"""Monte Carlo test of WEIGHT-rebalancing frequency inside the staggered-
sleeve random book (scripts/monte_carlo_random_book.py, default here
n=5 / hold=4 / refresh=5 / 4 sleeves). Names never change between sleeve
reviews -- each sleeve still holds its 5 names for its full 4 months --
only how often position weights are reset back to equal varies:

  drift   : no weight resets. Each sleeve is its own sub-account; at its
            review it splits its own value equally across its 5 new names,
            then lets them (and its value relative to the other sleeves)
            drift for the 4 months.
  monthly : every monthly date the whole book is reset to equal weight
            (1/20 per name-slot). This is what monte_carlo_random_book.py's
            engine does implicitly (it re-applies target weights each month).
  weekly  : monthly resets, plus a reset to equal weight on the first
            trading day of every week.

Every sim draws its sleeve books once (same RNG stream as
monte_carlo_random_book.py, so seed s here = seed s there) and replays them
under all three modes, so mode differences are paired, not noise. Costs are
charged on drift-aware turnover (|trade| vs. the drifted position) at
COST_BPS -- unlike compute_portfolio_returns, which charges turnover between
consecutive *targets* and so never pays for re-equalizing kept names.
Execution is T+1 on monthly dates (EXEC_LAG_DAYS); weekly resets need no
signal so they trade on the week's first session itself.

Usage: python scripts/monte_carlo_rebalance_frequency.py [--n 5] [--hold 4]
       [--refresh 5] [--sleeves 4] [--sims 500] [--seed 0] [--out path.json]
Writes: output/monte_carlo_random_book/rebalance_freq_n5_hold4_refresh5_sleeves4_sims500.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from backtesting.data_loader import SPY, QQQ                        # noqa: E402
from research.autoresearch.evaluate import (                         # noqa: E402
    COST_BPS, EXEC_LAG_DAYS, _snap_forward, bench_returns,
)
from research.strategies.engine import load_data                     # noqa: E402
from research.walkforward.portfolio import performance_metrics       # noqa: E402
from scripts.monte_carlo_random_book import _random_book             # noqa: E402

OUT_DIR = REPO / "output" / "monte_carlo_random_book"
MODES = ("drift", "monthly", "weekly")


def _sleeve_schedule(comp: dict, dates: list, rng: random.Random, book_size: int,
                     hold_months: int, refresh_n: int, sleeve_count: int) -> list:
    """Same loop (and RNG call order) as monte_carlo_random_book._random_targets,
    but keeps sleeve membership: returns, per date, (sleeve holdings list,
    set of sleeve indices reviewed that date)."""
    step = max(hold_months // sleeve_count, 1)
    sleeve_holdings = [[] for _ in range(sleeve_count)]
    out = []
    for i, d in enumerate(dates):
        reviewed = set()
        for j in range(sleeve_count):
            offset = j * step
            if i >= offset and (i - offset) % hold_months == 0:
                scores = comp.get(d)
                if scores is not None:
                    sleeve_holdings[j] = _random_book(
                        scores, sleeve_holdings[j], book_size, refresh_n, rng)
                    reviewed.add(j)
        out.append(([list(h) for h in sleeve_holdings], reviewed))
    return out


def _simulate(schedule: list, mode: str, px: np.ndarray, col: dict,
              exec_pos: list, week_pos: set, n_slots: int) -> np.ndarray:
    """Returns NAV at each monthly exec position (len = len(exec_pos)), NAV0=1.
    State per sleeve: (ticker column indices, dollar values) plus per-sleeve
    cash. Equal-weight modes pool cash book-wide by resetting every sleeve."""
    cost = COST_BPS / 1e4
    n_sleeves = len(schedule[0][0])
    idx = [np.array([], dtype=int) for _ in range(n_sleeves)]
    val = [np.array([]) for _ in range(n_sleeves)]
    cash = np.full(n_sleeves, 1.0 / n_sleeves)
    monthly_at = {p: i for i, p in enumerate(exec_pos)}
    events = sorted(set(exec_pos) | (week_pos if mode == "weekly" else set()))
    nav_out = np.zeros(len(exec_pos))
    last = events[0]

    for p in events:
        # drift every position from the previous event to this one
        if p != last:
            for j in range(n_sleeves):
                if len(idx[j]):
                    rel = px[p, idx[j]] / px[last, idx[j]]
                    val[j] = val[j] * np.where(np.isfinite(rel), rel, 1.0)
            last = p
        nav = float(sum(v.sum() for v in val) + cash.sum())

        m = monthly_at.get(p)
        if m is not None:
            nav_out[m] = nav
            holdings, reviewed = schedule[m]
            new_idx = [np.array([col[t] for t in h], dtype=int) for h in holdings]
        else:
            new_idx = idx          # weekly reset: same names
            reviewed = set()

        if mode == "drift":
            for j in reviewed:
                sleeve_val = val[j].sum() + cash[j]
                old = dict(zip(idx[j].tolist(), val[j].tolist()))
                k = len(new_idx[j])
                tgt = sleeve_val / k if k else 0.0
                traded = sum(abs(tgt - old.pop(c, 0.0)) for c in new_idx[j].tolist())
                traded += sum(abs(v) for v in old.values())
                net = sleeve_val - traded * cost
                idx[j] = new_idx[j]
                val[j] = np.full(k, net / k) if k else np.array([])
                cash[j] = 0.0 if k else net
        else:
            old = {}
            for j in range(n_sleeves):
                for c, v in zip(idx[j].tolist(), val[j].tolist()):
                    old[c] = old.get(c, 0.0) + v
            slot_tgt = nav / n_slots
            new_tot = {}
            for j in range(n_sleeves):
                for c in new_idx[j].tolist():
                    new_tot[c] = new_tot.get(c, 0.0) + slot_tgt
            traded = sum(abs(new_tot.get(c, 0.0) - old.get(c, 0.0))
                         for c in set(old) | set(new_tot))
            net = nav - traded * cost
            filled = sum(len(x) for x in new_idx)
            for j in range(n_sleeves):
                idx[j] = new_idx[j]
                val[j] = np.full(len(idx[j]), net / n_slots)
            cash[:] = net * (1.0 - filled / n_slots) / n_sleeves
    return nav_out


def _stats(pr: pd.Series, spy: pd.Series) -> dict:
    m = performance_metrics(pr, hold_months=1, benchmarks={"SPY": spy})
    return {"cagr": m["cagr"], "sharpe": m["sharpe"], "sortino": m["sortino"],
            "max_dd": m["max_drawdown"], "beta": m.get("spy_beta"),
            "alpha": m.get("spy_alpha")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5, help="book size per sleeve")
    ap.add_argument("--hold", type=int, default=4, help="hold months per sleeve review")
    ap.add_argument("--refresh", type=int, default=5, help="names evicted+refilled per review")
    ap.add_argument("--sleeves", type=int, default=4, help="number of staggered sleeves")
    ap.add_argument("--sims", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()
    n_slots = args.n * args.sleeves

    print("loading composite scores + price matrix ...", flush=True)
    data = load_data()
    rebal = list(data.rebal_dates)
    spy = bench_returns(data.matrix, SPY, rebal)
    qqq = bench_returns(data.matrix, QQQ, rebal)
    all_dates = rebal[:-1]

    pool_tickers = sorted({t for s in data.comp.values() for t in s.dropna().index}
                          & set(data.matrix.columns))
    col = {t: i for i, t in enumerate(pool_tickers)}
    px = data.matrix[pool_tickers].to_numpy(dtype=float)
    index = data.matrix.index
    # same T+1 positions as evaluate._lagged_prices uses for compute_portfolio_returns
    exec_pos = [min(index.get_loc(_snap_forward(index, d)) + EXEC_LAG_DAYS, len(index) - 1)
                for d in rebal]
    weeks = pd.DatetimeIndex(pd.to_datetime(index)).to_period("W")
    week_pos = {p for p in range(exec_pos[0] + 1, exec_pos[-1])
                if weeks[p] != weeks[p - 1]} - set(exec_pos)
    print(f"{len(exec_pos)} monthly dates, {len(week_pos)} extra weekly resets", flush=True)

    curves = {m: np.zeros((args.sims, len(all_dates))) for m in MODES}
    stats = {m: {k: [] for k in ("cagr", "sharpe", "sortino", "max_dd", "beta", "alpha")}
             for m in MODES}
    for s in range(args.sims):
        rng = random.Random(args.seed + s)
        sched = _sleeve_schedule(data.comp, rebal, rng, args.n, args.hold,
                                 args.refresh, args.sleeves)
        for mode in MODES:
            nav = _simulate(sched, mode, px, col, exec_pos, week_pos, n_slots)
            pr = pd.Series(nav[1:] / nav[:-1] - 1.0, index=all_dates)
            for k, v in _stats(pr, spy).items():
                stats[mode][k].append(v)
            curves[mode][s] = (1.0 + pr).cumprod().values
        if (s + 1) % 20 == 0:
            print(f"  sim {s + 1}/{args.sims}", flush=True)

    spy_r = spy.reindex(all_dates).fillna(0.0)
    qqq_r = qqq.reindex(all_dates).fillna(0.0)
    spy_equity = (1.0 + spy_r).cumprod().values
    qqq_equity = (1.0 + qqq_r).cumprod().values

    out_modes = {}
    print("\n=== mean across sims ===")
    for mode in MODES:
        c = curves[mode]
        final = c[:, -1] - 1.0
        mean_stats = {k: float(np.mean(v)) for k, v in stats[mode].items()}
        out_modes[mode] = {
            "median_curve": np.median(c, axis=0).tolist(),
            "p10_curve": np.percentile(c, 10, axis=0).tolist(),
            "p90_curve": np.percentile(c, 90, axis=0).tolist(),
            "stats": mean_stats,
            "sim_stats": stats[mode],
            "summary": {
                "median_final_return": float(np.median(final)),
                "p10_final_return": float(np.percentile(final, 10)),
                "p90_final_return": float(np.percentile(final, 90)),
                "pct_sims_beating_spy": float((final > spy_equity[-1] - 1.0).mean()),
                "pct_sims_beating_qqq": float((final > qqq_equity[-1] - 1.0).mean()),
            },
        }
        st = mean_stats
        print(f"  {mode:<8} cagr={st['cagr']:.4f}  sharpe={st['sharpe']:.3f}  "
              f"sortino={st['sortino']:.3f}  max_dd={st['max_dd']:.4f}  "
              f"beta={st['beta']:.3f}  alpha={st['alpha']:.4f}")

    # per-sim cumulative gap vs monthly (same names), so small mode differences
    # are visible instead of buried under the cross-sim spread
    gap_vs_monthly = {}
    for mode in ("weekly", "drift"):
        g = curves[mode] / curves["monthly"] - 1.0
        gap_vs_monthly[mode] = {"median": np.median(g, axis=0).tolist(),
                                "p10": np.percentile(g, 10, axis=0).tolist(),
                                "p90": np.percentile(g, 90, axis=0).tolist()}

    paired = {}
    print("\n=== paired vs drift (same names, per sim) ===")
    for mode in ("monthly", "weekly"):
        for k in ("cagr", "sharpe"):
            diff = np.array(stats[mode][k]) - np.array(stats["drift"][k])
            paired[f"{mode}_minus_drift_{k}"] = {
                "mean": float(diff.mean()), "pct_positive": float((diff > 0).mean())}
            print(f"  {mode}-drift {k}: mean {diff.mean():+.4f}  "
                  f"positive in {(diff > 0).mean():.0%} of sims")
    for k in ("cagr", "sharpe"):
        diff = np.array(stats["weekly"][k]) - np.array(stats["monthly"][k])
        paired[f"weekly_minus_monthly_{k}"] = {
            "mean": float(diff.mean()), "pct_positive": float((diff > 0).mean())}
        print(f"  weekly-monthly {k}: mean {diff.mean():+.4f}  "
              f"positive in {(diff > 0).mean():.0%} of sims")

    out = {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "params": {"book_size": args.n, "hold_months": args.hold,
                   "sleeve_count": args.sleeves, "refresh_n": args.refresh,
                   "n_sims": args.sims, "seed": args.seed, "cost_bps": COST_BPS,
                   "selection": "uniform random draw from composite_score==100 pool",
                   "modes": list(MODES)},
        "dates": [str(d) for d in all_dates],
        "modes": out_modes,
        "paired": paired,
        "gap_vs_monthly": gap_vs_monthly,
        "spy_curve": spy_equity.tolist(),
        "qqq_curve": qqq_equity.tolist(),
        "stats": {"spy": _stats(spy_r, spy_r), "qqq": _stats(qqq_r, spy_r)},
    }
    name = (f"rebalance_freq_n{args.n}_hold{args.hold}_refresh{args.refresh}"
            f"_sleeves{args.sleeves}_sims{args.sims}.json")
    out_path = Path(args.out) if args.out else OUT_DIR / name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        json.dump(out, fh, indent=2, default=lambda x: None if x != x else x)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
