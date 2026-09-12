"""Monte Carlo test of the v3 loop-engineering managed-book strategy
(research/strategies/generic_rule.py::StrategyConfig, exit rule = 9-month
time cap OR 10% trailing stop off the peak OR the name's composite rank
falling into the worst quartile of the full scored universe, with a 1-week
minimum hold suppressing soft exits early -- see research/loop_engineering/
README.md's "v3_loopeng_cap9_minhold1wk_quartile") with the ENTRY step
replaced by a uniform random draw from that review date's composite_score
== 100 pool, book size k=5, instead of research/strategies/engine.py's
default top-k-by-score entry.

Same question as scripts/monte_carlo_random_book.py, applied to a
different strategy family (single managed book + per-position exit rules,
rather than staggered calendar sleeves): if entries are basically
interchangeable among names tied at composite score 100, does this
strategy's edge survive random entry selection, or was it riding on which
specific top-k names got picked?

Usage: python scripts/monte_carlo_managed_book.py --mode random [--k 5]
       [--sims 200] [--seed 0] [--trail 0.10] [--cap-months 9]
       [--rank-floor 0.75] [--min-hold 0.25] [--out path.json]

       python scripts/monte_carlo_managed_book.py --mode all100
       -- deterministic: holds every ticker scored exactly 100 at each
       review (no randomness, no fixed k), same exit rule otherwise.

Writes: output/monte_carlo_managed_book/results.json (or --out)
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
from research.strategies.engine import (                             # noqa: E402
    Book, COST_BPS, _close, _mark, _open, load_data,
)
from research.strategies.generic_rule import StrategyConfig, build_rule  # noqa: E402
from research.walkforward.portfolio import performance_metrics       # noqa: E402
from scripts.run_strategy_sweep import bench_returns, monthly_returns  # noqa: E402

OUT = REPO / "output" / "monte_carlo_managed_book" / "results.json"


def _value_ok(vpct, t, min_value_pct):
    if min_value_pct is None:
        return True
    if vpct is None:
        return False
    v = vpct.get(t, np.nan)
    return not pd.isna(v) and v >= min_value_pct


def _review_random(day, book, data, exit_rule, k, cost_bps, rng, rebal_set, min_value_pct=None):
    is_review = day in rebal_set
    comp_today = data.comp.get(day) if is_review else None
    matrix = data.matrix

    for t in list(book.positions):
        p = book.positions[t]
        px = matrix.at[day, t] if t in matrix.columns else np.nan
        if pd.isna(px):
            continue
        if exit_rule(p, day, px, is_review, comp_today):
            _close(book, t, cost_bps)

    if not is_review:
        return
    open_slots = k - len(book.positions)
    if open_slots <= 0 or comp_today is None:
        return
    held = set(book.positions)
    vpct = data.value_pct.get(day) if min_value_pct is not None else None
    pool = [t for t in comp_today[comp_today == 100].index
           if t not in held and _value_ok(vpct, t, min_value_pct)]
    rng.shuffle(pool)
    fill = pool[:open_slots]
    if len(fill) < open_slots:
        universe = [t for t in comp_today.dropna().index
                   if t not in held and t not in fill and _value_ok(vpct, t, min_value_pct)]
        rng.shuffle(universe)
        fill += universe[:open_slots - len(fill)]
    for t in fill:
        px = matrix.at[day, t] if t in matrix.columns else np.nan
        sc = float(comp_today.get(t, np.nan))
        _open(book, t, day, px, k, cost_bps, score=sc)


def _review_all100(day, book, data, exit_rule, cost_bps, rebal_set):
    """Deterministic variant: hold every ticker scored exactly 100 that
    review date (no randomness, no fixed k) -- new entrants are sized as
    1/(size of that date's full score==100 pool), existing positions are
    left to drift (same managed-book convention as the k-slot engine: only
    entries are (re)sized, not a full rebalance every review)."""
    is_review = day in rebal_set
    comp_today = data.comp.get(day) if is_review else None
    matrix = data.matrix

    for t in list(book.positions):
        p = book.positions[t]
        px = matrix.at[day, t] if t in matrix.columns else np.nan
        if pd.isna(px):
            continue
        if exit_rule(p, day, px, is_review, comp_today):
            _close(book, t, cost_bps)

    if not is_review:
        return
    pool = list(comp_today[comp_today == 100].index)
    if not pool:
        return
    held = set(book.positions)
    new_entrants = [t for t in pool if t not in held]
    if not new_entrants:
        return
    k_dynamic = len(pool)
    for t in new_entrants:
        px = matrix.at[day, t] if t in matrix.columns else np.nan
        sc = float(comp_today.get(t, np.nan))
        _open(book, t, day, px, k_dynamic, cost_bps, score=sc)


def simulate_managed_book_all100(data, exit_rule, cost_bps=COST_BPS) -> pd.Series:
    rebal_set = set(data.rebal_dates)
    book = Book()
    day0 = data.trading_days[0]
    nav = {}
    _review_all100(day0, book, data, exit_rule, cost_bps, rebal_set)
    nav[day0] = book.total()
    prev_day = day0
    for day in data.trading_days[1:]:
        _mark(book, data.matrix, prev_day, day)
        _review_all100(day, book, data, exit_rule, cost_bps, rebal_set)
        nav[day] = book.total()
        prev_day = day
    return pd.Series(nav).sort_index()


def simulate_managed_book_random(data, exit_rule, k, rng, cost_bps=COST_BPS,
                                 min_value_pct=None) -> pd.Series:
    rebal_set = set(data.rebal_dates)
    book = Book()
    day0 = data.trading_days[0]
    nav = {}
    _review_random(day0, book, data, exit_rule, k, cost_bps, rng, rebal_set, min_value_pct)
    nav[day0] = book.total()
    prev_day = day0
    for day in data.trading_days[1:]:
        _mark(book, data.matrix, prev_day, day)
        _review_random(day, book, data, exit_rule, k, cost_bps, rng, rebal_set, min_value_pct)
        nav[day] = book.total()
        prev_day = day
    return pd.Series(nav).sort_index()


def _stats(pr: pd.Series, spy: pd.Series) -> dict:
    m = performance_metrics(pr, hold_months=1, benchmarks={"SPY": spy})
    return {"cagr": m["cagr"], "sharpe": m["sharpe"], "sortino": m["sortino"],
            "max_dd": m["max_drawdown"], "beta": m.get("spy_beta"),
            "alpha": m.get("spy_alpha")}


def _run_all100(data, exit_rule, rebal, spy_r, qqq_r, out_path) -> None:
    """Deterministic replay: hold every composite_score==100 name, no
    randomness, no fixed k -- see simulate_managed_book_all100."""
    print("running deterministic all-score-100 replay ...", flush=True)
    nav = simulate_managed_book_all100(data, exit_rule)
    nav_m = nav.reindex(rebal).dropna()
    dates = list(nav_m.index)
    curve = (nav_m / nav_m.iloc[0]).values
    pr = nav_m.pct_change().dropna()
    stats = _stats(pr, spy_r)

    spy_equity = (1.0 + spy_r.reindex(dates).fillna(0.0)).cumprod().values
    qqq_equity = (1.0 + qqq_r.reindex(dates).fillna(0.0)).cumprod().values
    spy_stats = _stats(spy_r.reindex(dates).dropna(), spy_r)
    qqq_stats = _stats(qqq_r.reindex(dates).dropna(), spy_r)

    print(f"  portfolio  cagr={stats['cagr']:.4f}  sharpe={stats['sharpe']:.3f}  "
         f"sortino={stats['sortino']:.3f}  max_dd={stats['max_dd']:.4f}  "
         f"beta={stats['beta']:.3f}  alpha={stats['alpha']:.4f}")

    out = {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "params": {"mode": "all100", "selection": "every ticker with composite_score==100, no randomness"},
        "dates": [str(d) for d in dates],
        "curve": curve.tolist(),
        "spy_curve": spy_equity.tolist(),
        "qqq_curve": qqq_equity.tolist(),
        "stats": {"portfolio": stats, "spy": spy_stats, "qqq": qqq_stats},
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        json.dump(out, fh, indent=2, default=lambda x: None if x != x else x)
    print(f"wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["random", "all100"], default="random")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--trail", type=float, default=0.10)
    ap.add_argument("--cap-months", type=float, default=9)
    ap.add_argument("--rank-floor", type=float, default=0.75)
    ap.add_argument("--min-hold", type=float, default=0.25)
    ap.add_argument("--min-value-pct", type=float, default=None,
                    help="v2 champion's Value-parent entry floor (e.g. 15.0)")
    ap.add_argument("--sims", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    print("loading composite scores + price matrix ...", flush=True)
    data = load_data()
    rebal = list(data.rebal_dates)
    spy_r = bench_returns(data.matrix, SPY, rebal)
    qqq_r = bench_returns(data.matrix, QQQ, rebal)

    cfg = StrategyConfig(k=args.k, trail_pct=args.trail, cap_months=args.cap_months,
                         rank_floor_pct=args.rank_floor, min_hold_months=args.min_hold)
    exit_rule = build_rule(cfg)

    if args.mode == "all100":
        out_path = Path(args.out) if args.out else OUT.parent / "all100_results.json"
        _run_all100(data, exit_rule, rebal, spy_r, qqq_r, out_path)
        return

    dates = None
    sim_curves = []
    sim_stats = {"cagr": [], "sharpe": [], "sortino": [], "max_dd": [], "beta": [], "alpha": []}

    for s in range(args.sims):
        rng = random.Random(args.seed + s)
        nav = simulate_managed_book_random(data, exit_rule, args.k, rng,
                                          min_value_pct=args.min_value_pct)
        nav_m = nav.reindex(rebal).dropna()
        if dates is None:
            dates = list(nav_m.index)
        curve = (nav_m / nav_m.iloc[0]).values
        sim_curves.append(curve)
        pr = nav_m.pct_change().dropna()
        st = _stats(pr, spy_r)
        for kk, v in st.items():
            sim_stats[kk].append(v)
        if (s + 1) % 20 == 0:
            print(f"  sim {s + 1}/{args.sims}", flush=True)

    sim_curves = np.array(sim_curves)
    median_curve = np.median(sim_curves, axis=0)
    p10 = np.percentile(sim_curves, 10, axis=0)
    p90 = np.percentile(sim_curves, 90, axis=0)

    spy_equity = (1.0 + spy_r.reindex(dates).fillna(0.0)).cumprod().values
    qqq_equity = (1.0 + qqq_r.reindex(dates).fillna(0.0)).cumprod().values

    portfolio_stats = {k2: float(np.mean(v)) for k2, v in sim_stats.items()}
    spy_stats = _stats(spy_r.reindex(dates).dropna(), spy_r)
    qqq_stats = _stats(qqq_r.reindex(dates).dropna(), spy_r)

    final_returns = sim_curves[:, -1] - 1.0
    summary = {
        "n_sims": args.sims,
        "median_final_return": float(np.median(final_returns)),
        "p10_final_return": float(np.percentile(final_returns, 10)),
        "p90_final_return": float(np.percentile(final_returns, 90)),
        "spy_final_return": float(spy_equity[-1] - 1.0),
        "qqq_final_return": float(qqq_equity[-1] - 1.0),
        "pct_sims_beating_spy": float((final_returns > (spy_equity[-1] - 1.0)).mean()),
        "pct_sims_beating_qqq": float((final_returns > (qqq_equity[-1] - 1.0)).mean()),
    }
    print(json.dumps(summary, indent=2))
    print("\n=== stats (mean across sims for portfolio) ===")
    for name, st in [("Portfolio", portfolio_stats), ("SPY", spy_stats), ("QQQ", qqq_stats)]:
        print(f"  {name:<10} cagr={st['cagr']:.4f}  sharpe={st['sharpe']:.3f}  "
             f"sortino={st['sortino']:.3f}  max_dd={st['max_dd']:.4f}  beta={st['beta']:.3f}  "
             f"alpha={st['alpha']:.4f}")

    out = {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "params": {"k": args.k, "trail_pct": args.trail, "cap_months": args.cap_months,
                   "rank_floor_pct": args.rank_floor, "min_hold_months": args.min_hold,
                   "min_value_pct": args.min_value_pct,
                   "n_sims": args.sims, "seed": args.seed,
                   "selection": "uniform random draw from composite_score==100 pool at entry"},
        "dates": [str(d) for d in dates],
        "sim_curves": sim_curves.tolist(),
        "median_curve": median_curve.tolist(),
        "p10_curve": p10.tolist(),
        "p90_curve": p90.tolist(),
        "spy_curve": spy_equity.tolist(),
        "qqq_curve": qqq_equity.tolist(),
        "summary": summary,
        "stats": {"portfolio": portfolio_stats, "spy": spy_stats, "qqq": qqq_stats},
        "sim_stats": sim_stats,
    }
    out_path = Path(args.out) if args.out else OUT
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        json.dump(out, fh, indent=2, default=lambda x: None if x != x else x)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
