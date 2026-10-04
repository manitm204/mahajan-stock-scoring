"""Variant of scripts/monte_carlo_market_neutral.py that hedges each long
by shorting a WEAK SINGLE NAME in the same sector, instead of the sector
SPDR ETF.

Hedge rule: for each long name (drawn as usual by uniform random pick from
the composite_score == 100 pool), look at its GICS sector, take the THREE
lowest composite-scored stocks in that sector as of the review date, and
uniformly randomly pick ONE of those three to short at `hedge` x the long
weight (default 1.0 -> dollar-neutral per name). So the short side is also
Monte-Carlo randomized, among names we score at the bottom of the sector.

To keep turnover sane and the short tied to its long, the chosen short is
PERSISTED for as long as that long name is held; it is only re-drawn when
the long name is (re-)entered. Longs that drop out release their short.

Question this answers: does replacing the broad sector-ETF hedge with a
short in a specifically weak same-sector name add return (capturing a
good-minus-bad within-sector spread) while still holding beta near zero?

Usage: python scripts/monte_carlo_short_worst_in_sector.py [--n 4]
       [--hold 4] [--refresh 3] [--sleeves 3] [--worst 3] [--hedge 1.0]
       [--sims 200] [--seed 0] [--out path.json]
Writes: output/monte_carlo_short_worst_in_sector/results.json (or --out)
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

from backtesting.data_loader import SPY, QQQ                         # noqa: E402
from research.autoresearch.evaluate import (                         # noqa: E402
    bench_returns, compute_portfolio_returns, targets_to_weight_matrix,
)
from research.strategies.engine import load_data                     # noqa: E402
from research.walkforward.portfolio import performance_metrics       # noqa: E402

OUT = REPO / "output" / "monte_carlo_short_worst_in_sector" / "results.json"


def _random_book(comp_date_scores: pd.Series, held: list, k: int,
                 refresh_n: int, rng: random.Random) -> list:
    """Long book: uniform random draw from the score==100 pool (identical
    to monte_carlo_random_book.py)."""
    pool = list(comp_date_scores[comp_date_scores == 100].index)
    if not pool:
        return held
    if not held:
        rng.shuffle(pool)
        return pool[:k]
    keep = list(held)
    n_evict = min(refresh_n, len(keep))
    to_evict = set(rng.sample(keep, n_evict))
    keep = [t for t in keep if t not in to_evict]
    need = k - len(keep)
    candidates = [t for t in pool if t not in keep]
    rng.shuffle(candidates)
    fill = candidates[:need]
    if len(fill) < need:
        universe = list(comp_date_scores.dropna().index)
        rng.shuffle(universe)
        extra = [t for t in universe if t not in keep and t not in fill]
        fill += extra[:need - len(fill)]
    return keep + fill


def _sector_worst(comp: dict, dates: list, sector: pd.Series,
                  worst_k: int) -> dict:
    """Precompute per date -> {sector: [worst_k lowest-scored tickers]}
    (ascending by composite score). Independent of the RNG, so it is built
    once and reused across all sims."""
    out = {}
    for d in dates:
        scores = comp.get(d)
        if scores is None:
            continue
        scored = scores.dropna()
        secs = sector.reindex(scored.index)
        per_sector = {}
        for sec, grp in scored.groupby(secs):
            per_sector[sec] = list(grp.sort_values().index[:worst_k])
        out[d] = per_sector
    return out


def _pick_short(long_ticker: str, d: str, sector: pd.Series,
                worst_by_date: dict, longs_today: set,
                rng: random.Random) -> str | None:
    """Uniformly pick one of the <=worst_k lowest-scored names in the long
    name's sector (excluding any name we are currently long). None if no
    candidate exists on this date."""
    sec = sector.get(long_ticker)
    cands = [t for t in worst_by_date.get(d, {}).get(sec, []) if t not in longs_today]
    if not cands:
        return None
    return rng.choice(cands)


def _random_targets(comp: dict, dates: list, rng: random.Random,
                    book_size: int, hold_months: int, refresh_n: int,
                    sleeve_count: int, hedge: float, sector: pd.Series,
                    worst_by_date: dict) -> pd.DataFrame:
    step = max(hold_months // sleeve_count, 1)
    sleeve_holdings = [[] for _ in range(sleeve_count)]
    weight_per_name = (1.0 / sleeve_count) / book_size
    short_for: dict[str, str] = {}   # long ticker -> its persisted short ticker
    rows = []
    for i, d in enumerate(dates):
        for j in range(sleeve_count):
            offset = j * step
            if i >= offset and (i - offset) % hold_months == 0:
                scores = comp.get(d)
                if scores is not None:
                    sleeve_holdings[j] = _random_book(
                        scores, sleeve_holdings[j], book_size, refresh_n, rng)
        # aggregate long weights per name across sleeves
        agg: dict[str, float] = {}
        for holdings in sleeve_holdings:
            for t in holdings:
                agg[t] = agg.get(t, 0.0) + weight_per_name
        longs_today = set(agg)
        # release shorts whose long is gone
        for lt in [lt for lt in short_for if lt not in longs_today]:
            del short_for[lt]
        # build short legs: one weak same-sector name per long, persisted
        short_agg: dict[str, float] = {}
        for lt, w in agg.items():
            st = short_for.get(lt)
            if st is None or st in longs_today:
                st = _pick_short(lt, d, sector, worst_by_date, longs_today, rng)
                if st is None:
                    continue
                short_for[lt] = st
            short_agg[st] = short_agg.get(st, 0.0) - hedge * w
        combined = dict(agg)
        for st, w in short_agg.items():
            combined[st] = combined.get(st, 0.0) + w
        for t, w in combined.items():
            if w != 0.0:
                rows.append({"date": d, "ticker": t, "weight": w})
    return pd.DataFrame(rows, columns=["date", "ticker", "weight"])


def _stats(pr: pd.Series, spy: pd.Series) -> dict:
    m = performance_metrics(pr, hold_months=1, benchmarks={"SPY": spy})
    return {"cagr": m["cagr"], "sharpe": m["sharpe"], "sortino": m["sortino"],
            "max_dd": m["max_drawdown"], "beta": m.get("spy_beta"),
            "alpha": m.get("spy_alpha")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4, help="book size per sleeve")
    ap.add_argument("--hold", type=int, default=4, help="hold months per sleeve review")
    ap.add_argument("--refresh", type=int, default=3, help="names evicted+refilled per review")
    ap.add_argument("--sleeves", type=int, default=3, help="number of staggered sleeves")
    ap.add_argument("--worst", type=int, default=3, help="pick short from the N lowest-scored in sector")
    ap.add_argument("--hedge", type=float, default=1.0,
                    help="short-leg ratio: short hedge*w of the chosen weak name per long w")
    ap.add_argument("--sims", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()
    book_size, hold_months, refresh_n = args.n, args.hold, args.refresh

    print("loading composite scores + price matrix ...", flush=True)
    data = load_data()
    worst_by_date = _sector_worst(data.comp, data.rebal_dates, data.sector, args.worst)

    rebal = list(data.rebal_dates)
    spy = bench_returns(data.matrix, SPY, rebal)
    qqq = bench_returns(data.matrix, QQQ, rebal)

    all_dates = rebal[:-1]  # compute_portfolio_returns drops the final open period
    sim_curves = np.zeros((args.sims, len(all_dates)))
    sim_stats = {"cagr": [], "sharpe": [], "sortino": [], "max_dd": [], "beta": [], "alpha": []}

    for s in range(args.sims):
        rng = random.Random(args.seed + s)
        targets = _random_targets(data.comp, rebal, rng, book_size, hold_months,
                                  refresh_n, args.sleeves, args.hedge, data.sector,
                                  worst_by_date)
        weights = targets_to_weight_matrix(targets, rebal)
        pr, _turnover = compute_portfolio_returns(weights, data.matrix, rebal)
        st = _stats(pr, spy)
        for k, v in st.items():
            sim_stats[k].append(v)
        pr = pr.reindex(all_dates).fillna(0.0)
        sim_curves[s] = (1.0 + pr).cumprod().values
        if (s + 1) % 20 == 0:
            print(f"  sim {s + 1}/{args.sims}", flush=True)

    median_curve = np.median(sim_curves, axis=0)
    p10 = np.percentile(sim_curves, 10, axis=0)
    p90 = np.percentile(sim_curves, 90, axis=0)

    spy_r = spy.reindex(all_dates).fillna(0.0)
    qqq_r = qqq.reindex(all_dates).fillna(0.0)
    spy_equity = (1.0 + spy_r).cumprod().values
    qqq_equity = (1.0 + qqq_r).cumprod().values

    portfolio_stats = {k: float(np.mean(v)) for k, v in sim_stats.items()}
    spy_stats = _stats(spy_r, spy_r)      # beta vs itself = 1.0 by construction
    qqq_stats = _stats(qqq_r, spy_r)      # beta vs SPY

    final_returns = sim_curves[:, -1] - 1.0
    summary = {
        "n_sims": args.sims,
        "median_final_return": float(np.median(final_returns)),
        "p10_final_return": float(np.percentile(final_returns, 10)),
        "p90_final_return": float(np.percentile(final_returns, 90)),
        "spy_final_return": float(spy_equity[-1] - 1.0),
        "qqq_final_return": float(qqq_equity[-1] - 1.0),
        "pct_sims_beating_spy": float((final_returns > (spy_equity[-1] - 1.0)).mean()),
        "pct_sims_positive": float((final_returns > 0).mean()),
    }
    print(json.dumps(summary, indent=2))
    print("\n=== stats (mean across sims for portfolio) ===")
    for name, st in [("ShortWorst", portfolio_stats), ("SPY", spy_stats), ("QQQ", qqq_stats)]:
        print(f"  {name:<10} cagr={st['cagr']:.4f}  sharpe={st['sharpe']:.3f}  "
             f"sortino={st['sortino']:.3f}  max_dd={st['max_dd']:.4f}  beta={st['beta']:.3f}  "
             f"alpha={st['alpha']:.4f}")

    out = {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "params": {"book_size": book_size, "hold_months": hold_months,
                   "sleeve_count": args.sleeves, "refresh_n": refresh_n,
                   "worst_k": args.worst, "hedge_ratio": args.hedge,
                   "n_sims": args.sims, "seed": args.seed,
                   "selection": "uniform random draw from composite_score==100 pool",
                   "hedge": "short hedge_ratio x long weight of a random name among the "
                            f"{args.worst} lowest-scored stocks in the long's GICS sector "
                            "(persisted while the long is held)"},
        "dates": [str(d) for d in all_dates],
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
