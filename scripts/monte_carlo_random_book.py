"""Monte Carlo test of the staggered-sleeve book structure (3 sleeves,
book size / hold months / refresh count all configurable via CLI -- default
matches "21_loopeng_book4_hold4_evict3" from
scripts/generate_strategy_lab_comparison.py) with the SELECTION step
replaced by uniform random draws from that date's composite_score == 100
pool (~11 tickers/date -- see research/autoresearch/candidate.py's
docstring on BOOK_SIZE=11 being a narrow, tied-at-the-max peak).

Question this answers: if every name at the top of the composite is tied at
score 100, does a deterministic eviction rule actually add value over
picking randomly among the tied names, for a given (N, hold, refresh)
config? Runs N independent random replays and plots them as a spaghetti
chart against a highlighted median path and SPY/QQQ, plus a stats table
(CAGR, Sharpe, Sortino, max drawdown, beta) for the portfolio (averaged
across sims), SPY, and QQQ.

Usage: python scripts/monte_carlo_random_book.py [--n 4] [--hold 4]
       [--refresh 3] [--sims 200] [--seed 0] [--out path.json]
Writes: output/monte_carlo_random_book/results.json (or --out)
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
    bench_returns, compute_portfolio_returns, targets_to_weight_matrix,
)
from research.strategies.engine import load_data                     # noqa: E402
from research.walkforward.portfolio import performance_metrics       # noqa: E402

OUT = REPO / "output" / "monte_carlo_random_book" / "results.json"


def _random_book(comp_date_scores: pd.Series, held: list, k: int,
                 refresh_n: int, rng: random.Random) -> list:
    """Pick a k-name book by uniform random draw from the score==100 pool.
    First fill for a sleeve: k random names from the pool. Later reviews:
    evict `refresh_n` currently-held names at random, refill from the pool
    (excluding what's kept), falling back to the full scored universe only
    if the pool itself is smaller than what's needed (rare -- pool is
    ~11 names, need at most k)."""
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


def _random_targets(comp: dict, dates: list, rng: random.Random,
                    book_size: int, hold_months: int, refresh_n: int,
                    sleeve_count: int) -> pd.DataFrame:
    step = max(hold_months // sleeve_count, 1)
    sleeve_holdings = [[] for _ in range(sleeve_count)]
    weight_per_name = (1.0 / sleeve_count) / book_size
    rows = []
    for i, d in enumerate(dates):
        for j in range(sleeve_count):
            offset = j * step
            if i >= offset and (i - offset) % hold_months == 0:
                scores = comp.get(d)
                if scores is not None:
                    sleeve_holdings[j] = _random_book(
                        scores, sleeve_holdings[j], book_size, refresh_n, rng)
        agg = {}
        for holdings in sleeve_holdings:
            for t in holdings:
                agg[t] = agg.get(t, 0.0) + weight_per_name
        for t, w in agg.items():
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
    ap.add_argument("--sims", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()
    book_size, hold_months, refresh_n = args.n, args.hold, args.refresh

    print("loading composite scores + price matrix ...", flush=True)
    data = load_data()
    rebal = list(data.rebal_dates)
    spy = bench_returns(data.matrix, SPY, rebal)
    qqq = bench_returns(data.matrix, QQQ, rebal)

    all_dates = rebal[:-1]  # compute_portfolio_returns drops the final open period
    sim_curves = np.zeros((args.sims, len(all_dates)))
    sim_stats = {"cagr": [], "sharpe": [], "sortino": [], "max_dd": [], "beta": [], "alpha": []}

    for s in range(args.sims):
        rng = random.Random(args.seed + s)
        targets = _random_targets(data.comp, rebal, rng, book_size, hold_months, refresh_n,
                                  args.sleeves)
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
        "params": {"book_size": book_size, "hold_months": hold_months,
                   "sleeve_count": args.sleeves, "refresh_n": refresh_n,
                   "n_sims": args.sims, "seed": args.seed,
                   "selection": "uniform random draw from composite_score==100 pool"},
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
