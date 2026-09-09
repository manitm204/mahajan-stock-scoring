"""Monte Carlo test of the book4/hold4/evict3 structure (see
scripts/generate_strategy_lab_comparison.py's "21_loopeng_book4_hold4_evict3"
replay: 3 staggered 4-name sleeves, 4-month holds, 3 evictions per review --
the neighbor the user identified as best in the book-size robustness sweep)
with the SELECTION step replaced by uniform random draws from that date's
composite_score == 100 pool (~11 tickers/date -- see
research/autoresearch/candidate.py's docstring on BOOK_SIZE=11 being a
narrow, tied-at-the-max peak).

Question this answers: if every name at the top of the composite is tied at
score 100, does the deterministic "worst-momentum-decline eviction" rule
actually add value over picking randomly among the tied names? Runs N
independent random replays and plots them as a spaghetti chart against a
highlighted median path and SPY/QQQ.

Usage: python scripts/monte_carlo_random_book.py [--sims 200] [--seed 0]
Writes: output/monte_carlo_random_book/results.json
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
    compute_portfolio_returns, targets_to_weight_matrix,
)
from research.strategies.engine import load_data                     # noqa: E402
from scripts.run_strategy_sweep import bench_returns                 # noqa: E402

OUT = REPO / "output" / "monte_carlo_random_book" / "results.json"

BOOK_SIZE = 4
HOLD_MONTHS = 4
SLEEVE_COUNT = 3
REFRESH_N = 3
STEP = max(HOLD_MONTHS // SLEEVE_COUNT, 1)


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


def _random_targets(comp: dict, dates: list, rng: random.Random) -> pd.DataFrame:
    sleeve_holdings = [[] for _ in range(SLEEVE_COUNT)]
    weight_per_name = (1.0 / SLEEVE_COUNT) / BOOK_SIZE
    rows = []
    for i, d in enumerate(dates):
        for j in range(SLEEVE_COUNT):
            offset = j * STEP
            if i >= offset and (i - offset) % HOLD_MONTHS == 0:
                scores = comp.get(d)
                if scores is not None:
                    sleeve_holdings[j] = _random_book(
                        scores, sleeve_holdings[j], BOOK_SIZE, REFRESH_N, rng)
        agg = {}
        for holdings in sleeve_holdings:
            for t in holdings:
                agg[t] = agg.get(t, 0.0) + weight_per_name
        for t, w in agg.items():
            rows.append({"date": d, "ticker": t, "weight": w})
    return pd.DataFrame(rows, columns=["date", "ticker", "weight"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print("loading composite scores + price matrix ...", flush=True)
    data = load_data()
    rebal = list(data.rebal_dates)
    spy = bench_returns(data.matrix, SPY, rebal)
    qqq = bench_returns(data.matrix, QQQ, rebal)

    all_dates = rebal[:-1]  # compute_portfolio_returns drops the final open period
    sim_curves = np.zeros((args.sims, len(all_dates)))

    for s in range(args.sims):
        rng = random.Random(args.seed + s)
        targets = _random_targets(data.comp, rebal, rng)
        weights = targets_to_weight_matrix(targets, rebal)
        pr, _turnover = compute_portfolio_returns(weights, data.matrix, rebal)
        pr = pr.reindex(all_dates).fillna(0.0)
        sim_curves[s] = (1.0 + pr).cumprod().values
        if (s + 1) % 20 == 0:
            print(f"  sim {s + 1}/{args.sims}", flush=True)

    median_curve = np.median(sim_curves, axis=0)
    p10 = np.percentile(sim_curves, 10, axis=0)
    p90 = np.percentile(sim_curves, 90, axis=0)

    spy_equity = (1.0 + spy.reindex(all_dates).fillna(0.0)).cumprod().values
    qqq_equity = (1.0 + qqq.reindex(all_dates).fillna(0.0)).cumprod().values

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

    out = {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "params": {"book_size": BOOK_SIZE, "hold_months": HOLD_MONTHS,
                   "sleeve_count": SLEEVE_COUNT, "refresh_n": REFRESH_N,
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
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as fh:
        json.dump(out, fh, indent=2)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
