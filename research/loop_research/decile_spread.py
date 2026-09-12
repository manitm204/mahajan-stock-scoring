"""Decile-spread validation of the composite score itself (user request
2026-09-09): "does the scoring actually work" -- rank the full universe by
composite score each date into 10 equal-sized groups, hold a random 10-name
book drawn from each decile (same staggered-sleeve mechanics as everywhere
else: 4 sleeves, 4-month hold, full turnover), and compare the Monte Carlo
distributions. If the composite score carries real information, decile 1
(highest scores) should clearly and fairly monotonically outperform decile
10 (lowest scores).

Equal-sized RANK deciles, not raw score-value bins -- the composite is a
per-sector percentile with a large tied mass at 100, so a raw-value bin
would make "decile 1" much bigger than the others. Ties are broken by
ticker ascending (deterministic, matches every other selector in this repo)
before splitting into 10 equal slices via a stable full-universe sort.

Usage: python -m research.loop_research.decile_spread
Writes: output/loop_research/decile_spread.json
"""
from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from backtesting.data_loader import SPY, QQQ                          # noqa: E402
from research.autoresearch.evaluate import bench_returns               # noqa: E402
from research.loop_research import harness as H                        # noqa: E402
from research.walkforward.portfolio import performance_metrics         # noqa: E402

OUT = REPO / "output" / "loop_research" / "decile_spread.json"
K = 10
N_DECILES = 10
N_SIMS = 500


def _decile_pool(scores: pd.Series, decile: int) -> list[str]:
    """decile=1 is the TOP 10% of scores that date, decile=10 the bottom."""
    s = scores.dropna().sort_index().sort_values(ascending=False, kind="stable")
    n = len(s)
    if n == 0:
        return []
    bins = np.array_split(np.arange(n), N_DECILES)
    idx = bins[decile - 1]
    return list(s.index[idx])


def make_decile_selector(decile: int):
    def selector(scores, parent_scores, held, k, refresh_n, rng, ctx=None):
        pool = _decile_pool(scores, decile)
        if not pool:
            return held
        if not held:
            p = list(pool)
            rng.shuffle(p)
            return p[:k]
        keep = list(held)
        n_evict = min(refresh_n, len(keep))
        to_evict = set(rng.sample(keep, n_evict))
        keep = [t for t in keep if t not in to_evict]
        need = k - len(keep)
        cands = [t for t in pool if t not in keep]
        rng.shuffle(cands)
        fill = cands[:need]
        if len(fill) < need:
            universe = [t for t in scores.dropna().index if t not in keep and t not in fill]
            rng.shuffle(universe)
            fill += universe[:need - len(fill)]
        return keep + fill
    selector.__name__ = f"decile_{decile}"
    return selector


def _stats(pr, spy):
    m = performance_metrics(pr, hold_months=1, benchmarks={"SPY": spy})
    return {"cagr": m["cagr"], "sharpe": m["sharpe"], "sortino": m["sortino"],
            "max_dd": m["max_drawdown"], "beta": m.get("spy_beta"),
            "alpha": m.get("spy_alpha"), "ir": m.get("spy_ir")}


def run_decile(bundle, decile, n_sims=N_SIMS, seed_base=0):
    data = bundle["data"]
    rebal = list(data.rebal_dates)
    spy = bench_returns(data.matrix, SPY, rebal)
    selector = make_decile_selector(decile)
    sim_stats = {"cagr": [], "sharpe": [], "sortino": [], "max_dd": [], "beta": [], "alpha": [], "ir": []}
    pct_beat_spy = []
    spy_total = float((1.0 + spy).prod() - 1.0)
    for s in range(n_sims):
        pr, turnover, targets = H.run_one(bundle, selector, seed_base + s, k=K, refresh_n=K)
        st = _stats(pr, spy)
        for kk, v in st.items():
            sim_stats[kk].append(v)
        pct_beat_spy.append(float((1.0 + pr).prod() - 1.0) > spy_total)
    return {kk: float(np.median(v)) for kk, v in sim_stats.items()} | {
        "pct_sims_beating_spy": float(np.mean(pct_beat_spy)),
    }


def main():
    bundle = H.get_data()
    results = {}
    t0 = time.time()
    for decile in range(1, N_DECILES + 1):
        results[decile] = run_decile(bundle, decile)
        elapsed = time.time() - t0
        print(f"decile {decile}/10  sharpe_median={results[decile]['sharpe']:.3f}  "
             f"cagr={results[decile]['cagr']:.3f}  ({elapsed:.0f}s elapsed)", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as fh:
        json.dump(results, fh, indent=2)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
