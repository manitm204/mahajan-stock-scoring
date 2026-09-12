"""Pool-robustness check for exp5 (user request 2026-09-09): does
"pick the best-insider names" still beat "pick randomly" when applied to a
DIFFERENT pool than the one it was found on? The composite_score==100 tied
pool is small and already the best-of-sector by construction, so a
skeptical read of exp5 is "of course insider looks good among the already-
elite ties -- would it hold up on a less pre-screened pool?"

Two pools, each date:
  top100  -- composite score == 100 (what exp5 was built/tested on)
  next11  -- the next 11 highest-scored names strictly BELOW 100 that same
             date (a distinctly less pre-screened group, same size as the
             typical top100 tied pool for a fair comparison)

Two selectors, each pool:
  random  -- uniform random draw of 5 from the pool (500-sim Monte Carlo,
             same staggered-sleeve mechanics as harness.py)
  insider -- deterministic top-5 by insider parent score within the pool

Writes output/loop_research/pool_comparison.json (full curves + stats) and
output/loop_research/pool_comparison.png (2-panel chart).
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from backtesting.data_loader import SPY, QQQ                          # noqa: E402
from research.autoresearch.evaluate import (                           # noqa: E402
    bench_returns, compute_portfolio_returns, targets_to_weight_matrix,
)
from research.loop_research import harness as H                        # noqa: E402
from research.walkforward.portfolio import performance_metrics         # noqa: E402

OUT_DIR = REPO / "output" / "loop_research"
N_SIMS = 500


def _pool(scores: pd.Series, mode: str) -> list[str]:
    s = scores.dropna()
    if mode == "top100":
        return list(s[s == 100].index)
    if mode == "next11":
        below = s[s < 100].sort_index().sort_values(ascending=False, kind="stable")
        return list(below.index[:11])
    raise ValueError(mode)


def _random_pool_selector(mode):
    def selector(scores, parent_scores, held, k, refresh_n, rng, ctx=None):
        pool = _pool(scores, mode)
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
    selector.__name__ = f"random_{mode}"
    return selector


def _insider_pool_selector(mode):
    def selector(scores, parent_scores, held, k, refresh_n, rng, ctx=None):
        pool = _pool(scores, mode)
        ins = parent_scores.get("insider", pd.Series(dtype=float)).reindex(scores.dropna().index)
        s = ins.loc[ins.index.isin(pool)].sort_index().sort_values(ascending=False, kind="stable")
        picks = list(s.index[:k])
        if len(picks) < k:
            remaining = ins[~ins.index.isin(picks)]
            remaining = remaining.sort_index().sort_values(ascending=False, kind="stable")
            picks += list(remaining.index[:k - len(picks)])
        return picks
    selector.__name__ = f"insider_{mode}"
    return selector


def _stats(pr, spy):
    m = performance_metrics(pr, hold_months=1, benchmarks={"SPY": spy})
    return {"cagr": m["cagr"], "sharpe": m["sharpe"], "sortino": m["sortino"],
            "max_dd": m["max_drawdown"], "beta": m.get("spy_beta"),
            "alpha": m.get("spy_alpha"), "ir": m.get("spy_ir")}


def run_pool(bundle, mode, n_sims=N_SIMS, seed_base=0):
    data = bundle["data"]
    rebal = list(data.rebal_dates)
    spy = bench_returns(data.matrix, SPY, rebal)
    qqq = bench_returns(data.matrix, QQQ, rebal)
    all_dates = rebal[:-1]

    random_sel = _random_pool_selector(mode)
    sim_curves = np.zeros((n_sims, len(all_dates)))
    sim_stats = {"cagr": [], "sharpe": [], "sortino": [], "max_dd": [], "beta": [], "alpha": [], "ir": []}
    for s in range(n_sims):
        rng = random.Random(seed_base + s)
        pr, turnover, targets = H.run_one(bundle, random_sel, seed_base + s)
        st = _stats(pr, spy)
        for kk, v in st.items():
            sim_stats[kk].append(v)
        pr_full = pr.reindex(all_dates).fillna(0.0)
        sim_curves[s] = (1.0 + pr_full).cumprod().values

    median_curve = np.median(sim_curves, axis=0)
    p10 = np.percentile(sim_curves, 10, axis=0)
    p90 = np.percentile(sim_curves, 90, axis=0)
    random_stats = {kk: float(np.median(v)) for kk, v in sim_stats.items()}

    insider_sel = _insider_pool_selector(mode)
    pr_ins, turnover_ins, targets_ins = H.run_one(bundle, insider_sel, seed_base)
    insider_stats = _stats(pr_ins, spy)
    insider_curve = (1.0 + pr_ins.reindex(all_dates).fillna(0.0)).cumprod().values

    diffs = np.array(sim_stats["sharpe"]) - insider_stats["sharpe"]
    win_rate_insider_vs_random = float((insider_stats["sharpe"] > np.array(sim_stats["sharpe"])).mean())

    spy_r = spy.reindex(all_dates).fillna(0.0)
    qqq_r = qqq.reindex(all_dates).fillna(0.0)

    return {
        "mode": mode,
        "dates": [str(d) for d in all_dates],
        "median_curve": median_curve.tolist(),
        "p10_curve": p10.tolist(),
        "p90_curve": p90.tolist(),
        "insider_curve": insider_curve.tolist(),
        "spy_curve": (1.0 + spy_r).cumprod().values.tolist(),
        "qqq_curve": (1.0 + qqq_r).cumprod().values.tolist(),
        "random_stats_median": random_stats,
        "insider_stats": insider_stats,
        "insider_unique_holdings": int(targets_ins["ticker"].nunique()),
        "win_rate_insider_beats_random_sim": win_rate_insider_vs_random,
    }


def main():
    bundle = H.get_data()
    results = {}
    for mode in ["top100", "next11"]:
        print(f"running pool={mode} ...", flush=True)
        results[mode] = run_pool(bundle, mode)
        r = results[mode]
        print(f"  random median sharpe={r['random_stats_median']['sharpe']:.3f}  "
             f"insider sharpe={r['insider_stats']['sharpe']:.3f}  "
             f"insider beats {r['win_rate_insider_beats_random_sim']:.1%} of random sims")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUT_DIR / "pool_comparison.json").open("w") as fh:
        json.dump(results, fh, indent=2)
    print(f"wrote {OUT_DIR / 'pool_comparison.json'}")


if __name__ == "__main__":
    main()
