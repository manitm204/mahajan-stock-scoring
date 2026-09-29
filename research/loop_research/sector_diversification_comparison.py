"""Sector diversification within the top-20 score pool (user request
2026-09-29, follow-up to mktcap_tilt_comparison.py -- market cap dropped):
does forcing the random 5-of-20 draw to hit 5 distinct GICS sectors (skip a
random candidate if its sector is already picked this period, draw again)
change outcomes versus the existing random_top20_selector baseline (plain
uniform random 5, sectors unconstrained)? Both selectors are stochastic, so
this compares two Monte Carlo distributions (matched seeds).

Same staggered-sleeve mechanics as harness.py (k=5, 4-month hold, 4 sleeves,
full turnover each review).

Usage: python -m research.loop_research.sector_diversification_comparison [--sims 500]
Writes: output/loop_research/sector_diversification_comparison.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from research.loop_research import harness as H

OUT = Path(__file__).resolve().parents[2] / "output" / "loop_research" / "sector_diversification_comparison.json"
N_SIMS = 500
STAT_KEYS = ["cagr", "sharpe", "sortino", "max_dd", "spy_alpha", "spy_ir"]


def _run(bundle, selector, n_sims, seed_base=0):
    data = bundle["data"]
    rebal = list(data.rebal_dates)
    all_dates = rebal[:-1]
    spy, qqq = H.bench_series(bundle)
    curves = np.zeros((n_sims, len(all_dates)))
    stats = {kk: [] for kk in STAT_KEYS}
    for s in range(n_sims):
        pr, turnover, targets = H.run_one(bundle, selector, seed_base + s)
        m = H.sim_metrics(pr, turnover, targets, spy, qqq)
        for kk in stats:
            stats[kk].append(m[kk])
        pr_full = pr.reindex(all_dates).fillna(0.0)
        curves[s] = (1.0 + pr_full).cumprod().values
    return curves, stats, all_dates


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=N_SIMS)
    args = ap.parse_args()

    bundle = H.get_data()

    variants = [
        ("random_top20", H.random_top20_selector),
        ("random_top20_sector", H.random_top20_sector_selector),
    ]

    curves, stats, dates = {}, {}, None
    for name, selector in variants:
        print(f"running {name} ...", flush=True)
        c, s, dates = _run(bundle, selector, args.sims)
        curves[name], stats[name] = c, s

    baseline_sharpe = np.array(stats["random_top20"]["sharpe"])
    result = {"n_sims": args.sims, "dates": [str(d) for d in dates]}
    for name, _ in variants:
        result[name] = {
            "median_curve": np.median(curves[name], axis=0).tolist(),
            "p10_curve": np.percentile(curves[name], 10, axis=0).tolist(),
            "p90_curve": np.percentile(curves[name], 90, axis=0).tolist(),
            "median_stats": {kk: float(np.median(v)) for kk, v in stats[name].items()},
        }
        if name != "random_top20":
            win_rate = float((np.array(stats[name]["sharpe"]) > baseline_sharpe).mean())
            result[name]["win_rate_vs_random_top20_matched_seed"] = win_rate

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as fh:
        json.dump(result, fh, indent=2)

    for name, _ in variants:
        line = f"{name:24s} median sharpe={result[name]['median_stats']['sharpe']:.3f}"
        if "win_rate_vs_random_top20_matched_seed" in result[name]:
            line += f"  beats baseline in {result[name]['win_rate_vs_random_top20_matched_seed']:.1%} of matched sims"
        print(line)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
