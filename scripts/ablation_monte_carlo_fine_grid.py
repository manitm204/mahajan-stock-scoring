"""Fine-grid follow-up to scripts/ablation_monte_carlo_random_book.py, zoomed
into the region that looked most interesting in the coarse grid (N=8-11,
hold=2-4, high refresh): sweeps N=3..10, HOLD_MONTHS=1..6, and REFRESH_N as
an absolute count from 2..N (not a fraction of N, at the user's request --
this grid asks "how many names to bring in" directly rather than "what
share of the book").

Same mechanic as the coarse grid: SLEEVE_COUNT fixed at 3, selection is a
uniform random draw from that date's composite_score==100 pool at every
fill/refill, each config scored by its own Monte Carlo distribution of
--sims independent replays.

Usage: python scripts/ablation_monte_carlo_fine_grid.py [--sims 100]
Writes: output/ablation_monte_carlo_fine_grid/results.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from backtesting.data_loader import SPY, QQQ                        # noqa: E402
from research.strategies.engine import load_data                     # noqa: E402
from scripts.ablation_monte_carlo_random_book import (               # noqa: E402
    SLEEVE_COUNT, _agg, _eta_squared, _random_targets, _sim_metrics,
)
from research.autoresearch.evaluate import (                         # noqa: E402
    compute_portfolio_returns, targets_to_weight_matrix,
)
from scripts.run_strategy_sweep import bench_returns                 # noqa: E402

OUT = REPO / "output" / "ablation_monte_carlo_fine_grid" / "results.json"

N_GRID = list(range(3, 11))       # 3..10
HOLD_MONTHS_GRID = list(range(1, 7))  # 1..6
# REFRESH_N is absolute (2..N), not a fraction -- depends on N so built per-N below


def build_configs() -> list[dict]:
    configs = []
    for n in N_GRID:
        for refresh_n in range(2, n + 1):
            for hold in HOLD_MONTHS_GRID:
                configs.append({"n": n, "hold_months": hold, "refresh_n": refresh_n})
    return configs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print("loading composite scores + price matrix ...", flush=True)
    data = load_data()
    rebal = list(data.rebal_dates)
    spy = bench_returns(data.matrix, SPY, rebal)
    qqq = bench_returns(data.matrix, QQQ, rebal)

    configs = build_configs()
    print(f"{len(configs)} configs x {args.sims} sims = "
         f"{len(configs) * args.sims} total simulations", flush=True)

    metric_names = ["cagr", "sharpe", "max_dd", "spy_beta", "spy_alpha",
                    "spy_ir", "qqq_ir", "alpha_tstat"]
    results = []
    per_config_sharpe, per_config_spy_ir = [], []
    t_start = time.time()

    for ci, cfg in enumerate(configs):
        per_sim = {m: [] for m in metric_names}
        for s in range(args.sims):
            rng = random.Random(args.seed * 1_000_003 + ci * 1009 + s)
            targets = _random_targets(data.comp, rebal, rng,
                                      cfg["n"], cfg["hold_months"], cfg["refresh_n"])
            weights = targets_to_weight_matrix(targets, rebal)
            pr, _turnover = compute_portfolio_returns(weights, data.matrix, rebal)
            sm = _sim_metrics(pr, spy, qqq)
            for m in metric_names:
                per_sim[m].append(sm[m])

        agg = {m: _agg(per_sim[m]) for m in metric_names}
        per_config_sharpe.append(per_sim["sharpe"])
        per_config_spy_ir.append(per_sim["spy_ir"])
        results.append({"config": cfg, "metrics": agg})

        if (ci + 1) % 10 == 0 or ci == len(configs) - 1:
            elapsed = time.time() - t_start
            rate = (ci + 1) / elapsed
            eta = (len(configs) - ci - 1) / rate if rate > 0 else float("nan")
            print(f"  config {ci + 1}/{len(configs)}  "
                 f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)", flush=True)

    eta2_sharpe = _eta_squared(per_config_sharpe)
    eta2_spy_ir = _eta_squared(per_config_spy_ir)

    ranked = sorted(results, key=lambda r: r["metrics"]["spy_ir"]["mean"]
                    if r["metrics"]["spy_ir"]["mean"] == r["metrics"]["spy_ir"]["mean"]
                    else -1e9, reverse=True)

    print("\n=== variance decomposition (does configuration matter?) ===")
    print(f"  eta^2(sharpe, config)  = {eta2_sharpe:.3f}  "
         f"({eta2_sharpe * 100:.1f}% of Sharpe variance explained by config choice)")
    print(f"  eta^2(spy_ir, config)  = {eta2_spy_ir:.3f}  "
         f"({eta2_spy_ir * 100:.1f}% of SPY-IR variance explained by config choice)")
    print("\n=== top 5 configs by mean SPY-IR ===")
    for r in ranked[:5]:
        c, m = r["config"], r["metrics"]
        print(f"  N={c['n']:<3} hold={c['hold_months']:<3} refresh={c['refresh_n']}/{c['n']}  "
             f"spy_ir={m['spy_ir']['mean']:.3f}+/-{m['spy_ir']['std']:.3f}  "
             f"sharpe={m['sharpe']['mean']:.3f}  cagr={m['cagr']['mean']:.3f}")
    print("\n=== bottom 5 configs by mean SPY-IR ===")
    for r in ranked[-5:]:
        c, m = r["config"], r["metrics"]
        print(f"  N={c['n']:<3} hold={c['hold_months']:<3} refresh={c['refresh_n']}/{c['n']}  "
             f"spy_ir={m['spy_ir']['mean']:.3f}+/-{m['spy_ir']['std']:.3f}  "
             f"sharpe={m['sharpe']['mean']:.3f}  cagr={m['cagr']['mean']:.3f}")

    out = {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "params": {"sleeve_count": SLEEVE_COUNT, "n_grid": N_GRID,
                   "hold_months_grid": HOLD_MONTHS_GRID,
                   "refresh_mode": "absolute_2_to_N",
                   "sims_per_config": args.sims, "seed": args.seed,
                   "selection": "uniform random draw from composite_score==100 pool"},
        "variance_decomposition": {"eta_squared_sharpe": eta2_sharpe,
                                   "eta_squared_spy_ir": eta2_spy_ir},
        "results": results,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as fh:
        json.dump(out, fh, indent=2, default=lambda x: None if x != x else x)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
