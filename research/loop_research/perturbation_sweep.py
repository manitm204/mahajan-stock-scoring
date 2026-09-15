"""Perturbation robustness sweep across multiple noise multiples (user
request 2026-09-13): how much does insider_revisions_min_top20's edge over
random_top20_selector degrade as the noise added to the ranking score gets
progressively larger (0.5x, 1x, 1.5x, 2x, 3x, 4x its own monthly
cross-sectional std)?

Usage: python -m research.loop_research.perturbation_sweep
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from research.loop_research import harness as H                      # noqa: E402
from research.loop_research.exploration_batch_2026_09_11 import (    # noqa: E402
    make_perturbed_selector,
)
from research.loop_research.exploration_batch_100 import (           # noqa: E402
    metric_insider_revisions_min,
)
from research.loop_research.promotion import compute_baseline_paths   # noqa: E402

N_SIMS = 500
NOISE_MULTS = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]


def main():
    bundle = H.get_data()
    random_baseline = H.random_top20_selector

    print("computing random-baseline paths ONCE (shared across all noise levels) ...")
    _, base_metrics = compute_baseline_paths(random_baseline, bundle=bundle, n_sims=N_SIMS)
    base_sharpes = np.array([m["sharpe"] for m in base_metrics])
    print(f"random baseline: median sharpe = {np.median(base_sharpes):.4f}\n")

    print(f"{'noise_mult':>10s} {'median_sharpe':>14s} {'win_rate':>10s} {'sharpe_vs_random':>18s}")
    for mult in NOISE_MULTS:
        perturbed_selector = make_perturbed_selector(metric_insider_revisions_min, False, mult)
        sims = H.run_monte_carlo(bundle, perturbed_selector, N_SIMS, seed_base=0, deterministic=False)
        cand_sharpes = np.array([m["sharpe"] for m in sims])
        win_rate = float(np.mean(cand_sharpes > base_sharpes))
        median_sharpe = float(np.median(cand_sharpes))
        print(f"{mult:>10.1f} {median_sharpe:>14.4f} {win_rate:>9.1%} {median_sharpe - np.median(base_sharpes):>+18.4f}")


if __name__ == "__main__":
    main()
