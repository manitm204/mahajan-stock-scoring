"""Pure random pick among new entrants (user request 2026-09-13): each
month, identify which stocks just entered the top-20 pool (weren't there
last month), then pick 5 of THEM entirely at random -- no insider/revisions
filtering, no score-based tie-break at all. This isolates whether "buy
freshly-arrived names" has any edge on its own, stripped of the
insider/revisions floor that `insider_revisions_min_new_entrant_top20`
layers on top of it.

Genuinely stochastic (not a deterministic ranking + noise), so each of the
Monte Carlo sims below is a REAL, independently-randomized realization, not
a noised version of one fixed choice.

Usage: python -m research.loop_research.random_new_entrant
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from research.loop_research import harness as H            # noqa: E402
from research.loop_research.candidates import TOP_N_POOL   # noqa: E402
from research.loop_research.promotion import compute_baseline_paths  # noqa: E402

N_SIMS = 100


def random_new_entrant_selector(scores, parent_scores, held, k, refresh_n, rng, ctx=None):
    ranked = scores.dropna().sort_values(ascending=False, kind="stable")
    pool = list(ranked.index[:TOP_N_POOL])
    if not pool:
        return held
    data = ctx["bundle"]["data"]
    i, dates = ctx["i"], ctx["dates"]
    if i == 0:
        rng.shuffle(pool)
        return pool[:k]
    prior_scores = data.comp.get(dates[i - 1])
    if prior_scores is None:
        rng.shuffle(pool)
        return pool[:k]
    prior_pool = set(prior_scores.dropna().sort_values(ascending=False, kind="stable").index[:TOP_N_POOL])
    new_entrants = [t for t in pool if t not in prior_pool]
    rng.shuffle(new_entrants)
    picks = new_entrants[:k]
    if len(picks) < k:
        remaining = [t for t in pool if t not in picks]
        rng.shuffle(remaining)
        picks += remaining[:k - len(picks)]
    return picks


def main():
    bundle = H.get_data()
    spy, qqq = H.bench_series(bundle)

    print(f"running {N_SIMS} real Monte Carlo simulations of 'random pick among new entrants' ...")
    sims = H.run_monte_carlo(bundle, random_new_entrant_selector, N_SIMS, seed_base=0, deterministic=False)
    df = pd.DataFrame(sims)

    print("computing random_top20_selector baseline (same N_SIMS, for a paired win rate) ...")
    _, base_metrics = compute_baseline_paths(H.random_top20_selector, bundle=bundle, n_sims=N_SIMS)
    base_df = pd.DataFrame(base_metrics)

    metrics = ["cagr", "sharpe", "sortino", "max_dd", "avg_turnover", "spy_beta",
              "spy_alpha", "spy_ir", "qqq_ir", "unique_holdings", "total_return"]

    print(f"\n=== random_new_entrant_top20: {N_SIMS} sims -- median / mean / std / min / max ===")
    print(f"{'metric':16s} {'median':>10s} {'mean':>10s} {'std':>10s} {'min':>10s} {'max':>10s}")
    for k in metrics:
        vals = df[k].values
        print(f"{k:16s} {np.median(vals):>10.4f} {np.mean(vals):>10.4f} {np.std(vals):>10.4f} {np.min(vals):>10.4f} {np.max(vals):>10.4f}")

    win_rate = float(np.mean(df["sharpe"].values > base_df["sharpe"].values))
    print(f"\nwin rate vs random_top20_selector (paired by sim index): {win_rate:.1%}")
    print(f"median sharpe: random_new_entrant={df['sharpe'].median():.4f}  random_top20={base_df['sharpe'].median():.4f}")
    print(f"median alpha:  random_new_entrant={df['spy_alpha'].median():.4f}  random_top20={base_df['spy_alpha'].median():.4f}")


if __name__ == "__main__":
    main()
