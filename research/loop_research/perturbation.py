"""Perturbation Monte Carlo for a DETERMINISTIC selector (user request
2026-09-09): top5_by_insider produces the exact same 140-ticker history every
time, so the earlier "500 sims" only ever compared that one fixed path
against 500 different random-baseline paths -- it never gave the candidate
itself any variance, so a 98% win rate there could just as easily mean
"insider score cleanly separates good picks from bad" as "the top-ranked
names happened to be a few ties away from a very different, much worse book,
and we got lucky."

This adds Gaussian noise (scaled to that date's tied-pool insider-score
spread) to the insider score before ranking, independently per sim, then
reselects top-5 under the perturbed ranking. If the edge survives realistic
noise, the top insider names are genuinely separated from the rest, not
riding on a razor-thin, noise-sensitive ranking.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from research.loop_research import harness as H   # noqa: E402


def make_perturbed_single_parent_selector(parent: str, noise_mult: float):
    """Generalized version of the insider-specific perturbation above: adds
    Gaussian noise (scaled to that date's tied-pool cross-sectional std of
    `parent`'s score) before ranking, independently per sim, then reselects
    top-k under the perturbed ranking. Works for any single-parent-tie-break
    selector built via batch_candidates.make_single_parent_top5_selector."""
    def selector(scores, parent_scores, held, k, refresh_n, rng, ctx=None):
        pool = list(scores[scores == 100].index)
        pscore = parent_scores.get(parent, pd.Series(dtype=float)).reindex(scores.dropna().index)
        pool_vals = pscore.reindex(pool).dropna()
        base_std = float(pool_vals.std()) if len(pool_vals) >= 2 and pool_vals.std() == pool_vals.std() else 1.0
        base_std = base_std or 1.0

        def _noisy_rank(names):
            noise = pd.Series({t: rng.gauss(0, noise_mult * base_std) for t in names})
            vals = pscore.reindex(names).fillna(0) + noise
            return vals.sort_index().sort_values(ascending=False, kind="stable")

        ranked = _noisy_rank(pool)
        picks = list(ranked.index[:k])
        if len(picks) < k:
            remaining_pool = [t for t in scores.dropna().index if t not in picks]
            remaining_ranked = _noisy_rank(remaining_pool)
            picks += list(remaining_ranked.index[:k - len(picks)])
        return picks
    selector.__name__ = f"{parent}_perturbed_{noise_mult}"
    return selector


def make_perturbed_insider_selector(noise_mult: float):
    """Back-compat alias -- exp5 (top5_by_insider) is the insider special case
    of the generalized single-parent perturbation above."""
    return make_perturbed_single_parent_selector("insider", noise_mult)


def overlap_with_exact(bundle, noise_mult, n_probe_dates=20, seed=0):
    """Diagnostic: at how many of a probe of rebal dates does the perturbed
    top-5 differ from the exact deterministic top-5, and by how much
    (Jaccard overlap)? Cheap sanity check alongside the full Sharpe-level
    Monte Carlo."""
    import random
    from research.loop_research import candidates as C
    data = bundle["data"]
    dates = list(data.rebal_dates)
    rng = random.Random(seed)
    perturbed_sel = make_perturbed_insider_selector(noise_mult)
    overlaps = []
    for d in dates[-n_probe_dates:]:
        scores = data.comp.get(d)
        pscores = bundle["parent_scores"].get(d, {})
        if scores is None:
            continue
        ctx = {"date": d, "dates": dates, "i": dates.index(d), "bundle": bundle}
        exact = set(C.top5_by_insider(scores, pscores, [], H.N, H.REFRESH_N, rng, ctx=ctx))
        pert = set(perturbed_sel(scores, pscores, [], H.N, H.REFRESH_N, rng, ctx=ctx))
        jaccard = len(exact & pert) / len(exact | pert) if (exact | pert) else 1.0
        overlaps.append(jaccard)
    return sum(overlaps) / len(overlaps) if overlaps else float("nan")
