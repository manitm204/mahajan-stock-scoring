"""Head-to-head comparison of three closely related top-20-pool strategies
(user request 2026-09-13):

1. `insider_revisions_min_top20` -- the champion: rank the whole top-20 pool
   by min(insider, revisions), no recency involved.
2. `insider_revisions_min_new_entrant_top20` -- champion floor's percentile
   rank + a flat +0.5 bonus for names new to the pool this month (the
   near-miss from the 10-idea batch).
3. `new_entrant_score_momentum_top20` (NEW, built for this comparison) --
   same additive structure as #2, but swaps the insider/revisions floor for
   1-month composite-score momentum: percentile_rank(score gain over the
   last month) + 0.5*is_new. Tests whether the near-miss's edge is about
   INSIDER/REVISIONS specifically, or would work with any reasonable
   "floor" signal riding on the same new-entrant tilt.

Usage: python -m research.loop_research.compare_three
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from research.loop_research import harness as H                      # noqa: E402
from research.loop_research.candidates import TOP_N_POOL             # noqa: E402
from research.loop_research.exploration_batch_2026_09_11 import (    # noqa: E402
    make_selector, make_perturbed_selector,
)
from research.loop_research.exploration_batch_100 import (           # noqa: E402
    metric_insider_revisions_min,
)
from research.loop_research.exploration_batch_2026_09_12 import (    # noqa: E402
    metric_insider_revisions_min_new_entrant,
)
from research.loop_research.promotion import (                       # noqa: E402
    evaluate_promotion, compute_baseline_paths, OUT_DIR,
)

N_SIMS = 500


def metric_score_momentum_1m(scores, parent_scores, pool, ctx):
    """Pure 1-month composite-score change, no new-entrant bonus and no
    restriction to new entrants -- literally 'whichever stock's composite
    score rose the most vs one month ago', across the whole top-20 pool."""
    data = ctx["bundle"]["data"]
    i, dates = ctx["i"], ctx["dates"]
    if i == 0:
        return None
    prior_scores = data.comp.get(dates[i - 1])
    if prior_scores is None:
        return None
    cur = scores.reindex(pool)
    prior = prior_scores.reindex(pool)
    return cur - prior


STRATEGIES = {
    "champion": ("insider_revisions_min_top20", metric_insider_revisions_min, False),
    "floor_plus_new_entrant": ("insider_revisions_min_new_entrant_top20", metric_insider_revisions_min_new_entrant, False),
    "score_momentum_1m": ("score_momentum_1m_top20", metric_score_momentum_1m, False),
}


def build_pick_history(selector, bundle, k=5):
    """Returns {date: set(top-k picks)} using the RAW ranking metric applied
    fresh each month (i.e. the selector's own internal top-k choice on that
    date, ignoring the sleeve-holding/refresh mechanics) -- for overlap
    analysis between strategies' underlying preferences, not the actual
    staggered book."""
    data = bundle["data"]
    dates = list(data.rebal_dates)
    parent_scores_by_date = bundle["parent_scores"]
    out = {}
    for i, d in enumerate(dates):
        s = data.comp.get(d)
        if s is None:
            continue
        ranked = s.dropna().sort_values(ascending=False, kind="stable")
        pool = list(ranked.index[:TOP_N_POOL])
        if not pool:
            continue
        ctx = {"date": d, "dates": dates, "i": i, "bundle": bundle}
        pscores = parent_scores_by_date.get(d, {})
        import random
        picks = selector(s, pscores, [], k, k, random.Random(0), ctx=ctx)
        out[d] = set(picks)
    return out


def main():
    bundle = H.get_data()

    selectors = {}
    for key, (name, metric_fn, ascending) in STRATEGIES.items():
        selectors[key] = make_selector(name, metric_fn, ascending)

    print("=== Overlap analysis: how similar are the actual monthly picks? ===\n")
    histories = {key: build_pick_history(sel, bundle) for key, sel in selectors.items()}
    common_dates = sorted(set.intersection(*[set(h.keys()) for h in histories.values()]))

    pairs = [("champion", "floor_plus_new_entrant"),
            ("champion", "score_momentum_1m"),
            ("floor_plus_new_entrant", "score_momentum_1m")]
    for a, b in pairs:
        overlaps = [len(histories[a][d] & histories[b][d]) for d in common_dates]
        print(f"{a:28s} vs {b:28s}: avg overlap {np.mean(overlaps):.2f}/5 picks "
             f"(exact match {sum(o==5 for o in overlaps)}/{len(overlaps)} months, "
             f"zero overlap {sum(o==0 for o in overlaps)}/{len(overlaps)} months)")

    print("\n=== Full-sample statistics for each (500-sim Monte Carlo) ===\n")
    print("computing shared champion baseline once ...")
    champ_selector = selectors["champion"]
    base_raw, base_metrics = compute_baseline_paths(champ_selector, bundle=bundle, n_sims=N_SIMS)

    results = {}
    for key, (name, metric_fn, ascending) in STRATEGIES.items():
        sel = selectors[key]

        def make_pert(noise_mult, _mf=metric_fn, _asc=ascending):
            return make_perturbed_selector(_mf, _asc, noise_mult)

        if key == "champion":
            report = evaluate_promotion(
                sel, baseline_selector=champ_selector, make_perturbed_selector=make_pert,
                n_sims=N_SIMS, deterministic_candidate=True, bundle=bundle,
                baseline_precomputed=(base_raw, base_metrics))
        else:
            report = evaluate_promotion(
                sel, baseline_selector=champ_selector, make_perturbed_selector=make_pert,
                n_sims=N_SIMS, deterministic_candidate=True, bundle=bundle,
                baseline_precomputed=(base_raw, base_metrics))
        results[key] = report
        out_path = OUT_DIR / f"promotion_{name}.json"
        with out_path.open("w") as fh:
            json.dump(report, fh, indent=2, default=float)

    print(f"{'strategy':30s} {'sharpe':>8s} {'alpha':>8s} {'max_dd':>8s} {'beta':>7s} "
         f"{'ΔSharpe':>9s} {'win%':>6s} {'pert%':>6s} {'yrs+':>6s}")
    for key, (name, _, _) in STRATEGIES.items():
        r = results[key]
        cm = r["candidate_metrics"]
        pw = r["perturbation_win_rate"]
        pw_s = f"{pw:.1%}" if pw is not None else "n/a"
        print(f"{name:30s} {cm['median_sharpe']:8.3f} {cm['median_alpha']:8.3f} "
             f"{cm['median_max_dd']:8.3f} {cm['median_beta']:7.3f} "
             f"{r['sharpe_improve']:+9.4f} {r['win_rate']:6.1%} {pw_s:>6s} "
             f"{r['n_years_positive']}/{r['n_years']:>4}")

    print("\nwrote full reports to output/loop_research/promotion_<name>.json")


if __name__ == "__main__":
    main()
