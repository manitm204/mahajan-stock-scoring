"""10 new top-20-pool ideas (user request 2026-09-12), tested one at a time
against the NEW champion baseline `insider_revisions_min_top20` (not the
random baseline used by every earlier batch) -- i.e. "can anything beat the
current champion", not "can anything beat random". Each idea is proposed,
tested, and recorded before the next is written, per user instruction.

Usage: python -m research.loop_research.exploration_batch_2026_09_12 <idea_key>
Writes: output/loop_research/promotion_<name>.json (full evaluate_promotion
report) and output/loop_research/experiments/<name>.json (compact summary),
same format as every prior candidate in this research loop.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from research.loop_research.exploration_batch_2026_09_11 import (   # noqa: E402
    make_selector, make_perturbed_selector,
)
from research.loop_research.exploration_batch_100 import (          # noqa: E402
    metric_insider_revisions_min, make_score_momentum, REP_SUB,
)
from research.loop_research.promotion import (                      # noqa: E402
    evaluate_promotion, build_summary, print_report, OUT_DIR,
)

import json  # noqa: E402

BASELINE_SELECTOR = make_selector(
    "insider_revisions_min_top20", metric_insider_revisions_min, ascending=False)


# ---------------------------------------------------------------------------
# Idea 1: three-way floor -- insider AND revisions AND quality must all clear
# a bar (extend the winning MIN/floor logic with a third already-validated
# parent, rather than just the pair).
def metric_min3_insider_revisions_quality(scores, parent_scores, pool, ctx):
    a = parent_scores.get("insider")
    b = parent_scores.get("revisions")
    c = parent_scores.get("quality")
    if a is None or b is None or c is None:
        return None
    return pd.concat([a, b, c], axis=1).min(axis=1).reindex(pool)


def metric_min3_insider_revisions_institutional(scores, parent_scores, pool, ctx):
    a = parent_scores.get("insider")
    b = parent_scores.get("revisions")
    c = parent_scores.get("institutional")
    if a is None or b is None or c is None:
        return None
    return pd.concat([a, b, c], axis=1).min(axis=1).reindex(pool)


SCORE_MOM_MONTHS = 3


def metric_insider_momentum(scores, parent_scores, pool, ctx):
    """Insider PARENT score risen the most over the trailing 3 months --
    distinct from composite-score momentum (already tested/rejected):
    isolates whether the INSIDER signal specifically is re-rating, not the
    whole composite."""
    data = ctx["bundle"]["data"]
    i, dates = ctx["i"], ctx["dates"]
    j = i - SCORE_MOM_MONTHS
    if j < 0:
        return None
    now_parents = ctx["bundle"]["parent_scores"].get(ctx["date"], {})
    prior_parents = ctx["bundle"]["parent_scores"].get(dates[j], {})
    now, prior = now_parents.get("insider"), prior_parents.get("insider")
    if now is None or prior is None:
        return None
    return now.reindex(pool) - prior.reindex(pool)


def metric_insider_revisions_rankproduct(scores, parent_scores, pool, ctx):
    """Percentile-rank-based product of insider and revisions, instead of
    raw scores -- the raw-score product (insider_revisions_product_top20,
    tested 2026-09-11/12) failed (bootstrap p=0.335); check whether that was
    a scale/skew artifact of raw scores rather than the product-interaction
    logic itself."""
    a = parent_scores.get("insider")
    b = parent_scores.get("revisions")
    if a is None or b is None:
        return None
    ra = a.rank(pct=True)
    rb = b.rank(pct=True)
    return (ra * rb).reindex(pool)


def metric_insider_revisions_min_riskadj(scores, parent_scores, pool, ctx):
    """Risk-adjust the champion's own floor signal: min(insider, revisions)
    divided by trailing 3-month realized volatility -- distinct from every
    prior riskadj idea (those applied vol-adjustment to PRICE momentum, not
    to a fundamental-signal floor)."""
    a, b = parent_scores.get("insider"), parent_scores.get("revisions")
    if a is None or b is None:
        return None
    floor = pd.concat([a, b], axis=1).min(axis=1)
    matrix = ctx["bundle"]["data"].matrix
    d = ctx["date"]
    if d not in matrix.index:
        return None
    idx = matrix.index.get_loc(d)
    start = max(idx - 63, 0)
    cols = [t for t in pool if t in matrix.columns]
    window = matrix.iloc[start:idx + 1][cols]
    rets = window.pct_change(fill_method=None).dropna(how="all")
    vol = rets.std().replace(0, float("nan"))
    return (floor.reindex(pool) / vol)


INSIDER_STABILITY_MONTHS = 6


def metric_insider_stability(scores, parent_scores, pool, ctx):
    """Steady insider conviction: lowest 6-month volatility of the INSIDER
    parent score -- distinct from insider_momentum (chasing recent change,
    already rejected) and composite-score stability (already rejected)."""
    data = ctx["bundle"]["data"]
    i, dates = ctx["i"], ctx["dates"]
    j = max(i - INSIDER_STABILITY_MONTHS + 1, 0)
    if j == i:
        return None
    frames = []
    for dd in dates[j:i + 1]:
        p = ctx["bundle"]["parent_scores"].get(dd, {})
        ins = p.get("insider")
        if ins is not None:
            frames.append(ins.reindex(pool))
    if len(frames) < 3:
        return None
    return pd.concat(frames, axis=1).std(axis=1)


def metric_insider_revisions_max(scores, parent_scores, pool, ctx):
    """OR logic: EITHER insider or revisions strong is enough (max) --
    direct contrast to the champion's AND/min logic, to confirm requiring
    BOTH is what actually drives the edge, not just 'insider or revisions
    is a good tie-break signal' in general."""
    a, b = parent_scores.get("insider"), parent_scores.get("revisions")
    if a is None or b is None:
        return None
    return pd.concat([a, b], axis=1).max(axis=1).reindex(pool)


def metric_subfactor_insider_revisions_min(scores, parent_scores, pool, ctx):
    """Finer-grained version of the champion: min of the single BEST
    representative insider subfactor and BEST representative revisions
    subfactor, computed directly at the subfactor level rather than the
    parent aggregate -- tests whether granularity below the parent helps or
    just adds noise."""
    s1, s2 = REP_SUB["insider"], REP_SUB["revisions"]
    frame = ctx["bundle"]["subfactor_frames"].get(ctx["date"])
    if frame is None or s1 not in frame.columns or s2 not in frame.columns:
        return None
    return frame[[s1, s2]].min(axis=1).reindex(pool)


def metric_insider_revisions_min_new_entrant(scores, parent_scores, pool, ctx):
    """Combine the champion floor with the (promising-but-unconfirmed-alone)
    freshness idea: among floor-qualified names, boost ones that were JUST
    promoted into the top-20 pool this month. Tests whether 'new entrant'
    adds anything on top of the floor rather than needing to stand alone."""
    a, b = parent_scores.get("insider"), parent_scores.get("revisions")
    if a is None or b is None:
        return None
    floor = pd.concat([a, b], axis=1).min(axis=1).reindex(pool)
    data = ctx["bundle"]["data"]
    i, dates = ctx["i"], ctx["dates"]
    if i == 0:
        return floor
    prior_scores = data.comp.get(dates[i - 1])
    if prior_scores is None:
        return floor
    prior_ranked = prior_scores.dropna().sort_values(ascending=False, kind="stable")
    from research.loop_research.candidates import TOP_N_POOL
    prior_pool = set(prior_ranked.index[:TOP_N_POOL])
    is_new = pd.Series({t: (1.0 if t not in prior_pool else 0.0) for t in pool})
    floor_rank = floor.rank(pct=True)
    return floor_rank + is_new * 0.5


TENURE_MAX_LOOKBACK_MONTHS = 24


def metric_insider_revisions_min_inverse_tenure(scores, parent_scores, pool, ctx):
    """Refine idea 9: replace the BINARY 'new this month' flag with a
    CONTINUOUS inverse-tenure bonus (fewer consecutive months already in the
    top-20 pool = bigger bonus) on top of the champion floor -- tests
    whether idea 9's gain is a general recency effect or specific to the
    exact 'brand new this month' cutoff."""
    a, b = parent_scores.get("insider"), parent_scores.get("revisions")
    if a is None or b is None:
        return None
    floor_rank = pd.concat([a, b], axis=1).min(axis=1).reindex(pool).rank(pct=True)

    data = ctx["bundle"]["data"]
    i, dates = ctx["i"], ctx["dates"]
    from research.loop_research.candidates import TOP_N_POOL
    tenure = {}
    for t in pool:
        cnt = 0
        for back in range(0, TENURE_MAX_LOOKBACK_MONTHS + 1):
            j = i - back
            if j < 0:
                break
            s = data.comp.get(dates[j])
            if s is None:
                break
            ranked = s.dropna().sort_values(ascending=False, kind="stable")
            if t in set(ranked.index[:TOP_N_POOL]):
                cnt += 1
            else:
                break
        tenure[t] = cnt
    tenure_s = pd.Series(tenure)
    inverse_tenure_bonus = 1.0 - (tenure_s / TENURE_MAX_LOOKBACK_MONTHS).clip(0, 1)
    return floor_rank + 0.5 * inverse_tenure_bonus


IDEAS = {
    "min3_insider_revisions_quality": (
        "min3_insider_revisions_quality_top20",
        "three-way floor: insider AND revisions AND quality must ALL be strong "
        "(min of all three), extending the winning insider/revisions MIN logic "
        "with a third already-validated parent",
        metric_min3_insider_revisions_quality, False,
    ),
    "min3_insider_revisions_institutional": (
        "min3_insider_revisions_institutional_top20",
        "three-way floor: insider AND revisions AND institutional ownership "
        "change must ALL be strong (min of all three) -- swap quality for "
        "institutional as the third leg, since idea 1 (quality) failed badly",
        metric_min3_insider_revisions_institutional, False,
    ),
    "insider_momentum": (
        "insider_momentum_top20",
        "insider PARENT score risen the most over the trailing 3 months -- "
        "isolates whether the insider signal itself is re-rating, distinct "
        "from composite-score momentum (already tested/rejected)",
        metric_insider_momentum, False,
    ),
    "insider_revisions_rankproduct": (
        "insider_revisions_rankproduct_top20",
        "percentile-rank product of insider and revisions (rank(insider)*"
        "rank(revisions)), testing whether the raw-score product's failure "
        "(p=0.335) was a scale artifact, not a flaw in the product logic",
        metric_insider_revisions_rankproduct, False,
    ),
    "insider_revisions_min_riskadj": (
        "insider_revisions_min_riskadj_top20",
        "risk-adjust the champion's own floor: min(insider,revisions) "
        "divided by trailing 3-month realized volatility",
        metric_insider_revisions_min_riskadj, False,
    ),
    "insider_stability": (
        "insider_stability_top20",
        "steady insider conviction: lowest 6-month volatility of the "
        "insider PARENT score (not composite), among the top-20 pool",
        metric_insider_stability, True,
    ),
    "insider_revisions_max": (
        "insider_revisions_max_top20",
        "OR logic (max of insider,revisions) as a contrast to the "
        "champion's AND/min logic -- tests whether requiring BOTH is what "
        "drives the edge",
        metric_insider_revisions_max, False,
    ),
    "subfactor_insider_revisions_min": (
        "subfactor_insider_revisions_min_top20",
        "finer-grained version of the champion: min of the single best "
        "representative insider subfactor and best revisions subfactor, at "
        "the subfactor level instead of the parent aggregate",
        metric_subfactor_insider_revisions_min, False,
    ),
    "insider_revisions_min_new_entrant": (
        "insider_revisions_min_new_entrant_top20",
        "champion floor (percentile rank) plus a bonus for names freshly "
        "promoted into the top-20 pool this month -- tests whether the "
        "promising-but-unconfirmed-alone new-entrant idea adds value on top "
        "of the floor",
        metric_insider_revisions_min_new_entrant, False,
    ),
    "insider_revisions_min_inverse_tenure": (
        "insider_revisions_min_inverse_tenure_top20",
        "champion floor plus a CONTINUOUS inverse-tenure bonus (fewer "
        "months already in the pool = bigger bonus), refining idea 9's "
        "binary new-entrant flag into a general recency continuum",
        metric_insider_revisions_min_inverse_tenure, False,
    ),
}


def run_experiment(key: str, n_sims: int = 500):
    name, hypothesis, metric_fn, ascending = IDEAS[key]
    candidate_selector = make_selector(name, metric_fn, ascending)

    def make_pert(noise_mult):
        return make_perturbed_selector(metric_fn, ascending, noise_mult)

    print(f"=== {name} ===\nhypothesis: {hypothesis}\nbaseline: insider_revisions_min_top20 (current champion)\n")
    report = evaluate_promotion(
        candidate_selector, baseline_selector=BASELINE_SELECTOR,
        make_perturbed_selector=make_pert, n_sims=n_sims,
        deterministic_candidate=True,
    )
    print_report(report)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"promotion_{name}.json"
    with out_path.open("w") as fh:
        json.dump(report, fh, indent=2, default=float)
    print(f"\nwrote {out_path}")

    summary = build_summary(report, name, hypothesis)
    summary["baseline"] = "insider_revisions_min_top20"
    exp_dir = OUT_DIR / "experiments"
    exp_dir.mkdir(parents=True, exist_ok=True)
    summary_path = exp_dir / f"{name}.json"
    with summary_path.open("w") as fh:
        json.dump(summary, fh, indent=2, default=float)
    print(f"wrote {summary_path}")
    return report, summary


if __name__ == "__main__":
    key = sys.argv[1]
    run_experiment(key)
