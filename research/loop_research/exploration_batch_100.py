"""100-idea systematic exploration batch (user request 2026-09-11: "come up
with a new strategy ... 100 different ideas/strategies ... one by one").

Honest framing: rather than hand-writing 100 individually bespoke economic
narratives (which would mostly be thin rationalizations at this volume),
this is a systematic COMBINATORIAL sweep over building blocks already
validated as meaningful in this research loop -- the same approach real
quant factor research uses at scale. Every idea is still a genuine,
distinct, testable selection rule on the top-20-by-composite-rank pool; the
systematization is in how the 100 are GENERATED, not a shortcut on how each
one is tested (each gets the full evaluate_promotion 8-gate rule + the same
perturbation-robustness gate as every hand-built candidate).

Six families (see generate_ideas() for exact composition, ~100 total):
  1. Parent-pair differences (all C(8,2)=28 pairs): rank by (parent_i -
     parent_j) descending -- "prefer parent_i strength unmatched by
     parent_j", a systematic generalization of the momentum-quality-gap
     idea tried on exp2.
  2. Representative-subfactor-pair differences (all C(8,2)=28 pairs, one
     representative subfactor chosen per parent): same idea one level more
     granular, per the finding that subfactor-level signals can behave very
     differently from their parent aggregate.
  3. Price-based multi-lookback family: momentum / volatility / risk-
     adjusted-momentum at lookbacks {21, 63, 126, 252} trading days, plus
     distance-from-N-day-high/low and moving-average-gap at multiple
     windows -- a systematic version of the single-lookback ideas tested in
     the first 10-idea batch.
  4. Score-dynamics family: composite-score momentum/stability at multiple
     lookback horizons, plus month-over-month percentile-rank change.
  5. Breadth/threshold-count family: count of parents or subfactors above
     various cross-sectional quantiles (median, top quartile), including a
     deliberate contrarian-direction negative control.
  6. Interaction family: combines the two known-strongest tie-break
     parents (insider, revisions) via product/sum/min, plus score-adjusted-
     for-risk and score-times-momentum blends, plus a rank-sum version of
     avg-parent-score.

Usage: python -m research.loop_research.exploration_batch_100
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from research.loop_research.candidates import TOP_N_POOL           # noqa: E402
from research.loop_research.exploration_batch_2026_09_11 import (  # noqa: E402
    make_selector, make_perturbed_selector,
)

PARENTS = ["momentum", "value", "quality", "growth", "revisions",
          "institutional", "insider", "short"]

MOM_LOOKBACKS = [21, 63, 126, 252]
VOL_LOOKBACKS = [21, 63, 126, 252]
RAM_LOOKBACKS = [21, 63, 126, 252]
DIST_WINDOWS = [63, 126, 252]
MA_WINDOWS = [50, 100, 150, 200]
SCORE_MOM_MONTHS = [1, 2, 3, 6, 9, 12]
SCORE_STABILITY_MONTHS = [3, 6, 9, 12]


def _rep_subfactors():
    from factors.parent_selection_v4 import SELECTED_SUBS
    return {p: list(w.keys())[0] for p, w in SELECTED_SUBS.items()}


REP_SUB = _rep_subfactors()


# --- family 1: parent-pair differences ---------------------------------
def make_parent_diff(p1, p2):
    def fn(scores, parent_scores, pool, ctx):
        a, b = parent_scores.get(p1), parent_scores.get(p2)
        if a is None or b is None:
            return None
        return (a - b).reindex(pool)
    return fn


# --- family 2: representative-subfactor-pair differences ---------------
def make_subfactor_diff(s1, s2):
    def fn(scores, parent_scores, pool, ctx):
        frame = ctx["bundle"]["subfactor_frames"].get(ctx["date"])
        if frame is None or s1 not in frame.columns or s2 not in frame.columns:
            return None
        return (frame[s1] - frame[s2]).reindex(pool)
    return fn


# --- family 3: price-based multi-lookback -------------------------------
def make_momentum(lookback):
    def fn(scores, parent_scores, pool, ctx):
        matrix = ctx["bundle"]["data"].matrix
        d = ctx["date"]
        if d not in matrix.index:
            return None
        idx = matrix.index.get_loc(d)
        start = max(idx - lookback, 0)
        cols = [t for t in pool if t in matrix.columns]
        window = matrix.iloc[start:idx + 1][cols]
        if len(window) < 2:
            return None
        return window.iloc[-1] / window.iloc[0] - 1.0
    return fn


def make_vol(lookback):
    def fn(scores, parent_scores, pool, ctx):
        matrix = ctx["bundle"]["data"].matrix
        d = ctx["date"]
        if d not in matrix.index:
            return None
        idx = matrix.index.get_loc(d)
        start = max(idx - lookback, 0)
        cols = [t for t in pool if t in matrix.columns]
        window = matrix.iloc[start:idx + 1][cols]
        rets = window.pct_change(fill_method=None).dropna(how="all")
        return rets.std()
    return fn


def make_riskadj_mom(lookback):
    mom_fn = make_momentum(lookback)
    vol_fn = make_vol(lookback)

    def fn(scores, parent_scores, pool, ctx):
        mom = mom_fn(scores, parent_scores, pool, ctx)
        vol = vol_fn(scores, parent_scores, pool, ctx)
        if mom is None or vol is None:
            return None
        return mom / vol.replace(0, float("nan"))
    return fn


def make_dist_from_high(window_len):
    def fn(scores, parent_scores, pool, ctx):
        matrix = ctx["bundle"]["data"].matrix
        d = ctx["date"]
        if d not in matrix.index:
            return None
        idx = matrix.index.get_loc(d)
        start = max(idx - window_len, 0)
        cols = [t for t in pool if t in matrix.columns]
        window = matrix.iloc[start:idx + 1][cols]
        return window.iloc[-1] / window.max() - 1.0
    return fn


def make_dist_from_low(window_len):
    def fn(scores, parent_scores, pool, ctx):
        matrix = ctx["bundle"]["data"].matrix
        d = ctx["date"]
        if d not in matrix.index:
            return None
        idx = matrix.index.get_loc(d)
        start = max(idx - window_len, 0)
        cols = [t for t in pool if t in matrix.columns]
        window = matrix.iloc[start:idx + 1][cols]
        return window.iloc[-1] / window.min() - 1.0
    return fn


def make_ma_gap(window_len):
    def fn(scores, parent_scores, pool, ctx):
        matrix = ctx["bundle"]["data"].matrix
        d = ctx["date"]
        if d not in matrix.index:
            return None
        idx = matrix.index.get_loc(d)
        start = max(idx - window_len + 1, 0)
        if idx - start < int(window_len * 0.75):
            return None
        cols = [t for t in pool if t in matrix.columns]
        window = matrix.iloc[start:idx + 1][cols]
        return window.iloc[-1] / window.mean() - 1.0
    return fn


# --- family 4: score dynamics --------------------------------------------
def make_score_momentum(months):
    def fn(scores, parent_scores, pool, ctx):
        data = ctx["bundle"]["data"]
        i, dates = ctx["i"], ctx["dates"]
        j = i - months
        if j < 0:
            return None
        now = data.comp.get(ctx["date"])
        prior = data.comp.get(dates[j])
        if now is None or prior is None:
            return None
        return now.reindex(pool) - prior.reindex(pool)
    return fn


def make_score_stability(months):
    def fn(scores, parent_scores, pool, ctx):
        data = ctx["bundle"]["data"]
        i, dates = ctx["i"], ctx["dates"]
        j = max(i - months + 1, 0)
        if j == i:
            return None
        frames = [data.comp[dd].reindex(pool) for dd in dates[j:i + 1] if dd in data.comp]
        if len(frames) < 3:
            return None
        return pd.concat(frames, axis=1).std(axis=1)
    return fn


def metric_score_rank_change(scores, parent_scores, pool, ctx):
    data = ctx["bundle"]["data"]
    i, dates = ctx["i"], ctx["dates"]
    if i == 0:
        return None
    prior = data.comp.get(dates[i - 1])
    if prior is None:
        return None
    cur_rank, prior_rank = scores.rank(pct=True), prior.rank(pct=True)
    return (cur_rank - prior_rank).reindex(pool)


# --- family 5: breadth / threshold counts --------------------------------
def make_parents_above_q(q, contrarian=False):
    def fn(scores, parent_scores, pool, ctx):
        if not parent_scores:
            return None
        df = pd.DataFrame({p: s for p, s in parent_scores.items()})
        thresh = df.quantile(q)
        above = (df >= thresh).sum(axis=1)
        return (-above if contrarian else above).reindex(pool)
    return fn


def make_subfactors_above_q(q):
    def fn(scores, parent_scores, pool, ctx):
        frame = ctx["bundle"]["subfactor_frames"].get(ctx["date"])
        if frame is None:
            return None
        thresh = frame.quantile(q)
        above = (frame >= thresh).sum(axis=1)
        return above.reindex(pool)
    return fn


# --- family 6: interactions ------------------------------------------------
def metric_insider_revisions_product(scores, parent_scores, pool, ctx):
    a, b = parent_scores.get("insider"), parent_scores.get("revisions")
    if a is None or b is None:
        return None
    return (a * b).reindex(pool)


def metric_insider_revisions_sum(scores, parent_scores, pool, ctx):
    a, b = parent_scores.get("insider"), parent_scores.get("revisions")
    if a is None or b is None:
        return None
    return (a + b).reindex(pool)


def metric_insider_revisions_min(scores, parent_scores, pool, ctx):
    a, b = parent_scores.get("insider"), parent_scores.get("revisions")
    if a is None or b is None:
        return None
    return pd.concat([a, b], axis=1).min(axis=1).reindex(pool)


def metric_insider_revisions_balance(scores, parent_scores, pool, ctx):
    """Negative control / diversification idea: prefer the SMALLEST gap
    between the two known-strongest tie-break parents (balanced strength in
    both) over having one dominate."""
    a, b = parent_scores.get("insider"), parent_scores.get("revisions")
    if a is None or b is None:
        return None
    return -(a - b).abs().reindex(pool)


def metric_score_over_vol(scores, parent_scores, pool, ctx):
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
    return scores.reindex(pool) / vol


def metric_score_times_score_momentum(scores, parent_scores, pool, ctx):
    mom_fn = make_score_momentum(3)
    mom = mom_fn(scores, parent_scores, pool, ctx)
    if mom is None:
        return None
    return scores.reindex(pool) * mom


def metric_sum_of_parent_ranks(scores, parent_scores, pool, ctx):
    if not parent_scores:
        return None
    df = pd.DataFrame({p: s for p, s in parent_scores.items()})
    ranks = df.rank(pct=True)
    return ranks.sum(axis=1).reindex(pool)


def generate_ideas():
    """Returns list of (name, hypothesis, metric_fn, ascending)."""
    ideas = []

    # Family 1: parent-pair differences (28)
    from itertools import combinations
    for p1, p2 in combinations(PARENTS, 2):
        name = f"parentdiff_{p1}_minus_{p2}_top20"
        hyp = f"prefer {p1} strength unmatched by {p2} weakness: rank by ({p1}-{p2}) score, descending"
        ideas.append((name, hyp, make_parent_diff(p1, p2), False))

    # Family 2: representative-subfactor-pair differences (28)
    for p1, p2 in combinations(PARENTS, 2):
        s1, s2 = REP_SUB[p1], REP_SUB[p2]
        name = f"subdiff_{s1}_minus_{s2}_top20"
        hyp = f"subfactor-level version: rank by ({s1}-{s2}), descending"
        ideas.append((name, hyp, make_subfactor_diff(s1, s2), False))

    # Family 3: price-based multi-lookback (4 mom + 4 vol + 4 riskadjmom + 3*2 dist + 4 ma = 23)
    for L in MOM_LOOKBACKS:
        ideas.append((f"momentum_{L}d_top20", f"highest {L}-trading-day total return", make_momentum(L), False))
    for L in VOL_LOOKBACKS:
        ideas.append((f"lowvol_{L}d_top20", f"lowest {L}-trading-day realized volatility", make_vol(L), True))
    for L in RAM_LOOKBACKS:
        ideas.append((f"riskadjmom_{L}d_top20", f"highest {L}-trading-day return/volatility ratio", make_riskadj_mom(L), False))
    for N in DIST_WINDOWS:
        ideas.append((f"near_{N}d_high_top20", f"closest to the trailing {N}-trading-day high", make_dist_from_high(N), False))
        ideas.append((f"near_{N}d_low_top20", f"closest to the trailing {N}-trading-day low (deep value/oversold)", make_dist_from_low(N), True))
    for W in MA_WINDOWS:
        ideas.append((f"above_{W}dma_top20", f"largest positive gap above the {W}-day moving average", make_ma_gap(W), False))

    # Family 4: score dynamics (6 + 4 + 1 = 11)
    for M in SCORE_MOM_MONTHS:
        ideas.append((f"scoremom_{M}m_top20", f"composite score risen the most over the trailing {M} month(s)", make_score_momentum(M), False))
    for M in SCORE_STABILITY_MONTHS:
        ideas.append((f"scorestable_{M}m_top20", f"lowest composite-score volatility over the trailing {M} months", make_score_stability(M), True))
    ideas.append(("scorerankchange_1m_top20", "biggest month-over-month improvement in composite-score percentile rank", metric_score_rank_change, False))

    # Family 5: breadth/threshold counts (2 + 2 + 1 = 5)
    ideas.append(("parents_above_top25pct_top20", "most parents scoring in the top quartile that day", make_parents_above_q(0.75), False))
    ideas.append(("parents_above_median_contrarian_top20", "FEWEST parents above the day's median (negative control)", make_parents_above_q(0.5, contrarian=True), False))
    ideas.append(("subs_above_median_top20", "most of the 24 subfactors above the day's median", make_subfactors_above_q(0.5), False))
    ideas.append(("subs_above_top25pct_top20", "most of the 24 subfactors in the day's top quartile", make_subfactors_above_q(0.75), False))
    ideas.append(("subs_above_bottom25pct_contrarian_top20", "FEWEST subfactors in the day's bottom quartile (negative control)", make_subfactors_above_q(0.25), False))

    # Family 6: interactions (7)
    ideas.append(("insider_revisions_product_top20", "reward simultaneous insider AND revisions strength (product)", metric_insider_revisions_product, False))
    ideas.append(("insider_revisions_sum_top20", "insider + revisions score sum", metric_insider_revisions_sum, False))
    ideas.append(("insider_revisions_min_top20", "BOTH insider and revisions must be strong (min)", metric_insider_revisions_min, False))
    ideas.append(("insider_revisions_balance_top20", "balanced strength across insider/revisions, not one dominating", metric_insider_revisions_balance, False))
    ideas.append(("score_over_vol_top20", "composite score adjusted for trailing 3-month risk (score/vol)", metric_score_over_vol, False))
    ideas.append(("score_times_scoremom_top20", "blend of score LEVEL and 3-month score TREND (score * scoremom)", metric_score_times_score_momentum, False))
    ideas.append(("sum_parent_ranks_top20", "rank-based (not raw-score) version of avg-parent-score, robust to scale differences", metric_sum_of_parent_ranks, False))

    return ideas
