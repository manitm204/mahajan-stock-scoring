"""One selector function per experiment. Each is deliberately small and
economically motivated -- see session_log.md for the hypothesis behind each
one and its result. This file accumulates every experiment ever tried,
accepted or not (per the research-hygiene rule: never delete a failure)."""
from __future__ import annotations

import pandas as pd

TOP_N_POOL = 20
_ALL_SUB_COLS = None


def _all_sub_cols():
    """Union of the 24 SELECTED_SUBS subfactor columns across all 8 V4
    parents (cached at module scope -- SELECTED_SUBS is a static constant)."""
    global _ALL_SUB_COLS
    if _ALL_SUB_COLS is None:
        from factors.parent_selection_v4 import SELECTED_SUBS
        cols = set()
        for weights in SELECTED_SUBS.values():
            cols.update(weights.keys())
        _ALL_SUB_COLS = sorted(cols)
    return _ALL_SUB_COLS


WEEK_LOOKBACK_DAYS = 5


def worst_week_return_top20(scores, parent_scores, held, k, refresh_n, rng, ctx=None):
    """New hypothesis (2026-09-11, user request): short-term oversold /
    mean-reversion idea -- within the top-20-by-composite-rank pool, prefer
    the names that fell the MOST over the trailing ~1 week (5 trading days
    ending at the review date), instead of any score-based quality proxy.
    No look-ahead: only uses prices at/before the review date (current price
    vs the price 5 trading days earlier, both already realized by d).
    Deterministic tie-break: ticker ascending. Falls back to a random draw
    from the pool if price history isn't available that far back."""
    ranked = scores.dropna().sort_values(ascending=False, kind="stable")
    pool = list(ranked.index[:TOP_N_POOL])
    if not pool:
        return held

    bundle = (ctx or {}).get("bundle")
    matrix = bundle["data"].matrix if bundle else None
    d = ctx["date"] if ctx else None
    if matrix is None or d not in matrix.index:
        rng.shuffle(pool)
        return pool[:k]

    idx = matrix.index.get_loc(d)
    prior_idx = max(idx - WEEK_LOOKBACK_DAYS, 0)
    prior_date = matrix.index[prior_idx]
    px_now = matrix.loc[d, [t for t in pool if t in matrix.columns]]
    px_prior = matrix.loc[prior_date, px_now.index]
    week_ret = (px_now / px_prior - 1.0).dropna()

    week_ret = week_ret.sort_index().sort_values(ascending=True, kind="stable")
    picks = list(week_ret.index[:k])
    if len(picks) < k:
        remaining = [t for t in pool if t not in picks]
        picks += remaining[:k - len(picks)]
    return picks


def avg_top3_parent_score_top20(scores, parent_scores, held, k, refresh_n, rng, ctx=None):
    """New baseline generation (2026-09-11): within the top-20-by-composite-
    rank pool, for each ticker average its 3 HIGHEST parent scores (out of
    the 8 V4 parents) -- rewards names with a few standout strengths rather
    than requiring uniform strength (avg_parent_score_top5's all-8-parent
    average) or penalizing any single weak spot (min_parent_score_top5).
    Rank pool descending by that top-3 average, take top k. Deterministic
    tie-break: ticker ascending."""
    ranked = scores.dropna().sort_values(ascending=False, kind="stable")
    pool = list(ranked.index[:TOP_N_POOL])
    if not pool:
        return held

    df = pd.DataFrame({p: s for p, s in parent_scores.items()})
    top3_avg = df.apply(lambda row: row.dropna().nlargest(3).mean(), axis=1)

    pool_ranked = top3_avg.reindex(pool).dropna()
    pool_ranked = pool_ranked.sort_index().sort_values(ascending=False, kind="stable")
    picks = list(pool_ranked.index[:k])
    if len(picks) < k:
        remaining = top3_avg[~top3_avg.index.isin(picks)]
        remaining = remaining.sort_index().sort_values(ascending=False, kind="stable")
        picks += list(remaining.index[:k - len(picks)])
    return picks


def min_subfactor_score_top20(scores, parent_scores, held, k, refresh_n, rng, ctx=None):
    """New baseline generation (2026-09-11): within the top-20-by-composite-
    rank pool (harness.random_top20_selector's pool, not the old ==100 tied
    pool), rank by the MINIMUM across all 24 individual subfactor scores
    (not the 8 parent aggregates -- a finer-grained version of exp1's
    min_parent_score_top5 idea) and take the top k. Deterministic tie-break:
    ticker ascending. Requires harness.get_subfactor_frames() to be present
    at ctx['bundle']['subfactor_frames'] (see run scripts)."""
    ranked = scores.dropna().sort_values(ascending=False, kind="stable")
    pool = list(ranked.index[:TOP_N_POOL])
    if not pool:
        return held

    sub_frames = (ctx or {}).get("bundle", {}).get("subfactor_frames", {})
    frame = sub_frames.get(ctx["date"]) if ctx else None
    if frame is None:
        # No subfactor data for this date -- fall back to the pool as-is,
        # random order (should not happen once subfactor_frames is wired in).
        rng.shuffle(pool)
        min_scores = pd.Series(0.0, index=pool)
    else:
        cols = [c for c in _all_sub_cols() if c in frame.columns]
        min_scores = frame[cols].min(axis=1)

    pool_ranked = min_scores.reindex(pool).dropna()
    pool_ranked = pool_ranked.sort_index().sort_values(ascending=False, kind="stable")
    picks = list(pool_ranked.index[:k])
    if len(picks) < k:
        remaining = min_scores[~min_scores.index.isin(picks)]
        remaining = remaining.sort_index().sort_values(ascending=False, kind="stable")
        picks += list(remaining.index[:k - len(picks)])
    return picks


def _min_parent_rank(scores: pd.Series, parent_scores: dict) -> pd.Series:
    """min across the 8 V4 parent scores, for every ticker scored that date."""
    df = pd.DataFrame({p: s for p, s in parent_scores.items()})
    df = df.loc[df.index.isin(scores.dropna().index)]
    return df.min(axis=1)


def min_parent_score_top5(scores, parent_scores, held, k, refresh_n, rng, ctx=None):
    """Experiment 1 (2026-09-09): among the composite==100 tied pool, prefer
    names with no glaring weak spot in any single parent family, instead of
    a uniform random draw. Rank by min(8 parent scores), take the top k.
    Deterministic tie-break: ticker ascending. Falls back to the full scored
    universe (still ranked by min-parent-score) if the ==100 pool is smaller
    than k."""
    pool = scores[scores == 100].index
    min_scores = _min_parent_rank(scores, parent_scores)
    pool_ranked = min_scores.loc[min_scores.index.isin(pool)]
    pool_ranked = pool_ranked.sort_index().sort_values(ascending=False, kind="stable")
    picks = list(pool_ranked.index[:k])
    if len(picks) < k:
        remaining = min_scores[~min_scores.index.isin(picks)]
        remaining = remaining.sort_index().sort_values(ascending=False, kind="stable")
        picks += list(remaining.index[:k - len(picks)])
    return picks


def avg_parent_score_top5(scores, parent_scores, held, k, refresh_n, rng, ctx=None):
    """Experiment 3 (2026-09-09): exp1's min-parent-score rule improved
    Sharpe (74% win rate) but was inconclusive on CI width; exp2's
    momentum-quality-gap rule improved CAGR/alpha/drawdown but hurt Sharpe,
    likely via idiosyncratic concentration in a narrow recurring subset.
    Try the broader, less narrowing middle ground on the pre-registered
    experiment list: rank by the simple average of all 8 V4 parent scores
    (not just the min, not just two parents), take the top k. Deterministic
    tie-break: ticker ascending."""
    df = pd.DataFrame({p: s for p, s in parent_scores.items()})
    avg = df.mean(axis=1).reindex(scores.dropna().index)
    pool = scores[scores == 100].index
    pool_avg = avg.loc[avg.index.isin(pool)]
    pool_avg = pool_avg.sort_index().sort_values(ascending=False, kind="stable")
    picks = list(pool_avg.index[:k])
    if len(picks) < k:
        remaining = avg[~avg.index.isin(picks)]
        remaining = remaining.sort_index().sort_values(ascending=False, kind="stable")
        picks += list(remaining.index[:k - len(picks)])
    return picks


def avg_parent_score_weighted_random(scores, parent_scores, held, k, refresh_n, rng, ctx=None):
    """Experiment 4 (2026-09-09): exp1/exp2/exp3 all deterministically
    top-ranked the pool and all three share the same recurring weakness --
    a bad 2022, the one clear regime-shock year in this window -- consistent
    with a working theory that the random baseline's incidental
    diversification across the tied pool acted as an accidental hedge that
    deterministic top-ranking gives up. Test whether keeping that
    diversification while still tilting toward better-supported names beats
    both the pure-random baseline and the pure-deterministic exp3: sample k
    names WITHOUT replacement from the pool, weighted by
    max(avg_parent_score, 1) (never zero-out a tied-100 name entirely --
    the composite already screened for that), instead of hard-ranking.
    Uses the per-sim rng, so like the baseline this has its own Monte Carlo
    distribution (not compared as a single deterministic run)."""
    pool = list(scores[scores == 100].index)
    if not pool:
        return held
    df = pd.DataFrame({p: s for p, s in parent_scores.items()})
    avg = df.mean(axis=1)
    weights = [max(float(avg.get(t, 1.0)), 1.0) for t in pool]

    def _weighted_sample(names, w, n):
        names, w = list(names), list(w)
        picks = []
        for _ in range(min(n, len(names))):
            total = sum(w)
            r = rng.random() * total
            acc = 0.0
            for i, wi in enumerate(w):
                acc += wi
                if acc >= r:
                    picks.append(names.pop(i))
                    w.pop(i)
                    break
        return picks

    if not held:
        return _weighted_sample(pool, weights, k)
    keep = list(held)
    n_evict = min(refresh_n, len(keep))
    to_evict = set(rng.sample(keep, n_evict))
    keep = [t for t in keep if t not in to_evict]
    need = k - len(keep)
    cand_pool = [t for t in pool if t not in keep]
    cand_w = [max(float(avg.get(t, 1.0)), 1.0) for t in cand_pool]
    fill = _weighted_sample(cand_pool, cand_w, need)
    if len(fill) < need:
        universe = [t for t in scores.dropna().index if t not in keep and t not in fill]
        rng.shuffle(universe)
        fill += universe[:need - len(fill)]
    return keep + fill


def top5_by_insider(scores, parent_scores, held, k, refresh_n, rng, ctx=None):
    """exp5 (2026-09-09), champion: the single strongest result out of a
    93-candidate batch (research/loop_research/batch_candidates.py, waveC).
    Among the score==100 pool, prefer the single highest insider parent
    score (ins_no_selling_flag/ins_cluster_buyers_180d/ins_sell_pressure_inv),
    ignore every other parent. Deterministic tie-break: ticker ascending.
    500-sim confirmation: win rate 98.4%, 95% CI [+0.037, +0.457] (does not
    touch zero), wins 6 of 7 calendar years incl. 2022 -- the one year every
    other parent-score-based candidate lost. See session_log.md / champion.md."""
    pool = scores[scores == 100].index
    ins = parent_scores.get("insider", pd.Series(dtype=float)).reindex(scores.dropna().index)
    return _rank_pick_local(ins, pool, k)


def _rank_pick_local(series, pool, k):
    s = series.loc[series.index.isin(pool)]
    s = s.sort_index().sort_values(ascending=False, kind="stable")
    picks = list(s.index[:k])
    if len(picks) < k:
        remaining = series[~series.index.isin(picks)]
        remaining = remaining.sort_index().sort_values(ascending=False, kind="stable")
        picks += list(remaining.index[:k - len(picks)])
    return picks


def momentum_overextension_penalty_top5(scores, parent_scores, held, k, refresh_n, rng, ctx=None):
    """Experiment 2 (2026-09-09): motivated by exp1's failure-mode check --
    the worst-2022 offenders in the score==100 pool skewed toward high
    momentum paired with weak quality (a momentum-crash signature: pure
    momentum plays overextend and reverse hardest when the regime turns,
    while quality provides no offsetting support). Rank the pool by
    (momentum_score - quality_score), ascending -- i.e. prefer names whose
    momentum isn't running far ahead of their quality -- and take the
    smallest-gap top k. Deterministic tie-break: ticker ascending."""
    pool = scores[scores == 100].index
    mom = parent_scores.get("momentum", pd.Series(dtype=float))
    qual = parent_scores.get("quality", pd.Series(dtype=float))
    gap = (mom - qual).reindex(scores.dropna().index)
    pool_gap = gap.loc[gap.index.isin(pool)]
    pool_gap = pool_gap.sort_index().sort_values(ascending=True, kind="stable")
    picks = list(pool_gap.index[:k])
    if len(picks) < k:
        remaining = gap[~gap.index.isin(picks)]
        remaining = remaining.sort_index().sort_values(ascending=True, kind="stable")
        picks += list(remaining.index[:k - len(picks)])
    return picks
