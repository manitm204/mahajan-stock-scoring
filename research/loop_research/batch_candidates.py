"""Batch of ~100 selector hypotheses (user request 2026-09-09: "try 100 more
strategies", generated in named waves rather than truly one-at-a-time so the
loop can actually finish in reasonable wall-clock time -- each wave is still
informed by the prior wave's results, and every single one, win or lose, is
logged individually in session_log.md / results.json, never silently
discarded). See PARENTS for the 8 V4 parent families this all draws on.

Each entry in ALL_CANDIDATES() is (name, hypothesis, selector_fn,
deterministic: bool). Selector signature matches harness.py's contract:
    selector(scores, parent_scores, held, k, refresh_n, rng, ctx=None)
`ctx` carries {'date','dates','i','bundle'} for selectors that need trailing
history (score trajectory) or price-derived risk metrics (vol/beta/dd).
"""
from __future__ import annotations

import pandas as pd

PARENTS = ["momentum", "value", "quality", "growth", "revisions",
          "institutional", "insider", "short"]


def _rank_pick(series: pd.Series, pool, k, ascending=False):
    s = series.loc[series.index.isin(pool)]
    s = s.sort_index().sort_values(ascending=ascending, kind="stable")
    picks = list(s.index[:k])
    if len(picks) < k:
        remaining = series[~series.index.isin(picks)]
        remaining = remaining.sort_index().sort_values(ascending=ascending, kind="stable")
        picks += list(remaining.index[:k - len(picks)])
    return picks


def _parent_frame(parent_scores):
    return pd.DataFrame({p: s for p, s in parent_scores.items()})


# --------------------------------------------------------------------------- #
# Wave A: parent-score aggregation variants (single scoring rule over the
# 8 parents, deterministic top-k) -- broader/narrower ways to combine
# information already screened by the composite.
# --------------------------------------------------------------------------- #
def make_aggregate_selector(name, agg_fn):
    def selector(scores, parent_scores, held, k, refresh_n, rng, ctx=None):
        df = _parent_frame(parent_scores)
        agg = agg_fn(df).reindex(scores.dropna().index)
        pool = scores[scores == 100].index
        return _rank_pick(agg, pool, k, ascending=False)
    selector.__name__ = name
    return selector


import numpy as np

_V4_WEIGHTS = pd.Series({"momentum": 0.0829, "value": 0.2302, "quality": 0.1717,
                         "growth": 0.1216, "revisions": 0.0978, "institutional": 0.0648,
                         "insider": 0.1562, "short": 0.0748})


def _weighted_avg(df):
    w = _V4_WEIGHTS.reindex(df.columns).fillna(0)
    return df.mul(w, axis=1).sum(axis=1)


WAVE_A = [
    ("waveA_max_parent_score", "Rank by MAX of the 8 parent scores (does having one standout strength beat needing broad support?)",
    make_aggregate_selector("max_agg", lambda df: df.max(axis=1))),
    ("waveA_weighted_avg_v4weights", "Rank by the production V4 parent-weight-weighted average (same weights the composite itself uses, just recomputed as a raw score instead of a sector percentile)",
    make_aggregate_selector("wavg_agg", _weighted_avg)),
    ("waveA_median_parent_score", "Rank by MEDIAN of the 8 parent scores (robust central tendency, less sensitive to one outlier parent than mean)",
    make_aggregate_selector("median_agg", lambda df: df.median(axis=1))),
    ("waveA_trimmed_mean_excl_worst2", "Rank by mean of the 8 parents EXCLUDING the worst 2 (drop the two weakest families before averaging, a softer version of exp1's min-rule)",
    make_aggregate_selector("trimmed_agg", lambda df: df.apply(lambda row: row.sort_values().iloc[2:].mean(), axis=1))),
    ("waveA_geometric_mean", "Rank by geometric mean of the 8 parent scores (penalizes any single very-low score more than arithmetic mean, without being as harsh as a hard min)",
    make_aggregate_selector("geo_agg", lambda df: np.exp(np.log(df.clip(lower=1)).mean(axis=1)))),
    ("waveA_rank_average", "Rank by the AVERAGE CROSS-SECTIONAL RANK across the 8 parents (rank-based instead of raw-score-based, so no single parent's scale/skew dominates)",
    make_aggregate_selector("rankavg_agg", lambda df: df.rank(pct=True).mean(axis=1))),
]


# --------------------------------------------------------------------------- #
# Wave B: single-parent floor filters -- exclude pool names below a floor on
# ONE parent, then draw randomly among survivors. Isolates whether any single
# parent family, used as a pure screen (not a ranking tilt), helps.
# --------------------------------------------------------------------------- #
def make_floor_filter_selector(parent, floor):
    def selector(scores, parent_scores, held, k, refresh_n, rng, ctx=None):
        pool = list(scores[scores == 100].index)
        pscore = parent_scores.get(parent, pd.Series(dtype=float))
        survivors = [t for t in pool if pscore.get(t, 0) >= floor]
        use_pool = survivors if len(survivors) >= k else pool
        if not use_pool:
            return held
        if not held:
            rng.shuffle(use_pool)
            return use_pool[:k]
        keep = list(held)
        n_evict = min(refresh_n, len(keep))
        to_evict = set(rng.sample(keep, n_evict))
        keep = [t for t in keep if t not in to_evict]
        need = k - len(keep)
        cands = [t for t in use_pool if t not in keep]
        rng.shuffle(cands)
        fill = cands[:need]
        if len(fill) < need:
            universe = [t for t in scores.dropna().index if t not in keep and t not in fill]
            rng.shuffle(universe)
            fill += universe[:need - len(fill)]
        return keep + fill
    selector.__name__ = f"floor_{parent}_{floor}"
    return selector


WAVE_B = [(f"waveB_floor_{p}_ge40", f"Exclude score==100 pool names with {p} parent score < 40, draw randomly among survivors (keeps exp1-style diversification, isolates whether {p} alone as a pure screen helps)",
          make_floor_filter_selector(p, 40)) for p in PARENTS]


# --------------------------------------------------------------------------- #
# Wave C: single-parent deterministic tie-break -- among the ==100 ties,
# prefer the highest score on ONE named parent (not an average/floor).
# --------------------------------------------------------------------------- #
def make_single_parent_top5_selector(parent):
    def selector(scores, parent_scores, held, k, refresh_n, rng, ctx=None):
        pscore = parent_scores.get(parent, pd.Series(dtype=float)).reindex(scores.dropna().index)
        pool = scores[scores == 100].index
        return _rank_pick(pscore, pool, k, ascending=False)
    selector.__name__ = f"top5_{parent}"
    return selector


WAVE_C = [(f"waveC_top5_by_{p}", f"Among the ==100 tied pool, break ties by preferring the single highest {p} score (not an average across parents -- isolates {p} alone as a tie-break rule)",
          make_single_parent_top5_selector(p)) for p in PARENTS]


# --------------------------------------------------------------------------- #
# Wave D: defensive / thematic parent-pair and triple combos, chosen for
# economic rationale (not an exhaustive C(8,2) sweep).
# --------------------------------------------------------------------------- #
def make_combo_selector(name, parents):
    def selector(scores, parent_scores, held, k, refresh_n, rng, ctx=None):
        df = _parent_frame({p: parent_scores[p] for p in parents if p in parent_scores})
        if df.empty:
            return held
        agg = df.mean(axis=1).reindex(scores.dropna().index)
        pool = scores[scores == 100].index
        return _rank_pick(agg, pool, k, ascending=False)
    selector.__name__ = name
    return selector


WAVE_D = [
    ("waveD_quality_institutional_insider", "Defensive combo: average of quality+institutional+insider (fundamentally-supported, smart-money-confirmed names, deliberately excluding momentum entirely)",
    make_combo_selector("qii", ["quality", "institutional", "insider"])),
    ("waveD_value_quality", "Classic value+quality tilt (cheap AND fundamentally sound, the textbook 'quality value' combo)",
    make_combo_selector("vq", ["value", "quality"])),
    ("waveD_growth_revisions", "Growth+revisions combo: forward-looking growth confirmed by analyst revisions, without price momentum's crash risk",
    make_combo_selector("gr", ["growth", "revisions"])),
    ("waveD_insider_institutional", "Smart-money combo: insider buying + institutional accumulation, no price-based or valuation signal at all",
    make_combo_selector("ii", ["insider", "institutional"])),
    ("waveD_value_short_contrarian", "Contrarian combo: value+short (cheap AND not heavily shorted -- avoid names the market's betting against even if they're statistically cheap)",
    make_combo_selector("vs", ["value", "short"])),
]


# --------------------------------------------------------------------------- #
# Wave E: momentum-overextension variants beyond exp2's momentum-quality gap
# (exp2 was rejected -- test whether a DIFFERENT anchor parent, or a hard
# momentum cap instead of a gap penalty, behaves differently).
# --------------------------------------------------------------------------- #
def make_gap_selector(name, anchor, other):
    def selector(scores, parent_scores, held, k, refresh_n, rng, ctx=None):
        a = parent_scores.get(anchor, pd.Series(dtype=float))
        b = parent_scores.get(other, pd.Series(dtype=float))
        gap = (a - b).reindex(scores.dropna().index)
        pool = scores[scores == 100].index
        return _rank_pick(gap, pool, k, ascending=True)
    selector.__name__ = name
    return selector


def make_momentum_cap_selector(cap):
    def selector(scores, parent_scores, held, k, refresh_n, rng, ctx=None):
        pool = list(scores[scores == 100].index)
        mom = parent_scores.get("momentum", pd.Series(dtype=float))
        survivors = [t for t in pool if mom.get(t, 0) <= cap]
        use_pool = survivors if len(survivors) >= k else pool
        if not use_pool:
            return held
        if not held:
            rng.shuffle(use_pool)
            return use_pool[:k]
        keep = list(held)
        n_evict = min(refresh_n, len(keep))
        to_evict = set(rng.sample(keep, n_evict))
        keep = [t for t in keep if t not in to_evict]
        need = k - len(keep)
        cands = [t for t in use_pool if t not in keep]
        rng.shuffle(cands)
        fill = cands[:need]
        if len(fill) < need:
            universe = [t for t in scores.dropna().index if t not in keep and t not in fill]
            rng.shuffle(universe)
            fill += universe[:need - len(fill)]
        return keep + fill
    selector.__name__ = f"mom_cap_{cap}"
    return selector


WAVE_E = [
    ("waveE_momentum_institutional_gap", "Momentum-overextension via institutional anchor: penalize momentum running far ahead of institutional accumulation instead of quality (institutional flow, not fundamentals, as the 'is this move supported' check)",
    make_gap_selector("mom_inst_gap", "momentum", "institutional")),
    ("waveE_momentum_insider_gap", "Momentum-overextension via insider anchor: penalize momentum without insider buying support",
    make_gap_selector("mom_ins_gap", "momentum", "insider")),
    ("waveE_momentum_short_gap", "Crowded-momentum check: penalize high momentum paired with high short-interest (short parent already scores against being heavily shorted, so a big gap = momentum + already-crowded)",
    make_gap_selector("mom_short_gap", "momentum", "short")),
    ("waveE_momentum_hardcap_80_random", "Hard-exclude momentum>80 from the pool entirely, draw randomly among survivors (avoids exp2's failure mode of over-narrowing by using a screen, not a full ranking)",
    make_momentum_cap_selector(80)),
    ("waveE_momentum_hardcap_90_random", "Same as above but a looser momentum<=90 cap (tests whether the cap threshold itself matters)",
    make_momentum_cap_selector(90)),
]


# --------------------------------------------------------------------------- #
# Wave F: score TRAJECTORY -- prefer names whose parent-score average is
# rising vs falling over the trailing hold period, distinct from every prior
# wave's use of the LEVEL of the score.
# --------------------------------------------------------------------------- #
def make_trajectory_selector(name, months_back, ascending_bad=False):
    def selector(scores, parent_scores, held, k, refresh_n, rng, ctx=None):
        pool = scores[scores == 100].index
        if ctx is None:
            return _rank_pick(pd.Series(0.0, index=scores.dropna().index), pool, k)
        dates, i, bundle = ctx["dates"], ctx["i"], ctx["bundle"]
        cur = _parent_frame(parent_scores).mean(axis=1)
        if i < months_back:
            return _rank_pick(cur.reindex(scores.dropna().index), pool, k, ascending=False)
        prev_date = dates[i - months_back]
        prev_pscores = bundle["parent_scores"].get(prev_date, {})
        prev = _parent_frame(prev_pscores).mean(axis=1) if prev_pscores else pd.Series(dtype=float)
        change = (cur - prev.reindex(cur.index)).reindex(scores.dropna().index)
        return _rank_pick(change, pool, k, ascending=False)
    selector.__name__ = name
    return selector


WAVE_F = [
    ("waveF_rising_score_3mo", "Prefer names whose average parent score IMPROVED the most over the trailing 3 months (a score-momentum signal, distinct from price momentum -- 'is the fundamental picture getting better, not just was it already good')",
    make_trajectory_selector("rising3", 3)),
    ("waveF_rising_score_1mo", "Same idea, 1-month trajectory (faster/noisier version of waveF_rising_score_3mo)",
    make_trajectory_selector("rising1", 1)),
]


# --------------------------------------------------------------------------- #
# Wave G: sector diversification cap, layered on top of the two leading
# rules found so far (random baseline, and exp3's avg-parent-score tilt).
# --------------------------------------------------------------------------- #
def make_sector_capped_selector(name, base_rank_fn, max_per_sector):
    def selector(scores, parent_scores, held, k, refresh_n, rng, ctx=None):
        pool = list(scores[scores == 100].index)
        if not pool or ctx is None:
            return held
        sector = ctx["bundle"]["data"].sector
        ranked = base_rank_fn(scores, parent_scores, pool, rng)
        picks, sector_count = [], {}
        for t in ranked:
            sec = sector.get(t, "UNKNOWN") if sector is not None else "UNKNOWN"
            if sector_count.get(sec, 0) >= max_per_sector:
                continue
            picks.append(t)
            sector_count[sec] = sector_count.get(sec, 0) + 1
            if len(picks) == k:
                break
        if len(picks) < k:
            for t in ranked:
                if t not in picks:
                    picks.append(t)
                if len(picks) == k:
                    break
        return picks
    selector.__name__ = name
    return selector


def _avg_rank_order(scores, parent_scores, pool, rng):
    df = _parent_frame(parent_scores)
    avg = df.mean(axis=1).reindex(scores.dropna().index)
    s = avg.loc[avg.index.isin(pool)].sort_index().sort_values(ascending=False, kind="stable")
    return list(s.index)


def _random_rank_order(scores, parent_scores, pool, rng):
    p = list(pool)
    rng.shuffle(p)
    return p


WAVE_G = [
    ("waveG_avgparent_sector_cap1", "exp3's avg-parent-score ranking, but capped at 1 name per GICS sector (forces cross-sector diversification instead of letting the ranking concentrate in whichever sector currently scores best)",
    make_sector_capped_selector("avgparent_cap1", _avg_rank_order, 1)),
    ("waveG_random_sector_cap2", "Random baseline, but capped at 2 names per sector (tests whether the ~74%/69% win rates from parent-score tilts are partly just an accidental diversification effect that a sector cap on the RANDOM baseline itself would already capture)",
    make_sector_capped_selector("random_cap2", _random_rank_order, 2)),
]


# --------------------------------------------------------------------------- #
# Wave H: risk-based filters using trailing 63-day realized vol / beta /
# drawdown-from-high (research.loop_research.risk_metrics).
# --------------------------------------------------------------------------- #
def make_risk_filter_selector(name, metric, keep_fn):
    def selector(scores, parent_scores, held, k, refresh_n, rng, ctx=None):
        pool = list(scores[scores == 100].index)
        if not pool or ctx is None:
            return held
        from research.loop_research.risk_metrics import build_risk_metrics
        rm = build_risk_metrics(ctx["bundle"])
        day_metrics = rm.get(ctx["date"], {}).get(metric)
        if day_metrics is None:
            survivors = pool
        else:
            survivors = [t for t in pool if keep_fn(day_metrics.get(t, float("nan")))]
        use_pool = survivors if len(survivors) >= k else pool
        if not held:
            rng.shuffle(use_pool)
            return use_pool[:k]
        keep = list(held)
        n_evict = min(refresh_n, len(keep))
        to_evict = set(rng.sample(keep, n_evict))
        keep = [t for t in keep if t not in to_evict]
        need = k - len(keep)
        cands = [t for t in use_pool if t not in keep]
        rng.shuffle(cands)
        fill = cands[:need]
        if len(fill) < need:
            universe = [t for t in scores.dropna().index if t not in keep and t not in fill]
            rng.shuffle(universe)
            fill += universe[:need - len(fill)]
        return keep + fill
    selector.__name__ = name
    return selector


WAVE_H = [
    ("waveH_exclude_high_vol", "Exclude the pool's top-tercile trailing-63d realized vol names, draw randomly among survivors (avoid the most volatile tied names -- direct test of whether vol itself, not any fundamental parent, explains 2022's damage)",
    make_risk_filter_selector("exvol", "vol", lambda v: v == v and v <= float("inf"))),  # threshold set at call site via closure below
    ("waveH_exclude_high_beta", "Exclude pool names with trailing beta-to-SPY > 1.3, draw randomly among survivors (lower-beta tilt within the tied pool, a direct defensive lever distinct from any parent score)",
    make_risk_filter_selector("exbeta", "beta", lambda b: b == b and b <= 1.3)),
    ("waveH_exclude_falling_knives", "Exclude pool names already down >15% from their trailing 3-month high, draw randomly among survivors (avoid 'catching a falling knife' -- distinct from parent scores, which don't see recent price action at all)",
    make_risk_filter_selector("exdd", "dd", lambda d: d == d and d >= -0.15)),
]
# fix the vol threshold selector to use a real per-date tercile cutoff instead of a no-op
def _make_vol_tercile_selector():
    def selector(scores, parent_scores, held, k, refresh_n, rng, ctx=None):
        pool = list(scores[scores == 100].index)
        if not pool or ctx is None:
            return held
        from research.loop_research.risk_metrics import build_risk_metrics
        rm = build_risk_metrics(ctx["bundle"])
        vol = rm.get(ctx["date"], {}).get("vol")
        if vol is None:
            survivors = pool
        else:
            pool_vol = vol.reindex(pool).dropna()
            if len(pool_vol) >= 3:
                cutoff = pool_vol.quantile(2 / 3)
                survivors = [t for t in pool if vol.get(t, 0) <= cutoff]
            else:
                survivors = pool
        use_pool = survivors if len(survivors) >= k else pool
        if not held:
            rng.shuffle(use_pool)
            return use_pool[:k]
        keep = list(held)
        n_evict = min(refresh_n, len(keep))
        to_evict = set(rng.sample(keep, n_evict))
        keep = [t for t in keep if t not in to_evict]
        need = k - len(keep)
        cands = [t for t in use_pool if t not in keep]
        rng.shuffle(cands)
        fill = cands[:need]
        if len(fill) < need:
            universe = [t for t in scores.dropna().index if t not in keep and t not in fill]
            rng.shuffle(universe)
            fill += universe[:need - len(fill)]
        return keep + fill
    return selector


WAVE_H[0] = ("waveH_exclude_high_vol", WAVE_H[0][1], _make_vol_tercile_selector())


# --------------------------------------------------------------------------- #
# Wave I: leave-one-out averages -- average of 7 of the 8 parents, excluding
# one at a time. Isolates which single parent's ABSENCE from the tilt matters.
# --------------------------------------------------------------------------- #
def make_leave_one_out_selector(excluded):
    def selector(scores, parent_scores, held, k, refresh_n, rng, ctx=None):
        remaining = {p: s for p, s in parent_scores.items() if p != excluded}
        df = _parent_frame(remaining)
        if df.empty:
            return held
        agg = df.mean(axis=1).reindex(scores.dropna().index)
        pool = scores[scores == 100].index
        return _rank_pick(agg, pool, k, ascending=False)
    selector.__name__ = f"loo_{excluded}"
    return selector


WAVE_I = [(f"waveI_avg_excl_{p}", f"Average of the OTHER 7 parent scores, excluding {p} entirely, top 5 (isolates whether {p}'s presence in exp3's average is helping or hurting)",
          make_leave_one_out_selector(p)) for p in PARENTS]


# --------------------------------------------------------------------------- #
# Wave J: additional thematic pair combos (academic-literature-motivated,
# distinct from Wave D's picks).
# --------------------------------------------------------------------------- #
WAVE_J = [
    ("waveJ_quality_momentum", "'Quality momentum' (Asness et al.): momentum confirmed by quality, the classic academic combo distinct from exp2's rejected momentum-quality GAP penalty",
    make_combo_selector("qm", ["quality", "momentum"])),
    ("waveJ_value_growth_garp", "GARP (growth at a reasonable price): value+growth combo",
    make_combo_selector("vg", ["value", "growth"])),
    ("waveJ_revisions_institutional", "Analyst-revisions confirmed by institutional flow (two independent 'is the market catching on' signals)",
    make_combo_selector("ri", ["revisions", "institutional"])),
    ("waveJ_short_quality", "Avoid crowded shorts + fundamentally sound (short+quality combo -- the inverse framing of Wave D's value+short)",
    make_combo_selector("sq", ["short", "quality"])),
    ("waveJ_all_except_momentum", "Average of all 7 non-momentum parents (the most defensive possible tilt -- fully excludes price momentum, the one parent implicated in the 2022 failure mode)",
    make_combo_selector("nomom", [p for p in PARENTS if p != "momentum"])),
]


# --------------------------------------------------------------------------- #
# Wave L: the remaining pair combos from the full C(8,2)=28 grid not already
# covered by Wave D/J's hand-picked thematic pairs -- a systematic sweep
# (deterministic, ~free to run) rather than a hand-picked story for each,
# reported with an eta-squared-style "does the specific pair matter at all"
# summary in the batch runner rather than one paragraph per pair.
# --------------------------------------------------------------------------- #
import itertools

_ALREADY_COVERED_PAIRS = {
    frozenset(["value", "quality"]), frozenset(["growth", "revisions"]),
    frozenset(["insider", "institutional"]), frozenset(["value", "short"]),
    frozenset(["quality", "momentum"]), frozenset(["value", "growth"]),
    frozenset(["revisions", "institutional"]), frozenset(["short", "quality"]),
}
WAVE_L = []
for p1, p2 in itertools.combinations(PARENTS, 2):
    if frozenset([p1, p2]) in _ALREADY_COVERED_PAIRS:
        continue
    WAVE_L.append((
        f"waveL_pair_{p1}_{p2}",
        f"Systematic pair-grid sweep (not hand-picked): average of {p1}+{p2}, the remaining untested cell in the full C(8,2) parent-pair grid.",
        make_combo_selector(f"{p1}_{p2}", [p1, p2]),
    ))


# --------------------------------------------------------------------------- #
# Wave K: negative controls, triples, momentum-focused combined refinements,
# and a second floor threshold -- rounds the batch out past 100 total
# candidates tested (incl. the original 4).
# --------------------------------------------------------------------------- #
WAVE_K = []

# Negative controls: deliberately pick the WORST-ranked names. If these don't
# clearly underperform, something is wrong with the harness/metric, not the
# hypothesis -- a sanity check on the whole batch, not a real strategy idea.
WAVE_K.append(("waveK_negctrl_worst_avg_parent", "NEGATIVE CONTROL: rank by avg parent score ASCENDING (worst first), top 5 -- should clearly underperform; if it doesn't, something is wrong with the metric/harness, not the hypothesis.",
              make_aggregate_selector("worst_avg", lambda df: -df.mean(axis=1))))
WAVE_K.append(("waveK_negctrl_worst_min_parent", "NEGATIVE CONTROL: rank by min parent score ASCENDING (worst weak-spot first), top 5.",
              make_aggregate_selector("worst_min", lambda df: -df.min(axis=1))))

# Triples beyond Wave D's quality+institutional+insider.
WAVE_K.append(("waveK_triple_value_quality_institutional", "'Safe value' triple: value+quality+institutional (cheap, sound, and smart-money-held)",
              make_combo_selector("vqi", ["value", "quality", "institutional"])))
WAVE_K.append(("waveK_triple_momentum_growth_revisions", "'Improving business' triple: momentum+growth+revisions (price, fundamentals, and analyst view all pointing the same way)",
              make_combo_selector("mgr", ["momentum", "growth", "revisions"])))
WAVE_K.append(("waveK_triple_quality_growth_revisions", "Quality-confirmed growth triple: quality+growth+revisions, no momentum or valuation at all",
              make_combo_selector("qgr", ["quality", "growth", "revisions"])))
WAVE_K.append(("waveK_triple_insider_short_institutional", "'Who's buying/selling' triple: insider+short+institutional -- pure ownership-flow signal, zero price or fundamental information",
              make_combo_selector("isi", ["insider", "short", "institutional"])))

# Combined refinements targeting the recurring 2022 weakness directly: layer
# exp3's avg-parent-score tilt with a momentum ceiling, instead of Wave E's
# momentum-cap-with-random-fill.
def make_capped_then_avg_selector(name, momentum_cap):
    def selector(scores, parent_scores, held, k, refresh_n, rng, ctx=None):
        pool = list(scores[scores == 100].index)
        mom = parent_scores.get("momentum", pd.Series(dtype=float))
        survivors = [t for t in pool if mom.get(t, 0) <= momentum_cap]
        use_pool = survivors if len(survivors) >= k else pool
        df = _parent_frame(parent_scores)
        avg = df.mean(axis=1).reindex(scores.dropna().index)
        return _rank_pick(avg, use_pool, k, ascending=False)
    selector.__name__ = name
    return selector


WAVE_K.append(("waveK_avgparent_momcap80", "exp3's avg-parent-score ranking, but pre-filtered to exclude momentum>80 names first (directly targets the 2022 momentum-crash weak spot shared by exp1/exp3, instead of Wave E's random-fill version)",
              make_capped_then_avg_selector("avgparent_momcap80", 80)))
WAVE_K.append(("waveK_avgparent_momcap70", "Same idea, tighter momentum<=70 cap",
              make_capped_then_avg_selector("avgparent_momcap70", 70)))
WAVE_K.append(("waveK_trimmed_momcap80", "Wave A's trimmed-mean (excl. worst 2 parents) combined with the momentum<=80 pre-filter",
              None))  # filled below, needs trimmed agg + filter

def _trimmed_momcap_selector(scores, parent_scores, held, k, refresh_n, rng, ctx=None):
    pool = list(scores[scores == 100].index)
    mom = parent_scores.get("momentum", pd.Series(dtype=float))
    survivors = [t for t in pool if mom.get(t, 0) <= 80]
    use_pool = survivors if len(survivors) >= k else pool
    df = _parent_frame(parent_scores)
    agg = df.apply(lambda row: row.sort_values().iloc[2:].mean(), axis=1).reindex(scores.dropna().index)
    return _rank_pick(agg, use_pool, k, ascending=False)


WAVE_K[-1] = ("waveK_trimmed_momcap80", "Wave A's trimmed-mean (excl. worst 2 parents) combined with the momentum<=80 pre-filter",
             _trimmed_momcap_selector)

# A second floor threshold (60 instead of Wave B's 40) for every parent --
# tests whether Wave B's negative/neutral results were just too loose a bar.
WAVE_K += [(f"waveK_floor_{p}_ge60", f"Same as Wave B's {p} floor filter but a stricter >=60 threshold (tests whether 40 was too loose to see an effect)",
           make_floor_filter_selector(p, 60)) for p in PARENTS]

# Additional risk-filter variants: median vol split (looser than Wave H's
# tercile) and a stricter beta cap.
WAVE_K.append(("waveK_exclude_high_vol_median", "Exclude pool names above the MEDIAN trailing vol (looser cut than Wave H's top-tercile exclusion)",
              None))

def _vol_median_selector(scores, parent_scores, held, k, refresh_n, rng, ctx=None):
    pool = list(scores[scores == 100].index)
    if not pool or ctx is None:
        return held
    from research.loop_research.risk_metrics import build_risk_metrics
    rm = build_risk_metrics(ctx["bundle"])
    vol = rm.get(ctx["date"], {}).get("vol")
    if vol is None:
        survivors = pool
    else:
        pool_vol = vol.reindex(pool).dropna()
        if len(pool_vol) >= 3:
            cutoff = pool_vol.median()
            survivors = [t for t in pool if vol.get(t, 0) <= cutoff]
        else:
            survivors = pool
    use_pool = survivors if len(survivors) >= k else pool
    if not held:
        rng.shuffle(use_pool)
        return use_pool[:k]
    keep = list(held)
    n_evict = min(refresh_n, len(keep))
    to_evict = set(rng.sample(keep, n_evict))
    keep = [t for t in keep if t not in to_evict]
    need = k - len(keep)
    cands = [t for t in use_pool if t not in keep]
    rng.shuffle(cands)
    fill = cands[:need]
    if len(fill) < need:
        universe = [t for t in scores.dropna().index if t not in keep and t not in fill]
        rng.shuffle(universe)
        fill += universe[:need - len(fill)]
    return keep + fill


WAVE_K[-1] = ("waveK_exclude_high_vol_median", "Exclude pool names above the MEDIAN trailing vol (looser cut than Wave H's top-tercile exclusion)",
             _vol_median_selector)
WAVE_K.append(("waveK_exclude_high_beta_1p0", "Stricter beta cap: exclude pool names with trailing beta>1.0 (vs Wave H's 1.3)",
              make_risk_filter_selector("exbeta10", "beta", lambda b: b == b and b <= 1.0)))

WAVE_K.append(("waveK_rising_score_6mo", "Score-trajectory (Wave F) at a slower 6-month lookback",
              make_trajectory_selector("rising6", 6)))
WAVE_K.append(("waveK_sector_cap2_avgparent", "exp3's avg-parent tilt with a looser 2-per-sector cap (vs Wave G's 1-per-sector)",
              make_sector_capped_selector("avgparent_cap2", _avg_rank_order, 2)))

# Mark every WAVE_K entry deterministic except the two random-fill floor/vol
# variants that use rng for tie-breaking among survivors -- flagged in the
# runner by name pattern instead of a parallel list, since all of these are
# still primarily deterministic screens with a thin random fill.
ALL_WAVES = {
    "A_aggregation": WAVE_A,
    "B_floor_filters": WAVE_B,
    "C_tie_break": WAVE_C,
    "D_thematic_combos": WAVE_D,
    "E_momentum_overextension": WAVE_E,
    "F_score_trajectory": WAVE_F,
    "G_sector_diversification": WAVE_G,
    "H_risk_filters": WAVE_H,
    "I_leave_one_out": WAVE_I,
    "J_more_thematic_combos": WAVE_J,
    "L_systematic_pair_grid": WAVE_L,
    "K_negctrl_triples_refinements": WAVE_K,
}
