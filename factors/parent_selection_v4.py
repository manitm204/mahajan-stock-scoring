"""Parent-Selection V4 factor registry.

Wraps the research subfactor library (``research/subfactor_expansion/library.py``
+ ``library_flow.py``) so the production scoring pipeline can emit **only the
subfactors chosen by the parent-selection framework**, each carrying the
intra-parent weight the selector assigned, without duplicating the underlying
computations into ``factors/*``. Consumed via ``sub_factor_set: parent_selection_v4``
in ``config.yaml``. Parent-level composite weights (V4: IC/IR-blend, capped 25 %)
live in ``config.factors.engine_weights``.

Sub-factor selection + intra-parent weights were produced by
``python run_parent_selection.py --source expansion`` — see
``output/parent_selection_expansion/selections.csv`` and REPORT.md. Composite
weight rationale in ``output/parent_eval/REPORT.md``
(scripts/parent_eval.py, variant V4).
"""
from __future__ import annotations

import pandas as pd

from .base import Factor, SubFactor
from .utils import DataContext


def _candidate_builders():
    """Late import of the research subfactor library.

    The research package imports back into ``factors`` (research/panel.py needs
    ALL_FACTORS), so a top-level import here creates a circular chain. Deferring
    to first call keeps ``factors`` importable stand-alone.
    """
    from research.subfactor_expansion.library import CANDIDATE_BUILDERS
    return CANDIDATE_BUILDERS

# --- Selected subs + intra-parent weights (V4 IC/IR-cap-25 % construction) ---
# All 8 parents cap at 3 sub-factors; weights ∝ mean IC (3M/6M) with a 50 % cap
# and renormalisation (see research/parent_selection.py). Names are the exact
# candidate names emitted by the research library — do NOT rename without
# regenerating output/parent_selection_expansion/selections.csv.
SELECTED_SUBS: dict[str, dict[str, float]] = {
    # 2026-07-29 re-selection after the PIT-availability fixes (13F partial-
    # quarter purge + ingest gate, short-interest settlement→dissemination lag,
    # analyst-grades month-complete admission). Panel rebuilt with the gated
    # loaders; quality stops at 2 subs (next candidate's mean IC ≤ 0).
    # Derivation: output/parent_selection_expansion/ (pre-fix outputs archived
    # in output/pre_pit_fix_2026-07-29/).
    "momentum":      {"mom_vol_adjusted":                0.449,
                      "mom_6m":                          0.354,
                      "mom_52w_high_prox":               0.197},
    "value":         {"val_shareholder_yield":           0.474,
                      "val_ev_revenue_inv":              0.418,
                      "val_buyback_yield":               0.107},
    "quality":       {"qual_earnings_stability_3y":      0.500,
                      "qual_altman_z":                   0.500},
    "growth":        {"grw_earnings_surprise":           0.409,
                      "grw_earnings_yoy":                0.324,
                      "grw_operating_income_growth":     0.267},
    # 2026-08-07 user-ratified re-selection after the _pt_event_features
    # look-ahead fix (data/grades.py: trailing-window query had no upper date
    # bound, so backfilled rows counted FUTURE events). On clean data
    # rev_pt_upgrade_ratio_30d is dead (IC −0.001 vs the artifact +0.025) and
    # is dropped; the clean selector restores rev_target_revision_raw_30d in
    # the third slot (momentum-overlap caveat noted, user chose the selector
    # pick over a two-sub override). rev_rating_surprise_90d = grade vs the
    # issuing firm's own PIT baseline; only revisions sub with positive true
    # OOS evidence (2016-2022: 3M +0.009 / 6M +0.013 — expect ~+0.01, not the
    # in-window +0.034). Derivation: output/parent_selection_expansion/ +
    # output/analyst_deep_dive/.
    "revisions":     {"rev_rating_surprise_90d":         0.475,
                      "rev_raise_and_bullish_90d":       0.333,
                      "rev_target_revision_raw_30d":     0.192},
    "institutional": {"inst_investors_holding_change":   0.500,
                      "inst_new_positions":              0.495,
                      "inst_net_share_change":           0.005},
    "insider":       {"ins_no_selling_flag":             0.500,
                      "ins_cluster_buyers_180d":         0.301,
                      "ins_sell_pressure_inv":           0.199},
    "short":         {"si_short_pct_float":              0.421,
                      "si_days_to_cover":                0.408,
                      "si_short_pct_float_pctile_252d":  0.171},
}


class _ResearchLibraryFactor(Factor):
    """Adapter: one production ``Factor`` per parent that sources its subs from
    the research library builder and stamps each with its selected weight.

    The Layer 2 scoring engine (:func:`factors.base.score_factor`) treats these
    exactly like a hand-coded factor: it sector-percentile-ranks each sub and
    weighted-averages them into a parent score. Config-side allowlists (e.g. the
    ``factor_sets.parent_selection_v4`` documentation entry) still apply.
    """

    def __init__(self, key: str) -> None:
        self.name = key
        self.weight_key = key
        self._weights = SELECTED_SUBS[key]

    def compute(self, ctx: DataContext) -> list[SubFactor]:
        keep = self._weights
        builder = _candidate_builders()[self.name]
        subs: list[SubFactor] = []
        for cand in builder(ctx):
            if cand.name not in keep:
                continue
            raw = cand.raw if isinstance(cand.raw, pd.Series) else pd.Series(cand.raw)
            subs.append(SubFactor(
                name=cand.name, raw=raw, higher_is_better=cand.higher_is_better,
                weight=keep[cand.name],
            ))
        if not subs:
            raise RuntimeError(
                f"parent-selection-v4: no candidate subs available for parent "
                f"{self.name!r}. Selected={list(keep)}; check that the research "
                f"library builder still emits them.")
        return subs


# Ordered registry (same parent order as ALL_FACTORS so weight vectors align).
ALL_FACTORS_V4: list[Factor] = [
    _ResearchLibraryFactor("momentum"),
    _ResearchLibraryFactor("value"),
    _ResearchLibraryFactor("quality"),
    _ResearchLibraryFactor("growth"),
    _ResearchLibraryFactor("revisions"),
    _ResearchLibraryFactor("short"),
    _ResearchLibraryFactor("insider"),
    _ResearchLibraryFactor("institutional"),
]

# V4 parent-level composite weights — mirrored into
# ``config.factors.engine_weights`` so run_scoring picks them up. Kept here as
# the source-of-truth reference so a mismatch is easy to catch.
# 2026-08-07 user-ratified METHOD B ("effective-exposure cap"): the incumbent
# IC/IR-blend vector re-trimmed so no parent's EFFECTIVE exposure (C·w — own
# weight plus what correlated parents import) exceeds the 25 % cap the nominal
# water-fill was meant to enforce. Momentum/institutional were at 0.31
# effective under the old vector. Derivation: scripts/derive_engine_weights_b.py
# (clean 42-month panel C); evidence: output/crowding/incremental_weights/
# REPORT.md (fully-PIT 2020/2021-start backtests + cap/λ sweeps). Rerun the
# derivation script after any SELECTED_SUBS change. Pre-B incumbent vector
# (2026-07-05 parent-eval): mom .170 / val .106 / qual .044 / grw .171 /
# rev .141 / inst .152 / ins .034 / short .182.
# 2026-08-07b re-trim after the revisions formula re-ratification (dead
# rev_pt_upgrade_ratio_30d out, rev_target_revision_raw_30d back in): the new
# formula raises revisions' block correlation (eff 0.286), so revisions is now
# trimmed too (.1515→.1389) and momentum slightly more (.1299→.1249).
# 2026-08-31 METHOD E ("cap + diversifier floor") — SUPERSEDED 2026-09-01,
# kept for the record: B's cap-trim run to convergence as before, then any
# parent whose effective exposure sits below 0.05 (value, insider —
# anti-correlated with the momentum/growth/revisions/institutional/short
# cluster, but B alone never rewards that) is floored at 8 % nominal weight,
# funded proportionally from parents above the floor; quality also gets
# floored once the loop redistributes far enough. Derivation:
# scripts/derive_engine_weights_e.py. Evidence: pre-registered walk-forward IC
# study (scripts/incremental_weight_study.py, output/crowding/
# incremental_weights_e_floor/results.json) — decisive pass, best blend IC of
# A/B/C/D/E (0.0663 vs incumbent A 0.0554, CI90 [+0.0070,+0.0149]) — AND the
# fully-PIT costed portfolio backtest (scripts/full_pit_backtest_e.py,
# output/crowding/incremental_weights_e_floor/pit_backtest_e.csv, 2021-2026
# OOS) — CAGR 18.6 %/Sharpe 1.078/Calmar 0.587/max_dd -31.6 %/alpha_t 1.84,
# best alpha-t of A/B/D/E, ties-or-beats B on every metric but Sortino (0.945
# vs 0.988). Unlike method D (won IC, lost the portfolio test), E won both.
#
# 2026-09-01 METHOD EQEFF ("equal effective exposure") — CURRENT. Solves
# w \propto C^{-1}\cdot 1 (correlation-matrix inverse applied to a vector of
# ones), clipped >=0, renormalized to sum 1 — the nominal weight vector that
# makes every parent's EFFECTIVE exposure (C\cdot w) come out EQUAL (all 8
# land at ~0.111 on the current panel, not exactly 1/8=0.125 — the common
# value is whatever the correlation structure implies, not a fixed target).
# No IC/IR information is used at all: pure correlation-structure risk
# parity. Derivation: scripts/derive_engine_weights_eqeff.py.
#
# Evidence (2026-09-01, extensive same-day study): tested against incumbent A,
# method E, plain equal-nominal-weight (EQ), and 10 further IC-informed
# hybrids (structural C^{-1}\cdotIC solves at multiple shrinkage levels, an
# IR-target variant, three EQEFF/A linear blends at 25/50/75 %, and two
# sequential equal-or-EQEFF + IC-tilt constructions) — scripts/
# weight_config_study.py, scripts/full_pit_backtest_eqeff.py + _2020.py,
# scripts/weight_config_analysis.py, scripts/factor_scorecard.py,
# scripts/full_pit_backtest_10configs.py. On a 10-metric PORTFOLIO-AGNOSTIC
# scorecard (IC, IR, hit rate, Q5-Q1, monotonicity, tent asymmetry,
# within-sector IC, Fama-MacBeth slope t-stat, score stability, bottom-decile
# drag — output/crowding/weight_config_study/factor_scorecard*.csv), EQEFF
# wins EVERY metric outright and none of the 10 hybrids beat it — ranking
# quality degrades MONOTONICALLY as IC information is blended back in
# (BLEND25 -> BLEND50 -> BLEND75 -> EQEFF is a clean one-directional
# dose-response curve). EQEFF's edge survives a within-sector control (not a
# sector-rotation artifact) and clears Fama-MacBeth significance (t=2.45)
# that the incumbent A does NOT clear (t=0.45).
#
# CAVEAT CARRIED FORWARD DELIBERATELY: a COSTED, CONCENTRATED (top 10-25 %)
# portfolio backtest does NOT show EQEFF beating method E — E remains better
# at that specific book concentration (scripts/weight_config_analysis.py's
# 12-construction sensitivity grid: E wins 5/12, EQEFF only 3/12, worst-of-
# four at top_pct 0.10/0.25; EQEFF only overtakes at top_pct >=0.40). Shipped
# anyway as a deliberate choice to optimize the composite SCORE's standalone
# ranking quality (what run_scoring.py / the Scoring dashboard describe)
# rather than the one specific top-25%/cap5/cap_match live portfolio
# construction E happened to be tuned against — user judged that construction
# too easy to find a false edge in relative to a portfolio-agnostic ranking
# metric. If the live portfolio construction is ever revisited, re-check E vs
# EQEFF at that construction specifically; don't assume this verdict
# transfers automatically.
V4_PARENT_WEIGHTS: dict[str, float] = {
    "momentum":      0.0829,
    "value":         0.2302,
    "quality":       0.1717,
    "growth":        0.1216,
    "revisions":     0.0978,
    "institutional": 0.0648,
    "insider":       0.1562,
    "short":         0.0748,
}


__all__ = ["ALL_FACTORS_V4", "SELECTED_SUBS", "V4_PARENT_WEIGHTS"]
