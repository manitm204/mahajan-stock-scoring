"""Predictive/Reliability/Production scores + CORE/REGIME_DEPENDENT/WATCHLIST/
EXCLUDED classification, consuming one regime_aware_evidence.as_of() table.
See docs/superpowers/specs/2026-07-11-hierarchical-regime-aware-factor-model-design.md
Section 2.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from research.parent_selection import _pct_rank

CORE_MIN_IC = 0.01
CORE_MIN_POS_YEARS = 0.60
CORE_MAX_REGIME_DEP = 0.03
EXCLUDED_MAX_POS_YEARS = 0.40
MIN_YEARS_FOR_VERDICT = 3

PREDICTIVE_WEIGHTS = {"expected_ic": 0.40, "ic_ir": 0.25,
                      "long_run_spread_ann": 0.25, "long_run_hit_rate": 0.10}
RELIABILITY_WEIGHTS = {"pct_positive_years": 0.35, "persistence_ir": 0.25,
                       "spread_consistency": 0.20, "coverage": 0.10, "n_months": 0.10}

# score_table()'s output schema: base evidence columns plus everything this module
# adds, so the empty-evidence path can return the exact same columns as the
# populated path instead of silently dropping the rank_*/score columns.
_ADDED_COLUMNS = (
    "ic_ir", "classification", "eligible",
    *(f"rank_{m}" for m in {**PREDICTIVE_WEIGHTS, **RELIABILITY_WEIGHTS}),
    "predictive_score", "reliability_score", "production_score",
)


def _classify_row(row: pd.Series) -> str:
    ic = row["long_run_mean_ic"]
    pos_years = row["pct_positive_years"]
    n_years = row["n_years"]
    regime_dep = row["regime_dependence"]
    if pd.isna(ic) or pd.isna(pos_years):
        return "WATCHLIST"
    if ic < 0 and pos_years <= EXCLUDED_MAX_POS_YEARS and n_years >= MIN_YEARS_FOR_VERDICT:
        return "EXCLUDED"
    if (ic >= CORE_MIN_IC and pos_years >= CORE_MIN_POS_YEARS
            and pd.notna(regime_dep) and regime_dep <= CORE_MAX_REGIME_DEP):
        return "CORE"
    if ic >= 0 and pd.notna(regime_dep) and regime_dep > CORE_MAX_REGIME_DEP:
        return "REGIME_DEPENDENT"
    return "WATCHLIST"


def classify_regime_subfactors(evidence: pd.DataFrame) -> pd.Series:
    """CORE / REGIME_DEPENDENT / WATCHLIST / EXCLUDED per row, absolute thresholds
    (not parent-relative) -- see spec Section 2 for the rule table."""
    return evidence.apply(_classify_row, axis=1)


def _add_predictive_reliability_scores(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    """Percentile-ranks every PREDICTIVE_WEIGHTS/RELIABILITY_WEIGHTS metric *within
    ``df`` as given* (caller controls grouping -- e.g. one groupby('parent') slice,
    or the whole frame for a global ranking) and adds predictive_score,
    reliability_score, and their product under ``score_col``. Shared by
    score_table (per-parent groups) and regime_aware_parents.parent_utility_table
    (whole-frame, cross-parent ranking) so the Predictive/Reliability formula lives
    in exactly one place.
    """
    df = df.copy()
    for metric in PREDICTIVE_WEIGHTS:
        df[f"rank_{metric}"] = _pct_rank(df[metric])
    for metric in RELIABILITY_WEIGHTS:
        df[f"rank_{metric}"] = _pct_rank(df[metric])
    df["predictive_score"] = sum(
        w * df[f"rank_{m}"] for m, w in PREDICTIVE_WEIGHTS.items())
    df["reliability_score"] = sum(
        w * df[f"rank_{m}"] for m, w in RELIABILITY_WEIGHTS.items())
    df[score_col] = df["predictive_score"] * df["reliability_score"]
    return df


def score_table(evidence: pd.DataFrame) -> pd.DataFrame:
    """Adds ic_ir, classification, an eligible flag (expected_ic > 0 and not
    EXCLUDED), and per-parent percentile-ranked Predictive/Reliability/Production
    scores. Ranking is grouped by ``parent`` -- a candidate's score reflects its
    standing among its own parent's siblings, never the global pool.
    """
    if evidence.empty:
        return pd.DataFrame(columns=[*evidence.columns, *_ADDED_COLUMNS])

    df = evidence.copy()
    df["ic_ir"] = (df["long_run_mean_ic"] / df["long_run_std_ic"]).replace(
        [np.inf, -np.inf], np.nan)
    df["classification"] = classify_regime_subfactors(df)
    df["eligible"] = (df["expected_ic"] > 0) & (df["classification"] != "EXCLUDED")

    parts = [_add_predictive_reliability_scores(g, "production_score")
             for _, g in df.groupby("parent")]
    return pd.concat(parts, ignore_index=True) if parts else df
