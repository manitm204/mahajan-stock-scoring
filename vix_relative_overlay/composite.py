"""Composite construction from frozen parent scores — reimplemented for this study.

Follows the production composite conventions (dispersion-equalise each parent
about the neutral 50, fill missing parents at 50, blend by the parent weights,
re-rank within GICS sector to a 0-100 percentile) so that a weight change is
the *only* lever separating the baseline from the VIX-adjusted variants.
Fidelity against the research implementation is asserted once at cache-build
time (``baseline_cache.verify_composite_fidelity``).

One deliberate deviation from the research path: composites are computed over
each date's point-in-time members only (the panel frame index), not the
all-time universe union — departed/future names never enter a date's ranking.
Applied identically to every variant.
"""
from __future__ import annotations

import pandas as pd

NEUTRAL = 50.0
TARGET_STD = 20.0
MIN_STD = 1e-6
MIN_SECTOR_OBS = 5


def normalize_parents(parents: pd.DataFrame,
                      target_std: float = TARGET_STD) -> pd.DataFrame:
    """Rescale each parent column to a common std about the neutral 50."""
    out = parents.copy()
    for col in out.columns:
        s = out[col]
        std = float(s.std())
        if std < MIN_STD:
            continue  # no cross-sectional signal → leave as-is (missing already 50)
        out[col] = NEUTRAL + (s - NEUTRAL) * (target_std / std)
    return out


def sector_percentile(values: pd.Series, sectors: pd.Series,
                      min_obs: int = MIN_SECTOR_OBS) -> pd.Series:
    """0-100 within-sector percentile ranks; missing values and sectors with
    fewer than ``min_obs`` valid names score the neutral 50."""
    frame = pd.DataFrame({"v": values}).reindex(sectors.index)
    frame["sec"] = sectors.values
    out = pd.Series(50.0, index=sectors.index, dtype="float64")
    for _, grp in frame.groupby("sec", sort=False):
        valid = grp["v"].dropna()
        if len(valid) < min_obs:
            continue
        out.loc[valid.index] = (valid.rank(pct=True) * 100.0).round(4)
    return out


def composite_scores(parent_frame: pd.DataFrame, weights: dict[str, float],
                     sectors: pd.Series) -> pd.Series:
    """0-100 sector-relative composite for one rebalance date.

    ``parent_frame`` is (ticker × parent); ``weights`` maps parent → composite
    weight (need not cover every column). Missing parents contribute the
    neutral 50; the blend renormalises by the weight actually used.
    """
    cols = [p for p in weights if p in parent_frame.columns and weights[p] > 0]
    blend = normalize_parents(parent_frame[cols])
    comp = pd.Series(0.0, index=parent_frame.index, dtype="float64")
    used = 0.0
    for p in cols:
        comp += weights[p] * blend[p].fillna(NEUTRAL)
        used += weights[p]
    if used > 0:
        comp /= used
    secs = sectors.reindex(parent_frame.index).fillna("Unknown")
    return sector_percentile(comp, secs)
