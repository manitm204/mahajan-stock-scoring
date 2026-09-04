"""Composite score and candidate classification.

Blends the parent factor scores with the (possibly regime-conditional) weight
vector, then **re-ranks the blend within each GICS sector** so the final
composite is itself sector-neutral on a 0–100 scale. Candidates are labeled from
that sector-relative percentile:

* top band (default >= 80th pct)   -> LONG
* bottom band (default <= 20th pct) -> SHORT
* middle                            -> WATCHLIST

Because classification uses the sector-relative percentile, roughly the same
fraction of every sector is eligible long and short — no sector dominates a side
purely because it screens cheap or expensive in aggregate.
"""
from __future__ import annotations

from typing import Sequence

import pandas as pd

from data.config import Config

from .base import FactorResult
from .utils import sector_percentile

NEUTRAL = 50.0
_MIN_STD = 1e-6


def _normalize_parents(
    parents: pd.DataFrame, mode: str, target_std: float
) -> pd.DataFrame:
    """Equalize each parent factor's cross-sectional dispersion before weighting.

    Parent scores are 0–100 sector-relative percentiles, but their spreads differ
    widely (Momentum std ~26 vs Insider ~12 — heavy ties at the neutral 50 for
    thin-data factors compress the latter). In a raw weighted blend the pull a
    factor exerts is ``weight × std``, so a wide factor out-votes a compressed one
    even at equal weight. ``zscore`` rescales every factor to the same std about
    the fixed neutral 50::

        normalized = 50 + (parent - 50) · (target_std / std)

    After this each factor contributes identical dispersion, so ``weight`` becomes
    the sole influence lever. ``target_std`` is a global constant that cancels in
    the downstream sector-percentile re-rank, so it changes nothing in the final
    ``composite_score`` — it only keeps ``composite_raw`` on a readable ~0–100
    scale. Factors with no dispersion (all names neutral → std≈0) are left at 50
    so a dead factor adds no push. ``mode="none"`` returns the parents unchanged
    (legacy behaviour).
    """
    if mode == "none":
        return parents
    if mode != "zscore":
        raise ValueError(f"Unknown score_normalization mode: {mode!r}")
    out = parents.copy()
    for col in out.columns:
        s = out[col]
        std = float(s.std())
        if std < _MIN_STD:
            continue  # no cross-sectional signal → leave as-is (missing already 50)
        out[col] = NEUTRAL + (s - NEUTRAL) * (target_std / std)
    return out


def build_composite(
    results: Sequence[FactorResult],
    weights: dict[str, float],
    sectors: pd.Series,
    cfg: Config,
    min_obs: int = 5,
) -> pd.DataFrame:
    """Assemble the composite/classification frame indexed by ticker."""
    universe = sectors.index
    parents = pd.DataFrame({res.key: res.parent.reindex(universe) for res in results})

    # Normalize per-factor dispersion so `weight` is the true lever (see
    # _normalize_parents). Blend uses the normalized copy; the raw parent
    # percentiles are still reported unchanged as the `*_score` columns.
    norm_mode = str(cfg.get("factors", "score_normalization", default="none"))
    target_std = float(cfg.get("factors", "normalization_target_std", default=20.0))
    blend_parents = _normalize_parents(parents, norm_mode, target_std)

    # Weighted blend of (normalized) parent scores; missing already neutral at 50.
    composite_raw = pd.Series(0.0, index=universe)
    used = 0.0
    for key, weight in weights.items():
        if key in blend_parents.columns:
            composite_raw += weight * blend_parents[key].fillna(NEUTRAL)
            used += weight
    if used > 0:
        composite_raw /= used     # renormalize if any weighted factor is absent

    # Re-rank the blend within sector -> final sector-neutral composite score.
    composite_score = sector_percentile(composite_raw, sectors, higher_is_better=True, min_obs=min_obs)

    out = parents.copy()
    out.columns = [f"{c}_score" for c in out.columns]
    out.insert(0, "sector", sectors)
    out["composite_raw"] = composite_raw.round(4)
    out["composite_score"] = composite_score.round(4)
    out["sector_rank"] = (
        composite_score.groupby(sectors).rank(ascending=False, method="min").astype("Int64")
    )

    long_pct = float(cfg.get("factors", "classification", "long_percentile", default=80))
    short_pct = float(cfg.get("factors", "classification", "short_percentile", default=20))
    out["long_short_flag"] = _classify(composite_score, long_pct, short_pct)
    return out.sort_values("composite_score", ascending=False)


def _classify(score: pd.Series, long_pct: float, short_pct: float) -> pd.Series:
    flag = pd.Series("WATCHLIST", index=score.index, dtype=object)
    flag[score >= long_pct] = "LONG"
    flag[score <= short_pct] = "SHORT"
    return flag
