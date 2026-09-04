"""Overlay variants: map the rebalance-date VIX to a tilt strength + a
comparable-VIX training sample, blend the parent weights, recompute composites.

Variants (kept deliberately few — no optimisation grid):

* ``baseline``       — frozen rolling-5Y weights, no VIX awareness.
* ``fixed_bucket``   — absolute Low (<15) / Medium (15-25) / High (>25) VIX
                       buckets; a fixed 30 % tilt toward the same-bucket regime
                       weights outside Medium.
* ``pctile``         — training-relative percentile ladder: no tilt in the
                       30-70th percentile band, then 10 % / 20 % / 30 % as the
                       rebalance-date VIX moves through the 15/85, 5/95 and
                       extreme tails of the training-window distribution.
* ``pctile_strong``  — the same ladder at twice the strength (20/40/60 %) —
                       one dose-response check, not a grid.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .baseline_cache import FrozenWindow
from .composite import composite_scores
from .regime import regime_weights

VIX_LOW, VIX_HIGH = 15.0, 25.0          # fixed-bucket thresholds (repo convention)
FIXED_BUCKET_STRENGTH = 0.30
TAIL_LO, TAIL_HI = 30.0, 70.0           # no-tilt band / regime-sample tails


@dataclass(frozen=True)
class Variant:
    name: str
    kind: str            # "none" | "fixed" | "pctile"
    scale: float = 1.0


VARIANTS: list[Variant] = [
    Variant("baseline", "none"),
    Variant("fixed_bucket", "fixed"),
    Variant("pctile", "pctile", 1.0),
    Variant("pctile_strong", "pctile", 2.0),
]


@dataclass
class RebalDecision:
    """Everything the overlay decided at one test rebalance."""

    window: str
    date: str
    vix: float
    vix_pctile: float          # vs the training-window distribution
    strength: float
    sample_key: str            # "" when no tilt
    sample_n: int
    weights: dict[str, float]
    l1_vs_baseline: float      # 0.5 · Σ|Δw|


def percentile_of(train_values: np.ndarray, v: float) -> float:
    """Mean-rank percentile (0-100) of ``v`` within the training distribution."""
    n = len(train_values)
    if n == 0 or not np.isfinite(v):
        return float("nan")
    less = float((train_values < v).sum())
    equal = float((train_values == v).sum())
    return (less + 0.5 * equal) / n * 100.0


def pctile_strength(pct: float, scale: float = 1.0) -> float:
    """The training-relative ladder: 0 % in the broad middle, stronger only in
    unusually low/high VIX."""
    if not np.isfinite(pct):
        return 0.0
    dist = abs(pct - 50.0)     # symmetric bands around the median
    if dist <= 20.0:           # 30th-70th percentile
        s = 0.0
    elif dist <= 35.0:         # 15th-30th / 70th-85th
        s = 0.10
    elif dist <= 45.0:         # 5th-15th / 85th-95th
        s = 0.20
    else:                      # below 5th / above 95th
        s = 0.30
    return min(1.0, s * scale)


def blend_weights(base: dict[str, float], regime: dict[str, float],
                  strength: float) -> dict[str, float]:
    keys = set(base) | set(regime)
    return {p: (1.0 - strength) * base.get(p, 0.0) + strength * regime.get(p, 0.0)
            for p in keys}


def window_decisions(fw: FrozenWindow, variant: Variant) -> list[RebalDecision]:
    """Per-test-rebalance adjusted weights for one (window, variant)."""
    base = fw.parent_weights
    vt = fw.vix_train.to_numpy(dtype=float)
    lo_val, hi_val = np.percentile(vt, TAIL_LO), np.percentile(vt, TAIL_HI)

    spot_train = {d: fw.vix_at_rebal[d] for d in fw.train_rebals}
    samples = {
        "low_tail": [d for d, v in spot_train.items() if v < lo_val],
        "high_tail": [d for d, v in spot_train.items() if v > hi_val],
        "fixed_low": [d for d, v in spot_train.items() if v < VIX_LOW],
        "fixed_high": [d for d, v in spot_train.items() if v > VIX_HIGH],
    }
    rw_cache: dict[str, tuple[dict | None, dict]] = {}

    def rw(key: str) -> tuple[dict | None, dict]:
        if key not in rw_cache:
            rw_cache[key] = regime_weights(fw.train_stats, samples[key],
                                           fw.train_rebals)
        return rw_cache[key]

    out: list[RebalDecision] = []
    for d in fw.test_rebals:
        v = fw.vix_at_rebal[d]
        pct = percentile_of(vt, v)
        strength, key = 0.0, ""
        if variant.kind == "fixed":
            if v < VIX_LOW:
                key, strength = "fixed_low", FIXED_BUCKET_STRENGTH
            elif v > VIX_HIGH:
                key, strength = "fixed_high", FIXED_BUCKET_STRENGTH
        elif variant.kind == "pctile":
            strength = pctile_strength(pct, variant.scale)
            if strength > 0:
                key = "low_tail" if pct < 50.0 else "high_tail"

        weights, n = base, 0
        if strength > 0 and key:
            rweights, meta = rw(key)
            n = meta["n_sample"]
            if rweights is None:
                strength = 0.0        # thin/degenerate sample → baseline weights
            else:
                weights = blend_weights(base, rweights, strength)
        l1 = 0.5 * sum(abs(weights.get(p, 0.0) - base.get(p, 0.0))
                       for p in set(weights) | set(base))
        out.append(RebalDecision(fw.label, d, v, pct, strength, key if strength else "",
                                 n if strength else 0, weights, l1))
    return out


def variant_scores(fw: FrozenWindow, decisions: list[RebalDecision],
                   sectors: pd.Series) -> dict[str, pd.Series]:
    """Recompute the composite (and thus the ranking) at each test rebalance
    with that rebalance's adjusted weights."""
    out: dict[str, pd.Series] = {}
    for dec in decisions:
        pframe = fw.parent_scores.get(dec.date)
        if pframe is None:
            continue
        out[dec.date] = composite_scores(pframe, dec.weights, sectors)
    return out
