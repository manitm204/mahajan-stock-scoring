"""VIX-banded momentum tilt on the composite parent weights.

Validated in the walk-forward study ``output/vixtilt_smooth/`` (variant ``lit_band``,
2026-07-13): a fixed, literature-motivated rule — cut momentum and redistribute to
quality+value when volatility is elevated, mildly boost momentum / trim value when it
is unusually calm — beat the frozen baseline net of costs at every book size and in
14/19 semiannual OOS windows, while global/estimated tilt variants did not.

The multipliers are continuous piecewise-linear in the spot VIX so a hair's-width VIX
move cannot flip the composite (no cliff at the 15/25 thresholds):

* VIX ≤ 13         momentum ×1.25, value ×0.75 (full low-vol boost)
* 13 → 15          boost fades linearly to neutral
* 15 → 23          neutral — baseline weights untouched
* 23 → 27          momentum ramps ×1.0 → ×0.5, freed weight → quality+value pro-rata
* VIX ≥ 27         pinned at the ×0.5 floor (never deeper; ×0 tested worse in 2020)

All constants are frozen as validated; the only runtime switch is
``factors.vix_tilt.enabled`` in config.yaml.
"""
from __future__ import annotations

import math

# Frozen rule constants — see module docstring; do not tune without re-running
# the pre-registered walk-forward (run_vixtilt_study.py --variant-set smooth).
MOM_BOOST = 1.25          # low-vol momentum multiplier
VAL_CUT = 0.75            # low-vol value multiplier
MOM_FLOOR = 0.50          # high-vol momentum multiplier floor
LO_FULL, LO_EDGE = 13.0, 15.0     # low-side band: full boost ≤13, neutral ≥15
HI_EDGE, HI_FULL = 23.0, 27.0     # high-side band: neutral ≤23, floor ≥27
PARENT_CAP = 0.25         # same cap as the V4 parent-weight construction


def band_multipliers(vix: float | None) -> tuple[float, float]:
    """(momentum_mult, value_mult) — continuous piecewise-linear in spot VIX."""
    if vix is None or not math.isfinite(vix):
        return 1.0, 1.0
    if vix >= HI_EDGE:
        frac = min(1.0, (vix - HI_EDGE) / (HI_FULL - HI_EDGE))
        return 1.0 - (1.0 - MOM_FLOOR) * frac, 1.0
    if vix <= LO_EDGE:
        frac = min(1.0, (LO_EDGE - vix) / (LO_EDGE - LO_FULL))
        return 1.0 + frac * (MOM_BOOST - 1.0), 1.0 - frac * (1.0 - VAL_CUT)
    return 1.0, 1.0


def _cap_renorm(weights: dict[str, float], cap: float = PARENT_CAP) -> dict[str, float]:
    """Water-fill weights to the per-parent cap and renormalise to 1 (sorted
    iteration for run-to-run determinism)."""
    w = {p: float(weights[p]) for p in sorted(weights) if weights[p] > 0}
    if not w:
        return {}
    tot = sum(w.values())
    w = {p: x / tot for p, x in w.items()}
    for _ in range(20):
        over = [p for p, x in w.items() if x > cap + 1e-12]
        if not over:
            break
        excess = sum(w[p] - cap for p in over)
        for p in over:
            w[p] = cap
        under = [p for p, x in w.items() if x < cap - 1e-12]
        pool = sum(w[p] for p in under)
        if not under or pool <= 1e-12:
            break
        for p in under:
            w[p] += excess * w[p] / pool
    tot = sum(w.values())
    return {p: x / tot for p, x in w.items()}


def apply_vix_tilt(weights: dict[str, float],
                   vix: float | None) -> tuple[dict[str, float], float]:
    """Return (tilted weight vector summing to 1, momentum multiplier applied).

    A multiplier of 1.0 means the input vector is returned unchanged (neutral VIX
    band, missing VIX, or no momentum weight to act on).
    """
    mom_m, val_m = band_multipliers(vix)
    if (abs(mom_m - 1.0) < 1e-12 and abs(val_m - 1.0) < 1e-12) or not weights:
        return dict(weights), 1.0
    w = dict(weights)
    if mom_m < 1.0:                       # high side: cut momentum, free → Q+V
        freed = w.get("momentum", 0.0) * (1.0 - mom_m)
        if "momentum" in w:
            w["momentum"] *= mom_m
        qv_tot = w.get("quality", 0.0) + w.get("value", 0.0)
        for p in ("quality", "value"):
            share = (w.get(p, 0.0) / qv_tot) if qv_tot > 0 else 0.5
            w[p] = w.get(p, 0.0) + freed * share
    else:                                 # low side: boost momentum, trim value
        if "momentum" in w:
            w["momentum"] *= mom_m
        if "value" in w:
            w["value"] *= val_m
    return _cap_renorm(w), mom_m
