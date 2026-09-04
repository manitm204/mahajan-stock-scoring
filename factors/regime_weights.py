"""Regime-conditional composite weights.

The market is bucketed into a volatility regime from the VIX close and, when
enabled, the composite uses a regime-specific weight table (low vol tilts toward
momentum/growth; high vol toward quality/value). Weighting is opt-in via
``--regime-weights`` or ``factors.regime_conditional_weights`` in config; the
default is the static "normal" blend.

Every weight table is validated to sum to 1.0 at resolution time so a mis-edited
config fails loudly instead of silently skewing the composite.
"""
from __future__ import annotations

from dataclasses import dataclass

from data.config import Config

from .utils import DataContext
from .vix_tilt import apply_vix_tilt

LOW_VOL = "low_vol"
NORMAL = "normal"
HIGH_VOL = "high_vol"


@dataclass
class RegimeDecision:
    weights: dict[str, float]
    regime: str             # detected volatility regime
    applied: str            # "regime:<x>" or "static"
    vix: float | None


def detect_regime(vix: float | None, low_max: float, high_min: float) -> str:
    """Bucket the VIX close into a volatility regime."""
    if vix is None:
        return NORMAL
    if vix < low_max:
        return LOW_VOL
    if vix > high_min:
        return HIGH_VOL
    return NORMAL


def validate_weights(weights: dict[str, float], label: str, tol: float = 1e-6) -> None:
    total = sum(weights.values())
    if abs(total - 1.0) > tol:
        raise ValueError(
            f"Factor weights for '{label}' sum to {total:.6f}, expected 1.0. "
            "Fix config.yaml -> factors weights."
        )


def resolve_weights(cfg: Config, ctx: DataContext, use_regime: bool,
                    use_engine: bool | None = None) -> RegimeDecision:
    """Pick and validate the composite weight vector for this run.

    Precedence: Factor Weight Engine weights (when enabled) > regime table >
    static default. ``use_engine`` overrides the ``factors.use_engine_weights``
    config toggle when not None.
    """
    default = {k: float(v) for k, v in cfg.get("factors", "default_weights", default={}).items()}
    validate_weights(default, "default_weights")

    vix = ctx.vix()
    low_max = float(cfg.get("factors", "regime", "low_vol_max", default=15.0))
    high_min = float(cfg.get("factors", "regime", "high_vol_min", default=25.0))
    regime = detect_regime(vix, low_max, high_min)

    def _finalize(weights: dict[str, float], applied: str) -> RegimeDecision:
        # VIX-banded momentum tilt (factors/vix_tilt.py) on whichever base vector
        # won the precedence above. Neutral band / missing VIX → unchanged.
        if bool(cfg.get("factors", "vix_tilt", "enabled", default=False)):
            weights, mom_mult = apply_vix_tilt(weights, vix)
            if abs(mom_mult - 1.0) > 1e-12:
                applied = f"{applied}+vix_tilt(mom×{mom_mult:.2f})"
        return RegimeDecision(weights, regime, applied, vix)

    engine_on = (use_engine if use_engine is not None
                 else bool(cfg.get("factors", "use_engine_weights", default=False)))
    if engine_on:
        table = cfg.get("factors", "engine_weights", default=None)
        if table:
            weights = {k: float(v) for k, v in table.items()}
            validate_weights(weights, "engine_weights")
            return _finalize(weights, "engine")

    enabled = use_regime or bool(cfg.get("factors", "regime_conditional_weights", default=False))
    if not enabled:
        return _finalize(default, "static")

    table = cfg.get("factors", "regime", "weights", regime, default=None)
    if not table:
        return _finalize(default, "static")
    weights = {k: float(v) for k, v in table.items()}
    validate_weights(weights, f"regime.{regime}")
    return _finalize(weights, f"regime:{regime}")
