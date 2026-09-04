"""Tests for the VIX-banded momentum tilt (factors/vix_tilt.py) and its wiring."""
from __future__ import annotations

import math

import pytest

from factors.vix_tilt import PARENT_CAP, apply_vix_tilt, band_multipliers

BASE = {
    "momentum": 0.187, "value": 0.071, "quality": 0.012, "growth": 0.216,
    "revisions": 0.141, "institutional": 0.149, "insider": 0.108, "short": 0.116,
}


# --------------------------------------------------------------------------- #
# band_multipliers — shape and continuity
# --------------------------------------------------------------------------- #
def test_anchor_points():
    assert band_multipliers(10.0) == (1.25, 0.75)
    assert band_multipliers(13.0) == (1.25, 0.75)
    assert band_multipliers(15.0) == (1.0, 1.0)
    assert band_multipliers(20.0) == (1.0, 1.0)
    assert band_multipliers(23.0) == (1.0, 1.0)
    assert band_multipliers(25.0) == (0.75, 1.0)
    assert band_multipliers(27.0) == (0.5, 1.0)
    assert band_multipliers(53.0) == (0.5, 1.0)      # floor, never deeper


def test_no_cliff_anywhere():
    """Adjacent VIX values (0.1 apart) never move the momentum mult by > 0.02."""
    prev = band_multipliers(5.0)[0]
    v = 5.1
    while v < 60.0:
        cur = band_multipliers(v)[0]
        assert abs(cur - prev) < 0.02, f"jump at VIX {v:.1f}"
        prev, v = cur, v + 0.1


def test_missing_vix_is_neutral():
    assert band_multipliers(None) == (1.0, 1.0)
    assert band_multipliers(float("nan")) == (1.0, 1.0)


# --------------------------------------------------------------------------- #
# apply_vix_tilt — weight vector behaviour
# --------------------------------------------------------------------------- #
def test_neutral_band_returns_weights_unchanged():
    w, mult = apply_vix_tilt(BASE, 18.0)
    assert mult == 1.0
    assert w == BASE


def test_high_vix_cuts_momentum_and_feeds_quality_value():
    w, mult = apply_vix_tilt(BASE, 27.0)
    assert mult == 0.5
    assert sum(w.values()) == pytest.approx(1.0)
    # momentum roughly halved (small drift from the final cap/renormalise)
    assert w["momentum"] / BASE["momentum"] == pytest.approx(0.5, abs=0.05)
    # freed weight went to quality and value
    assert w["quality"] > BASE["quality"]
    assert w["value"] > BASE["value"]
    # untouched parents keep their relative ordering
    assert w["growth"] > w["revisions"] > w["short"]


def test_low_vix_boosts_momentum_trims_value():
    w, mult = apply_vix_tilt(BASE, 14.0)          # mid-band: frac = 0.5
    assert mult == pytest.approx(1.125)
    assert sum(w.values()) == pytest.approx(1.0)
    assert w["momentum"] > BASE["momentum"]
    assert w["value"] < BASE["value"]


def test_cap_respected():
    w, _ = apply_vix_tilt(BASE, 40.0)
    assert max(w.values()) <= PARENT_CAP + 1e-9
    assert sum(w.values()) == pytest.approx(1.0)


def test_missing_momentum_key_is_safe():
    base = {"value": 0.5, "quality": 0.5}
    w, mult = apply_vix_tilt(base, 30.0)
    assert sum(w.values()) == pytest.approx(1.0)
    assert math.isfinite(mult)


# --------------------------------------------------------------------------- #
# resolve_weights wiring — config gate on/off
# --------------------------------------------------------------------------- #
class _FakeCfg:
    def __init__(self, tilt_enabled: bool):
        self._tilt = tilt_enabled

    def get(self, *keys, default=None):
        path = tuple(keys)
        if path == ("factors", "default_weights"):
            return dict(BASE)
        if path == ("factors", "vix_tilt", "enabled"):
            return self._tilt
        if path == ("factors", "regime", "low_vol_max"):
            return 15.0
        if path == ("factors", "regime", "high_vol_min"):
            return 25.0
        return default


class _FakeCtx:
    def __init__(self, vix: float | None):
        self._vix = vix

    def vix(self):
        return self._vix


@pytest.mark.parametrize("vix,expect_tilt", [(30.0, True), (18.0, False), (None, False)])
def test_resolve_weights_gate(vix, expect_tilt):
    from factors.regime_weights import resolve_weights

    dec = resolve_weights(_FakeCfg(tilt_enabled=True), _FakeCtx(vix), use_regime=False)
    assert sum(dec.weights.values()) == pytest.approx(1.0)
    assert ("vix_tilt" in dec.applied) == expect_tilt
    if expect_tilt:
        assert dec.weights["momentum"] < BASE["momentum"]

    dec_off = resolve_weights(_FakeCfg(tilt_enabled=False), _FakeCtx(vix),
                              use_regime=False)
    assert "vix_tilt" not in dec_off.applied
    assert dec_off.weights == BASE
