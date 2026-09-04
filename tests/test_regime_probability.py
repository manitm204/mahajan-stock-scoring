"""Tests for smooth VIX regime probabilities, shrinkage, and the expected_ic blend
(research/walkforward/regime_probability.py)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.walkforward.regime_probability import (
    REGIME_ORDER, regime_probabilities, shrink_regime_ic,
    effective_regime_stats, expected_ic,
)


def test_regime_probabilities_sum_to_one_and_nonnegative():
    for vix in [5.0, 15.0, 20.0, 25.0, 40.0]:
        probs = regime_probabilities(vix)
        assert abs(sum(probs.values()) - 1.0) < 1e-9
        assert all(p >= 0.0 for p in probs.values())


def test_regime_probabilities_boundary_is_half_split():
    # At the boundary itself, Low/Medium split exactly 50/50 regardless of width
    # (sigmoid(0)==0.5 always); with width=4 the *other* boundary (High, 10 points
    # away) still bleeds in a little (~0.076) -- that bleed is the point of a smooth
    # transition, not a bug, so this checks the exact values at width=4 rather than
    # assuming High is ~0.
    probs = regime_probabilities(15.0)
    assert probs["Low (<15)"] == pytest.approx(0.5, abs=1e-9)
    assert probs["Medium (15-25)"] == pytest.approx(0.424142, abs=1e-4)
    assert probs["High (>25)"] == pytest.approx(0.075858, abs=1e-4)


def test_regime_probabilities_center_dominates():
    # At the exact bucket center (vix=20, halfway between the 15/25 boundaries),
    # Medium is the plurality but width=4 is wide enough that it isn't overwhelming
    # (~0.55, not ~1.0) -- that's the smooth-transition tradeoff the spec's width=4
    # explicitly chooses over a near-hard cutoff.
    probs = regime_probabilities(20.0)
    assert probs["Medium (15-25)"] > 0.5
    assert probs["Medium (15-25)"] > probs["Low (<15)"]
    assert probs["Medium (15-25)"] > probs["High (>25)"]


def test_regime_probabilities_nan_vix_returns_nan():
    probs = regime_probabilities(float("nan"))
    assert all(np.isnan(p) for p in probs.values())


def test_shrink_pulls_toward_long_run_with_thin_sample():
    shrunk = shrink_regime_ic(regime_ic=0.10, long_run_ic=0.01, n_eff=1.0, k=24.0)
    lam = 1.0 / (1.0 + 24.0)
    assert shrunk == pytest.approx(lam * 0.10 + (1 - lam) * 0.01, abs=1e-9)


def test_shrink_trusts_observed_with_large_sample():
    shrunk = shrink_regime_ic(regime_ic=0.10, long_run_ic=0.01, n_eff=1000.0, k=24.0)
    assert shrunk == pytest.approx(0.10, abs=0.01)


def test_shrink_falls_back_to_long_run_when_no_observations():
    shrunk = shrink_regime_ic(regime_ic=float("nan"), long_run_ic=0.02, n_eff=0.0, k=24.0)
    assert shrunk == pytest.approx(0.02, abs=1e-9)


def test_effective_regime_stats_weights_by_probability():
    # Two dates: VIX=10 (heavily Low) and VIX=30 (heavily High), IC=0.10 and IC=0.02.
    vix = pd.Series({"2020-01-31": 10.0, "2020-02-29": 30.0})
    ic = pd.Series({"2020-01-31": 0.10, "2020-02-29": 0.02})
    stats = effective_regime_stats(vix, ic)
    assert set(stats) == set(REGIME_ORDER)
    # Low regime's n_eff should come almost entirely from the VIX=10 date.
    assert stats["Low (<15)"]["n_eff"] == pytest.approx(
        regime_probabilities(10.0)["Low (<15)"] + regime_probabilities(30.0)["Low (<15)"],
        abs=1e-9)
    assert stats["Low (<15)"]["regime_ic"] == pytest.approx(0.10, abs=0.02)
    assert stats["High (>25)"]["regime_ic"] == pytest.approx(0.02, abs=0.02)


def test_effective_regime_stats_empty_series_returns_nan_stats():
    stats = effective_regime_stats(pd.Series(dtype=float), pd.Series(dtype=float))
    for r in REGIME_ORDER:
        assert stats[r]["n_eff"] == 0.0
        assert np.isnan(stats[r]["regime_ic"])


def test_expected_ic_blends_50_25_25():
    # Flat regime evidence equal to long_run_ic → regime component collapses to
    # long_run_ic too, so expected_ic == 0.75*long_run + 0.25*recent exactly.
    regime_stats = {r: {"n_eff": 1000.0, "regime_ic": 0.05} for r in REGIME_ORDER}
    result = expected_ic(long_run_ic=0.05, recent_ic=0.09, vix_now=20.0,
                         regime_stats=regime_stats)
    assert result == pytest.approx(0.75 * 0.05 + 0.25 * 0.09, abs=1e-6)


def test_expected_ic_falls_back_to_long_run_when_recent_missing():
    regime_stats = {r: {"n_eff": 0.0, "regime_ic": float("nan")} for r in REGIME_ORDER}
    result = expected_ic(long_run_ic=0.03, recent_ic=float("nan"), vix_now=20.0,
                         regime_stats=regime_stats)
    assert result == pytest.approx(0.03, abs=1e-9)


def test_expected_ic_nan_long_run_is_nan():
    regime_stats = {r: {"n_eff": 0.0, "regime_ic": float("nan")} for r in REGIME_ORDER}
    result = expected_ic(long_run_ic=float("nan"), recent_ic=0.05, vix_now=20.0,
                         regime_stats=regime_stats)
    assert np.isnan(result)
