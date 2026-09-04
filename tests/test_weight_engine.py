"""Tests for the Factor Weight Engine (regime, metrics, weighting).

The engine integration tests build a synthetic world where one factor
("momentum") perfectly predicts forward returns and the others are noise, then
assert the adaptive overlay tilts toward the predictive factor *without*
breaching the stable-baseline discipline (bounds, sum-to-one, bounded tilt).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from factors.factor_effectiveness import (
    PeriodObs,
    compute_effectiveness,
    quintile_spread,
    spearman_ic,
    summarize_factor,
)
from factors.market_regime import (
    NEUTRAL,
    RISK_OFF,
    RISK_ON,
    classify_regime,
)
from factors.weight_engine import FactorWeightEngine, project_to_simplex_box


# ---------------------------------------------------------------------------
# Constraint projection
# ---------------------------------------------------------------------------
def test_projection_sums_to_one_within_bounds():
    out = project_to_simplex_box({"a": 0.9, "b": 0.05, "c": 0.05}, lo=0.1, hi=0.5)
    assert math.isclose(sum(out.values()), 1.0, rel_tol=1e-9)
    assert all(0.1 - 1e-9 <= v <= 0.5 + 1e-9 for v in out.values())
    # The over-cap factor 'a' is pulled down to the cap.
    assert out["a"] == pytest.approx(0.5, abs=1e-6)


def test_projection_identity_when_feasible():
    w = {"a": 0.25, "b": 0.25, "c": 0.5}
    out = project_to_simplex_box(w, lo=0.0, hi=1.0)
    assert all(out[k] == pytest.approx(w[k], abs=1e-9) for k in w)


def test_projection_lifts_below_floor():
    out = project_to_simplex_box({"a": 0.0, "b": 0.0, "c": 1.0}, lo=0.2, hi=0.6)
    assert all(v >= 0.2 - 1e-9 for v in out.values())
    assert all(v <= 0.6 + 1e-9 for v in out.values())
    assert math.isclose(sum(out.values()), 1.0, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# Metric primitives
# ---------------------------------------------------------------------------
def test_spearman_ic_perfect_rank():
    s = pd.Series(range(30), index=[f"T{i}" for i in range(30)])
    f = pd.Series(np.arange(30) * 0.01, index=s.index)
    assert spearman_ic(s, f, min_names=20) == pytest.approx(1.0, abs=1e-9)


def test_spearman_ic_too_few_names_returns_none():
    s = pd.Series(range(10), index=[f"T{i}" for i in range(10)])
    f = pd.Series(np.arange(10) * 0.01, index=s.index)
    assert spearman_ic(s, f, min_names=20) is None


def test_quintile_spread_positive_when_high_scores_win():
    idx = [f"T{i}" for i in range(40)]
    s = pd.Series(range(40), index=idx)
    f = pd.Series(np.arange(40) * 0.001, index=idx)
    spread = quintile_spread(s, f, q=0.2, min_names=20)
    assert spread is not None and spread > 0


def test_effectiveness_zscore_ranks_factors():
    metrics = {
        "good": summarize_factor("good", [PeriodObs("d", 0.2, 0.02)] * 6),
        "bad": summarize_factor("bad", [PeriodObs("d", -0.2, -0.02)] * 6),
        "flat": summarize_factor("flat", [PeriodObs("d", 0.0, 0.0)] * 6),
    }
    eff = compute_effectiveness(metrics)
    assert eff["good"] > eff["flat"] > eff["bad"]


# ---------------------------------------------------------------------------
# Regime classification
# ---------------------------------------------------------------------------
def test_regime_risk_on_calm_and_uptrend():
    st = classify_regime(vix=12.0, spy_to_200dma=0.05, low_vix=15, high_vix=25)
    assert st.label == RISK_ON


def test_regime_risk_off_on_fear_or_downtrend():
    assert classify_regime(30.0, 0.05).label == RISK_OFF      # fear
    assert classify_regime(12.0, -0.05).label == RISK_OFF     # downtrend


def test_regime_neutral_when_mixed_or_unknown():
    assert classify_regime(20.0, 0.05).label == NEUTRAL       # mid vix, up
    assert classify_regime(None, None).label == NEUTRAL       # no inputs


# ---------------------------------------------------------------------------
# Engine integration — synthetic predictive world
# ---------------------------------------------------------------------------
KEYS = ["momentum", "value", "quality", "growth"]
N_TICKERS = 40
N_DATES = 16


def _build_world(seed: int = 0):
    """Prices where forward returns rank by a fixed 'momentum' ordering.

    Momentum scores are fixed and equal to the true return ranking (IC ~ +1
    every period); the other factor scores are reshuffled each date so their
    ICs average to ~0. Returns (dates, price_matrix, scores_by_date).
    """
    rng = np.random.default_rng(seed)
    tickers = [f"T{i:02d}" for i in range(N_TICKERS)]
    dates = [f"2024-{m:02d}-01" for m in range(1, N_DATES + 1)]
    rank = np.arange(N_TICKERS)                      # 0..39, higher = better
    mom_score = pd.Series(rank.astype(float) / (N_TICKERS - 1) * 100.0, index=tickers)

    # Build prices: each period return is monotonically increasing in rank.
    prices = {dates[0]: pd.Series(100.0, index=tickers)}
    for k in range(1, N_DATES):
        ret = 0.0008 * (rank - rank.mean())          # high rank -> positive return
        prices[dates[k]] = prices[dates[k - 1]] * (1.0 + ret)
    matrix = pd.DataFrame(prices).T                  # date x ticker
    matrix.index.name = "date"

    scores_by_date = {}
    for d in dates:
        cols = {"momentum_score": mom_score}
        for key in KEYS:
            if key == "momentum":
                continue
            shuffled = rng.permutation(N_TICKERS).astype(float) / (N_TICKERS - 1) * 100.0
            cols[f"{key}_score"] = pd.Series(shuffled, index=tickers)
        scores_by_date[d] = pd.DataFrame(cols)
    return dates, matrix, scores_by_date


def _run(engine, dates, scores_by_date, regime):
    decisions = {}
    for d in dates:
        decisions[d] = engine.weights_for(d, regime)
        engine.record(d, scores_by_date[d], regime)
    return decisions


def test_engine_warmup_returns_baseline():
    dates, matrix, scores = _build_world()
    base = {k: 0.25 for k in KEYS}
    engine = FactorWeightEngine(KEYS, base, matrix, min_periods=5, smoothing=0.0)
    dec = engine.weights_for(dates[0], classify_regime(20.0, 0.0))
    assert dec.applied == "baseline:warmup"
    assert all(dec.weights[k] == pytest.approx(0.25, abs=1e-9) for k in KEYS)


def test_engine_tilts_toward_predictive_factor():
    dates, matrix, scores = _build_world()
    base = {k: 0.25 for k in KEYS}
    engine = FactorWeightEngine(
        KEYS, base, matrix, overlay_strength=0.5, min_weight=0.05, max_weight=0.6,
        smoothing=0.0, min_periods=4, regime_blend=0.0)
    neutral = classify_regime(20.0, 0.0)
    decisions = _run(engine, dates, scores, neutral)

    final = decisions[dates[-1]]
    assert final.applied == "overlay"
    # Momentum is the predictive factor -> it should carry the most weight and
    # exceed its baseline; the noise factors should be at/under baseline.
    assert final.weights["momentum"] == max(final.weights.values())
    assert final.weights["momentum"] > 0.25
    for noise in ("value", "quality", "growth"):
        assert final.weights[noise] <= 0.25 + 1e-9
    assert math.isclose(sum(final.weights.values()), 1.0, rel_tol=1e-9)
    assert all(0.05 - 1e-9 <= w <= 0.6 + 1e-9 for w in final.weights.values())
    # The overlay is bounded: it cannot move more weight than overlay_strength.
    moved = sum(abs(final.weights[k] - final.baseline[k]) for k in KEYS)
    assert moved <= 0.5 + 1e-6
    assert final.confidence > 0.0
    assert final.effectiveness["momentum"] > 0.0


def test_engine_smoothing_dampens_changes():
    dates, matrix, scores = _build_world()
    base = {k: 0.25 for k in KEYS}
    sharp = FactorWeightEngine(KEYS, base, matrix, overlay_strength=0.5,
                               smoothing=0.0, min_periods=4, regime_blend=0.0)
    smooth = FactorWeightEngine(KEYS, base, matrix, overlay_strength=0.5,
                                smoothing=0.8, min_periods=4, regime_blend=0.0)
    neutral = classify_regime(20.0, 0.0)
    d_sharp = _run(sharp, dates, scores, neutral)[dates[-1]]
    d_smooth = _run(smooth, dates, scores, neutral)[dates[-1]]
    # Both tilt the same direction, but smoothing keeps momentum closer to base.
    assert d_smooth.weights["momentum"] < d_sharp.weights["momentum"]
    assert d_smooth.weights["momentum"] > 0.25


def test_engine_confidence_scales_overlay():
    dates, matrix, scores = _build_world()
    base = {k: 0.25 for k in KEYS}
    engine = FactorWeightEngine(KEYS, base, matrix, overlay_strength=0.4,
                                smoothing=0.0, min_periods=4, regime_blend=0.0)
    neutral = classify_regime(20.0, 0.0)
    final = _run(engine, dates, scores, neutral)[dates[-1]]
    assert 0.0 < final.confidence <= 1.0
    assert final.overlay_strength_effective == pytest.approx(
        0.4 * final.confidence, abs=1e-9)


def test_engine_rejects_infeasible_bounds():
    with pytest.raises(ValueError):
        FactorWeightEngine(KEYS, {k: 0.25 for k in KEYS}, pd.DataFrame(),
                           max_weight=0.2)  # 0.2 * 4 < 1 -> infeasible
