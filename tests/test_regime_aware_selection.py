from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research import compute_forward_returns
from research.panel import ScorePanel
from research.walkforward.regime_aware_selection import (
    _apply_degradation_check, _composite_ic_stats, parent_subfactor_weights,
    select_parent_subfactors,
)


def test_parent_subfactor_weights_proportional_cap_and_floor():
    scores = pd.Series({"s1": 0.9, "s2": 0.08, "s3": 0.02})
    w = parent_subfactor_weights(["s1", "s2", "s3"], scores, cap=0.50, min_weight=0.10)
    assert abs(sum(w.values()) - 1.0) < 1e-9
    assert max(w.values()) <= 0.50 + 1e-9
    assert all(v >= 0.10 - 1e-9 for v in w.values())


def test_parent_subfactor_weights_single_sub_gets_full_weight():
    w = parent_subfactor_weights(["s1"], pd.Series({"s1": 0.5}))
    assert w == {"s1": pytest.approx(1.0)}


def test_parent_subfactor_weights_floor_survives_renormalization():
    # s3's raw share lands below min_weight only *after* cap water-filling redistributes
    # s1's excess -- naively renormalizing everyone (including the floored s3) by the
    # post-floor total would push s3 back under 0.10. The floor must be the final word.
    scores = pd.Series({"s1": 0.85, "s2": 0.14, "s3": 0.01})
    w = parent_subfactor_weights(["s1", "s2", "s3"], scores, cap=0.50, min_weight=0.10)
    assert abs(sum(w.values()) - 1.0) < 1e-9
    assert all(v >= 0.10 - 1e-9 for v in w.values())
    assert w["s3"] == pytest.approx(0.10, abs=1e-9)


def test_parent_subfactor_weights_empty_selection():
    assert parent_subfactor_weights([], pd.Series(dtype=float)) == {}


def test_apply_degradation_check_stops_on_material_drop():
    metric_map = pd.DataFrame({"production_score": [0.9, 0.5]}, index=["s1", "s2"])
    # Adding s2 drops mean_ic from 0.05 to 0.03 -- a 40% relative drop, breaching 10%.
    fake_stats = {
        frozenset({"s1"}): {"mean_ic": 0.05, "ic_ir": 1.0, "spread": 0.05},
        frozenset({"s1", "s2"}): {"mean_ic": 0.03, "ic_ir": 1.0, "spread": 0.05},
    }
    kept, stop = _apply_degradation_check(
        ["s1", "s2"], metric_map, lambda w: fake_stats[frozenset(w)],
        cap=0.5, min_weight=0.10, tol=0.10)
    assert kept == ["s1"]
    assert stop is not None and "degraded" in stop


def test_apply_degradation_check_keeps_non_degrading_addition():
    metric_map = pd.DataFrame({"production_score": [0.9, 0.5]}, index=["s1", "s2"])
    fake_stats = {
        frozenset({"s1"}): {"mean_ic": 0.05, "ic_ir": 1.0, "spread": 0.05},
        frozenset({"s1", "s2"}): {"mean_ic": 0.048, "ic_ir": 1.1, "spread": 0.06},
    }
    kept, stop = _apply_degradation_check(
        ["s1", "s2"], metric_map, lambda w: fake_stats[frozenset(w)],
        cap=0.5, min_weight=0.10, tol=0.10)
    assert kept == ["s1", "s2"]
    assert stop is None


UNIVERSE = [f"T{i}" for i in range(30)]
DATES = pd.date_range("2018-01-31", periods=10, freq="ME").strftime("%Y-%m-%d").tolist()


def _toy_panel(dates: list[str], universe: list[str]) -> ScorePanel:
    """One parent 'p1' with two subs: s1 ranks ascending with ticker order, s2 ranks
    descending -- same construction as tests/test_regime_aware_evidence.py."""
    scores = {}
    for k, d in enumerate(dates):
        base = np.linspace(10, 90, len(universe)) + k
        scores[d] = pd.DataFrame({"s1": base, "s2": base[::-1]}, index=universe)
    return ScorePanel(rebal_dates=list(dates), scores=scores,
                      parent_keys=["p1"], sub_by_parent={"p1": ["s1", "s2"]},
                      universe=list(universe))


def _toy_matrix(dates: list[str], universe: list[str]) -> pd.DataFrame:
    """Prices grow faster for higher-index tickers -> a signal ranking tickers
    ascending (s1) has positive IC; descending (s2) has negative IC."""
    idx = pd.date_range(dates[0], periods=len(dates) + 8, freq="ME").strftime("%Y-%m-%d")
    data = {t: (1.01 + i * 0.002) ** np.arange(len(idx)) for i, t in enumerate(universe)}
    return pd.DataFrame(data, index=idx)


def test_composite_ic_stats_computes_real_composite_metrics():
    panel = _toy_panel(DATES, UNIVERSE)
    matrix = _toy_matrix(DATES, UNIVERSE)
    fwd_by_h = compute_forward_returns(matrix, DATES, {"3M": 3, "6M": 6})
    stats_s1 = _composite_ic_stats(panel, {"s1": 1.0}, fwd_by_h)
    stats_s2 = _composite_ic_stats(panel, {"s2": 1.0}, fwd_by_h)
    assert stats_s1["mean_ic"] > 0
    assert stats_s2["mean_ic"] < 0


def test_composite_ic_stats_empty_weights_returns_nan():
    panel = _toy_panel(DATES, UNIVERSE)
    matrix = _toy_matrix(DATES, UNIVERSE)
    fwd_by_h = compute_forward_returns(matrix, DATES, {"3M": 3, "6M": 6})
    stats = _composite_ic_stats(panel, {}, fwd_by_h)
    assert all(v != v for v in stats.values())   # all NaN


def test_select_parent_subfactors_end_to_end_wiring():
    """s2 is ineligible (negative expected_ic) so it must never be considered, and
    s1 alone should be selected and weighted 100%."""
    panel = _toy_panel(DATES, UNIVERSE)
    matrix = _toy_matrix(DATES, UNIVERSE)
    fwd_by_h = compute_forward_returns(matrix, DATES, {"3M": 3, "6M": 6})
    corr = pd.DataFrame({"s1": [1.0, -1.0], "s2": [-1.0, 1.0]}, index=["s1", "s2"])
    scored = pd.DataFrame([
        {"sub_factor": "s1", "parent": "p1", "expected_ic": 0.05, "production_score": 0.9,
         "eligible": True},
        {"sub_factor": "s2", "parent": "p1", "expected_ic": -0.05, "production_score": 0.1,
         "eligible": False},
    ])
    result = select_parent_subfactors(scored, corr, panel, fwd_by_h)
    assert result["selected"] == ["s1"]
    assert result["weights"] == {"s1": pytest.approx(1.0)}


def test_select_parent_subfactors_rejects_redundant_sub():
    """s1 and s1_dup are perfectly correlated (R²=1.0 >= the 0.60 gate); only the
    higher-ranked one should survive, even though both are eligible."""
    panel = _toy_panel(DATES, UNIVERSE)
    matrix = _toy_matrix(DATES, UNIVERSE)
    fwd_by_h = compute_forward_returns(matrix, DATES, {"3M": 3, "6M": 6})
    corr = pd.DataFrame({"s1": [1.0, 1.0], "s1_dup": [1.0, 1.0]}, index=["s1", "s1_dup"])
    scored = pd.DataFrame([
        {"sub_factor": "s1", "parent": "p1", "expected_ic": 0.05, "production_score": 0.9,
         "eligible": True},
        {"sub_factor": "s1_dup", "parent": "p1", "expected_ic": 0.04, "production_score": 0.7,
         "eligible": True},
    ])
    result = select_parent_subfactors(scored, corr, panel, fwd_by_h)
    assert result["selected"] == ["s1"]
    assert "s1_dup" not in result["weights"]


def test_select_parent_subfactors_no_eligible_candidates():
    scored = pd.DataFrame([
        {"sub_factor": "s1", "parent": "p1", "expected_ic": -0.01, "production_score": 0.9,
         "eligible": False},
    ])
    corr = pd.DataFrame({"s1": [1.0]}, index=["s1"])
    panel = _toy_panel(DATES, UNIVERSE)
    matrix = _toy_matrix(DATES, UNIVERSE)
    fwd_by_h = compute_forward_returns(matrix, DATES, {"3M": 3, "6M": 6})
    result = select_parent_subfactors(scored, corr, panel, fwd_by_h)
    assert result == {"selected": [], "weights": {}, "stop_reason": "no eligible candidate",
                      "decisions": []}
