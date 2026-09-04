from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.walkforward.regime_aware_parents import (
    HysteresisState, adaptive_weight, apply_change_caps, apply_quarterly_membership,
    blend_parent_weights, parent_utility_table, update_streaks,
)


def test_parent_utility_table_ranks_across_all_parents():
    evidence = pd.DataFrame([
        {"sub_factor": "A", "parent": "A", "expected_ic": 0.05, "long_run_mean_ic": 0.05,
         "long_run_std_ic": 0.02, "long_run_spread_ann": 0.08, "long_run_hit_rate": 0.7,
         "pct_positive_years": 0.8, "persistence_ir": 1.0, "spread_consistency": 0.9,
         "coverage": 0.95, "n_months": 60},
        {"sub_factor": "B", "parent": "B", "expected_ic": 0.01, "long_run_mean_ic": 0.01,
         "long_run_std_ic": 0.03, "long_run_spread_ann": 0.02, "long_run_hit_rate": 0.5,
         "pct_positive_years": 0.5, "persistence_ir": 0.2, "spread_consistency": 0.5,
         "coverage": 0.80, "n_months": 40},
    ])
    out = parent_utility_table(evidence)
    a = out[out.sub_factor == "A"].iloc[0]
    b = out[out.sub_factor == "B"].iloc[0]
    assert a["utility_score"] > b["utility_score"]
    assert a["utility_score"] == pytest.approx(1.0, abs=1e-9)   # dominates every metric


def test_adaptive_weight_excludes_negative_ic_parents():
    # 4 eligible parents (p2 excluded) so a 25% cap is actually satisfiable at sum=1
    # (with only 2-3 eligible parents, 0.25 * n < 1.0 makes cap and sum=1 mutually
    # infeasible, and _water_fill will breach the cap to preserve sum=1 -- see
    # research/walkforward/vix_overlay.py::_water_fill).
    utility = pd.Series({"p1": 0.9, "p2": 0.6, "p3": 0.5, "p4": 0.4, "p5": 0.3})
    expected_ic = pd.Series({"p1": 0.02, "p2": -0.01, "p3": 0.03, "p4": 0.01, "p5": 0.02})
    w = adaptive_weight(utility, expected_ic, cap=0.25)
    assert "p2" not in w
    assert abs(sum(w.values()) - 1.0) < 1e-9
    assert max(w.values()) <= 0.25 + 1e-9


def test_blend_parent_weights_zeroes_negative_ic_and_renormalizes():
    # 4 nonzero-after-zeroing parents (p2 zeroed) so cap=0.25 stays feasible at sum=1
    # -- see the feasibility note on test_adaptive_weight_excludes_negative_ic_parents.
    base = {"p1": 0.30, "p2": 0.25, "p3": 0.20, "p4": 0.15, "p5": 0.10}
    adaptive = {"p1": 0.30, "p3": 0.30, "p4": 0.25, "p5": 0.15}    # p2: no adaptive weight
    expected_ic = {"p1": 0.02, "p2": -0.01, "p3": 0.03, "p4": 0.01, "p5": 0.015}
    out = blend_parent_weights(base, adaptive, expected_ic, base_weight_frac=0.70, cap=0.25)
    assert out.get("p2", 0.0) == pytest.approx(0.0, abs=1e-9)
    assert abs(sum(out.values()) - 1.0) < 1e-9
    assert max(out.values()) <= 0.25 + 1e-9


def test_apply_change_caps_limits_monthly_move():
    target = {"p1": 0.30, "p2": 0.70}          # a big jump from prior weights
    prev_month = {"p1": 0.10, "p2": 0.90}
    quarter_start = {"p1": 0.10, "p2": 0.90}
    out = apply_change_caps(target, prev_month, quarter_start,
                            monthly_cap=0.02, quarterly_cap=0.05)
    assert out["p1"] <= 0.10 + 0.02 + 1e-9
    assert abs(sum(out.values()) - 1.0) < 1e-9


def test_apply_change_caps_limits_quarterly_move_by_third_month():
    """Three consecutive monthly calls, all pushing the same direction: by the third
    month the quarterly +/-5pp band is what's binding, not the monthly +/-2pp band."""
    quarter_start = {"p1": 0.10, "p2": 0.90}
    prev = dict(quarter_start)
    target = {"p1": 0.30, "p2": 0.70}
    for _ in range(3):
        prev = apply_change_caps(target, prev, quarter_start,
                                 monthly_cap=0.02, quarterly_cap=0.05)
    assert prev["p1"] == pytest.approx(quarter_start["p1"] + 0.05, abs=1e-9)


def test_apply_change_caps_renormalization_does_not_breach_band():
    # With 3 asymmetric parents, clipping independently lands at sum=1.02 (not 1.0):
    # p1/p2 clip to 0.12 each, p3 clips to its lo=0.78 floor. A naive uniform
    # renormalise-by-1.02 would push p3 to ~0.7647, breaching its own 0.78 floor
    # (a monthly-cap violation). The excess must instead come only from p1/p2, which
    # still have room above their own lo bound.
    target = {"p1": 0.30, "p2": 0.30, "p3": 0.40}
    prev_month = {"p1": 0.10, "p2": 0.10, "p3": 0.80}
    quarter_start = {"p1": 0.10, "p2": 0.10, "p3": 0.80}
    out = apply_change_caps(target, prev_month, quarter_start,
                            monthly_cap=0.02, quarterly_cap=0.05)
    assert out["p3"] >= 0.80 - 0.02 - 1e-9
    assert abs(sum(out.values()) - 1.0) < 1e-9


def test_update_streaks_tracks_above_and_below():
    state = HysteresisState()
    update_streaks(state, would_select=["s1"], eligible={"s1", "s2"}, immediate_exit=set())
    assert state.above_streak["s1"] == 1
    assert state.below_streak.get("s2", 0) == 1    # eligible but not picked -> below streak
    update_streaks(state, would_select=["s1"], eligible={"s1", "s2"}, immediate_exit=set())
    assert state.above_streak["s1"] == 2


def test_apply_quarterly_membership_entry_requires_two_months():
    state = HysteresisState()
    update_streaks(state, would_select=["s1"], eligible={"s1"}, immediate_exit=set())
    apply_quarterly_membership(state, would_select=["s1"])
    assert "s1" not in state.members            # only 1 month so far -> not enough
    update_streaks(state, would_select=["s1"], eligible={"s1"}, immediate_exit=set())
    apply_quarterly_membership(state, would_select=["s1"])
    assert "s1" in state.members                 # 2 consecutive months -> entry


def test_apply_quarterly_membership_exit_requires_three_months():
    state = HysteresisState(members={"s1"})
    for _ in range(2):
        update_streaks(state, would_select=[], eligible=set(), immediate_exit=set())
    apply_quarterly_membership(state, would_select=[])
    assert "s1" in state.members                 # only 2 months below -> not enough yet
    update_streaks(state, would_select=[], eligible=set(), immediate_exit=set())
    apply_quarterly_membership(state, would_select=[])
    assert "s1" not in state.members              # 3rd consecutive month -> exit


def test_update_streaks_immediate_exit_bypasses_hysteresis():
    state = HysteresisState(members={"s1"}, above_streak={"s1": 5})
    update_streaks(state, would_select=["s1"], eligible={"s1"}, immediate_exit={"s1"})
    assert "s1" not in state.members              # removed immediately despite a strong streak


def test_apply_quarterly_membership_respects_max_subs():
    state = HysteresisState(members={"s1", "s2", "s3"})
    for _ in range(2):
        update_streaks(state, would_select=["s4"], eligible={"s1", "s2", "s3", "s4"},
                       immediate_exit=set())
    apply_quarterly_membership(state, would_select=["s4"], max_subs=3)
    assert "s4" not in state.members               # no free slot -- 3 members block entry
