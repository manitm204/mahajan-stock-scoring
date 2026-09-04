"""Regression: non-members must never enter research books as neutral ghosts.

Bug (found 2026-07-15 via SBNY appearing in a 2024 book): build_parent_panel
reindexed each date's frame to the all-time union universe, creating all-NaN
rows for names not in that date's PIT universe; composite_from_parents then
filled every missing parent with NEUTRAL, so ghosts scored dead-neutral,
passed the top-75% screen, and could never be vetoed (veto_set skips NaN).
"""
import numpy as np
import pandas as pd

from research.panel import ScorePanel
from research.walkforward.compose import build_parent_panel
from vixtilt.backtest import composite_from_parents


def _panel() -> ScorePanel:
    # date d1 has members A,B,C; the union universe also holds ghost X
    frame = pd.DataFrame(
        {"quality_sub": [80.0, 50.0, 20.0], "value_sub": [70.0, 50.0, 30.0]},
        index=["A", "B", "C"])
    return ScorePanel(
        rebal_dates=["d1"], scores={"d1": frame},
        parent_keys=["quality", "value"],
        sub_by_parent={"quality": ["quality_sub"], "value": ["value_sub"]},
        universe=["A", "B", "C", "X"])


def test_build_parent_panel_keeps_pit_index():
    pp = build_parent_panel(_panel(), {"quality": {"quality_sub": 1.0},
                                       "value": {"value_sub": 1.0}})
    assert list(pp.scores["d1"].index) == ["A", "B", "C"]
    assert "X" not in pp.scores["d1"].index


def test_composite_masks_all_nan_rows():
    # even if a ghost row sneaks into the frame, its composite must be NaN
    frame = pd.DataFrame(
        {"quality": [80.0, 50.0, 20.0, np.nan],
         "value": [70.0, 50.0, 30.0, np.nan]},
        index=["A", "B", "C", "X"])
    sectors = pd.Series("Tech", index=frame.index)
    comp = composite_from_parents(
        frame, {"quality": 0.5, "value": 0.5}, sectors, min_obs=2)
    assert np.isnan(comp["X"])
    assert comp.drop("X").notna().all()
