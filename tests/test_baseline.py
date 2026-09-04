"""Tests for the corrected OOF baseline factor model (research/baseline.py).

A synthetic world where one parent's sub-factor genuinely predicts forward
returns and another is pure noise, so IC-weighting should tilt toward the real
signal. Separately we assert the mechanical guarantees that make the headline
honest: weights normalise, never invert a sign, the equal-weight composite
reproduces the panel's own parent mean, and the walk-forward uses no return
realised on/after the date it scores.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.panel import ScorePanel
from research.baseline import (
    BaselineConfig, WeightSet, compose_one, equal_weights, mean_ic,
    static_composite, train_weights, walk_forward_composite, weights_from_ic,
)


def _make_panel(n: int = 300, n_dates: int = 24, seed: int = 11):
    """Two parents: 'a' has a real sub-predictor + a noise sub; 'b' is all noise."""
    rng = np.random.default_rng(seed)
    tickers = [f"T{i:03d}" for i in range(n)]
    dates = [f"20{22 + (m // 12):02d}-{(m % 12) + 1:02d}-01" for m in range(n_dates)]
    sub_by_parent = {"a": ["a_good", "a_noise"], "b": ["b_n1", "b_n2"]}

    scores: dict[str, pd.DataFrame] = {}
    fwd: dict[str, pd.Series] = {}
    for d in dates:
        u = rng.random(n)
        cols = {
            "a_good": 100.0 * u,
            "a_noise": 100.0 * rng.random(n),
            "b_n1": 100.0 * rng.random(n),
            "b_n2": 100.0 * rng.random(n),
        }
        frame = pd.DataFrame(cols, index=tickers)
        for p, subs in sub_by_parent.items():
            frame[p] = frame[subs].mean(axis=1)
        scores[d] = frame
        fwd[d] = pd.Series(0.10 * (u - 0.5) + rng.normal(0, 0.01, n), index=tickers)

    panel = ScorePanel(rebal_dates=dates, scores=scores,
                       parent_keys=["a", "b"], sub_by_parent=sub_by_parent,
                       universe=tickers)
    return panel, fwd


# ---------------------------------------------------------------------------
# Weight derivation
# ---------------------------------------------------------------------------
def test_weights_normalise_and_floor_negatives():
    ic = {"x": 0.05, "y": -0.20, "z": 0.0}
    w = weights_from_ic(ic, ["x", "y", "z"], shrink=0.0, ic_floor=0.0)
    assert sum(w.values()) == pytest.approx(1.0)
    # negative-IC signal never gets a sign flip / negative weight
    assert all(v >= 0 for v in w.values())
    assert w["x"] == pytest.approx(1.0)        # only positive-IC key carries weight
    assert w["y"] == 0.0


def test_all_nonpositive_ic_falls_back_to_equal():
    ic = {"x": -0.1, "y": -0.2}
    w = weights_from_ic(ic, ["x", "y"], shrink=0.0, ic_floor=0.0)
    assert w == pytest.approx({"x": 0.5, "y": 0.5})


def test_shrink_one_is_equal_weight():
    ic = {"x": 0.30, "y": 0.01}
    w = weights_from_ic(ic, ["x", "y"], shrink=1.0, ic_floor=0.0)
    assert w == pytest.approx({"x": 0.5, "y": 0.5})


def test_train_weights_prefers_real_predictor():
    panel, fwd = _make_panel()
    cfg = BaselineConfig(shrink=0.0, min_names=20)
    ws = train_weights(panel, fwd, panel.rebal_dates, cfg)
    # within parent 'a', the genuine predictor outweighs the noise sub
    assert ws.sub["a"]["a_good"] > ws.sub["a"]["a_noise"]
    # parent 'a' (has signal) outweighs parent 'b' (pure noise)
    assert ws.parent["a"] > ws.parent["b"]


# ---------------------------------------------------------------------------
# Composite construction
# ---------------------------------------------------------------------------
def test_equal_weight_composite_matches_panel_parent_mean():
    panel, _ = _make_panel()
    eq = WeightSet(parent=equal_weights(panel.parent_keys),
                   sub={p: equal_weights(s) for p, s in panel.sub_by_parent.items()})
    d = panel.rebal_dates[0]
    comp = compose_one(panel, d, eq)
    # equal sub + equal parent == mean of the two parent columns
    expected = panel.scores[d][["a", "b"]].mean(axis=1)
    pd.testing.assert_series_equal(comp.sort_index(), expected.sort_index(),
                                   check_names=False)


def test_static_composite_covers_all_dates():
    panel, _ = _make_panel()
    eq = WeightSet(parent=equal_weights(panel.parent_keys),
                   sub={p: equal_weights(s) for p, s in panel.sub_by_parent.items()})
    comp = static_composite(panel, eq)
    assert set(comp) == set(panel.rebal_dates)


# ---------------------------------------------------------------------------
# Out-of-fold guarantee
# ---------------------------------------------------------------------------
def test_walk_forward_warmup_is_equal_weight():
    panel, fwd = _make_panel()
    cfg = BaselineConfig(min_train_periods=12, settle_lag=1, shrink=0.0)
    _, hist = walk_forward_composite(panel, fwd, cfg)
    # early dates lack enough settled history -> warm-up equal weights
    assert hist[panel.rebal_dates[0]].is_warmup
    assert hist[panel.rebal_dates[0]].parent == pytest.approx(
        equal_weights(panel.parent_keys))
    # a late date has trained (non-warm-up) weights
    assert not hist[panel.rebal_dates[-1]].is_warmup


def test_walk_forward_no_lookahead():
    """The weights at date i must not change if any future panel/return is altered.

    We corrupt all data strictly after a pivot date and confirm the pivot's
    learned weights are identical — proving they depend only on settled history.
    """
    panel, fwd = _make_panel()
    cfg = BaselineConfig(min_train_periods=6, settle_lag=1, shrink=0.0)
    _, hist = walk_forward_composite(panel, fwd, cfg)
    pivot_i = 20
    pivot = panel.rebal_dates[pivot_i]
    baseline_w = hist[pivot].parent.copy()

    # Build a future-corrupted copy.
    corrupt_fwd = dict(fwd)
    corrupt_scores = dict(panel.scores)
    for d in panel.rebal_dates[pivot_i:]:
        corrupt_fwd[d] = fwd[d] * -3.0 + 99.0
        corrupt_scores[d] = panel.scores[d] * 0.0
    cpanel = ScorePanel(rebal_dates=panel.rebal_dates, scores=corrupt_scores,
                        parent_keys=panel.parent_keys,
                        sub_by_parent=panel.sub_by_parent, universe=panel.universe)
    _, chist = walk_forward_composite(cpanel, corrupt_fwd, cfg)
    assert chist[pivot].parent == pytest.approx(baseline_w)


def test_mean_ic_recovers_positive_for_predictor():
    panel, fwd = _make_panel()
    ic_good = mean_ic(panel, fwd, "a_good", panel.rebal_dates, min_names=20)
    ic_noise = mean_ic(panel, fwd, "b_n1", panel.rebal_dates, min_names=20)
    assert ic_good > 0.2
    assert ic_good > ic_noise
