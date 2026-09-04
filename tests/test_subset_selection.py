"""Tests for the sub-factor subset-selection engine (research.subset_selection).

A synthetic two-parent world is built where the answer is known by construction:
`mom_good` and `val_unique` each *equally* predict forward returns but on offset
strength cycles (so combining them lowers IC volatility), `mom_dup` is a near-duplicate
of `mom_good` (redundant), and `val_noise` is pure noise. The engine should therefore:
recover positive IC for the two real predictors, flag the duplicate pair as redundant,
and — crucially — *prefer breadth*: its information-ratio objective must rank
{mom_good, val_unique} above {mom_good, mom_dup}, and forward selection must pick the
two uncorrelated predictors while rejecting the duplicate and the noise. This pins the
Fundamental-Law behaviour (breadth beats a single strong factor) without any DB access.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.panel import ScorePanel
from research import subset_selection as ss

SUBS_BY_PARENT = {"momentum": ["mom_good", "mom_dup"],
                  "value": ["val_unique", "val_noise"]}
ALL_SUBS = ["mom_good", "mom_dup", "val_unique", "val_noise"]


def _make_panel(n: int = 300, n_dates: int = 24, seed: int = 11):
    """Two parents; mom_good & val_unique predict on offset cycles, mom_dup duplicates
    mom_good, val_noise is noise. Returns (panel, fwd_by_h) with all 4 horizons."""
    rng = np.random.default_rng(seed)
    tickers = [f"T{i:03d}" for i in range(n)]
    dates = [f"20{22 + (m // 12):02d}-{(m % 12) + 1:02d}-01" for m in range(n_dates)]
    scores: dict[str, pd.DataFrame] = {}
    fwd: dict[str, pd.Series] = {}
    for i, d in enumerate(dates):
        u_good, u_unique = rng.random(n), rng.random(n)
        s_good = 100.0 * u_good
        s_dup = np.clip(s_good + rng.normal(0, 1.5, n), 0, 100)     # near-duplicate
        s_unique = 100.0 * u_unique
        s_noise = 100.0 * rng.random(n)
        scores[d] = pd.DataFrame(
            {"mom_good": s_good, "mom_dup": s_dup,
             "val_unique": s_unique, "val_noise": s_noise}, index=tickers)
        # Offset strength cycles: momentum leads when value lags and vice versa, so a
        # two-factor composite has a *steadier* IC than either factor alone.
        a = 0.06 * (1.0 + np.sin(i))
        b = 0.06 * (1.0 + np.cos(i))
        ret = a * (u_good - 0.5) + b * (u_unique - 0.5) + rng.normal(0, 0.01, n)
        fwd[d] = pd.Series(ret, index=tickers)
    panel = ScorePanel(rebal_dates=dates, scores=scores,
                       parent_keys=["momentum", "value"],
                       sub_by_parent={k: list(v) for k, v in SUBS_BY_PARENT.items()},
                       universe=tickers)
    fwd_by_h = {h: fwd for h in ss.DEFAULT_HORIZONS}
    return panel, fwd_by_h


@pytest.fixture(scope="module")
def world():
    return _make_panel()


# --- inventory / per-sub performance ---------------------------------------
def test_inventory_one_row_per_sub(world):
    panel, _ = world
    inv = ss.inventory(panel)
    assert set(inv["sub_factor"]) == set(ALL_SUBS)
    assert (inv["coverage"] > 0.5).all()          # all subs disperse the universe
    assert (inv["n_active_dates"] == len(panel.rebal_dates)).all()


def test_performance_predictor_beats_noise(world):
    panel, fwd = world
    perf = ss.subfactor_performance(panel, fwd).set_index("sub_factor")
    assert perf.loc["mom_good", "ic_spearman"] > 0.05
    assert perf.loc["val_unique", "ic_spearman"] > 0.05
    assert abs(perf.loc["val_noise", "ic_spearman"]) < 0.03
    # real predictors carry a positive Q5-Q1 spread
    assert perf.loc["mom_good", "spread_q5_q1"] > 0


# --- redundancy ------------------------------------------------------------
def test_correlation_flags_duplicate(world):
    panel, _ = world
    corr = ss.correlation_matrix(panel)
    assert corr.loc["mom_good", "mom_dup"] > ss.CORR_HIGH
    assert abs(corr.loc["mom_good", "val_unique"]) < 0.3
    pairs = ss.redundant_pairs(corr, ss.CORR_HIGH)
    flagged = {frozenset((r.a, r.b)) for r in pairs.itertuples()}
    assert frozenset(("mom_good", "mom_dup")) in flagged


# --- the core claim: IR rewards breadth over a redundant duplicate ---------
def test_ir_objective_prefers_breadth(world):
    panel, fwd = world
    folds = ss.time_folds(list(fwd["1M"]), 4)
    ir_breadth = ss.evaluate_subset(panel, ["mom_good", "val_unique"], fwd, folds)["oos_ic"]
    ir_dup = ss.evaluate_subset(panel, ["mom_good", "mom_dup"], fwd, folds)["oos_ic"]
    ir_solo = ss.evaluate_subset(panel, ["mom_good"], fwd, folds)["oos_ic"]
    # Adding an uncorrelated equal predictor raises the IR; adding a duplicate does not.
    assert ir_breadth > ir_solo
    assert ir_breadth > ir_dup


# --- forward selection + backward pruning ----------------------------------
def test_forward_select_picks_uncorrelated_predictors(world):
    panel, fwd = world
    folds = ss.time_folds(list(fwd["1M"]), 4)
    selected, steps = ss.forward_select(panel, ALL_SUBS, fwd, folds)
    assert "mom_good" in selected and "val_unique" in selected
    assert "val_noise" not in selected            # noise never earns a place
    assert "mom_dup" not in selected              # duplicate adds no breadth
    assert not steps.empty and steps["oos_ic"].is_monotonic_increasing


def test_backward_prune_keeps_additive_pair(world):
    panel, fwd = world
    folds = ss.time_folds(list(fwd["1M"]), 4)
    kept, _ = ss.backward_prune(panel, ["mom_good", "val_unique"], fwd, folds)
    assert set(kept) == {"mom_good", "val_unique"}   # neither is dropped


def test_backward_prune_drops_redundant(world):
    panel, fwd = world
    folds = ss.time_folds(list(fwd["1M"]), 4)
    kept, steps = ss.backward_prune(panel, ["mom_good", "mom_dup", "val_unique"], fwd, folds)
    assert "val_unique" in kept
    assert len(kept) < 3                             # the redundant duplicate is pruned


# --- selection_score + momentum dependence ---------------------------------
def test_selection_score_ranks_predictors_high(world):
    panel, fwd = world
    inv = ss.inventory(panel)
    perf = ss.subfactor_performance(panel, fwd)
    corr = ss.correlation_matrix(panel)
    scored = ss.selection_score(perf, inv, ss.max_abs_corr(corr))
    top2 = set(scored.head(2)["sub_factor"])
    assert "val_noise" not in top2
    assert "val_unique" in top2 or "mom_good" in top2


def test_momentum_dependence_reports_share(world):
    panel, fwd = world
    selected = ["mom_good", "val_unique"]
    comp = ss.ew_composite(panel, selected, list(fwd["1M"]))
    md = ss.momentum_dependence(panel, selected, comp, fwd["1M"])
    assert md["n_selected"] == 2 and md["n_momentum"] == 1
    assert md["momentum_share"] == pytest.approx(0.5)
    # dropping the momentum leg costs IC (the non-momentum sub alone is weaker)
    assert md["ic_full"] > md["ic_without_momentum"]


# --- weighting comparison integrates with the baseline engine --------------
def test_weighting_composites_four_methods(world):
    panel, fwd = world
    comps = ss.weighting_composites(panel, ["mom_good", "val_unique"], fwd["1M"])
    assert set(comps) == {"equal_flat", "equal_parent_sub", "ic_weighted", "shrunk_ic"}
    for series_by_date in comps.values():
        assert series_by_date                       # non-empty
        any_series = next(iter(series_by_date.values()))
        assert isinstance(any_series, pd.Series)
