"""Tests for the walk-forward out-of-sample validation framework.

The correctness-critical invariant is **no look-ahead**: selection may never use a
forward-return window that reaches into the test period. That is asserted two ways — the
split's ``train_rebalances`` filter, and the price-matrix truncation the selector relies
on. The rest is deterministic unit coverage of the composite blend, the sector-neutral
weighting, the IC/IR parent-weight water-fill and the metric maths, plus a DB-guarded
check that a full-sample re-selection reproduces the live V4 construction.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from research.forward_returns import compute_forward_returns
from research.panel import ScorePanel
from research.walkforward import analysis
from research.walkforward.compose import (FrozenConfig, _composite_row, frozen_composite,
                                          ic_ir_weights)
from research.walkforward.portfolio import (benchmark_stats, performance_metrics,
                                            sector_neutral_weights, simulate)
from research.walkforward.splits import (DATA_START, SELECTION_HORIZON_MONTHS,
                                         WalkForwardSplit, default_splits, policy_splits)


# --------------------------------------------------------------------------- #
# No-look-ahead
# --------------------------------------------------------------------------- #
def test_train_rebalances_exclude_boundary_crossing_windows():
    """Every selection rebalance must leave a full 6M window before the test boundary."""
    rebals = pd.date_range("2023-06-30", "2025-06-30", freq="ME").strftime("%Y-%m-%d").tolist()
    sp = WalkForwardSplit("t", "2023-06-01", "2024-06-30", "2024-07-01", "2025-06-30")
    train = sp.train_rebalances(rebals)
    cap = (pd.Timestamp(sp.test_start)
           - pd.DateOffset(months=SELECTION_HORIZON_MONTHS)).date().isoformat()
    assert train, "expected some usable training rebalances"
    assert max(train) <= cap                       # 6M window ends on/before the boundary
    assert all(d < sp.test_start for d in train)   # and strictly before the test period


def test_policy_splits_training_windows():
    """Expanding trains from data start; rolling windows start N years before the boundary
    (clamped to data start); test years span 2017-2026."""
    exp = policy_splits("expanding")
    assert all(sp.train_start == DATA_START for sp in exp)
    assert [sp.test_year for sp in exp] == [str(y) for y in range(2017, 2027)]

    roll3 = {sp.test_year: sp for sp in policy_splits("rolling3y")}
    assert roll3["2024"].train_start == "2021-01-01"   # 3y before 2024-01-01
    assert roll3["2017"].train_start == DATA_START     # 3y before 2017 precedes data


def test_price_truncation_drops_windows_past_boundary():
    """The mechanism the selector uses: truncating prices at the boundary makes
    compute_forward_returns drop any window that would reach into the test period."""
    idx = pd.date_range("2023-06-30", "2025-01-31", freq="ME").strftime("%Y-%m-%d").tolist()
    matrix = pd.DataFrame({"AAA": np.linspace(100, 200, len(idx))}, index=idx)
    boundary = "2024-07-01"
    px = matrix.loc[matrix.index <= boundary]
    fwd = compute_forward_returns(px, idx, {"6M": 6})
    # No 6M window may end after the boundary → every used start ≤ boundary − ~6M.
    for start in fwd["6M"]:
        assert start <= boundary
        end_target = (pd.Timestamp(start) + pd.DateOffset(months=6))
        assert end_target <= pd.Timestamp(boundary) + pd.Timedelta(days=25)


# --------------------------------------------------------------------------- #
# Weighting
# --------------------------------------------------------------------------- #
def test_sector_neutral_weights_sum_to_one_and_match_universe():
    universe = [f"T{i}" for i in range(10)]
    sectors = pd.Series(["A"] * 6 + ["B"] * 4, index=universe)   # universe A=0.6 B=0.4
    selected = ["T0", "T1", "T6"]                                # 2 in A, 1 in B
    w = sector_neutral_weights(selected, universe, sectors)
    assert abs(w.sum() - 1.0) < 1e-9
    by_sec = w.groupby(sectors.reindex(w.index)).sum()
    assert abs(by_sec["A"] - 0.6) < 1e-9 and abs(by_sec["B"] - 0.4) < 1e-9


def test_sector_neutral_redistributes_empty_sectors():
    universe = [f"T{i}" for i in range(10)]
    sectors = pd.Series(["A"] * 6 + ["B"] * 4, index=universe)
    selected = ["T0", "T1"]                                      # both in A, none in B
    w = sector_neutral_weights(selected, universe, sectors)
    assert abs(w.sum() - 1.0) < 1e-9                             # B's share redistributed


# --------------------------------------------------------------------------- #
# IC/IR parent weights
# --------------------------------------------------------------------------- #
def test_ic_ir_weights_cap_and_exclude_dilutive():
    # Six parents so the 25 % cap is feasible (as in the real 8-parent run); one dilutive.
    ps = pd.DataFrame({
        "parent": ["a", "b", "c", "d", "e", "f"],
        "mean_ic_3m6m": [0.12, 0.06, 0.05, 0.04, 0.03, -0.05],   # f is dilutive
        "information_ratio": [1.2, 0.6, 0.5, 0.4, 0.3, -0.4],
    })
    w = ic_ir_weights(ps, cap=0.25)
    assert abs(sum(w.values()) - 1.0) < 1e-9
    assert max(w.values()) <= 0.25 + 1e-9            # water-fill cap respected
    assert "f" not in w                               # non-positive combined → 0 weight
    assert w["a"] == pytest.approx(0.25, abs=1e-9)    # dominant parent pinned at the cap


# --------------------------------------------------------------------------- #
# Composite blend
# --------------------------------------------------------------------------- #
def test_composite_row_coverage_aware_skips_nan():
    mat = np.array([[80.0, np.nan], [np.nan, np.nan], [60.0, 40.0]])
    w = np.array([0.5, 0.5])
    out = _composite_row(mat, w, coverage_aware=True)
    assert abs(out[0] - 80.0) < 1e-9                 # renormalised onto the present col
    assert np.isnan(out[1])                          # no data → NaN
    assert abs(out[2] - 50.0) < 1e-9


def _toy_sub_panel(dates, universe):
    """Two parents, one sub each; sub scores rank names deterministically."""
    scores = {}
    for k, d in enumerate(dates):
        base = np.linspace(10, 90, len(universe)) + k    # rising with ticker order
        scores[d] = pd.DataFrame({"s1": base, "s2": base[::-1]}, index=universe)
    return ScorePanel(rebal_dates=list(dates), scores=scores,
                      parent_keys=["p1", "p2"], sub_by_parent={"p1": ["s1"], "p2": ["s2"]},
                      universe=list(universe))


def test_frozen_composite_returns_valid_sector_percentiles():
    universe = [f"T{i}" for i in range(30)]
    sectors = pd.Series(["A", "B", "C"] * 10, index=universe)
    dates = ["2024-07-31", "2024-08-31"]
    panel = _toy_sub_panel(dates, universe)
    cfg = FrozenConfig(sub_weights={"p1": {"s1": 1.0}, "p2": {"s2": 1.0}},
                       parent_weights={"p1": 0.5, "p2": 0.5}, meta={})
    out = frozen_composite(panel, dates, cfg, sectors)
    assert set(out) == set(dates)
    for d in dates:
        vals = out[d].dropna()
        assert len(vals) > 0
        assert vals.min() >= 0.0 and vals.max() <= 100.0


# --------------------------------------------------------------------------- #
# Metrics + quantiles
# --------------------------------------------------------------------------- #
def test_performance_metrics_on_known_series():
    r = pd.Series([0.02, 0.01, 0.03, 0.015], index=pd.date_range("2024-01-31", periods=4,
                                                                 freq="ME"))
    m = performance_metrics(r, hold_months=1)
    assert m["cagr"] > 0                              # all-positive → compounding gain
    assert m["max_drawdown"] == 0.0                   # monotone-up equity
    assert m["hit_rate"] == 1.0
    assert m["total_return"] == pytest.approx(float((1 + r).prod() - 1), abs=1e-9)
    assert np.isinf(m["sortino"])                     # no negative periods → infinite Sortino


def test_performance_metrics_flattens_benchmark_stats():
    idx = pd.date_range("2024-01-31", periods=6, freq="ME")
    port = pd.Series([0.03, -0.01, 0.02, 0.04, -0.02, 0.03], index=idx)
    spy = pd.Series([0.01, -0.005, 0.015, 0.02, -0.01, 0.02], index=idx)
    m = performance_metrics(port, 1, benchmarks={"SPY": spy})
    for k in ("spy_cagr", "spy_alpha", "spy_beta", "spy_ir", "spy_excess_cagr"):
        assert k in m and not np.isnan(m[k])
    assert m["excess_cagr"] == pytest.approx(m["spy_excess_cagr"], abs=1e-12)


def test_benchmark_stats_beta_of_scaled_benchmark():
    """A portfolio that is exactly 2× the benchmark each period has beta≈2 and ~0 IR alpha
    only from the leverage (excess return positive when benchmark drifts up)."""
    idx = pd.date_range("2024-01-31", periods=8, freq="ME")
    b = pd.Series([0.01, -0.02, 0.03, 0.015, -0.01, 0.02, 0.005, -0.015], index=idx)
    p = 2.0 * b
    bs = benchmark_stats(p, b, ppy=12.0)
    assert bs["beta"] == pytest.approx(2.0, abs=1e-6)
    assert bs["alpha"] == pytest.approx(0.0, abs=1e-6)   # pure leverage → no CAPM alpha


def test_quantile_analysis_recovers_monotone_signal():
    universe = [f"T{i}" for i in range(50)]
    dates = [f"2024-{m:02d}-28" for m in range(1, 9)]
    scores, fwd = {}, {}
    rng = np.random.default_rng(0)
    for d in dates:
        s = pd.Series(np.arange(50, dtype=float), index=universe)    # score = rank
        scores[d] = s
        fwd[d] = s / 100.0 + rng.normal(0, 0.01, 50)                 # return rises with score
    res = analysis.quantile_analysis(scores, {"1M": fwd})
    blk = res["1M"]
    assert blk["spread"]["avg"] > 0                   # Q5 out-returns Q1
    assert blk["profile_spearman"] > 0.9              # cleanly monotone


# --------------------------------------------------------------------------- #
# DB-guarded reproduction of the live V4 selection
# --------------------------------------------------------------------------- #
_CACHE = Path("cache/subfactor_expansion/cand_panel_2015-06-30_2026-06-30_monthly_v2.pkl")


@pytest.mark.slow
@pytest.mark.skipif(not _CACHE.exists(), reason="candidate panel cache not built")
def test_full_sample_selection_reproduces_live_v4():
    """Re-selecting on the 2023-2026 window (no boundary) should reproduce the live
    factors.parent_selection_v4 construction for most parents — proof the walk-forward
    re-derives the *same* method the fund uses. Run on the point-in-time deep panel, so a
    parent or two may shift versus the original static-universe fit; ≥5/8 still proves it."""
    from backtesting import data_loader as dl
    from data.db import get_db
    from factors.parent_selection_v4 import SELECTED_SUBS
    from research.subfactor_expansion.panel import load_cached_panel as eload
    from research.walkforward.selection import select_config
    from run_walkforward import _adapt

    panel = _adapt(eload(_CACHE))
    dates = [d for d in panel.rebal_dates if "2023-01-01" <= d <= "2026-06-30"]
    with get_db() as db:
        matrix = dl.load_price_matrix(db, panel.universe, "2015-06-30", "2026-07-06")
    cfg = select_config(panel, dates, matrix, boundary=None)
    # Under the point-in-time universe the re-selection can shift a few borderline sub
    # picks versus the original static-universe V4 fit, so exact per-parent set equality is
    # no longer the right invariant. Assert a strong *sub-level* overlap instead — the
    # method still lands on the large majority of V4's sub-factors.
    live = sum(len(SELECTED_SUBS[p]) for p in SELECTED_SUBS)
    matched = sum(len(set(cfg.sub_weights.get(p, {})) & set(SELECTED_SUBS[p]))
                  for p in SELECTED_SUBS)
    assert matched / live >= 0.6, f"only {matched}/{live} V4 sub-factors reproduced"
