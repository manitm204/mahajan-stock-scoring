"""Tests for the Factor Research & Validation Framework.

A synthetic world is built where, within one parent, some sub-factors genuinely
predict forward returns, one is a near-duplicate of another, and one is pure
noise. The framework should then: rank returns monotonically across quintiles for
a clean signal, recover positive IC, credit unique incremental information only to
the independent predictor, flag the duplicate as redundant, and classify rows the
way the documented cascade says.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.panel import ScorePanel
from research.forward_returns import compute_forward_returns
from research.quintiles import quintile_profile, _monotonicity, aggregate_quintiles
from research.ic import period_ic, ic_timeseries, summarize_ic
from research.incremental import incremental_metrics
from research.redundancy import max_sibling_corr
from research.classify import (
    classify_subfactors, redundancy_watch, Thresholds, KEEP, MERGE, REMOVE,
    INSUFFICIENT,
)


def _make_panel(n: int = 300, n_dates: int = 15, seed: int = 7):
    """Panel with one parent 'p' whose subs have known roles, plus matching fwd."""
    rng = np.random.default_rng(seed)
    tickers = [f"T{i:03d}" for i in range(n)]
    dates = [f"20{24 + (m // 12):02d}-{(m % 12) + 1:02d}-01" for m in range(n_dates)]
    subs = ["p_good", "p_dup", "p_unique", "p_noise"]

    scores: dict[str, pd.DataFrame] = {}
    fwd: dict[str, pd.Series] = {}
    for d in dates:
        u_good = rng.random(n)
        u_unique = rng.random(n)
        s_good = 100.0 * u_good
        s_dup = np.clip(s_good + rng.normal(0, 1.5, n), 0, 100)   # near-duplicate
        s_unique = 100.0 * u_unique
        s_noise = 100.0 * rng.random(n)
        frame = pd.DataFrame(
            {"p_good": s_good, "p_dup": s_dup, "p_unique": s_unique, "p_noise": s_noise},
            index=tickers,
        )
        frame["p"] = frame[subs].mean(axis=1)
        scores[d] = frame
        ret = 0.08 * (u_good - 0.5) + 0.08 * (u_unique - 0.5) + rng.normal(0, 0.01, n)
        fwd[d] = pd.Series(ret, index=tickers)

    panel = ScorePanel(
        rebal_dates=dates, scores=scores, parent_keys=["p"],
        sub_by_parent={"p": subs}, universe=tickers,
    )
    return panel, fwd


# ---------------------------------------------------------------------------
# Quintiles
# ---------------------------------------------------------------------------
def test_quintile_profile_monotonic():
    scores = pd.Series(np.arange(100, dtype=float))
    fwd = scores / 100.0    # forward return rises perfectly with the score
    prof = quintile_profile(scores, fwd, min_names=20)
    assert prof is not None
    assert np.all(np.diff(prof) > 0)          # Q1 < Q2 < ... < Q5
    assert prof[-1] - prof[0] > 0             # positive Q5-Q1 spread
    assert _monotonicity(prof) == pytest.approx(1.0)


def test_quintile_profile_too_small():
    scores = pd.Series([1.0, 2.0, 3.0])
    fwd = pd.Series([0.1, 0.2, 0.3])
    assert quintile_profile(scores, fwd, min_names=25) is None


def test_aggregate_quintiles_spread_positive_for_predictor():
    panel, fwd = _make_panel()
    table = aggregate_quintiles(panel, fwd, ["p_good", "p_noise"])
    good = table[table["signal"] == "p_good"].iloc[0]
    noise = table[table["signal"] == "p_noise"].iloc[0]
    assert good["spread_q5_q1"] > noise["spread_q5_q1"]
    assert good["spread_q5_q1"] > 0


# ---------------------------------------------------------------------------
# IC
# ---------------------------------------------------------------------------
def test_period_ic_sign():
    scores = pd.Series(np.arange(50, dtype=float))
    fwd = scores * 0.001 + 0.0
    assert period_ic(scores, fwd, min_names=20) > 0.9
    assert period_ic(scores, -fwd, min_names=20) < -0.9


def test_summarize_ic_predictor_beats_noise():
    panel, fwd = _make_panel()
    ts = ic_timeseries(panel, fwd, ["p_good", "p_noise"], horizon="1M")
    summ = summarize_ic(ts, signals=["p_good", "p_noise"]).set_index("signal")
    assert summ.loc["p_good", "mean_ic"] > 0
    assert summ.loc["p_good", "mean_ic"] > summ.loc["p_noise", "mean_ic"]
    assert 0.0 <= summ.loc["p_good", "hit_rate"] <= 1.0


# ---------------------------------------------------------------------------
# Incremental contribution
# ---------------------------------------------------------------------------
def test_incremental_credits_unique_not_noise():
    panel, fwd = _make_panel()
    inc = incremental_metrics(panel, fwd).set_index("sub_factor")
    # The independent predictor keeps a real edge after orthogonalizing on siblings.
    assert inc.loc["p_unique", "incremental_ic"] > 0.02
    # Noise has ~no unique predictive content.
    assert inc.loc["p_unique", "incremental_ic"] > inc.loc["p_noise", "incremental_ic"]
    # Standalone IC of the noise sub is near zero.
    assert abs(inc.loc["p_noise", "standalone_ic"]) < 0.05


def test_incremental_duplicate_loses_edge():
    panel, fwd = _make_panel()
    inc = incremental_metrics(panel, fwd).set_index("sub_factor")
    # p_good is predictive standalone but its near-duplicate sibling absorbs most
    # of that edge, so its *incremental* IC is far below its standalone IC.
    assert inc.loc["p_good", "standalone_ic"] > 0.05
    assert inc.loc["p_good", "incremental_ic"] < inc.loc["p_good", "standalone_ic"]


# ---------------------------------------------------------------------------
# Redundancy
# ---------------------------------------------------------------------------
def test_redundancy_flags_duplicate():
    panel, _ = _make_panel()
    red = max_sibling_corr(panel).set_index("sub_factor")
    assert red.loc["p_dup", "max_abs_sibling_corr"] > 0.7
    assert red.loc["p_dup", "closest_sibling"] == "p_good"
    assert red.loc["p_noise", "max_abs_sibling_corr"] < 0.5


# ---------------------------------------------------------------------------
# Classification cascade
# ---------------------------------------------------------------------------
def _row(**kw):
    base = dict(sub_factor="x", parent="p", n_periods=30, coverage=0.6,
                mean_ic=0.0, information_ratio=0.1, monotonicity=0.5,
                max_abs_sibling_corr=0.3, closest_sibling="y",
                incremental_ic=0.01, delta_parent_ic=0.005)
    base.update(kw)
    return base


def test_classify_insufficient_on_thin_history():
    df = pd.DataFrame([_row(sub_factor="thin", n_periods=4)])
    out = classify_subfactors(df).set_index("sub_factor")
    assert out.loc["thin", "verdict"] == INSUFFICIENT


def test_classify_remove_on_no_edge():
    df = pd.DataFrame([_row(sub_factor="dead", information_ratio=-0.2,
                            incremental_ic=0.0, delta_parent_ic=-0.001)])
    out = classify_subfactors(df).set_index("sub_factor")
    assert out.loc["dead", "verdict"] == REMOVE


def test_classify_merge_on_redundant():
    df = pd.DataFrame([_row(sub_factor="twin", information_ratio=0.2,
                            max_abs_sibling_corr=0.88, incremental_ic=0.0,
                            delta_parent_ic=0.0)])
    out = classify_subfactors(df).set_index("sub_factor")
    assert out.loc["twin", "verdict"] == MERGE


def test_classify_keep_on_additive():
    df = pd.DataFrame([_row(sub_factor="solid", information_ratio=0.2,
                            max_abs_sibling_corr=0.3, incremental_ic=0.02,
                            delta_parent_ic=0.01)])
    out = classify_subfactors(df).set_index("sub_factor")
    assert out.loc["solid", "verdict"] == KEEP


def test_redundancy_watch_flags_keep_pair():
    # Two mutually-correlated Keep siblings -> the weaker is suggested to fold in.
    df = pd.DataFrame([
        _row(sub_factor="a", incremental_ic=0.02, max_abs_sibling_corr=0.92,
             closest_sibling="b"),
        _row(sub_factor="b", incremental_ic=0.02, max_abs_sibling_corr=0.92,
             closest_sibling="a"),
    ])
    classified = classify_subfactors(df)
    assert (classified["verdict"] == KEEP).all()
    watch = redundancy_watch(classified, corr_high=0.70)
    assert len(watch) == 1
    assert set([watch.iloc[0]["fold"], watch.iloc[0]["into"]]) == {"a", "b"}


# ---------------------------------------------------------------------------
# Forward returns
# ---------------------------------------------------------------------------
def test_forward_returns_horizons():
    dates = pd.date_range("2024-01-01", periods=14, freq="MS").strftime("%Y-%m-%d")
    matrix = pd.DataFrame(
        {"AAA": np.linspace(100, 126, len(dates)), "BBB": np.linspace(100, 87, len(dates))},
        index=dates,
    )
    rebal = list(dates)
    fwd = compute_forward_returns(matrix, rebal, {"1M": 1, "3M": 3, "12M": 12})
    # 1M exists for all but the last; 12M only for the first couple.
    assert len(fwd["1M"]) >= len(dates) - 2
    assert len(fwd["12M"]) >= 1
    first = rebal[0]
    assert fwd["3M"][first]["AAA"] > 0     # rising price -> positive forward return
    assert fwd["3M"][first]["BBB"] < 0     # falling price -> negative forward return
