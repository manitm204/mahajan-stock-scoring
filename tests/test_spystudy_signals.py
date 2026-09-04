"""Point-in-time and correctness guards for the SPY signal study.

This repo has twice shipped studies whose conclusions came from look-ahead
(the insider window bug, the ghost-member bug). Every signal here must depend
only on data at or before its own date.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spystudy import signals as sg
from spystudy.data import SECTORS, forward_return


@pytest.fixture
def close():
    """Deterministic sector panel long enough to warm up every signal."""
    rng = np.random.default_rng(0)
    n = 900
    idx = pd.bdate_range("2015-01-01", periods=n)
    common = rng.normal(0, 0.008, n)
    data = {"SPY": 100 * np.exp(np.cumsum(common))}
    for i, s in enumerate(SECTORS):
        r = common * (0.5 + 0.05 * i) + rng.normal(0, 0.006, n)
        data[s] = 50 * np.exp(np.cumsum(r))
    return pd.DataFrame(data, index=idx)


def test_rsi_matches_wilder_reference():
    """The canonical Wilder / StockCharts worked example: RSI(14) = 70.53."""
    s = pd.Series([44.3389, 44.0902, 44.1497, 43.6124, 44.3278, 44.8264,
                   45.0955, 45.4245, 45.8433, 46.0826, 45.8931, 46.0328,
                   45.6140, 46.2820, 46.2820])
    out = sg.rsi(s, period=14)
    assert out.iloc[:14].isna().all(), "RSI must not emit before its warmup"
    assert out.iloc[14] == pytest.approx(70.53, abs=0.01)


def test_rsi_bounded_and_monotone_direction():
    rng = np.random.default_rng(1)
    up = pd.Series(np.cumsum(np.abs(rng.normal(1, 0.1, 100))) + 100)
    down = pd.Series(-np.cumsum(np.abs(rng.normal(1, 0.1, 100))) + 500)
    r_up, r_down = sg.rsi(up).dropna(), sg.rsi(down).dropna()
    assert (r_up.between(0, 100)).all() and (r_down.between(0, 100)).all()
    assert r_up.iloc[-1] > 95, "a pure uptrend should pin RSI near 100"
    assert r_down.iloc[-1] < 5, "a pure downtrend should pin RSI near 0"


@pytest.mark.parametrize("name", ["rsi14", "sector_corr_21d", "sector_corr_60d",
                                  "trend_4m", "ma200_dist", "absorption",
                                  "absorption_shift", "absorption_chg"])
def test_signals_are_point_in_time(close, name):
    """Truncating the future must not change any past signal value.

    If a signal peeks ahead, the values computed on the full panel will differ
    from those computed on the truncated panel.
    """
    cut = 700
    full = sg.build(close)[name].iloc[:cut]
    trunc = sg.build(close.iloc[:cut])[name]
    pd.testing.assert_series_equal(full, trunc, check_names=False,
                                   rtol=1e-10, atol=1e-12)


def test_forward_return_looks_forward_only():
    s = pd.Series(np.arange(1.0, 51.0))
    f = forward_return(s, days=21)
    assert f.iloc[-21:].isna().all(), "no forward return without a full window"
    assert f.iloc[0] == pytest.approx(s.iloc[21] / s.iloc[0] - 1)


def test_absorption_ratio_is_a_variance_share(close):
    ar = sg.absorption_ratio(close).dropna()
    assert not ar.empty
    assert (ar > 0).all() and (ar <= 1).all()
    # 2 of 11 sector PCs on correlated returns should dominate but not saturate
    assert 0.4 < ar.mean() < 0.99


def test_sector_correlation_ignores_unlisted_sectors():
    """A sector that is all-NaN in the window must be dropped, not filled."""
    rng = np.random.default_rng(2)
    n = 200
    idx = pd.bdate_range("2015-01-01", periods=n)
    data = {"SPY": pd.Series(100.0, index=idx)}
    for s in SECTORS:
        data[s] = pd.Series(50 * np.exp(np.cumsum(rng.normal(0, 0.01, n))),
                            index=idx)
    df = pd.DataFrame(data)
    base = sg.sector_correlation(df, 60)

    df2 = df.copy()
    df2["XLC"] = np.nan  # as if XLC had not listed yet
    later = sg.sector_correlation(df2, 60)
    assert later.notna().sum() > 0, "10 sectors still clears the minimum"
    assert not np.allclose(base.dropna(), later.dropna()), \
        "dropping a sector must change the mean pairwise correlation"
