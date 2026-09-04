"""Tests for the T+N delayed-execution helper and the diagnostics module."""
import numpy as np
import pandas as pd

from research.ablation.diagnostics import bootstrap_ci, drop_best, rolling_alpha
from research.ablation.engine import _lagged_prices


def _matrix():
    dates = pd.date_range("2020-01-01", periods=30, freq="B").strftime("%Y-%m-%d")
    return pd.DataFrame({"A": np.arange(1.0, 31.0),
                         "B": np.arange(2.0, 32.0)}, index=dates)


def test_lag_zero_is_noop():
    m = _matrix()
    form = [m.index[0], m.index[10], m.index[20]]
    px = _lagged_prices(m, form, 0)
    assert (px.values == m.loc[form].values).all()
    assert list(px.index) == form


def test_lag_shifts_by_trading_days():
    m = _matrix()
    form = [m.index[0], m.index[10], m.index[20]]
    px = _lagged_prices(m, form, 1)
    assert list(px.index) == form                      # labels keep formation dates
    assert px.iloc[0]["A"] == m.iloc[1]["A"]           # but prices are T+1
    assert px.iloc[1]["A"] == m.iloc[11]["A"]


def test_lag_clamps_at_series_end():
    m = _matrix()
    form = [m.index[-1]]
    px = _lagged_prices(m, form, 5)
    assert px.iloc[0]["A"] == m.iloc[-1]["A"]


def _paired_series(n=120, seed=0):
    rng = np.random.default_rng(seed)
    b = pd.Series(rng.normal(0.008, 0.04, n),
                  index=[f"2016-{i:04d}" for i in range(n)])
    p = b * 1.0 + rng.normal(0.002, 0.01, n)           # true alpha ~ +0.2%/period
    return p, b


def test_bootstrap_ci_brackets_alpha():
    p, b = _paired_series()
    out = bootstrap_ci(p, b, ppy=12.0, n_boot=300)
    lo, hi = out["alpha_ci90"]
    assert lo < 0.024 * 12 / 12 * 12 + 1               # sanity: finite
    assert lo < hi
    assert 0.0 < out["p_alpha_neg"] < 1.0 or out["p_alpha_neg"] in (0.0, 1.0)


def test_drop_best_reduces_excess():
    p, b = _paired_series()
    out = drop_best(p, b, ppy=12.0)
    assert out["excess_wo_best3_periods"] <= out["excess_full"]
    assert out["best_year"] in out["excess_by_year"]


def test_rolling_alpha_shape():
    p, b = _paired_series()
    ra = rolling_alpha(p, b, ppy=12.0, window=36)
    assert len(ra) == len(p) - 36 + 1
    assert ra.notna().all()
