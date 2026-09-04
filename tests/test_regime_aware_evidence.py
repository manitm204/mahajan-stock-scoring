from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.panel import ScorePanel
from research.walkforward.regime_aware_evidence import (
    RECENT_MONTHS, build_monthly_cache, as_of,
)


def _toy_panel(dates: list[str], universe: list[str]) -> ScorePanel:
    """One parent 'p1' with two subs whose scores rank the universe deterministically
    (s1 ascending with ticker order, s2 descending), so IC direction is predictable."""
    scores = {}
    for k, d in enumerate(dates):
        base = np.linspace(10, 90, len(universe)) + k
        scores[d] = pd.DataFrame({"s1": base, "s2": base[::-1]}, index=universe)
    return ScorePanel(rebal_dates=list(dates), scores=scores,
                      parent_keys=["p1"], sub_by_parent={"p1": ["s1", "s2"]},
                      universe=list(universe))


def _toy_matrix(dates: list[str], universe: list[str], trend: dict[str, float]) -> pd.DataFrame:
    """Monotone price paths so forward returns are deterministic: ticker T{i} grows at
    (trend intercept + i * trend slope) per period, giving s1 a positive IC and s2 a
    negative IC by construction (s1 ranks tickers ascending, prices grow ascending too)."""
    idx = pd.date_range(dates[0], periods=len(dates) + 8, freq="ME").strftime("%Y-%m-%d")
    data = {}
    for i, t in enumerate(universe):
        growth = 1.0 + trend["intercept"] + i * trend["slope"]
        data[t] = growth ** np.arange(len(idx))
    return pd.DataFrame(data, index=idx)


UNIVERSE = [f"T{i}" for i in range(30)]
DATES = pd.date_range("2018-01-31", periods=10, freq="ME").strftime("%Y-%m-%d").tolist()


def test_build_monthly_cache_shape_and_columns():
    panel = _toy_panel(DATES, UNIVERSE)
    matrix = _toy_matrix(DATES, UNIVERSE, {"intercept": 0.01, "slope": 0.002})
    vix = pd.Series(20.0, index=pd.to_datetime(matrix.index))
    cache = build_monthly_cache(panel, matrix, vix)
    assert set(cache.columns) == {
        "sub_factor", "parent", "date", "vix_level", "coverage",
        "ic_3M", "spread_3M_raw", "ic_6M", "spread_6M_raw",
    }
    assert set(cache["sub_factor"].unique()) == {"s1", "s2"}
    # s1 ranks tickers ascending and prices grow ascending with ticker index -> positive IC.
    s1_ic = cache[cache.sub_factor == "s1"]["ic_6M"].dropna()
    assert (s1_ic > 0).all()
    s2_ic = cache[cache.sub_factor == "s2"]["ic_6M"].dropna()
    assert (s2_ic < 0).all()


def _hand_cache(sub: str, parent: str, dates: list[str], ics: list[float],
                spreads6: list[float], vix_levels: list[float],
                coverage: float = 1.0) -> pd.DataFrame:
    """A hand-built single-sub cache: ic_3M==ic_6M==ics[i] for simplicity."""
    return pd.DataFrame({
        "sub_factor": sub, "parent": parent, "date": dates,
        "vix_level": vix_levels, "coverage": coverage,
        "ic_3M": ics, "ic_6M": ics,
        "spread_3M_raw": spreads6, "spread_6M_raw": spreads6,
    })


def test_as_of_long_run_and_recent_match_hand_computation():
    # 36 months, Jan-2015..Dec-2017 (panel_inception = 2015-06-30 per DATA_START, so the
    # first 5 months fall before inception and must be excluded from long_run).
    dates = pd.date_range("2015-01-31", periods=36, freq="ME").strftime("%Y-%m-%d").tolist()
    ics = [0.01 + 0.001 * i for i in range(36)]        # trending up over time
    spreads = [0.02] * 36
    vix_levels = [20.0] * 36                            # always Medium regime
    cache = _hand_cache("s1", "p1", dates, ics, spreads, vix_levels)
    cutoff = dates[-1]
    vix = pd.Series(vix_levels, index=pd.to_datetime(dates))

    out = as_of(cache, cutoff, vix, panel_inception="2015-06-30")
    row = out[out.sub_factor == "s1"].iloc[0]

    in_window = [ic for d, ic in zip(dates, ics) if d >= "2015-06-30"]
    assert row["long_run_mean_ic"] == pytest.approx(np.mean(in_window), abs=1e-9)
    assert row["n_months"] == len(in_window)

    recent_start = (pd.Timestamp(cutoff) - pd.DateOffset(months=RECENT_MONTHS)).date().isoformat()
    recent = [ic for d, ic in zip(dates, ics) if d >= recent_start]
    assert row["recent_24m_ic"] == pytest.approx(np.mean(recent), abs=1e-9)

    # Constant VIX=20 (pure Medium) -> almost all regime weight in Medium.
    assert row["regime_n_eff_medium"] > row["regime_n_eff_low"]
    assert row["regime_n_eff_medium"] > row["regime_n_eff_high"]

    # spread always +0.02 raw at 6M -> annualised = 0.02 * (12/6) = 0.04
    assert row["long_run_spread_ann"] == pytest.approx(0.04, abs=1e-9)


def test_as_of_excludes_dates_after_cutoff():
    dates = pd.date_range("2016-01-31", periods=24, freq="ME").strftime("%Y-%m-%d").tolist()
    ics = [0.05] * 12 + [-0.05] * 12          # sign flips halfway through
    spreads = [0.01] * 24
    vix_levels = [20.0] * 24
    cache = _hand_cache("s1", "p1", dates, ics, spreads, vix_levels)
    vix = pd.Series(vix_levels, index=pd.to_datetime(dates))

    cutoff = dates[11]                         # exactly at the sign flip boundary
    out = as_of(cache, cutoff, vix, panel_inception="2015-06-30")
    row = out[out.sub_factor == "s1"].iloc[0]
    # Only the first 12 (all +0.05) months should be visible -- no look-ahead into the
    # -0.05 months that come after cutoff.
    assert row["long_run_mean_ic"] == pytest.approx(0.05, abs=1e-9)
    assert row["pct_positive_years"] == pytest.approx(1.0, abs=1e-9)


def test_as_of_empty_window_returns_all_expected_columns():
    # cutoff before panel_inception -> the PIT filter empties the window entirely;
    # the empty-path DataFrame must still carry every column the populated path
    # produces (including the 6 regime_* columns), so a caller doing df["regime_ic_low"]
    # on either path never KeyErrors.
    cache = _hand_cache("s1", "p1", ["2020-01-31"], [0.05], [0.01], [20.0])
    vix = pd.Series([20.0], index=pd.to_datetime(["2020-01-31"]))
    out = as_of(cache, "2015-01-01", vix, panel_inception="2015-06-30")
    assert out.empty
    expected_cols = {
        "sub_factor", "parent", "long_run_mean_ic", "long_run_std_ic",
        "long_run_hit_rate", "long_run_spread_ann", "recent_24m_ic",
        "pct_positive_years", "persistence_ir", "spread_consistency",
        "coverage", "n_months", "n_years", "regime_dependence", "expected_ic",
        "regime_ic_low", "regime_n_eff_low", "regime_ic_medium", "regime_n_eff_medium",
        "regime_ic_high", "regime_n_eff_high",
    }
    assert set(out.columns) == expected_cols
