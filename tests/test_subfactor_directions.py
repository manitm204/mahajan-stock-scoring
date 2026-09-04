"""Direction-consistency tests for the sub-factor sign fixes (2026-07-06).

Each test seeds a tiny synthetic cross-section covering the pathological cases
(negative EBITDA, negative equity, net cash, shrinking volume, deep drawdown)
and asserts the *ordering* of the resulting raw values / percentile scores is
economically sane. Guards against the negative-denominator sign flips that let
distressed names rank best on inverted ratios.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factors.utils import sector_percentile


SECTOR = pd.Series("Tech", index=["GOOD", "LEVERED", "DISTRESSED", "NETCASH", "MEH", "BLAH"])


def _scores(raw: dict[str, float], higher_is_better: bool) -> pd.Series:
    s = pd.Series(raw, dtype=float).reindex(SECTOR.index)
    return sector_percentile(s, SECTOR, higher_is_better=higher_is_better, min_obs=5)


# ---------------------------------------------------------------------------
# EV multiples in yield form (val_ev_ebitda_inv / val_ev_fcf_inv / val_ev_revenue_inv)
# ---------------------------------------------------------------------------
def test_ebitda_yield_ranks_negative_ebitda_worst():
    ev = pd.Series({"GOOD": 100.0, "LEVERED": 100.0, "DISTRESSED": 100.0,
                    "NETCASH": -20.0, "MEH": 100.0, "BLAH": 100.0})
    ebitda = pd.Series({"GOOD": 20.0, "LEVERED": 10.0, "DISTRESSED": -15.0,
                        "NETCASH": 5.0, "MEH": 8.0, "BLAH": 5.0})
    ebitda_yield = ebitda / ev.where(ev > 0)
    scores = _scores(ebitda_yield.to_dict(), higher_is_better=True)
    # Cheapest genuine cash generator wins; negative EBITDA is the worst.
    assert scores["GOOD"] == scores.max()
    assert scores["DISTRESSED"] == scores.drop("NETCASH").min()
    # Non-positive EV is meaningless -> neutral, never "cheapest".
    assert scores["NETCASH"] == 50.0


def test_old_inverted_multiple_was_the_bug():
    """Documents the failure mode the fix removes: EV/EBITDA with
    higher_is_better=False ranks the negative-EBITDA name best."""
    ev_ebitda = {"GOOD": 5.0, "LEVERED": 10.0, "DISTRESSED": -6.7,
                 "NETCASH": -4.0, "MEH": 12.5, "BLAH": 20.0}
    scores = _scores(ev_ebitda, higher_is_better=False)
    assert scores["DISTRESSED"] > scores["GOOD"]   # the bug, kept as a tripwire


# ---------------------------------------------------------------------------
# Net debt / |EBITDA| (qual_debt_to_ebitda_inv)
# ---------------------------------------------------------------------------
def test_debt_to_abs_ebitda_all_quadrants():
    net_debt = pd.Series({"GOOD": -50.0, "LEVERED": 300.0, "DISTRESSED": 300.0,
                          "NETCASH": -50.0, "MEH": 50.0, "BLAH": 100.0})
    ebitda = pd.Series({"GOOD": 100.0, "LEVERED": 100.0, "DISTRESSED": -100.0,
                        "NETCASH": -100.0, "MEH": 100.0, "BLAH": 100.0})
    ratio = net_debt / ebitda.abs().replace(0, np.nan)
    scores = _scores(ratio.to_dict(), higher_is_better=False)
    # Net cash scores at the good end regardless of EBITDA sign...
    assert scores["GOOD"] > scores["MEH"] > scores["LEVERED"]
    assert scores["NETCASH"] == scores["GOOD"]
    # ...and an indebted money-loser can never rank above an indebted earner.
    assert scores["DISTRESSED"] <= scores["LEVERED"]


# ---------------------------------------------------------------------------
# Debt / |equity| (qual_debt_to_equity_inv)
# ---------------------------------------------------------------------------
def test_debt_to_abs_equity_penalizes_negative_equity():
    debt = pd.Series({"GOOD": 20.0, "LEVERED": 150.0, "DISTRESSED": 150.0,
                      "NETCASH": 0.0, "MEH": 60.0, "BLAH": 80.0})
    equity = pd.Series({"GOOD": 100.0, "LEVERED": 100.0, "DISTRESSED": -30.0,
                        "NETCASH": 100.0, "MEH": 100.0, "BLAH": 100.0})
    ratio = debt / equity.abs().replace(0, np.nan)
    scores = _scores(ratio.to_dict(), higher_is_better=False)
    # Negative-equity + debt is the most levered, not the least.
    assert scores["DISTRESSED"] == scores.min()
    assert scores["NETCASH"] == scores.max()


# ---------------------------------------------------------------------------
# ROE / CFO-to-NI masks
# ---------------------------------------------------------------------------
def test_roe_mask_neutralizes_double_negative():
    equity = pd.Series({"GOOD": 100.0, "DISTRESSED": -50.0})
    roe = pd.Series({"GOOD": 0.20, "DISTRESSED": 0.40})  # -20/-50 reads +0.40
    masked = roe.mask(equity <= 0)
    assert np.isnan(masked["DISTRESSED"])
    assert masked["GOOD"] == pytest.approx(0.20)


def test_cfo_to_ni_mask_neutralizes_negative_ni():
    ni = pd.Series({"GOOD": 10.0, "DISTRESSED": -10.0})
    cfo_to_ni = pd.Series({"GOOD": 1.1, "DISTRESSED": -0.8})  # CFO +8 / NI -10
    masked = cfo_to_ni.mask(ni <= 0)
    assert np.isnan(masked["DISTRESSED"])


# ---------------------------------------------------------------------------
# Momentum: volume confirmation weight must never flip the return's sign
# ---------------------------------------------------------------------------
def test_volume_confirmation_weight_is_positive():
    rel_vol = pd.Series([0.05, 0.5, 1.0, 2.0, 10.0])
    weight = np.clip(1.0 + np.log(rel_vol.clip(lower=0.1)), 0.25, 2.0)
    assert (weight > 0).all()
    ret = pd.Series([0.30, 0.30, 0.30, -0.30, -0.30])
    conf = ret * weight
    assert (np.sign(conf) == np.sign(ret)).all()


def test_max_drawdown_direction_milder_is_better():
    # dd values are <= 0; closer to zero = milder drawdown = better.
    dd = {"GOOD": -0.05, "LEVERED": -0.20, "DISTRESSED": -0.60,
          "NETCASH": -0.10, "MEH": -0.30, "BLAH": -0.40}
    scores = _scores(dd, higher_is_better=True)   # the fixed direction
    assert scores["GOOD"] == scores.max()
    assert scores["DISTRESSED"] == scores.min()


# ---------------------------------------------------------------------------
# Builders emit the fixed directions (guards against regression at the source)
# ---------------------------------------------------------------------------
def test_library_direction_flags():
    from research.subfactor_expansion import library

    src = {
        "val_ev_ebitda_inv": True, "val_ev_fcf_inv": True,
        "val_ev_revenue_inv": True, "mom_max_drawdown_252d": True,
        "qual_debt_to_ebitda_inv": False,
    }
    import inspect
    text = inspect.getsource(library)
    for name, expected in src.items():
        assert f'"{name}"' in text, f"{name} missing from library"
    assert 'Candidate("mom_max_drawdown_252d", mdd, True' in text
    assert 'Candidate("val_ev_ebitda_inv", ebitda_yield, True' in text
    assert 'Candidate("qual_debt_to_ebitda_inv", debt_to_ebitda, False' in text
    assert "val_sales_to_ev" not in [
        line.split('"')[1] for line in text.splitlines()
        if line.strip().startswith('Candidate("')
    ]


def test_production_value_direction_flags():
    import inspect
    from factors import value

    text = inspect.getsource(value)
    assert 'SubFactor("val_ev_ebitda_inv", ebitda_yield, higher_is_better=True)' in text
