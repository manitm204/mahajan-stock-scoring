"""Regression tests for delisting realization in the walk-forward return paths.

Before the 2026-07-17 fix, a name whose price series ended mid-window (delisting,
acquisition, bankruptcy) was silently dropped: ``compute_forward_returns`` via
``fwd.dropna()`` and the portfolio simulator via ``_period_return``'s survivor
renormalization. That was a survivorship leak — terminal losses vanished from the
cross-section. Now the last print is carried forward (proceeds held as cash), so a
collapse into delisting is realized. These tests pin that behavior and the
invariants the fix must NOT disturb (no look-ahead, unlisted names stay excluded).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.forward_returns import compute_forward_returns, realize_delistings
from research.walkforward.portfolio import simulate

DATES = ["2020-01-31", "2020-02-28", "2020-03-31", "2020-04-30", "2020-05-29"]


def _matrix() -> pd.DataFrame:
    """SURV trades throughout; GONE collapses to 40 then delists (NaN after Feb);
    LATE lists in April (NaN before)."""
    return pd.DataFrame(
        {
            "SURV": [100.0, 102.0, 104.0, 106.0, 108.0],
            "GONE": [100.0, 40.0, np.nan, np.nan, np.nan],
            "LATE": [np.nan, np.nan, np.nan, 50.0, 55.0],
        },
        index=DATES,
    )


def test_realize_delistings_fills_after_last_print_only():
    filled = realize_delistings(_matrix())
    assert filled.loc["2020-05-29", "GONE"] == 40.0        # carried forward
    assert np.isnan(filled.loc["2020-01-31", "LATE"])      # never back-filled
    assert list(filled.index) == DATES                     # index untouched


def test_forward_returns_realize_delisted_name():
    fwd = compute_forward_returns(_matrix(), DATES, {"3M": 3})
    r = fwd["3M"]["2020-01-31"]
    assert r["GONE"] == pytest.approx(40.0 / 100.0 - 1.0)  # −60% realized, not dropped
    assert "LATE" not in r.index                           # unlisted at start stays out
    assert r["SURV"] == pytest.approx(106.0 / 100.0 - 1.0)


def test_forward_returns_still_drop_windows_past_data_end():
    """Delisting fill must not extend the matrix: windows reaching past the last date
    are still dropped (the no-look-ahead truncation mechanism)."""
    fwd = compute_forward_returns(_matrix(), DATES, {"3M": 3})
    assert "2020-04-30" not in fwd["3M"]                   # 3M end falls past May


def test_simulate_realizes_delisting_loss_instead_of_renormalizing():
    """Equal-weight book of SURV+GONE over Jan→Feb→Mar. GONE has no March print, so the
    old path renormalized onto SURV alone (loss vanished); now Feb→Mar must realize
    GONE at its last print: 0.5·(104/102−1) + 0.5·(40/40−1)."""
    scores = {d: pd.Series({"SURV": 1.0, "GONE": 1.0}) for d in DATES[:3]}
    sectors = pd.Series({"SURV": "Tech", "GONE": "Fin", "LATE": "Tech"})
    res = simulate(scores, _matrix(), sectors, top_pct=1.0, mode="equal")
    jan_feb = res.period_returns["2020-02-28"]
    feb_mar = res.period_returns["2020-03-31"]
    assert jan_feb == pytest.approx(0.5 * (102 / 100 - 1) + 0.5 * (40 / 100 - 1))
    assert feb_mar == pytest.approx(0.5 * (104 / 102 - 1) + 0.5 * 0.0)
    # Old (buggy) behavior would have given the survivor-only return — pin against it.
    assert feb_mar != pytest.approx(104 / 102 - 1)
