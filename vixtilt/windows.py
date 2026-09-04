"""Semiannual rolling-5y train/test windows with explicit no-look-ahead caps.

Each 6-month OOS test window (H1 = Jan-Jun, H2 = Jul-Dec, 2017 → latest half) trains on
the trailing 5 years (clamped at the panel start). Two caps keep training strictly
point-in-time:

* ``selection_rebals`` — capped ``SELECTION_HORIZON_MONTHS`` (6M) before the test start,
  because baseline selection consumes 3M/6M forward returns (same invariant as the
  production walk-forward);
* ``stat_rebals`` — capped ``STAT_HORIZON_MONTHS`` (1M) before the test start, because
  the VIX-regime statistics use 1M forward returns only. Both are further protected by
  truncating the price matrix at the test boundary when forward returns are computed.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

DATA_START = "2015-06-30"
FIRST_TEST_YEAR = 2017
LAST_TEST_END = "2026-06-30"
TRAIN_YEARS = 5
SELECTION_HORIZON_MONTHS = 6   # baseline selection uses 3M/6M forward returns
STAT_HORIZON_MONTHS = 1        # regime statistics use 1M forward returns


@dataclass(frozen=True)
class Window:
    label: str          # e.g. "2020-H1"
    train_start: str
    train_end: str      # test_start - 1 day
    test_start: str
    test_end: str

    def _cap(self, months: int) -> str:
        return (pd.Timestamp(self.test_start)
                - pd.DateOffset(months=months)).date().isoformat()

    def selection_rebals(self, rebal_dates: list[str]) -> list[str]:
        cap = self._cap(SELECTION_HORIZON_MONTHS)
        return [d for d in rebal_dates if self.train_start <= d <= min(self.train_end, cap)]

    def stat_rebals(self, rebal_dates: list[str]) -> list[str]:
        cap = self._cap(STAT_HORIZON_MONTHS)
        return [d for d in rebal_dates if self.train_start <= d <= min(self.train_end, cap)]

    def test_rebals(self, rebal_dates: list[str]) -> list[str]:
        return [d for d in rebal_dates if self.test_start <= d <= self.test_end]

    @property
    def year(self) -> int:
        return int(self.label.split("-", 1)[0])


def semiannual_windows(first_test_year: int = FIRST_TEST_YEAR,
                       last_end: str = LAST_TEST_END,
                       train_years: int = TRAIN_YEARS,
                       data_start: str = DATA_START) -> list[Window]:
    out: list[Window] = []
    last_ts = pd.Timestamp(last_end)
    year = first_test_year
    while True:
        for half, start, end in (("H1", f"{year}-01-01", f"{year}-06-30"),
                                 ("H2", f"{year}-07-01", f"{year}-12-31")):
            if pd.Timestamp(start) > last_ts:
                return out
            test_end = min(pd.Timestamp(end), last_ts).date().isoformat()
            rolled = (pd.Timestamp(start) - pd.DateOffset(years=train_years)).date().isoformat()
            train_start = max(rolled, data_start)
            train_end = (pd.Timestamp(start) - pd.Timedelta(days=1)).date().isoformat()
            out.append(Window(f"{year}-{half}", train_start, train_end, start, test_end))
        year += 1


BLOCK_DEFS: list[tuple[str, int, int]] = [
    ("2017-2019", 2017, 2019),
    ("2020-2022", 2020, 2022),
    ("2023-2025", 2023, 2025),
    ("2026+",     2026, 2099),
]


def block_of(label: str) -> str:
    y = int(label.split("-", 1)[0])
    for name, lo, hi in BLOCK_DEFS:
        if lo <= y <= hi:
            return name
    return "other"
