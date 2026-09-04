"""Semiannual walk-forward grid with the rolling-5Y training policy.

Follows the walk-forward conventions used throughout the repo — 6-month H1/H2
test windows from 2017, trailing 5-year training clamped at the 2015-06-30 data
start, and training rebalances capped 6 months before the test boundary so no
selection or regime statistic can consume a forward-return window that reaches
into the test period — implemented independently for this study.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

DATA_START = "2015-06-30"
FIRST_TEST_YEAR = 2017
TRAIN_YEARS = 5
# Longest forward-return horizon any training statistic consumes (the V4
# selection and the regime IC/spread stats both top out at 6M).
SELECTION_HORIZON_MONTHS = 6


@dataclass(frozen=True)
class StudyWindow:
    """One 6-month out-of-sample window plus its rolling-5Y training span."""

    label: str            # e.g. "2020-H1"
    train_start: str
    test_start: str       # the PIT boundary; training ends the day before
    test_end: str

    @property
    def rebal_cap(self) -> str:
        """Latest training rebalance whose 6M forward window ends by the boundary."""
        return (pd.Timestamp(self.test_start)
                - pd.DateOffset(months=SELECTION_HORIZON_MONTHS)).date().isoformat()

    def train_rebalances(self, rebal_dates: list[str]) -> list[str]:
        cap = self.rebal_cap
        return [d for d in rebal_dates if self.train_start <= d <= cap]

    def test_rebalances(self, rebal_dates: list[str]) -> list[str]:
        return [d for d in rebal_dates if self.test_start <= d <= self.test_end]

    @property
    def year(self) -> int:
        return int(self.label.split("-", 1)[0])


def build_windows(last_end: str,
                  first_test_year: int = FIRST_TEST_YEAR) -> list[StudyWindow]:
    """H1/H2 windows from ``first_test_year`` through ``last_end`` (inclusive)."""
    out: list[StudyWindow] = []
    last = pd.Timestamp(last_end)
    year = first_test_year
    while True:
        for half, start, end in (("H1", f"{year}-01-01", f"{year}-06-30"),
                                 ("H2", f"{year}-07-01", f"{year}-12-31")):
            if pd.Timestamp(start) > last:
                return out
            train_start = max(
                (pd.Timestamp(start) - pd.DateOffset(years=TRAIN_YEARS))
                .date().isoformat(),
                DATA_START)
            eff_end = min(pd.Timestamp(end), last).date().isoformat()
            out.append(StudyWindow(f"{year}-{half}", train_start, start, eff_end))
        year += 1


# 3-year reporting blocks (matches the repo's regime-study convention).
BLOCKS: list[tuple[str, int, int]] = [
    ("2017-2019", 2017, 2019),
    ("2020-2022", 2020, 2022),
    ("2023-2025", 2023, 2025),
    ("2026+", 2026, 2099),
]


def block_of(window_label: str) -> str:
    y = int(window_label.split("-", 1)[0])
    for name, lo, hi in BLOCKS:
        if lo <= y <= hi:
            return name
    return "other"
