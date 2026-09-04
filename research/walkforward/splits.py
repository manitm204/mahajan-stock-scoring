"""Walk-forward train/test splits over the full 2015→2026 point-in-time history.

Adjusted-close prices reach back to 2014-06 and the top-weighted momentum sub-factor
(``mom_12_1``) needs ~12 months of history, so full-strength scores begin ~2015-06.
The default scheme draws **annual expanding** splits: for each out-of-sample test year the
model re-selects its entire construction on all prior history, then scores the unseen year.

**The no-look-ahead invariant lives here.** Selection ICs use forward returns out to
``SELECTION_HORIZON_MONTHS`` (6M — the longest horizon the V4 sub/parent pick uses). A
training rebalance may only inform selection if its 6M forward window *ends on or before*
the test-start boundary, otherwise selection would have peeked at test-period returns.
:meth:`WalkForwardSplit.train_rebalances` enforces this by capping the usable training
rebalances 6 months before the boundary; :func:`research.walkforward.selection.select_config`
additionally truncates the training price matrix there so any straddling window is dropped.

The **training-window study** (Section 8) reuses the same test years but varies how far back
training reaches — expanding (all history) vs rolling 5y/3y/2y — via :func:`policy_splits`.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

# The longest forward-return horizon the selection pipeline consumes (V4 picks
# sub-factors and parent weights on the 3M/6M mean IC, so 6M is the binding one).
SELECTION_HORIZON_MONTHS = 6

# Earliest rebalance with a mature 12-month momentum signal (prices from 2014-06).
DATA_START = "2015-06-30"
# Panel span (first/last monthly rebalance the candidate panel is built over).
PANEL_START = "2015-06-30"
PANEL_END = "2026-06-30"
# First out-of-sample test year and the (partial) last one.
FIRST_TEST_YEAR = 2017
LAST_TEST_YEAR = 2026
LAST_TEST_END = "2026-06-30"          # 2026 is a half year of data

# Training-window policies for the Section-8 study. ``None`` = expanding (all history);
# an integer N = rolling N-year window ending at the test boundary.
TRAIN_POLICIES: dict[str, int | None] = {
    "expanding": None,
    "rolling5y": 5,
    "rolling3y": 3,
    "rolling2y": 2,
}


@dataclass(frozen=True)
class WalkForwardSplit:
    """One train/test split.

    ``train_start``/``train_end`` bound the *rebalance dates* eligible for selection;
    ``test_start``/``test_end`` bound the out-of-sample evaluation rebalances. All are ISO
    date strings. ``label`` is a short human tag (the test year, optionally policy-prefixed);
    ``policy`` records the training-window policy (``expanding``/``rolling5y``/…).
    """

    label: str
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    policy: str = "expanding"

    @property
    def test_year(self) -> str:
        return self.test_start[:4]

    @property
    def selection_price_end(self) -> str:
        """Date to truncate the training price matrix at, so no selection forward-return
        window reaches into the test period. Equals the test-start boundary."""
        return self.test_start

    def _cap(self) -> str:
        return (pd.Timestamp(self.selection_price_end)
                - pd.DateOffset(months=SELECTION_HORIZON_MONTHS)).date().isoformat()

    def train_rebalances(self, rebal_dates: list[str]) -> list[str]:
        """Rebalance dates used for selection: within ``[train_start, train_end]`` **and**
        early enough that a full ``SELECTION_HORIZON_MONTHS`` forward window ends by the
        test boundary — the condition that keeps selection strictly out-of-sample."""
        cap = self._cap()
        return [d for d in rebal_dates
                if self.train_start <= d <= self.train_end and d <= cap]

    def test_rebalances(self, rebal_dates: list[str]) -> list[str]:
        return [d for d in rebal_dates if self.test_start <= d <= self.test_end]


def _test_window(year: int) -> tuple[str, str]:
    start = f"{year}-01-01"
    end = LAST_TEST_END if year >= LAST_TEST_YEAR else f"{year}-12-31"
    return start, end


def policy_splits(policy: str, *, first_test_year: int = FIRST_TEST_YEAR,
                  data_start: str = DATA_START) -> list[WalkForwardSplit]:
    """Annual splits for one training-window ``policy``.

    ``expanding`` trains on all history from ``data_start``; ``rolling{N}y`` trains only on
    the trailing N years before each test boundary (clamped to ``data_start`` when the
    window would start before the data does — early rolling splits then coincide with
    expanding, reported honestly)."""
    if policy not in TRAIN_POLICIES:
        raise ValueError(f"unknown training-window policy: {policy!r}")
    n = TRAIN_POLICIES[policy]
    out: list[WalkForwardSplit] = []
    for year in range(first_test_year, LAST_TEST_YEAR + 1):
        test_start, test_end = _test_window(year)
        if n is None:
            train_start = data_start
        else:
            rolled = (pd.Timestamp(test_start) - pd.DateOffset(years=n)).date().isoformat()
            train_start = max(rolled, data_start)
        train_end = (pd.Timestamp(test_start) - pd.Timedelta(days=1)).date().isoformat()
        label = str(year) if policy == "expanding" else f"{policy}:{year}"
        out.append(WalkForwardSplit(label, train_start, train_end,
                                    test_start, test_end, policy=policy))
    return out


def default_splits() -> list[WalkForwardSplit]:
    """The annual expanding walk-forward — the primary scheme for Sections 1-7."""
    return policy_splits("expanding")


def resolve_splits(scheme: str) -> list[WalkForwardSplit]:
    if scheme in ("warmup", "annual", "expanding"):
        return default_splits()
    if scheme in TRAIN_POLICIES:
        return policy_splits(scheme)
    raise ValueError(f"unknown split scheme: {scheme!r}")


# --------------------------------------------------------------------------- #
# Semiannual (6-month) test windows — regime study.
# --------------------------------------------------------------------------- #
SEMI_TRAIN_POLICIES: dict[str, int] = {"rolling5y": 5, "rolling2y": 2}


def _semi_windows(first_year: int, last_end: str) -> list[tuple[str, str, str]]:
    """(label, test_start, test_end) list, 6M windows H1/H2 from first_year to last_end."""
    out: list[tuple[str, str, str]] = []
    last_end_ts = pd.Timestamp(last_end)
    year = first_year
    while True:
        for half, start, end in (("H1", f"{year}-01-01", f"{year}-06-30"),
                                 ("H2", f"{year}-07-01", f"{year}-12-31")):
            if pd.Timestamp(start) > last_end_ts:
                return out
            eff_end = min(pd.Timestamp(end), last_end_ts).date().isoformat()
            out.append((f"{year}-{half}", start, eff_end))
        year += 1


def semiannual_policy_splits(policy: str, *, first_test_year: int = FIRST_TEST_YEAR,
                             last_end: str = LAST_TEST_END,
                             data_start: str = DATA_START,
                             ) -> list[WalkForwardSplit]:
    """Semiannual (6-month) test windows for a rolling training-window ``policy``.

    Windows are H1 (Jan-Jun) and H2 (Jul-Dec) each year from ``first_test_year`` up to
    ``last_end``. Training window is trailing N-years clamped at ``data_start``. Preserves
    the same no-look-ahead invariant as the annual splits.
    """
    if policy not in SEMI_TRAIN_POLICIES:
        raise ValueError(f"unknown semiannual training policy: {policy!r}")
    n = SEMI_TRAIN_POLICIES[policy]
    out: list[WalkForwardSplit] = []
    for label_root, test_start, test_end in _semi_windows(first_test_year, last_end):
        rolled = (pd.Timestamp(test_start) - pd.DateOffset(years=n)).date().isoformat()
        train_start = max(rolled, data_start)
        train_end = (pd.Timestamp(test_start) - pd.Timedelta(days=1)).date().isoformat()
        label = f"{policy}:{label_root}"
        out.append(WalkForwardSplit(label, train_start, train_end,
                                    test_start, test_end, policy=policy))
    return out


def semiannual_labels(first_test_year: int = FIRST_TEST_YEAR,
                      last_end: str = LAST_TEST_END) -> list[str]:
    """Raw '{year}-H{1,2}' labels for the semiannual grid — used to align across policies."""
    return [w[0] for w in _semi_windows(first_test_year, last_end)]
