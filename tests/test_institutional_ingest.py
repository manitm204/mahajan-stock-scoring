"""FMP whole-market 13F ownership ingestion + signal derivation."""
from __future__ import annotations

from pathlib import Path

import pytest

from data.db import Database
from data import institutional as inst


@pytest.fixture()
def db(tmp_path: Path) -> Database:
    d = Database(path=tmp_path / "test.db", wal=False)
    yield d
    d.close()


def test_quarter_grid_spans_quarter_ends():
    grid = inst._quarter_grid("2024-02-10", "2025-01-05")
    # Start rounds to the quarter containing Feb (Q1), end to the quarter of Jan (Q1).
    assert grid[0] == (2024, 1, "2024-03-31")
    assert (2024, 4, "2024-12-31") in grid
    assert grid[-1] == (2025, 1, "2025-03-31")


def test_map_summary_pulls_expected_fields():
    raw = {"cik": "0000320193", "investorsHolding": 5856,
           "investorsHoldingChange": -66, "numberOf13Fshares": 9.3e9,
           "numberOf13FsharesChange": -5.6e7, "totalInvested": 2.08e12,
           "ownershipPercent": 62.6, "ownershipPercentChange": -0.01,
           "newPositions": 166, "increasedPositions": 2217,
           "reducedPositions": 1800, "closedPositions": 241, "putCallRatio": 0.8}
    m = inst._map_summary("AAPL", "2025-03-31", raw, "now")
    assert m["investors_holding"] == 5856
    assert m["shares_change"] == pytest.approx(-5.6e7)
    assert m["increased_positions"] == 2217
    assert m["ownership_percent"] == pytest.approx(62.6)


def test_compute_signals_from_ownership(db: Database):
    db.upsert("institutional_ownership_summary", [
        inst._map_summary("AAPL", "2025-03-31", {
            "investorsHolding": 5856, "increasedPositions": 2217,
            "reducedPositions": 1800, "newPositions": 166,
            "closedPositions": 241, "numberOf13FsharesChange": -5.6e7}, "now"),
    ], conflict=["ticker", "report_date"])

    n = inst.compute_signals_from_ownership(db)
    assert n == 1
    sig = db.query("SELECT * FROM institutional_signals WHERE ticker='AAPL'")[0]
    assert sig["fund_count"] == 5856               # whole-market breadth, not ~12
    assert sig["position_increases"] == 2217
    assert sig["new_positions"] == 166
    assert sig["net_share_change"] == pytest.approx(-5.6e7)


def test_pit_admits_by_filing_lag(db: Database):
    """A quarter's signal is only visible ~45 days after quarter end."""
    from factors.utils import DataContext

    db.upsert("universe", [{"ticker": "AAA", "gics_sector": "Tech",
                            "company_name": "A"}], conflict=["ticker"])
    db.upsert("institutional_signals", [{
        "ticker": "AAA", "report_date": "2024-03-31", "fund_count": 500,
        "position_increases": 100, "position_decreases": 50, "new_positions": 20,
        "closed_positions": 10, "net_share_change": 1000.0, "computed_at": "now"}],
        conflict=["ticker", "report_date"])

    # Cutoff 2024-04-15 (< report_date + 45d = 2024-05-15) → not yet available:
    # no rows admitted, so the factor sees no fund_count for AAA.
    early = DataContext(db=db, as_of="2024-04-15", universe=["AAA"], reporting_lag=True)
    early_df = early._institutional_pit()
    assert "fund_count" not in early_df.columns or early_df["fund_count"].isna().all()
    # Cutoff 2024-06-01 (> 2024-05-15) → available.
    late = DataContext(db=db, as_of="2024-06-01", universe=["AAA"], reporting_lag=True)
    assert late._institutional_pit().loc["AAA", "fund_count"] == 500


class _FakeOwnershipProvider:
    def __init__(self):
        self.calls: list[tuple[str, int, int]] = []

    def get_institutional_ownership(self, ticker, year, quarter):
        self.calls.append((ticker, year, quarter))
        return {"investorsHolding": 100, "numberOf13FsharesChange": 1.0}


class _FakeRegistry:
    def __init__(self, provider):
        self._provider = provider

    def fundamentals(self):
        return self._provider


def test_backfill_skips_quarters_before_filing_deadline(db: Database):
    """A quarter is ingested only after its 45-day 13F deadline has passed.

    Regression: MSFT 2026-06-30 was fetched in July (deadline Aug 14) with only
    244 of ~6,400 filers reported, producing shares_change = -5.3B.
    """
    provider = _FakeOwnershipProvider()
    stats = inst.fmp_update_ownership(
        db, ["AAA"], start="2026-01-01", end="2026-07-29",
        registry=_FakeRegistry(provider))

    # Q1 deadline (2026-05-15) has passed; Q2 deadline (2026-08-14) has not.
    assert (("AAA", 2026, 1) in provider.calls)
    assert (("AAA", 2026, 2) not in provider.calls)
    assert stats["rows"] == 1
    dates = [r["report_date"] for r in db.query(
        "SELECT report_date FROM institutional_ownership_summary")]
    assert dates == ["2026-03-31"]


def test_ownership_latest_respects_filing_lag(db: Database):
    """PIT reads of the ownership summary admit a quarter only after +45d."""
    from factors.utils import DataContext
    from research.subfactor_expansion.library_flow import _ownership_latest

    db.upsert("universe", [{"ticker": "AAA", "gics_sector": "Tech",
                            "company_name": "A"}], conflict=["ticker"])
    db.upsert("institutional_ownership_summary", [
        inst._map_summary("AAA", "2024-03-31", {"investorsHolding": 100}, "now"),
        inst._map_summary("AAA", "2024-06-30", {"investorsHolding": 110}, "now"),
    ], conflict=["ticker", "report_date"])

    # 2024-07-15: Q2's 13Fs (due 2024-08-14) are not yet public → Q1 is latest.
    pit = DataContext(db=db, as_of="2024-07-15", universe=["AAA"], reporting_lag=True)
    assert _ownership_latest(pit).loc["AAA", "report_date"] == "2024-03-31"
    # Without the PIT lag (live mode) the raw report_date bound applies.
    live = DataContext(db=db, as_of="2024-07-15", universe=["AAA"])
    assert _ownership_latest(live).loc["AAA", "report_date"] == "2024-06-30"
