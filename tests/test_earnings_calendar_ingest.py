"""FMP /stable/earnings ingestion into earnings_calendar."""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pytest

from data.db import Database
from data import earnings_calendar as ec


@pytest.fixture()
def db(tmp_path: Path) -> Database:
    d = Database(path=tmp_path / "test.db", wal=False)
    yield d
    d.close()


TODAY = date.today()
PAST = (TODAY - timedelta(days=30)).isoformat()
FUTURE = (TODAY + timedelta(days=60)).isoformat()
FUTURE_SHIFTED = (TODAY + timedelta(days=67)).isoformat()


class _FakeFMP:
    name = "fmp"

    def __init__(self, rows):
        self.rows = rows

    def get_earnings(self, ticker, limit=1000):
        return self.rows


def test_fmp_rows_map_eps_and_revenue(db: Database):
    provider = _FakeFMP([
        {"symbol": "AAA", "date": FUTURE, "epsActual": None,
         "epsEstimated": 2.03, "revenueActual": None,
         "revenueEstimated": 1.14e11},
        {"symbol": "AAA", "date": PAST, "epsActual": 2.01,
         "epsEstimated": 1.94, "revenueActual": 1.11e11,
         "revenueEstimated": 1.09e11},
    ])
    n = ec.update_ticker(db, "AAA", provider)
    assert n == 2
    past = db.query_one(
        "SELECT * FROM earnings_calendar WHERE ticker='AAA' AND earnings_date=?",
        (PAST,))
    assert past["eps_actual"] == pytest.approx(2.01)
    assert past["eps_estimate"] == pytest.approx(1.94)
    assert past["revenue_actual"] == pytest.approx(1.11e11)
    assert past["source"] == "fmp"
    fut = db.query_one(
        "SELECT * FROM earnings_calendar WHERE ticker='AAA' AND earnings_date=?",
        (FUTURE,))
    assert fut["eps_actual"] is None
    assert fut["eps_estimate"] == pytest.approx(2.03)


def test_shifted_forecast_date_replaces_stale_future_row(db: Database):
    """A future announcement date that moves must not leave a phantom row."""
    provider = _FakeFMP([{"symbol": "AAA", "date": FUTURE, "epsActual": None,
                          "epsEstimated": 1.0, "revenueActual": None,
                          "revenueEstimated": None}])
    ec.update_ticker(db, "AAA", provider)
    provider.rows = [{"symbol": "AAA", "date": FUTURE_SHIFTED, "epsActual": None,
                      "epsEstimated": 1.0, "revenueActual": None,
                      "revenueEstimated": None}]
    ec.update_ticker(db, "AAA", provider)
    dates = [r["earnings_date"] for r in db.query(
        "SELECT earnings_date FROM earnings_calendar WHERE ticker='AAA'")]
    assert dates == [FUTURE_SHIFTED]


def test_empty_fmp_falls_back_to_yfinance(db: Database, monkeypatch):
    """A blocked/empty FMP response must not wipe the yfinance path."""
    monkeypatch.setattr(ec, "fetch_dates", lambda ticker: [
        {"earnings_date": PAST, "earnings_time": None,
         "fiscal_quarter": "Q1-2026", "eps_estimate": 1.0, "eps_actual": 1.1}])
    n = ec.update_ticker(db, "AAA", _FakeFMP([]))
    assert n == 1
    row = db.query_one("SELECT * FROM earnings_calendar WHERE ticker='AAA'")
    assert row["source"] == "yfinance"
    assert row["eps_actual"] == pytest.approx(1.1)
