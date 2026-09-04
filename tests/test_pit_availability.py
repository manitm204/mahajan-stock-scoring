"""PIT availability gates for period-keyed tables that publish after their key
date: FINRA short interest (settlement + ~14d) and monthly analyst grade rows
(keyed at month start, accreting in place until the month ends)."""
from __future__ import annotations

from pathlib import Path

import pytest

from data.db import Database
from factors.utils import DataContext


@pytest.fixture()
def db(tmp_path: Path) -> Database:
    d = Database(path=tmp_path / "test.db", wal=False)
    d.upsert("universe", [{"ticker": "AAA", "gics_sector": "Tech",
                           "company_name": "A", "active": 1}], conflict=["ticker"])
    yield d
    d.close()


def _si_row(date: str, pct: float) -> dict:
    return {"ticker": "AAA", "date": date, "short_percent_of_float": pct,
            "source": "polygon", "fetched_at": "now"}


def test_short_interest_pit_admits_after_dissemination_lag(db: Database):
    """A settlement row is invisible until ~14 days later in PIT mode."""
    db.upsert("short_interest", [_si_row("2024-05-31", 2.0),
                                 _si_row("2024-06-28", 3.0)],
              conflict=["ticker", "date"])

    # 2024-07-05: the 06-28 settlement (published ~07-12) is not yet available.
    early = DataContext(db=db, as_of="2024-07-05", universe=["AAA"],
                        reporting_lag=True)
    assert early.short_interest().loc["AAA", "short_percent_of_float"] == 2.0
    # 2024-07-15: published.
    late = DataContext(db=db, as_of="2024-07-15", universe=["AAA"],
                       reporting_lag=True)
    assert late.short_interest().loc["AAA", "short_percent_of_float"] == 3.0
    # Live mode (no reporting_lag) keeps the raw settlement-date bound.
    live = DataContext(db=db, as_of="2024-07-05", universe=["AAA"])
    assert live.short_interest().loc["AAA", "short_percent_of_float"] == 3.0


def _grade_row(date: str, strong_buy: int, hold: int) -> dict:
    return {"ticker": "AAA", "date": date, "strong_buy": strong_buy, "buy": 0,
            "hold": hold, "sell": 0, "strong_sell": 0,
            "total": strong_buy + hold, "fetched_at": "now"}


def test_grade_diffusion_admits_only_completed_months(db: Database):
    """The current month's grade row mutates in place — PIT reads skip it."""
    from research.subfactor_expansion.library_flow import _grade_diffusion

    # May diffusion = 2/4 = 0.5; June diffusion = 4/4 = 1.0.
    db.upsert("analyst_grades", [_grade_row("2024-05-01", 2, 2),
                                 _grade_row("2024-06-01", 4, 0)],
              conflict=["ticker", "date"])

    # Mid-June: the June row is still accreting → May is the latest admitted.
    pit = DataContext(db=db, as_of="2024-06-15", universe=["AAA"],
                      reporting_lag=True)
    latest, _ = _grade_diffusion(pit, days=90)
    assert latest["AAA"] == pytest.approx(0.5)
    # After June ends the June row is final and admitted.
    done = DataContext(db=db, as_of="2024-07-02", universe=["AAA"],
                       reporting_lag=True)
    latest, _ = _grade_diffusion(done, days=90)
    assert latest["AAA"] == pytest.approx(1.0)
    # Live mode still sees the in-progress month.
    live = DataContext(db=db, as_of="2024-06-15", universe=["AAA"])
    latest, _ = _grade_diffusion(live, days=90)
    assert latest["AAA"] == pytest.approx(1.0)


def test_short_history_features_respect_lag(db: Database):
    """Windowed short-interest history also excludes unpublished settlements."""
    from research.subfactor_expansion.library_flow import _short_history_features

    db.upsert("short_interest", [
        {"ticker": "AAA", "date": d, "short_percent_of_float": p,
         "short_percent_float_change": c, "source": "polygon", "fetched_at": "now"}
        for d, p, c in [("2024-05-15", 2.0, 0.1), ("2024-05-31", 2.5, 0.5),
                        ("2024-06-15", 3.0, 0.5), ("2024-06-28", 4.0, 1.0)]
    ], conflict=["ticker", "date"])

    pit = DataContext(db=db, as_of="2024-07-05", universe=["AAA"],
                      reporting_lag=True)
    own_pct, _, _, covering = _short_history_features(pit, days=252)
    # Latest available row is 06-15 (published ~06-29); 06-28 is not out yet,
    # so the covering signal is -(latest change) = -0.5, not -1.0.
    assert covering["AAA"] == pytest.approx(-0.5)
