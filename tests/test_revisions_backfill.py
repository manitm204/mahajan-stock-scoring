"""Point-in-time correctness of the analyst-revision feature backfill.

The daily flow only snapshots *today*; `backfill_features` replays the feature
table over history from the already-stored dated `analyst_grades` +
`analyst_price_target_events`. These tests pin the PIT rules: a snapshot dated D
must use only rows dated on/before D, and rating changes/PT momentum must be
reconstructed from that windowed history.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from data.db import Database
from data import grades as g


@pytest.fixture()
def db(tmp_path: Path) -> Database:
    d = Database(path=tmp_path / "test.db", wal=False)
    yield d
    d.close()


def _grade(ticker, date, sb, b, h, s, ss):
    return {"ticker": ticker, "date": date, "strong_buy": sb, "buy": b,
            "hold": h, "sell": s, "strong_sell": ss, "total": sb + b + h + s + ss,
            "source": "test", "fetched_at": "t"}


def _pt_event(ticker, published, target, when):
    return {"ticker": ticker, "published_date": published, "analyst_company": "X",
            "analyst_name": "Y", "price_target": target, "adj_price_target": target,
            "price_when_posted": when, "news_title": "t", "news_url": "u",
            "source": "test", "fetched_at": "t"}


def test_month_end_grid_bounds():
    grid = g._month_end_grid("2022-11-01", "2023-02-15")
    assert grid[0] == "2022-11-30"
    assert grid[-1] == "2023-02-15"          # trailing partial month included
    assert "2023-01-31" in grid


def test_rating_change_is_point_in_time(db: Database):
    # Net score improves over time: all-hold (net 0) → all strong-buy (net 2).
    db.upsert("analyst_grades", [
        _grade("AAA", "2023-01-15", 0, 0, 10, 0, 0),   # net 0
        _grade("AAA", "2023-04-15", 10, 0, 0, 0, 0),   # net 2 (~90d later)
    ], conflict=["ticker", "date"])

    stats = g.backfill_features(db, ["AAA"], "2023-01-31", "2023-05-31")
    assert stats["rows"] > 0

    # At 2023-02-28 only the Jan grade is visible → net 0, no 90d change yet.
    feb = db.query("SELECT rating_net_score, rating_change_90d FROM "
                   "analyst_revision_features WHERE ticker='AAA' AND snapshot_date='2023-02-28'")[0]
    assert feb["rating_net_score"] == 0.0
    assert feb["rating_change_90d"] is None      # nothing ~90d before Feb

    # At 2023-04-30 the Apr grade is visible (net 2) and the ~90d-ago value is
    # the Jan grade (net 0) → change of +2.
    apr = db.query("SELECT rating_net_score, rating_change_90d FROM "
                   "analyst_revision_features WHERE ticker='AAA' AND snapshot_date='2023-04-30'")[0]
    assert apr["rating_net_score"] == 2.0
    assert apr["rating_change_90d"] == pytest.approx(2.0)


def test_no_future_leakage(db: Database):
    # A big upgrade lands only in May; an April snapshot must not see it.
    db.upsert("analyst_grades", [
        _grade("BBB", "2023-01-10", 0, 0, 10, 0, 0),   # net 0
        _grade("BBB", "2023-05-10", 10, 0, 0, 0, 0),   # net 2
    ], conflict=["ticker", "date"])
    g.backfill_features(db, ["BBB"], "2023-04-01", "2023-05-31")
    apr = db.query("SELECT rating_net_score FROM analyst_revision_features "
                   "WHERE ticker='BBB' AND snapshot_date='2023-04-30'")[0]
    assert apr["rating_net_score"] == 0.0        # May upgrade invisible in April


def test_pt_momentum_from_dated_events(db: Database):
    # Targets rise from ~100 (prior-month window) to ~120 (trailing month).
    db.insert_ignore("analyst_price_target_events", [
        _pt_event("CCC", "2023-03-05", 100, 95),   # in (D-90, D-60] for D=2023-05-31
        _pt_event("CCC", "2023-03-10", 100, 96),
        _pt_event("CCC", "2023-05-20", 120, 110),  # in (D-30, D]
        _pt_event("CCC", "2023-05-25", 120, 111),
    ])
    g.backfill_features(db, ["CCC"], "2023-05-31", "2023-05-31")
    row = db.query("SELECT pt_momentum, pt_event_count_30d, pt_upgrade_ratio_30d "
                   "FROM analyst_revision_features WHERE ticker='CCC' "
                   "AND snapshot_date='2023-05-31'")[0]
    assert row["pt_momentum"] == pytest.approx(120 / 100 - 1.0)   # +0.20
    assert row["pt_event_count_30d"] == 2
    assert row["pt_upgrade_ratio_30d"] == pytest.approx(1.0)      # both targets > price


def test_empty_ticker_writes_nothing(db: Database):
    stats = g.backfill_features(db, ["ZZZ"], "2023-01-31", "2023-03-31")
    assert stats["rows"] == 0
    assert "ZZZ" in stats["empty"]
