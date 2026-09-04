"""PIT universe history: change-feed parsing, backward spell replay,
members_as_of queries, and the FMP deep-price path used by the backfill."""
from __future__ import annotations

from pathlib import Path

import pytest

from data.db import Database
from data.providers import FMPProvider
from data.universe_history import FEED_ORIGIN, build_spells, fetch_change_events


@pytest.fixture()
def db(tmp_path: Path) -> Database:
    d = Database(path=tmp_path / "test.db", wal=False)
    yield d
    d.close()


# --------------------------------------------------------------------------- #
# Change-feed parsing
# --------------------------------------------------------------------------- #
class _FakeFMP:
    """Stands in for FMPProvider._get with canned feed rows."""

    def __init__(self, rows):
        self._rows = rows

    def _get(self, path, params=None):
        return self._rows


def test_pure_removal_rows_are_not_additions():
    """FMP reuses `symbol` for the removed ticker when addedSecurity is empty
    (verified against the real NBL/RTN/M/BHF rows) — must parse as removal."""
    rows = [
        {"date": "2020-10-09", "symbol": "NBL", "addedSecurity": "",
         "removedTicker": "NBL", "removedSecurity": "Noble Energy Inc"},
        {"date": "2020-10-12", "symbol": "VNT", "addedSecurity": "Vontier",
         "removedTicker": "", "removedSecurity": ""},
    ]
    events = fetch_change_events(_FakeFMP(rows))
    assert events[0] == {"date": "2020-10-09", "added": "", "removed": "NBL"}
    assert events[1] == {"date": "2020-10-12", "added": "VNT", "removed": ""}


def test_change_events_normalize_symbols_and_sort():
    rows = [
        {"date": "2021-06-04", "symbol": "brk.b", "addedSecurity": "Berkshire",
         "removedTicker": "", "removedSecurity": ""},
        {"date": "2020-01-02", "symbol": "AAA", "addedSecurity": "A Corp",
         "removedTicker": "zzz.a", "removedSecurity": "Z Corp"},
    ]
    events = fetch_change_events(_FakeFMP(rows))
    assert [e["date"] for e in events] == ["2020-01-02", "2021-06-04"]
    assert events[1]["added"] == "BRK-B"
    assert events[0]["removed"] == "ZZZ-A"


# --------------------------------------------------------------------------- #
# Backward spell replay
# --------------------------------------------------------------------------- #
def test_build_spells_replay():
    current = {"AAA", "BBB"}
    events = [
        # CCC removed 2020-05-01 (replaced by BBB the same day)
        {"date": "2020-05-01", "added": "BBB", "removed": "CCC"},
        # CCC itself was added 2018-03-01
        {"date": "2018-03-01", "added": "CCC", "removed": ""},
    ]
    spells, anomalies = build_spells(current, events)
    by = {(s["ticker"], s["start_date"]): s["end_date"] for s in spells}
    assert by[("BBB", "2020-05-01")] is None          # current, added in-feed
    assert by[("CCC", "2018-03-01")] == "2020-05-01"  # closed spell
    assert by[("AAA", FEED_ORIGIN)] is None           # member since before feed
    assert not anomalies


def test_build_spells_readded_ticker_gets_two_spells():
    current = {"XXX"}
    events = [
        {"date": "2024-01-05", "added": "XXX", "removed": ""},   # re-added
        {"date": "2020-06-01", "added": "", "removed": "XXX"},   # removed
        {"date": "2015-02-01", "added": "XXX", "removed": ""},   # first added
    ]
    spells, anomalies = build_spells(current, events)
    xxx = sorted([s for s in spells if s["ticker"] == "XXX"],
                 key=lambda s: s["start_date"])
    assert len(xxx) == 2
    assert (xxx[0]["start_date"], xxx[0]["end_date"]) == ("2015-02-01", "2020-06-01")
    assert (xxx[1]["start_date"], xxx[1]["end_date"]) == ("2024-01-05", None)
    assert not anomalies


def test_build_spells_flags_unknown_addition_as_anomaly():
    # An add event for a name with no later membership (rename/merger noise)
    # must be surfaced, not silently turned into a spell.
    spells, anomalies = build_spells(set(), [
        {"date": "2019-04-02", "added": "GHOST", "removed": ""}])
    assert not [s for s in spells if s["ticker"] == "GHOST"]
    assert len(anomalies) == 1


# --------------------------------------------------------------------------- #
# members_as_of
# --------------------------------------------------------------------------- #
def test_members_as_of_boundaries(db: Database):
    db.upsert("universe_history", [
        {"ticker": "AAA", "start_date": "2015-01-01", "end_date": None},
        {"ticker": "BBB", "start_date": "2016-06-01", "end_date": "2020-03-15"},
    ], conflict=["ticker", "start_date"])
    assert db.members_as_of("2015-06-30") == ["AAA"]
    assert db.members_as_of("2016-06-01") == ["AAA", "BBB"]   # start inclusive
    assert db.members_as_of("2020-03-14") == ["AAA", "BBB"]
    assert db.members_as_of("2020-03-15") == ["AAA"]          # end exclusive
    assert db.members_as_of("2014-12-31") == []


def test_members_as_of_empty_table_returns_empty(db: Database):
    assert db.members_as_of("2020-01-01") == []


# --------------------------------------------------------------------------- #
# FMP deep price path
# --------------------------------------------------------------------------- #
def test_fmp_get_prices_normalizes_and_sorts(monkeypatch):
    p = FMPProvider("k", "https://example.test/stable")
    bars = [  # FMP serves newest-first; one bad row must be dropped
        {"date": "2016-01-05", "open": 10.1, "high": 10.5, "low": 9.9,
         "close": 10.2, "volume": 1000},
        {"date": "2016-01-04", "open": 10.0, "high": 10.4, "low": 9.8,
         "close": None, "volume": 900},
        {"date": "2016-01-03", "open": 9.9, "high": 10.2, "low": 9.7,
         "close": 10.0, "volume": 800},
    ]
    monkeypatch.setattr(p, "_get", lambda path, params=None: bars)
    df = p.get_prices("AAA", "2016-01-01", "2016-01-31")
    assert list(df["date"]) == ["2016-01-03", "2016-01-05"]   # sorted, bad row gone
    assert (df["adj_close"] == df["close"]).all()             # price-return convention


def test_fmp_get_prices_empty_on_blocked(monkeypatch):
    p = FMPProvider("k", "https://example.test/stable")
    monkeypatch.setattr(p, "_get", lambda path, params=None: None)
    assert p.get_prices("AAA", "2016-01-01", "2016-01-31").empty
