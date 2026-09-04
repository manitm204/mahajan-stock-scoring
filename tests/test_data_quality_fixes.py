"""Regression tests for the 2026-08-02 data-quality fixes.

1. Split-artifact fix: price-target consumers read the split-adjusted target
   (``COALESCE(adj_price_target, price_target)``) and drop rows outside the
   target/price plausibility band — so an FMP row whose ``price_target`` is in
   pre-split units while ``price_when_posted`` was retroactively adjusted
   (CMG/AMZN/AVGO 2022-24 pattern) no longer produces a fake +2,800% upside.
2. Sector taxonomy fix: both universe ingestion paths normalize vendor sector
   labels onto the canonical GICS set so ``sector_percentile`` peers stocks
   correctly.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from data.db import Database
from data import grades as g
from data.universe import SECTOR_NORMALIZE, normalize_sector


@pytest.fixture()
def db(tmp_path: Path) -> Database:
    d = Database(path=tmp_path / "test.db", wal=False)
    yield d
    d.close()


def _pt_event(ticker, published, target, adj_target, when):
    return {"ticker": ticker, "published_date": published, "analyst_company": "X",
            "analyst_name": "Y", "price_target": target,
            "adj_price_target": adj_target, "price_when_posted": when,
            "news_title": "t", "news_url": "u", "source": "test", "fetched_at": "t"}


def test_pt_event_features_use_adjusted_target(db: Database):
    # CMG-style split artifact: target quoted pre-split (2100), price_when_posted
    # retroactively split-adjusted (28.33), adj_price_target = 2100/50 = 42.
    db.upsert("analyst_price_target_events", [
        _pt_event("AAA", "2024-06-10T00:00:00.000Z", 2100.0, 42.0, 28.33),
        _pt_event("AAA", "2024-06-12T00:00:00.000Z", 30.0, 30.0, 28.0),
    ], conflict=["ticker", "published_date", "analyst_company", "price_target"])

    feat = g._pt_event_features(db, "AAA", "2024-06-30", window_days=30)
    assert feat["pt_event_count_30d"] == 2
    # Adjusted basis: upsides are 42/28.33-1 ≈ +0.483 and 30/28-1 ≈ +0.071 —
    # NOT the artifact 2100/28.33-1 ≈ +73x.
    assert feat["pt_upgrade_ratio_30d"] == 1.0
    assert 0.2 < feat["pt_target_upside_30d"] < 0.4


def test_pt_event_features_drop_junk_rows(db: Database):
    # Row with no usable adjusted target and an impossible ratio on the raw
    # basis (400 / 0.287 ≈ 1394x) must be excluded entirely.
    db.upsert("analyst_price_target_events", [
        _pt_event("BBB", "2024-06-10T00:00:00.000Z", 400.0, None, 0.287),
        _pt_event("BBB", "2024-06-12T00:00:00.000Z", 110.0, 110.0, 100.0),
    ], conflict=["ticker", "published_date", "analyst_company", "price_target"])

    feat = g._pt_event_features(db, "BBB", "2024-06-30", window_days=30)
    assert feat["pt_event_count_30d"] == 1
    assert abs(feat["pt_target_upside_30d"] - 0.10) < 1e-9


def test_pt_event_features_exclude_future_events(db: Database):
    # Look-ahead regression (found 2026-08-04): a backfilled snapshot must see
    # only events inside its trailing window — never events published AFTER the
    # snapshot date. Same-day events are admitted (matches the live daily pull).
    db.upsert("analyst_price_target_events", [
        _pt_event("CCC", "2022-05-20T00:00:00.000Z", 120.0, 120.0, 100.0),  # in window
        _pt_event("CCC", "2022-06-10T00:00:00.000Z", 90.0, 90.0, 100.0),   # same-day
        _pt_event("CCC", "2023-01-05T00:00:00.000Z", 300.0, 300.0, 100.0),  # FUTURE
        _pt_event("CCC", "2026-01-05T00:00:00.000Z", 400.0, 400.0, 100.0),  # FUTURE
    ], conflict=["ticker", "published_date", "analyst_company", "price_target"])

    feat = g._pt_event_features(db, "CCC", "2022-06-10", window_days=30)
    assert feat["pt_event_count_30d"] == 2          # not 4
    assert feat["pt_upgrade_ratio_30d"] == 0.5      # 120 up, 90 down
    assert abs(feat["pt_target_upside_30d"] - 0.05) < 1e-9


def test_pt_window_mean_uses_adjusted_and_bounds(db: Database):
    # AMZN-style: pre-split-unit target 3450 with adj 172.5; plus one junk row
    # far outside the plausibility band that must be dropped from the mean.
    db.upsert("analyst_price_target_events", [
        _pt_event("CCC", "2022-06-01T00:00:00.000Z", 3450.0, 172.5, 120.0),
        _pt_event("CCC", "2022-06-05T00:00:00.000Z", 180.0, 180.0, 121.0),
        _pt_event("CCC", "2022-06-07T00:00:00.000Z", 5000.0, None, 1.0),
    ], conflict=["ticker", "published_date", "analyst_company", "price_target"])

    mean = g._pt_window_mean(db, "CCC", "2022-05-31", "2022-06-30")
    assert mean == pytest.approx((172.5 + 180.0) / 2)


def test_normalize_sector_maps_all_fmp_labels():
    for fmp_label, gics_label in SECTOR_NORMALIZE.items():
        assert normalize_sector(fmp_label) == gics_label
    # GICS labels pass through unchanged; None stays None.
    assert normalize_sector("Health Care") == "Health Care"
    assert normalize_sector("Information Technology") == "Information Technology"
    assert normalize_sector(None) is None
    assert normalize_sector(" Technology ") == "Information Technology"
