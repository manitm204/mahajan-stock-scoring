"""Analyst grades & revision module (FMP).

Backfills the Estimate Revisions factor from genuinely point-in-time analyst
data, unlike the yfinance estimates flow which can only accumulate one snapshot
per run. Two FMP series are used:

* ``grades-historical`` — a dated monthly history of analyst rating counts
  (strong-buy .. strong-sell). A net rating score is derived per month and the
  change over 30/60/90 days captures upgrade/downgrade momentum from a single
  fetch (the full history arrives at once).
* ``price-target-summary`` — consensus price targets bucketed by last
  month/quarter/year, from which a price-target momentum signal is computed.

Raw grade history is stored in ``analyst_grades``; the derived revision signals
land in ``analyst_revision_features``, which Layer 2's revisions factor reads.

Run standalone:
    python -m data.grades --tickers AAPL MSFT
"""
from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta, timezone

import pandas as pd

from .config import load_config
from .db import Database, get_db
from .providers import ProviderRegistry
from .utils import get_logger, safe_float, safe_int, to_iso_date

# Split-artifact guard (2026-08-02). FMP retroactively split-adjusts
# ``price_when_posted`` but leaves ``price_target`` in as-published units, so
# across a split the stored pair is unit-inconsistent (CMG/AMZN/AVGO 2022-24
# splits produced fake +300%..+2,800% implied upsides on ~1.7k rows).
# ``adj_price_target`` is FMP's split-adjusted target — the same present-day
# basis as ``price_when_posted`` and ``daily_prices.adj_close`` — so every
# consumer reads the target via ``PT_TARGET_SQL``. The ratio bounds then drop
# the residual handful of rows that are junk on either basis: a real 12-month
# sell-side target essentially never implies more than +300% upside or less
# than -90% downside at issue.
PT_TARGET_SQL = "COALESCE(adj_price_target, price_target)"
PT_MIN_RATIO = 0.10   # target / price_when_posted floor (upside -90%)
PT_MAX_RATIO = 4.00   # target / price_when_posted cap   (upside +300%)

log = get_logger("grades")

SUPPORTED_WINDOWS = (30, 60, 90)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _net_score(sb: int, b: int, h: int, s: int, ss: int) -> tuple[float | None, int]:
    """Net rating score in [-2, 2] and the total analyst count.

    Weights strong opinions double: (2*strong_buy + buy - sell - 2*strong_sell)
    normalized by total ratings. Returns (None, 0) when no analysts cover it.
    """
    total = sb + b + h + s + ss
    if total <= 0:
        return None, 0
    return (2 * sb + b - s - 2 * ss) / total, total


def _normalize_grades(ticker: str, rows: list[dict], now: str) -> list[dict]:
    """Map raw FMP grade rows to storage rows, newest-first, with a net score."""
    out: list[dict] = []
    for r in rows:
        d = to_iso_date(r.get("date"))
        if not d:
            continue
        sb = safe_int(r.get("analystRatingsStrongBuy")) or 0
        b = safe_int(r.get("analystRatingsBuy")) or 0
        h = safe_int(r.get("analystRatingsHold")) or 0
        s = safe_int(r.get("analystRatingsSell")) or 0
        ss = safe_int(r.get("analystRatingsStrongSell")) or 0
        net, total = _net_score(sb, b, h, s, ss)
        out.append({
            "ticker": ticker, "date": d, "strong_buy": sb, "buy": b, "hold": h,
            "sell": s, "strong_sell": ss, "total": total, "source": "fmp",
            "fetched_at": now, "_net": net,
        })
    out.sort(key=lambda x: x["date"], reverse=True)
    return out


def _net_at_or_before(rows: list[dict], target: str) -> float | None:
    """Net score of the newest grade row dated on/before ``target`` (None if none)."""
    for r in rows:  # rows are newest-first
        if r["date"] <= target and r["_net"] is not None:
            return r["_net"]
    return None


def _load_grades(db: Database, ticker: str) -> list[dict]:
    """Stored grade history for a ticker, newest-first, with the net score.

    Mirrors :func:`_normalize_grades` output (date, counts, total, ``_net``) but
    reads the already-ingested ``analyst_grades`` table, so revision features can
    be replayed for any historical date without another API call.
    """
    rows = db.query(
        "SELECT date, strong_buy, buy, hold, sell, strong_sell, total "
        "FROM analyst_grades WHERE ticker = ? ORDER BY date DESC", (ticker,))
    out: list[dict] = []
    for r in rows:
        net, total = _net_score(
            safe_int(r["strong_buy"]) or 0, safe_int(r["buy"]) or 0,
            safe_int(r["hold"]) or 0, safe_int(r["sell"]) or 0,
            safe_int(r["strong_sell"]) or 0)
        out.append({"date": r["date"], "total": total, "_net": net})
    return out


def _pt_window_mean(db: Database, ticker: str, lo: str, hi: str) -> float | None:
    """Mean consensus (split-adjusted) price target from events in ``(lo, hi]``.

    Rows whose target/price ratio falls outside the plausibility bounds are
    dropped — see the PT_* constants at module top.
    """
    rows = db.query(
        f"SELECT {PT_TARGET_SQL} AS price_target, price_when_posted "
        "FROM analyst_price_target_events "
        "WHERE ticker = ? AND published_date > ? AND published_date <= ?",
        (ticker, lo, hi))
    vals = []
    for r in rows:
        v = safe_float(r["price_target"])
        if v is None or v <= 0:
            continue
        pw = safe_float(r["price_when_posted"])
        if pw is not None and pw > 0 and not (PT_MIN_RATIO <= v / pw <= PT_MAX_RATIO):
            continue
        vals.append(v)
    return sum(vals) / len(vals) if vals else None


def _pt_features(db: Database, ticker: str, today: str) -> dict:
    """Dated price-target consensus + momentum from the event stream.

    Point-in-time (unlike the live ``price-target-summary`` bucket): compares the
    mean target over the trailing month ``(D-30, D]`` with the prior-month window
    ``(D-90, D-60]`` so it is reconstructable at any historical ``D``.
    """
    base = datetime.strptime(today, "%Y-%m-%d").date()
    d30 = (base - timedelta(days=30)).isoformat()
    d60 = (base - timedelta(days=60)).isoformat()
    d90 = (base - timedelta(days=90)).isoformat()
    recent = _pt_window_mean(db, ticker, d30, today)
    prior = _pt_window_mean(db, ticker, d90, d60)
    feat: dict[str, object] = {}
    if recent not in (None, 0):
        feat["pt_consensus"] = recent
    if recent not in (None, 0) and prior not in (None, 0):
        feat["pt_momentum"] = recent / prior - 1.0
    return feat


def _normalize_pt_events(ticker: str, raw: list[dict], now: str) -> list[dict]:
    """Map raw FMP price-target-news rows to storage rows."""
    out: list[dict] = []
    for r in raw:
        published = (r.get("publishedDate") or "").strip()
        if not published:
            continue
        pt = safe_float(r.get("priceTarget"))
        if pt is None:
            continue
        out.append({
            "ticker": ticker,
            "published_date": published,
            "analyst_company": r.get("analystCompany"),
            "analyst_name": r.get("analystName"),
            "price_target": pt,
            "adj_price_target": safe_float(r.get("adjPriceTarget")),
            "price_when_posted": safe_float(r.get("priceWhenPosted")),
            "news_title": r.get("newsTitle"),
            "news_url": r.get("newsURL"),
            "source": "fmp",
            "fetched_at": now,
        })
    return out


def _pt_event_features(db: Database, ticker: str, today: str,
                       window_days: int = 30) -> dict:
    """Trailing-window aggregates over analyst_price_target_events.

    Returns three signals that capture revision momentum from dated events:
      * `pt_event_count_{w}d`   — coverage / activity intensity
      * `pt_upgrade_ratio_{w}d` — share of events with target > price-when-posted
      * `pt_target_upside_{w}d` — mean (target / price-when-posted - 1)
    """
    base = datetime.strptime(today, "%Y-%m-%d").date()
    since = (base - timedelta(days=window_days)).isoformat()
    # Upper bound is essential: without it every BACKFILLED snapshot would count
    # all future events as "in window" (found 2026-08-04 — a 2015 AAPL snapshot
    # carried 254 'trailing-30d' events, i.e. the entire 2021-2026 history).
    # `< today+1d` admits same-day events, matching the live daily snapshot.
    until = (base + timedelta(days=1)).isoformat()
    rows = db.query(
        f"SELECT {PT_TARGET_SQL} AS price_target, price_when_posted "
        "FROM analyst_price_target_events "
        "WHERE ticker = ? AND published_date >= ? AND published_date < ?",
        (ticker, since, until))
    feat: dict[str, object] = {}
    pairs = [(safe_float(r["price_target"]), safe_float(r["price_when_posted"]))
             for r in rows]
    pairs = [(pt, pw) for pt, pw in pairs
             if pt is not None and pw is not None and pw > 0
             and PT_MIN_RATIO <= pt / pw <= PT_MAX_RATIO]
    feat[f"pt_event_count_{window_days}d"] = len(pairs)
    if pairs:
        upgrades = sum(1 for pt, pw in pairs if pt > pw)
        feat[f"pt_upgrade_ratio_{window_days}d"] = upgrades / len(pairs)
        feat[f"pt_target_upside_{window_days}d"] = sum(
            (pt - pw) / pw for pt, pw in pairs) / len(pairs)
    return feat


def compute_features(db: Database, ticker: str, grades: list[dict],
                     today: str) -> dict:
    """Assemble the revision feature row for snapshot date ``today``.

    Uses only already-stored dated tables (``analyst_grades`` +
    ``analyst_price_target_events``), so the same computation serves both the
    live daily snapshot and the historical backfill and is point-in-time correct.
    ``grades`` are the ticker's rating rows newest-first with a ``_net`` score
    (from :func:`_normalize_grades` live, or :func:`_load_grades` on replay).
    """
    feat: dict[str, object] = {"ticker": ticker, "snapshot_date": today,
                               "source": "fmp", "computed_at": _now()}
    net_now = _net_at_or_before(grades, today)
    if net_now is not None:
        # total_analysts from the same grade row that set net_now.
        latest = next((g for g in grades if g["date"] <= today and g["_net"] is not None), None)
        feat["rating_net_score"] = net_now
        if latest is not None:
            feat["total_analysts"] = latest["total"]
        base = datetime.strptime(today, "%Y-%m-%d").date()
        for w in SUPPORTED_WINDOWS:
            then = _net_at_or_before(grades, (base - timedelta(days=w)).isoformat())
            if then is not None:
                feat[f"rating_change_{w}d"] = net_now - then
    feat.update(_pt_features(db, ticker, today))
    feat.update(_pt_event_features(db, ticker, today))
    return feat


def _has_signal(feat: dict) -> bool:
    """True if the feature row carries a real signal worth storing.

    Ignores identity/metadata keys and the bare event *count* (which
    :func:`_pt_event_features` always emits, even as 0 for names with no
    coverage) so a snapshot with neither grades nor price-target events is not
    persisted as an empty row.
    """
    meta = {"ticker", "snapshot_date", "source", "computed_at", "pt_event_count_30d"}
    return any(k not in meta and v is not None for k, v in feat.items())


def update_ticker(db: Database, provider, ticker: str, today: str) -> bool:
    raw = provider.get_analyst_grades(ticker)
    grades = _normalize_grades(ticker, raw, _now())
    if grades:
        db.upsert("analyst_grades",
                  [{k: v for k, v in g.items() if k != "_net"} for g in grades],
                  conflict=["ticker", "date"])

    # Price-target-news events feed the new 30d revision aggregates. Ingest
    # before feature computation so the trailing-window query sees today's pull.
    events_raw = provider.get_price_target_news(ticker, max_pages=5)
    events = _normalize_pt_events(ticker, events_raw, _now())
    if events:
        db.insert_ignore("analyst_price_target_events", events)

    if not grades and not events:
        return False

    feat = compute_features(db, ticker, grades, today)
    db.upsert("analyst_revision_features", [feat],
              conflict=["ticker", "snapshot_date"])
    return True


def _month_end_grid(start: str, end: str) -> list[str]:
    """Month-end snapshot dates in ``[start, end]`` (inclusive of both ends)."""
    dates = [d.date().isoformat() for d in pd.date_range(start, end, freq="ME")]
    if not dates or dates[-1] < end:
        dates.append(end)
    return dates


def backfill_features(db: Database, tickers: list[str],
                      start: str, end: str | None = None) -> dict:
    """Replay ``analyst_revision_features`` over history from stored raw tables.

    The daily flow only snapshots *today*, so the feature table starts empty of
    history even though ``analyst_grades`` (monthly, back to 2022) and
    ``analyst_price_target_events`` are fully dated. This recomputes a feature row
    for each month-end from ``start`` to ``end`` using no API calls, unlocking the
    revisions factor for historical scoring/backtests.
    """
    end = end or date.today().isoformat()
    grid = _month_end_grid(start, end)
    written = 0
    empty: list[str] = []
    for i, ticker in enumerate(tickers, 1):
        grades = _load_grades(db, ticker)
        rows = []
        for d in grid:
            feat = compute_features(db, ticker, grades, d)
            if _has_signal(feat):
                rows.append(feat)
        if rows:
            written += db.upsert("analyst_revision_features", rows,
                                 conflict=["ticker", "snapshot_date"])
        else:
            empty.append(ticker)
        if i % 50 == 0 or i == len(tickers):
            log.info("Revisions backfill: %d/%d (%d rows, %d snapshots/grid)",
                     i, len(tickers), written, len(grid))
    return {"rows": written, "snapshots": len(grid), "empty": empty}


def _normalize_grade_actions(ticker: str, raw: list[dict], now: str) -> list[dict]:
    """Map raw FMP /stable/grades rows to storage rows."""
    out: list[dict] = []
    for r in raw:
        d = to_iso_date(r.get("date"))
        if not d:
            continue
        out.append({
            "ticker": ticker,
            "date": d,
            "grading_company": r.get("gradingCompany"),
            "previous_grade": r.get("previousGrade"),
            "new_grade": r.get("newGrade"),
            "action": r.get("action"),
            "source": "fmp",
            "fetched_at": now,
        })
    return out


def update_grade_events(db: Database, tickers: list[str],
                        registry: ProviderRegistry | None = None) -> dict:
    """Ingest individual rating actions (/stable/grades) for ``tickers``.

    Full history per call, insert-ignore on the natural key — safe to re-run.
    """
    cfg = load_config()
    registry = registry or ProviderRegistry(cfg)
    provider = registry.fundamentals()
    if provider is None or not hasattr(provider, "get_grade_actions"):
        log.warning("No FMP provider available for grade actions; skipping")
        return {"rows": 0, "failed": []}
    now = _now()
    rows_added = 0
    failed: list[str] = []
    for i, ticker in enumerate(tickers, 1):
        try:
            raw = provider.get_grade_actions(ticker)
            rows = _normalize_grade_actions(ticker, raw, now)
            if rows:
                rows_added += db.insert_ignore("analyst_grade_events", rows)
        except Exception as exc:  # noqa: BLE001
            log.warning("Grade actions failed for %s: %s", ticker, exc)
            failed.append(ticker)
        if i % 50 == 0 or i == len(tickers):
            log.info("Grade actions: %d/%d (%d rows)", i, len(tickers), rows_added)
    return {"rows": rows_added, "failed": failed}


def update_grades(db: Database, tickers: list[str],
                  registry: ProviderRegistry | None = None) -> dict:
    cfg = load_config()
    registry = registry or ProviderRegistry(cfg)
    provider = registry.fundamentals()
    if provider is None or not hasattr(provider, "get_analyst_grades"):
        log.warning("No FMP provider available for analyst grades; skipping")
        return {"records": 0, "failed": []}

    today = date.today().isoformat()
    added = 0
    failed: list[str] = []
    for i, ticker in enumerate(tickers, 1):
        try:
            if update_ticker(db, provider, ticker, today):
                added += 1
        except Exception as exc:  # noqa: BLE001
            log.warning("Grades failed for %s: %s", ticker, exc)
            failed.append(ticker)
        if i % 50 == 0 or i == len(tickers):
            log.info("Grades: %d/%d (%d revision rows)", i, len(tickers), added)
    return {"records": added, "failed": failed}


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill analyst grades & revisions")
    parser.add_argument("--tickers", nargs="*")
    parser.add_argument("--backfill", action="store_true",
                        help="Replay revision features over history from stored "
                             "grade/PT tables (no API calls)")
    parser.add_argument("--grade-actions", action="store_true",
                        help="Ingest individual rating actions (/stable/grades) "
                             "into analyst_grade_events")
    parser.add_argument("--start", default="2022-11-01",
                        help="Backfill start date (default: earliest grade history)")
    parser.add_argument("--end", default=None, help="Backfill end date (default: today)")
    args = parser.parse_args()
    cfg = load_config()
    get_logger("grades", log_file=cfg.log_file)
    with get_db() as db:
        tickers = [t.upper() for t in args.tickers] if args.tickers else db.universe_tickers()
        if args.backfill:
            stats = backfill_features(db, tickers, args.start, args.end)
            print(f"Revision feature rows : {stats['rows']}")
            print(f"Snapshots per ticker  : {stats['snapshots']}")
            print(f"Tickers w/o signal    : {len(stats['empty'])}")
        elif args.grade_actions:
            stats = update_grade_events(db, tickers)
            print(f"Grade-action rows     : {stats['rows']}")
            print(f"Failed                : {len(stats['failed'])}")
        else:
            stats = update_grades(db, tickers)
            print(f"Revision rows written : {stats['records']}")
            print(f"Failed                : {len(stats['failed'])}")


if __name__ == "__main__":
    main()
