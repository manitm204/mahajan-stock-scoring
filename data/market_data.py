"""Market data module — daily OHLCV.

Fetches daily prices via the provider abstraction (Polygon if keyed, else
yfinance), maintaining ~4 years of history with incremental updates. Existing
bars are never deleted; the most recent window is re-fetched on each run so
split/dividend adjustments propagate into ``adj_close``.

Run standalone:
    python -m data.market_data                      # all universe + benchmarks
    python -m data.market_data --tickers AAPL MSFT  # subset
    python -m data.market_data --full               # force full backfill
"""
from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta

from .config import load_config
from .db import Database, get_db
from .providers import ProviderRegistry
from .utils import get_logger, safe_float, safe_int

log = get_logger("market_data")

# Map warehouse tickers to provider-specific source symbols (indices etc.).
SOURCE_SYMBOL_OVERRIDES = {"VIX": "^VIX"}


def source_symbol(ticker: str) -> str:
    return SOURCE_SYMBOL_OVERRIDES.get(ticker, ticker)


def _start_for(db: Database, ticker: str, history_years: int,
               buffer_days: int) -> str:
    """Incremental start date: just before the latest stored bar, or full
    backfill window if the ticker is new."""
    latest = db.max_value("daily_prices", "date", "ticker = ?", (ticker,))
    if latest:
        last_dt = datetime.strptime(latest, "%Y-%m-%d").date()
        return (last_dt - timedelta(days=buffer_days)).isoformat()
    return (date.today() - timedelta(days=int(365.25 * history_years))).isoformat()


def update_ticker(db: Database, provider, ticker: str, history_years: int,
                  buffer_days: int, full: bool) -> int:
    """Fetch and upsert prices for one ticker. Returns net new bars stored."""
    if full:
        start = (date.today() - timedelta(days=int(365.25 * history_years))).isoformat()
    else:
        start = _start_for(db, ticker, history_years, buffer_days)
    end = (date.today() + timedelta(days=1)).isoformat()

    df = provider.get_prices(source_symbol(ticker), start, end)
    if df is None or df.empty:
        return 0

    before = db.count("daily_prices", "ticker = ?", (ticker,))
    rows = []
    for r in df.to_dict("records"):
        close = safe_float(r.get("close"))
        if not r.get("date") or close is None:
            continue  # skip provisional/missing bars (e.g. unsettled index close)
        rows.append({
            "ticker": ticker,
            "date": r["date"],
            "open": safe_float(r.get("open")),
            "high": safe_float(r.get("high")),
            "low": safe_float(r.get("low")),
            "close": close,
            "adj_close": safe_float(r.get("adj_close")),
            "volume": safe_int(r.get("volume")),
            "source": provider.name,
        })
    if not rows:
        return 0
    db.upsert("daily_prices", rows, conflict=["ticker", "date"],
              update=["open", "high", "low", "close", "adj_close", "volume", "source"])
    after = db.count("daily_prices", "ticker = ?", (ticker,))
    return max(0, after - before)


def update_prices(db: Database, tickers: list[str], full: bool = False,
                  registry: ProviderRegistry | None = None) -> dict:
    cfg = load_config()
    registry = registry or ProviderRegistry(cfg)
    provider = registry.prices()
    if provider is None:
        log.error("No price provider available; skipping market data")
        return {"tickers": 0, "bars_added": 0, "failed": []}

    history_years = int(cfg.get("market_data", "history_years", default=4))
    buffer_days = int(cfg.get("market_data", "lookback_buffer_days", default=5))

    bars_added = 0
    failed: list[str] = []
    total = len(tickers)
    for i, ticker in enumerate(tickers, 1):
        try:
            added = update_ticker(db, provider, ticker, history_years,
                                  buffer_days, full)
            bars_added += added
        except Exception as exc:  # noqa: BLE001 - one bad ticker must not stop the run
            log.warning("Price update failed for %s: %s", ticker, exc)
            failed.append(ticker)
        if i % 50 == 0 or i == total:
            log.info("Prices: %d/%d tickers (%d new bars so far)",
                     i, total, bars_added)
    log.info("Market data complete: %d bars added, %d failed", bars_added, len(failed))
    return {"tickers": total, "bars_added": bars_added, "failed": failed}


DEEP_START_DEFAULT = "2014-06-01"   # 12m momentum warm-up before a 2015-06 score
_SPLICE_TOLERANCE = 0.005           # 0.5% close disagreement at the FMP/Polygon seam


def _deep_universe(db: Database) -> list[str]:
    """Every name ever in the PIT window (universe_history) + current universe
    + benchmarks (except VIX, which FMP/Polygon don't serve — FRED does)."""
    hist = [r["ticker"] for r in
            db.query("SELECT DISTINCT ticker FROM universe_history")]
    out = set(hist) | set(db.universe_tickers()) | set(db.benchmark_tickers())
    out.discard("VIX")
    return sorted(out)


def _splice_check(db: Database, ticker: str, df) -> None:
    """Warn if FMP disagrees with an already-stored bar on an overlapping date
    (would indicate an adjustment-convention mismatch at the splice)."""
    overlap = df[df["date"] >= "2022-06-21"].head(3)
    for r in overlap.to_dict("records"):
        row = db.query_one(
            "SELECT close FROM daily_prices WHERE ticker=? AND date=?",
            (ticker, r["date"]))
        if row and row["close"] and r["close"]:
            rel = abs(r["close"] - row["close"]) / row["close"]
            if rel > _SPLICE_TOLERANCE:
                log.warning("Splice mismatch %s @ %s: stored=%.4f fmp=%.4f",
                            ticker, r["date"], row["close"], r["close"])


def deep_backfill(db: Database, start: str = DEEP_START_DEFAULT,
                  tickers: list[str] | None = None,
                  registry: ProviderRegistry | None = None) -> dict:
    """One-time deep price backfill via FMP (plus VIX via FRED).

    Existing bars always win (INSERT OR IGNORE): Polygon stays the daily
    incremental source and FMP only fills the pre-2022 gap and departed
    names. Re-runnable; tickers already covered back to ``start`` are skipped.
    """
    cfg = load_config()
    registry = registry or ProviderRegistry(cfg)
    fmp = registry.transcripts()      # the FMP instance
    if fmp is None or fmp.name != "fmp":
        log.error("FMP provider unavailable; cannot deep-backfill")
        return {"tickers": 0, "bars_added": 0, "failed": []}

    tickers = tickers or _deep_universe(db)
    end = date.today().isoformat()
    bars_added, skipped, failed = 0, 0, []
    for i, ticker in enumerate(tickers, 1):
        try:
            earliest = db.scalar(
                "SELECT MIN(date) FROM daily_prices WHERE ticker=?", (ticker,))
            if earliest and earliest <= start:
                skipped += 1
                continue
            df = fmp.get_prices(source_symbol(ticker), start,
                                earliest or end)
            if df is None or df.empty:
                failed.append(ticker)
                continue
            _splice_check(db, ticker, df)
            rows = [{"ticker": ticker, "date": r["date"], "open": r["open"],
                     "high": r["high"], "low": r["low"], "close": r["close"],
                     "adj_close": r["adj_close"], "volume": r["volume"],
                     "source": "fmp"}
                    for r in df.to_dict("records") if r.get("close") is not None]
            bars_added += db.insert_ignore("daily_prices", rows)
        except Exception as exc:  # noqa: BLE001 - one bad ticker must not stop the run
            log.warning("Deep backfill failed for %s: %s", ticker, exc)
            failed.append(ticker)
        if i % 50 == 0 or i == len(tickers):
            log.info("Deep backfill: %d/%d tickers (%d bars, %d skipped)",
                     i, len(tickers), bars_added, skipped)

    vix_added = _backfill_vix(db, registry, start)
    log.info("Deep backfill complete: %d bars (+%d VIX), %d skipped, %d empty",
             bars_added, vix_added, skipped, len(failed))
    return {"tickers": len(tickers), "bars_added": bars_added,
            "vix_bars": vix_added, "skipped": skipped, "failed": failed}


def _backfill_vix(db: Database, registry: ProviderRegistry, start: str) -> int:
    """VIX closes via FRED VIXCLS (free, full history) into daily_prices."""
    fred = registry.macro()
    if fred is None:
        log.warning("No FRED provider; VIX backfill skipped")
        return 0
    df = fred.get_series("VIXCLS", start=start)
    rows = [{"ticker": "VIX", "date": r["date"], "close": r["value"],
             "adj_close": r["value"], "source": "fred"}
            for r in df.to_dict("records") if r.get("value") is not None]
    return db.insert_ignore("daily_prices", rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Update daily OHLCV market data")
    parser.add_argument("--tickers", nargs="*", help="Specific tickers (default: all)")
    parser.add_argument("--full", action="store_true", help="Force full backfill")
    parser.add_argument("--no-benchmarks", action="store_true",
                        help="Skip benchmark tickers")
    parser.add_argument("--deep", action="store_true",
                        help="One-time deep historical backfill via FMP "
                             "(incl. departed PIT-universe names + VIX via FRED)")
    parser.add_argument("--deep-start", default=DEEP_START_DEFAULT,
                        help=f"Deep backfill start (default {DEEP_START_DEFAULT})")
    args = parser.parse_args()

    cfg = load_config()
    get_logger("market_data", log_file=cfg.log_file)
    with get_db() as db:
        if args.deep:
            tickers = [t.upper() for t in args.tickers] if args.tickers else None
            stats = deep_backfill(db, start=args.deep_start, tickers=tickers)
            print(f"Tickers processed : {stats['tickers']}")
            print(f"Bars added        : {stats['bars_added']} (+{stats.get('vix_bars', 0)} VIX)")
            print(f"Skipped (covered) : {stats.get('skipped', 0)}")
            print(f"No data           : {len(stats['failed'])}")
            if stats["failed"]:
                print(f"  {', '.join(stats['failed'][:40])}"
                      + (" ..." if len(stats["failed"]) > 40 else ""))
            return
        if args.tickers:
            tickers = [t.upper() for t in args.tickers]
        else:
            tickers = db.universe_tickers()
            if not args.no_benchmarks:
                tickers = sorted(set(tickers) | set(db.benchmark_tickers()))
        if not tickers:
            print("No tickers found. Run `python -m data.universe` first.")
            return
        stats = update_prices(db, tickers, full=args.full)
    print(f"Tickers processed : {stats['tickers']}")
    print(f"Bars added        : {stats['bars_added']}")
    print(f"Failed            : {len(stats['failed'])}")


if __name__ == "__main__":
    main()
