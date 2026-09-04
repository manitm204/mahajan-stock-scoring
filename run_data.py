"""Mahajan Hedge Fund - Layer 1 (Data Layer) main pipeline.

Orchestrates the full data build in dependency order, logging everything to
output/run.log and printing an execution summary. Every stage is independently
runnable via its own module; this script wires them into one incremental run.

Examples:
    python run_data.py                                  # full universe
    python run_data.py --tickers AAPL MSFT NVDA         # subset
    python run_data.py --no-filings --no-13f            # skip slow EDGAR stages
    python run_data.py --forms 10-K 4 --days 30         # tune SEC scope
    python run_data.py --transcripts-only --tickers AAPL  # candidates only
"""
from __future__ import annotations

import argparse
import time
from datetime import date, datetime, timedelta, timezone

from data.config import load_config
from data.db import get_db
from data.providers import ProviderRegistry
from data.utils import get_logger

from data import (
    earnings_calendar, estimates, features, fundamentals, grades, institutional,
    market_data, quality, sec_data, short_interest, transcripts, universe,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build the Mahajan Hedge Fund Layer 1 data warehouse")
    p.add_argument("--tickers", nargs="*", help="Restrict to specific tickers")
    p.add_argument("--no-filings", action="store_true", help="Skip SEC filings/insiders")
    p.add_argument("--no-13f", action="store_true", help="Skip 13F institutional holdings")
    p.add_argument("--short-interest", action="store_true",
                   help="Fetch short interest (skipped by default: Polygon free "
                        "tier is paced to 5 req/min, ~2.4h for the full "
                        "universe; FINRA data is bi-monthly)")
    p.add_argument("--no-estimates", action="store_true", help="Skip analyst estimates")
    p.add_argument("--no-grades", action="store_true",
                   help="Skip analyst grades / revision backfill (FMP)")
    p.add_argument("--no-earnings-calendar", action="store_true", help="Skip earnings calendar")
    p.add_argument("--no-transcripts", action="store_true", help="Skip transcripts")
    p.add_argument("--no-prices", action="store_true", help="Skip price update")
    p.add_argument("--no-fundamentals", action="store_true", help="Skip fundamentals")
    p.add_argument("--no-features", action="store_true", help="Skip derived features")
    p.add_argument("--transcripts-only", action="store_true",
                   help="Only fetch transcripts for --tickers (candidates)")
    p.add_argument("--forms", nargs="*", help="SEC form types (default from config)")
    p.add_argument("--days", type=int, help="SEC lookback window in days")
    p.add_argument("--insider-backfill-pages", type=int, default=1,
                   help="FMP insider-trading pages per ticker (1000 txns/page). "
                        "Default 1 = incremental daily pull; the deep history "
                        "is already stored — pass 5+ only for a re-backfill.")
    p.add_argument("--full", action="store_true",
                   help="Force full refetch/recompute: price history backfill, "
                        "fundamentals for every ticker, and a full "
                        "price-feature rewrite (use after splits)")
    p.add_argument("--no-benchmarks", action="store_true",
                   help="Exclude benchmark ETFs from the price set")
    return p.parse_args()


# FINRA publishes short interest bi-monthly; past this age a cycle was missed.
_SHORT_INTEREST_STALE_DAYS = 15


def _check_short_interest_freshness(db, log) -> None:
    last = db.scalar("SELECT MAX(fetched_at) FROM short_interest")
    if not last:
        log.warning("Short interest: no data stored; run "
                    "`python run_data.py --short-interest` or "
                    "`python -m data.short_interest`")
        return
    age = (datetime.now(timezone.utc) - datetime.fromisoformat(last)).days
    if age >= _SHORT_INTEREST_STALE_DAYS:
        log.warning("Short interest last fetched %d days ago (FINRA publishes "
                    "bi-monthly); consider `python run_data.py "
                    "--short-interest` (~2.4h at the Polygon free-tier pace)",
                    age)
    else:
        log.info("Short interest skipped (fetched %d days ago; still fresh)", age)


def _hdr(log, stage: str) -> float:
    log.info("=" * 70)
    log.info("STAGE: %s", stage)
    log.info("=" * 70)
    return time.time()


def main() -> None:
    args = parse_args()
    cfg = load_config()
    log = get_logger("pipeline", log_file=cfg.log_file)
    started = datetime.now(timezone.utc)
    run_id = started.strftime("run-%Y%m%dT%H%M%S")
    log.info("###### Mahajan Hedge Fund - Layer 1 data build (%s) ######", run_id)

    summary: dict[str, object] = {
        "tickers_processed": 0, "price_bars_added": 0, "fundamentals_updated": 0,
        "price_feature_rows": 0, "fundamental_feature_rows": 0,
        "short_interest_records": 0, "estimate_snapshots": 0, "revision_rows": 0,
        "earnings_dates_updated": 0, "transcripts_cached": 0,
        "sec_filings_cached": 0, "insider_transactions": 0, "insider_flags": 0,
        "institutional_filings": 0, "institutional_signals": 0,
        "data_quality_warnings": 0, "failed_tickers": set(),
    }

    with get_db() as db:
        registry = ProviderRegistry(cfg)

        # 1. Universe -------------------------------------------------------
        _hdr(log, "Universe & benchmarks")
        universe.refresh_universe(db)

        # 2. Provider initialization ---------------------------------------
        _hdr(log, "Provider initialization")
        for domain, name in registry.describe().items():
            log.info("  provider[%s] = %s", domain, name)

        # Resolve the working ticker set.
        if args.tickers:
            tickers = [t.upper() for t in args.tickers]
        else:
            tickers = db.universe_tickers()
        price_tickers = list(tickers)
        if not args.tickers and not args.no_benchmarks:
            price_tickers = sorted(set(tickers) | set(db.benchmark_tickers()))
        summary["tickers_processed"] = len(tickers)

        # Transcripts-only fast path ---------------------------------------
        if args.transcripts_only:
            _hdr(log, "Transcripts (candidates only)")
            t = transcripts.update_transcripts(db, tickers, registry=registry)
            summary["transcripts_cached"] = t["cached"]
            _print_summary(summary, started)
            return

        # 3. Prices ---------------------------------------------------------
        if not args.no_prices:
            _hdr(log, "Daily prices (OHLCV)")
            r = market_data.update_prices(db, price_tickers, full=args.full,
                                          registry=registry)
            summary["price_bars_added"] = r["bars_added"]
            summary["failed_tickers"].update(r["failed"])

        # 4. Fundamentals ---------------------------------------------------
        if not args.no_fundamentals:
            _hdr(log, "Fundamentals")
            r = fundamentals.update_fundamentals(db, tickers, registry=registry,
                                                 force=args.full)
            summary["fundamentals_updated"] = r["periods"]
            summary["failed_tickers"].update(r["failed"])

        # 5 & 6. Derived features ------------------------------------------
        if not args.no_features:
            _hdr(log, "Derived price features")
            summary["price_feature_rows"] = features.update_price_features(
                db, price_tickers, full=args.full)
            _hdr(log, "Derived fundamental features")
            summary["fundamental_feature_rows"] = features.update_fundamental_features(db, tickers)

        # 7. Short interest (opt-in; see --short-interest help) -------------
        if args.short_interest:
            _hdr(log, "Short interest")
            r = short_interest.update_short_interest(db, tickers, registry=registry)
            summary["short_interest_records"] = r["records"]
            summary["failed_tickers"].update(r["failed"])
        else:
            _check_short_interest_freshness(db, log)

        # 8. Analyst estimates ---------------------------------------------
        if not args.no_estimates:
            _hdr(log, "Analyst estimates")
            r = estimates.update_estimates(db, tickers, registry=registry)
            summary["estimate_snapshots"] = r["records"]
            summary["failed_tickers"].update(r["failed"])

        # 8b. Analyst grades / revisions (FMP) -----------------------------
        if not args.no_grades:
            _hdr(log, "Analyst grades & revisions")
            r = grades.update_grades(db, tickers, registry=registry)
            summary["revision_rows"] = r["records"]
            summary["failed_tickers"].update(r["failed"])

        # 9. Earnings calendar ---------------------------------------------
        if not args.no_earnings_calendar:
            _hdr(log, "Earnings calendar")
            r = earnings_calendar.update_earnings_calendar(db, tickers)
            summary["earnings_dates_updated"] = r["updated"]
            summary["failed_tickers"].update(r["failed"])

        # 10. Transcripts ---------------------------------------------------
        if not args.no_transcripts:
            _hdr(log, "Earnings transcripts")
            # Only candidate tickers when explicitly supplied; never the universe.
            cand = tickers if args.tickers else []
            r = transcripts.update_transcripts(db, cand, registry=registry)
            summary["transcripts_cached"] = r["cached"]

        # 11 & 12. SEC filings + insider transactions ----------------------
        if not args.no_filings:
            _hdr(log, "SEC filings & insider transactions")
            r = sec_data.update_sec(db, tickers, forms=args.forms, days=args.days,
                                    registry=registry,
                                    insider_backfill_pages=args.insider_backfill_pages)
            summary["sec_filings_cached"] = r["filings"]
            summary["insider_transactions"] = r["insider_txns"]
            summary["insider_flags"] = r["flags"]
            summary["failed_tickers"].update(r["failed"])

        # 13. Institutional holdings (13F) ---------------------------------
        # Whole-market FMP ownership summary (deep, universe-wide). The daily run
        # only refreshes the two most recent quarters; the one-time deep history
        # backfill is `python -m data.institutional --source fmp --start 2019-01-01`.
        if not args.no_13f:
            _hdr(log, "Institutional ownership (13F)")
            recent_start = (date.today() - timedelta(days=210)).isoformat()
            r = institutional.fmp_update_ownership(db, tickers, start=recent_start,
                                                   registry=registry)
            summary["institutional_filings"] = r["rows"]
            summary["institutional_signals"] = r["signals"]

        # 14. Data quality validation --------------------------------------
        _hdr(log, "Data quality validation")
        q = quality.run_quality_checks(db, run_id=run_id)
        summary["data_quality_warnings"] = q["total_warnings"]

    _print_summary(summary, started)


def _print_summary(summary: dict, started: datetime) -> None:
    elapsed = (datetime.now(timezone.utc) - started).total_seconds()
    failed = sorted(summary.pop("failed_tickers"))
    log = get_logger("pipeline")
    lines = [
        "", "=" * 60, " EXECUTION SUMMARY ".center(60, "="), "=" * 60,
        f" Tickers processed          : {summary['tickers_processed']}",
        f" Price bars added           : {summary['price_bars_added']}",
        f" Fundamentals updated       : {summary['fundamentals_updated']}",
        f" Price feature rows         : {summary['price_feature_rows']}",
        f" Fundamental feature rows   : {summary['fundamental_feature_rows']}",
        f" Short interest records     : {summary['short_interest_records']}",
        f" Analyst estimate snapshots : {summary['estimate_snapshots']}",
        f" Revision rows (grades/PT)  : {summary['revision_rows']}",
        f" Earnings dates updated     : {summary['earnings_dates_updated']}",
        f" Transcripts cached         : {summary['transcripts_cached']}",
        f" SEC filings cached         : {summary['sec_filings_cached']}",
        f" Insider transactions parsed: {summary['insider_transactions']}",
        f" Insider flags generated    : {summary['insider_flags']}",
        f" 13F holdings parsed        : {summary['institutional_filings']}",
        f" 13F signals computed       : {summary['institutional_signals']}",
        f" Data quality warnings      : {summary['data_quality_warnings']}",
        f" Failed tickers             : {len(failed)}"
        + (f" ({', '.join(failed[:15])}{'...' if len(failed) > 15 else ''})"
           if failed else ""),
        f" Elapsed                    : {elapsed:.1f}s",
        "=" * 60,
    ]
    text = "\n".join(lines)
    log.info("Run complete in %.1fs", elapsed)
    print(text)


if __name__ == "__main__":
    main()
