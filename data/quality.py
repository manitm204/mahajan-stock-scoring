"""Data quality validation.

Runs non-fatal integrity checks over the warehouse and records warnings in
``data_quality_log`` (and the run log). Checks never raise -- they surface
problems so the pipeline can keep building history. Designed to be extended
with additional checks as new modules land.

Run standalone:
    python -m data.quality
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone

from .config import load_config
from .db import Database, get_db
from .utils import get_logger

log = get_logger("quality")


def _run_id() -> str:
    return datetime.now(timezone.utc).strftime("run-%Y%m%dT%H%M%S")


def _emit(db: Database, run_id: str, check: str, msg: str,
          ticker: str | None = None, severity: str = "warning") -> None:
    db.log_quality(run_id, check, msg, ticker=ticker, severity=severity)
    log.warning("[%s] %s%s", check, f"{ticker}: " if ticker else "", msg)


def check_missing_ohlcv(db: Database, run_id: str) -> int:
    """Rows with a close but a missing OHLC component or non-positive price."""
    n = db.count("daily_prices",
                 "close IS NOT NULL AND (open IS NULL OR high IS NULL OR "
                 "low IS NULL OR close <= 0)")
    if n:
        _emit(db, run_id, "missing_ohlcv", f"{n} price bars with missing/invalid OHLC")
    return n


def check_negative_shares(db: Database, run_id: str) -> int:
    n = db.count("fundamentals", "shares_outstanding IS NOT NULL AND shares_outstanding < 0")
    n += db.count("institutional_holdings", "shares_held < 0")
    if n:
        _emit(db, run_id, "negative_shares", f"{n} rows with negative share counts")
    return n


def check_duplicate_filings(db: Database, run_id: str) -> int:
    """Same accession appearing under multiple tickers (unexpected)."""
    rows = db.query(
        "SELECT accession_number, COUNT(DISTINCT ticker) c FROM sec_filings "
        "WHERE accession_number IS NOT NULL GROUP BY accession_number HAVING c > 1")
    if rows:
        _emit(db, run_id, "duplicate_filings",
              f"{len(rows)} accession numbers span multiple tickers")
    return len(rows)


def check_duplicate_insiders(db: Database, run_id: str) -> int:
    rows = db.query(
        "SELECT accession_number, insider_name, transaction_date, transaction_code, "
        "shares, price, COUNT(*) c FROM insider_transactions "
        "GROUP BY accession_number, insider_name, transaction_date, transaction_code, "
        "shares, price HAVING c > 1")
    if rows:
        _emit(db, run_id, "duplicate_insiders",
              f"{len(rows)} duplicated insider transactions")
    return len(rows)


def check_invalid_dates(db: Database, run_id: str) -> int:
    total = 0
    for table, col in (("daily_prices", "date"), ("fundamentals", "fiscal_date"),
                       ("sec_filings", "filing_date"),
                       ("institutional_holdings", "report_date")):
        n = db.count(table, f"{col} IS NOT NULL AND {col} NOT GLOB "
                            "'[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]'")
        if n:
            _emit(db, run_id, "invalid_dates", f"{n} malformed dates in {table}.{col}")
        total += n
    return total


def check_missing_fundamentals(db: Database, run_id: str) -> int:
    """Universe names with prices but no fundamentals."""
    rows = db.query(
        "SELECT u.ticker FROM universe u WHERE u.active = 1 "
        "AND EXISTS (SELECT 1 FROM daily_prices p WHERE p.ticker = u.ticker) "
        "AND NOT EXISTS (SELECT 1 FROM fundamentals f WHERE f.ticker = u.ticker)")
    if rows:
        sample = ", ".join(r["ticker"] for r in rows[:10])
        _emit(db, run_id, "missing_fundamentals",
              f"{len(rows)} priced names lack fundamentals (e.g. {sample})")
    return len(rows)


def check_stale_prices(db: Database, run_id: str) -> int:
    """Active names whose latest bar is far behind the warehouse max date."""
    market_max = db.max_value("daily_prices", "date")
    if not market_max:
        return 0
    rows = db.query(
        "SELECT u.ticker, MAX(p.date) last FROM universe u "
        "JOIN daily_prices p ON p.ticker = u.ticker WHERE u.active = 1 "
        "GROUP BY u.ticker HAVING last < date(?, '-10 day')", (market_max,))
    if rows:
        _emit(db, run_id, "stale_prices",
              f"{len(rows)} active names have prices >10d behind market")
    return len(rows)


CHECKS = [
    check_missing_ohlcv, check_negative_shares, check_duplicate_filings,
    check_duplicate_insiders, check_invalid_dates, check_missing_fundamentals,
    check_stale_prices,
]


def run_quality_checks(db: Database, run_id: str | None = None) -> dict:
    run_id = run_id or _run_id()
    results: dict[str, int] = {}
    for check in CHECKS:
        try:
            n = check(db, run_id)
        except Exception as exc:  # noqa: BLE001 - a broken check must not crash the run
            log.error("Quality check %s errored: %s", check.__name__, exc)
            n = 0
        results[check.__name__] = n
    results["total_warnings"] = sum(results.values())
    log.info("Data quality: %d warning(s) across %d checks",
             results["total_warnings"], len(CHECKS))
    return results


def main() -> None:
    cfg = load_config()
    get_logger("quality", log_file=cfg.log_file)
    argparse.ArgumentParser(description="Run data quality checks").parse_args()
    with get_db() as db:
        results = run_quality_checks(db)
    for name, n in results.items():
        print(f"  {name:26s}: {n}")


if __name__ == "__main__":
    main()
