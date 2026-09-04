"""Backfill per-share cash-dividend history for the scoring universe.

The `val_dividend_yield` subfactor needs a real dividend series; the raw
``fundamentals.dividends_paid`` field is unreliably populated (~0.6% coverage)
because it depends on the fields FMP exposes in the cash-flow statement, not on
whether the company actually paid a dividend. This script pulls the dedicated
``/stable/dividends`` endpoint (per-share ex-date history, deep) and lands it in
``historical_dividends``.

Non-dividend payers return an empty list and are recorded as a zero-row ticker
so re-runs don't hit FMP again for them (see ``meta`` sentinel).

Idempotent: `INSERT ... ON CONFLICT ex_date DO UPDATE` refreshes the row.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

# Repo root on sys.path so `data.*` imports resolve when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data.db import Database, get_db  # noqa: E402
from data.providers import FMPProvider, ProviderRegistry  # noqa: E402
from data.utils import get_logger  # noqa: E402

log = get_logger("subfactor_expansion.fetch_dividends")

# Sentinel key we drop into ``meta`` for tickers with no dividend history, so
# subsequent runs skip the API round-trip.
_NODIV_META = "hdiv_no_dividends_"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _normalize_row(ticker: str, raw: dict) -> dict | None:
    """Map an FMP `/stable/dividends` row to the historical_dividends schema.

    Skips rows without an ex-date or a numeric dividend value.
    """
    ex_date = raw.get("date") or raw.get("recordDate")
    if not ex_date:
        return None
    dividend = raw.get("dividend")
    adj = raw.get("adjDividend")
    if dividend is None and adj is None:
        return None
    return {
        "ticker": ticker,
        "ex_date": str(ex_date)[:10],
        "payment_date": (raw.get("paymentDate") or None),
        "record_date": (raw.get("recordDate") or None),
        "declaration_date": (raw.get("declarationDate") or None),
        "dividend": _safe_float(dividend),
        "adj_dividend": _safe_float(adj if adj is not None else dividend),
        "source": "fmp",
        "fetched_at": _now_iso(),
    }


def _safe_float(v: object) -> float | None:
    try:
        f = float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return f


def _already_marked_nodiv(db: Database, ticker: str) -> bool:
    key = f"{_NODIV_META}{ticker}"
    row = db.query_one("SELECT value FROM meta WHERE key=?", (key,))
    return row is not None


def _mark_nodiv(db: Database, ticker: str) -> None:
    db.upsert(
        "meta",
        [{"key": f"{_NODIV_META}{ticker}", "value": "1", "updated_at": _now_iso()}],
        conflict=["key"],
    )


def _last_ex_date(db: Database, ticker: str) -> str | None:
    row = db.query_one(
        "SELECT MAX(ex_date) AS d FROM historical_dividends WHERE ticker=?", (ticker,)
    )
    return row["d"] if row and row["d"] else None


def fetch_ticker(fmp: FMPProvider, db: Database, ticker: str,
                 skip_seen: bool = True) -> tuple[int, str]:
    """Return (rows written, status). status ∈ {ok, skipped, nodiv}."""
    if skip_seen and _already_marked_nodiv(db, ticker):
        return 0, "skipped"
    raw_rows = fmp.get_historical_dividends(ticker)
    if not raw_rows:
        _mark_nodiv(db, ticker)
        return 0, "nodiv"
    normalized = [r for r in (_normalize_row(ticker, x) for x in raw_rows) if r]
    if not normalized:
        _mark_nodiv(db, ticker)
        return 0, "nodiv"
    db.upsert("historical_dividends", normalized, conflict=["ticker", "ex_date"])
    return len(normalized), "ok"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tickers", nargs="*", help="Explicit ticker list; default is the full universe.")
    parser.add_argument("--limit", type=int, help="Process only the first N tickers (smoke-test).")
    parser.add_argument("--force", action="store_true",
                        help="Ignore the 'no dividends' sentinel and re-query FMP.")
    args = parser.parse_args()

    db = get_db()
    reg = ProviderRegistry()
    fmp = reg.transcripts()  # resolves the FMPProvider by config priority
    if not isinstance(fmp, FMPProvider):
        log.error("FMP provider unavailable; check FMP_API_KEY in .env.")
        return 2

    tickers = args.tickers or db.universe_tickers()
    if args.limit:
        tickers = tickers[: args.limit]

    counts = {"ok": 0, "nodiv": 0, "skipped": 0, "rows": 0}
    for i, tkr in enumerate(tickers, 1):
        rows, status = fetch_ticker(fmp, db, tkr, skip_seen=not args.force)
        counts[status] += 1
        counts["rows"] += rows
        if i % 25 == 0 or i == len(tickers):
            log.info("progress %d/%d — ok=%d nodiv=%d skipped=%d rows=%d",
                     i, len(tickers), counts["ok"], counts["nodiv"],
                     counts["skipped"], counts["rows"])

    log.info("done — tickers=%d ok=%d nodiv=%d skipped=%d rows=%d",
             len(tickers), counts["ok"], counts["nodiv"],
             counts["skipped"], counts["rows"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
