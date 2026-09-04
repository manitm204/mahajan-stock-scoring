"""Earnings transcript module (FMP-gated).

Only activates when an FMP_API_KEY is present. Transcripts are fetched ONLY
for explicitly supplied candidate tickers (long/short candidates, recent
earnings names, or manual requests) -- never for the entire universe. The
schema reserves columns for future NLP features (tone, guidance sentiment,
risk/margin/AI/capex/restructuring mentions) which stay NULL until a later
layer computes them.

Run standalone (requires FMP_API_KEY in .env):
    python -m data.transcripts --tickers AAPL MSFT
    python -m data.transcripts --tickers AAPL --year 2025 --quarter 4
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone

from .config import load_config
from .db import Database, get_db
from .providers import ProviderRegistry
from .utils import get_logger, to_iso_date

log = get_logger("transcripts")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def target_periods(db: Database, ticker: str, limit: int = 4) -> list[tuple[int, int]]:
    """Recent (year, quarter) pairs from past earnings dates for this ticker."""
    rows = db.query(
        "SELECT earnings_date FROM earnings_calendar "
        "WHERE ticker = ? AND earnings_date <= date('now') "
        "ORDER BY earnings_date DESC LIMIT ?", (ticker, limit))
    periods: list[tuple[int, int]] = []
    for r in rows:
        try:
            d = datetime.strptime(r["earnings_date"], "%Y-%m-%d").date()
        except (ValueError, TypeError):
            continue
        periods.append((d.year, ((d.month - 1) // 3) + 1))
    if not periods:
        today = datetime.now().date()
        periods = [(today.year, ((today.month - 1) // 3) + 1)]
    return periods


def _parse_quarter(payload: dict) -> int | None:
    """Quarter from FMP. Legacy gave ``quarter: 1``; /stable gives ``period:
    "Q1"``."""
    q = payload.get("quarter")
    if q is None:
        period = str(payload.get("period") or "").upper().lstrip("Q")
        q = period or None
    try:
        return int(q) if q is not None else None
    except (TypeError, ValueError):
        return None


def store_transcript(db: Database, ticker: str, payload: dict, source: str) -> bool:
    text = payload.get("content") or payload.get("transcript")
    if not text:
        return False
    year = payload.get("year")
    quarter = _parse_quarter(payload)
    if year is None or quarter is None:
        return False
    row = {
        "ticker": ticker, "fiscal_year": int(year), "fiscal_quarter": int(quarter),
        "call_date": to_iso_date(payload.get("date")), "transcript_text": text,
        "source": source, "fetched_at": _now(),
    }
    # NLP feature columns are intentionally left NULL for a future layer.
    db.upsert("transcripts", [row],
              conflict=["ticker", "fiscal_year", "fiscal_quarter"],
              update=["call_date", "transcript_text", "source", "fetched_at"])
    return True


def update_transcripts(db: Database, tickers: list[str], year: int | None = None,
                       quarter: int | None = None, limit: int = 4,
                       registry: ProviderRegistry | None = None) -> dict:
    cfg = load_config()
    registry = registry or ProviderRegistry(cfg)
    provider = registry.transcripts()
    if provider is None:
        log.info("No FMP_API_KEY present; transcript module is inactive (skipped)")
        return {"cached": 0, "skipped": True, "failed": []}
    if not tickers:
        log.info("No candidate tickers supplied; transcripts fetch nothing by design")
        return {"cached": 0, "skipped": False, "failed": []}

    cached = 0
    failed: list[str] = []
    for ticker in tickers:
        periods = ([(year, quarter)] if year and quarter
                   else target_periods(db, ticker, limit))
        for yr, qtr in periods:
            try:
                payload = provider.get_transcript(ticker, yr, qtr)
                if payload and store_transcript(db, ticker, payload, provider.name):
                    cached += 1
            except Exception as exc:  # noqa: BLE001
                log.warning("Transcript failed for %s %sQ%s: %s", ticker, yr, qtr, exc)
                failed.append(f"{ticker}:{yr}Q{qtr}")
    log.info("Transcripts complete: %d cached", cached)
    return {"cached": cached, "skipped": False, "failed": failed}


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch earnings transcripts (FMP)")
    parser.add_argument("--tickers", nargs="*", required=False,
                        help="Candidate tickers (required; never the whole universe)")
    parser.add_argument("--year", type=int)
    parser.add_argument("--quarter", type=int)
    parser.add_argument("--limit", type=int, default=4,
                        help="Recent quarters per ticker when year/quarter omitted")
    args = parser.parse_args()
    cfg = load_config()
    get_logger("transcripts", log_file=cfg.log_file)
    with get_db() as db:
        tickers = [t.upper() for t in (args.tickers or [])]
        stats = update_transcripts(db, tickers, year=args.year, quarter=args.quarter,
                                   limit=args.limit)
    if stats.get("skipped"):
        print("Transcripts inactive (no FMP_API_KEY).")
    else:
        print(f"Transcripts cached : {stats['cached']}")
        print(f"Failed             : {len(stats['failed'])}")


if __name__ == "__main__":
    main()
