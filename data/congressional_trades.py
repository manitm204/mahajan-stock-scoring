"""Congressional trading module.

Stores Senate/House stock-trade disclosures under the STOCK Act, from FMP's
`/stable/senate-trades` and `/stable/house-trades`. Treated like a second,
independent "informed trader" population alongside the insider (Form 4)
parent — cluster buy/sell pressure from politicians rather than corporate
officers.

PIT note: the STOCK Act allows up to 45 days between `transaction_date` and
public disclosure, so every consumer MUST gate on `disclosure_date`, never
`transaction_date` — mirrors the 13F `report_date`-is-not-`filing_date`
lesson already learned in this codebase.

Run standalone:
    python -m data.congressional_trades --tickers AAPL TSLA
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone

from .config import load_config
from .db import Database, get_db
from .providers import ProviderRegistry
from .utils import get_logger

log = get_logger("congressional_trades")

CHAMBERS = ("senate", "house")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _dedup_key(ticker: str, chamber: str, row: dict) -> str:
    """`link` points to the source disclosure document but one filing can
    list several line items, so it alone is not a unique key — compose it
    with the fields that vary between line items on the same disclosure."""
    parts = [ticker, chamber, row.get("senateID") or "", row.get("transactionDate") or "",
             row.get("type") or "", row.get("amount") or "", row.get("link") or ""]
    return "|".join(parts)


def _row_to_record(ticker: str, chamber: str, row: dict) -> dict | None:
    disclosure_date = row.get("disclosureDate")
    transaction_date = row.get("transactionDate")
    if not disclosure_date or not transaction_date:
        return None
    return {
        "ticker": ticker,
        "chamber": chamber,
        "member_id": row.get("senateID"),
        "first_name": row.get("firstName"),
        "last_name": row.get("lastName"),
        "office": row.get("office"),
        "district": row.get("district"),
        "owner": row.get("owner"),
        "asset_description": row.get("assetDescription"),
        "asset_type": row.get("assetType"),
        "transaction_type": row.get("type"),
        "transaction_date": transaction_date,
        "disclosure_date": disclosure_date,
        "amount_range": row.get("amount"),
        "capital_gains_over_200": int(str(row.get("capitalGainsOver200USD")).lower() == "true"),
        "link": row.get("link"),
        "source": "fmp",
        "fetched_at": _now(),
        "dedup_key": _dedup_key(ticker, chamber, row),
    }


def update_congressional_trades(db: Database, tickers: list[str],
                                registry: ProviderRegistry | None = None,
                                max_pages: int = 5) -> dict:
    cfg = load_config()
    registry = registry or ProviderRegistry(cfg)
    provider = registry.congressional_trades()
    if provider is None or not hasattr(provider, "get_congressional_trades"):
        log.warning("No provider for congressional trades; skipping")
        return {"records": 0, "failed": list(tickers)}

    added = 0
    failed: list[str] = []
    for i, ticker in enumerate(tickers, 1):
        try:
            for chamber in CHAMBERS:
                raw = provider.get_congressional_trades(ticker, chamber, max_pages=max_pages)
                rows = [r for r in (_row_to_record(ticker, chamber, row) for row in raw) if r]
                if rows:
                    added += db.insert_ignore("congressional_trades", rows)
        except Exception as exc:  # noqa: BLE001
            log.warning("Congressional trades failed for %s: %s", ticker, exc)
            failed.append(ticker)
        if i % 50 == 0 or i == len(tickers):
            log.info("Congressional trades: %d/%d (%d rows)", i, len(tickers), added)
    return {"records": added, "failed": failed}


def main() -> None:
    parser = argparse.ArgumentParser(description="Update Senate/House trade disclosures")
    parser.add_argument("--tickers", nargs="*")
    parser.add_argument("--max-pages", type=int, default=5)
    args = parser.parse_args()
    cfg = load_config()
    get_logger("congressional_trades", log_file=cfg.log_file)
    with get_db() as db:
        tickers = [t.upper() for t in args.tickers] if args.tickers else db.universe_tickers()
        stats = update_congressional_trades(db, tickers, max_pages=args.max_pages)
    print(f"Rows added : {stats['records']}")
    print(f"Failed     : {len(stats['failed'])}")


if __name__ == "__main__":
    main()
