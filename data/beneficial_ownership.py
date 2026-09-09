"""Beneficial ownership (13D/13G) module.

Stores Schedule 13D/13G activist/institutional-stake filings from FMP's
`/stable/acquisition-of-beneficial-ownership`. A new 13D from an activist
investor crossing the 5% ownership threshold is one of the fastest-arriving
"informed trader" signals available — filed within 10 days of the crossing,
unlike the quarterly-cadence 13F institutional summary already in the stack.

PIT note: `filing_date`/`accepted_date` (identical in practice) already ARE
the real public-disclosure date, so no additional availability lag is needed
on top of them (unlike short interest's FINRA dissemination lag, or 13F's
regulatory-deadline lag).

Run standalone:
    python -m data.beneficial_ownership --tickers AAPL TSLA
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone

from .config import load_config
from .db import Database, get_db
from .providers import ProviderRegistry
from .utils import get_logger, safe_float

log = get_logger("beneficial_ownership")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _dedup_key(ticker: str, row: dict) -> str:
    """`url` is the SEC filing document link — unique per reporting-person
    filing. Falls back to a composite key on the rare row missing one."""
    url = row.get("url")
    if url:
        return f"{ticker}|{url}"
    parts = [ticker, row.get("filingDate") or "", row.get("nameOfReportingPerson") or "",
             row.get("percentOfClass") or ""]
    return "|".join(parts)


def _filing_type(url: str | None) -> str | None:
    """Parse '13D'/'13G' out of the SEC filing-viewer URL.

    Only the post-~2019 XSL-viewer URL format embeds the schedule type (e.g.
    ``.../xslSCHEDULE_13G_X02/primary_doc.xml``); older filings use a plain
    accession filename with no schedule marker and return None here. Callers
    that need a clean 13D-only sample (real activist stakes, not routine 13G
    passive crossings) must filter on this column and accept the resulting
    survivorship to recent years.
    """
    if not url:
        return None
    u = url.upper()
    if "13D" in u:
        return "13D"
    if "13G" in u:
        return "13G"
    return None


def _row_to_record(ticker: str, row: dict) -> dict | None:
    filing_date = row.get("filingDate")
    if not filing_date:
        return None
    return {
        "ticker": ticker,
        "cik": str(row.get("cik") or ""),
        "filing_date": filing_date,
        "accepted_date": row.get("acceptedDate"),
        "cusip": row.get("cusip"),
        "reporting_person": row.get("nameOfReportingPerson"),
        "citizenship": row.get("citizenshipOrPlaceOfOrganization"),
        "sole_voting_power": safe_float(row.get("soleVotingPower")),
        "shared_voting_power": safe_float(row.get("sharedVotingPower")),
        "sole_dispositive_power": safe_float(row.get("soleDispositivePower")),
        "shared_dispositive_power": safe_float(row.get("sharedDispositivePower")),
        "amount_beneficially_owned": safe_float(row.get("amountBeneficiallyOwned")),
        "percent_of_class": safe_float(row.get("percentOfClass")),
        "reporting_person_type": row.get("typeOfReportingPerson"),
        "filing_type": _filing_type(row.get("url")),
        "url": row.get("url"),
        "source": "fmp",
        "fetched_at": _now(),
        "dedup_key": _dedup_key(ticker, row),
    }


def update_beneficial_ownership(db: Database, tickers: list[str],
                                registry: ProviderRegistry | None = None) -> dict:
    cfg = load_config()
    registry = registry or ProviderRegistry(cfg)
    provider = registry.beneficial_ownership()
    if provider is None or not hasattr(provider, "get_beneficial_ownership"):
        log.warning("No provider for beneficial ownership; skipping")
        return {"records": 0, "failed": list(tickers)}

    added = 0
    failed: list[str] = []
    for i, ticker in enumerate(tickers, 1):
        try:
            raw = provider.get_beneficial_ownership(ticker)
            rows = [r for r in (_row_to_record(ticker, row) for row in raw) if r]
            if rows:
                added += db.insert_ignore("beneficial_ownership", rows)
        except Exception as exc:  # noqa: BLE001
            log.warning("Beneficial ownership failed for %s: %s", ticker, exc)
            failed.append(ticker)
        if i % 50 == 0 or i == len(tickers):
            log.info("Beneficial ownership: %d/%d (%d rows)", i, len(tickers), added)
    return {"records": added, "failed": failed}


def main() -> None:
    parser = argparse.ArgumentParser(description="Update 13D/13G beneficial ownership")
    parser.add_argument("--tickers", nargs="*")
    args = parser.parse_args()
    cfg = load_config()
    get_logger("beneficial_ownership", log_file=cfg.log_file)
    with get_db() as db:
        tickers = [t.upper() for t in args.tickers] if args.tickers else db.universe_tickers()
        stats = update_beneficial_ownership(db, tickers)
    print(f"Rows added : {stats['records']}")
    print(f"Failed     : {len(stats['failed'])}")


if __name__ == "__main__":
    main()
