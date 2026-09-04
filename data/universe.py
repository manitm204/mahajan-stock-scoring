"""Universe & benchmark module.

Scrapes the S&P 500 constituent list from Wikipedia and maintains a curated
set of benchmark tickers (market, sector, factor, regime, breadth, alt). The
constituent list is cached locally and refreshed weekly; benchmarks are
static metadata seeded on every run.

Run standalone:
    python -m data.universe            # refresh if stale
    python -m data.universe --force    # force re-scrape
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone

import pandas as pd
import requests
from bs4 import BeautifulSoup

from .config import load_config
from .db import Database, get_db
from .utils import get_logger, retry

log = get_logger("universe")

# Canonical GICS sector labels (2026-08-02 fix). Two ingestion paths write
# ``universe.gics_sector``: the Wikipedia S&P scrape (GICS labels) and FMP
# company profiles for departed members (FMP's own taxonomy). Mixing the two
# splits real sectors in half — a "Healthcare" ticker never competes against
# its "Health Care" peers in ``factors.utils.sector_percentile`` — so every
# write path funnels through :func:`normalize_sector`.
SECTOR_NORMALIZE: dict[str, str] = {
    "Healthcare": "Health Care",
    "Consumer Cyclical": "Consumer Discretionary",
    "Consumer Defensive": "Consumer Staples",
    "Financial Services": "Financials",
    "Basic Materials": "Materials",
    "Technology": "Information Technology",
}


def normalize_sector(sector: str | None) -> str | None:
    """Map vendor sector labels onto the canonical GICS set."""
    if not sector:
        return sector
    return SECTOR_NORMALIZE.get(sector.strip(), sector.strip())

# Benchmark universe: ticker -> (display name, category). Stored in `benchmarks`.
BENCHMARKS: dict[str, tuple[str, str]] = {
    # Market benchmarks
    "SPY": ("SPDR S&P 500 ETF", "market"),
    "QQQ": ("Invesco QQQ (Nasdaq 100)", "market"),
    "IWM": ("iShares Russell 2000", "market"),
    "DIA": ("SPDR Dow Jones Industrial Average", "market"),
    # Sector ETFs
    "XLK": ("Technology Select Sector", "sector"),
    "XLF": ("Financials Select Sector", "sector"),
    "XLV": ("Health Care Select Sector", "sector"),
    "XLE": ("Energy Select Sector", "sector"),
    "XLI": ("Industrials Select Sector", "sector"),
    "XLC": ("Communication Services Select Sector", "sector"),
    "XLY": ("Consumer Discretionary Select Sector", "sector"),
    "XLP": ("Consumer Staples Select Sector", "sector"),
    "XLB": ("Materials Select Sector", "sector"),
    "XLRE": ("Real Estate Select Sector", "sector"),
    "XLU": ("Utilities Select Sector", "sector"),
    # Factor benchmarks
    "MTUM": ("iShares MSCI USA Momentum Factor", "factor"),
    "QUAL": ("iShares MSCI USA Quality Factor", "factor"),
    "VLUE": ("iShares MSCI USA Value Factor", "factor"),
    "SIZE": ("iShares MSCI USA Size Factor", "factor"),
    # Regime indicators
    "VIX": ("CBOE Volatility Index", "regime"),
    "TLT": ("iShares 20+ Year Treasury Bond", "regime"),
    "HYG": ("iShares iBoxx High Yield Corporate Bond", "regime"),
    "LQD": ("iShares iBoxx Investment Grade Corporate Bond", "regime"),
    # Breadth
    "RSP": ("Invesco S&P 500 Equal Weight", "breadth"),
    # Alternative assets
    "GLD": ("SPDR Gold Shares", "alt"),
    "USO": ("United States Oil Fund", "alt"),
}

UNIVERSE_REFRESH_META_KEY = "universe_last_refresh"
_WIKI_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (compatible; MahajanHedgeFund/1.0; "
                   "research; manitm204@gmail.com)")
}


def normalize_ticker(symbol: str) -> str:
    """Normalize Wikipedia symbols to the dash form used by yfinance/EDGAR.

    e.g. ``BRK.B`` -> ``BRK-B``. Also strips footnote whitespace.
    """
    return symbol.strip().upper().replace(".", "-")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _is_stale(db: Database, refresh_days: int) -> bool:
    last = db.get_meta(UNIVERSE_REFRESH_META_KEY)
    if not last:
        return True
    try:
        last_dt = datetime.fromisoformat(last)
    except ValueError:
        return True
    age_days = (datetime.now(timezone.utc) - last_dt).total_seconds() / 86400
    return age_days >= refresh_days


def scrape_sp500(url: str) -> pd.DataFrame:
    """Scrape the S&P 500 constituent table from Wikipedia.

    Returns a DataFrame with columns: ticker, company_name, gics_sector,
    gics_sub_industry, date_added. Empty DataFrame on failure.
    """
    def _fetch() -> str:
        r = requests.get(url, headers=_WIKI_HEADERS, timeout=30)
        r.raise_for_status()
        return r.text

    html = retry(_fetch, logger=log, what="scrape S&P 500 from Wikipedia")
    if not html:
        return pd.DataFrame()

    soup = BeautifulSoup(html, "lxml")
    table = soup.find("table", {"id": "constituents"}) or soup.find("table",
                                                                    class_="wikitable")
    if table is None:
        log.error("Could not locate S&P 500 constituents table on the page")
        return pd.DataFrame()

    rows = []
    body_rows = table.find_all("tr")[1:]  # skip header
    for tr in body_rows:
        cells = tr.find_all(["td", "th"])
        if len(cells) < 4:
            continue
        text = [c.get_text(strip=True) for c in cells]
        rows.append({
            "ticker": normalize_ticker(text[0]),
            "company_name": text[1],
            "gics_sector": text[2],
            "gics_sub_industry": text[3],
            "date_added": text[5] if len(text) > 5 else None,
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df[df["ticker"].str.len() > 0].drop_duplicates(subset="ticker")
    return df


def store_universe(db: Database, df: pd.DataFrame) -> int:
    now = _now()
    existing = set(db.universe_tickers(active_only=False))
    rows = []
    for r in df.to_dict("records"):
        rows.append({
            "ticker": r["ticker"],
            "company_name": r.get("company_name"),
            "gics_sector": normalize_sector(r.get("gics_sector")),
            "gics_sub_industry": r.get("gics_sub_industry"),
            "date_added": r.get("date_added"),
            "source": "wikipedia",
            "active": 1,
            "first_seen": now if r["ticker"] not in existing else None,
            "last_updated": now,
        })
    # Don't clobber first_seen for rows we've seen before.
    db.upsert(
        "universe", rows, conflict=["ticker"],
        update=["company_name", "gics_sector", "gics_sub_industry",
                "date_added", "source", "active", "last_updated"],
    )
    # Deactivate names that dropped out of the index (kept for history).
    current = set(df["ticker"])
    dropped = existing - current
    for ticker in dropped:
        db.upsert("universe",
                  [{"ticker": ticker, "active": 0, "last_updated": now}],
                  conflict=["ticker"], update=["active", "last_updated"])
    if dropped:
        log.info("Deactivated %d names that left the index", len(dropped))
    return len(rows)


def store_benchmarks(db: Database) -> int:
    now = _now()
    rows = [{"ticker": t, "name": name, "category": cat, "last_updated": now}
            for t, (name, cat) in BENCHMARKS.items()]
    db.upsert("benchmarks", rows, conflict=["ticker"],
              update=["name", "category", "last_updated"])
    return len(rows)


def refresh_universe(db: Database, force: bool = False) -> dict[str, int]:
    """Refresh the S&P 500 universe (if stale) and reseed benchmarks."""
    cfg = load_config()
    url = cfg.get("universe", "sp500_url")
    refresh_days = int(cfg.get("universe", "refresh_days", default=7))
    cache_file = cfg.path("universe", "cache_file",
                          default="cache/sp500_constituents.csv")

    n_bench = store_benchmarks(db)
    stats = {"benchmarks": n_bench, "constituents": 0, "scraped": 0}

    if not force and not _is_stale(db, refresh_days):
        existing = db.count("universe", "active = 1")
        log.info("Universe is fresh (%d active names); skipping scrape", existing)
        stats["constituents"] = existing
        return stats

    df = scrape_sp500(url)
    if df.empty:
        # Fall back to last good cache if the scrape failed.
        if cache_file.exists():
            log.warning("Scrape failed; loading cached constituents from %s", cache_file)
            df = pd.read_csv(cache_file, dtype=str)
        else:
            log.error("Scrape failed and no cache available; universe unchanged")
            stats["constituents"] = db.count("universe", "active = 1")
            return stats
    else:
        df.to_csv(cache_file, index=False)
        log.info("Cached %d constituents to %s", len(df), cache_file)

    stats["scraped"] = len(df)
    store_universe(db, df)
    db.set_meta(UNIVERSE_REFRESH_META_KEY, _now())
    stats["constituents"] = db.count("universe", "active = 1")
    log.info("Universe refreshed: %d active constituents, %d benchmarks",
             stats["constituents"], n_bench)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh S&P 500 universe & benchmarks")
    parser.add_argument("--force", action="store_true",
                        help="Force re-scrape even if cache is fresh")
    args = parser.parse_args()
    cfg = load_config()
    get_logger("universe", log_file=cfg.log_file)
    with get_db() as db:
        stats = refresh_universe(db, force=args.force)
    print(f"Constituents (active): {stats['constituents']}")
    print(f"Benchmarks            : {stats['benchmarks']}")


if __name__ == "__main__":
    main()
