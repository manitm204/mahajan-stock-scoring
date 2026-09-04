"""Data access — wraps the validated loaders in pairtrading.study."""
from __future__ import annotations

from pairtrading.study import (DB_PATH, load_composites, load_prices,  # noqa: F401
                               load_sectors, members_as_of, score_asof)

# ETFs present in daily_prices (verified 2026-07-18; XLRE from 2015-10,
# XLC from 2018-06, the rest full 2014-06 → 2026-07). ETFs are NOT S&P
# members, so they never appear in members_as_of — always eligible here.
SECTOR_ETF = {
    "Information Technology": "XLK",
    "Financials": "XLF",
    "Health Care": "XLV",
    "Energy": "XLE",
    "Industrials": "XLI",
    "Consumer Discretionary": "XLY",
    "Consumer Staples": "XLP",
    "Utilities": "XLU",
    "Materials": "XLB",
    "Real Estate": "XLRE",
    "Communication Services": "XLC",
}
BROAD_ETFS = ["SPY", "QQQ", "DIA", "IWM", "RSP",          # broad / equal-weight
              "MTUM", "QUAL", "VLUE", "SIZE",             # factor ETFs
              "GLD", "TLT", "HYG", "LQD", "USO"]          # cross-asset
# USO note: Apr-2020 8:1 reverse split + contango restructuring — series is
# split-consistent but its economic behavior changed regime that month.
ALL_ETFS = sorted(set(SECTOR_ETF.values()) | set(BROAD_ETFS))

# dual share classes of the same company (checked against daily_prices at
# discovery time; missing tickers are skipped)
SHARE_CLASS_PAIRS = [("GOOGL", "GOOG"), ("FOXA", "FOX"), ("NWSA", "NWS")]
