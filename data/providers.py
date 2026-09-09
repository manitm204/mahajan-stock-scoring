"""Provider abstraction layer.

Selects the best available data source per domain based on which API keys are
present in ``.env``. Priority order is configured in ``config.yaml``:

    Polygon  -> prices
    FMP      -> structured financials + transcripts
    FRED     -> macro / regime data
    Yahoo    -> universal fallback
    SEC      -> filings

Domain modules ask the registry for a provider and call a normalized method,
so adding a new vendor means writing one class and listing it in config -- no
changes to the consuming modules.
"""
from __future__ import annotations

import time
from datetime import date
from typing import Any, Protocol

import pandas as pd
import requests

from .config import Config, load_config
from .utils import get_logger, retry, safe_float, safe_int

log = get_logger("providers")


# ---------------------------------------------------------------------------
# Interfaces
# ---------------------------------------------------------------------------
class PriceProvider(Protocol):
    name: str

    def get_prices(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        """Return normalized OHLCV: columns date, open, high, low, close,
        adj_close, volume. Empty DataFrame on failure."""


PRICE_COLUMNS = ["date", "open", "high", "low", "close", "adj_close", "volume"]


def _empty_prices() -> pd.DataFrame:
    return pd.DataFrame(columns=PRICE_COLUMNS)


# ---------------------------------------------------------------------------
# yfinance — universal free fallback (prices + fundamentals)
# ---------------------------------------------------------------------------
class YFinanceProvider:
    name = "yfinance"
    requires_key = None

    def get_prices(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        import yfinance as yf

        def _fetch() -> pd.DataFrame:
            t = yf.Ticker(ticker)
            return t.history(start=start, end=end, interval="1d",
                             auto_adjust=False, actions=False)

        raw = retry(_fetch, logger=log, what=f"yfinance prices {ticker}")
        if raw is None or raw.empty:
            return _empty_prices()
        df = raw.reset_index()
        # yfinance returns 'Date' (sometimes tz-aware) and title-case columns.
        df.columns = [str(c).lower().replace(" ", "_") for c in df.columns]
        if "date" not in df.columns and "datetime" in df.columns:
            df = df.rename(columns={"datetime": "date"})
        if "adj_close" not in df.columns:
            df["adj_close"] = df.get("close")
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        for col in PRICE_COLUMNS:
            if col not in df.columns:
                df[col] = None
        return df[PRICE_COLUMNS]

    def get_statements(self, ticker: str) -> dict[str, Any]:
        """Return raw Yahoo statement frames + info dict for normalization."""
        import yfinance as yf

        def _fetch() -> dict[str, Any]:
            t = yf.Ticker(ticker)
            return {
                "income_annual": t.income_stmt,
                "income_quarterly": t.quarterly_income_stmt,
                "balance_annual": t.balance_sheet,
                "balance_quarterly": t.quarterly_balance_sheet,
                "cashflow_annual": t.cashflow,
                "cashflow_quarterly": t.quarterly_cashflow,
                "info": _safe_info(t),
            }

        result = retry(_fetch, logger=log, what=f"yfinance financials {ticker}")
        return result or {}


def _safe_info(ticker_obj: Any) -> dict[str, Any]:
    try:
        return dict(ticker_obj.info)
    except Exception:  # noqa: BLE001 - yfinance .info is flaky
        return {}


# ---------------------------------------------------------------------------
# Polygon — preferred prices (requires POLYGON_API_KEY)
# ---------------------------------------------------------------------------
class PolygonProvider:
    name = "polygon"
    requires_key = "POLYGON_API_KEY"
    BASE = "https://api.polygon.io"
    MIN_REQUEST_INTERVAL = 12.5  # free tier: 5 requests/minute

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key
        self._last_request = 0.0

    def _pace(self) -> None:
        wait = self._last_request + self.MIN_REQUEST_INTERVAL - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        self._last_request = time.monotonic()

    def get_prices(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        url = (f"{self.BASE}/v2/aggs/ticker/{ticker}/range/1/day/"
               f"{start}/{end}")
        params = {"adjusted": "true", "sort": "asc", "limit": 50000,
                  "apiKey": self.api_key}

        def _fetch() -> dict[str, Any]:
            r = requests.get(url, params=params, timeout=30)
            r.raise_for_status()
            return r.json()

        data = retry(_fetch, logger=log, what=f"polygon prices {ticker}")
        if not data or "results" not in data or not data["results"]:
            return _empty_prices()
        rows = []
        for bar in data["results"]:
            d = pd.to_datetime(bar["t"], unit="ms").strftime("%Y-%m-%d")
            close = safe_float(bar.get("c"))
            rows.append({
                "date": d,
                "open": safe_float(bar.get("o")),
                "high": safe_float(bar.get("h")),
                "low": safe_float(bar.get("l")),
                "close": close,
                "adj_close": close,  # polygon aggs are split/dividend adjusted
                "volume": safe_int(bar.get("v")),
            })
        return pd.DataFrame(rows, columns=PRICE_COLUMNS)

    def get_short_interest(self, ticker: str, since: str | None = None,
                           max_pages: int = 20) -> list[dict[str, Any]]:
        """Bi-monthly FINRA short interest history from /stocks/v1/short-interest.

        Returns rows oldest-first with `settlement_date`, `short_interest`,
        `days_to_cover`, `avg_daily_volume`. Cursor-paginates via `next_url`;
        each page is capped at 50000 rows by the API.
        """
        url = f"{self.BASE}/stocks/v1/short-interest"
        params: dict[str, Any] = {"ticker": ticker, "limit": 1000,
                                  "sort": "settlement_date.asc",
                                  "apiKey": self.api_key}
        if since:
            params["settlement_date.gte"] = since
        out: list[dict[str, Any]] = []
        next_url: str | None = None
        for _ in range(max_pages):
            def _fetch(u=next_url, p=params if next_url is None else None):
                self._pace()
                r = requests.get(u or url, params=p, timeout=30)
                r.raise_for_status()
                return r.json()

            data = retry(_fetch, logger=log,
                         what=f"polygon short-interest {ticker}")
            if not data:
                break
            for row in data.get("results", []) or []:
                out.append(row)
            next_url = data.get("next_url")
            if not next_url:
                break
            # Polygon's next_url already encodes the cursor; we just append apiKey.
            if "apiKey=" not in next_url:
                sep = "&" if "?" in next_url else "?"
                next_url = f"{next_url}{sep}apiKey={self.api_key}"
        return out


# ---------------------------------------------------------------------------
# FMP — structured financials + transcripts (requires FMP_API_KEY)
# ---------------------------------------------------------------------------
# FMP retired the legacy /api/v3 path on 2025-08-31; keys now use the /stable
# API, where the ticker is a `symbol` query param rather than part of the path.
_FMP_BLOCKED = object()  # sentinel: endpoint not available on this key's plan
_FMP_PLAN_CODES = {401, 402, 403}


class FMPProvider:
    name = "fmp"
    requires_key = "FMP_API_KEY"

    def __init__(self, api_key: str, base_url: str) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self._blocked: set[str] = set()  # endpoint families already warned about

    def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        params = dict(params or {})
        params["apikey"] = self.api_key
        url = f"{self.base_url}/{path.lstrip('/')}"

        def _fetch() -> Any:
            r = requests.get(url, params=params, timeout=30)
            # Plan/auth errors are permanent — don't retry, warn once per
            # endpoint, and let the caller treat it as "no data".
            if r.status_code in _FMP_PLAN_CODES:
                if path not in self._blocked:
                    self._blocked.add(path)
                    log.warning("FMP %s -> HTTP %d (endpoint not in this key's "
                                "plan); skipping", path, r.status_code)
                return _FMP_BLOCKED
            r.raise_for_status()
            return r.json()

        result = retry(_fetch, logger=log, what=f"fmp {path}")
        return None if result is _FMP_BLOCKED else result

    def get_prices(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        """Daily OHLCV from /stable/historical-price-eod/full.

        Split-adjusted close (verified to match Polygon aggs to the cent at the
        2022 splice), served arbitrarily deep in one call — used for the deep
        historical backfill that Polygon's plan-limited ~5y lookback can't
        reach, including delisted tickers. Same price-return convention as the
        Polygon path: ``adj_close = close`` (no dividend adjustment).
        """
        data = self._get("historical-price-eod/full",
                         {"symbol": ticker, "from": start, "to": end})
        if not isinstance(data, list) or not data:
            return _empty_prices()
        rows = []
        for bar in data:
            close = safe_float(bar.get("close"))
            d = bar.get("date")
            if not d or close is None:
                continue
            rows.append({
                "date": d,
                "open": safe_float(bar.get("open")),
                "high": safe_float(bar.get("high")),
                "low": safe_float(bar.get("low")),
                "close": close,
                "adj_close": close,
                "volume": safe_int(bar.get("volume")),
            })
        rows.sort(key=lambda r: r["date"])
        return pd.DataFrame(rows, columns=PRICE_COLUMNS)

    def get_statements(self, ticker: str, limit: int = 20) -> dict[str, Any]:
        """``limit`` = periods per statement (20 default; the deep-history
        backfill passes ~60 so quarterly coverage reaches 2011, not 2019)."""
        out: dict[str, Any] = {}
        for period in ("annual", "quarter"):
            out[f"income_{period}"] = self._get(
                "income-statement", {"symbol": ticker, "period": period, "limit": limit})
            out[f"balance_{period}"] = self._get(
                "balance-sheet-statement", {"symbol": ticker, "period": period, "limit": limit})
            out[f"cashflow_{period}"] = self._get(
                "cash-flow-statement", {"symbol": ticker, "period": period, "limit": limit})
        return out

    def get_transcript(self, ticker: str, year: int, quarter: int) -> dict[str, Any] | None:
        data = self._get("earning-call-transcript",
                         {"symbol": ticker, "year": year, "quarter": quarter})
        if isinstance(data, list) and data:
            return data[0]
        if isinstance(data, dict) and data:
            return data
        return None

    def get_analyst_grades(self, ticker: str, limit: int = 500) -> list[dict[str, Any]]:
        """Dated monthly history of analyst rating counts (strongBuy..strongSell).

        Returns rows newest-first; empty list if the endpoint is blocked/empty.
        This is genuine point-in-time history, so revision windows are derivable
        from a single fetch (no snapshot accumulation needed). The default limit
        pulls the full monthly history (~7-8 years) so rating-change features can
        be replayed deep into the past, not just the trailing year.
        """
        data = self._get("grades-historical", {"symbol": ticker, "limit": limit})
        return data if isinstance(data, list) else []

    def get_grade_actions(self, ticker: str) -> list[dict[str, Any]]:
        """Individual rating actions from /stable/grades.

        One row per firm action with `gradingCompany`, `previousGrade`,
        `newGrade`, `action` (upgrade/downgrade/initialise/maintain/…) and the
        action `date`. Full history arrives in one call (no pagination
        observed; AAPL ≈ 1.8k rows back to ~2016), so the series is fully
        replayable for point-in-time features — and unlike the monthly
        `grades-historical` counts it carries firm identity.
        """
        data = self._get("grades", {"symbol": ticker})
        return data if isinstance(data, list) else []

    def get_earnings(self, ticker: str, limit: int = 1000) -> list[dict[str, Any]]:
        """Full earnings history + upcoming dates from /stable/earnings.

        Rows are newest-first with ``date`` (announcement date), ``epsActual``/
        ``epsEstimated`` and ``revenueActual``/``revenueEstimated``. Historical
        rows freeze the consensus estimate as of the report, so surprises are
        replayable deep into the past (~25y for large caps vs yfinance's ~24
        rows); future rows carry estimates with null actuals.
        """
        data = self._get("earnings", {"symbol": ticker, "limit": limit})
        return data if isinstance(data, list) else []

    def get_institutional_ownership(self, ticker: str, year: int,
                                    quarter: int) -> dict[str, Any] | None:
        """Whole-market 13F ownership summary for one symbol/quarter (FMP Ultimate).

        Aggregates every 13F filer into per-symbol breadth: `investorsHolding`
        (total holders), `newPositions`/`increasedPositions`/`reducedPositions`/
        `closedPositions`, `numberOf13Fshares` (+change), `ownershipPercent`
        (+change) and `putCallRatio`. Available quarterly back to ~2014; returns
        None if the quarter is missing/blocked. Requires both year and quarter.
        """
        data = self._get("institutional-ownership/symbol-positions-summary",
                         {"symbol": ticker, "year": year, "quarter": quarter})
        if isinstance(data, list) and data:
            return data[0]
        if isinstance(data, dict) and data:
            return data
        return None

    def get_analyst_estimates(self, ticker: str, period: str = "annual",
                              limit: int = 40) -> list[dict[str, Any]]:
        """Consensus forward estimates per fiscal period (FMP Ultimate).

        Each row carries `date` (fiscal period end), `revenueAvg`, `epsAvg`,
        `ebitdaAvg`, plus low/high bounds and `numAnalystsEps`/`numAnalystsRevenue`.
        This is the *current* consensus for each forward period (not a dated
        revision series), so revision momentum is accrued by snapshotting daily.
        """
        data = self._get("analyst-estimates",
                         {"symbol": ticker, "period": period, "limit": limit})
        return data if isinstance(data, list) else []

    def get_price_target_summary(self, ticker: str) -> dict[str, Any] | None:
        """Consensus price-target averages bucketed by lastMonth/Quarter/Year."""
        data = self._get("price-target-summary", {"symbol": ticker})
        if isinstance(data, list) and data:
            return data[0]
        if isinstance(data, dict) and data:
            return data
        return None

    def get_price_target_news(self, ticker: str, max_pages: int = 5,
                              limit: int = 100) -> list[dict[str, Any]]:
        """Dated per-analyst price-target events from /stable/price-target-news.

        Each row carries `publishedDate`, `analystCompany`, `analystName`,
        `priceTarget`, `adjPriceTarget`, `priceWhenPosted` — a genuine
        revision-momentum time series, unlike the snapshot-only Yahoo path.
        Pages are 0-indexed.
        """
        out: list[dict[str, Any]] = []
        for page in range(max_pages):
            data = self._get("price-target-news",
                             {"symbol": ticker, "page": page, "limit": limit})
            if not isinstance(data, list) or not data:
                break
            out.extend(data)
            if len(data) < limit:
                break
        return out

    def get_historical_dividends(self, ticker: str) -> list[dict[str, Any]]:
        """Full per-share cash-dividend history from /stable/dividends.

        One row per ex-date with `date` (ex_date), `adjDividend` (split-adjusted),
        `dividend` (unadjusted), plus payment/record/declaration dates. Ordering
        is newest-first; empty list if the ticker never paid a dividend or the
        endpoint is blocked. Non-dividend payers (~50% of the S&P 500) simply
        return [] — the caller records "no dividend" not "missing".
        """
        data = self._get("dividends", {"symbol": ticker})
        return data if isinstance(data, list) else []

    def get_insider_trades(self, ticker: str, max_pages: int = 5,
                           limit: int = 1000) -> list[dict[str, Any]]:
        """Paginated Form-4 history from /stable/insider-trading/search.

        Goes back multiple years per ticker, far deeper than the 90-day
        EDGAR default. Each row is a parsed insider transaction with
        `transactionDate`, `transactionType`, `securitiesTransacted`, `price`,
        `reportingName`, `typeOfOwner`, `acquisitionOrDisposition`, etc.
        """
        out: list[dict[str, Any]] = []
        for page in range(max_pages):
            data = self._get("insider-trading/search",
                             {"symbol": ticker, "page": page, "limit": limit})
            if not isinstance(data, list) or not data:
                break
            out.extend(data)
            if len(data) < limit:
                break
        return out

    def get_beneficial_ownership(self, ticker: str) -> list[dict[str, Any]]:
        """13D/13G beneficial-ownership filings from /stable/acquisition-of-beneficial-ownership.

        One row per reporting-person filing: `filingDate`/`acceptedDate` (the
        real disclosure date — filed within 10 days of crossing 5% ownership,
        no additional PIT lag needed), `nameOfReportingPerson`,
        `amountBeneficiallyOwned`, `percentOfClass`, `typeOfReportingPerson`.
        Full history arrives in one call (AAPL back to 1998; no pagination).
        """
        data = self._get("acquisition-of-beneficial-ownership", {"symbol": ticker})
        return data if isinstance(data, list) else []

    def get_congressional_trades(self, ticker: str, chamber: str,
                                 max_pages: int = 5) -> list[dict[str, Any]]:
        """Senate/House stock-trade disclosures for one symbol.

        `chamber` is ``"senate"`` or ``"house"`` -> `/stable/senate-trades` or
        `/stable/house-trades`. Each row carries `transactionDate` (when the
        trade happened) and `disclosureDate` (when it became public under the
        STOCK Act, up to 45 days later) — PIT gating must use `disclosureDate`,
        never `transactionDate`. `amount` is a disclosed dollar range, not an
        exact figure.
        """
        path = f"{chamber}-trades"
        out: list[dict[str, Any]] = []
        for page in range(max_pages):
            data = self._get(path, {"symbol": ticker, "page": page})
            if not isinstance(data, list) or not data:
                break
            out.extend(data)
            if len(data) < 100:
                break
        return out


# ---------------------------------------------------------------------------
# FRED — macro / regime series (requires FRED_API_KEY)
# ---------------------------------------------------------------------------
class FREDProvider:
    name = "fred"
    requires_key = "FRED_API_KEY"
    BASE = "https://api.stlouisfed.org/fred"

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key

    def get_series(self, series_id: str, start: str | None = None) -> pd.DataFrame:
        params = {"series_id": series_id, "api_key": self.api_key,
                  "file_type": "json"}
        if start:
            params["observation_start"] = start

        def _fetch() -> dict[str, Any]:
            r = requests.get(f"{self.BASE}/series/observations",
                            params=params, timeout=30)
            r.raise_for_status()
            return r.json()

        data = retry(_fetch, logger=log, what=f"fred {series_id}")
        if not data or "observations" not in data:
            return pd.DataFrame(columns=["date", "value"])
        rows = [{"date": o["date"], "value": safe_float(o.get("value"))}
                for o in data["observations"]]
        return pd.DataFrame(rows, columns=["date", "value"])


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
_PROVIDER_CLASSES = {
    "yfinance": YFinanceProvider,
    "polygon": PolygonProvider,
    "fmp": FMPProvider,
    "fred": FREDProvider,
    "sec": None,  # filings handled directly by data/sec_data.py
}


class ProviderRegistry:
    """Resolves and caches the active provider per domain, logging choices."""

    def __init__(self, cfg: Config | None = None) -> None:
        self.cfg = cfg or load_config()
        self._cache: dict[str, Any] = {}
        self._logged: set[str] = set()

    def _instantiate(self, name: str) -> Any | None:
        cls = _PROVIDER_CLASSES.get(name)
        if cls is None:
            return None
        if cls is YFinanceProvider:
            return YFinanceProvider()
        if cls is PolygonProvider:
            key = self.cfg.env("POLYGON_API_KEY")
            return PolygonProvider(key) if key else None
        if cls is FMPProvider:
            key = self.cfg.env("FMP_API_KEY")
            base = self.cfg.get("transcripts", "fmp_base_url",
                                default="https://financialmodelingprep.com/stable")
            return FMPProvider(key, base) if key else None
        if cls is FREDProvider:
            key = self.cfg.env("FRED_API_KEY")
            return FREDProvider(key) if key else None
        return None

    def resolve(self, domain: str) -> Any | None:
        if domain in self._cache:
            return self._cache[domain]
        priority = self.cfg.get("providers", domain, default=[]) or []
        for name in priority:
            provider = self._instantiate(name)
            if provider is not None:
                self._cache[domain] = provider
                if domain not in self._logged:
                    log.info("Using %s for %s", provider.name, domain)
                    self._logged.add(domain)
                return provider
        log.warning("No provider available for %s (priority=%s)", domain, priority)
        self._cache[domain] = None
        return None

    # Convenience accessors --------------------------------------------------
    def prices(self) -> PriceProvider | None:
        return self.resolve("prices")

    def fundamentals(self) -> Any | None:
        return self.resolve("fundamentals")

    def transcripts(self) -> FMPProvider | None:
        return self.resolve("transcripts")

    def macro(self) -> FREDProvider | None:
        return self.resolve("macro")

    def short_interest(self) -> Any | None:
        return self.resolve("short_interest")

    def insiders(self) -> Any | None:
        return self.resolve("insiders")

    def earnings(self) -> Any | None:
        return self.resolve("earnings")

    def beneficial_ownership(self) -> Any | None:
        return self.resolve("beneficial_ownership")

    def congressional_trades(self) -> Any | None:
        return self.resolve("congressional_trades")

    def describe(self) -> dict[str, str]:
        """Resolve every domain once and return a name map (for run summary)."""
        summary = {}
        for domain in ("prices", "fundamentals", "transcripts", "macro",
                       "filings", "short_interest", "insiders", "earnings",
                       "beneficial_ownership", "congressional_trades"):
            if domain == "filings":
                summary[domain] = "sec"
                continue
            p = self.resolve(domain)
            summary[domain] = p.name if p else "none"
        return summary


if __name__ == "__main__":
    reg = ProviderRegistry()
    print("Resolved providers:")
    for domain, name in reg.describe().items():
        print(f"  {domain:14s}: {name}")
