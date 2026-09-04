"""Fundamentals module — income statement, balance sheet, cash flow.

Pulls annual and quarterly statements via the provider abstraction and
normalizes them into the wide ``fundamentals`` table. Every raw field the
provider returns is preserved in ``raw_json`` so future factors/ML can use
historical data without backfilling. Historical periods are never deleted;
each (ticker, period_type, fiscal_date) row is upserted.

Run standalone:
    python -m data.fundamentals --tickers AAPL MSFT
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import pandas as pd

from .config import load_config
from .db import Database, get_db
from .providers import ProviderRegistry
from .utils import get_logger, safe_float

log = get_logger("fundamentals")

# Map warehouse column -> ordered candidate labels in the provider's frames.
# First label that yields a value wins (handles Yahoo's naming drift).
INCOME_MAP = {
    "revenue": ["Total Revenue", "Operating Revenue", "revenue"],
    "cost_of_revenue": ["Cost Of Revenue", "costOfRevenue"],
    "gross_profit": ["Gross Profit", "grossProfit"],
    "operating_income": ["Operating Income", "operatingIncome"],
    "ebit": ["EBIT", "Operating Income", "ebit"],
    "ebitda": ["EBITDA", "Normalized EBITDA", "ebitda"],
    "net_income": ["Net Income", "Net Income Common Stockholders", "netIncome"],
    "eps_basic": ["Basic EPS", "epsBasic", "eps"],
    "eps_diluted": ["Diluted EPS", "epsDiluted", "epsdiluted"],
    "rnd_expense": ["Research And Development", "researchAndDevelopmentExpenses"],
    "interest_expense": ["Interest Expense", "interestExpense"],
}
BALANCE_MAP = {
    "total_assets": ["Total Assets", "totalAssets"],
    "current_assets": ["Current Assets", "totalCurrentAssets"],
    "cash": ["Cash And Cash Equivalents",
             "Cash Cash Equivalents And Short Term Investments",
             "cashAndCashEquivalents"],
    "total_liabilities": ["Total Liabilities Net Minority Interest",
                          "totalLiabilities"],
    "current_liabilities": ["Current Liabilities", "totalCurrentLiabilities"],
    "debt": ["Total Debt", "totalDebt"],
    "net_debt": ["Net Debt", "netDebt"],
    "working_capital": ["Working Capital"],
    "retained_earnings": ["Retained Earnings", "retainedEarnings"],
    "shareholder_equity": ["Stockholders Equity",
                           "Total Equity Gross Minority Interest",
                           "totalStockholdersEquity"],
    "shares_outstanding": ["Ordinary Shares Number", "Share Issued",
                           "weightedAverageShsOut"],
}
CASHFLOW_MAP = {
    "operating_cash_flow": ["Operating Cash Flow", "operatingCashFlow"],
    "free_cash_flow": ["Free Cash Flow", "freeCashFlow"],
    "capex": ["Capital Expenditure", "capitalExpenditure"],
    "dividends_paid": ["Cash Dividends Paid", "Common Stock Dividend Paid",
                       "dividendsPaid"],
    "buybacks": ["Repurchase Of Capital Stock", "commonStockRepurchased"],
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _pick(df: pd.DataFrame, col, labels: list[str]) -> float | None:
    """Return first present value among ``labels`` for column ``col``."""
    for label in labels:
        if label in df.index:
            val = safe_float(df.loc[label, col])
            if val is not None:
                return val
    return None


def _json_safe(obj) -> object:
    if isinstance(obj, float) and (obj != obj):  # NaN
        return None
    return obj


def _normalize_yahoo(statements: dict, period_type: str) -> list[dict]:
    """Build per-fiscal-date rows from Yahoo statement frames."""
    suffix = "annual" if period_type == "annual" else "quarterly"
    income = statements.get(f"income_{suffix}")
    balance = statements.get(f"balance_{suffix}")
    cashflow = statements.get(f"cashflow_{suffix}")
    frames = {"income": income, "balance": balance, "cashflow": cashflow}

    # Collect the union of period-end dates across the three statements.
    dates = set()
    for df in frames.values():
        if isinstance(df, pd.DataFrame) and not df.empty:
            dates.update(df.columns)
    if not dates:
        return []

    rows = []
    for col in sorted(dates, reverse=True):
        try:
            fiscal_date = pd.to_datetime(col).strftime("%Y-%m-%d")
        except Exception:  # noqa: BLE001
            continue
        row: dict[str, object] = {"fiscal_date": fiscal_date}
        raw: dict[str, object] = {}
        for mapping, df in ((INCOME_MAP, income), (BALANCE_MAP, balance),
                            (CASHFLOW_MAP, cashflow)):
            if not isinstance(df, pd.DataFrame) or df.empty or col not in df.columns:
                continue
            for field, labels in mapping.items():
                if row.get(field) is None:
                    row[field] = _pick(df, col, labels)
            # Preserve every raw line item for this period.
            for label in df.index:
                raw[str(label)] = _json_safe(safe_float(df.loc[label, col]))

        # Derived/cleanup: net_debt fallback = debt - cash.
        if row.get("net_debt") is None and row.get("debt") is not None:
            cash = row.get("cash") or 0
            row["net_debt"] = row["debt"] - cash
        row["raw_json"] = json.dumps(raw)
        rows.append(row)
    return rows


def _normalize_fmp(statements: dict, period_type: str) -> list[dict]:
    """Build rows from FMP statement lists (list[dict] per statement)."""
    suffix = "annual" if period_type == "annual" else "quarter"
    income = statements.get(f"income_{suffix}") or []
    balance = statements.get(f"balance_{suffix}") or []
    cashflow = statements.get(f"cashflow_{suffix}") or []

    def index_by_date(items):
        out = {}
        for it in items if isinstance(items, list) else []:
            if isinstance(it, dict) and it.get("date"):
                out[it["date"][:10]] = it
        return out

    inc, bal, cf = map(index_by_date, (income, balance, cashflow))
    dates = set(inc) | set(bal) | set(cf)
    rows = []
    for fiscal_date in sorted(dates, reverse=True):
        row: dict[str, object] = {"fiscal_date": fiscal_date}
        raw: dict[str, object] = {}
        for mapping, src in ((INCOME_MAP, inc.get(fiscal_date, {})),
                             (BALANCE_MAP, bal.get(fiscal_date, {})),
                             (CASHFLOW_MAP, cf.get(fiscal_date, {}))):
            for field, labels in mapping.items():
                if row.get(field) is None:
                    for label in labels:
                        if label in src:
                            row[field] = safe_float(src[label])
                            break
            raw.update({k: _json_safe(v) for k, v in src.items()})
        if row.get("net_debt") is None and row.get("debt") is not None:
            row["net_debt"] = row["debt"] - (row.get("cash") or 0)
        row["raw_json"] = json.dumps(raw)
        rows.append(row)
    return rows


def update_ticker(db: Database, provider, ticker: str, period_types: list[str],
                  limit: int | None = None) -> int:
    if limit is not None and provider.name == "fmp":
        statements = provider.get_statements(ticker, limit=limit)
    else:
        statements = provider.get_statements(ticker)
    if not statements:
        return 0
    is_yahoo = any(isinstance(v, pd.DataFrame) for v in statements.values())
    currency = None
    info = statements.get("info") or {}
    if isinstance(info, dict):
        currency = info.get("financialCurrency") or info.get("currency")

    stored = 0
    now = _now()
    core = ("revenue", "net_income", "total_assets", "operating_cash_flow",
            "shareholder_equity")
    for period_type in period_types:
        rows = (_normalize_yahoo(statements, period_type) if is_yahoo
                else _normalize_fmp(statements, period_type))
        # Drop empty trailing periods that providers pad with all-null columns.
        rows = [r for r in rows if any(r.get(c) is not None for c in core)]
        if not rows:
            continue
        for row in rows:
            row.update({"ticker": ticker, "period_type": period_type,
                        "currency": currency, "source": provider.name,
                        "fetched_at": now})
        db.upsert("fundamentals", rows,
                  conflict=["ticker", "period_type", "fiscal_date"])
        stored += len(rows)
    return stored


def _stale_tickers(db: Database, tickers: list[str], refresh_days: int) -> list[str]:
    """Tickers whose statements could have changed since the last fetch.

    Statements move quarterly, so a ticker is refetched only when (a) it has
    never been fetched, (b) an earnings date has passed since the last fetch
    (the calendar is populated by earlier runs; earnings dates are known in
    advance), or (c) the last fetch is older than ``refresh_days`` — a safety
    net for names with missing/late calendar coverage.
    """
    today = datetime.now(timezone.utc).date().isoformat()
    last_fetch = {r["ticker"]: str(r["f"])[:10] for r in db.query(
        "SELECT ticker, MAX(fetched_at) AS f FROM fundamentals GROUP BY ticker")}
    last_earn = {r["ticker"]: str(r["e"])[:10] for r in db.query(
        "SELECT ticker, MAX(earnings_date) AS e FROM earnings_calendar "
        "WHERE earnings_date <= ? GROUP BY ticker", (today,))}
    cutoff = (datetime.now(timezone.utc) - pd.Timedelta(days=refresh_days)
              ).date().isoformat()
    out = []
    for t in tickers:
        fetched = last_fetch.get(t)
        if (fetched is None or fetched < cutoff
                or last_earn.get(t, "") > fetched):
            out.append(t)
    return out


def update_fundamentals(db: Database, tickers: list[str],
                        registry: ProviderRegistry | None = None,
                        force: bool = False, limit: int | None = None) -> dict:
    cfg = load_config()
    registry = registry or ProviderRegistry(cfg)
    provider = registry.fundamentals()
    if provider is None:
        log.error("No fundamentals provider available")
        return {"tickers": 0, "periods": 0, "failed": []}
    period_types = cfg.get("fundamentals", "period_types",
                           default=["annual", "quarterly"])

    # Incremental gate: skip tickers with no possible new statement since the
    # last fetch (each fetch costs ~6 provider calls; statements are quarterly).
    if not force:
        refresh_days = int(cfg.get("fundamentals", "refresh_days", default=7))
        work = _stale_tickers(db, tickers, refresh_days)
        log.info("Fundamentals: %d/%d tickers stale (refresh_days=%d, "
                 "rest skipped as fresh)", len(work), len(tickers), refresh_days)
    else:
        work = list(tickers)

    periods = 0
    failed: list[str] = []
    total = len(work)
    for i, ticker in enumerate(work, 1):
        try:
            periods += update_ticker(db, provider, ticker, period_types, limit)
        except Exception as exc:  # noqa: BLE001
            log.warning("Fundamentals failed for %s: %s", ticker, exc)
            failed.append(ticker)
        if i % 25 == 0 or i == total:
            log.info("Fundamentals: %d/%d tickers (%d periods stored)",
                     i, total, periods)
    log.info("Fundamentals complete: %d periods, %d failed", periods, len(failed))
    return {"tickers": total, "periods": periods, "failed": failed}


def main() -> None:
    parser = argparse.ArgumentParser(description="Update fundamentals")
    parser.add_argument("--tickers", nargs="*", help="Specific tickers")
    parser.add_argument("--limit", type=int, default=None,
                        help="Statement periods per fetch (FMP; e.g. 60 for "
                             "the deep-history backfill). Implies --force.")
    parser.add_argument("--force", action="store_true",
                        help="Refetch even if statements look fresh")
    args = parser.parse_args()
    cfg = load_config()
    get_logger("fundamentals", log_file=cfg.log_file)
    with get_db() as db:
        tickers = [t.upper() for t in args.tickers] if args.tickers else db.universe_tickers()
        if not tickers:
            print("No tickers. Run `python -m data.universe` first.")
            return
        stats = update_fundamentals(db, tickers,
                                    force=args.force or args.limit is not None,
                                    limit=args.limit)
    print(f"Tickers processed : {stats['tickers']}")
    print(f"Periods stored    : {stats['periods']}")
    print(f"Failed            : {len(stats['failed'])}")


if __name__ == "__main__":
    main()
