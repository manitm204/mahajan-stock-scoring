"""Derived feature generation.

Computes and stores derived price features and fundamental features in
dedicated tables so research/factor code never recomputes them. Features are
computed over full history (a panel) so future ML/factor models can train on
point-in-time values.

Price features:    20/60/252-day returns, 20-day volatility, distance from
                   52w high/low, dollar volume, relative volume, beta vs SPY,
                   sector relative strength.
Fundamental:       growth (YoY/QoQ/CAGR), quality (margins, ROE, ROIC,
                   turnover), health (leverage, coverage), valuation
                   (FCF yield, P/S, EV/EBITDA).

Run standalone:
    python -m data.features --tickers AAPL          # both feature sets
    python -m data.features --prices-only
    python -m data.features --fundamentals-only
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .config import load_config
from .db import Database, get_db
from .utils import divide, get_logger, pct_change, safe_float

log = get_logger("features")

TRADING_DAYS = 252
TAX_SHIELD = 0.79  # approximate (1 - 21% statutory) for NOPAT in ROIC

# GICS sector -> sector ETF used for relative strength.
SECTOR_ETF = {
    "Information Technology": "XLK",
    "Financials": "XLF",
    "Health Care": "XLV",
    "Energy": "XLE",
    "Industrials": "XLI",
    "Communication Services": "XLC",
    "Consumer Discretionary": "XLY",
    "Consumer Staples": "XLP",
    "Materials": "XLB",
    "Real Estate": "XLRE",
    "Utilities": "XLU",
}

PRICE_FEATURE_COLS = [
    "return_20d", "return_60d", "return_252d", "volatility_20d",
    "distance_from_52w_high", "distance_from_52w_low", "dollar_volume",
    "relative_volume", "beta_vs_spy", "sector_relative_strength",
]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _load_prices(db: Database, ticker: str) -> pd.DataFrame:
    df = db.query_df(
        "SELECT date, close, adj_close, volume FROM daily_prices "
        "WHERE ticker = ? ORDER BY date", (ticker,))
    if not df.empty:
        df["date"] = df["date"].astype(str)
    return df


def _returns_series(db: Database, ticker: str) -> pd.Series:
    df = _load_prices(db, ticker)
    if df.empty:
        return pd.Series(dtype=float)
    s = df.set_index("date")["adj_close"].astype(float)
    return s.pct_change()


def _nday_return_series(db: Database, ticker: str, window: int) -> pd.Series:
    df = _load_prices(db, ticker)
    if df.empty:
        return pd.Series(dtype=float)
    s = df.set_index("date")["adj_close"].astype(float)
    return s.pct_change(window)


# ---------------------------------------------------------------------------
# Price features
# ---------------------------------------------------------------------------
def compute_price_features(db: Database, ticker: str, spy_ret: pd.Series,
                           sector_ret60: pd.Series | None,
                           since: str | None = None) -> list[dict]:
    """Rolling price features for one ticker.

    Rolling stats always use the full stored history (a 252d window needs it),
    but when ``since`` is given only rows *after* that date are emitted — the
    older rows are already stored and a feature value at date d depends only on
    bars <= d, so re-upserting them is pure churn. Pass ``since=None`` for a
    full rewrite (e.g. after a split retro-adjusts the stored adj_close).
    """
    df = _load_prices(db, ticker)
    if len(df) < 21:
        return []
    df = df.copy()
    adj = df["adj_close"].astype(float)
    close = df["close"].astype(float)
    vol = df["volume"].astype(float)
    ret = adj.pct_change()

    df["return_20d"] = adj.pct_change(20)
    df["return_60d"] = adj.pct_change(60)
    df["return_252d"] = adj.pct_change(252)
    df["volatility_20d"] = ret.rolling(20).std() * np.sqrt(TRADING_DAYS)
    roll_max = close.rolling(TRADING_DAYS, min_periods=20).max()
    roll_min = close.rolling(TRADING_DAYS, min_periods=20).min()
    df["distance_from_52w_high"] = close / roll_max - 1.0
    df["distance_from_52w_low"] = close / roll_min - 1.0
    df["dollar_volume"] = close * vol
    avg_vol = vol.rolling(20).mean()
    df["relative_volume"] = vol / avg_vol

    # Beta vs SPY over a 252-day rolling window (cov/var of daily returns).
    idx = df["date"]
    sret = spy_ret.reindex(idx).astype(float).reset_index(drop=True)
    sret.index = df.index
    df["beta_vs_spy"] = (ret.rolling(TRADING_DAYS).cov(sret)
                         / sret.rolling(TRADING_DAYS).var())

    # Sector relative strength = stock 60d return - sector ETF 60d return.
    if sector_ret60 is not None and not sector_ret60.empty:
        sec = sector_ret60.reindex(idx).astype(float).reset_index(drop=True)
        sec.index = df.index
        df["sector_relative_strength"] = df["return_60d"] - sec
    else:
        df["sector_relative_strength"] = np.nan

    df = df.replace([np.inf, -np.inf], np.nan)
    if since is not None:
        df = df[df["date"] > since]
    now = _now()
    rows = []
    for r in df.to_dict("records"):
        feat = {c: safe_float(r.get(c)) for c in PRICE_FEATURE_COLS}
        if all(v is None for v in feat.values()):
            continue
        feat.update({"ticker": ticker, "date": r["date"], "computed_at": now})
        rows.append(feat)
    return rows


def update_price_features(db: Database, tickers: list[str],
                          full: bool = False) -> int:
    cfg = load_config()
    bench = cfg.get("features", "market_benchmark", default="SPY")
    spy_ret = _returns_series(db, bench)
    if spy_ret.empty:
        log.warning("No %s price history; beta will be null", bench)

    # Precompute each sector ETF's 60d return series once.
    sector_ret60: dict[str, pd.Series] = {}
    for etf in set(SECTOR_ETF.values()):
        sector_ret60[etf] = _nday_return_series(db, etf, 60)

    sectors = {r["ticker"]: r["gics_sector"]
               for r in db.query("SELECT ticker, gics_sector FROM universe")}

    # Incremental: only emit rows newer than what is stored (minus a small
    # overlap for late-arriving bars). ``full=True`` rewrites all history —
    # needed after a split/dividend retro-adjusts stored adj_close values.
    last_stored: dict[str, str] = {}
    if not full:
        last_stored = {r["ticker"]: str(r["d"]) for r in db.query(
            "SELECT ticker, MAX(date) AS d FROM price_features GROUP BY ticker")}

    total_rows = 0
    for i, ticker in enumerate(tickers, 1):
        etf = SECTOR_ETF.get(sectors.get(ticker, ""))
        sec_series = sector_ret60.get(etf) if etf else None
        since = None
        if not full and ticker in last_stored:
            since = (pd.Timestamp(last_stored[ticker])
                     - pd.Timedelta(days=5)).date().isoformat()
        try:
            rows = compute_price_features(db, ticker, spy_ret, sec_series,
                                          since=since)
            if rows:
                db.upsert("price_features", rows, conflict=["ticker", "date"])
                total_rows += len(rows)
        except Exception as exc:  # noqa: BLE001
            log.warning("Price features failed for %s: %s", ticker, exc)
        if i % 50 == 0 or i == len(tickers):
            log.info("Price features: %d/%d tickers (%d rows)", i, len(tickers), total_rows)
    return total_rows


# ---------------------------------------------------------------------------
# Fundamental features
# ---------------------------------------------------------------------------
FLOW_COLS = ["revenue", "net_income", "ebit", "ebitda", "operating_cash_flow",
             "free_cash_flow", "interest_expense"]


def _price_on(db: Database, ticker: str, on_date: str) -> float | None:
    return safe_float(db.scalar(
        "SELECT close FROM daily_prices WHERE ticker = ? AND date <= ? "
        "ORDER BY date DESC LIMIT 1", (ticker, on_date)))


def _ttm(df: pd.DataFrame) -> pd.DataFrame:
    """Add trailing-twelve-month sums of flow columns for quarterly data."""
    out = df.copy()
    for col in FLOW_COLS:
        if col in out.columns:
            out[f"ttm_{col}"] = out[col].rolling(4, min_periods=4).sum()
    return out


def compute_fundamental_features(db: Database, ticker: str) -> list[dict]:
    rows: list[dict] = []
    now = _now()
    for period_type in ("annual", "quarterly"):
        df = db.query_df(
            "SELECT * FROM fundamentals WHERE ticker = ? AND period_type = ? "
            "ORDER BY fiscal_date", (ticker, period_type))
        if df.empty:
            continue
        quarterly = period_type == "quarterly"
        yoy_lag = 4 if quarterly else 1
        cagr_lag = 12 if quarterly else 3
        if quarterly:
            df = _ttm(df)

        for pos in range(len(df)):
            cur = df.iloc[pos]

            def g(col: str):  # flow value, TTM-adjusted for quarterly ratios
                if quarterly and f"ttm_{col}" in df.columns:
                    return safe_float(cur.get(f"ttm_{col}"))
                return safe_float(cur.get(col))

            rev, ni = g("revenue"), g("net_income")
            ebit, ebitda = g("ebit"), g("ebitda")
            ocf, fcf = g("operating_cash_flow"), g("free_cash_flow")
            int_exp = g("interest_expense")
            equity = safe_float(cur.get("shareholder_equity"))
            assets = safe_float(cur.get("total_assets"))
            debt = safe_float(cur.get("debt"))
            net_debt = safe_float(cur.get("net_debt"))
            cur_assets = safe_float(cur.get("current_assets"))
            cur_liab = safe_float(cur.get("current_liabilities"))
            shares = safe_float(cur.get("shares_outstanding"))
            eps = safe_float(cur.get("eps_diluted")) or safe_float(cur.get("eps_basic"))
            fiscal_date = cur["fiscal_date"]

            feat: dict[str, object] = {
                "ticker": ticker, "period_type": period_type,
                "fiscal_date": fiscal_date, "computed_at": now,
            }
            # Growth (single-period revenue/eps, not TTM, for cleaner YoY/QoQ).
            rev_raw = safe_float(cur.get("revenue"))
            eps_raw = eps
            if pos - yoy_lag >= 0:
                prev = df.iloc[pos - yoy_lag]
                feat["revenue_growth_yoy"] = pct_change(rev_raw, safe_float(prev.get("revenue")))
                pe = safe_float(prev.get("eps_diluted")) or safe_float(prev.get("eps_basic"))
                feat["eps_growth_yoy"] = pct_change(eps_raw, pe)
            if quarterly and pos - 1 >= 0:
                prev = df.iloc[pos - 1]
                feat["revenue_growth_qoq"] = pct_change(rev_raw, safe_float(prev.get("revenue")))
                pe = safe_float(prev.get("eps_diluted")) or safe_float(prev.get("eps_basic"))
                feat["eps_growth_qoq"] = pct_change(eps_raw, pe)
            if pos - cagr_lag >= 0:
                base = df.iloc[pos - cagr_lag]
                feat["revenue_cagr_3y"] = _cagr(rev_raw, safe_float(base.get("revenue")), 3)
                be = safe_float(base.get("eps_diluted")) or safe_float(base.get("eps_basic"))
                feat["eps_cagr_3y"] = _cagr(eps_raw, be, 3)

            # Quality
            feat["roe"] = divide(ni, equity)
            invested = (debt or 0) + (equity or 0)
            feat["roic"] = divide((ebit or 0) * TAX_SHIELD, invested) if invested else None
            feat["gross_margin"] = divide(_gross(cur, df, pos, quarterly), rev)
            feat["operating_margin"] = divide(ebit, rev)
            feat["net_margin"] = divide(ni, rev)
            feat["cfo_to_net_income"] = divide(ocf, ni)
            feat["asset_turnover"] = divide(rev, assets)
            # Health
            feat["debt_to_equity"] = divide(debt, equity)
            feat["current_ratio"] = divide(cur_assets, cur_liab)
            feat["interest_coverage"] = divide(ebit, abs(int_exp)) if int_exp else None
            feat["net_debt_to_ebitda"] = divide(net_debt, ebitda)
            # Valuation (needs market cap at the fiscal date)
            price = _price_on(db, ticker, fiscal_date)
            mcap = price * shares if (price and shares) else None
            feat["fcf_yield"] = divide(fcf, mcap)
            feat["price_to_sales"] = divide(mcap, rev)
            ev = (mcap + (net_debt or 0)) if mcap is not None else None
            feat["ev_to_ebitda"] = divide(ev, ebitda)

            rows.append(feat)
    return rows


def _gross(cur, df, pos, quarterly) -> float | None:
    """Gross profit (TTM for quarterly) for margin computation."""
    if quarterly:
        window = df.iloc[max(0, pos - 3):pos + 1]
        if len(window) < 4:
            return None
        return safe_float(window["gross_profit"].sum())
    return safe_float(cur.get("gross_profit"))


def _cagr(current, base, years: int) -> float | None:
    c, b = safe_float(current), safe_float(base)
    if c is None or b is None or b <= 0 or c <= 0:
        return None
    return (c / b) ** (1.0 / years) - 1.0


def update_fundamental_features(db: Database, tickers: list[str]) -> int:
    total = 0
    for i, ticker in enumerate(tickers, 1):
        try:
            rows = compute_fundamental_features(db, ticker)
            if rows:
                db.upsert("fundamental_features", rows,
                          conflict=["ticker", "period_type", "fiscal_date"])
                total += len(rows)
        except Exception as exc:  # noqa: BLE001
            log.warning("Fundamental features failed for %s: %s", ticker, exc)
        if i % 25 == 0 or i == len(tickers):
            log.info("Fundamental features: %d/%d (%d rows)", i, len(tickers), total)
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate derived features")
    parser.add_argument("--tickers", nargs="*")
    parser.add_argument("--prices-only", action="store_true")
    parser.add_argument("--fundamentals-only", action="store_true")
    parser.add_argument("--full", action="store_true",
                        help="Rewrite all price-feature history (required after "
                             "extending price history backwards — incremental "
                             "mode only appends past the last stored date)")
    args = parser.parse_args()
    cfg = load_config()
    get_logger("features", log_file=cfg.log_file)
    with get_db() as db:
        tickers = [t.upper() for t in args.tickers] if args.tickers else db.universe_tickers()
        pf = ff = 0
        if not args.fundamentals_only:
            pf = update_price_features(db, tickers, full=args.full)
        if not args.prices_only:
            ff = update_fundamental_features(db, tickers)
    print(f"Price feature rows       : {pf}")
    print(f"Fundamental feature rows : {ff}")


if __name__ == "__main__":
    main()
