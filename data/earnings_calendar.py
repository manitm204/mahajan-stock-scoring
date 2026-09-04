"""Earnings calendar module.

Stores past and upcoming earnings dates (with EPS/revenue estimate/actual
where available). FMP's /stable/earnings is the primary source (~25y of
announcement dates with report-time-frozen estimates); yfinance remains the
no-key fallback (~24 rows). Proximity signals (days_until_earnings and
within-7/14/30-day flags) are derived on demand from the stored dates
relative to a reference date, so they never go stale in storage.

Run standalone:
    python -m data.earnings_calendar --tickers AAPL MSFT
"""
from __future__ import annotations

import argparse
from datetime import date, datetime, timezone

import pandas as pd

from .config import load_config
from .db import Database, get_db
from .providers import ProviderRegistry
from .utils import get_logger, safe_float, to_iso_date

log = get_logger("earnings_calendar")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _fiscal_quarter(iso_date: str) -> str:
    try:
        d = datetime.strptime(iso_date, "%Y-%m-%d").date()
    except ValueError:
        return ""
    return f"Q{((d.month - 1) // 3) + 1}-{d.year}"


def fetch_dates(ticker: str) -> list[dict]:
    import yfinance as yf
    t = yf.Ticker(ticker)
    rows: list[dict] = []
    seen: set[str] = set()

    # earnings_dates: historical + a few upcoming, with estimate/actual columns.
    try:
        df = t.get_earnings_dates(limit=24)
    except Exception:  # noqa: BLE001
        df = None
    if df is not None and not df.empty:
        df = df.reset_index()
        date_col = df.columns[0]
        for r in df.to_dict("records"):
            iso = to_iso_date(r.get(date_col))
            if not iso or iso in seen:
                continue
            seen.add(iso)
            ts = pd.to_datetime(r.get(date_col), errors="coerce")
            etime = None
            if ts is not None and not pd.isna(ts):
                etime = "bmo" if ts.hour and ts.hour < 12 else ("amc" if ts.hour else None)
            rows.append({
                "earnings_date": iso,
                "earnings_time": etime,
                "fiscal_quarter": _fiscal_quarter(iso),
                "eps_estimate": safe_float(r.get("EPS Estimate")),
                "eps_actual": safe_float(r.get("Reported EPS")),
            })

    # calendar: ensures the very next earnings date is captured.
    try:
        cal = t.calendar or {}
        ed = cal.get("Earnings Date")
        dates = ed if isinstance(ed, list) else ([ed] if ed else [])
        for d in dates:
            iso = to_iso_date(d)
            if iso and iso not in seen:
                seen.add(iso)
                rows.append({
                    "earnings_date": iso, "earnings_time": None,
                    "fiscal_quarter": _fiscal_quarter(iso),
                    "eps_estimate": None, "eps_actual": None,
                })
    except Exception:  # noqa: BLE001
        pass
    return rows


def fetch_dates_fmp(provider, ticker: str) -> list[dict]:
    rows: list[dict] = []
    for r in provider.get_earnings(ticker):
        iso = to_iso_date(r.get("date"))
        if not iso:
            continue
        rows.append({
            "earnings_date": iso,
            "earnings_time": None,
            "fiscal_quarter": _fiscal_quarter(iso),
            "eps_estimate": safe_float(r.get("epsEstimated")),
            "eps_actual": safe_float(r.get("epsActual")),
            "revenue_estimate": safe_float(r.get("revenueEstimated")),
            "revenue_actual": safe_float(r.get("revenueActual")),
        })
    return rows


def update_ticker(db: Database, ticker: str, provider=None) -> int:
    source = "yfinance"
    rows: list[dict] = []
    if provider is not None and provider.name == "fmp":
        rows = fetch_dates_fmp(provider, ticker)
        if rows:
            source = "fmp"
    if not rows:
        rows = fetch_dates(ticker)
    if not rows:
        return 0
    now = _now()
    for r in rows:
        r.update({"ticker": ticker, "source": source, "fetched_at": now})
    # Future-dated rows are forecasts whose dates can shift between fetches;
    # replace them wholesale so stale announcement dates don't linger.
    today = date.today().isoformat()
    with db.transaction() as conn:
        conn.execute("DELETE FROM earnings_calendar WHERE ticker = ? "
                     "AND earnings_date > ?", (ticker, today))
    db.upsert("earnings_calendar", rows, conflict=["ticker", "earnings_date"],
              update=["earnings_time", "fiscal_quarter", "eps_estimate",
                      "eps_actual", "revenue_estimate", "revenue_actual",
                      "source", "fetched_at"])
    return len(rows)


def update_earnings_calendar(db: Database, tickers: list[str]) -> dict:
    provider = ProviderRegistry().earnings()
    updated = 0
    failed: list[str] = []
    for i, ticker in enumerate(tickers, 1):
        try:
            if update_ticker(db, ticker, provider):
                updated += 1
        except Exception as exc:  # noqa: BLE001
            log.warning("Earnings calendar failed for %s: %s", ticker, exc)
            failed.append(ticker)
        if i % 50 == 0 or i == len(tickers):
            log.info("Earnings calendar: %d/%d (%d updated)", i, len(tickers), updated)
    return {"updated": updated, "failed": failed}


def proximity_features(db: Database, tickers: list[str] | None = None,
                       as_of: str | None = None) -> dict[str, dict]:
    """Derive days_until_earnings + within-N-day flags for the next upcoming
    earnings date per ticker, relative to ``as_of`` (default today)."""
    cfg = load_config()
    windows = cfg.get("earnings_calendar", "proximity_windows", default=[7, 14, 30])
    ref = datetime.strptime(as_of, "%Y-%m-%d").date() if as_of else date.today()
    where = ""
    params: list = [ref.isoformat()]
    if tickers:
        where = f" AND ticker IN ({','.join('?' * len(tickers))})"
        params += tickers
    rows = db.query(
        "SELECT ticker, MIN(earnings_date) nxt FROM earnings_calendar "
        "WHERE earnings_date >= ?" + where + " GROUP BY ticker", params)
    out: dict[str, dict] = {}
    for r in rows:
        nxt = r["nxt"]
        days = (datetime.strptime(nxt, "%Y-%m-%d").date() - ref).days
        feat = {"next_earnings_date": nxt, "days_until_earnings": days}
        for w in windows:
            feat[f"earnings_within_{w}d"] = int(0 <= days <= w)
        out[r["ticker"]] = feat
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Update earnings calendar")
    parser.add_argument("--tickers", nargs="*")
    args = parser.parse_args()
    cfg = load_config()
    get_logger("earnings_calendar", log_file=cfg.log_file)
    with get_db() as db:
        tickers = [t.upper() for t in args.tickers] if args.tickers else db.universe_tickers()
        stats = update_earnings_calendar(db, tickers)
        prox = proximity_features(db, tickers if args.tickers else None)
    print(f"Tickers updated : {stats['updated']}")
    print(f"Failed          : {len(stats['failed'])}")
    for tkr, f in list(prox.items())[:5]:
        print(f"  {tkr}: next {f['next_earnings_date']} ({f['days_until_earnings']}d)")


if __name__ == "__main__":
    main()
