"""Short interest module.

Stores short-interest history (shares short, short ratio, % of float) and
derives change/days-to-cover/squeeze-risk signals. Polygon's
``/stocks/v1/short-interest`` provides the bi-monthly FINRA history; the
yfinance ``.info`` snapshot remains as a fallback for tickers Polygon does not
cover or when the Polygon key is absent.

Run standalone:
    python -m data.short_interest --tickers AAPL TSLA
"""
from __future__ import annotations

import argparse
from datetime import date, datetime, timezone

from .config import load_config
from .db import Database, get_db
from .providers import ProviderRegistry
from .utils import get_logger, pct_change, safe_float

log = get_logger("short_interest")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _avg_volume(db: Database, ticker: str, window: int = 20) -> float | None:
    rows = db.query(
        "SELECT volume FROM daily_prices WHERE ticker = ? "
        "ORDER BY date DESC LIMIT ?", (ticker, window))
    vols = [r["volume"] for r in rows if r["volume"]]
    return sum(vols) / len(vols) if vols else None


def _prior_snapshot(db: Database, ticker: str, before: str) -> dict | None:
    row = db.query_one(
        "SELECT * FROM short_interest WHERE ticker = ? AND date < ? "
        "ORDER BY date DESC LIMIT 1", (ticker, before))
    return dict(row) if row else None


def _float_shares(db: Database, ticker: str) -> float | None:
    """Latest known float for this ticker (set by the yfinance path, or NULL).
    Used to derive `short_percent_of_float` for Polygon rows."""
    return safe_float(db.scalar(
        "SELECT float_shares FROM short_interest WHERE ticker = ? "
        "AND float_shares IS NOT NULL ORDER BY date DESC LIMIT 1",
        (ticker,)))


def _squeeze_flag(spf: float | None, dtc: float | None, cfg) -> int:
    pct_thr = float(cfg.get("short_interest", "squeeze_short_pct_threshold",
                            default=0.20))
    dtc_thr = float(cfg.get("short_interest", "squeeze_days_to_cover_threshold",
                            default=5.0))
    return int(bool(spf and spf >= pct_thr) and bool(dtc and dtc >= dtc_thr))


# ---------------------------------------------------------------------------
# Polygon path — bi-monthly FINRA history
# ---------------------------------------------------------------------------
def update_ticker_polygon(db: Database, provider, ticker: str, cfg,
                          since: str | None) -> int:
    raw = provider.get_short_interest(ticker, since=since)
    if not raw:
        return 0
    raw = sorted(raw, key=lambda r: r.get("settlement_date") or "")
    floats = _float_shares(db, ticker)
    rows: list[dict] = []
    prev_si: float | None = None
    prev_spf: float | None = None
    now = _now()
    for r in raw:
        sdate = r.get("settlement_date")
        si = safe_float(r.get("short_interest"))
        if not sdate or si is None:
            continue
        dtc = safe_float(r.get("days_to_cover"))
        spf = (si / floats) if floats else None
        si_chg = pct_change(si, prev_si) if prev_si is not None else None
        spf_chg = (spf - prev_spf if spf is not None and prev_spf is not None
                   else None)
        rows.append({
            "ticker": ticker,
            "date": sdate,
            "shares_short": si,
            "short_ratio": dtc,
            "short_percent_of_float": spf,
            "shares_outstanding": None,  # Polygon doesn't supply this
            "float_shares": floats,
            "short_interest_change": si_chg,
            "short_percent_float_change": spf_chg,
            "days_to_cover": dtc,
            "short_squeeze_risk_flag": _squeeze_flag(spf, dtc, cfg),
            "source": "polygon",
            "fetched_at": now,
        })
        prev_si = si
        prev_spf = spf
    if rows:
        db.upsert("short_interest", rows, conflict=["ticker", "date"])
    return len(rows)


# ---------------------------------------------------------------------------
# yfinance path — single snapshot fallback
# ---------------------------------------------------------------------------
def fetch_snapshot(ticker: str) -> dict | None:
    import yfinance as yf
    try:
        info = yf.Ticker(ticker).info
    except Exception as exc:  # noqa: BLE001
        log.warning("Short interest info failed for %s: %s", ticker, exc)
        return None
    if not info:
        return None
    return {
        "shares_short": safe_float(info.get("sharesShort")),
        "short_ratio": safe_float(info.get("shortRatio")),
        "short_percent_of_float": safe_float(info.get("shortPercentOfFloat")),
        "shares_outstanding": safe_float(info.get("sharesOutstanding")),
        "float_shares": safe_float(info.get("floatShares")),
    }


def update_ticker_yfinance(db: Database, ticker: str, cfg, today: str) -> int:
    snap = fetch_snapshot(ticker)
    if not snap or snap.get("shares_short") is None:
        return 0

    prior = _prior_snapshot(db, ticker, today)
    snap["short_interest_change"] = (
        pct_change(snap["shares_short"], prior.get("shares_short")) if prior else None)
    snap["short_percent_float_change"] = (
        (snap["short_percent_of_float"] - prior["short_percent_of_float"])
        if prior and snap["short_percent_of_float"] is not None
        and prior.get("short_percent_of_float") is not None else None)

    dtc = snap.get("short_ratio")
    if dtc is None:
        avg_vol = _avg_volume(db, ticker)
        dtc = (snap["shares_short"] / avg_vol) if avg_vol else None
    snap["days_to_cover"] = dtc
    snap["short_squeeze_risk_flag"] = _squeeze_flag(
        snap.get("short_percent_of_float"), dtc, cfg)
    snap.update({"ticker": ticker, "date": today, "source": "yfinance",
                 "fetched_at": _now()})
    db.upsert("short_interest", [snap], conflict=["ticker", "date"])
    return 1


def update_short_interest(db: Database, tickers: list[str],
                          registry: ProviderRegistry | None = None,
                          since: str | None = None) -> dict:
    cfg = load_config()
    registry = registry or ProviderRegistry(cfg)
    provider = registry.short_interest()
    today = date.today().isoformat()
    use_polygon = provider is not None and hasattr(provider, "get_short_interest")
    if not use_polygon:
        log.info("Short interest: using yfinance snapshot fallback")
    added = 0
    failed: list[str] = []
    for i, ticker in enumerate(tickers, 1):
        try:
            if use_polygon:
                n = update_ticker_polygon(db, provider, ticker, cfg, since)
                if n == 0:  # Polygon empty -> try yfinance for at least the snapshot
                    n = update_ticker_yfinance(db, ticker, cfg, today)
            else:
                n = update_ticker_yfinance(db, ticker, cfg, today)
            added += n
        except Exception as exc:  # noqa: BLE001
            log.warning("Short interest failed for %s: %s", ticker, exc)
            failed.append(ticker)
        if i % 50 == 0 or i == len(tickers):
            log.info("Short interest: %d/%d (%d rows)", i, len(tickers), added)
    return {"records": added, "failed": failed}


def main() -> None:
    parser = argparse.ArgumentParser(description="Update short interest history")
    parser.add_argument("--tickers", nargs="*")
    parser.add_argument("--since", default=None,
                        help="Earliest settlement_date to fetch (Polygon path)")
    args = parser.parse_args()
    cfg = load_config()
    get_logger("short_interest", log_file=cfg.log_file)
    with get_db() as db:
        tickers = [t.upper() for t in args.tickers] if args.tickers else db.universe_tickers()
        stats = update_short_interest(db, tickers, since=args.since)
    print(f"Rows added : {stats['records']}")
    print(f"Failed     : {len(stats['failed'])}")


if __name__ == "__main__":
    main()
