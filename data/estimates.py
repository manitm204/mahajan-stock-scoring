"""Analyst estimates module.

Stores daily snapshots of forward EPS, revenue estimate, price targets, and
analyst count sourced from FMP Ultimate (`analyst-estimates` +
`price-target-summary`). Revision features (30/60/90-day changes in forward EPS
and price target, plus upside and breadth changes) are computed by comparing the
current snapshot against the snapshot nearest to N days ago, so they accrue
automatically as daily history accumulates.

The consensus endpoint returns the *current* consensus per forward fiscal period
(not a dated revision series), so numeric revision momentum is forward-accruing;
the dated grade/price-target-event history in :mod:`data.grades` supplies the
back-history for the revisions factor.

Run standalone:
    python -m data.estimates --tickers AAPL MSFT
"""
from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta, timezone

from .config import load_config
from .db import Database, get_db
from .providers import ProviderRegistry
from .utils import get_logger, safe_float, safe_int, to_iso_date

log = get_logger("estimates")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _forward_period(rows: list[dict], today: str) -> dict | None:
    """The next full fiscal period on/after ``today`` (else the latest available).

    FMP returns consensus rows keyed by fiscal-period-end `date`; the forward
    estimate is the nearest period that has not yet closed.
    """
    dated = [(to_iso_date(r.get("date")), r) for r in rows]
    dated = [(d, r) for d, r in dated if d]
    if not dated:
        return None
    future = sorted((d, r) for d, r in dated if d >= today)
    if future:
        return future[0][1]
    return sorted(dated)[-1][1]  # all periods past → most recent consensus


def fetch_snapshot(ticker: str, db: Database, provider) -> dict | None:
    """Current forward consensus for ``ticker`` from FMP Ultimate."""
    today = date.today().isoformat()
    annual = provider.get_analyst_estimates(ticker, period="annual", limit=40)
    fwd = _forward_period(annual, today) or {}
    forward_eps = safe_float(fwd.get("epsAvg"))
    revenue_estimate = safe_float(fwd.get("revenueAvg"))
    n_analysts = safe_int(fwd.get("numAnalystsEps"))

    # Consensus price target from the dedicated summary endpoint.
    target_mean = target_low = target_high = None
    summary = provider.get_price_target_summary(ticker) if hasattr(
        provider, "get_price_target_summary") else None
    if summary:
        target_mean = safe_float(summary.get("lastMonthAvgPriceTarget")) or \
            safe_float(summary.get("lastQuarterAvgPriceTarget"))

    # Current price from the warehouse close.
    current_price = safe_float(db.scalar(
        "SELECT close FROM daily_prices WHERE ticker = ? ORDER BY date DESC LIMIT 1",
        (ticker,)))

    if all(v is None for v in (forward_eps, revenue_estimate, target_mean, n_analysts)):
        return None
    return {
        "forward_eps": forward_eps,
        "revenue_estimate": revenue_estimate,
        "consensus_price_target": target_mean,
        "low_price_target": target_low,
        "high_price_target": target_high,
        "recommendation_mean": None,
        "number_of_analysts": n_analysts,
        "current_price": current_price,
    }


def _value_n_days_ago(db: Database, ticker: str, column: str, today: str,
                      days: int) -> float | None:
    """Snapshot value at-or-before (today - days), nearest available."""
    target = (datetime.strptime(today, "%Y-%m-%d").date()
              - timedelta(days=days)).isoformat()
    return safe_float(db.scalar(
        f"SELECT {column} FROM analyst_estimates "
        "WHERE ticker = ? AND snapshot_date <= ? ORDER BY snapshot_date DESC LIMIT 1",
        (ticker, target)))


def compute_revisions(db: Database, ticker: str, snap: dict, today: str,
                      windows: list[int]) -> dict:
    feat: dict[str, object] = {"ticker": ticker, "snapshot_date": today,
                               "computed_at": _now()}
    for w in windows:
        prev_eps = _value_n_days_ago(db, ticker, "forward_eps", today, w)
        if prev_eps is not None and snap.get("forward_eps") is not None:
            feat[f"forward_eps_revision_{w}d"] = snap["forward_eps"] - prev_eps
        prev_pt = _value_n_days_ago(db, ticker, "consensus_price_target", today, w)
        if prev_pt is not None and snap.get("consensus_price_target") is not None:
            feat[f"price_target_revision_{w}d"] = snap["consensus_price_target"] - prev_pt

    pt, price = snap.get("consensus_price_target"), snap.get("current_price")
    if pt is not None and price not in (None, 0):
        feat["price_target_upside_pct"] = (pt - price) / price

    base_w = windows[0] if windows else 30
    prev_n = _value_n_days_ago(db, ticker, "number_of_analysts", today, base_w)
    if prev_n is not None and snap.get("number_of_analysts") is not None:
        feat["analyst_count_change"] = int(snap["number_of_analysts"] - prev_n)
        feat["estimate_breadth_change"] = snap["number_of_analysts"] - prev_n
    return feat


def update_ticker(db: Database, provider, ticker: str, today: str,
                  windows: list[int]) -> bool:
    snap = fetch_snapshot(ticker, db, provider)
    if not snap:
        return False
    snap.update({"ticker": ticker, "snapshot_date": today,
                 "source": "fmp", "fetched_at": _now()})
    db.upsert("analyst_estimates", [snap], conflict=["ticker", "snapshot_date"])
    feat = compute_revisions(db, ticker, snap, today, windows)
    db.upsert("analyst_estimate_features", [feat],
              conflict=["ticker", "snapshot_date"])
    return True


def update_estimates(db: Database, tickers: list[str],
                     registry: ProviderRegistry | None = None) -> dict:
    cfg = load_config()
    registry = registry or ProviderRegistry(cfg)
    provider = registry.fundamentals()
    if provider is None or not hasattr(provider, "get_analyst_estimates"):
        log.warning("No FMP provider available for analyst estimates; skipping")
        return {"records": 0, "failed": []}
    today = date.today().isoformat()
    windows = cfg.get("estimates", "revision_windows", default=[30, 60, 90])
    added = 0
    failed: list[str] = []
    for i, ticker in enumerate(tickers, 1):
        try:
            if update_ticker(db, provider, ticker, today, windows):
                added += 1
        except Exception as exc:  # noqa: BLE001
            log.warning("Estimates failed for %s: %s", ticker, exc)
            failed.append(ticker)
        if i % 50 == 0 or i == len(tickers):
            log.info("Estimates: %d/%d (%d snapshots)", i, len(tickers), added)
    return {"records": added, "failed": failed}


def main() -> None:
    parser = argparse.ArgumentParser(description="Update analyst estimate snapshots")
    parser.add_argument("--tickers", nargs="*")
    args = parser.parse_args()
    cfg = load_config()
    get_logger("estimates", log_file=cfg.log_file)
    with get_db() as db:
        tickers = [t.upper() for t in args.tickers] if args.tickers else db.universe_tickers()
        stats = update_estimates(db, tickers)
    print(f"Snapshots added : {stats['records']}")
    print(f"Failed          : {len(stats['failed'])}")


if __name__ == "__main__":
    main()
