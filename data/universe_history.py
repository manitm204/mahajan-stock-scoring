"""Point-in-time S&P 500 membership history (FMP constituent-change feed).

Reconstructs which tickers were in the index on any historical date by
replaying FMP's ``historical-sp500-constituent`` change events *backwards*
from today's known membership. Each membership spell lands in the
``universe_history`` table (start_date inclusive, end_date exclusive, NULL =
still a member); ``Database.members_as_of(date)`` serves point-in-time
queries so research and backtests never score a survivorship-biased set.

Departed names that are absent from the static ``universe`` table are also
profiled via FMP (sector/name) and stored there with ``active=0`` so
sector-neutral logic keeps working for them.

Run standalone:
    python -m data.universe_history                     # build + profiles
    python -m data.universe_history --window-start 2015-01-01
    python -m data.universe_history --dry-run           # report only
"""
from __future__ import annotations

import argparse
from datetime import date, datetime, timezone

from .config import load_config
from .db import Database, get_db
from .providers import FMPProvider, ProviderRegistry
from .universe import normalize_sector, normalize_ticker
from .utils import get_logger, to_iso_date

log = get_logger("universe_history")

# The feed's own origin; used as the spell start for names whose addition
# predates every change event we have (documented, not fabricated precision).
FEED_ORIGIN = "1957-03-03"
DEFAULT_WINDOW_START = "2015-01-01"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def fetch_change_events(provider: FMPProvider) -> list[dict]:
    """All index change events, oldest-first: {date, added, removed}."""
    raw = provider._get("historical-sp500-constituent") or []
    events = []
    for row in raw:
        d = to_iso_date(row.get("date"))
        if not d:
            continue
        # Pure removals reuse `symbol` for the *removed* ticker with an empty
        # addedSecurity — only treat `symbol` as an addition when a security
        # was actually added (verified against NBL/RTN/M/BHF feed rows).
        is_add = bool((row.get("addedSecurity") or "").strip())
        events.append({
            "date": d,
            "added": normalize_ticker(row.get("symbol") or "") if is_add else "",
            "removed": normalize_ticker(row.get("removedTicker") or ""),
        })
    events.sort(key=lambda e: e["date"])
    return events


def build_spells(current_members: set[str],
                 events: list[dict]) -> tuple[list[dict], list[str]]:
    """Replay change events backwards from today's membership.

    Returns (spells, anomalies). Each spell is {ticker, start_date, end_date};
    end_date None = still in the index. Anomalies are logged inconsistencies
    (e.g. an add event for a name we don't believe was a member afterwards) —
    expected in small numbers from renames/mergers the feed records unevenly.
    """
    members = set(current_members)
    # ticker -> end_date of the spell we are currently "inside" walking back.
    open_end: dict[str, str | None] = {t: None for t in members}
    spells: list[dict] = []
    anomalies: list[str] = []

    for ev in sorted(events, key=lambda e: e["date"], reverse=True):
        d, added, removed = ev["date"], ev["added"], ev["removed"]
        if added:
            if added in members:
                # Before d the name was not yet in the index: close its spell.
                spells.append({"ticker": added, "start_date": d,
                               "end_date": open_end.pop(added, None)})
                members.discard(added)
            else:
                anomalies.append(
                    f"add {added} @ {d} but no later membership known "
                    f"(rename/merger?)")
        if removed:
            if removed in members:
                anomalies.append(
                    f"remove {removed} @ {d} while already a member earlier "
                    f"(overlapping spells?)")
            else:
                # Before d the name WAS in the index: open a spell ending at d.
                members.add(removed)
                open_end[removed] = d

    # Names still "in" after the full replay were members since before the
    # earliest event we processed for them.
    for t in members:
        spells.append({"ticker": t, "start_date": FEED_ORIGIN,
                       "end_date": open_end.get(t)})
    return spells, anomalies


def store_spells(db: Database, spells: list[dict], window_start: str) -> int:
    """Persist spells overlapping [window_start, today]. Rebuilds the table."""
    keep = [s for s in spells
            if s["end_date"] is None or s["end_date"] > window_start]
    now = _now()
    names = {r["ticker"]: r["company_name"] for r in db.query(
        "SELECT ticker, company_name FROM universe")}
    rows = [{"ticker": s["ticker"], "start_date": s["start_date"],
             "end_date": s["end_date"], "company_name": names.get(s["ticker"]),
             "source": "fmp", "last_updated": now} for s in keep]
    with db.transaction() as conn:
        conn.execute("DELETE FROM universe_history")
    db.upsert("universe_history", rows, conflict=["ticker", "start_date"])
    return len(rows)


def profile_departed(db: Database, provider: FMPProvider,
                     window_start: str) -> dict[str, int]:
    """Fetch FMP profiles for historical members missing from ``universe``.

    Stores them with active=0/source='fmp_history' so sectors resolve for
    departed names. Returns {"needed": n, "profiled": ok, "missing": fail}.
    """
    known = set(db.universe_tickers(active_only=False))
    hist = {r["ticker"] for r in db.query(
        "SELECT DISTINCT ticker FROM universe_history "
        "WHERE end_date IS NULL OR end_date > ?", (window_start,))}
    todo = sorted(hist - known)
    stats = {"needed": len(todo), "profiled": 0, "missing": 0}
    now = _now()
    for i, ticker in enumerate(todo, 1):
        data = provider._get("profile", {"symbol": ticker})
        prof = data[0] if isinstance(data, list) and data else None
        if not prof:
            stats["missing"] += 1
            log.warning("No FMP profile for departed member %s", ticker)
            continue
        db.upsert("universe", [{
            "ticker": ticker,
            "company_name": prof.get("companyName"),
            "gics_sector": normalize_sector(prof.get("sector")),
            "gics_sub_industry": prof.get("industry"),
            "source": "fmp_history",
            "active": 0,
            "first_seen": now,
            "last_updated": now,
        }], conflict=["ticker"],
            update=["company_name", "gics_sector", "gics_sub_industry",
                    "source", "last_updated"])
        stats["profiled"] += 1
        if i % 50 == 0 or i == len(todo):
            log.info("Departed profiles: %d/%d", i, len(todo))
    return stats


def validate(db: Database, window_start: str) -> list[str]:
    """Cross-checks; returns human-readable findings (also logged)."""
    findings: list[str] = []
    today = date.today().isoformat()
    live = set(db.universe_tickers(active_only=True))
    pit = set(db.members_as_of(today))
    only_live, only_pit = sorted(live - pit), sorted(pit - live)
    if only_live or only_pit:
        findings.append(f"today mismatch: universe-only={only_live[:8]} "
                        f"history-only={only_pit[:8]}")
    for y in range(int(window_start[:4]), int(today[:4]) + 1):
        n = len(db.members_as_of(f"{y}-12-31" if y < int(today[:4]) else today))
        if not 480 <= n <= 520:
            findings.append(f"member count {n} as of {y} outside [480, 520]")
    for f in findings:
        log.warning("validate: %s", f)
    return findings


def build_history(db: Database, window_start: str = DEFAULT_WINDOW_START,
                  dry_run: bool = False,
                  registry: ProviderRegistry | None = None) -> dict:
    registry = registry or ProviderRegistry(load_config())
    provider = registry.transcripts()  # FMP instance (same key/base for all)
    if provider is None or provider.name != "fmp":
        log.error("FMP provider unavailable; cannot build universe history")
        return {"events": 0, "spells": 0}

    events = fetch_change_events(provider)
    current = set(db.universe_tickers(active_only=True))
    spells, anomalies = build_spells(current, events)
    in_window = [s for s in spells
                 if s["end_date"] is None or s["end_date"] > window_start]
    log.info("Feed: %d events; %d spells total, %d overlap window >= %s "
             "(%d anomalies)", len(events), len(spells), len(in_window),
             window_start, len(anomalies))
    for a in anomalies[:20]:
        log.info("anomaly: %s", a)

    summary = {"events": len(events), "spells": len(in_window),
               "anomalies": len(anomalies)}
    if dry_run:
        return summary
    summary["stored"] = store_spells(db, spells, window_start)
    summary["profiles"] = profile_departed(db, provider, window_start)
    summary["findings"] = validate(db, window_start)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build point-in-time S&P 500 membership history")
    parser.add_argument("--window-start", default=DEFAULT_WINDOW_START,
                        help="Keep spells overlapping this date onward")
    parser.add_argument("--dry-run", action="store_true",
                        help="Fetch and report; do not write")
    args = parser.parse_args()
    cfg = load_config()
    get_logger("universe_history", log_file=cfg.log_file)
    with get_db() as db:
        summary = build_history(db, args.window_start, args.dry_run)
    print(f"Change events      : {summary.get('events', 0)}")
    print(f"Spells in window   : {summary.get('spells', 0)}")
    print(f"Stored             : {summary.get('stored', '-')}")
    print(f"Departed profiles  : {summary.get('profiles', '-')}")
    if summary.get("findings"):
        print("Validation findings:")
        for f in summary["findings"]:
            print(f"  ! {f}")


if __name__ == "__main__":
    main()
