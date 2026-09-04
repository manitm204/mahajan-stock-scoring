"""Subfactor Expansion — end-to-end research pipeline entry point.

Steps:

    1. --audit          list current & candidate subfactor counts and quit
    2. --backfill       fetch new raw data (dividends)
    3. --build          score every candidate at a monthly grid
    4. --validate       compute coverage/IC/quintiles/sign/year-stability
    5. --report         write the REPORT.md deliverable
    (--all runs 3+4+5)

Runs against the production database with strict PIT (``reporting_lag=True``)
via :class:`DataContext`. Outputs land under ``output/subfactor_expansion/``;
production tables and factor modules are never touched.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

# The library uses ``groupby().apply()`` extensively; pandas has flagged the
# grouping-columns behavior for future removal (still functional today).
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")
warnings.filterwarnings("ignore", category=FutureWarning,
                        module="research.subfactor_expansion")

import pandas as pd  # noqa: E402

from data.config import load_config
from data.db import get_db
from data.utils import get_logger

from research.subfactor_expansion.library import iter_parents
from research.subfactor_expansion.panel import (
    CandidatePanel,
    build_candidate_panel,
    cache_key,
    load_cached_panel,
    save_cached_panel,
)
from research.subfactor_expansion.report import write_report
from research.subfactor_expansion.validation import run_validation

log = get_logger("run_subfactor_expansion")

OUT_DIR = Path("output/subfactor_expansion")
CACHE_DIR = Path("cache/subfactor_expansion")


def _monthly_rebalances(db, start: str, end: str) -> list[str]:
    """Trading dates closest to each month-end in [start, end]."""
    df = db.query_df(
        "SELECT DISTINCT date FROM daily_prices WHERE date >= ? AND date <= ? "
        "ORDER BY date", (start, end)
    )
    if df.empty:
        return []
    dates = pd.to_datetime(df["date"])
    month_ends: list[str] = []
    for period, grp in dates.groupby(dates.dt.to_period("M")):
        month_ends.append(str(grp.max().date()))
    return month_ends


def _price_matrix(db, start: str, end: str) -> pd.DataFrame:
    df = db.query_df(
        "SELECT ticker, date, adj_close FROM daily_prices "
        "WHERE date >= ? AND date <= ? ORDER BY date", (start, end)
    )
    if df.empty:
        return pd.DataFrame()
    wide = df.pivot_table(index="date", columns="ticker", values="adj_close").sort_index()
    return wide


def _do_audit() -> int:
    """Print the audit table so users see the plan before committing to --build."""
    from research.subfactor_expansion.report import _OLD_COUNTS

    # Late import so the audit is cheap even when the DB is offline.
    from research.subfactor_expansion.library import CANDIDATE_BUILDERS
    from factors.utils import DataContext

    db = get_db()
    ctx = DataContext(db)
    try:
        print("Parent           | Old (lean) | Proposed | Δ")
        print("-----------------+------------+----------+----")
        total_old = total_new = 0
        for parent in iter_parents():
            try:
                new = len(CANDIDATE_BUILDERS[parent](ctx))
            except Exception as exc:  # noqa: BLE001
                log.warning("audit build %s failed: %s", parent, exc)
                new = 0
            old = _OLD_COUNTS.get(parent, 0)
            total_old += old
            total_new += new
            print(f"{parent:16s} | {old:>10d} | {new:>8d} | +{new - old}")
        print("-----------------+------------+----------+----")
        print(f"{'total':16s} | {total_old:>10d} | {total_new:>8d} | "
              f"+{total_new - total_old}")
    finally:
        ctx.close()
    return 0


def _do_backfill() -> int:
    """Kick off the dividends backfill by delegating to the script."""
    from scripts.subfactor_expansion.fetch_dividends import main as fetch_main
    return fetch_main()


def _do_build(start: str, end: str, use_cache: bool) -> Path:
    db = get_db()
    cfg = load_config()
    rebals = _monthly_rebalances(db, start, end)
    if not rebals:
        raise SystemExit(f"no trading dates found in [{start}, {end}]")
    cache_path = CACHE_DIR / cache_key(start, end, "monthly")
    panel = load_cached_panel(cache_path) if use_cache else None
    if panel is not None:
        log.info("loaded cached panel from %s (%d dates)", cache_path, len(panel.rebal_dates))
        return cache_path
    log.info("building panel for %d rebalance dates", len(rebals))
    panel = build_candidate_panel(db, rebals, cfg, verbose=True)
    save_cached_panel(panel, cache_path)
    log.info("saved panel -> %s", cache_path)
    return cache_path


def _do_validate(panel: CandidatePanel, start: str, end: str,
                 ic_horizon: str) -> dict:
    db = get_db()
    matrix = _price_matrix(db, start, end)
    if matrix.empty:
        raise SystemExit("empty price matrix; check the price ingestion")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    return run_validation(panel, matrix, OUT_DIR, ic_horizon=ic_horizon)


def _load_panel_or_die(cache_path: Path) -> CandidatePanel:
    panel = load_cached_panel(cache_path)
    if panel is None:
        raise SystemExit(f"no cached panel at {cache_path}; run --build first")
    return panel


def _do_report(panel: CandidatePanel, validation: dict, ic_horizon: str,
               new_endpoints: list[str] | None,
               backfill_status: dict[str, str] | None) -> Path:
    return write_report(
        panel, validation, OUT_DIR / "REPORT.md",
        ic_horizon=ic_horizon,
        new_endpoints=new_endpoints, backfill_status=backfill_status,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--audit", action="store_true", help="print old/new counts and exit")
    action.add_argument("--backfill", action="store_true", help="fetch dividends from FMP")
    action.add_argument("--build", action="store_true", help="build the candidate panel")
    action.add_argument("--validate", action="store_true", help="run coverage/IC/etc.")
    action.add_argument("--report", action="store_true", help="write REPORT.md")
    action.add_argument("--charts", action="store_true",
                       help="write ic_by_subfactor / scorecard / correlation heatmap")
    action.add_argument("--all", action="store_true",
                       help="build + validate + report + charts")

    parser.add_argument("--start", default="2023-01-01",
                        help="first rebalance date (default 2023-01-01 — 3y before end)")
    parser.add_argument("--end", default="2026-06-30",
                        help="last rebalance date (default 2026-06-30)")
    parser.add_argument("--ic-horizon", default="3M",
                        choices=["1M", "3M", "6M", "12M"],
                        help="IC horizon used in the report/summary")
    parser.add_argument("--no-cache", action="store_true",
                        help="ignore any cached panel and rebuild from scratch")
    parser.add_argument("--selected", default=None,
                        help="path to a parent-selection selections.csv; its selected "
                             "candidates are highlighted (starred/bold) in the charts "
                             f"(default auto-detects {_SELECTIONS_CSV})")

    args = parser.parse_args()
    selected_csv = Path(args.selected) if args.selected else None

    if args.audit:
        return _do_audit()
    if args.backfill:
        return _do_backfill()

    # For --build / --validate / --report / --all we need at least the cache
    # path resolved.
    cache_path = CACHE_DIR / cache_key(args.start, args.end, "monthly")
    if args.build:
        _do_build(args.start, args.end, use_cache=not args.no_cache)
        return 0

    if args.validate or args.report:
        panel = _load_panel_or_die(cache_path)
        validation = _do_validate(panel, args.start, args.end, args.ic_horizon)
        if args.report:
            _do_report(panel, validation, args.ic_horizon,
                       new_endpoints=None, backfill_status=None)
        return 0

    if args.charts:
        panel = _load_panel_or_die(cache_path)
        _do_charts(panel, args.ic_horizon, selected_csv)
        return 0

    if args.all:
        cache_path = _do_build(args.start, args.end, use_cache=not args.no_cache)
        panel = _load_panel_or_die(cache_path)
        validation = _do_validate(panel, args.start, args.end, args.ic_horizon)
        _do_report(panel, validation, args.ic_horizon,
                   new_endpoints=_derive_endpoints_used(),
                   backfill_status=_backfill_status())
        _do_charts(panel, args.ic_horizon, selected_csv)
        return 0

    return 0


# Default place the parent-selection expansion run drops its selections.
_SELECTIONS_CSV = Path("output/parent_selection_expansion/selections.csv")


def _load_selected(path: Path | None) -> set[str] | None:
    """Union of parent-selected candidate names from a parent-selection ``selections.csv``.

    Parses the comma-joined ``selected`` column. Returns None (charts stay
    selection-agnostic) when the file is absent so the charts still render without a
    prior parent-selection run."""
    path = path or _SELECTIONS_CSV
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if "selected" not in df.columns:
        return None
    names: set[str] = set()
    for cell in df["selected"].dropna():
        names.update(s.strip() for s in str(cell).split(",") if s.strip())
    return names or None


def _do_charts(panel: CandidatePanel, ic_horizon: str,
               selected_csv: Path | None = None) -> None:
    """Emit the three analysis charts + redundant-pairs CSV."""
    from research.subfactor_expansion import charts

    summary_path = OUT_DIR / f"summary_{ic_horizon}.csv"
    if not summary_path.exists():
        raise SystemExit(
            f"no {summary_path.name} on disk; run --validate --ic-horizon "
            f"{ic_horizon} first"
        )
    summary = pd.read_csv(summary_path)
    selected = _load_selected(selected_csv)
    if selected:
        log.info("highlighting %d parent-selected candidates from %s",
                 len(selected), selected_csv or _SELECTIONS_CSV)
    written = charts.write_all(panel, summary, OUT_DIR / "analysis",
                               horizon=ic_horizon, selected=selected)
    log.info("wrote charts:")
    for name, p in written.items():
        log.info("  %-22s %s", name, p)


def _derive_endpoints_used() -> list[str]:
    return ["FMP /stable/dividends — per-share cash-dividend history"]


def _backfill_status() -> dict[str, str]:
    """Report the row count of the newly-fetched tables so the report shows it."""
    db = get_db()
    div_rows = db.count("historical_dividends")
    div_tickers = db.scalar("SELECT COUNT(DISTINCT ticker) FROM historical_dividends") or 0
    return {
        "historical_dividends": f"{div_rows} rows across {div_tickers} tickers",
    }


if __name__ == "__main__":
    sys.exit(main())
