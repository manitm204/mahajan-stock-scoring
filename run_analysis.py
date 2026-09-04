"""Mahajan Hedge Fund - Layer 3 (Qualitative Research Engine) entry point.

Examples::

    # Single-ticker memo
    python run_analysis.py --ticker AAPL

    # Whole-sector pass (one memo per company + sector view)
    python run_analysis.py --sector "Information Technology"

    # Pre-flight cost estimate (no API calls actually made)
    python run_analysis.py --ticker AAPL --estimate-cost

    # Full run: top 25 LONG + top 25 SHORT candidates by Layer 2 composite
    python run_analysis.py --full-run --budget 1.50

The default $1.00 hard budget protects against runaway spend. Override
with ``--budget``; abort behaviour is enforced by :class:`CostTracker`.
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, datetime, timezone, timedelta
from pathlib import Path
from typing import Iterable

from data.config import load_config
from data.db import get_db
from data.utils import get_logger

from analysis import __default_model__
from analysis.api_client import ClaudeClient
from analysis.base import AnalyzerContext
from analysis.cache import AnalysisCache
from analysis.cost_tracker import BudgetExceededError, CostTracker, estimate_cost
from analysis.data_access import list_candidates, list_sector
from analysis.overlay_store import persist_overlays
from analysis.runner import analyze_ticker, write_report
from analysis.sector_analysis import analyze_sector

log = get_logger("run_analysis")

DEFAULT_BUDGET_USD = 1.00


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Mahajan Hedge Fund - Layer 3 qualitative research engine",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    target = p.add_mutually_exclusive_group(required=True)
    target.add_argument("--ticker", help="Analyze a single ticker")
    target.add_argument("--sector", help="Analyze every ticker in a GICS sector")
    target.add_argument("--full-run", action="store_true",
                        help="Top 25 LONG + top 25 SHORT candidates from Layer 2")

    p.add_argument("--budget", type=float, default=DEFAULT_BUDGET_USD,
                   help="Hard $ budget for this run")
    p.add_argument("--model", default=__default_model__,
                   help="Anthropic model ID")
    p.add_argument("--no-cache", action="store_true",
                   help="Skip the SQLite analysis cache (forces re-analysis)")
    p.add_argument("--cache-ttl-days", type=int, default=30,
                   help="Cache TTL in days")
    p.add_argument("--top-n", type=int, default=25,
                   help="(--full-run) Top N per side")
    p.add_argument("--estimate-cost", action="store_true",
                   help="Pre-flight cost estimate only; no API calls")
    p.add_argument("--run-date", help="Override report sub-directory date (YYYY-MM-DD)")
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip tickers that already have a report for the run date "
                        "(output/reports/<run-date>/<TICKER>.md), so a name analyzed "
                        "earlier today is not re-generated.")
    p.add_argument("--no-report", action="store_true",
                   help="Run analyzers but skip writing markdown memos")
    p.add_argument("--no-sector-view", action="store_true",
                   help="(--sector / --full-run) skip the cross-company synthesis call")
    p.add_argument("--no-persist-overlay", action="store_true",
                   help="Skip writing research_overlays rows (display only)")
    p.add_argument("--skip-if-fresh", action="store_true",
                   help="Skip tickers whose research_overlays row is recent (see "
                        "--fresh-days) AND no material events have arrived since "
                        "that analysis: earnings, institutional filings, insider "
                        "transactions, or SEC forms (8-K/10-Q/10-K/SC 13D/SC 13G).")
    p.add_argument("--fresh-days", type=int, default=7,
                   help="(--skip-if-fresh) Max age in days for an overlay to be "
                        "considered fresh. Default 7.")
    p.add_argument("--long-only", action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Cost estimation (no API calls)
# ---------------------------------------------------------------------------
# Rough per-analyzer token budgets derived from the prompt sizes and typical
# response lengths. These are deliberately conservative so the operator sees
# the worst-case figure before opting in.
ESTIMATED_TOKENS_PER_CALL = {
    "earnings": (20_000, 1_500),   # transcript + system prompt; ~1.5k JSON out
    "filing":   ( 3_000, 1_200),   # 8q metrics table + system
    "risk":     (25_000, 1_500),   # two 10-K sections diffed
    "insider":  ( 2_500, 700),
    "overlay":  ( 5_000, 1_500),   # quant context + 4 analyzer JSONs
    "sector":   ( 6_000, 1_500),
}


def estimate_total_cost(model: str, n_tickers: int,
                        include_sector_calls: int = 0) -> float:
    """Sum the per-analyzer estimates for ``n_tickers`` names."""
    per_ticker = 0.0
    for analyzer in ("earnings", "filing", "risk", "insider", "overlay"):
        in_t, out_t = ESTIMATED_TOKENS_PER_CALL[analyzer]
        per_ticker += estimate_cost(model, in_t, out_t)
    total = per_ticker * n_tickers
    if include_sector_calls > 0:
        in_t, out_t = ESTIMATED_TOKENS_PER_CALL["sector"]
        total += include_sector_calls * estimate_cost(model, in_t, out_t)
    return total


# ---------------------------------------------------------------------------
# Resolution: figure out the target ticker set up front
# ---------------------------------------------------------------------------
def resolve_tickers(args: argparse.Namespace, db) -> tuple[list[str], list[str]]:
    """Return ``(tickers, sectors_for_view)``."""
    if args.ticker:
        return [args.ticker.upper()], []
    if args.sector:
        scores = list_sector(db, args.sector)
        if not scores:
            log.error("No tickers found for sector %r", args.sector)
            return [], []
        return [s.ticker for s in scores], [args.sector]
    # --full-run
    longs  = list_candidates(db, "LONG",  limit=args.top_n)
    shorts = list_candidates(db, "SHORT", limit=args.top_n)

    if (args.long_only):
        tickers = [s.ticker for s in longs]
    else: 
        tickers = [s.ticker for s in longs] + [s.ticker for s in shorts]
    # Deduplicate while preserving order.
    seen, ordered = set(), []
    for t in tickers:
        if t not in seen:
            seen.add(t)
            ordered.append(t)
    sectors = sorted({s.sector for s in longs + shorts})
    return ordered, sectors


def filter_existing_reports(cfg, tickers: list[str],
                            run_date: str | None) -> tuple[list[str], list[str]]:
    """Split ``tickers`` into (to_analyze, already_done) by report presence.

    A ticker is "already done" when ``output/reports/<run_date>/<TICKER>.md``
    exists, mirroring :func:`analysis.report_generator.save_memo`'s layout.
    """
    day = run_date or date.today().isoformat()
    report_dir = Path(cfg.root) / "output" / "reports" / day
    to_run, done = [], []
    for t in tickers:
        if (report_dir / f"{t}.md").exists():
            done.append(t)
        else:
            to_run.append(t)
    return to_run, done


# ---------------------------------------------------------------------------
# Event-aware freshness filter
# ---------------------------------------------------------------------------
# Form types that represent material new information worth re-analyzing.
_MATERIAL_FORMS = ("8-K", "10-Q", "10-K", "SC 13D", "SC 13G", "SC 13D/A", "SC 13G/A")


def filter_fresh_tickers(
    db,
    tickers: list[str],
    fresh_days: int,
) -> tuple[list[str], list[str]]:
    """Return ``(to_analyze, skipped_fresh)``.

    A ticker is *fresh* (skipped) when ALL of the following hold:

    1. A ``research_overlays`` row exists computed within ``fresh_days`` days.
    2. No new earnings transcript has been loaded since that analysis.
    3. No earnings calendar result (``eps_actual`` filled) appeared since then.
    4. No institutional ownership report arrived since then.
    5. No insider transactions were recorded since then.
    6. No material SEC filing (8-K, 10-Q, 10-K, SC 13D/G) was filed since then.
    """
    cutoff = datetime.now(tz=timezone.utc) - timedelta(days=fresh_days)
    cutoff_iso = cutoff.isoformat()

    to_run: list[str] = []
    fresh: list[str] = []

    for ticker in tickers:
        # Step 1: find the most recent overlay within the fresh window.
        row = db.query_one(
            "SELECT computed_at FROM research_overlays "
            "WHERE ticker = ? AND computed_at >= ? "
            "ORDER BY computed_at DESC LIMIT 1",
            (ticker, cutoff_iso),
        )
        if row is None:
            # No recent overlay → always re-analyze.
            to_run.append(ticker)
            continue

        last_at = row["computed_at"]  # ISO 8601 string — safe for text comparison

        # Step 2-6: check for any new material events since last_at.
        has_new_transcript = db.scalar(
            "SELECT 1 FROM transcripts WHERE ticker = ? AND call_date > ? LIMIT 1",
            (ticker, last_at[:10]),  # call_date is DATE; compare to date portion
        )
        has_new_earnings = db.scalar(
            "SELECT 1 FROM earnings_calendar "
            "WHERE ticker = ? AND earnings_date > ? AND eps_actual IS NOT NULL LIMIT 1",
            (ticker, last_at[:10]),
        )
        has_new_institutional = db.scalar(
            "SELECT 1 FROM institutional_ownership_summary "
            "WHERE ticker = ? AND report_date > ? LIMIT 1",
            (ticker, last_at[:10]),
        )
        has_new_insider = db.scalar(
            "SELECT 1 FROM insider_transactions "
            "WHERE ticker = ? AND transaction_date > ? AND transaction_date <= date('now') LIMIT 1",
            (ticker, last_at[:10]),
        )
        placeholders = ",".join("?" * len(_MATERIAL_FORMS))
        has_new_filing = db.scalar(
            f"SELECT 1 FROM sec_filings "
            f"WHERE ticker = ? AND filing_date > ? AND form_type IN ({placeholders}) LIMIT 1",
            (ticker, last_at[:10], *_MATERIAL_FORMS),
        )

        if any([has_new_transcript, has_new_earnings, has_new_institutional,
                has_new_insider, has_new_filing]):
            to_run.append(ticker)
        else:
            fresh.append(ticker)

    return to_run, fresh


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    args = parse_args()
    cfg = load_config()
    log_path = cfg.log_file

    with get_db() as db:
        tickers, sectors = resolve_tickers(args, db)
        if not tickers:
            print("No candidate tickers to analyze.")
            return 1

        if args.skip_existing:
            tickers, already = filter_existing_reports(cfg, tickers, args.run_date)
            if already:
                day = args.run_date or date.today().isoformat()
                print(f"Skipping {len(already)} ticker(s) already reported on {day}: "
                      f"{', '.join(already)}")
            if not tickers:
                print("All candidate tickers already have reports for the run date.")
                return 0

        if args.skip_if_fresh:
            tickers, stale_skipped = filter_fresh_tickers(db, tickers, args.fresh_days)
            if stale_skipped:
                print(f"Skipping {len(stale_skipped)} ticker(s) with a fresh overlay "
                      f"(≤{args.fresh_days}d old) and no new earnings/institutional/"
                      f"insider/filing events: {', '.join(stale_skipped)}")
            if not tickers:
                print("All candidate tickers are fresh — nothing to re-analyze.")
                return 0

        # ----------- pre-flight estimate ---------------------------------
        n_sector_calls = 0 if (args.no_sector_view or args.ticker) else len(sectors)
        est = estimate_total_cost(args.model, len(tickers), n_sector_calls)
        print(f"Layer 3 plan")
        print(f"  model              : {args.model}")
        print(f"  tickers            : {len(tickers)}")
        print(f"  sector synth calls : {n_sector_calls}")
        print(f"  estimated cost     : ${est:.4f}")
        print(f"  budget cap         : ${args.budget:.2f}")
        if args.estimate_cost:
            print("--estimate-cost set; exiting without making API calls.")
            return 0
        if est > args.budget:
            print(f"ABORT: estimated cost ${est:.4f} exceeds budget ${args.budget:.2f}.")
            print("       Re-run with --budget to raise the cap.")
            return 2

        # ----------- build shared services -------------------------------
        tracker = CostTracker(budget_usd=args.budget)
        client = ClaudeClient(model=args.model, cost_tracker=tracker)
        cache  = None if args.no_cache else AnalysisCache(ttl_days=args.cache_ttl_days)
        ctx = AnalyzerContext(client=client, cache=cache)

        try:
            _run_pipeline(db, ctx, tickers, sectors, args)
        except BudgetExceededError as e:
            log.error("Budget exceeded mid-run: %s", e)
            print(f"\nABORTED: {e}")
        finally:
            print()
            print(tracker.format_summary())
            if cache is not None:
                cache.close()

    return 0


def _run_pipeline(db, ctx: AnalyzerContext, tickers: Iterable[str],
                  sectors: list[str], args: argparse.Namespace) -> None:
    analyses = []
    sector_views: dict[str, dict | None] = {}

    # Per-ticker pass.
    for t in tickers:
        log.info("Analyzing %s ...", t)
        result = analyze_ticker(db, ctx, t)
        analyses.append(result)
        print(_one_line_status(result))

    # Sector synthesis pass (one call per sector).
    if not args.no_sector_view and sectors and not args.ticker:
        for sec in sectors:
            in_sec = [a for a in analyses if a.composite and a.composite.sector == sec]
            if not in_sec:
                continue
            overlays = {a.ticker: a.overlay for a in in_sec}
            view = analyze_sector(
                ctx, sec, [a.composite for a in in_sec], overlays,
            )
            sector_views[sec] = view

    # Reports.
    if not args.no_report:
        written = 0
        for a in analyses:
            sec = a.composite.sector if a.composite else None
            path = write_report(a, sector_view=sector_views.get(sec) if sec else None,
                                run_date=args.run_date)
            if path:
                written += 1
        print(f"\nWrote {written} memo(s).")

    # Persist research overlays for the dashboard / approvals step. The quant
    # composite is the only ranking score; overlays are informational.
    if not args.no_persist_overlay:
        n = persist_overlays(db, analyses, model=ctx.model)
        print(f"Persisted {n} research_overlays row(s).")


def _one_line_status(result) -> str:
    if result.composite is None:
        return f"  {result.ticker:6s} - skipped (no composite)"
    o = result.overlay or {}
    parts = [
        f"quant={result.composite.composite_score:5.1f}",
        f"status={o.get('final_research_status') or 'n/a'}",
        f"review={o.get('quant_signal_review') or 'n/a'}",
        f"qual_risk={o.get('qualitative_risk_level') or 'n/a'}",
    ]
    return f"  {result.ticker:6s} - " + "  ".join(parts)


if __name__ == "__main__":
    sys.exit(main())
