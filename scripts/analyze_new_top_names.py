#!/usr/bin/env python
"""Gate the Layer-3 LLM analysis to only the names that actually need it.

Run this *after* ``run_data.py`` + ``run_scoring.py``. It looks at the latest
composite scores and runs ``run_analysis.py`` for:

  * NEW names in the top-N long **or** short book that have no memo anywhere in
    this past week's report folders (``output/reports/<date>/<TICKER>.md``), and
  * names that reported EARNINGS in the recent window (``eps_actual`` filled),
    which are re-analyzed fresh (``--no-cache``) even if already covered.

Everything else in the top-N already has a memo from earlier in the week and is
left alone, so you don't re-pay for LLM coverage.

Why this exists: ``run_analysis.py --skip-existing`` only dedupes against the
*current* run-date folder. When the score date rolls to a new day that folder is
empty, so a plain ``--full-run --skip-existing`` re-analyzes all 2*N names. This
wrapper dedupes against the whole week and adds the earnings-rerun rule.

Examples
--------
    # See the plan without spending anything:
    python scripts/analyze_new_top_names.py --dry-run

    # Actually run it (memos land in output/reports/<score_date>/):
    python scripts/analyze_new_top_names.py --budget 5
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path

# Allow running as `python scripts/analyze_new_top_names.py` from repo root.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.data_access import list_candidates  # noqa: E402
from data.db import get_db  # noqa: E402


def _covered_this_week(score_date: str, week_days: int) -> dict[str, str]:
    """Map ticker -> most-recent memo date within the trailing ``week_days``."""
    anchor = date.fromisoformat(score_date)
    covered: dict[str, str] = {}
    for i in range(week_days + 1):
        day = (anchor - timedelta(days=i)).isoformat()
        folder = ROOT / "output" / "reports" / day
        if not folder.exists():
            continue
        for memo in folder.glob("*.md"):
            covered.setdefault(memo.stem, day)  # newest first -> keep most recent
    return covered


def _earnings_names(db, score_date: str, earnings_days: int) -> dict[str, str]:
    """Candidates that reported earnings (eps_actual set) in the recent window."""
    anchor = date.fromisoformat(score_date)
    lo = (anchor - timedelta(days=earnings_days)).isoformat()
    rows = db.query(
        "SELECT ticker, earnings_date FROM earnings_calendar "
        "WHERE earnings_date BETWEEN ? AND ? AND eps_actual IS NOT NULL",
        (lo, score_date),
    )
    return {r["ticker"]: r["earnings_date"] for r in rows}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--top-n", type=int, default=15,
                   help="Top N per side (long AND short). Default 15.")
    p.add_argument("--week-days", type=int, default=7,
                   help="Trailing window (days) that counts as 'covered this "
                        "week'. Default 7.")
    p.add_argument("--earnings-days", type=int, default=1,
                   help="Lookback (days from score date) for the earnings-rerun "
                        "rule. Default 1 (i.e. reported yesterday).")
    p.add_argument("--budget", type=float, default=5.0,
                   help="Hard $ budget passed to run_analysis.py. Default 5.")
    p.add_argument("--model", default=None, help="Override Anthropic model ID.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the plan and exit without any API calls.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    with get_db() as db:
        row = db.query_one("SELECT MAX(as_of_date) AS d FROM composite_scores")
        score_date = row["d"] if row else None
        if not score_date:
            print("No composite_scores found. Run run_scoring.py first.")
            return 1

        longs = [c.ticker for c in list_candidates(db, "LONG", args.top_n, score_date)]
        shorts = [c.ticker for c in list_candidates(db, "SHORT", args.top_n, score_date)]
        candidates = list(dict.fromkeys(longs + shorts))

        covered = _covered_this_week(score_date, args.week_days)
        earnings = _earnings_names(db, score_date, args.earnings_days)

    new_names = [t for t in candidates if t not in covered]
    earn_reruns = [t for t in candidates if t in earnings]
    # New names run normally; earnings names force a fresh (uncached) analysis.
    to_run = list(dict.fromkeys(new_names + earn_reruns))
    skipped = [t for t in candidates if t not in to_run]

    print(f"Score date        : {score_date}")
    print(f"Top-{args.top_n} long     : {', '.join(longs)}")
    print(f"Top-{args.top_n} short    : {', '.join(shorts)}")
    print(f"Covered this week : {len(covered)} memo(s) in trailing {args.week_days}d")
    print("-" * 60)
    print(f"NEW (uncovered)   : {', '.join(new_names) or '(none)'}")
    print(f"EARNINGS rerun    : "
          f"{', '.join(f'{t}@{earnings[t]}' for t in earn_reruns) or '(none)'}")
    print(f"--> WILL ANALYZE  : {', '.join(to_run) or '(none)'}")
    print(f"Already covered   : {', '.join(skipped) or '(none)'}")
    print("-" * 60)

    if not to_run:
        print("Nothing to analyze. Done.")
        return 0
    if args.dry_run:
        print("--dry-run set; exiting without API calls.")
        return 0

    rc = 0
    for t in to_run:
        cmd = [sys.executable, str(ROOT / "run_analysis.py"),
               "--ticker", t, "--run-date", score_date, "--budget", str(args.budget)]
        if args.model:
            cmd += ["--model", args.model]
        if t in earnings:  # earnings names: bypass the 30-day analysis cache
            cmd += ["--no-cache"]
        print(f"\n=== run_analysis {t}"
              f"{' (earnings, --no-cache)' if t in earnings else ''} ===")
        rc |= subprocess.call(cmd, cwd=str(ROOT))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
