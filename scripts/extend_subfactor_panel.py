"""Extend the cached subfactor candidate panel with new monthly dates,
instead of rebuilding the whole 2015-2026 history from scratch.

Each rebalance date's candidate frame is built independently (its own
point-in-time DataContext), so new dates can just be scored and merged into
the existing cache -- no need to recompute anything before the old cache's
last date.

Usage: python scripts/extend_subfactor_panel.py --end 2026-09-04
Writes: cache/subfactor_expansion/cand_panel_<start>_<end>_monthly_v2.pkl
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from data.config import load_config
from data.db import get_db

from research.subfactor_expansion.panel import (
    CandidatePanel, build_candidate_panel, cache_key, load_cached_panel, save_cached_panel,
)
from run_subfactor_expansion import _monthly_rebalances

OLD_PANEL = REPO / "cache" / "subfactor_expansion" / "cand_panel_2015-06-30_2026-06-30_monthly_v2.pkl"
START = "2015-06-30"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--end", required=True, help="new end date, e.g. 2026-09-04")
    args = ap.parse_args()

    old = load_cached_panel(OLD_PANEL)
    if old is None:
        raise SystemExit(f"base panel not found: {OLD_PANEL}")
    print(f"loaded base panel: {len(old.rebal_dates)} dates, "
         f"{old.rebal_dates[0]}..{old.rebal_dates[-1]}")

    db = get_db()
    cfg = load_config()
    all_months = _monthly_rebalances(db, START, args.end)
    new_dates = [d for d in all_months if d > old.rebal_dates[-1]]
    if not new_dates:
        raise SystemExit(f"no new dates after {old.rebal_dates[-1]} through {args.end}")
    print(f"building {len(new_dates)} new dates: {new_dates}")

    new = build_candidate_panel(db, new_dates, cfg)

    merged_candidates_by_parent = {p: list(old.candidates_by_parent.get(p, []))
                                   for p in old.candidates_by_parent}
    for p, subs in new.candidates_by_parent.items():
        existing = set(merged_candidates_by_parent.setdefault(p, []))
        merged_candidates_by_parent[p] += [s for s in subs if s not in existing]

    merged = CandidatePanel(
        rebal_dates=sorted(set(old.rebal_dates) | set(new.rebal_dates)),
        scores={**old.scores, **new.scores},
        raws={**old.raws, **new.raws},
        candidates_by_parent=merged_candidates_by_parent,
        universe=sorted(set(old.universe) | set(new.universe)),
        parent_by_candidate={**old.parent_by_candidate, **new.parent_by_candidate},
        higher_by_candidate={**old.higher_by_candidate, **new.higher_by_candidate},
    )

    out_path = OLD_PANEL.parent / cache_key(START, args.end, "monthly")
    save_cached_panel(merged, out_path)
    print(f"wrote {out_path}: {len(merged.rebal_dates)} dates, "
         f"{merged.rebal_dates[0]}..{merged.rebal_dates[-1]}")


if __name__ == "__main__":
    main()
