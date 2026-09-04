"""Reset live-account state to the clean $10k baseline.

The warehouse accumulates state from prior backtest runs ($1M gross
portfolios, simulated rejections, smoke-test approvals) that is
*orthogonal* to running the live $10k workflow on top of the Layer 7
dashboard. This script wipes only that live-account state and leaves
the research warehouse (composite_scores, blended_scores, transcripts,
parent_factor_scores, daily_prices, …) untouched.

Wiped:

* ``portfolio_positions``   — legacy holdings
* ``portfolio_history``     — legacy NAV/exposure snapshots
* ``position_approvals``    — Layer-6 trade audit log
* ``approved_candidates``   — Layer-7 approval pipeline (incl. terminal rows)
* ``output/risk_alerts/pre_trade_rejections.jsonl``
* ``cache/risk_state.json`` — halt/breaker/forced-derisk flags

Run ``python scripts/reset_to_clean_baseline.py --dry-run`` first to
see counts. Add ``--yes`` to actually delete.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from data.db import get_db


LIVE_ACCOUNT_TABLES = (
    "portfolio_positions",
    "portfolio_history",
    "position_approvals",
    "approved_candidates",
)
LIVE_ACCOUNT_FILES = (
    Path("output/risk_alerts/pre_trade_rejections.jsonl"),
    Path("cache/risk_state.json"),
)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--yes", action="store_true",
                   help="Actually delete. Without this, runs in dry-run mode.")
    p.add_argument("--dry-run", action="store_true",
                   help="Report counts only; no writes.")
    args = p.parse_args()

    dry = args.dry_run or not args.yes

    db = get_db()

    print("=" * 60)
    print("Live-account state inventory")
    print("=" * 60)
    for tbl in LIVE_ACCOUNT_TABLES:
        try:
            n = db.scalar(f"SELECT COUNT(*) FROM {tbl}") or 0
        except Exception as exc:                                # noqa: BLE001
            n = f"(table missing: {exc})"
        print(f"  {tbl:<28} {n}")

    for path in LIVE_ACCOUNT_FILES:
        exists = path.exists()
        size = path.stat().st_size if exists else 0
        print(f"  {str(path):<28} "
              f"{'exists, ' + str(size) + ' bytes' if exists else 'absent'}")

    print()
    if dry:
        print("DRY RUN — nothing deleted. Pass --yes to wipe.")
        return 0

    print("=" * 60)
    print("Wiping live-account state…")
    print("=" * 60)

    with db.transaction() as conn:
        for tbl in LIVE_ACCOUNT_TABLES:
            try:
                cur = conn.execute(f"DELETE FROM {tbl}")
                print(f"  {tbl:<28} -{cur.rowcount} rows")
            except Exception as exc:                            # noqa: BLE001
                print(f"  {tbl:<28} skipped ({exc})")

    for path in LIVE_ACCOUNT_FILES:
        if path.exists():
            path.unlink()
            print(f"  {str(path):<28} deleted")
        else:
            print(f"  {str(path):<28} (already absent)")

    print()
    print("Done. The dashboard should now reflect the clean $10k baseline.")
    print("Reload the Streamlit pages (R or the sidebar Refresh button) to "
          "clear cached views.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
