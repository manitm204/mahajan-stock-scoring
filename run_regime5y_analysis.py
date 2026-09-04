"""Regime-by-regime performance analysis focused on the rolling-5y model.

Re-runs the semiannual walk-forward for the rolling-5y policy only, captures a rich
per-window metric bundle (portfolio + SPY-standalone + composite IC + Q5-Q1 spread),
then writes six diagnostic charts and three summary tables (per-window / 3-year buckets /
full period) to ``output/regime5y_focus/``.

Usage:
    python run_regime5y_analysis.py                 # full run using the cached panel
    python run_regime5y_analysis.py --rebuild-panel
    python run_regime5y_analysis.py --out output/regime5y_focus
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

from backtesting import data_loader as dl
from data.db import get_db
from research.walkforward import regime5y_report
from research.walkforward.regime5y_focus import build_focused_summary, run_focused_5y
from research.walkforward.splits import LAST_TEST_END, PANEL_END, PANEL_START

from run_walkforward import PRICE_END, _load_panel

OUT_DIR = Path("output/regime5y_focus")


def main() -> int:
    ap = argparse.ArgumentParser(description="Regime-by-regime rolling-5y analysis")
    ap.add_argument("--rebuild-panel", action="store_true")
    ap.add_argument("--out", default=str(OUT_DIR))
    ap.add_argument("--first-test-year", type=int, default=2017)
    ap.add_argument("--last-end", default=LAST_TEST_END)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading candidate panel ({PANEL_START} → {PANEL_END})…")
    panel = _load_panel(args.rebuild_panel)

    with get_db() as db:
        matrix = dl.load_price_matrix(db, panel.universe, PANEL_START, PRICE_END)
        sectors = dl.global_sectors(db)
    print(f"Price matrix: {matrix.shape[1]} tickers × {matrix.shape[0]} days; "
          f"sectors for {sectors.notna().sum()} names.")

    print(f"\n=== Rolling-5y focused walk-forward "
          f"({args.first_test_year} → {args.last_end}) ===")
    run = run_focused_5y(panel, matrix, sectors,
                         first_test_year=args.first_test_year,
                         last_end=args.last_end)
    if not run.windows:
        raise SystemExit("no usable semiannual windows")

    print("\n=== Aggregating (per-window / 3y buckets / full) ===")
    summary = build_focused_summary(run)

    print("\n=== Writing charts + report ===")
    regime5y_report.write_focused_reports(run, summary, out_dir)

    pw = summary["per_window"]
    if not pw.empty:
        beats = int((pw["excess_cagr"].fillna(-1) > 0).sum())
        total = int(pw["excess_cagr"].notna().sum())
        print(f"\nBeat SPY in {beats}/{total} test windows.")
    print(f"Wrote reports to {out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
