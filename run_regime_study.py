"""Semiannual walk-forward regime study — rolling 5y vs rolling 2y vs hybrid blends.

Rebuilds the entire Parent-Selection-V4 construction (sub-factor selection, intra-parent
weights, parent construction, parent/composite weights) on train-only data for every
6-month test window (H1/H2) rolled forward from 2017 through the latest available half-
year, under four training-window policies:

    rolling5y     — trailing 5-year window
    rolling2y     — trailing 2-year window
    blend_comp    — 50/50 average of the 5y and 2y composite scores
    blend_parent  — 50/50 average of the 5y and 2y frozen configs

Writes CSVs, PNG charts, and a top-level ``REGIME_REPORT.md`` to
``output/regime_study/``. Point-in-time universe throughout, selection forward-return
windows capped 6 months before each test-start boundary (no look-ahead). Read-only; no
mutation of production factors, config, or DB.

Usage:
    python run_regime_study.py                 # full run using the cached panel
    python run_regime_study.py --rebuild-panel # re-score the candidate panel first
    python run_regime_study.py --out output/regime_study
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

from backtesting import data_loader as dl
from data.db import get_db
from research.walkforward import regime_report
from research.walkforward.regime_study import build_summaries, run_regime_study
from research.walkforward.splits import PANEL_END, PANEL_START, LAST_TEST_END

# Reuse the panel loader from run_walkforward — it already handles caching PIT-safely.
from run_walkforward import PRICE_END, _load_panel

OUT_DIR = Path("output/regime_study")


def main() -> int:
    ap = argparse.ArgumentParser(description="Semiannual walk-forward regime study")
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

    print("\n=== Semiannual regime walk-forward "
          f"(first {args.first_test_year}, last-end {args.last_end}) ===")
    run = run_regime_study(panel, matrix, sectors,
                           first_test_year=args.first_test_year,
                           last_end=args.last_end)
    if not any(run.windows.values()):
        raise SystemExit("no usable semiannual windows — check data span / panel coverage")

    print("\n=== Aggregating (per-window / 3-year buckets / full pool) ===")
    summaries = build_summaries(run)

    print("\n=== Writing reports + charts ===")
    regime_report.write_reports(run, summaries, out_dir)

    port = summaries["full_portfolio"]
    if not port.empty:
        best = port[port["top_pct"] == 0.20].sort_values("sharpe", ascending=False)
        print("\nTop-20 % pooled OOS by Sharpe:")
        for _, r in best.iterrows():
            print(f"  {r['policy']:<14s}  Sharpe {r.get('sharpe', float('nan')):+.2f}  "
                  f"CAGR {r.get('cagr', float('nan')):+.1%}  "
                  f"SPY-excess {r.get('spy_excess_cagr', float('nan')):+.1%}  "
                  f"maxDD {r.get('max_drawdown', float('nan')):+.1%}")

    print(f"\nWrote reports to {out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
