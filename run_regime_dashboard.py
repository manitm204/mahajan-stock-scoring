"""6-Month Regime Dashboard renderer.

Reads ``output/regime5y_focus/per_window.csv`` (produced by
``run_regime5y_analysis.py``) plus daily VIX closes from the local DB, then
writes a chronological heatmap, a VIX-sorted heatmap, and a correlation matrix
to ``output/regime_dashboard/``.

Usage:
    python run_regime_dashboard.py
    python run_regime_dashboard.py --per-window output/regime5y_focus/per_window.csv \\
                                   --out output/regime_dashboard
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

from data.db import get_db
from research.walkforward import regime_dashboard as rd

DEFAULT_IN = Path("output/regime5y_focus/per_window.csv")
DEFAULT_OUT = Path("output/regime_dashboard")


def main() -> int:
    ap = argparse.ArgumentParser(description="6-Month Regime Dashboard")
    ap.add_argument("--per-window", default=str(DEFAULT_IN),
                    help="Path to per_window.csv from run_regime5y_analysis.py")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    per_window = Path(args.per_window)
    if not per_window.exists():
        raise SystemExit(f"missing {per_window} — run run_regime5y_analysis.py first")
    out_dir = Path(args.out)

    print(f"Loading VIX from local DB…")
    with get_db() as db:
        vix = rd.load_vix(db)
    print(f"VIX series: {len(vix)} daily closes ({vix.index[0].date()} → {vix.index[-1].date()})")

    print(f"Assembling per-window frame from {per_window}…")
    frame = rd.build_frame(per_window, vix)
    print(f"Frame: {len(frame)} windows × {len(frame.columns)} columns")

    print(f"Rendering dashboard artefacts to {out_dir}/ …")
    paths = rd.write_dashboard(frame, out_dir)
    for name, path in paths.items():
        print(f"  {name:>18}  {path}")

    beats = int((frame["excess_cagr"].fillna(-1) > 0).sum())
    total = int(frame["excess_cagr"].notna().sum())
    print(f"\nBeat SPY in {beats}/{total} windows. VIX range: "
          f"{frame['avg_vix'].min():.1f} → {frame['avg_vix'].max():.1f}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
