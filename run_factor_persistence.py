"""Factor Persistence & Regime Analysis entry point.

Loads the cached subfactor-expansion panel, computes per-year IC / IR / spread /
coverage / VIX-regime statistics for every subfactor and parent, and writes all
outputs (CSVs + heatmap PNGs + FACTOR_PERSISTENCE_REPORT.md) to
``output/factor_persistence/``.

Usage:
    python run_factor_persistence.py
    python run_factor_persistence.py --rebuild-panel   # re-score first
    python run_factor_persistence.py --first-year 2018 --last-year 2025
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import pandas as pd

from backtesting import data_loader as dl
from data.config import load_config
from data.db import get_db
from research.subfactor_expansion.panel import (build_candidate_panel,
                                                cache_key as exp_cache_key,
                                                load_cached_panel as exp_load,
                                                save_cached_panel as exp_save)
from research.panel import ScorePanel
from research.walkforward.splits import PANEL_START, PANEL_END
from research.walkforward.vix_regime_study import load_vix_series
from research.walkforward.factor_persistence import run_persistence_analysis, write_outputs

CACHE_DIR = Path("cache/subfactor_expansion")
OUT_DIR = Path("output/factor_persistence")
PRICE_END = "2026-07-06"


def _adapt(cand) -> ScorePanel:
    parent_keys = [p for p, subs in cand.candidates_by_parent.items() if subs]
    return ScorePanel(
        rebal_dates=list(cand.rebal_dates),
        scores=cand.scores,
        parent_keys=parent_keys,
        sub_by_parent={p: list(cand.candidates_by_parent[p]) for p in parent_keys},
        universe=list(cand.universe),
    )


def _month_ends(db, start: str, end: str) -> list[str]:
    df = db.query_df(
        "SELECT DISTINCT date FROM daily_prices WHERE date >= ? AND date <= ? ORDER BY date",
        (start, end),
    )
    if df.empty:
        return []
    dates = pd.to_datetime(df["date"])
    return [str(g.max().date()) for _, g in dates.groupby(dates.dt.to_period("M"))]


def _load_panel(rebuild: bool) -> ScorePanel:
    ckey = CACHE_DIR / exp_cache_key(PANEL_START, PANEL_END, "monthly")
    cand = None if rebuild else exp_load(ckey)
    if cand is None:
        cfg = load_config()
        with get_db() as db:
            rebals = _month_ends(db, PANEL_START, PANEL_END)
            print(f"Building PIT candidate panel over {len(rebals)} rebalances "
                  f"({rebals[0]} → {rebals[-1]})…")
            cand = build_candidate_panel(db, rebals, cfg, verbose=True, keep_raw=False)
        exp_save(cand, ckey)
    else:
        print(f"Loaded candidate panel: {len(cand.rebal_dates)} rebalances "
              f"({cand.rebal_dates[0]}→{cand.rebal_dates[-1]}), "
              f"{len(cand.all_candidates)} candidates, {len(cand.universe)} PIT-union names.")
    return _adapt(cand)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rebuild-panel", action="store_true",
                        help="Re-score the candidate panel from scratch")
    parser.add_argument("--first-year", type=int, default=2017,
                        help="First calendar year to include (default 2017)")
    parser.add_argument("--last-year", type=int, default=2026,
                        help="Last calendar year to include (default 2026)")
    args = parser.parse_args()

    print("=== Factor Persistence & Regime Analysis ===\n")

    # Load panel
    print("[1/4] Loading subfactor panel…")
    panel = _load_panel(args.rebuild_panel)

    # Load price matrix and VIX
    print("[2/4] Loading price matrix and VIX…")
    with get_db() as db:
        matrix = dl.load_price_matrix(db, panel.universe, PANEL_START, PRICE_END)
        vix = load_vix_series(db)

    print(f"      Price matrix: {matrix.shape[0]} dates × {matrix.shape[1]} tickers")
    print(f"      VIX series: {len(vix)} days ({vix.index[0].date()} → {vix.index[-1].date()})")

    # Run analysis
    print(f"\n[3/4] Computing per-year stats ({args.first_year}–{args.last_year})…")
    results = run_persistence_analysis(
        panel, matrix, vix,
        first_year=args.first_year,
        last_year=args.last_year,
        verbose=True,
    )

    # Write outputs
    print(f"\n[4/4] Writing outputs to {OUT_DIR}/ …")
    write_outputs(results, OUT_DIR, verbose=True)

    print("\nDone.  Key files:")
    print(f"  {OUT_DIR}/FACTOR_PERSISTENCE_REPORT.md")
    print(f"  {OUT_DIR}/subfactor_persistence_table.csv")
    print(f"  {OUT_DIR}/parent_persistence_table.csv")
    print(f"  {OUT_DIR}/heatmap_subfactor_ic.png")
    print(f"  {OUT_DIR}/heatmap_parent_ic.png")
    print(f"  {OUT_DIR}/heatmap_regime_subfactor_ic.png")


if __name__ == "__main__":
    main()
