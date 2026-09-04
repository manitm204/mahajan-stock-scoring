"""Factor Research & Validation Framework — entry point.

Scores every parent factor and sub-factor over a monthly point-in-time rebalance
grid, measures predictive power (quintiles, IC) at 1/3/6/12-month horizons, tests
sub-factor redundancy and incremental contribution, and classifies each
sub-factor Keep / Merge / Remove / Insufficient Data. Writes CSVs, charts and a
markdown report to ``output/factor_research/``.

Usage:
    python run_factor_research.py
    python run_factor_research.py --start 2022-01-01 --end 2026-06-01 --freq monthly
    python run_factor_research.py --no-cache        # force a fresh re-score
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from backtesting import data_loader as dl
from data.config import load_config
from data.db import get_db

from research import (
    HORIZON_MONTHS, Thresholds, aggregate_quintiles, classify_subfactors,
    compute_forward_returns, ic_timeseries, incremental_metrics, max_sibling_corr,
    parent_score_correlation, summarize_ic,
)
from research.classify import redundancy_watch
from research.panel import (
    build_score_panel, cache_key, load_cached_panel, save_cached_panel,
)
from research import report as report_mod

DEFAULT_START = "2022-01-01"
DEFAULT_END = "2026-06-01"
DEFAULT_FREQ = "monthly"
OUT_DIR = Path("output/factor_research")
HORIZONS = list(HORIZON_MONTHS.keys())   # 1M, 3M, 6M, 12M


def _series_map(df: pd.DataFrame, key: str, value: str) -> dict:
    if df.empty or key not in df.columns or value not in df.columns:
        return {}
    return dict(zip(df[key], df[value]))


def signal_coverage(panel, signals: list[str]) -> dict[str, float]:
    """Average non-neutral (score != 50) fraction per signal across rebalances."""
    sums = {s: [] for s in signals}
    for d in panel.rebal_dates:
        frame = panel.scores[d]
        for s in signals:
            if s in frame.columns:
                col = frame[s].dropna()
                if len(col):
                    sums[s].append(float((col.sub(50.0).abs() > 1e-6).mean()))
    return {s: (float(np.mean(v)) if v else float("nan")) for s, v in sums.items()}


def build_factor_table(panel, quint_by_h, ic_by_h) -> pd.DataFrame:
    ft = pd.DataFrame({"parent": panel.parent_keys})
    ic1 = ic_by_h["1M"]
    ft["n_periods"] = ft["parent"].map(_series_map(ic1, "signal", "n_periods"))
    for h in HORIZONS:
        ft[f"ic_{h}"] = ft["parent"].map(_series_map(ic_by_h[h], "signal", "mean_ic"))
        ft[f"spread_{h}"] = ft["parent"].map(
            _series_map(quint_by_h[h], "signal", "spread_q5_q1"))
    ft["ir_1M"] = ft["parent"].map(_series_map(ic1, "signal", "information_ratio"))
    ft["hit_1M"] = ft["parent"].map(_series_map(ic1, "signal", "hit_rate"))
    ft["stability_1M"] = ft["parent"].map(_series_map(ic1, "signal", "stability"))
    ft["mono_1M"] = ft["parent"].map(_series_map(quint_by_h["1M"], "signal", "monotonicity"))
    return ft.sort_values("ir_1M", ascending=False, na_position="last").reset_index(drop=True)


def build_subfactor_table(panel, quint_by_h, ic_by_h, redundancy_df,
                          incremental_df, coverage) -> pd.DataFrame:
    st = pd.DataFrame({"sub_factor": panel.all_subs})
    st["parent"] = st["sub_factor"].map(panel.parent_of)
    ic1 = ic_by_h["1M"]
    st["n_periods"] = st["sub_factor"].map(_series_map(ic1, "signal", "n_periods"))
    st["coverage"] = st["sub_factor"].map(coverage)
    st["ic_1M"] = st["sub_factor"].map(_series_map(ic1, "signal", "mean_ic"))
    st["ir_1M"] = st["sub_factor"].map(_series_map(ic1, "signal", "information_ratio"))
    st["hit_1M"] = st["sub_factor"].map(_series_map(ic1, "signal", "hit_rate"))
    st["stability_1M"] = st["sub_factor"].map(_series_map(ic1, "signal", "stability"))
    for h in ["3M", "6M", "12M"]:
        st[f"ic_{h}"] = st["sub_factor"].map(_series_map(ic_by_h[h], "signal", "mean_ic"))
    st["spread_1M"] = st["sub_factor"].map(
        _series_map(quint_by_h["1M"], "signal", "spread_q5_q1"))
    st["monotonicity"] = st["sub_factor"].map(
        _series_map(quint_by_h["1M"], "signal", "monotonicity"))
    st["max_abs_sibling_corr"] = st["sub_factor"].map(
        _series_map(redundancy_df, "sub_factor", "max_abs_sibling_corr"))
    st["closest_sibling"] = st["sub_factor"].map(
        _series_map(redundancy_df, "sub_factor", "closest_sibling"))
    st["standalone_ic"] = st["sub_factor"].map(
        _series_map(incremental_df, "sub_factor", "standalone_ic"))
    st["incremental_ic"] = st["sub_factor"].map(
        _series_map(incremental_df, "sub_factor", "incremental_ic"))
    st["delta_parent_ic"] = st["sub_factor"].map(
        _series_map(incremental_df, "sub_factor", "delta_parent_ic"))
    return st


def main() -> None:
    ap = argparse.ArgumentParser(description="Factor research & validation framework")
    ap.add_argument("--start", default=DEFAULT_START)
    ap.add_argument("--end", default=DEFAULT_END)
    ap.add_argument("--freq", default=DEFAULT_FREQ,
                    choices=["weekly", "monthly", "quarterly"])
    ap.add_argument("--out", default=str(OUT_DIR))
    ap.add_argument("--no-cache", action="store_true",
                    help="ignore any cached score panel and re-score")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_config()

    with get_db() as db:
        universe = db.universe_tickers()
        trading = dl.trading_calendar(db)
        matrix = dl.load_price_matrix(db, universe, args.start, args.end)
        if matrix.empty:
            raise RuntimeError("No price data in the requested date range.")
        rebal_dates = dl.generate_rebalance_dates(trading, args.start, args.end, args.freq)
        rebal_dates = [d for d in rebal_dates if d in matrix.index]
        if len(rebal_dates) < 4:
            raise RuntimeError("Need at least four rebalances for forward returns.")

        cache_path = out_dir / cache_key(args.start, args.end, args.freq)
        panel = None if args.no_cache else load_cached_panel(cache_path)
        if panel is None:
            print(f"Scoring {len(rebal_dates)} rebalances "
                  f"({rebal_dates[0]} → {rebal_dates[-1]})...")
            panel = build_score_panel(db, rebal_dates, cfg)
            save_cached_panel(panel, cache_path)
        else:
            print(f"Loaded cached score panel ({len(panel.rebal_dates)} rebalances).")

    # ---- forward returns & per-horizon metrics ----------------------------
    fwd = compute_forward_returns(matrix, panel.rebal_dates, HORIZON_MONTHS)
    all_signals = panel.parent_keys + panel.all_subs

    quint_by_h: dict[str, pd.DataFrame] = {}
    ic_by_h: dict[str, pd.DataFrame] = {}
    ic_ts_all: list[pd.DataFrame] = []
    quint_long: list[pd.DataFrame] = []
    for h in HORIZONS:
        quint_by_h[h] = aggregate_quintiles(panel, fwd[h], all_signals)
        quint_long.append(quint_by_h[h].assign(horizon=h))
        ts = ic_timeseries(panel, fwd[h], all_signals, horizon=h)
        ic_ts_all.append(ts)
        ic_by_h[h] = summarize_ic(ts, signals=all_signals)
        print(f"  {h}: {len(fwd[h])} periods, "
              f"{ic_by_h[h]['n_periods'].gt(0).sum()} signals with IC")

    # ---- redundancy & incremental (sub-factor level) ----------------------
    redundancy_df = max_sibling_corr(panel)
    redundancy_matrices = {
        p: parent_score_correlation(panel, p) for p in panel.parent_keys
    }
    redundancy_matrices = {p: m for p, m in redundancy_matrices.items() if m is not None}
    incremental_df = incremental_metrics(panel, fwd["1M"])
    coverage = signal_coverage(panel, all_signals)

    # ---- assemble tables & classify ---------------------------------------
    factor_table = build_factor_table(panel, quint_by_h, ic_by_h)
    subfactor_table = build_subfactor_table(
        panel, quint_by_h, ic_by_h, redundancy_df, incremental_df, coverage)
    thresholds = Thresholds()
    subfactor_table = classify_subfactors(subfactor_table, thresholds)
    subfactor_table = subfactor_table.sort_values(
        ["verdict", "keep_score"], ascending=[True, False]).reset_index(drop=True)
    watch = redundancy_watch(subfactor_table, thresholds.corr_high)

    # ---- write outputs ----------------------------------------------------
    quint_long_df = pd.concat(quint_long, ignore_index=True)
    ic_long_df = pd.concat(ic_ts_all, ignore_index=True) if ic_ts_all else pd.DataFrame()
    report_mod.write_csv_outputs(out_dir, factor_table, subfactor_table,
                                 quint_long_df, ic_long_df, redundancy_matrices)
    report_mod.render_parent_quintiles(out_dir, quint_by_h, panel.parent_keys)
    report_mod.render_subfactor_quintiles(out_dir, quint_by_h, panel)
    report_mod.render_parent_ic_by_horizon(out_dir, factor_table, HORIZONS)
    report_mod.render_subfactor_ic_ir(out_dir, subfactor_table)
    report_mod.render_redundancy(out_dir, redundancy_matrices)
    report_mod.render_verdict_summary(out_dir, subfactor_table)
    report_mod.write_markdown(
        out_dir,
        meta={"start": args.start, "end": args.end, "freq": args.freq,
              "n_rebalances": len(panel.rebal_dates), "n_universe": len(panel.universe)},
        factor_table=factor_table, subfactor_table=subfactor_table,
        thresholds=thresholds, horizons=HORIZONS, watch=watch)

    _print_summary(factor_table, subfactor_table, watch, out_dir)


def _print_summary(factor_table, subfactor_table, watch, out_dir) -> None:
    pd.set_option("display.width", 200)
    print("\n" + "=" * 90)
    print(" PARENT FACTOR SCORECARD (sorted by 1M IR)")
    print("=" * 90)
    cols = [c for c in ["parent", "n_periods", "ic_1M", "ic_3M", "ic_6M", "ic_12M",
                        "ir_1M", "hit_1M", "mono_1M"] if c in factor_table.columns]
    print(factor_table[cols].to_string(index=False, float_format=lambda v: f"{v:7.4f}"))

    print("\n" + "=" * 90)
    print(" SUB-FACTOR VERDICTS")
    print("=" * 90)
    for verdict in ["Keep", "Merge", "Remove", "Insufficient Data"]:
        sub = subfactor_table[subfactor_table["verdict"] == verdict]
        names = ", ".join(sub["sub_factor"].tolist()) or "(none)"
        print(f"\n  {verdict} ({len(sub)}): {names}")
    if watch is not None and not watch.empty:
        print("\n" + "=" * 90)
        print(" REDUNDANCY WATCH (near-duplicate Keep pairs — consider folding by hand)")
        print("=" * 90)
        for _, r in watch.iterrows():
            print(f"  fold {r['fold']:>26}  ->  {r['into']:<26} (corr {r['corr']:.2f})")
    print(f"\nSaved CSVs, charts and REPORT.md to {out_dir}/")


if __name__ == "__main__":
    main()
