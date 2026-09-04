"""VIX regime study: parent effectiveness by VIX regime + train/test mismatch.

Two studies built on top of the existing rolling-5y semiannual walk-forward:

  Study 1 — VIX regime vs parent factor effectiveness
    Collects per-window VIX stats and per-parent OOS IC / Q5-Q1 spread / model
    weight, buckets windows into Low/Medium/High VIX regimes, and produces
    three heatmaps (parent × regime) plus per-window raw data and correlations.

  Study 2 — Train/test VIX mismatch vs OOS performance
    For each window computes |avg_vix_test − avg_vix_train| and the signed
    direction, then correlates those against excess CAGR, Sharpe, IR, composite
    IC, and Q5-Q1 spread. Outputs scatter-plots and a correlation summary table.

Usage:
    python run_vix_regime_study.py
    python run_vix_regime_study.py --rebuild-panel
    python run_vix_regime_study.py --out output/vix_regime
"""
from __future__ import annotations

import argparse
import textwrap
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from backtesting import data_loader as dl
from data.db import get_db
from research.walkforward.splits import LAST_TEST_END, PANEL_END, PANEL_START
from research.walkforward.vix_regime_study import (
    PERF_COLS, REGIME_ORDER,
    load_vix_series,
    plot_mismatch_scatterplots, plot_regime_heatmaps, plot_vix_direction_scatterplots,
    run_vix_study,
    study1_regime_tables, study2_correlations, study2_mismatch_table,
)

from run_walkforward import PRICE_END, _load_panel

OUT_DIR = Path("output/vix_regime")


# --------------------------------------------------------------------------- #
# Markdown report
# --------------------------------------------------------------------------- #
def _fmt(v, fmt=".3f") -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    return f"{v:{fmt}}"


def _df_md(df: pd.DataFrame, float_cols: dict[str, str] | None = None) -> str:
    """Render a small DataFrame as a Github-flavoured markdown table."""
    fc = float_cols or {}
    lines = []
    lines.append("| " + " | ".join(str(c) for c in df.columns) + " |")
    lines.append("|" + "|".join("---" for _ in df.columns) + "|")
    for _, row in df.iterrows():
        cells = []
        for c in df.columns:
            v = row[c]
            fmt = fc.get(c)
            if fmt:
                cells.append(_fmt(v, fmt))
            elif isinstance(v, float):
                cells.append(_fmt(v))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _heatmap_md(df: pd.DataFrame, fmt: str = ".3f") -> str:
    if df.empty:
        return "_No data._"
    tmp = df.copy()
    for col in tmp.columns:
        tmp[col] = tmp[col].apply(lambda v: _fmt(v, fmt))
    tmp.insert(0, "Parent", df.index)
    return _df_md(tmp)


def _write_report(out_dir: Path, regime_tables: dict[str, pd.DataFrame],
                  mismatch_df: pd.DataFrame, corr_df: pd.DataFrame,
                  run) -> None:
    n_windows = len(run.windows)
    regimes_seen = mismatch_df["test_regime"].value_counts().to_dict()
    regime_counts = ", ".join(f"{r}: {regimes_seen.get(r, 0)}"
                              for r in REGIME_ORDER if r in regimes_seen)

    # Per-parent weight correlation summary
    corr_s1 = regime_tables["correlations"]

    lines: list[str] = [
        "# VIX Regime Study",
        "",
        f"Rolling-5y semiannual walk-forward, 2017-H1 through {run.windows[-1].label if run.windows else '?'}",
        f"**{n_windows} test windows** | Regime distribution: {regime_counts}",
        "",
        "---",
        "",
        "## Study 1: VIX Regime vs Parent Factor Effectiveness",
        "",
        "Each semiannual test window is bucketed by the **average VIX during the test",
        "window + 3-month holding period** (Low <15, Medium 15-25, High >25).",
        "Parent weights are frozen at training time (rolling-5y); IC and Q5-Q1 are",
        "measured on the unseen test rebalances.",
        "",
        "### Heatmap 1A — Parent OOS IC (6M) by VIX Regime",
        "_Positive = factor correctly ranks future returns in that VIX environment._",
        "",
        _heatmap_md(regime_tables["ic_heatmap"], ".3f"),
        "",
        "### Heatmap 1B — Parent Q5-Q1 Spread (6M, annualised) by VIX Regime",
        "_Q5-Q1 annualised return spread; positive = top quintile beats bottom quintile._",
        "",
        _heatmap_md(regime_tables["q5q1_heatmap"], ".1%"),
        "",
        "### Heatmap 1C — Parent Model Weight by VIX Regime",
        "_Average frozen composite weight assigned in windows where the test-period VIX",
        "fell in each regime. Reflects training-period fit, not test-period VIX awareness._",
        "",
        _heatmap_md(regime_tables["weight_heatmap"], ".1%"),
        "",
        "### Study 1 Correlations (Spearman, per-window series)",
        "_ρ(VIX_avg_test_plus_3m, parent_IC_6M) and ρ(VIX, parent_weight)_",
        "",
        _df_md(corr_s1, {"spearman_vix_vs_ic": ".3f", "spearman_vix_vs_weight": ".3f"}),
        "",
        "---",
        "",
        "## Study 2: Train/Test VIX Mismatch vs OOS Performance",
        "",
        "**VIX mismatch** = |avg VIX during test+3M − avg VIX during 5y training window|",
        "**VIX direction** = avg VIX test+3M − avg VIX train (positive = more volatile test)",
        "",
        "### Correlation Table",
        "_Spearman ρ between mismatch/direction and each performance metric._",
        "",
        _df_md(corr_df[["metric", "n_windows", "spearman_mismatch_vs_perf",
                         "spearman_direction_vs_perf"]],
               {"spearman_mismatch_vs_perf": "+.3f", "spearman_direction_vs_perf": "+.3f"}),
        "",
        "### Per-Window Mismatch Summary",
        "_First 20 windows sorted by VIX mismatch (largest first)._",
        "",
    ]

    top20 = mismatch_df.sort_values("vix_mismatch", ascending=False).head(20)
    show_cols = ["window", "train_regime", "test_regime", "vix_avg_train",
                 "vix_avg_test_plus_3m", "vix_mismatch", "vix_direction",
                 "composite_ic_6m", "spy_excess_cagr", "port_sharpe"]
    top20_show = top20[[c for c in show_cols if c in top20.columns]].reset_index(drop=True)
    lines.append(_df_md(top20_show, {
        "vix_avg_train": ".1f", "vix_avg_test_plus_3m": ".1f",
        "vix_mismatch": ".1f", "vix_direction": "+.1f",
        "composite_ic_6m": "+.3f", "spy_excess_cagr": "+.1%", "port_sharpe": ".2f",
    }))

    lines += [
        "",
        "---",
        "",
        "## Key Questions",
        "",
        "### Q1 — Do different VIX regimes favour different parent factors?",
    ]
    # Find best parent per regime
    ic_hm = regime_tables["ic_heatmap"]
    if not ic_hm.empty:
        for regime in REGIME_ORDER:
            if regime not in ic_hm.columns:
                continue
            col = ic_hm[regime].dropna()
            if col.empty:
                continue
            best_p = col.idxmax()
            worst_p = col.idxmin()
            lines.append(f"- **{regime} VIX**: best parent = `{best_p}` "
                         f"(IC {_fmt(col[best_p])}),"
                         f" worst = `{worst_p}` (IC {_fmt(col[worst_p])})")

    lines += [
        "",
        "### Q2 — Does the model assign different parent weights by VIX regime?",
    ]
    wt_hm = regime_tables["weight_heatmap"]
    if not wt_hm.empty and len(wt_hm.columns) > 1:
        for p in wt_hm.index:
            row_vals = wt_hm.loc[p].dropna()
            if len(row_vals) < 2:
                continue
            rng = float(row_vals.max() - row_vals.min())
            if rng > 0.05:
                lines.append(f"- `{p}`: weight range {_fmt(row_vals.min(), '.1%')} – "
                             f"{_fmt(row_vals.max(), '.1%')} across regimes "
                             f"(Δ={_fmt(rng, '.1%')}) — notable variation")

    lines += [
        "",
        "### Q3 — Does performance deteriorate with larger train/test VIX mismatch?",
    ]
    if not corr_df.empty:
        for _, row in corr_df.iterrows():
            ρ_dist = row["spearman_mismatch_vs_perf"]
            direction = "deteriorates" if ρ_dist < -0.2 else ("improves" if ρ_dist > 0.2 else "flat")
            lines.append(f"- **{row['metric']}**: ρ(mismatch) = {_fmt(ρ_dist, '+.3f')} → {direction}")

    lines += [
        "",
        "---",
        "",
        "## Output Files",
        "| File | Description |",
        "|------|-------------|",
        "| `heatmap_parent_ic_by_vix_regime.png` | Parent IC heatmap |",
        "| `heatmap_parent_q5q1_by_vix_regime.png` | Parent Q5-Q1 heatmap |",
        "| `heatmap_parent_weight_by_vix_regime.png` | Parent weight heatmap |",
        "| `scatterplot_vix_mismatch_vs_performance.png` | Mismatch vs perf scatter |",
        "| `scatterplot_vix_direction_vs_performance.png` | Direction vs perf scatter |",
        "| `study1_per_window.csv` | Per-window VIX + parent IC/weight raw data |",
        "| `study1_ic_heatmap.csv` | Parent IC by regime |",
        "| `study1_q5q1_heatmap.csv` | Parent Q5-Q1 by regime |",
        "| `study1_weight_heatmap.csv` | Parent weight by regime |",
        "| `study1_correlations.csv` | VIX vs parent IC/weight correlations |",
        "| `study2_mismatch.csv` | Per-window mismatch + performance |",
        "| `study2_correlations.csv` | Mismatch/direction vs performance correlations |",
        "",
    ]

    (out_dir / "VIX_REGIME_REPORT.md").write_text("\n".join(lines))
    print(f"  → VIX_REGIME_REPORT.md")


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description="VIX regime study (2017→2026)")
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
        vix_series = load_vix_series(db)

    print(f"Price matrix: {matrix.shape[1]} tickers × {matrix.shape[0]} days")
    print(f"VIX: {vix_series.index[0].date()} → {vix_series.index[-1].date()} "
          f"({len(vix_series)} daily observations)")

    print(f"\n=== Running VIX regime study (rolling-5y, "
          f"{args.first_test_year}→{args.last_end}) ===")
    run = run_vix_study(panel, matrix, sectors, vix_series,
                        first_test_year=args.first_test_year,
                        last_end=args.last_end, verbose=True)

    if not run.windows:
        raise SystemExit("No usable windows — check data span / panel coverage")

    print(f"\nCompleted {len(run.windows)} windows.")

    # ---- Study 1 tables ----
    print("\nBuilding Study 1 regime tables…")
    regime_tables = study1_regime_tables(run)

    # ---- Study 2 tables ----
    print("Building Study 2 mismatch table…")
    mismatch_df = study2_mismatch_table(run)
    corr_df = study2_correlations(mismatch_df)

    # ---- Plots ----
    print("Generating heatmaps…")
    plot_regime_heatmaps(regime_tables, out_dir)
    print("Generating scatter plots…")
    plot_mismatch_scatterplots(mismatch_df, out_dir)
    plot_vix_direction_scatterplots(mismatch_df, out_dir)

    # ---- CSVs ----
    print("Writing CSVs…")
    regime_tables["per_window"].to_csv(out_dir / "study1_per_window.csv", index=False)
    regime_tables["ic_heatmap"].to_csv(out_dir / "study1_ic_heatmap.csv")
    regime_tables["q5q1_heatmap"].to_csv(out_dir / "study1_q5q1_heatmap.csv")
    regime_tables["weight_heatmap"].to_csv(out_dir / "study1_weight_heatmap.csv")
    regime_tables["correlations"].to_csv(out_dir / "study1_correlations.csv", index=False)
    mismatch_df.to_csv(out_dir / "study2_mismatch.csv", index=False)
    corr_df.to_csv(out_dir / "study2_correlations.csv", index=False)

    # ---- Markdown report ----
    print("Writing report…")
    _write_report(out_dir, regime_tables, mismatch_df, corr_df, run)

    print(f"\nAll outputs written to {out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
