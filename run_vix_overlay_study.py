"""VIX-aware parent-weighting overlay study runner.

Runs 4 composite variants (baseline + 3 overlays) in a strict PIT semiannual
walk-forward (rolling-5y, 2017–2026) and compares their OOS metrics.

Outputs (output/vix_overlay/):
  Heatmaps: regime weights for each overlay variant
  Line chart: composite IC per window per overlay
  Bar chart: full-period summary (Sharpe / Excess CAGR / IC)
  CSVs: per_window.csv, full_summary.csv, regime_weights_*.csv
  VIX_OVERLAY_REPORT.md

Usage:
    python run_vix_overlay_study.py
    python run_vix_overlay_study.py --rebuild-panel
    python run_vix_overlay_study.py --out output/vix_overlay
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from backtesting import data_loader as dl
from data.db import get_db
from research.walkforward.splits import LAST_TEST_END, PANEL_END, PANEL_START
from research.walkforward.vix_overlay import (
    METRIC_COLS, OVERLAY_TYPES, PARENT_CAP, CONSERVATIVE_CAP,
    OverlayRun, build_overlay_summary, run_overlay_study,
)
from research.walkforward.vix_regime_study import REGIME_ORDER, load_vix_series

from run_walkforward import PRICE_END, _load_panel

OUT_DIR = Path("output/vix_overlay")

OVERLAY_LABELS = {
    "baseline": "Baseline (no VIX tilt)",
    "overlay_full": "Full Tilt (IC-proportional by regime)",
    "overlay_cons": f"Conservative (±{CONSERVATIVE_CAP:.0%} cap)",
    "overlay_pos": "Positive-Only (IC>0 & Q5-Q1>0)",
}
OVERLAY_COLORS = {
    "baseline": "steelblue",
    "overlay_full": "darkorange",
    "overlay_cons": "forestgreen",
    "overlay_pos": "crimson",
}


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
def _heatmap(df: pd.DataFrame, title: str, cmap: str, fmt: str, out: Path) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    if df.empty:
        return
    data = df.values.astype(float)
    fig, ax = plt.subplots(figsize=(max(5, len(df.columns) * 2.2),
                                    max(4, len(df) * 0.65 + 1.5)))
    finite = data[np.isfinite(data)]
    vabs = float(np.nanmax(np.abs(finite))) if len(finite) else 1.0
    norm = (mcolors.TwoSlopeNorm(vmin=-vabs, vcenter=0, vmax=vabs)
            if cmap == "RdYlGn" else None)
    im = ax.imshow(data, cmap=cmap, aspect="auto", norm=norm,
                   vmin=None if norm else 0, vmax=None if norm else vabs)
    ax.set_xticks(range(len(df.columns)))
    ax.set_xticklabels(df.columns, fontsize=10)
    ax.set_yticks(range(len(df.index)))
    ax.set_yticklabels(df.index, fontsize=9)
    for i in range(len(df.index)):
        for j in range(len(df.columns)):
            v = data[i, j]
            txt = f"{v:{fmt}}" if np.isfinite(v) else "—"
            ax.text(j, i, txt, ha="center", va="center", fontsize=9,
                    color="black" if abs(v) < 0.65 * vabs else "white"
                    if np.isfinite(v) else "gray")
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title(title, fontsize=11, pad=10)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_regime_weight_heatmaps(regime_weights: dict[str, pd.DataFrame],
                                 out_dir: Path) -> None:
    for ot, df in regime_weights.items():
        _heatmap(df, f"Parent Weights by VIX Regime — {OVERLAY_LABELS[ot]}",
                 "Blues", ".1%", out_dir / f"heatmap_regime_weights_{ot}.png")


def plot_ic_by_window(per_window: pd.DataFrame, out_dir: Path) -> None:
    import matplotlib.pyplot as plt
    windows = per_window["window"].unique()
    fig, ax = plt.subplots(figsize=(max(12, len(windows) * 0.7), 5))
    for ot in OVERLAY_TYPES:
        sub = per_window[per_window["overlay"] == ot].set_index("window")
        ax.plot(windows, sub.reindex(windows)["ic_6m"].values,
                marker="o", markersize=4, linewidth=1.5,
                color=OVERLAY_COLORS[ot], label=OVERLAY_LABELS[ot])
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xticks(range(len(windows)))
    ax.set_xticklabels(windows, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("OOS Composite IC (6M)", fontsize=10)
    ax.set_title("Composite IC by Test Window — Baseline vs VIX Overlays", fontsize=11)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "ic_by_window.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_summary_bars(full_summary: pd.DataFrame, out_dir: Path) -> None:
    import matplotlib.pyplot as plt
    metrics = [("sharpe", "Sharpe"), ("spy_excess_cagr", "Excess CAGR vs SPY"),
               ("ic_6m", "Mean IC (6M)"), ("spy_ir", "Info Ratio vs SPY")]
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    for ax, (col, label) in zip(axes, metrics):
        vals = []
        labels = []
        colors = []
        for ot in OVERLAY_TYPES:
            sub = full_summary[full_summary["overlay"] == ot]
            if sub.empty or col not in sub.columns:
                continue
            v = float(sub.iloc[0][col])
            vals.append(v if np.isfinite(v) else 0.0)
            labels.append(OVERLAY_LABELS[ot].split("(")[0].strip())
            colors.append(OVERLAY_COLORS[ot])
        bars = ax.bar(range(len(vals)), vals, color=colors)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.axhline(0, color="gray", linewidth=0.7, linestyle="--")
        ax.set_title(label, fontsize=10)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{v:+.3f}" if abs(v) < 0.5 else f"{v:+.1%}",
                    ha="center", va="bottom" if v >= 0 else "top", fontsize=8)
    fig.suptitle("Full-Period Summary: Baseline vs VIX Overlays", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "summary_bars.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Markdown report
# --------------------------------------------------------------------------- #
def _f(v, fmt=".3f") -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    return f"{v:{fmt}}"


def _pct(v) -> str:
    return _f(v, "+.1%")


def _df_md(df: pd.DataFrame, fmts: dict[str, str] | None = None) -> str:
    fmts = fmts or {}
    lines = ["| " + " | ".join(str(c) for c in df.columns) + " |",
             "|" + "|".join("---" for _ in df.columns) + "|"]
    for _, row in df.iterrows():
        cells = []
        for c in df.columns:
            v = row[c]
            fmt = fmts.get(c)
            cells.append(_f(v, fmt) if fmt else (str(v) if not isinstance(v, float) else _f(v)))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _write_report(out_dir: Path, summaries: dict[str, pd.DataFrame],
                  run: OverlayRun) -> None:
    full = summaries["full_summary"]
    pw = summaries["per_window"]
    rw = summaries["regime_weights"]

    best_ot = "—"
    if not full.empty and "sharpe" in full.columns:
        valid = full.dropna(subset=["sharpe"])
        if not valid.empty:
            best_ot = str(valid.loc[valid["sharpe"].idxmax(), "overlay"])

    lines: list[str] = [
        "# VIX-Aware Parent Weighting Overlay Study",
        "",
        f"Rolling-5y semiannual walk-forward, {run.windows[0].label if run.windows else '?'}"
        f" through {run.windows[-1].label if run.windows else '?'}",
        f"**{len(run.windows)} test windows** | 4 overlay variants compared against baseline",
        "",
        "## Overlay Definitions",
        "",
        "| Overlay | Logic |",
        "|---------|-------|",
        "| **Baseline** | Standard rolling-5y V4 parent weights (no VIX awareness) |",
        "| **Full Tilt** | Weights ∝ max(0, IC_in_VIX_regime); negative-IC parents zeroed |",
        f"| **Conservative** | Baseline + tilt toward Full-Tilt, capped at ±{CONSERVATIVE_CAP:.0%} per parent |",
        "| **Positive-Only** | Only parents with positive IC **and** positive Q5-Q1 in the regime; rest zeroed |",
        "",
        "_Regime weights are derived from training data only. At each test rebalance the "
        "current spot VIX selects which regime vector to apply._",
        "",
        "---",
        "",
        "## Full-Period Summary",
        "",
    ]
    summary_cols = ["overlay", "n_windows", "ic_6m", "q5q1_ann", "sharpe",
                    "sortino", "cagr", "max_drawdown", "spy_excess_cagr",
                    "spy_ir", "spy_beta", "avg_turnover"]
    show_full = full[[c for c in summary_cols if c in full.columns]].copy()
    show_full["overlay"] = show_full["overlay"].map(OVERLAY_LABELS).fillna(show_full["overlay"])
    lines.append(_df_md(show_full, {
        "ic_6m": "+.3f", "q5q1_ann": "+.1%",
        "sharpe": ".2f", "sortino": ".2f", "cagr": "+.1%",
        "max_drawdown": ".1%", "spy_excess_cagr": "+.1%",
        "spy_ir": ".2f", "spy_beta": ".2f", "avg_turnover": ".2f",
    }))

    lines += [
        "",
        f"**Best overlay by Sharpe**: {OVERLAY_LABELS.get(best_ot, best_ot)}",
        "",
        "---",
        "",
        "## Regime Weight Heatmaps",
        "",
        "_Average parent weight assigned in each VIX regime for each overlay._",
        "",
    ]
    for ot in OVERLAY_TYPES:
        df = rw.get(ot, pd.DataFrame())
        if df.empty:
            continue
        lines += [f"### {OVERLAY_LABELS[ot]}", ""]
        tmp = df.copy()
        for col in tmp.columns:
            tmp[col] = tmp[col].apply(lambda v: _f(v, ".1%"))
        tmp.insert(0, "Parent", df.index)
        lines.append(_df_md(tmp))
        lines.append("")

    lines += [
        "---",
        "",
        "## Per-Window IC Comparison",
        "",
        "| Window | Baseline IC | Full Tilt IC | Conservative IC | Positive-Only IC |",
        "|--------|------------|--------------|-----------------|-----------------|",
    ]
    windows = pw["window"].unique()
    for wlbl in windows:
        sub = pw[pw["window"] == wlbl].set_index("overlay")
        cells = [wlbl] + [_f(sub.loc[ot, "ic_6m"]) if ot in sub.index else "—"
                          for ot in OVERLAY_TYPES]
        lines.append("| " + " | ".join(cells) + " |")

    lines += [
        "",
        "---",
        "",
        "## Key Findings",
        "",
        "### Does VIX-aware tilting improve OOS performance?",
    ]
    base_row = full[full["overlay"] == "baseline"]
    for ot in ["overlay_full", "overlay_cons", "overlay_pos"]:
        ot_row = full[full["overlay"] == ot]
        if base_row.empty or ot_row.empty:
            continue
        dsharp = _f(float(ot_row.iloc[0].get("sharpe", float("nan")))
                    - float(base_row.iloc[0].get("sharpe", float("nan"))), "+.2f")
        dic = _f(float(ot_row.iloc[0].get("ic_6m", float("nan")))
                 - float(base_row.iloc[0].get("ic_6m", float("nan"))), "+.3f")
        lines.append(f"- **{OVERLAY_LABELS[ot]}**: ΔSharpe={dsharp}, ΔIC={dic}")

    lines += [
        "",
        "### Do regime weights match the expected factor rotation?",
        "_From Study 1: Momentum best in Low VIX, Quality best in High VIX, Short stable._",
    ]
    for ot in ["overlay_full", "overlay_pos"]:
        df = rw.get(ot, pd.DataFrame())
        if df.empty or len(df.columns) < 2:
            continue
        lines.append(f"\n**{OVERLAY_LABELS[ot]}**:")
        for p in ["momentum", "quality", "short", "value"]:
            if p not in df.index:
                continue
            row = df.loc[p].dropna()
            if not row.empty:
                vals = " / ".join(f"{r.split(' ')[0]}: {_f(v, '.1%')}"
                                  for r, v in row.items())
                lines.append(f"  - `{p}`: {vals}")

    lines += [
        "",
        "---",
        "",
        "## Output Files",
        "| File | Description |",
        "|------|-------------|",
        "| `heatmap_regime_weights_*.png` | Parent weights by VIX regime per overlay |",
        "| `ic_by_window.png` | OOS composite IC per window per overlay |",
        "| `summary_bars.png` | Full-period Sharpe / Excess CAGR / IC / IR bars |",
        "| `per_window.csv` | Per (window, overlay) IC + portfolio metrics |",
        "| `full_summary.csv` | Full-period pooled metrics per overlay |",
        "| `regime_weights_*.csv` | Regime weight heatmap data per overlay |",
        "",
    ]
    (out_dir / "VIX_OVERLAY_REPORT.md").write_text("\n".join(lines))
    print("  → VIX_OVERLAY_REPORT.md")


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description="VIX-aware overlay study (2017→2026)")
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
    print(f"VIX: {vix_series.index[0].date()} → {vix_series.index[-1].date()}")

    print(f"\n=== VIX overlay study (rolling-5y, {args.first_test_year}→{args.last_end}) ===")
    run = run_overlay_study(panel, matrix, sectors, vix_series,
                            first_test_year=args.first_test_year,
                            last_end=args.last_end, verbose=True)

    if not run.windows:
        raise SystemExit("No usable windows — check data span / panel coverage")
    print(f"\nCompleted {len(run.windows)} windows.")

    print("\nBuilding summaries…")
    summaries = build_overlay_summary(run)

    print("Generating plots…")
    plot_regime_weight_heatmaps(summaries["regime_weights"], out_dir)
    plot_ic_by_window(summaries["per_window"], out_dir)
    plot_summary_bars(summaries["full_summary"], out_dir)

    print("Writing CSVs…")
    summaries["per_window"].to_csv(out_dir / "per_window.csv", index=False)
    summaries["full_summary"].to_csv(out_dir / "full_summary.csv", index=False)
    for ot, df in summaries["regime_weights"].items():
        df.to_csv(out_dir / f"regime_weights_{ot}.csv")

    print("Writing report…")
    _write_report(out_dir, summaries, run)

    print(f"\nAll outputs written to {out_dir}/")

    # Print quick summary to terminal
    fs = summaries["full_summary"]
    print("\n=== Full-period summary ===")
    show = fs[["overlay", "ic_6m", "q5q1_ann", "sharpe", "spy_excess_cagr",
               "spy_ir"]].copy()
    show["overlay"] = show["overlay"].map(OVERLAY_LABELS).fillna(show["overlay"])
    pd.set_option("display.float_format", "{:.3f}".format)
    pd.set_option("display.max_colwidth", 40)
    print(show.to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
