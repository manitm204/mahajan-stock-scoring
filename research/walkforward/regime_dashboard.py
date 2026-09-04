"""6-Month regime dashboard: heatmap of per-window portfolio vs SPY metrics.

Consumes ``output/regime5y_focus/per_window.csv``, joins per-window VIX stats from
``daily_prices`` (ticker='VIX'), and renders two heatmaps (chronological + VIX-sorted)
plus a correlation matrix. Nothing here rebuilds the walk-forward — it operates purely
on the artefacts produced by ``run_regime5y_analysis.py``.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Column configuration
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ColumnSpec:
    """One heatmap column.

    ``direction``: 'high_good' (RdYlGn) or 'low_good' (RdYlGn_r).
    ``center``: 'zero' | 'one' | 'median' — anchors TwoSlopeNorm.
    ``fmt``: value format inside cells.
    """
    label: str
    source: str            # column in per_window.csv or derived name
    direction: str         # high_good | low_good
    center: str            # zero | one | median
    fmt: str               # .1% | .2f etc.


COLUMNS: list[ColumnSpec] = [
    ColumnSpec("Avg VIX",        "avg_vix",           "low_good",  "median", ".1f"),
    ColumnSpec("VIX Chg",        "vix_change",        "low_good",  "zero",   "+.1f"),
    ColumnSpec("Port CAGR",      "port_cagr",         "high_good", "median", "+.1%"),
    ColumnSpec("SPY CAGR",       "spy_cagr",          "high_good", "median", "+.1%"),
    ColumnSpec("Excess CAGR",    "excess_cagr",       "high_good", "zero",   "+.1%"),
    ColumnSpec("Port Sharpe",    "port_sharpe",       "high_good", "median", "+.2f"),
    ColumnSpec("SPY Sharpe",     "spy_sharpe",        "high_good", "median", "+.2f"),
    ColumnSpec("Sharpe Diff",    "sharpe_diff",       "high_good", "zero",   "+.2f"),
    ColumnSpec("Port Sortino",   "port_sortino",      "high_good", "median", "+.2f"),
    ColumnSpec("SPY Sortino",    "spy_sortino",       "high_good", "median", "+.2f"),
    ColumnSpec("Info Ratio",     "info_ratio",        "high_good", "zero",   "+.2f"),
    ColumnSpec("Beta",           "beta",              "low_good",  "one",    ".2f"),
    ColumnSpec("Port Max DD",    "port_max_drawdown", "high_good", "median", ".1%"),
    ColumnSpec("SPY Max DD",     "spy_max_drawdown",  "high_good", "median", ".1%"),
    ColumnSpec("Rel DD",         "rel_max_drawdown",  "high_good", "zero",   "+.1%"),
    ColumnSpec("IC 3M",          "composite_ic_3M",   "high_good", "zero",   "+.3f"),
    ColumnSpec("Q5-Q1",          "q5_q1_spread_annualized", "high_good", "zero", "+.1%"),
]

CORR_COLS: list[tuple[str, str]] = [
    ("Avg VIX",       "avg_vix"),
    ("VIX Chg",       "vix_change"),
    ("Excess CAGR",   "excess_cagr"),
    ("Sharpe Diff",   "sharpe_diff"),
    ("Sortino Diff",  "sortino_diff"),
    ("Info Ratio",    "info_ratio"),
    ("IC 3M",         "composite_ic_3M"),
    ("Q5-Q1",         "q5_q1_spread_annualized"),
]


# --------------------------------------------------------------------------- #
# VIX per-window stats
# --------------------------------------------------------------------------- #
def _window_bounds(label: str) -> tuple[str, str]:
    """"YYYY-H1" → ('YYYY-01-01', 'YYYY-06-30'); H2 → ('YYYY-07-01','YYYY-12-31')."""
    year, half = label.split("-", 1)
    if half == "H1":
        return f"{year}-01-01", f"{year}-06-30"
    return f"{year}-07-01", f"{year}-12-31"


def load_vix(db) -> pd.Series:
    """Return VIX daily-close series indexed by ISO date string."""
    df = db.query_df("SELECT date, close FROM daily_prices WHERE ticker = 'VIX' "
                     "ORDER BY date")
    if df.empty:
        raise RuntimeError("No VIX rows in daily_prices — run backfill first")
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")["close"].astype(float).sort_index()


def per_window_vix(vix: pd.Series, labels: list[str]) -> pd.DataFrame:
    """Average VIX + VIX change (end - start close) for each 6M test window."""
    rows = []
    for lbl in labels:
        start, end = _window_bounds(lbl)
        sl = vix.loc[start:end]
        if sl.empty:
            rows.append({"window": lbl, "avg_vix": np.nan, "vix_change": np.nan})
            continue
        rows.append({
            "window": lbl,
            "avg_vix": float(sl.mean()),
            "vix_change": float(sl.iloc[-1] - sl.iloc[0]),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Dashboard frame assembly
# --------------------------------------------------------------------------- #
def build_frame(per_window_csv: Path, vix: pd.Series) -> pd.DataFrame:
    """Join per-window metrics with VIX stats and derive Sharpe / Sortino diffs."""
    df = pd.read_csv(per_window_csv)
    if "port_sortino" not in df.columns:
        raise RuntimeError("per_window.csv is missing 'port_sortino' — re-run "
                           "run_regime5y_analysis.py after adding Sortino fields")
    vix_df = per_window_vix(vix, df["window"].tolist())
    df = df.merge(vix_df, on="window", how="left")
    df["sharpe_diff"] = df["port_sharpe"] - df["spy_sharpe"]
    df["sortino_diff"] = df["port_sortino"] - df["spy_sortino"]
    return df


# --------------------------------------------------------------------------- #
# Colour normalization
# --------------------------------------------------------------------------- #
def _norm(values: pd.Series, spec: ColumnSpec) -> mcolors.TwoSlopeNorm:
    """Two-slope diverging norm with a metric-specific centre."""
    v = values.dropna().astype(float)
    if v.empty:
        return mcolors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    if spec.center == "zero":
        centre = 0.0
    elif spec.center == "one":
        centre = 1.0
    else:
        centre = float(v.median())
    span = max(float(v.max()) - centre, centre - float(v.min()), 1e-9)
    vmin = centre - span
    vmax = centre + span
    if vmin == vmax:
        vmin, vmax = centre - 1e-6, centre + 1e-6
    return mcolors.TwoSlopeNorm(vmin=vmin, vcenter=centre, vmax=vmax)


def _cmap(direction: str) -> mcolors.Colormap:
    return plt.get_cmap("RdYlGn" if direction == "high_good" else "RdYlGn_r")


def _fmt_cell(value: float, fmt: str) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    return format(value, fmt)


# --------------------------------------------------------------------------- #
# Heatmap renderer
# --------------------------------------------------------------------------- #
def render_heatmap(df: pd.DataFrame, *, order_by: str, title: str,
                   out_path: Path) -> None:
    """Render one dashboard image. ``order_by`` = 'window' or 'avg_vix'."""
    ordered = (df.sort_values("window") if order_by == "window"
               else df.sort_values("avg_vix"))
    rows = ordered["window"].tolist()
    n_rows, n_cols = len(rows), len(COLUMNS)

    fig, ax = plt.subplots(figsize=(1.15 * n_cols + 2, 0.42 * n_rows + 1.8))

    for j, spec in enumerate(COLUMNS):
        series = ordered[spec.source]
        norm = _norm(df[spec.source], spec)
        cmap = _cmap(spec.direction)
        for i, val in enumerate(series.tolist()):
            color = cmap(norm(val)) if not np.isnan(val) else (0.9, 0.9, 0.9, 1.0)
            ax.add_patch(plt.Rectangle((j, n_rows - 1 - i), 1, 1,
                                       facecolor=color, edgecolor="white", linewidth=0.6))
            text = _fmt_cell(val, spec.fmt)
            luminance = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
            text_color = "white" if luminance < 0.45 else "black"
            ax.text(j + 0.5, n_rows - 1 - i + 0.5, text,
                    ha="center", va="center", fontsize=8, color=text_color)

    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows)
    ax.set_xticks([j + 0.5 for j in range(n_cols)])
    ax.set_xticklabels([s.label for s in COLUMNS], rotation=45, ha="right", fontsize=9)
    ax.set_yticks([n_rows - 1 - i + 0.5 for i in range(n_rows)])
    ax.set_yticklabels(rows, fontsize=9)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title, fontsize=13, pad=12, loc="left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Correlation matrix
# --------------------------------------------------------------------------- #
def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    labels = [lbl for lbl, _ in CORR_COLS]
    cols = [src for _, src in CORR_COLS]
    sub = df[cols].astype(float)
    corr = sub.corr(method="pearson")
    corr.index = labels
    corr.columns = labels
    return corr


def render_correlation(corr: pd.DataFrame, out_path: Path) -> None:
    n = len(corr)
    fig, ax = plt.subplots(figsize=(1.1 * n + 1.6, 0.9 * n + 1.4))
    norm = mcolors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    cmap = plt.get_cmap("RdYlGn")
    for i, row_lbl in enumerate(corr.index):
        for j, col_lbl in enumerate(corr.columns):
            val = float(corr.iat[i, j])
            color = cmap(norm(val))
            ax.add_patch(plt.Rectangle((j, n - 1 - i), 1, 1,
                                       facecolor=color, edgecolor="white", linewidth=0.6))
            luminance = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
            text_color = "white" if luminance < 0.45 else "black"
            ax.text(j + 0.5, n - 1 - i + 0.5, f"{val:+.2f}",
                    ha="center", va="center", fontsize=9, color=text_color)
    ax.set_xlim(0, n)
    ax.set_ylim(0, n)
    ax.set_xticks([j + 0.5 for j in range(n)])
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=10)
    ax.set_yticks([n - 1 - i + 0.5 for i in range(n)])
    ax.set_yticklabels(corr.index, fontsize=10)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("Regime Correlation Matrix (Pearson, 19 semi-annual windows)",
                 fontsize=13, pad=12, loc="left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Entry
# --------------------------------------------------------------------------- #
def write_dashboard(df: pd.DataFrame, out_dir: Path) -> dict[str, Path]:
    """Render all four artefacts and return their paths."""
    out_dir.mkdir(parents=True, exist_ok=True)

    dash_csv = out_dir / "dashboard.csv"
    display_cols = ["window"] + [s.source for s in COLUMNS] + ["sortino_diff"]
    df.loc[:, [c for c in display_cols if c in df.columns]].to_csv(dash_csv, index=False)

    chron_png = out_dir / "regime_dashboard_chronological.png"
    render_heatmap(df, order_by="window",
                   title="6-Month Regime Dashboard — chronological (2017-H1 → 2026-H1)",
                   out_path=chron_png)

    vix_png = out_dir / "regime_dashboard_by_vix.png"
    render_heatmap(df, order_by="avg_vix",
                   title="6-Month Regime Dashboard — sorted by average VIX (low → high)",
                   out_path=vix_png)

    corr = correlation_matrix(df)
    corr_csv = out_dir / "regime_correlations.csv"
    corr.to_csv(corr_csv)
    corr_png = out_dir / "regime_correlations.png"
    render_correlation(corr, corr_png)

    return {"dashboard_csv": dash_csv, "chronological": chron_png,
            "by_vix": vix_png, "correlation_csv": corr_csv,
            "correlation_png": corr_png}
