"""Reports + charts for the focused rolling-5y regime analysis.

Six visualisations plus per-window / 3-year-bucket / full-period tables and a top-level
``REGIME5Y_REPORT.md`` under ``output/regime5y_focus/``. The charts:

1. ``timeseries_portfolio_vs_spy.png`` — cumulative equity of portfolio vs SPY, with the
   H1/H2 window boundaries marked.
2. ``rolling_excess_bars.png`` — per-window annualised excess vs SPY, bar-coloured by sign.
3. ``rolling_sharpe.png`` — per-window portfolio Sharpe vs SPY Sharpe.
4. ``rolling_max_drawdown.png`` — per-window max drawdown, portfolio vs SPY.
5. ``rolling_beta.png`` — per-window beta to SPY (reference line at 1.0).
6. ``rolling_ic_spread.png`` — per-window composite 6M IC (bars) + Q5-Q1 annualised
   spread (line, right axis).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from backtesting.data_loader import SPY
from . import portfolio as pf
from .regime5y_focus import (BUCKET_DEFS, FocusRun, TOP_PCT, _year,
                             build_focused_summary, pooled_returns)


# --------------------------------------------------------------------------- #
# Chart helpers
# --------------------------------------------------------------------------- #
def _equity(returns: pd.Series) -> pd.Series:
    r = returns.dropna()
    if r.empty:
        return pd.Series(dtype=float)
    return (1.0 + r).cumprod()


def plot_timeseries_vs_spy(run: FocusRun, out_path: Path) -> None:
    ret, spy, _ = pooled_returns(run, run.windows)
    if ret.empty:
        return
    fig, ax = plt.subplots(figsize=(11, 5.5))
    eq_port = _equity(ret)
    eq_spy = _equity(spy.reindex(eq_port.index))
    ax.plot(pd.to_datetime(eq_port.index), eq_port.values,
            label="rolling5y portfolio (top-20%)", color="#1f77b4", linewidth=1.8)
    ax.plot(pd.to_datetime(eq_spy.index), eq_spy.values,
            label="SPY", color="#7f7f7f", linewidth=1.5, linestyle="--")
    # H1 boundaries as light vertical lines
    for w in run.windows:
        if w.label.endswith("H1"):
            try:
                ax.axvline(pd.Timestamp(w.split.test_start), color="grey",
                           linewidth=0.4, alpha=0.35)
            except Exception:
                pass
    ax.set_title("rolling5y portfolio vs SPY — cumulative return, 2017–present")
    ax.set_ylabel("Growth of $1")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _diverging_bar(ax, labels: list[str], values: np.ndarray, *,
                    pos_color: str = "#2ca02c", neg_color: str = "#d62728") -> None:
    colors = [pos_color if v is not None and not np.isnan(v) and v >= 0 else neg_color
              for v in values]
    ax.bar(range(len(labels)), values, color=colors, alpha=0.85)
    ax.axhline(0, color="black", linewidth=0.7, alpha=0.7)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.grid(True, alpha=0.25)


def plot_rolling_excess(per_window: pd.DataFrame, out_path: Path) -> None:
    if per_window.empty:
        return
    labels = per_window["window"].tolist()
    vals = per_window["excess_cagr"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(11, 4.8))
    _diverging_bar(ax, labels, vals)
    ax.set_title("Excess CAGR vs SPY, per test window")
    ax.set_ylabel("Excess CAGR (annualised)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{100*v:+.0f}%"))
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_rolling_sharpe(per_window: pd.DataFrame, out_path: Path) -> None:
    if per_window.empty:
        return
    labels = per_window["window"].tolist()
    port = per_window["port_sharpe"].to_numpy(dtype=float)
    spy = per_window["spy_sharpe"].to_numpy(dtype=float)
    x = np.arange(len(labels))
    w = 0.4
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.bar(x - w/2, port, w, label="portfolio", color="#1f77b4", alpha=0.9)
    ax.bar(x + w/2, spy,  w, label="SPY",       color="#7f7f7f", alpha=0.85)
    ax.axhline(0, color="black", linewidth=0.7, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Annualised Sharpe (per window)")
    ax.set_title("Rolling Sharpe — portfolio vs SPY, per test window")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_rolling_max_drawdown(per_window: pd.DataFrame, out_path: Path) -> None:
    if per_window.empty:
        return
    labels = per_window["window"].tolist()
    port = per_window["port_max_drawdown"].to_numpy(dtype=float)
    spy = per_window["spy_max_drawdown"].to_numpy(dtype=float)
    x = np.arange(len(labels))
    w = 0.4
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.bar(x - w/2, port, w, label="portfolio", color="#d62728", alpha=0.85)
    ax.bar(x + w/2, spy,  w, label="SPY",       color="#7f7f7f", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Max drawdown (per window)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{100*v:+.0f}%"))
    ax.set_title("Rolling max drawdown — portfolio vs SPY, per test window")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_rolling_beta(per_window: pd.DataFrame, out_path: Path) -> None:
    if per_window.empty:
        return
    labels = per_window["window"].tolist()
    beta = per_window["beta"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.bar(range(len(labels)), beta, color="#4c72b0", alpha=0.85)
    ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--", alpha=0.7,
               label="β = 1")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Beta to SPY")
    ax.set_title("Rolling beta to SPY, per test window")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_rolling_ic_spread(per_window: pd.DataFrame, out_path: Path) -> None:
    if per_window.empty:
        return
    labels = per_window["window"].tolist()
    ic6 = per_window["composite_ic_6M"].to_numpy(dtype=float)
    spread = per_window["q5_q1_spread_annualized"].to_numpy(dtype=float)
    fig, ax1 = plt.subplots(figsize=(11, 4.8))
    _diverging_bar(ax1, labels, ic6, pos_color="#2ca02c", neg_color="#d62728")
    ax1.set_ylabel("Composite 6M IC (bars)")
    ax1.axhline(0, color="black", linewidth=0.7, alpha=0.7)
    ax2 = ax1.twinx()
    ax2.plot(range(len(labels)), spread, color="#9467bd", marker="o", markersize=4,
             linewidth=1.6, label="Q5−Q1 spread (ann)")
    ax2.set_ylabel("Q5−Q1 annualised spread")
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{100*v:+.0f}%"))
    ax2.legend(loc="upper right", frameon=False)
    ax1.set_title("Composite 6M IC + Q5–Q1 spread, per test window")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Markdown helpers
# --------------------------------------------------------------------------- #
def _fmt(v, fmt: str = ".3f") -> str:
    if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
        return "—"
    try:
        return format(float(v), fmt)
    except Exception:
        return str(v)


def _pct(v) -> str:
    if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
        return "—"
    return f"{100 * float(v):+.2f}%"


def _md_per_window(per_window: pd.DataFrame) -> str:
    if per_window.empty:
        return "_no data_"
    lines = [
        "| window | port CAGR | SPY CAGR | excess | port Sharpe | SPY Sharpe | "
        "port maxDD | SPY maxDD | β | IR | 6M IC | Q5−Q1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in per_window.iterrows():
        lines.append(
            f"| {r['window']} | {_pct(r.get('port_cagr'))} | "
            f"{_pct(r.get('spy_cagr'))} | {_pct(r.get('excess_cagr'))} | "
            f"{_fmt(r.get('port_sharpe'), '.2f')} | {_fmt(r.get('spy_sharpe'), '.2f')} | "
            f"{_pct(r.get('port_max_drawdown'))} | {_pct(r.get('spy_max_drawdown'))} | "
            f"{_fmt(r.get('beta'), '.2f')} | {_fmt(r.get('info_ratio'), '.2f')} | "
            f"{_fmt(r.get('composite_ic_6M'))} | {_pct(r.get('q5_q1_spread_annualized'))} |")
    return "\n".join(lines)


def _md_bucket(bucket: pd.DataFrame) -> str:
    if bucket.empty:
        return "_no data_"
    lines = [
        "| bucket | n wins | n periods | port CAGR | SPY CAGR | excess | "
        "port Sharpe | SPY Sharpe | port maxDD | SPY maxDD | β | IR | 6M IC | Q5−Q1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in bucket.iterrows():
        lines.append(
            f"| {r['grouping']} | {int(r.get('n_windows', 0) or 0)} | "
            f"{int(r.get('n_periods', 0) or 0)} | "
            f"{_pct(r.get('port_cagr'))} | {_pct(r.get('spy_cagr'))} | "
            f"{_pct(r.get('excess_cagr'))} | {_fmt(r.get('port_sharpe'), '.2f')} | "
            f"{_fmt(r.get('spy_sharpe'), '.2f')} | "
            f"{_pct(r.get('port_max_drawdown'))} | {_pct(r.get('spy_max_drawdown'))} | "
            f"{_fmt(r.get('beta'), '.2f')} | {_fmt(r.get('info_ratio'), '.2f')} | "
            f"{_fmt(r.get('composite_ic_6M'))} | {_pct(r.get('q5_q1_spread_annualized'))} |")
    return "\n".join(lines)


def _md_full(full: pd.DataFrame) -> str:
    if full.empty:
        return "_no data_"
    r = full.iloc[0]
    return (
        f"* n windows = {int(r.get('n_windows', 0) or 0)}, "
        f"n monthly periods = {int(r.get('n_periods', 0) or 0)}\n"
        f"* portfolio: CAGR {_pct(r.get('port_cagr'))}, "
        f"Sharpe {_fmt(r.get('port_sharpe'), '.2f')}, "
        f"max-DD {_pct(r.get('port_max_drawdown'))}, "
        f"vol {_pct(r.get('port_ann_vol'))}\n"
        f"* SPY:       CAGR {_pct(r.get('spy_cagr'))}, "
        f"Sharpe {_fmt(r.get('spy_sharpe'), '.2f')}, "
        f"max-DD {_pct(r.get('spy_max_drawdown'))}, "
        f"vol {_pct(r.get('spy_ann_vol'))}\n"
        f"* excess vs SPY: CAGR {_pct(r.get('excess_cagr'))}, "
        f"α {_pct(r.get('alpha'))}, β {_fmt(r.get('beta'), '.2f')}, "
        f"IR {_fmt(r.get('info_ratio'), '.2f')}, "
        f"TE {_pct(r.get('tracking_error'))}, "
        f"rel-DD {_pct(r.get('rel_max_drawdown'))}\n"
        f"* composite: 1M IC {_fmt(r.get('composite_ic_1M'))}, "
        f"3M IC {_fmt(r.get('composite_ic_3M'))}, "
        f"6M IC {_fmt(r.get('composite_ic_6M'))}, "
        f"12M IC {_fmt(r.get('composite_ic_12M'))}; "
        f"Q5−Q1 spread (ann) {_pct(r.get('q5_q1_spread_annualized'))}, "
        f"monotonic-rate {_fmt(r.get('q5_q1_monotonic_rate'), '.2%')}"
    )


def _summarize_regimes(per_window: pd.DataFrame) -> str:
    """Human read of which regimes drove the outcome — beats/miss counts + best/worst."""
    if per_window.empty:
        return "_no data_"
    d = per_window.dropna(subset=["excess_cagr"])
    if d.empty:
        return "_no windows produced an excess vs SPY_"
    n_beat = int((d["excess_cagr"] > 0).sum())
    n_total = int(len(d))
    best = d.sort_values("excess_cagr", ascending=False).iloc[0]
    worst = d.sort_values("excess_cagr", ascending=True).iloc[0]
    ic_beat = int((per_window.get("composite_ic_6M", pd.Series(dtype=float)).fillna(0) > 0).sum())
    return (
        f"Beat SPY in **{n_beat}/{n_total}** test windows ({100*n_beat/n_total:.0f}%). "
        f"Best window: **{best['window']}** with excess {_pct(best['excess_cagr'])} "
        f"(port CAGR {_pct(best['port_cagr'])}, SPY {_pct(best['spy_cagr'])}). "
        f"Worst window: **{worst['window']}** with excess {_pct(worst['excess_cagr'])} "
        f"(port CAGR {_pct(worst['port_cagr'])}, SPY {_pct(worst['spy_cagr'])}). "
        f"Positive 6M composite IC in **{ic_beat}/{n_total}** windows."
    )


# --------------------------------------------------------------------------- #
# Main entry
# --------------------------------------------------------------------------- #
def write_focused_reports(run: FocusRun, summary: dict[str, pd.DataFrame],
                          out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    per_window = summary["per_window"]
    bucket = summary["bucket"]
    full = summary["full"]

    # CSVs
    per_window.to_csv(out_dir / "per_window.csv", index=False)
    bucket.to_csv(out_dir / "bucket_3y.csv", index=False)
    full.to_csv(out_dir / "full_period.csv", index=False)

    # Charts
    plot_timeseries_vs_spy(run, out_dir / "timeseries_portfolio_vs_spy.png")
    plot_rolling_excess(per_window, out_dir / "rolling_excess_bars.png")
    plot_rolling_sharpe(per_window, out_dir / "rolling_sharpe.png")
    plot_rolling_max_drawdown(per_window, out_dir / "rolling_max_drawdown.png")
    plot_rolling_beta(per_window, out_dir / "rolling_beta.png")
    plot_rolling_ic_spread(per_window, out_dir / "rolling_ic_spread.png")

    # Markdown
    regime_read = _summarize_regimes(per_window)
    lines: list[str] = [
        "# Regime-by-regime performance analysis — rolling-5y model",
        "",
        "Focused deep dive on the winning `rolling5y` policy from the semiannual walk-forward "
        f"regime study. For each 6-month test window we form a monthly-rebalanced top-{int(TOP_PCT*100)}% "
        "long book on the frozen composite scores and measure it against SPY on the identical "
        "grid; per-window composite IC (all horizons) and Q5–Q1 spread come from the "
        "point-in-time forward-return machinery.",
        "",
        "> **Coverage note.** Each 6-month window uses monthly formation with 1-month realised "
        "returns. Because the composite 3M IC (and any 3-month forward hold at the last "
        "formation of a window) settles inside the next window, the *effective* evaluation "
        "coverage of a nominal H1/H2 window is ≈9 months once overlapping holdings are counted. "
        "Per-window CAGR/Sharpe below are annualised from the strict 6-month monthly returns; "
        "read them alongside the 3-year buckets which pool the returns cleanly.",
        "",
        "## Regime read",
        "",
        regime_read,
        "",
        "## Full-period aggregate",
        "",
        _md_full(full),
        "",
        "## 3-year buckets",
        "",
        _md_bucket(bucket),
        "",
        "## Per test window",
        "",
        _md_per_window(per_window),
        "",
        "## Files",
        "",
        "* `per_window.csv` — all metrics per test window",
        "* `bucket_3y.csv` — 3-year rolling buckets (2017-2019, 2020-2022, 2023-2025, 2026+)",
        "* `full_period.csv` — full-sample aggregate",
        "* `timeseries_portfolio_vs_spy.png` — cumulative equity, portfolio vs SPY",
        "* `rolling_excess_bars.png` — per-window excess CAGR vs SPY (signed bars)",
        "* `rolling_sharpe.png` — per-window Sharpe, portfolio vs SPY",
        "* `rolling_max_drawdown.png` — per-window max DD, portfolio vs SPY",
        "* `rolling_beta.png` — per-window beta to SPY (β=1 reference)",
        "* `rolling_ic_spread.png` — per-window 6M composite IC (bars) + Q5–Q1 spread (line)",
        "",
    ]
    (out_dir / "REGIME5Y_REPORT.md").write_text("\n".join(lines))
