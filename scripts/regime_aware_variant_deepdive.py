"""Deeper visual breakdown of the regime-aware study, beyond the full-period-mean
bar chart: per-window trajectories, distribution/consistency across windows,
parent-weight allocation compared across evidence modes, and head-to-head win
rates. Reads output/regime_aware/{walkforward_comparison,state_history}.csv,
writes 4 PNGs to output/regime_aware/.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

OUT_DIR = Path("output/regime_aware")
MODES = [("B", "5Y (long-run)", "#4C72B0"), ("C", "VIX-weighted", "#DD8452"),
         ("24M", "24-month", "#55A868")]
WINDOW_ORDER = None  # filled from data, kept chronological via categorical order


def _ordered_windows(df: pd.DataFrame) -> list[str]:
    return sorted(df["window"].unique(), key=lambda w: (w.split("-")[0], w.split("-")[1]))


def plot_timeseries_panel(df: pd.DataFrame) -> None:
    windows = _ordered_windows(df)
    metrics = [("sharpe", "Sharpe"), ("ic_6m", "IC (6M)"), ("max_drawdown", "Max Drawdown")]
    fig, axes = plt.subplots(len(metrics), 1, figsize=(13, 10), sharex=True)
    for ax, (metric, label) in zip(axes, metrics):
        for key, name, color in MODES:
            for variant, style, alpha in ((key, "-", 1.0), (f"{key}+2%floor", "--", 0.7)):
                sub = df[df["variant"] == variant].set_index("window").reindex(windows)
                ax.plot(windows, sub[metric], style, color=color, alpha=alpha,
                       label=f"{name}{' +floor' if style == '--' else ''}", linewidth=1.6)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_ylabel(label)
        ax.grid(alpha=0.25)
    axes[0].legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8, frameon=False)
    axes[-1].set_xticks(range(len(windows)))
    axes[-1].set_xticklabels(windows, rotation=60, ha="right", fontsize=8)
    fig.suptitle("Regime-Aware Study: Per-Window Trajectory (solid = no floor, dashed = 2% floor)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "variant_timeseries_panel.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT_DIR / 'variant_timeseries_panel.png'}")


def plot_distribution_boxplots(df: pd.DataFrame) -> None:
    order = ["B", "B+2%floor", "C", "C+2%floor", "24M", "24M+2%floor"]
    metrics = [("sharpe", "Sharpe"), ("ic_6m", "IC (6M)"),
              ("max_drawdown", "Max Drawdown"), ("avg_turnover", "Avg Turnover")]
    fig, axes = plt.subplots(1, len(metrics), figsize=(15, 4.5))
    for ax, (metric, label) in zip(axes, metrics):
        data = [df[df["variant"] == v][metric].dropna() for v in order]
        # 2021-H1 (COVID-recovery window) is a genuine outlier on sharpe/ic/drawdown
        # that's already visible in variant_timeseries_panel.png -- showfliers=False
        # here so it doesn't blow out the axis and hide the actual box shapes.
        ax.boxplot(data, tick_labels=order, showmeans=True, showfliers=False)
        ax.set_title(label)
        ax.tick_params(axis="x", rotation=60, labelsize=8)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.grid(alpha=0.25, axis="y")
    fig.suptitle("Regime-Aware Study: Distribution Across 18 of 19 Windows (2021-H1 outlier "
                "excluded for scale -- see variant_timeseries_panel.png)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "variant_distribution_boxplots.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT_DIR / 'variant_distribution_boxplots.png'}")


def plot_parent_weight_comparison(state_log: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    for ax, (key, name, _) in zip(axes, MODES):
        sub = state_log[state_log["variant"] == key]
        mat = sub.pivot(index="cutoff", columns="parent", values="weight").sort_index()
        dates = pd.to_datetime(mat.index)
        ax.stackplot(dates, mat.T.to_numpy(dtype=float), labels=mat.columns, alpha=0.85)
        ax.set_ylim(0, 1)
        ax.set_title(name)
        ax.tick_params(axis="x", rotation=45)
    axes[0].set_ylabel("Realized parent weight")
    axes[-1].legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8, frameon=False)
    fig.suptitle("Monthly Realized Parent-Weight Path by Evidence Mode (no floor)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "parent_weight_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT_DIR / 'parent_weight_comparison.png'}")


def plot_head_to_head(df: pd.DataFrame) -> None:
    windows = _ordered_windows(df)
    pairs = [("C", "B", "No floor"), ("C+2%floor", "B+2%floor", "2% floor"),
            ("24M", "B", "No floor"), ("24M+2%floor", "B+2%floor", "2% floor")]
    labels = ["C vs B\n(no floor)", "C+fl vs B+fl\n(floor)",
             "24M vs B\n(no floor)", "24M+fl vs B+fl\n(floor)"]
    wins, losses, ties = [], [], []
    for a, b, _ in pairs:
        da = df[df["variant"] == a].set_index("window")["sharpe"].reindex(windows)
        db = df[df["variant"] == b].set_index("window")["sharpe"].reindex(windows)
        diff = (da - db).dropna()
        wins.append(int((diff > 0).sum()))
        losses.append(int((diff < 0).sum()))
        ties.append(int((diff == 0).sum()))

    fig, ax = plt.subplots(figsize=(8, 5))
    x = range(len(pairs))
    ax.bar(x, wins, label="Beats B on Sharpe", color="#55A868")
    ax.bar(x, losses, bottom=wins, label="Loses to B on Sharpe", color="#C44E52")
    ax.bar(x, ties, bottom=[w + l for w, l in zip(wins, losses)], label="Tie",
          color="#CCCCCC")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel(f"Windows (of {len(windows)})")
    ax.set_title("Head-to-Head Win Rate vs Plain 5Y Lookback (B), by Window")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "head_to_head_winrate.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT_DIR / 'head_to_head_winrate.png'}")


def _spy_cagr_by_window(df: pd.DataFrame, windows: list[str]) -> pd.Series:
    """SPY's own per-window CAGR isn't a stored column, but
    spy_excess_cagr = strategy_cagr - spy_cagr (see performance_metrics/
    benchmark_stats in research/walkforward/portfolio.py), so spy_cagr =
    cagr - spy_excess_cagr. Every variant sees the identical SPY series for a
    given window, so this is recomputed per (window, variant) row and then
    averaged across variants -- the small cross-variant spread is just
    period-alignment noise, not a real difference in what SPY did."""
    spy_cagr = df["cagr"] - df["spy_excess_cagr"]
    return spy_cagr.groupby(df["window"]).mean().reindex(windows)


def plot_spy_comparison(df: pd.DataFrame) -> None:
    windows = _ordered_windows(df)
    spy_cagr = _spy_cagr_by_window(df, windows)

    fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=True)

    # (1) Cumulative growth, indexed to 1.0 -- each semiannual window's CAGR is
    # de-annualized back to a ~6-month period return and compounded across the
    # 19 windows. Approximation: assumes each window is exactly 0.5 years,
    # which is true for every window except a possibly-truncated final partial
    # window -- fine for an illustrative growth-path comparison, not a precise
    # equity curve.
    ax = axes[0]
    spy_growth = (1.0 + spy_cagr.fillna(0)) ** 0.5
    spy_equity = spy_growth.cumprod()
    ax.plot(windows, spy_equity, color="black", linewidth=2.2, label="SPY (derived)")
    for key, name, color in MODES:
        sub = df[df["variant"] == key].set_index("window")["cagr"].reindex(windows)
        growth = (1.0 + sub.fillna(0)) ** 0.5
        ax.plot(windows, growth.cumprod(), color=color, linewidth=1.6, label=name)
    ax.set_ylabel("Cumulative growth (start = 1.0)")
    ax.set_yscale("log")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title("Cumulative Growth vs SPY (log scale, no-floor modes)")

    # (2) Excess CAGR vs SPY per window
    ax = axes[1]
    for key, name, color in MODES:
        sub = df[df["variant"] == key].set_index("window")["spy_excess_cagr"].reindex(windows)
        ax.plot(windows, sub, color=color, linewidth=1.6, label=name, marker="o", markersize=3)
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_ylabel("Excess CAGR vs SPY")
    ax.grid(alpha=0.25)
    ax.set_title("Excess CAGR vs SPY, Per Window")

    # (3) Information ratio vs SPY per window -- the closest available analog
    # to a "Sharpe vs SPY": mean(excess return)/tracking-error, annualised.
    ax = axes[2]
    for key, name, color in MODES:
        sub = df[df["variant"] == key].set_index("window")["spy_ir"].reindex(windows)
        ax.plot(windows, sub, color=color, linewidth=1.6, label=name, marker="o", markersize=3)
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_ylabel("Information Ratio vs SPY")
    ax.grid(alpha=0.25)
    ax.set_title("Information Ratio vs SPY, Per Window (SPY-relative risk-adjusted return)")
    ax.set_xticks(range(len(windows)))
    ax.set_xticklabels(windows, rotation=60, ha="right", fontsize=8)

    fig.suptitle("Regime-Aware Study vs SPY Benchmark")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "spy_comparison_panel.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT_DIR / 'spy_comparison_panel.png'}")


def main() -> None:
    df = pd.read_csv(OUT_DIR / "walkforward_comparison.csv")
    state_log = pd.read_csv(OUT_DIR / "state_history.csv")
    plot_timeseries_panel(df)
    plot_distribution_boxplots(df)
    plot_parent_weight_comparison(state_log)
    plot_spy_comparison(df)
    plot_head_to_head(df)


if __name__ == "__main__":
    main()
