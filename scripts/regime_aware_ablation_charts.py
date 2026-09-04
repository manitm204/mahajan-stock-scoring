"""Charts + tables for the single-ingredient ablation battery (5Y, B, 12M,
VIXOnly, VIX+12M, VIX+LongRun, C). Reads output/regime_aware/{walkforward_
comparison,state_history}.csv, writes PNGs to output/regime_aware/ablation/.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

OUT_DIR = Path("output/regime_aware")
FIG_DIR = OUT_DIR / "ablation"
FIG_DIR.mkdir(parents=True, exist_ok=True)

ORDER = ["5Y", "12M", "B", "VIXOnly", "VIX+12M", "VIX+LongRun", "C"]
COLORS = {"5Y": "#4C72B0", "12M": "#DD8452", "B": "#55A868", "VIXOnly": "#C44E52",
         "VIX+12M": "#8172B2", "VIX+LongRun": "#937860", "C": "#64B5CD"}
LABELS = {"5Y": "5Y only", "12M": "12M only", "B": "Long-run only",
         "VIXOnly": "VIX only", "VIX+12M": "VIX+12M (50/50)",
         "VIX+LongRun": "VIX+LongRun (50/50)", "C": "C (reference blend)"}


def _ordered_windows(df: pd.DataFrame) -> list[str]:
    return sorted(df["window"].unique(), key=lambda w: (w.split("-")[0], w.split("-")[1]))


def plot_summary_table(df: pd.DataFrame) -> None:
    cols = [("sharpe", ".2f"), ("ic_6m", ".4f"), ("ic_ir_6m", ".3f"),
           ("max_drawdown", ".1%"), ("avg_turnover", ".2f"),
           ("subfactor_churn_per_year", ".1f"), ("spy_excess_cagr", ".1%")]
    headers = ["Variant", "Sharpe", "IC(6M)", "IC-IR", "MaxDD", "Turnover",
              "Churn/yr", "Excess vs SPY"]
    means = df.groupby("variant")[[c for c, _ in cols]].mean()
    rows = []
    for v in ORDER:
        r = means.loc[v]
        rows.append([LABELS[v]] + [format(r[c], fmt) for c, fmt in cols])

    fig, ax = plt.subplots(figsize=(13.5, 0.6 + 0.5 * len(ORDER)))
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=headers, cellLoc="center", loc="center",
                     colWidths=[0.19] + [0.135] * (len(headers) - 1))
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)
    for j in range(len(headers)):
        table[0, j].set_facecolor("#4C72B0")
        table[0, j].set_text_props(color="white", weight="bold")
    for i, v in enumerate(ORDER, start=1):
        table[i, 0].set_facecolor(COLORS[v] + "40")
    fig.suptitle("Full-Period Summary (mean across 19 windows, 2017-2026)", y=0.95)
    fig.savefig(FIG_DIR / "01_summary_table.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {FIG_DIR / '01_summary_table.png'}")


def plot_summary_bars(df: pd.DataFrame) -> None:
    metrics = [("sharpe", "Sharpe"), ("ic_6m", "IC (6M)"), ("ic_ir_6m", "IC-IR (6M)"),
              ("max_drawdown", "Max Drawdown"), ("avg_turnover", "Avg Turnover"),
              ("subfactor_churn_per_year", "Sub-factor Churn/yr")]
    means = df.groupby("variant")[[m for m, _ in metrics]].mean()
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    for ax, (metric, label) in zip(axes.flat, metrics):
        vals = [means.loc[v, metric] for v in ORDER]
        bars = ax.bar(range(len(ORDER)), vals, color=[COLORS[v] for v in ORDER])
        ax.set_xticks(range(len(ORDER)))
        ax.set_xticklabels([LABELS[v] for v in ORDER], rotation=45, ha="right", fontsize=8)
        ax.set_title(label)
        ax.axhline(0, color="black", linewidth=0.6)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v,
                   f"{v:.3f}" if abs(v) < 1 else f"{v:.2f}",
                   ha="center", va="bottom" if v >= 0 else "top", fontsize=7)
    fig.suptitle("Single-Ingredient Ablation: Full-Period Means")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "02_summary_bars.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {FIG_DIR / '02_summary_bars.png'}")


def plot_timeseries_panel(df: pd.DataFrame) -> None:
    windows = _ordered_windows(df)
    metrics = [("sharpe", "Sharpe"), ("ic_6m", "IC (6M)"), ("max_drawdown", "Max Drawdown")]
    fig, axes = plt.subplots(len(metrics), 1, figsize=(14, 10), sharex=True)
    for ax, (metric, label) in zip(axes, metrics):
        for v in ORDER:
            sub = df[df["variant"] == v].set_index("window")[metric].reindex(windows)
            ax.plot(windows, sub, color=COLORS[v], linewidth=1.4, label=LABELS[v])
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_ylabel(label)
        ax.grid(alpha=0.25)
    axes[0].legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8, frameon=False)
    axes[-1].set_xticks(range(len(windows)))
    axes[-1].set_xticklabels(windows, rotation=60, ha="right", fontsize=8)
    fig.suptitle("Per-Window Trajectory, All 7 Variants")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "03_timeseries_panel.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {FIG_DIR / '03_timeseries_panel.png'}")


def plot_distribution_boxplots(df: pd.DataFrame) -> None:
    metrics = [("sharpe", "Sharpe"), ("ic_6m", "IC (6M)"),
              ("max_drawdown", "Max Drawdown"), ("avg_turnover", "Avg Turnover")]
    fig, axes = plt.subplots(1, len(metrics), figsize=(16, 4.5))
    for ax, (metric, label) in zip(axes, metrics):
        data = [df[df["variant"] == v][metric].dropna() for v in ORDER]
        ax.boxplot(data, tick_labels=[LABELS[v] for v in ORDER], showmeans=True,
                  showfliers=False)
        ax.set_title(label)
        ax.tick_params(axis="x", rotation=60, labelsize=8)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.grid(alpha=0.25, axis="y")
    fig.suptitle("Distribution Across 18 of 19 Windows (2021-H1 outlier excluded for scale)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "04_distribution_boxplots.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {FIG_DIR / '04_distribution_boxplots.png'}")


def plot_head_to_head(df: pd.DataFrame) -> None:
    windows = _ordered_windows(df)
    others = [v for v in ORDER if v != "B"]
    wins, losses = [], []
    for v in others:
        da = df[df["variant"] == v].set_index("window")["sharpe"].reindex(windows)
        db = df[df["variant"] == "B"].set_index("window")["sharpe"].reindex(windows)
        diff = (da - db).dropna()
        wins.append(int((diff > 0).sum()))
        losses.append(int((diff <= 0).sum()))

    fig, ax = plt.subplots(figsize=(9, 5))
    x = range(len(others))
    ax.bar(x, wins, label="Beats B on Sharpe", color="#55A868")
    ax.bar(x, losses, bottom=wins, label="Loses/ties vs B", color="#C44E52")
    ax.set_xticks(list(x))
    ax.set_xticklabels([LABELS[v] for v in others], rotation=30, ha="right", fontsize=9)
    ax.set_ylabel(f"Windows (of {len(windows)})")
    ax.set_title("Head-to-Head Win Rate vs Long-Run-Only (B), by Window")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "05_head_to_head_vs_B.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {FIG_DIR / '05_head_to_head_vs_B.png'}")


def plot_spy_comparison(df: pd.DataFrame) -> None:
    windows = _ordered_windows(df)
    spy_cagr = (df["cagr"] - df["spy_excess_cagr"]).groupby(df["window"]).mean().reindex(windows)

    fig, ax = plt.subplots(figsize=(14, 6))
    spy_equity = ((1.0 + spy_cagr.fillna(0)) ** 0.5).cumprod()
    ax.plot(windows, spy_equity, color="black", linewidth=2.4, label="SPY (derived)")
    for v in ORDER:
        sub = df[df["variant"] == v].set_index("window")["cagr"].reindex(windows)
        growth = (1.0 + sub.fillna(0)) ** 0.5
        ax.plot(windows, growth.cumprod(), color=COLORS[v], linewidth=1.4, label=LABELS[v])
    ax.set_ylabel("Cumulative growth (start = 1.0)")
    ax.set_yscale("log")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title("Cumulative Growth vs SPY (log scale)")
    ax.set_xticks(range(len(windows)))
    ax.set_xticklabels(windows, rotation=60, ha="right", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "06_spy_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {FIG_DIR / '06_spy_comparison.png'}")


def plot_parent_weight_comparison(state_log: pd.DataFrame) -> None:
    pure = ["B", "12M", "VIXOnly"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    for ax, key in zip(axes, pure):
        sub = state_log[state_log["variant"] == key]
        mat = sub.pivot(index="cutoff", columns="parent", values="weight").sort_index()
        dates = pd.to_datetime(mat.index)
        ax.stackplot(dates, mat.T.to_numpy(dtype=float), labels=mat.columns, alpha=0.85)
        ax.set_ylim(0, 1)
        ax.set_title(LABELS[key])
        ax.tick_params(axis="x", rotation=45)
    axes[0].set_ylabel("Realized parent weight")
    axes[-1].legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8, frameon=False)
    fig.suptitle("Monthly Realized Parent-Weight Path: the 3 \"Pure\" Ingredients")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "07_parent_weight_pure_ingredients.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {FIG_DIR / '07_parent_weight_pure_ingredients.png'}")


def plot_ranking_table(df: pd.DataFrame) -> None:
    cols = ["sharpe", "ic_ir_6m", "subfactor_churn_per_year"]
    means = df.groupby("variant")[cols].mean()
    rank_sharpe = means["sharpe"].rank(ascending=False).astype(int)
    rank_icir = means["ic_ir_6m"].rank(ascending=False).astype(int)
    rank_churn = means["subfactor_churn_per_year"].rank(ascending=True).astype(int)

    rows = []
    for v in ORDER:
        rows.append([LABELS[v], f"#{rank_sharpe[v]}", f"#{rank_icir[v]}", f"#{rank_churn[v]}"])
    headers = ["Variant", "Sharpe rank", "IC-IR rank", "Churn rank (lower=more stable)"]

    fig, ax = plt.subplots(figsize=(11, 0.6 + 0.5 * len(ORDER)))
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=headers, cellLoc="center", loc="center",
                     colWidths=[0.24, 0.2, 0.2, 0.32])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)
    for j in range(len(headers)):
        table[0, j].set_facecolor("#4C72B0")
        table[0, j].set_text_props(color="white", weight="bold")
    for i, v in enumerate(ORDER, start=1):
        table[i, 0].set_facecolor(COLORS[v] + "40")
    fig.suptitle("Rank Summary (1 = best)", y=0.95)
    fig.savefig(FIG_DIR / "08_rank_table.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {FIG_DIR / '08_rank_table.png'}")


def main() -> None:
    df = pd.read_csv(OUT_DIR / "walkforward_comparison.csv")
    state_log = pd.read_csv(OUT_DIR / "state_history.csv")
    plot_summary_table(df)
    plot_summary_bars(df)
    plot_timeseries_panel(df)
    plot_distribution_boxplots(df)
    plot_head_to_head(df)
    plot_spy_comparison(df)
    plot_parent_weight_comparison(state_log)
    plot_ranking_table(df)


if __name__ == "__main__":
    main()
