"""CSV, chart and markdown outputs for the factor research framework.

Pure rendering: every function takes already-assembled tables and writes to
``out_dir``. Nothing here computes a metric or makes a verdict — that lives in the
metric modules and :mod:`research.classify` — so the report can be regenerated
without re-scoring.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .classify import KEEP, MERGE, REMOVE, INSUFFICIENT

VERDICT_COLOR = {KEEP: "#2ca02c", MERGE: "#ff7f0e", REMOVE: "#d62728",
                 INSUFFICIENT: "#7f7f7f"}
_QCOLS = ["q1_ret", "q2_ret", "q3_ret", "q4_ret", "q5_ret"]


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------
def write_csv_outputs(
    out_dir: Path,
    factor_table: pd.DataFrame,
    subfactor_table: pd.DataFrame,
    quint_long: pd.DataFrame,
    ic_long: pd.DataFrame,
    redundancy_matrices: dict[str, pd.DataFrame],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    factor_table.to_csv(out_dir / "factor_level_summary.csv", index=False)
    subfactor_table.to_csv(out_dir / "subfactor_summary.csv", index=False)
    quint_long.to_csv(out_dir / "quintile_returns.csv", index=False)
    ic_long.to_csv(out_dir / "ic_monthly.csv", index=False)
    for parent, corr in redundancy_matrices.items():
        corr.to_csv(out_dir / f"redundancy_{parent}.csv")


# ---------------------------------------------------------------------------
# Quintile charts
# ---------------------------------------------------------------------------
def render_parent_quintiles(
    out_dir: Path, quint_by_h: dict[str, pd.DataFrame], parent_keys: list[str]
) -> None:
    for horizon, table in quint_by_h.items():
        rows = table[table["signal"].isin(parent_keys)]
        rows = rows[rows["n_periods"] > 0]
        if rows.empty:
            continue
        n = len(rows)
        ncol = min(4, n)
        nrow = int(np.ceil(n / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3.2 * nrow),
                                 squeeze=False)
        for ax, (_, r) in zip(axes.flat, rows.iterrows()):
            vals = [r.get(c, np.nan) for c in _QCOLS]
            colors = ["#d62728" if v < 0 else "#2ca02c" for v in vals]
            ax.bar([f"Q{i+1}" for i in range(5)], vals, color=colors)
            ax.axhline(0, color="black", lw=0.7)
            ax.set_title(f"{r['signal']}  (mono {r.get('monotonicity', np.nan):+.2f}, "
                         f"Q5-Q1 {r.get('spread_q5_q1', np.nan):+.3f})", fontsize=9)
            ax.grid(True, axis="y", alpha=0.3)
        for ax in axes.flat[n:]:
            ax.axis("off")
        fig.suptitle(f"Parent factor quintiles — {horizon} forward return "
                     f"(Q1=low score, Q5=high)", fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.savefig(out_dir / f"parent_quintiles_{horizon}.png", dpi=140)
        plt.close(fig)


def render_subfactor_quintiles(
    out_dir: Path, quint_by_h: dict[str, pd.DataFrame], panel, horizon: str = "1M"
) -> None:
    table = quint_by_h.get(horizon)
    if table is None:
        return
    parents = [p for p, subs in panel.sub_by_parent.items() if len(subs) >= 2]
    for parent in parents:
        subs = panel.sub_by_parent[parent]
        rows = table[table["signal"].isin(subs)]
        rows = rows[rows["n_periods"] > 0]
        if rows.empty:
            continue
        n = len(rows)
        ncol = min(3, n)
        nrow = int(np.ceil(n / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3 * nrow),
                                 squeeze=False)
        for ax, (_, r) in zip(axes.flat, rows.iterrows()):
            vals = [r.get(c, np.nan) for c in _QCOLS]
            colors = ["#d62728" if v < 0 else "#2ca02c" for v in vals]
            ax.bar([f"Q{i+1}" for i in range(5)], vals, color=colors)
            ax.axhline(0, color="black", lw=0.7)
            ax.set_title(f"{r['signal']}  (mono {r.get('monotonicity', np.nan):+.2f})",
                         fontsize=8)
            ax.grid(True, axis="y", alpha=0.3)
        for ax in axes.flat[n:]:
            ax.axis("off")
        fig.suptitle(f"{parent}: sub-factor quintiles — {horizon} forward return",
                     fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.savefig(out_dir / f"subfactor_quintiles_{parent}.png", dpi=140)
        plt.close(fig)


# ---------------------------------------------------------------------------
# IC charts
# ---------------------------------------------------------------------------
def render_parent_ic_by_horizon(
    out_dir: Path, factor_table: pd.DataFrame, horizons: list[str]
) -> None:
    cols = [f"ic_{h}" for h in horizons if f"ic_{h}" in factor_table.columns]
    if factor_table.empty or not cols:
        return
    df = factor_table.set_index("parent")[cols]
    fig, ax = plt.subplots(figsize=(max(8, 1.2 * len(df)), 5))
    x = np.arange(len(df))
    width = 0.8 / len(cols)
    for i, c in enumerate(cols):
        ax.bar(x + i * width, df[c].values, width, label=c.replace("ic_", ""))
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x + width * (len(cols) - 1) / 2)
    ax.set_xticklabels(df.index, rotation=30, ha="right")
    ax.set_ylabel("Mean IC")
    ax.set_title("Parent factor mean IC by forward horizon")
    ax.legend(title="horizon"); ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(out_dir / "parent_ic_by_horizon.png", dpi=140)
    plt.close(fig)


def render_subfactor_ic_ir(out_dir: Path, subfactor_table: pd.DataFrame) -> None:
    df = subfactor_table[subfactor_table["n_periods"] > 0].copy()
    if df.empty:
        return
    df = df.sort_values("ir_1M", na_position="first")
    colors = [VERDICT_COLOR.get(v, "#7f7f7f") for v in df["verdict"]]
    fig, axes = plt.subplots(1, 2, figsize=(15, max(6, 0.32 * len(df))), sharey=True)
    axes[0].barh(df["sub_factor"], df["ir_1M"], color=colors)
    axes[0].axvline(0, color="black", lw=0.8)
    axes[0].set_title("1M Information Ratio"); axes[0].grid(True, axis="x", alpha=0.3)
    axes[1].barh(df["sub_factor"], df["ic_1M"], color=colors)
    axes[1].axvline(0, color="black", lw=0.8)
    axes[1].set_title("1M Mean IC"); axes[1].grid(True, axis="x", alpha=0.3)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in VERDICT_COLOR.values()]
    axes[1].legend(handles, list(VERDICT_COLOR.keys()), loc="lower right", fontsize=8)
    fig.suptitle("Sub-factor predictive power, colored by verdict", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(out_dir / "subfactor_ic_ir.png", dpi=140); plt.close(fig)


def render_redundancy(out_dir: Path, redundancy_matrices: dict[str, pd.DataFrame]) -> None:
    items = [(p, m) for p, m in redundancy_matrices.items() if m is not None and m.shape[0] >= 2]
    if not items:
        return
    n = len(items)
    ncol = min(3, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.5 * ncol, 5 * nrow), squeeze=False)
    for ax, (parent, corr) in zip(axes.flat, items):
        im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(corr))); ax.set_yticks(range(len(corr)))
        ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(corr.index, fontsize=7)
        ax.set_title(f"{parent}: sub-factor score correlation", fontsize=10)
        for i in range(len(corr)):
            for j in range(len(corr)):
                v = corr.values[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6,
                            color="white" if abs(v) > 0.6 else "black")
        plt.colorbar(im, ax=ax, fraction=0.046)
    for ax in axes.flat[n:]:
        ax.axis("off")
    fig.tight_layout(); fig.savefig(out_dir / "redundancy_heatmaps.png", dpi=140)
    plt.close(fig)


def render_verdict_summary(out_dir: Path, subfactor_table: pd.DataFrame) -> None:
    df = subfactor_table.copy()
    if df.empty:
        return
    order = [KEEP, MERGE, REMOVE, INSUFFICIENT]
    counts = df["verdict"].value_counts().reindex(order).fillna(0)
    judged = df[df["verdict"].isin([KEEP, MERGE, REMOVE])].copy()
    judged = judged.sort_values("keep_score")
    fig, axes = plt.subplots(1, 2, figsize=(15, max(6, 0.32 * max(len(judged), 1))))
    axes[0].bar(order, counts.values, color=[VERDICT_COLOR[v] for v in order])
    axes[0].set_title("Sub-factor verdicts")
    for i, v in enumerate(counts.values):
        axes[0].text(i, v, int(v), ha="center", va="bottom")
    axes[0].grid(True, axis="y", alpha=0.3)
    if not judged.empty:
        colors = [VERDICT_COLOR.get(v, "#7f7f7f") for v in judged["verdict"]]
        axes[1].barh(judged["sub_factor"], judged["keep_score"], color=colors)
        axes[1].axvline(0, color="black", lw=0.8)
        axes[1].set_title("Keep-score ranking (judged sub-factors)")
        axes[1].grid(True, axis="x", alpha=0.3)
    fig.tight_layout(); fig.savefig(out_dir / "verdict_summary.png", dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------
def write_markdown(
    out_dir: Path,
    *,
    meta: dict,
    factor_table: pd.DataFrame,
    subfactor_table: pd.DataFrame,
    thresholds,
    horizons: list[str],
    watch: pd.DataFrame | None = None,
) -> None:
    lines: list[str] = []
    w = lines.append
    w("# Factor Research & Validation Report")
    w("")
    w(f"- **Window:** {meta['start']} → {meta['end']}  ·  **rebalance:** {meta['freq']}")
    w(f"- **Rebalances scored:** {meta['n_rebalances']}  ·  "
      f"**universe:** {meta['n_universe']} names")
    w(f"- **Forward horizons:** {', '.join(horizons)}")
    w("")
    w("All metrics are strictly point-in-time: a score at date *d* is paired only "
      "with returns realized after *d*. Multi-month horizons overlap, so "
      "information ratios are comparative, not independent-sample t-stats.")
    w("")

    counts = subfactor_table["verdict"].value_counts()
    w("## Verdict summary")
    w("")
    for v in [KEEP, MERGE, REMOVE, INSUFFICIENT]:
        w(f"- **{v}:** {int(counts.get(v, 0))}")
    w("")
    n_judged = int(counts.get(KEEP, 0) + counts.get(MERGE, 0) + counts.get(REMOVE, 0))
    w(f"Of {len(subfactor_table)} sub-factors, {n_judged} have enough point-in-time "
      f"history to judge; {int(counts.get(INSUFFICIENT, 0))} are untested (thin Layer 1 "
      f"history) and left in place pending data.")
    w("")

    w("## Leaner recommended model")
    w("")
    keep = subfactor_table[subfactor_table["verdict"] == KEEP]
    for parent in factor_table["parent"]:
        kept = keep[keep["parent"] == parent]["sub_factor"].tolist()
        insuff = subfactor_table[(subfactor_table["parent"] == parent)
                                 & (subfactor_table["verdict"] == INSUFFICIENT)]["sub_factor"].tolist()
        bits = []
        if kept:
            bits.append("keep " + ", ".join(kept))
        if insuff:
            bits.append("untested (retain): " + ", ".join(insuff))
        w(f"- **{parent}** — {'; '.join(bits) if bits else 'no sub-factor retained'}")
    w("")

    w("## Parent factor scorecard")
    w("")
    w(_md_table(factor_table, _parent_cols(factor_table, horizons)))
    w("")

    w("## Sub-factor verdicts")
    w("")
    cols = ["sub_factor", "parent", "verdict", "n_periods", "ic_1M", "ir_1M",
            "hit_1M", "spread_1M", "monotonicity", "max_abs_sibling_corr",
            "incremental_ic", "delta_parent_ic"]
    cols = [c for c in cols if c in subfactor_table.columns]
    ordered = subfactor_table.sort_values(
        ["verdict", "keep_score"], ascending=[True, False])
    w(_md_table(ordered, cols))
    w("")
    w("## Why each judged sub-factor got its verdict")
    w("")
    for _, r in ordered.iterrows():
        if r["verdict"] == INSUFFICIENT:
            continue
        w(f"- **{r['sub_factor']}** → *{r['verdict']}*: {r['reason']}")
    w("")

    if watch is not None and not watch.empty:
        w("## Redundancy watch (near-duplicate Keep pairs)")
        w("")
        w("Both sides cleared the Keep bar (each adds a little unique signal), but "
          "they rank names alike above the correlation cut — candidates to fold by "
          "hand for a leaner model:")
        w("")
        for _, r in watch.iterrows():
            w(f"- fold **{r['fold']}** into **{r['into']}** (score corr {r['corr']:.2f})")
        w("")

    w("## Methodology & thresholds")
    w("")
    w("- **Quintiles:** names bucketed Q1(low)→Q5(high) by score; Q5−Q1 spread and "
      "Spearman monotonicity of bucket means.")
    w("- **IC:** cross-sectional Spearman of score vs forward return; IR = mean/std; "
      "hit rate = share of positive periods; stability = share of rolling-6 windows "
      "keeping the full-sample sign.")
    w("- **Redundancy:** average per-date Spearman correlation among a parent's "
      "sub-factor scores (do they rank names alike?).")
    w("- **Incremental:** IC of the sub residualized on its siblings (unique edge), "
      "and the change in parent IC from including the sub (drop-one).")
    w(f"- **Cut-offs:** min_periods={thresholds.min_periods}, "
      f"corr_high={thresholds.corr_high}, incremental_min={thresholds.incremental_min}, "
      f"delta_keep={thresholds.delta_keep}, ir_weak={thresholds.ir_weak}.")
    w("")

    (out_dir / "REPORT.md").write_text("\n".join(lines))


def _parent_cols(table: pd.DataFrame, horizons: list[str]) -> list[str]:
    cols = ["parent", "n_periods"]
    cols += [f"ic_{h}" for h in horizons if f"ic_{h}" in table.columns]
    cols += [c for c in ["ir_1M", "hit_1M", "stability_1M", "spread_1M", "mono_1M"]
             if c in table.columns]
    return cols


def _md_table(df: pd.DataFrame, cols: list[str]) -> str:
    cols = [c for c in cols if c in df.columns]
    head = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    out = [head, sep]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, float):
                cells.append("" if pd.isna(v) else f"{v:.4f}")
            else:
                cells.append("" if pd.isna(v) else str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)
