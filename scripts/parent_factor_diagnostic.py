"""Parent-factor diagnostic study: rolling-5y regime read vs V4 composite weights.

Derives subfactor selection, intra-parent weights, and parent weights entirely from
the last 5 years of panel data (rolling-5y window, no external split / holdout).
Evaluates each of the 8 parent factors individually on the same 5-year window:

  - Q1-Q5 quintile buckets across the PIT universe
  - 3M (default) forward returns per bucket
  - IC, IR, hit rate, Q5-Q1 spread, monotonicity
  - Derived sub-weights and parent weights vs V4 production weights

NOTE: selection and evaluation use the same 5-year data — this is an IN-SAMPLE
regime read, not a held-out OOS study. Use output/walkforward/ for the strict OOS
10-year analysis. Purpose here is: "given the last 5 years, what would the model
pick and how well does each parent explain returns in that recent regime?"

Outputs to output/parent_diagnostic/:
  parent_oos_2025.csv          — per-parent IC/IR/hit/spread/mono (rolling 5y)
  parent_quintiles_3M.csv      — Q1-Q5 annualized returns per parent
  weight_vs_edge.csv           — derived vs V4 production weights vs in-sample IC
  parent_diagnostic.md         — full text report
  parent_ic_bar.png            — IC bar chart (helping / neutral / hurting)
  parent_quintile_profiles.png — Q1-Q5 return profiles for each parent
  weight_vs_edge.png           — weight vs IC scatter / bar comparison

Usage:
    python scripts/parent_factor_diagnostic.py
    python scripts/parent_factor_diagnostic.py --horizon 6M   # use 6M IC instead
    python scripts/parent_factor_diagnostic.py --rolling-years 3
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from backtesting import data_loader as dl
from data.db import get_db
from factors.parent_selection_v4 import V4_PARENT_WEIGHTS, SELECTED_SUBS
from research import HORIZON_MONTHS, compute_forward_returns
from research.panel import ScorePanel
from research.subfactor_expansion.panel import (
    cache_key as exp_cache_key,
    load_cached_panel as exp_load,
)
from research.walkforward.analysis import (
    composite_ic,
    quantile_analysis,
)
from research.walkforward.compose import build_parent_panel
from research.walkforward.compose import FrozenConfig, frozen_composite
from research.walkforward.selection import select_config, slice_panel
from research.walkforward.splits import PANEL_END, PANEL_START

CACHE_DIR = Path("cache/subfactor_expansion")
OUT_DIR = Path("output/parent_diagnostic")
PRICE_END = "2026-07-06"

PARENT_ORDER = [
    "momentum", "value", "quality", "growth",
    "revisions", "institutional", "insider", "short",
]

COLORS = {
    "helping":  "#2ca02c",  # green
    "neutral":  "#ff7f0e",  # orange
    "hurting":  "#d62728",  # red
}

IC_HELP = 0.008    # IC > this → helping
IC_HURT = -0.008   # IC < this → hurting


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #

def _adapt(cand) -> ScorePanel:
    parent_keys = [p for p, subs in cand.candidates_by_parent.items() if subs]
    return ScorePanel(
        rebal_dates=list(cand.rebal_dates), scores=cand.scores,
        parent_keys=parent_keys,
        sub_by_parent={p: list(cand.candidates_by_parent[p]) for p in parent_keys},
        universe=list(cand.universe),
    )


def _load_panel() -> ScorePanel:
    ckey = CACHE_DIR / exp_cache_key(PANEL_START, PANEL_END, "monthly")
    cand = exp_load(ckey)
    if cand is None:
        raise SystemExit(
            f"Panel cache not found at {ckey}. "
            "Run: python run_walkforward.py --rebuild-panel"
        )
    print(f"Loaded panel: {len(cand.rebal_dates)} rebalances, "
          f"{len(cand.universe)} PIT names.")
    return _adapt(cand)


# --------------------------------------------------------------------------- #
# Core computation
# --------------------------------------------------------------------------- #

def _run_rolling_window(panel: ScorePanel, matrix: pd.DataFrame,
                        sectors: pd.Series, rolling_years: int = 5) -> dict:
    """Derive sub-selection, sub-weights, and parent weights from the last N years.

    Uses all rebalances in [panel_end − N years, panel_end] for both selection and
    evaluation. boundary=None so forward-return windows can extend to the last
    available price date (incomplete final windows are dropped automatically).

    This is IN-SAMPLE for the rolling window — selection and evaluation share the
    same data. Purpose: regime read ("what works in the recent N-year regime").
    """
    all_dates = sorted(panel.rebal_dates)
    panel_end = pd.Timestamp(all_dates[-1])
    window_start = (panel_end - pd.DateOffset(years=rolling_years)).date().isoformat()
    window_rebals = [d for d in all_dates if d >= window_start]

    print(f"Rolling {rolling_years}y window: {window_rebals[0]}→{window_rebals[-1]} "
          f"({len(window_rebals)} rebalances)")

    print("Deriving sub-selection + weights from rolling window …")
    cfg = select_config(panel, window_rebals, matrix, boundary=None)

    print("Derived parent weights:")
    for p, w in sorted(cfg.parent_weights.items(), key=lambda x: -x[1]):
        print(f"  {p:15s} {w:.3f}")

    print("Derived sub-factor selections:")
    for p in PARENT_ORDER:
        wmap = cfg.sub_weights.get(p, {})
        subs_str = ", ".join(f"{s}={w:.3f}" for s, w in sorted(wmap.items(), key=lambda x: -x[1]))
        print(f"  {p:15s} {subs_str}")

    window_panel = slice_panel(panel, window_rebals)
    parent_panel = build_parent_panel(window_panel, cfg.sub_weights)

    parent_scores: dict[str, dict[str, pd.Series]] = {}
    for d in parent_panel.rebal_dates:
        frame = parent_panel.scores.get(d)
        if frame is None:
            continue
        for parent in parent_panel.parent_keys:
            col = frame.get(parent)
            if col is not None:
                parent_scores.setdefault(parent, {})[d] = col

    # Composite scores using the derived weights (for Q1-Q5 reference)
    composite_scores = frozen_composite(window_panel, window_rebals, cfg, sectors)

    fwd = compute_forward_returns(matrix, window_rebals, HORIZON_MONTHS)

    # 2025-only slice — same derived sub_weights, different date range
    rebals_2025 = [d for d in window_rebals if d.startswith("2025")]
    parent_scores_2025: dict[str, dict[str, pd.Series]] = {}
    if rebals_2025:
        panel_2025 = slice_panel(panel, rebals_2025)
        pp_2025 = build_parent_panel(panel_2025, cfg.sub_weights)
        for d in pp_2025.rebal_dates:
            frame = pp_2025.scores.get(d)
            if frame is None:
                continue
            for parent in pp_2025.parent_keys:
                col = frame.get(parent)
                if col is not None:
                    parent_scores_2025.setdefault(parent, {})[d] = col
    fwd_2025 = compute_forward_returns(matrix, rebals_2025, HORIZON_MONTHS) if rebals_2025 else {}

    return {
        "cfg": cfg,
        "sub_weights": cfg.sub_weights,
        "derived_parent_weights": cfg.parent_weights,
        "parent_scores": parent_scores,
        "parent_scores_2025": parent_scores_2025,
        "composite_scores": composite_scores,
        "fwd": fwd,
        "fwd_2025": fwd_2025,
        "window_rebals": window_rebals,
        "rebals_2025": rebals_2025,
        "window_start": window_rebals[0],
        "rolling_years": rolling_years,
    }


def _compute_parent_metrics(parent_scores: dict[str, dict[str, pd.Series]],
                             fwd: dict[str, dict[str, pd.Series]],
                             horizon: str = "3M") -> pd.DataFrame:
    """IC / IR / hit / spread / mono per parent for 2025 OOS."""
    rows = []
    fwd_h = fwd.get(horizon, {})
    months = HORIZON_MONTHS[horizon]
    ppy = 12.0 / months

    for parent in PARENT_ORDER:
        by_date = parent_scores.get(parent, {})
        ics = []
        spreads = []  # per-period Q5-Q1
        mono_flags = []

        for d, fwd_r in fwd_h.items():
            sc = by_date.get(d)
            if sc is None:
                continue
            df = pd.DataFrame({"s": sc, "f": fwd_r}).dropna()
            if len(df) < 25:
                continue
            ic = df["s"].corr(df["f"], method="spearman")
            if pd.notna(ic):
                ics.append(ic)

            # per-period Q5-Q1 spread
            try:
                buckets = pd.qcut(df["s"].rank(method="first"), 5, labels=False)
                qmeans = df["f"].groupby(buckets).mean()
                if len(qmeans) == 5 and qmeans.notna().all():
                    spreads.append(float(qmeans.iloc[4] - qmeans.iloc[0]))
                    diffs = np.diff(qmeans.values)
                    mono_flags.append(bool(np.all(diffs > 0)))
            except ValueError:
                pass

        arr = np.array(ics, dtype=float)
        n = int(arr.size)
        if n == 0:
            rows.append({"parent": parent, "n_periods": 0})
            continue
        mean_ic = float(arr.mean())
        std_ic = float(arr.std(ddof=1)) if n > 1 else np.nan
        ir = mean_ic / std_ic if std_ic and std_ic > 1e-9 else np.nan
        hit = float((arr > 0).mean())

        spread_arr = np.array(spreads, dtype=float)
        mean_spread = float(spread_arr.mean()) if spread_arr.size > 0 else np.nan
        ann_spread = float((1 + mean_spread) ** ppy - 1) if not np.isnan(mean_spread) else np.nan
        mono_rate = float(np.mean(mono_flags)) if mono_flags else np.nan

        verdict = ("helping" if mean_ic > IC_HELP
                   else "hurting" if mean_ic < IC_HURT
                   else "neutral")
        rows.append({
            "parent": parent,
            "n_periods": n,
            "mean_ic": mean_ic,
            "ic_std": std_ic,
            "ir": ir,
            "hit_rate": hit,
            "mean_spread_3m": mean_spread,
            "ann_spread": ann_spread,
            "mono_rate": mono_rate,
            "verdict": verdict,
        })
    return pd.DataFrame(rows)


def _compute_quintile_profiles(parent_scores: dict[str, dict[str, pd.Series]],
                                fwd: dict[str, dict[str, pd.Series]],
                                horizon: str = "3M") -> dict[str, pd.DataFrame]:
    """Per-parent average Q1-Q5 forward return (annualized)."""
    fwd_h = fwd.get(horizon, {})
    months = HORIZON_MONTHS[horizon]
    ppy = 12.0 / months
    out: dict[str, pd.DataFrame] = {}

    for parent in PARENT_ORDER:
        by_date = parent_scores.get(parent, {})
        bucket_rets: list[np.ndarray] = []

        for d, fwd_r in fwd_h.items():
            sc = by_date.get(d)
            if sc is None:
                continue
            df = pd.DataFrame({"s": sc, "f": fwd_r}).dropna()
            if len(df) < 25:
                continue
            try:
                buckets = pd.qcut(df["s"].rank(method="first"), 5, labels=False)
                qm = df["f"].groupby(buckets).mean().reindex(range(5))
                if qm.notna().all():
                    bucket_rets.append(qm.to_numpy(dtype=float))
            except ValueError:
                pass

        if not bucket_rets:
            out[parent] = pd.DataFrame(
                {"quintile": [f"Q{i+1}" for i in range(5)],
                 "avg_return_3m": [np.nan] * 5,
                 "ann_return": [np.nan] * 5})
            continue

        mat = np.array(bucket_rets)
        avg = mat.mean(axis=0)
        ann = (1 + avg) ** ppy - 1
        out[parent] = pd.DataFrame({
            "quintile": [f"Q{i+1}" for i in range(5)],
            "avg_return_3m": avg.tolist(),
            "ann_return": ann.tolist(),
        })
    return out


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #

def _plot_ic_bar(metrics: pd.DataFrame, horizon: str, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    parents = metrics["parent"].tolist()
    ics = metrics["mean_ic"].tolist()
    verdicts = metrics.get("verdict", pd.Series(["neutral"] * len(parents))).tolist()
    colors = [COLORS.get(v, COLORS["neutral"]) for v in verdicts]

    bars = ax.bar(parents, ics, color=colors, edgecolor="black", linewidth=0.5, alpha=0.85)

    # Error bars (±1 SE = ic_std / sqrt(n))
    se = (metrics["ic_std"] / np.sqrt(metrics["n_periods"].clip(lower=1))).tolist()
    ax.errorbar(parents, ics, yerr=se, fmt="none", color="black",
                capsize=4, linewidth=1.2)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.axhline(IC_HELP, color=COLORS["helping"], linewidth=0.7, linestyle=":", alpha=0.6)
    ax.axhline(IC_HURT, color=COLORS["hurting"], linewidth=0.7, linestyle=":", alpha=0.6)

    ax.set_title(f"Parent factor {horizon} OOS IC — 2025 (train frozen on ≤2024)",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel(f"Mean Spearman IC ({horizon})")
    ax.set_xlabel("Parent Factor")
    ax.tick_params(axis="x", rotation=15)

    legend_patches = [
        mpatches.Patch(color=COLORS["helping"], label=f"Helping (IC > {IC_HELP:+.3f})"),
        mpatches.Patch(color=COLORS["neutral"], label="Neutral"),
        mpatches.Patch(color=COLORS["hurting"], label=f"Hurting (IC < {IC_HURT:+.3f})"),
    ]
    ax.legend(handles=legend_patches, loc="upper right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_dir / "parent_ic_bar.png", dpi=150)
    plt.close(fig)
    print("  Saved parent_ic_bar.png")


def _plot_quintile_profiles(profiles: dict[str, pd.DataFrame],
                             composite_scores: dict[str, pd.Series],
                             fwd: dict[str, dict[str, pd.Series]],
                             horizon: str, out_dir: Path) -> None:
    """3-row × 3-col grid: 8 parents + 1 composite Q1-Q5 profiles."""
    months = HORIZON_MONTHS[horizon]
    ppy = 12.0 / months

    # Composite quintile profile
    comp_profile = _single_quintile_profile(composite_scores, fwd.get(horizon, {}),
                                            months, ppy)

    fig, axes = plt.subplots(3, 3, figsize=(14, 10), sharey=False)
    axes_flat = axes.flatten()

    all_parents = PARENT_ORDER + ["composite"]
    for i, label in enumerate(all_parents):
        ax = axes_flat[i]
        if label == "composite":
            ann = comp_profile
            title = "COMPOSITE"
            color = "#1f77b4"
        else:
            df = profiles.get(label, pd.DataFrame())
            if df.empty or df["ann_return"].isna().all():
                ax.set_title(label, fontsize=9)
                ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                        ha="center", va="center", color="grey")
                continue
            ann = df["ann_return"].tolist()
            title = label
            color = "#1f77b4"

        quintiles = [f"Q{j+1}" for j in range(5)]
        bar_colors = [COLORS["hurting"], COLORS["neutral"], COLORS["neutral"],
                      COLORS["neutral"], COLORS["helping"]]
        ax.bar(quintiles, ann, color=bar_colors, edgecolor="black",
               linewidth=0.4, alpha=0.80)
        ax.axhline(0, color="black", linewidth=0.6, linestyle="--")
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_ylabel("Ann. return", fontsize=7)
        ax.tick_params(labelsize=7)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))

    # Hide unused subplot
    for j in range(len(all_parents), len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(f"Q1–Q5 {horizon} return profiles — rolling {horizon} window\n"
                 f"(Q1=lowest ranked, Q5=highest ranked; monthly rebalance)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_dir / "parent_quintile_profiles.png", dpi=150)
    plt.close(fig)
    print("  Saved parent_quintile_profiles.png")


def _single_quintile_profile(scores: dict[str, pd.Series],
                              fwd_h: dict[str, pd.Series],
                              months: int, ppy: float) -> list[float]:
    bucket_rets: list[np.ndarray] = []
    for d, fwd_r in fwd_h.items():
        sc = scores.get(d)
        if sc is None:
            continue
        df = pd.DataFrame({"s": sc, "f": fwd_r}).dropna()
        if len(df) < 25:
            continue
        try:
            buckets = pd.qcut(df["s"].rank(method="first"), 5, labels=False)
            qm = df["f"].groupby(buckets).mean().reindex(range(5))
            if qm.notna().all():
                bucket_rets.append(qm.to_numpy(dtype=float))
        except ValueError:
            pass
    if not bucket_rets:
        return [np.nan] * 5
    mat = np.array(bucket_rets)
    avg = mat.mean(axis=0)
    return ((1 + avg) ** ppy - 1).tolist()


def _plot_weight_vs_edge(metrics: pd.DataFrame, derived_weights: dict[str, float],
                          horizon: str, out_dir: Path) -> None:
    parents = [p for p in PARENT_ORDER if p in metrics["parent"].values]
    v4_w = [V4_PARENT_WEIGHTS.get(p, 0.0) for p in parents]
    dr_w = [derived_weights.get(p, 0.0) for p in parents]
    ics = []
    verdicts = []
    for p in parents:
        row = metrics[metrics["parent"] == p]
        ics.append(float(row["mean_ic"].iloc[0]) if not row.empty else np.nan)
        verdicts.append(str(row["verdict"].iloc[0]) if not row.empty else "neutral")

    x = np.arange(len(parents))
    width = 0.28

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), gridspec_kw={"height_ratios": [2, 1]})

    # Top: weight comparison
    ax1.bar(x - width / 2, v4_w, width, label="V4 prod weights (fixed)",
            color="#1f77b4", alpha=0.80, edgecolor="black", linewidth=0.4)
    ax1.bar(x + width / 2, dr_w, width, label="Derived rolling-5y weights",
            color="#ff7f0e", alpha=0.80, edgecolor="black", linewidth=0.4)
    ax1.set_xticks(x)
    ax1.set_xticklabels(parents, rotation=15)
    ax1.set_ylabel("Parent weight")
    ax1.set_title("Parent weights: V4 production vs rolling-5y derived vs in-sample IC",
                  fontsize=12, fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.grid(axis="y", alpha=0.3)

    # Bottom: in-sample IC
    ic_colors = [COLORS.get(v, COLORS["neutral"]) for v in verdicts]
    ax2.bar(x, ics, color=ic_colors, edgecolor="black", linewidth=0.4, alpha=0.85)
    ax2.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax2.axhline(IC_HELP, color=COLORS["helping"], linewidth=0.7, linestyle=":", alpha=0.6)
    ax2.axhline(IC_HURT, color=COLORS["hurting"], linewidth=0.7, linestyle=":", alpha=0.6)
    ax2.set_xticks(x)
    ax2.set_xticklabels(parents, rotation=15)
    ax2.set_ylabel(f"{horizon} in-sample IC (rolling 5y)")
    legend_patches = [
        mpatches.Patch(color=COLORS["helping"], label="Helping"),
        mpatches.Patch(color=COLORS["neutral"], label="Neutral"),
        mpatches.Patch(color=COLORS["hurting"], label="Hurting"),
    ]
    ax2.legend(handles=legend_patches, fontsize=8, loc="upper right")
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_dir / "weight_vs_edge.png", dpi=150)
    plt.close(fig)
    print("  Saved weight_vs_edge.png")


def _plot_scatter(df: pd.DataFrame, title: str, subtitle: str,
                   label_n_obs: str, out_path: Path,
                   x_label: str = "Composite Score (0–100, sector-relative percentile)",
                   show_quintile_diamonds: bool = True) -> None:
    """Composite score vs 3M return scatter, coloured H1/H2."""
    from scipy import stats as sc_stats
    sub = df[["score", "ret"]].dropna()
    if len(sub) < 10:
        print(f"  Skipping {out_path.name}: too few observations ({len(sub)}).")
        return
    x, y = sub["score"].values, sub["ret"].values
    slope, intercept, r, p, se = sc_stats.linregress(x, y)
    r_sp, p_sp = sc_stats.spearmanr(x, y)
    t_stat = slope / se if se > 0 else 0.0

    fig, ax = plt.subplots(figsize=(14, 8))
    h1 = df[df.get("half", pd.Series(["H1"] * len(df), index=df.index)) == "H1"]
    h2 = df[df.get("half", pd.Series(["H2"] * len(df), index=df.index)) == "H2"]
    ax.scatter(h1["score"], h1["ret"], alpha=0.3, s=10, color="#6baed6",
               label="H1 2025 (Jan–Jun)", zorder=2)
    ax.scatter(h2["score"], h2["ret"], alpha=0.3, s=10, color="#fc8d59",
               label="H2 2025 (Jul–Dec)", zorder=2)

    xline = np.linspace(x.min(), x.max(), 200)
    ax.plot(xline, slope * xline + intercept, "--", color="black", lw=1.5,
            label=f"OLS fit  (slope = {slope:.4f})", zorder=3)

    if show_quintile_diamonds:
        df2 = df.dropna(subset=["score", "ret"]).copy()
        df2["q"] = pd.qcut(df2["score"], 5, labels=False, duplicates="drop")
        qm = df2.groupby("q").agg(sm=("score", "mean"), rm=("ret", "mean"))
        ax.scatter(qm["sm"], qm["rm"], marker="D", s=80, color="green", zorder=5,
                   label="Quintile mean return")
        ax.plot(qm["sm"], qm["rm"], color="green", lw=1, zorder=4)

    stats_text = (
        f"n = {len(sub):,} observations\n"
        f"({label_n_obs})\n\n"
        f"OLS slope = {slope:.4f}\n"
        f"R² = {r**2:.4f}\n"
        f"Pearson r = {r:.4f}  (p = {p:.3f})\n"
        f"Spearman ρ = {r_sp:.4f}  (p = {p_sp:.3f})\n"
        f"t-stat = {t_stat:.3f}  (H₀: slope = 0)"
    )
    ax.text(0.02, 0.97, stats_text, transform=ax.transAxes, fontsize=9,
            va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85))
    ax.axhline(0, color="gray", lw=0.5, ls=":")
    ax.set_xlabel(x_label, fontsize=11)
    ax.set_ylabel("3-Month Forward Return (%)", fontsize=11)
    ax.set_title(f"{title}\n{subtitle}", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def _generate_scatter_plots(composite_scores: dict[str, pd.Series],
                              fwd_2025: dict[str, dict[str, pd.Series]],
                              out_dir: Path) -> None:
    """Full-universe and Q5 scatter plots using rolling-5y weights applied to 2025."""
    subtitle = ("2025 — Rolling-5y derived weights | "
                "Sub-factors & parent weights from 2021-2026 in-sample calibration")
    fwd_3m = fwd_2025.get("3M", {})

    # Build rows for the full-universe scatter
    full_rows = []
    for d, sc in sorted(composite_scores.items()):
        fwd_r = fwd_3m.get(d)
        if fwd_r is None or fwd_r.empty:
            continue
        half = "H1" if d <= "2025-06-30" else "H2"
        comb = pd.DataFrame({"score": sc, "ret": fwd_r * 100}).dropna()
        comb["half"] = half
        full_rows.append(comb)

    if not full_rows:
        print("  No 2025 data for scatter plots.")
        return

    full_df = pd.concat(full_rows, ignore_index=True)
    n_rebals = len(composite_scores)

    _plot_scatter(
        full_df,
        title="Full Universe: Composite Score vs 3-Month Forward Return",
        subtitle=subtitle,
        label_n_obs=f"{n_rebals} rebalances, full universe",
        out_path=out_dir / "full_universe_scatter.png",
        show_quintile_diamonds=True,
    )

    # Q5 scatter (top quintile only)
    q5_rows = []
    for d, sc in sorted(composite_scores.items()):
        fwd_r = fwd_3m.get(d)
        if fwd_r is None or fwd_r.empty:
            continue
        half = "H1" if d <= "2025-06-30" else "H2"
        df_period = pd.DataFrame({"score": sc, "ret": fwd_r * 100}).dropna()
        n = len(df_period)
        if n < 25:
            continue
        cutoff = df_period["score"].quantile(0.80)
        q5 = df_period[df_period["score"] >= cutoff].copy()
        q5["half"] = half
        q5_rows.append(q5)

    if q5_rows:
        q5_df = pd.concat(q5_rows, ignore_index=True)
        _plot_scatter(
            q5_df,
            title="Q5 Top-Quintile: Composite Score vs 3M Forward Return",
            subtitle=subtitle,
            label_n_obs=f"{n_rebals} rebalances × top-quintile",
            out_path=out_dir / "q5_scatter.png",
            show_quintile_diamonds=False,
            x_label="Composite Score (0–100, sector-relative)",
        )


def _plot_ic_comparison(metrics_5y: pd.DataFrame, metrics_2025: pd.DataFrame,
                         horizon: str, out_dir: Path) -> None:
    """Grouped bar: rolling-5y in-sample IC vs 2025-only IC, same sub-weights."""
    parents = PARENT_ORDER
    ic_5y = []
    ic_25 = []
    for p in parents:
        r5 = metrics_5y[metrics_5y["parent"] == p]
        r25 = metrics_2025[metrics_2025["parent"] == p]
        ic_5y.append(float(r5["mean_ic"].iloc[0]) if not r5.empty and r5["n_periods"].iloc[0] > 0 else np.nan)
        ic_25.append(float(r25["mean_ic"].iloc[0]) if not r25.empty and r25["n_periods"].iloc[0] > 0 else np.nan)

    x = np.arange(len(parents))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width / 2, ic_5y, width, label="Rolling 5y in-sample (2021–2026)",
           color="#1f77b4", alpha=0.82, edgecolor="black", linewidth=0.4)
    ax.bar(x + width / 2, ic_25, width, label="2025 only (same sub-weights)",
           color="#ff7f0e", alpha=0.82, edgecolor="black", linewidth=0.4)

    # SE bars for 2025 (narrower window)
    se_25 = []
    for p in parents:
        r = metrics_2025[metrics_2025["parent"] == p]
        n = int(r["n_periods"].iloc[0]) if not r.empty else 0
        se_25.append(float(r["ic_std"].iloc[0]) / np.sqrt(max(n, 1)) if not r.empty and n > 0 else 0)
    ax.errorbar(x + width / 2, ic_25, yerr=se_25, fmt="none",
                color="black", capsize=3, linewidth=1)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.axhline(IC_HELP, color=COLORS["helping"], linewidth=0.7, linestyle=":", alpha=0.6)
    ax.axhline(IC_HURT, color=COLORS["hurting"], linewidth=0.7, linestyle=":", alpha=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels(parents, rotation=15)
    ax.set_ylabel(f"Mean Spearman IC ({horizon})")
    ax.set_title(
        f"Parent {horizon} IC: rolling-5y in-sample vs 2025-only\n"
        f"(same rolling-5y derived sub-weights in both; ±1 SE bars on 2025)",
        fontsize=12, fontweight="bold"
    )
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_dir / "ic_comparison_5y_vs_2025.png", dpi=150)
    plt.close(fig)
    print("  Saved ic_comparison_5y_vs_2025.png")


# --------------------------------------------------------------------------- #
# Markdown report
# --------------------------------------------------------------------------- #

def _write_report(metrics: pd.DataFrame, metrics_2025: pd.DataFrame,
                   profiles: dict[str, pd.DataFrame],
                   derived_weights: dict[str, float],
                   sub_weights: dict[str, dict[str, float]],
                   horizon: str, rolling_years: int, window_start: str,
                   out_dir: Path) -> None:
    n_rebals = metrics["n_periods"].max() if not metrics.empty else "?"
    lines = [
        f"# Parent Factor Diagnostic — Rolling {rolling_years}y In-Sample",
        "",
        f"**Window:** {window_start} → present ({rolling_years} years, ~{n_rebals} monthly rebalances).",
        f"**Horizon analysed:** {horizon} forward returns.",
        f"**Metric:** Spearman rank IC of per-parent score vs forward returns.",
        f"**Note:** selection AND evaluation use the same window — this is an in-sample regime",
        f"read. For strict OOS see output/walkforward/.",
        "",
        "---",
        "",
        "## 1. Derived Sub-Factor Selections & Weights",
        "",
        "Sub-factors and intra-parent weights derived by the IC/IR-cap-50% greedy selection",
        f"run on the rolling {rolling_years}y window.",
        "",
        "| Parent | Sub-factor | Intra-parent weight |",
        "|--------|-----------|---------------------|",
    ]
    for parent in PARENT_ORDER:
        wmap = sub_weights.get(parent, {})
        if not wmap:
            lines.append(f"| {parent} | — | — |")
            continue
        for i, (sub, w) in enumerate(sorted(wmap.items(), key=lambda x: -x[1])):
            p_col = parent if i == 0 else ""
            lines.append(f"| {p_col} | {sub} | {w:.3f} |")

    lines += [
        "",
        "---",
        "",
        "## 2. Derived Parent Weights vs V4 Production",
        "",
        "| Parent | Derived (rolling 5y) | V4 Production (fixed) | Delta |",
        "|--------|---------------------|----------------------|-------|",
    ]
    for parent in PARENT_ORDER:
        dr = derived_weights.get(parent, 0.0)
        v4 = V4_PARENT_WEIGHTS.get(parent, 0.0)
        delta = dr - v4
        lines.append(f"| {parent} | {dr:.1%} | {v4:.1%} | {delta:+.1%} |")

    lines += [
        "",
        "---",
        "",
        f"## 3. Per-Parent In-Sample Metrics ({horizon})",
        "",
        "| Parent | n | IC | IC Std | IR | Hit | Ann Spread | Mono | Verdict |",
        "|--------|---|-----|--------|----|----|-----------|------|---------|",
    ]
    for _, r in metrics.iterrows():
        if r.get("n_periods", 0) == 0:
            lines.append(f"| {r['parent']} | 0 | — | — | — | — | — | — | no data |")
            continue
        lines.append(
            f"| {r['parent']} | {r['n_periods']:.0f} "
            f"| {r['mean_ic']:+.4f} | {r['ic_std']:.4f} "
            f"| {r['ir']:+.3f} | {r['hit_rate']:.0%} "
            f"| {r['ann_spread']:+.1%} | {r['mono_rate']:.0%} "
            f"| **{r['verdict']}** |"
        )

    lines += [
        "",
        f"_n = number of monthly {horizon} IC observations in the rolling window._",
        "_Ann Spread = annualized Q5−Q1 return._",
        "_Mono = fraction of periods where returns rise strictly Q1<Q2<Q3<Q4<Q5._",
        "",
        "---",
        "",
        f"## 4. Q1-Q5 Annualized Return Profiles ({horizon})",
        "",
        "| Parent | Q1 | Q2 | Q3 | Q4 | Q5 | Q5-Q1 |",
        "|--------|----|----|----|----|----|----|",
    ]
    for parent in PARENT_ORDER:
        df = profiles.get(parent, pd.DataFrame())
        if df.empty or df["ann_return"].isna().all():
            lines.append(f"| {parent} | — | — | — | — | — | — |")
            continue
        ann = df["ann_return"].tolist()
        spread = ann[4] - ann[0]
        lines.append(
            "| " + parent + " | "
            + " | ".join(f"{v:+.1%}" for v in ann)
            + f" | {spread:+.1%} |"
        )

    lines += [
        "",
        "---",
        "",
        "## 5. Weight vs Edge Summary",
        "",
        "| Parent | Derived weight | V4 weight | IC | Alignment |",
        "|--------|---------------|-----------|-----|-----------|",
    ]
    for parent in PARENT_ORDER:
        dr = derived_weights.get(parent, 0.0)
        v4 = V4_PARENT_WEIGHTS.get(parent, 0.0)
        row = metrics[metrics["parent"] == parent]
        ic = float(row["mean_ic"].iloc[0]) if not row.empty and row["n_periods"].iloc[0] > 0 else np.nan
        verdict = str(row["verdict"].iloc[0]) if not row.empty else "—"
        if np.isnan(ic):
            align = "—"
        elif verdict == "hurting" and max(dr, v4) >= 0.10:
            align = "OVER-WEIGHTED ⚠"
        elif verdict == "helping" and max(dr, v4) <= 0.06:
            align = "UNDER-WEIGHTED ⚠"
        elif verdict == "helping":
            align = "good"
        elif verdict == "hurting":
            align = "drag — cut weight"
        else:
            align = "neutral"
        lines.append(f"| {parent} | {dr:.1%} | {v4:.1%} | {ic:+.4f} | {align} |")

    # Comparison table: rolling-5y vs 2025-only
    lines += [
        "",
        "---",
        "",
        f"## 6. Rolling-5y vs 2025-Only IC Comparison",
        "",
        f"Both columns use the same rolling-5y derived sub-weights.",
        f"Rolling-5y is in-sample; 2025 is a narrower recent test (~12 periods, SE ≈ ±0.03).",
        "",
        f"| Parent | 5y IC | 2025 IC | Delta | Consistent? |",
        "|--------|-------|---------|-------|-------------|",
    ]
    for parent in PARENT_ORDER:
        r5 = metrics[metrics["parent"] == parent]
        r25 = metrics_2025[metrics_2025["parent"] == parent]
        ic5 = float(r5["mean_ic"].iloc[0]) if not r5.empty and r5["n_periods"].iloc[0] > 0 else np.nan
        ic25 = float(r25["mean_ic"].iloc[0]) if not r25.empty and r25["n_periods"].iloc[0] > 0 else np.nan
        if np.isnan(ic5) or np.isnan(ic25):
            lines.append(f"| {parent} | — | — | — | — |")
            continue
        delta = ic25 - ic5
        # Consistent if both positive or both negative
        consistent = "yes ✓" if (ic5 > 0) == (ic25 > 0) else "regime shift ⚠"
        # Flag large deterioration in 2025
        if ic5 > IC_HELP and ic25 < IC_HURT:
            consistent = "degraded ⚠"
        elif ic5 < IC_HURT and ic25 > IC_HELP:
            consistent = "improved ✓"
        lines.append(
            f"| {parent} | {ic5:+.4f} | {ic25:+.4f} | {delta:+.4f} | {consistent} |"
        )

    lines += ["", "---", "", "## 7. Verdicts", ""]

    helping = metrics[metrics["verdict"] == "helping"]["parent"].tolist()
    neutral = metrics[metrics["verdict"] == "neutral"]["parent"].tolist()
    hurting = metrics[metrics["verdict"] == "hurting"]["parent"].tolist()

    lines += [
        f"**Helping (IC > {IC_HELP:+.3f}):** " + (", ".join(helping) or "none"),
        "",
        f"**Neutral:** " + (", ".join(neutral) or "none"),
        "",
        f"**Hurting (IC < {IC_HURT:+.3f}):** " + (", ".join(hurting) or "none"),
        "",
    ]

    flags = []
    for parent in hurting:
        dr = derived_weights.get(parent, 0.0)
        v4 = V4_PARENT_WEIGHTS.get(parent, 0.0)
        if max(dr, v4) >= 0.10:
            ic_v = metrics[metrics["parent"] == parent]["mean_ic"].iloc[0]
            flags.append(
                f"- **{parent}** is hurting (IC={ic_v:+.4f}) but carries "
                f"{dr:.1%} derived / {v4:.1%} V4 weight — highest drag risk."
            )
        elif max(dr, v4) > 0:
            ic_v = metrics[metrics["parent"] == parent]["mean_ic"].iloc[0]
            flags.append(
                f"- **{parent}** is hurting (IC={ic_v:+.4f}) with "
                f"{dr:.1%} derived / {v4:.1%} V4 weight — removing would reduce drag."
            )
    for parent in neutral:
        dr = derived_weights.get(parent, 0.0)
        v4 = V4_PARENT_WEIGHTS.get(parent, 0.0)
        if max(dr, v4) >= 0.18:
            ic_v = metrics[metrics["parent"] == parent]["mean_ic"].iloc[0]
            flags.append(
                f"- **{parent}** is neutral (IC={ic_v:+.4f}) but "
                f"{dr:.1%} derived / {v4:.1%} V4 weight — diluting the composite."
            )
    for parent in helping:
        dr = derived_weights.get(parent, 0.0)
        v4 = V4_PARENT_WEIGHTS.get(parent, 0.0)
        if max(dr, v4) <= 0.06:
            ic_v = metrics[metrics["parent"] == parent]["mean_ic"].iloc[0]
            flags.append(
                f"- **{parent}** is helping (IC={ic_v:+.4f}) but only "
                f"{dr:.1%} derived / {v4:.1%} V4 weight — under-utilised."
            )
    if flags:
        lines += ["**Misalignments:**", ""] + flags
    else:
        lines.append("No severe weight-edge misalignments.")

    lines += [
        "",
        "---",
        "",
        "## 8. Caveats",
        "",
        f"- **In-sample:** rolling-5y selection and evaluation share the same window. "
          "Results are optimistic vs strict OOS — use output/walkforward/ for the held-out read.",
        f"- IC standard error ≈ IC_std / √n. With ~{n_rebals} rolling observations "
          "the SE is ~0.02–0.03; with 12 observations (2025 only) the SE is ~0.03–0.04.",
        "- Monotonicity is a strict per-period bar (all 5 diffs positive). "
          "Q5-Q1 spread is the more informative metric.",
        "- The derived weights here are what the model WOULD pick if recalibrated on "
          "this window; production uses the fixed V4 weights.",
        "- 2025 column uses the same rolling-5y derived sub-composition for like-for-like "
          "comparison (differs from V4 production subs used in the original 2025 OOS run).",
    ]

    (out_dir / "parent_diagnostic.md").write_text("\n".join(lines))
    print("  Saved parent_diagnostic.md")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser(description="Parent factor diagnostic — rolling-Ny in-sample")
    ap.add_argument("--horizon", default="3M", choices=["1M", "3M", "6M", "12M"])
    ap.add_argument("--rolling-years", type=int, default=5)
    ap.add_argument("--out", default=str(OUT_DIR))
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    panel = _load_panel()
    with get_db() as db:
        print("Loading price matrix …")
        matrix = dl.load_price_matrix(db, panel.universe, PANEL_START, PRICE_END)
        sectors = dl.global_sectors(db)
    print(f"Price matrix: {matrix.shape[1]} tickers × {matrix.shape[0]} dates")

    result = _run_rolling_window(panel, matrix, sectors, rolling_years=args.rolling_years)
    parent_scores = result["parent_scores"]
    parent_scores_2025 = result["parent_scores_2025"]
    composite_scores = result["composite_scores"]
    fwd = result["fwd"]
    fwd_2025 = result["fwd_2025"]
    derived_weights = result["derived_parent_weights"]
    sub_weights = result["sub_weights"]
    window_start = result["window_start"]

    print(f"\nComputing per-parent {args.horizon} metrics …")
    metrics = _compute_parent_metrics(parent_scores, fwd, horizon=args.horizon)
    profiles = _compute_quintile_profiles(parent_scores, fwd, horizon=args.horizon)

    # 2025-only metrics (same sub-weights)
    metrics_2025 = _compute_parent_metrics(parent_scores_2025, fwd_2025, horizon=args.horizon)

    # Save CSVs
    metrics.to_csv(out_dir / "parent_rolling5y.csv", index=False)
    print("  Saved parent_rolling5y.csv")

    metrics_2025.to_csv(out_dir / "parent_2025_only.csv", index=False)
    print("  Saved parent_2025_only.csv")

    rows = []
    for parent, df in profiles.items():
        for _, r in df.iterrows():
            rows.append({"parent": parent, **r.to_dict()})
    pd.DataFrame(rows).to_csv(out_dir / "parent_quintiles_3M.csv", index=False)
    print("  Saved parent_quintiles_3M.csv")

    wt_rows = []
    for parent in PARENT_ORDER:
        row5 = metrics[metrics["parent"] == parent]
        row25 = metrics_2025[metrics_2025["parent"] == parent]
        wt_rows.append({
            "parent": parent,
            "v4_prod_weight": V4_PARENT_WEIGHTS.get(parent, 0.0),
            "derived_rolling5y_weight": derived_weights.get(parent, 0.0),
            "ic_5y": float(row5["mean_ic"].iloc[0]) if not row5.empty and row5["n_periods"].iloc[0] > 0 else np.nan,
            "ic_2025": float(row25["mean_ic"].iloc[0]) if not row25.empty and row25["n_periods"].iloc[0] > 0 else np.nan,
            "verdict_5y": str(row5["verdict"].iloc[0]) if not row5.empty else "—",
            "verdict_2025": str(row25["verdict"].iloc[0]) if not row25.empty else "—",
        })
    pd.DataFrame(wt_rows).to_csv(out_dir / "weight_vs_edge.csv", index=False)
    print("  Saved weight_vs_edge.csv")

    # Sub-factor selections CSV
    sub_rows = []
    for parent in PARENT_ORDER:
        for sub, w in sorted(sub_weights.get(parent, {}).items(), key=lambda x: -x[1]):
            sub_rows.append({"parent": parent, "sub_factor": sub, "intra_parent_weight": w})
    pd.DataFrame(sub_rows).to_csv(out_dir / "derived_sub_weights.csv", index=False)
    print("  Saved derived_sub_weights.csv")

    # Comprehensive summary table CSV
    # Pull parent scorecard (IC + IR that drove the weight formula) from cfg.meta
    sc_records = result["cfg"].meta.get("parent_scorecard", [])
    sc_df = pd.DataFrame(sc_records).rename(columns={
        "mean_ic_3m6m": "scorecard_ic_3m6m",
        "information_ratio": "scorecard_ir",
    }) if sc_records else pd.DataFrame()

    summary_rows = []
    for parent in PARENT_ORDER:
        r5  = metrics[metrics["parent"] == parent]
        r25 = metrics_2025[metrics_2025["parent"] == parent]
        sc_row = sc_df[sc_df["parent"] == parent] if not sc_df.empty else pd.DataFrame()

        ic_scorecard = float(sc_row["scorecard_ic_3m6m"].iloc[0]) if not sc_row.empty and "scorecard_ic_3m6m" in sc_row.columns else np.nan
        ir_scorecard = float(sc_row["scorecard_ir"].iloc[0]) if not sc_row.empty and "scorecard_ir" in sc_row.columns else np.nan
        ic_5y  = float(r5["mean_ic"].iloc[0])  if not r5.empty  and r5["n_periods"].iloc[0]  > 0 else np.nan
        ic_25  = float(r25["mean_ic"].iloc[0]) if not r25.empty and r25["n_periods"].iloc[0] > 0 else np.nan
        summary_rows.append({
            "parent":            parent,
            "scorecard_mean_ic": ic_scorecard,
            "scorecard_ir":      ir_scorecard,
            "derived_weight":    derived_weights.get(parent, 0.0),
            "v4_prod_weight":    V4_PARENT_WEIGHTS.get(parent, 0.0),
            "ic_5y_insample":    ic_5y,
            "ic_2025_only":      ic_25,
            "delta_ic":          ic_25 - ic_5y if not (np.isnan(ic_25) or np.isnan(ic_5y)) else np.nan,
            "verdict_5y":        str(r5["verdict"].iloc[0])  if not r5.empty  else "—",
            "verdict_2025":      str(r25["verdict"].iloc[0]) if not r25.empty else "—",
        })
    summary_df = pd.DataFrame(summary_rows).sort_values("derived_weight", ascending=False)
    summary_df.to_csv(out_dir / "parent_summary.csv", index=False)
    print("  Saved parent_summary.csv")

    print("\nGenerating plots …")
    _plot_ic_bar(metrics, args.horizon, out_dir)
    _plot_quintile_profiles(profiles, composite_scores, fwd, args.horizon, out_dir)
    _plot_weight_vs_edge(metrics, derived_weights, args.horizon, out_dir)
    _plot_ic_comparison(metrics, metrics_2025, args.horizon, out_dir)
    _generate_scatter_plots(result["composite_scores"], fwd_2025, out_dir)

    _write_report(metrics, metrics_2025, profiles, derived_weights, sub_weights,
                  args.horizon, args.rolling_years, window_start, out_dir)

    print("\n=== Rolling-5y summary ===")
    for _, r in metrics.iterrows():
        if r.get("n_periods", 0) == 0:
            continue
        print(f"  {r['parent']:15s}  IC={r['mean_ic']:+.4f}  IR={r['ir']:+.3f}  "
              f"hit={r['hit_rate']:.0%}  → {r['verdict'].upper()}")

    print("\n=== 2025-only summary (same sub-weights) ===")
    for _, r in metrics_2025.iterrows():
        if r.get("n_periods", 0) == 0:
            continue
        print(f"  {r['parent']:15s}  IC={r['mean_ic']:+.4f}  IR={r['ir']:+.3f}  "
              f"hit={r['hit_rate']:.0%}  → {r['verdict'].upper()}")

    print(f"\nAll outputs saved to {out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
