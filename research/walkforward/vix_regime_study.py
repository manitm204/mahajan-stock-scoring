"""VIX regime study — two analyses using the rolling-5y semiannual walk-forward.

Study 1: VIX regime vs parent factor effectiveness.
  Buckets each semiannual test window by the average VIX during the test+holding
  period (Low <15 / Medium 15-25 / High >25) and reports per-parent OOS IC,
  Q5-Q1 spread, and model weight per bucket.

Study 2: Train/test VIX mismatch vs OOS performance.
  For each window computes the VIX gap between the 5-year training period and the
  test period, then correlates that gap with excess CAGR, Sharpe, IR, composite IC.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from backtesting.data_loader import SPY
from research import HORIZON_MONTHS, compute_forward_returns
from research.ic import period_ic
from research.panel import ScorePanel

from . import analysis
from . import portfolio as pf
from .compose import FrozenConfig, build_parent_panel, frozen_composite
from .selection import select_config, slice_panel
from .splits import WalkForwardSplit, semiannual_labels, semiannual_policy_splits

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
VIX_LOW = 15.0
VIX_HIGH = 25.0
REGIME_ORDER = ["Low (<15)", "Medium (15-25)", "High (>25)"]
TOP_PCT = 0.20
IC_HORIZON = "6M"


def vix_regime_label(vix: float) -> str:
    if not np.isfinite(vix):
        return "Unknown"
    if vix < VIX_LOW:
        return REGIME_ORDER[0]
    if vix <= VIX_HIGH:
        return REGIME_ORDER[1]
    return REGIME_ORDER[2]


# --------------------------------------------------------------------------- #
# Data containers
# --------------------------------------------------------------------------- #
@dataclass
class VixWindowData:
    label: str
    split: WalkForwardSplit
    config: FrozenConfig
    # VIX
    vix_spot_at_test_start: float
    vix_avg_test: float            # avg VIX over test window
    vix_avg_test_plus_3m: float    # avg VIX over test window + 3-month hold
    vix_regime_test: str           # bucketed by vix_avg_test_plus_3m
    vix_avg_train: float           # avg VIX over 5y training window
    vix_mismatch: float            # abs(test_plus_3m - train)
    vix_direction: float           # test_plus_3m - train (positive = higher VIX in test)
    # Portfolio performance
    port_cagr: float
    port_sharpe: float
    port_sortino: float
    spy_excess_cagr: float
    spy_ir: float
    spy_rel_max_drawdown: float
    # Composite IC
    composite_ic_3m: float
    composite_ic_6m: float
    # Q5-Q1
    q5q1_spread_ann: float
    q5q1_spread_sharpe: float
    # Per-parent (keyed by parent name)
    parent_weights: dict[str, float]
    parent_ic_6m: dict[str, float]
    parent_q5q1_ann: dict[str, float]


@dataclass
class VixStudyRun:
    windows: list[VixWindowData] = field(default_factory=list)
    matrix: pd.DataFrame = field(default_factory=pd.DataFrame)
    sectors: pd.Series = field(default_factory=lambda: pd.Series(dtype=object))
    vix_series: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))


# --------------------------------------------------------------------------- #
# VIX helpers
# --------------------------------------------------------------------------- #
def load_vix_series(db) -> pd.Series:
    """Daily VIX close indexed by Timestamp."""
    df = db.query_df(
        "SELECT date, adj_close FROM daily_prices WHERE ticker='VIX' ORDER BY date"
    )
    if df.empty:
        return pd.Series(dtype=float)
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")["adj_close"].astype(float)


def _vix_spot(vix: pd.Series, date: str) -> float:
    ts = pd.Timestamp(date)
    sub = vix[vix.index <= ts]
    return float(sub.iloc[-1]) if not sub.empty else float("nan")


def _vix_avg(vix: pd.Series, start: str, end: str) -> float:
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    sub = vix[(vix.index >= s) & (vix.index <= e)]
    return float(sub.mean()) if not sub.empty else float("nan")


# --------------------------------------------------------------------------- #
# Per-parent analytics for one test window
# --------------------------------------------------------------------------- #
def _parent_ic_window(parent_scores_by_date: dict[str, dict[str, pd.Series]],
                      test_rebals: list[str],
                      fwd_by_h: dict[str, dict[str, pd.Series]]) -> dict[str, float]:
    """Mean OOS IC at IC_HORIZON for each parent across test rebalances."""
    fwd_h = fwd_by_h.get(IC_HORIZON, {})
    out: dict[str, float] = {}
    for parent, by_date in parent_scores_by_date.items():
        ics = [period_ic(by_date[d], fwd_h[d], min_names=20)
               for d in test_rebals if d in by_date and d in fwd_h]
        ics = [x for x in ics if x is not None]
        out[parent] = float(np.mean(ics)) if ics else float("nan")
    return out


def _parent_q5q1_window(parent_scores_by_date: dict[str, dict[str, pd.Series]],
                        test_rebals: list[str],
                        fwd_by_h: dict[str, dict[str, pd.Series]],
                        matrix: pd.DataFrame) -> dict[str, float]:
    """Annualised Q5-Q1 spread at IC_HORIZON for each parent in one test window."""
    out: dict[str, float] = {}
    for parent, by_date in parent_scores_by_date.items():
        scores = {d: by_date[d] for d in test_rebals if d in by_date}
        if not scores:
            out[parent] = float("nan")
            continue
        qres = analysis.quantile_analysis(scores, fwd_by_h)
        q = qres.get(IC_HORIZON, {})
        out[parent] = float(q.get("spread", {}).get("annualized", float("nan")))
    return out


# --------------------------------------------------------------------------- #
# Main runner
# --------------------------------------------------------------------------- #
def run_vix_study(panel: ScorePanel, matrix: pd.DataFrame, sectors: pd.Series,
                  vix_series: pd.Series, *, first_test_year: int = 2017,
                  last_end: str | None = None, verbose: bool = True) -> VixStudyRun:
    """Rolling-5y semiannual walk-forward with per-window VIX and parent analytics."""
    kwargs: dict = {"first_test_year": first_test_year}
    if last_end is not None:
        kwargs["last_end"] = last_end

    splits = semiannual_policy_splits("rolling5y", **kwargs)
    run = VixStudyRun(matrix=matrix, sectors=sectors, vix_series=vix_series)

    for sp in splits:
        train_rebals = sp.train_rebalances(panel.rebal_dates)
        test_rebals = sp.test_rebalances(panel.rebal_dates)
        if not train_rebals or not test_rebals:
            if verbose:
                print(f"  skip {sp.label}: train={len(train_rebals)} test={len(test_rebals)}")
            continue
        if verbose:
            print(f"  {sp.label}: select on {len(train_rebals)} "
                  f"({train_rebals[0]}→{train_rebals[-1]}), "
                  f"test {len(test_rebals)} ({test_rebals[0]}→{test_rebals[-1]})")

        cfg = select_config(panel, train_rebals, matrix, boundary=sp.test_start)
        scores = frozen_composite(panel, test_rebals, cfg, sectors)

        # Parent panel for per-parent analytics
        sub_p = slice_panel(panel, test_rebals)
        parent_panel = build_parent_panel(sub_p, cfg.sub_weights)
        parent_scores_by_date: dict[str, dict[str, pd.Series]] = {}
        for parent in parent_panel.parent_keys:
            pbucket: dict[str, pd.Series] = {}
            for d in parent_panel.rebal_dates:
                col = parent_panel.scores[d].get(parent)
                if col is not None:
                    pbucket[d] = col
            parent_scores_by_date[parent] = pbucket

        # Forward returns for test period
        fwd = compute_forward_returns(matrix, list(scores), HORIZON_MONTHS)

        # Composite IC
        ic_df = analysis.composite_ic(scores, fwd)
        def _g_ic(h: str) -> float:
            r = ic_df[ic_df["horizon"] == h]
            return float(r.iloc[0]["mean_ic"]) if not r.empty else float("nan")

        # Q5-Q1 spread (composite)
        qres = analysis.quantile_analysis(scores, fwd)
        q6 = qres.get(IC_HORIZON, {})
        spr = q6.get("spread") or {}

        # Portfolio metrics
        book = pf.simulate(scores, matrix, sectors, top_pct=TOP_PCT,
                           mode="equal", hold_months=1)
        m = book.metrics

        # Per-parent IC and Q5-Q1
        p_ic = _parent_ic_window(parent_scores_by_date, test_rebals, fwd)
        p_q5q1 = _parent_q5q1_window(parent_scores_by_date, test_rebals, fwd, matrix)

        # VIX stats
        test_end_plus_3m = (pd.Timestamp(sp.test_end)
                            + pd.DateOffset(months=3)).date().isoformat()
        vix_spot_at = _vix_spot(vix_series, sp.test_start)
        vix_avg_t = _vix_avg(vix_series, sp.test_start, sp.test_end)
        vix_avg_t3 = _vix_avg(vix_series, sp.test_start, test_end_plus_3m)
        vix_avg_train = _vix_avg(vix_series, sp.train_start, sp.train_end)
        mismatch = abs(vix_avg_t3 - vix_avg_train)
        direction = vix_avg_t3 - vix_avg_train

        label_root = sp.label.split(":", 1)[1]
        run.windows.append(VixWindowData(
            label=label_root, split=sp, config=cfg,
            vix_spot_at_test_start=vix_spot_at,
            vix_avg_test=vix_avg_t,
            vix_avg_test_plus_3m=vix_avg_t3,
            vix_regime_test=vix_regime_label(vix_avg_t3),
            vix_avg_train=vix_avg_train,
            vix_mismatch=mismatch,
            vix_direction=direction,
            port_cagr=float(m.get("cagr", float("nan"))),
            port_sharpe=float(m.get("sharpe", float("nan"))),
            port_sortino=float(m.get("sortino", float("nan"))),
            spy_excess_cagr=float(m.get("spy_excess_cagr", float("nan"))),
            spy_ir=float(m.get("spy_ir", float("nan"))),
            spy_rel_max_drawdown=float(m.get("spy_rel_max_drawdown", float("nan"))),
            composite_ic_3m=_g_ic("3M"),
            composite_ic_6m=_g_ic(IC_HORIZON),
            q5q1_spread_ann=float(spr.get("annualized", float("nan"))),
            q5q1_spread_sharpe=float(spr.get("sharpe", float("nan"))),
            parent_weights=dict(cfg.parent_weights),
            parent_ic_6m=p_ic,
            parent_q5q1_ann=p_q5q1,
        ))
    return run


# --------------------------------------------------------------------------- #
# Study 1 — regime aggregation tables
# --------------------------------------------------------------------------- #
def study1_regime_tables(run: VixStudyRun) -> dict[str, pd.DataFrame]:
    """Three heatmap-ready tables: parent IC / Q5-Q1 / weight by VIX regime.

    Returns DataFrames indexed by parent, columns = VIX regime labels, values = means.
    Also returns per-window raw data and correlation tables.
    """
    parents = sorted({p for w in run.windows for p in w.parent_ic_6m})

    regime_rows: list[dict] = []
    for w in run.windows:
        row = {"window": w.label, "vix_avg_test_plus_3m": w.vix_avg_test_plus_3m,
               "vix_avg_test": w.vix_avg_test, "vix_spot": w.vix_spot_at_test_start,
               "regime": w.vix_regime_test, "composite_ic_6m": w.composite_ic_6m,
               "q5q1_spread_ann": w.q5q1_spread_ann}
        for p in parents:
            row[f"ic_{p}"] = w.parent_ic_6m.get(p, float("nan"))
            row[f"q5q1_{p}"] = w.parent_q5q1_ann.get(p, float("nan"))
            row[f"w_{p}"] = w.parent_weights.get(p, 0.0)
        regime_rows.append(row)
    per_window_df = pd.DataFrame(regime_rows)

    def _agg_heatmap(col_prefix: str) -> pd.DataFrame:
        """Aggregate by regime → DataFrame[parents × regimes]."""
        rows: dict[str, dict] = {p: {} for p in parents}
        for regime in REGIME_ORDER:
            sub = per_window_df[per_window_df["regime"] == regime]
            for p in parents:
                col = f"{col_prefix}{p}"
                if col in sub.columns:
                    vals = sub[col].dropna()
                    rows[p][regime] = float(vals.mean()) if not vals.empty else float("nan")
        df = pd.DataFrame(rows).T
        df.index.name = "parent"
        return df[[r for r in REGIME_ORDER if r in df.columns]]

    ic_heatmap = _agg_heatmap("ic_")
    q5q1_heatmap = _agg_heatmap("q5q1_")
    weight_heatmap = _agg_heatmap("w_")

    # Correlation: VIX vs per-parent IC and weight (per-window series)
    corr_rows: list[dict] = []
    for p in parents:
        ic_col = f"ic_{p}"
        w_col = f"w_{p}"
        vix_col = "vix_avg_test_plus_3m"
        if ic_col in per_window_df.columns and vix_col in per_window_df.columns:
            sub = per_window_df[[vix_col, ic_col, w_col]].dropna()
            corr_ic = float(sub[vix_col].corr(sub[ic_col], method="spearman")) \
                if len(sub) > 2 else float("nan")
            corr_w = float(sub[vix_col].corr(sub[w_col], method="spearman")) \
                if len(sub) > 2 else float("nan")
        else:
            corr_ic = corr_w = float("nan")
        corr_rows.append({"parent": p, "spearman_vix_vs_ic": corr_ic,
                          "spearman_vix_vs_weight": corr_w})

    return {
        "per_window": per_window_df,
        "ic_heatmap": ic_heatmap,
        "q5q1_heatmap": q5q1_heatmap,
        "weight_heatmap": weight_heatmap,
        "correlations": pd.DataFrame(corr_rows),
    }


# --------------------------------------------------------------------------- #
# Study 2 — train/test VIX mismatch table
# --------------------------------------------------------------------------- #
PERF_COLS = {
    "spy_excess_cagr": "Excess CAGR vs SPY",
    "port_sharpe": "Portfolio Sharpe",
    "port_sortino": "Portfolio Sortino",
    "spy_ir": "Information Ratio vs SPY",
    "spy_rel_max_drawdown": "Relative Max Drawdown vs SPY",
    "composite_ic_6m": "Composite IC (6M)",
    "q5q1_spread_ann": "Q5-Q1 Spread Annualised",
}


def study2_mismatch_table(run: VixStudyRun) -> pd.DataFrame:
    """Per-window mismatch stats and performance metrics."""
    rows: list[dict] = []
    for w in run.windows:
        rows.append({
            "window": w.label,
            "vix_avg_train": w.vix_avg_train,
            "vix_avg_test_plus_3m": w.vix_avg_test_plus_3m,
            "vix_mismatch": w.vix_mismatch,
            "vix_direction": w.vix_direction,
            "train_regime": vix_regime_label(w.vix_avg_train),
            "test_regime": w.vix_regime_test,
            **{k: getattr(w, k) for k in PERF_COLS},
        })
    return pd.DataFrame(rows)


def study2_correlations(mismatch_df: pd.DataFrame) -> pd.DataFrame:
    """Spearman correlations between VIX mismatch/direction and performance metrics."""
    rows: list[dict] = []
    for col, label in PERF_COLS.items():
        sub = mismatch_df[["vix_mismatch", "vix_direction", col]].dropna()
        n = len(sub)
        corr_dist = float(sub["vix_mismatch"].corr(sub[col], method="spearman")) \
            if n > 2 else float("nan")
        corr_dir = float(sub["vix_direction"].corr(sub[col], method="spearman")) \
            if n > 2 else float("nan")
        rows.append({"metric": label, "column": col, "n_windows": n,
                     "spearman_mismatch_vs_perf": corr_dist,
                     "spearman_direction_vs_perf": corr_dir})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Plotting helpers
# --------------------------------------------------------------------------- #
def _make_heatmap(df: pd.DataFrame, title: str, cmap: str, fmt: str,
                  out_path: Path) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(max(6, len(df.columns) * 2),
                                    max(4, len(df.index) * 0.6 + 1.5)))
    data = df.values.astype(float)
    vabs = np.nanmax(np.abs(data[np.isfinite(data)])) if np.any(np.isfinite(data)) else 1.0
    if cmap == "RdYlGn":
        norm = mcolors.TwoSlopeNorm(vmin=-vabs, vcenter=0, vmax=vabs)
    else:
        norm = None
    im = ax.imshow(data, cmap=cmap, aspect="auto",
                   norm=norm if norm else None,
                   vmin=None if norm else 0,
                   vmax=None if norm else vabs)
    ax.set_xticks(range(len(df.columns)))
    ax.set_xticklabels(df.columns, fontsize=10)
    ax.set_yticks(range(len(df.index)))
    ax.set_yticklabels(df.index, fontsize=9)
    for i in range(len(df.index)):
        for j in range(len(df.columns)):
            v = data[i, j]
            txt = f"{v:{fmt}}" if np.isfinite(v) else "—"
            ax.text(j, i, txt, ha="center", va="center", fontsize=9,
                    color="black" if abs(v) < 0.7 * vabs else "white"
                    if np.isfinite(v) else "gray")
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title(title, fontsize=12, pad=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_regime_heatmaps(tables: dict[str, pd.DataFrame], out_dir: Path) -> None:
    _make_heatmap(tables["ic_heatmap"],
                  "Parent OOS IC (6M) by VIX Regime",
                  "RdYlGn", ".3f",
                  out_dir / "heatmap_parent_ic_by_vix_regime.png")
    _make_heatmap(tables["q5q1_heatmap"],
                  "Parent Q5-Q1 Spread (6M, annualised) by VIX Regime",
                  "RdYlGn", ".2%",
                  out_dir / "heatmap_parent_q5q1_by_vix_regime.png")
    _make_heatmap(tables["weight_heatmap"],
                  "Parent Model Weight by VIX Regime (frozen at training)",
                  "Blues", ".2%",
                  out_dir / "heatmap_parent_weight_by_vix_regime.png")


def plot_mismatch_scatterplots(mismatch_df: pd.DataFrame, out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    metrics = [
        ("spy_excess_cagr", "Excess CAGR vs SPY"),
        ("port_sharpe", "Portfolio Sharpe"),
        ("spy_ir", "Information Ratio vs SPY"),
        ("composite_ic_6m", "Composite IC (6M)"),
        ("q5q1_spread_ann", "Q5-Q1 Spread (annualised)"),
    ]
    n = len(metrics)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes_flat = axes.flatten()
    for ax_i, (col, label) in enumerate(metrics):
        ax = axes_flat[ax_i]
        sub = mismatch_df[["vix_mismatch", "vix_direction", "window",
                            "train_regime", "test_regime", col]].dropna()
        if sub.empty:
            ax.set_visible(False)
            continue
        colors = {"Low (<15)": "steelblue", "Medium (15-25)": "gold", "High (>25)": "crimson"}
        for regime, grp in sub.groupby("test_regime"):
            c = colors.get(regime, "gray")
            ax.scatter(grp["vix_mismatch"], grp[col], c=c, label=regime, alpha=0.75, s=50)
        corr = float(sub["vix_mismatch"].corr(sub[col], method="spearman"))
        ax.set_xlabel("VIX Mismatch |test − train|", fontsize=9)
        ax.set_ylabel(label, fontsize=9)
        ax.set_title(f"{label}\n(ρ={corr:+.2f})", fontsize=9)
        ax.legend(fontsize=7, title="Test Regime")
    for ax_i in range(n, len(axes_flat)):
        axes_flat[ax_i].set_visible(False)
    fig.suptitle("Study 2: Train/Test VIX Mismatch vs OOS Performance", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "scatterplot_vix_mismatch_vs_performance.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_vix_direction_scatterplots(mismatch_df: pd.DataFrame, out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    metrics = [
        ("spy_excess_cagr", "Excess CAGR vs SPY"),
        ("port_sharpe", "Portfolio Sharpe"),
        ("composite_ic_6m", "Composite IC (6M)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (col, label) in zip(axes, metrics):
        sub = mismatch_df[["vix_direction", col, "test_regime"]].dropna()
        if sub.empty:
            ax.set_visible(False)
            continue
        colors = {"Low (<15)": "steelblue", "Medium (15-25)": "gold", "High (>25)": "crimson"}
        for regime, grp in sub.groupby("test_regime"):
            ax.scatter(grp["vix_direction"], grp[col],
                       c=colors.get(regime, "gray"), label=regime, alpha=0.75, s=50)
        corr = float(sub["vix_direction"].corr(sub[col], method="spearman"))
        ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_xlabel("VIX Direction (test − train)\n← calmer test | more volatile test →", fontsize=9)
        ax.set_ylabel(label, fontsize=9)
        ax.set_title(f"{label}\n(ρ={corr:+.2f})", fontsize=9)
        ax.legend(fontsize=7, title="Test Regime")
    fig.suptitle("Study 2: VIX Direction vs OOS Performance", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "scatterplot_vix_direction_vs_performance.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)
