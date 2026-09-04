"""Regime-study reporting: CSV/Markdown tables + PNG charts under ``output/regime_study/``.

Consumes the artefacts produced by :func:`research.walkforward.regime_study.run_regime_study`
and :func:`research.walkforward.regime_study.build_summaries`. Every table is written as
both a CSV (for downstream analysis) and a small Markdown extract in ``REGIME_REPORT.md``.

Charts:
* ``equity_curves.png`` — pooled top-20 % equity curve per policy vs SPY on the shared grid
* ``parent_weight_evolution_{policy}.png`` — stacked area of parent weights across windows
* ``parent_ic_evolution_{policy}.png`` — per-parent OOS 6M IC by window
* ``regime_divergence.png`` — |Δparent-weight| L1 turnover (5y vs 2y) and 6M IC gap
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .regime_study import (BUCKET_DEFS, POLICIES, RegimeRun, divergence_table,
                           parent_ic_matrix, parent_weight_matrix)

PALETTE = {
    "rolling5y":     "#1f77b4",
    "rolling2y":     "#d62728",
    "blend_comp":    "#2ca02c",
    "blend_parent":  "#9467bd",
    "SPY":           "#7f7f7f",
}


# --------------------------------------------------------------------------- #
# Chart helpers
# --------------------------------------------------------------------------- #
def _equity_curve(returns: pd.Series) -> pd.Series:
    r = returns.dropna()
    if r.empty:
        return pd.Series(dtype=float)
    return (1.0 + r).cumprod()


def plot_equity_curves(run: RegimeRun, out_path: Path, top_pct: float = 0.20) -> None:
    """Concatenate the per-window portfolio returns per policy → shared equity curves."""
    from backtesting.data_loader import SPY
    fig, ax = plt.subplots(figsize=(11, 5.5))
    spy_series: pd.Series | None = None
    for pol in POLICIES:
        windows = run.windows.get(pol, [])
        if not windows:
            continue
        rets = pd.concat([w.portfolios[top_pct].period_returns for w in windows
                          if top_pct in w.portfolios]).sort_index()
        rets = rets[~rets.index.duplicated(keep="first")]
        eq = _equity_curve(rets)
        if eq.empty:
            continue
        ax.plot(pd.to_datetime(eq.index), eq.values, label=pol,
                color=PALETTE[pol], linewidth=1.6)
        if spy_series is None:
            spys = pd.concat([w.portfolios[top_pct].bench_period_returns.get(
                SPY, pd.Series(dtype=float)) for w in windows]).sort_index()
            spys = spys[~spys.index.duplicated(keep="first")]
            spy_series = spys
    if spy_series is not None and not spy_series.empty:
        eq = _equity_curve(spy_series)
        ax.plot(pd.to_datetime(eq.index), eq.values, label="SPY",
                color=PALETTE["SPY"], linewidth=1.4, linestyle="--")
    ax.set_title(f"OOS equity curves — top-{int(top_pct*100)}% (monthly, semiannual walk-forward)")
    ax.set_ylabel("Growth of $1")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_parent_weight_evolution(run: RegimeRun, policy: str, out_path: Path) -> None:
    """Stacked-area of parent-weight evolution across the semiannual windows."""
    mat = parent_weight_matrix(run, policy)
    if mat.empty:
        return
    labels = list(mat.columns)
    parents = list(mat.index)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.stackplot(range(len(labels)), mat.to_numpy(dtype=float), labels=parents, alpha=0.85)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Parent weight")
    ax.set_title(f"Parent-weight evolution — {policy}")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_parent_ic_evolution(run: RegimeRun, policy: str, out_path: Path,
                             horizon: str = "6M") -> None:
    """One line per parent showing OOS IC (at ``horizon``) across the semiannual windows."""
    mat = parent_ic_matrix(run, policy, horizon=horizon)
    if mat.empty:
        return
    labels = list(mat.columns)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for parent in mat.index:
        y = mat.loc[parent].to_numpy(dtype=float)
        ax.plot(range(len(labels)), y, label=parent, marker="o", markersize=3,
                linewidth=1.2, alpha=0.85)
    ax.axhline(0, color="black", linewidth=0.7, alpha=0.6)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(f"OOS parent IC ({horizon})")
    ax.set_title(f"Parent OOS IC evolution — {policy}")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_regime_divergence(run: RegimeRun, out_path: Path) -> None:
    """Two-panel: (top) 5y-vs-2y parent-weight L1 turnover; (bottom) 6M IC gap 5y - 2y."""
    div = divergence_table(run)
    if div.empty:
        return
    labels = div["window"].tolist()
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True)
    axes[0].bar(range(len(labels)), div["parent_weight_l1"].to_numpy(dtype=float),
                color="#4c72b0", alpha=0.85)
    axes[0].set_ylabel("|Δ parent weight| (L1, 5y − 2y)")
    axes[0].set_title("Regime divergence — 5y vs 2y training window")
    axes[0].grid(True, alpha=0.25)
    axes[1].axhline(0, color="black", linewidth=0.7, alpha=0.6)
    axes[1].plot(range(len(labels)), div["ic_6M_5y"].to_numpy(dtype=float),
                 label="5y  6M IC", color=PALETTE["rolling5y"], marker="o", markersize=3)
    axes[1].plot(range(len(labels)), div["ic_6M_2y"].to_numpy(dtype=float),
                 label="2y  6M IC", color=PALETTE["rolling2y"], marker="o", markersize=3)
    axes[1].fill_between(range(len(labels)),
                         div["ic_6M_5y"].to_numpy(dtype=float),
                         div["ic_6M_2y"].to_numpy(dtype=float),
                         color="grey", alpha=0.15, label="Gap")
    axes[1].set_ylabel("OOS 6M composite IC")
    axes[1].set_xticks(range(len(labels)))
    axes[1].set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    axes[1].legend(loc="upper right", frameon=False, fontsize=8)
    axes[1].grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Markdown / CSV
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


def _md_table_full_ic(full_ic: pd.DataFrame) -> str:
    if full_ic.empty:
        return "_no data_"
    lines = ["| policy | horizon | n_periods | mean_ic | IR | hit_rate |",
             "|---|---|---:|---:|---:|---:|"]
    for _, r in full_ic.sort_values(["policy", "horizon"]).iterrows():
        lines.append(f"| {r['policy']} | {r['horizon']} | "
                     f"{int(r.get('n_periods', 0) or 0)} | "
                     f"{_fmt(r.get('mean_ic'))} | "
                     f"{_fmt(r.get('information_ratio'))} | "
                     f"{_fmt(r.get('hit_rate'), '.2%')} |")
    return "\n".join(lines)


def _md_table_full_portfolio(full_port: pd.DataFrame) -> str:
    if full_port.empty:
        return "_no data_"
    lines = ["| policy | top | n | CAGR | Sharpe | Sortino | vol | max DD | "
             "turnover | hit | SPY excess | α | β | IR |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for _, r in full_port.sort_values(["policy", "top_pct"]).iterrows():
        lines.append(
            f"| {r['policy']} | {int(r['top_pct']*100)}% | {int(r.get('n_periods', 0) or 0)} | "
            f"{_pct(r.get('cagr'))} | {_fmt(r.get('sharpe'), '.2f')} | "
            f"{_fmt(r.get('sortino'), '.2f')} | {_pct(r.get('ann_vol'))} | "
            f"{_pct(r.get('max_drawdown'))} | {_fmt(r.get('avg_turnover'), '.2f')} | "
            f"{_fmt(r.get('hit_rate'), '.2%')} | {_pct(r.get('spy_excess_cagr'))} | "
            f"{_pct(r.get('spy_alpha'))} | {_fmt(r.get('spy_beta'), '.2f')} | "
            f"{_fmt(r.get('spy_ir'), '.2f')} |")
    return "\n".join(lines)


def _md_table_bucket_portfolio(bucket_port: pd.DataFrame, top_pct: float = 0.20) -> str:
    b = bucket_port[bucket_port["top_pct"] == top_pct].copy()
    if b.empty:
        return "_no data_"
    lines = [f"Top-{int(top_pct*100)}% portfolio, per 3-year bucket:",
             "",
             "| bucket | policy | n | CAGR | Sharpe | max DD | SPY excess | α | β | IR |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for _, r in b.sort_values(["grouping", "policy"]).iterrows():
        lines.append(
            f"| {r['grouping']} | {r['policy']} | {int(r.get('n_periods', 0) or 0)} | "
            f"{_pct(r.get('cagr'))} | {_fmt(r.get('sharpe'), '.2f')} | "
            f"{_pct(r.get('max_drawdown'))} | {_pct(r.get('spy_excess_cagr'))} | "
            f"{_pct(r.get('spy_alpha'))} | {_fmt(r.get('spy_beta'), '.2f')} | "
            f"{_fmt(r.get('spy_ir'), '.2f')} |")
    return "\n".join(lines)


def _md_table_bucket_ic(bucket_ic: pd.DataFrame, horizon: str = "6M") -> str:
    b = bucket_ic[bucket_ic["horizon"] == horizon].copy()
    if b.empty:
        return "_no data_"
    lines = [f"OOS composite IC ({horizon}), per 3-year bucket (mean of window IC):",
             "",
             "| bucket | policy | n_windows | mean_ic | IR_across | hit |",
             "|---|---|---:|---:|---:|---:|"]
    for _, r in b.sort_values(["bucket", "policy"]).iterrows():
        lines.append(
            f"| {r['bucket']} | {r['policy']} | {int(r.get('n_windows', 0) or 0)} | "
            f"{_fmt(r.get('mean_ic'))} | "
            f"{_fmt(r.get('ir_across_windows'), '.2f')} | "
            f"{_fmt(r.get('hit_rate_windows'), '.2%')} |")
    return "\n".join(lines)


def _pick_recommendation(summaries: dict[str, pd.DataFrame]) -> str:
    port = summaries["full_portfolio"]
    ic = summaries["full_ic"]
    if port.empty or ic.empty:
        return "**Inconclusive** — no policies produced usable pooled results."
    p = port[port["top_pct"] == 0.20].set_index("policy")
    ic_wide = ic[ic["horizon"].isin(["3M", "6M"])].groupby("policy")["mean_ic"].mean()
    if p.empty or ic_wide.empty:
        return "**Inconclusive** — pooled slices empty at the 20% top."
    # Rank blend of Sharpe and 3M/6M mean IC.
    rank = (p["sharpe"].rank(ascending=True) + ic_wide.reindex(p.index).rank(ascending=True))
    best = rank.sort_values(ascending=False).index[0]
    sharpe_5 = p.loc["rolling5y", "sharpe"] if "rolling5y" in p.index else np.nan
    sharpe_2 = p.loc["rolling2y", "sharpe"] if "rolling2y" in p.index else np.nan
    ic_5 = ic_wide.get("rolling5y", np.nan)
    ic_2 = ic_wide.get("rolling2y", np.nan)
    blend_wins = best.startswith("blend_")
    if blend_wins:
        head = (f"**Hybrid wins:** `{best}` earns the best pooled 3M/6M mean IC "
                f"({_fmt(ic_wide[best])}) and top-20 % Sharpe ({_fmt(p.loc[best, 'sharpe'], '.2f')}) "
                f"across {int(p.loc[best, 'n_periods'])} months of OOS. That is evidence a blended "
                "training signal beats either 5-year or 2-year alone in this regime span.")
    elif best == "rolling2y":
        head = (f"**Short-window wins:** `rolling2y` produces the strongest pooled OOS read "
                f"(mean 3M/6M IC {_fmt(ic_2)}, Sharpe {_fmt(sharpe_2, '.2f')}) — evidence that "
                "factor efficacy has decayed and recent data is more predictive. Longer 5y "
                "training over-weights stale regimes here.")
    else:
        head = (f"**Long-window wins:** `rolling5y` remains the risk-adjusted best "
                f"(mean 3M/6M IC {_fmt(ic_5)}, Sharpe {_fmt(sharpe_5, '.2f')}) — the signal is "
                "stable enough that more history helps and 2-year windows over-fit recent noise.")
    return head


# --------------------------------------------------------------------------- #
# Main entry
# --------------------------------------------------------------------------- #
def write_reports(run: RegimeRun, summaries: dict[str, pd.DataFrame],
                  out_dir: Path) -> None:
    """Write all CSVs, PNGs, and REGIME_REPORT.md into ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- CSVs
    for key, df in summaries.items():
        if df is None or df.empty:
            continue
        df.to_csv(out_dir / f"{key}.csv", index=False)

    # per-policy diagnostic matrices
    for pol in POLICIES:
        pw = parent_weight_matrix(run, pol)
        if not pw.empty:
            pw.to_csv(out_dir / f"parent_weights_{pol}.csv")
        pic = parent_ic_matrix(run, pol, horizon="6M")
        if not pic.empty:
            pic.to_csv(out_dir / f"parent_ic_6M_{pol}.csv")

    div = divergence_table(run)
    if not div.empty:
        div.to_csv(out_dir / "regime_divergence.csv", index=False)

    # --- Charts
    plot_equity_curves(run, out_dir / "equity_curves.png")
    for pol in POLICIES:
        plot_parent_weight_evolution(run, pol,
                                     out_dir / f"parent_weight_evolution_{pol}.png")
        plot_parent_ic_evolution(run, pol,
                                 out_dir / f"parent_ic_evolution_{pol}.png",
                                 horizon="6M")
    plot_regime_divergence(run, out_dir / "regime_divergence.png")

    # --- Markdown
    rec = _pick_recommendation(summaries)
    md_lines: list[str] = [
        "# Semiannual Walk-Forward Regime Study",
        "",
        "6-month test windows (H1/H2) rolled forward from 2017. For each window the full "
        "Parent-Selection-V4 chain (sub-factor selection, intra-parent weights, parent "
        "construction, parent/composite weights) is rebuilt on train-only data under four "
        "training-window policies:",
        "",
        "* **rolling5y** — trailing 5-year training",
        "* **rolling2y** — trailing 2-year training",
        "* **blend_comp** — 50/50 average of the 5y and 2y composite scores on each test date",
        "* **blend_parent** — 50/50 average of the 5y and 2y frozen configs (parent + sub weights)",
        "",
        "Point-in-time universe, no look-ahead: selection forward-return windows are truncated "
        "6 months before each test-start boundary. Nothing here mutates production.",
        "",
        "> **Early-window note.** The trailing 5y and 2y windows share the same clamp at the "
        "data start (2015-06-30) until the 2y rollback actually crosses it (~2018), so the "
        "earliest handful of windows are identical across all four policies by construction. "
        "Read the 2020+ buckets for the honest 5y-vs-2y contrast.",
        "",
        f"## Verdict",
        "",
        rec,
        "",
        "## Full-sample pooled OOS IC",
        "",
        _md_table_full_ic(summaries["full_ic"]),
        "",
        "## Full-sample pooled OOS portfolios",
        "",
        _md_table_full_portfolio(summaries["full_portfolio"]),
        "",
        "## Per-bucket portfolios (top-20 %)",
        "",
        _md_table_bucket_portfolio(summaries["bucket_portfolio"], top_pct=0.20),
        "",
        "## Per-bucket composite IC (6M)",
        "",
        _md_table_bucket_ic(summaries["bucket_ic"], horizon="6M"),
        "",
        "## Files",
        "",
        "* `per_window_ic.csv` — per (policy, window, horizon) composite IC",
        "* `per_window_portfolio.csv` — per (policy, window, top_pct) portfolio metrics",
        "* `bucket_ic.csv`, `bucket_portfolio.csv`, `bucket_quantile.csv` — 3-year buckets",
        "* `full_ic.csv`, `full_portfolio.csv`, `full_quantile.csv` — pooled",
        "* `parent_weights_{policy}.csv`, `parent_ic_6M_{policy}.csv` — evolution matrices",
        "* `regime_divergence.csv` — 5y vs 2y L1 weight distance + 6M IC gap per window",
        "* `equity_curves.png` — pooled top-20 % OOS equity per policy vs SPY",
        "* `parent_weight_evolution_{policy}.png` — stacked-area evolution",
        "* `parent_ic_evolution_{policy}.png` — per-parent 6M IC over time",
        "* `regime_divergence.png` — 5y-vs-2y disagreement over time",
        "",
    ]
    (out_dir / "REGIME_REPORT.md").write_text("\n".join(md_lines))
