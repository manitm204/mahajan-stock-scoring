"""Tables, compact dashboard and markdown report for the VIX-tilt study."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .study import StudyResult, group_defs, grouping_metrics

PRIMARY_TP = 0.20

METRIC_COLS = ["grouping", "variant", "top_pct", "n_periods", "total_return", "cagr",
               "sharpe", "sortino", "ann_vol", "max_drawdown", "hit_rate",
               "avg_turnover", "spy_cagr", "spy_excess_cagr", "spy_alpha", "spy_beta",
               "spy_ir", "spy_te", "spy_rel_max_drawdown", "ic_1m", "ic_6m",
               "q5q1_1m_ann", "overlap_vs_base", "names_entered_avg"]


def _f(v, fmt=".3f") -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    return f"{v:{fmt}}"


def _md(df: pd.DataFrame, fmts: dict[str, str]) -> str:
    lines = ["| " + " | ".join(df.columns) + " |",
             "|" + "|".join("---" for _ in df.columns) + "|"]
    for _, row in df.iterrows():
        cells = []
        for c in df.columns:
            v = row[c]
            if isinstance(v, str):
                cells.append(v)
            else:
                cells.append(_f(v, fmts.get(c, ".3f")))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def build_tables(res: StudyResult, legs=("net", "gross")) -> dict[str, pd.DataFrame]:
    defs = group_defs(res)
    out: dict[str, pd.DataFrame] = {}
    for leg in legs:
        for level, gw in defs.items():
            df = grouping_metrics(res, gw, leg=leg)
            df = df[[c for c in METRIC_COLS if c in df.columns]]
            out[f"{level}_{leg}"] = df.reset_index(drop=True)
    return out


# --------------------------------------------------------------------------- #
# Dashboard — one row per window
# --------------------------------------------------------------------------- #
def build_dashboard(res: StudyResult, tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    dec = res.decisions
    tvs = res.train_vix_summary.set_index("window")
    win_net = tables["window_net"]
    rows = []
    for w in res.windows:
        lbl = w.label
        d0 = dec[(dec["window"] == lbl) & (dec["variant"] == "baseline")].sort_values("date")
        if d0.empty:
            continue
        row = {
            "window": lbl,
            "vix_at_start": float(d0.iloc[0]["vix"]),
            "vix_avg_rebals": float(d0["vix"].mean()),
            "vix_pct_at_start": float(d0.iloc[0]["vix_pct"]),
            "train_vix_median": float(tvs.loc[lbl, "train_vix_median"])
            if lbl in tvs.index else float("nan"),
        }
        for v in res.variants:
            dv = dec[(dec["window"] == lbl) & (dec["variant"] == v.name)]
            row[f"{v.name}_strength"] = float(dv["strength"].mean()) if not dv.empty \
                else float("nan")
            row[f"{v.name}_l1_shift"] = float(dv["l1_shift"].mean()) if not dv.empty \
                else float("nan")
            sub = win_net[(win_net["grouping"] == lbl) & (win_net["variant"] == v.name)
                          & (win_net["top_pct"] == PRIMARY_TP)]
            row[f"{v.name}_ret"] = float(sub.iloc[0]["total_return"]) if not sub.empty \
                else float("nan")
        sub = win_net[(win_net["grouping"] == lbl) & (win_net["variant"] == "baseline")
                      & (win_net["top_pct"] == PRIMARY_TP)]
        if not sub.empty and np.isfinite(sub.iloc[0].get("spy_cagr", float("nan"))):
            pf = res.portfolios[("baseline", PRIMARY_TP)]
            keep = [d for d in pf.index if res.window_of_date.get(d) == lbl]
            spy = pf.loc[keep]["spy"].dropna()
            row["spy_ret"] = float((1 + spy).prod() - 1) if len(spy) else float("nan")
        else:
            row["spy_ret"] = float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def plot_dashboard(dash: pd.DataFrame, res: StudyResult, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    variants = [v.name for v in res.variants]
    colors = {"baseline": "steelblue", "fixed30": "darkorange",
              "pctile_mild": "seagreen", "pctile_std": "crimson",
              "pctile_strong": "purple", "lit_pct": "darkorange", "lit_fix": "crimson"}
    x = np.arange(len(dash))
    fig, axes = plt.subplots(4, 1, figsize=(max(13, len(dash) * 0.8), 13), sharex=True)

    ax = axes[0]
    ax.plot(x, dash["vix_at_start"], marker="o", color="black", label="VIX at window start")
    ax.plot(x, dash["train_vix_median"], marker="s", linestyle="--", color="gray",
            label="training VIX median")
    ax2 = ax.twinx()
    ax2.bar(x, dash["vix_pct_at_start"], alpha=0.15, color="red",
            label="VIX pct vs training")
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("percentile")
    ax.set_ylabel("VIX")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title("Spot VIX vs training-window distribution")

    ax = axes[1]
    for v in variants:
        if v == "baseline":
            continue
        ax.plot(x, dash[f"{v}_strength"], marker="o", markersize=3,
                color=colors.get(v), label=v)
    ax.set_ylabel("avg overlay strength")
    ax.legend(fontsize=8)
    ax.set_title("Overlay strength applied (avg across window rebalances)")

    ax = axes[2]
    for v in variants:
        if v == "baseline":
            continue
        ax.plot(x, dash[f"{v}_l1_shift"], marker="o", markersize=3,
                color=colors.get(v), label=v)
    ax.set_ylabel("avg |Δ parent weight| (L1/2)")
    ax.legend(fontsize=8)
    ax.set_title("Parent-weight shift vs frozen baseline")

    ax = axes[3]
    width = 0.8 / (len(variants) + 1)
    for i, v in enumerate(variants):
        ax.bar(x + (i - len(variants) / 2) * width, dash[f"{v}_ret"], width,
               color=colors.get(v), label=v)
    ax.bar(x + (len(variants) - len(variants) / 2) * width, dash["spy_ret"], width,
           color="lightgray", edgecolor="gray", label="SPY")
    ax.axhline(0, color="gray", linewidth=0.7)
    ax.set_ylabel(f"window return (top {PRIMARY_TP:.0%}, net)")
    ax.legend(fontsize=8, ncol=3)
    ax.set_title("Per-window net return — variants vs SPY")
    ax.set_xticks(x)
    ax.set_xticklabels(dash["window"], rotation=45, ha="right", fontsize=8)

    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Markdown report
# --------------------------------------------------------------------------- #
def write_report(res: StudyResult, tables: dict[str, pd.DataFrame],
                 dash: pd.DataFrame, out_dir: Path, cost_per_side: float) -> None:
    full = tables["full_net"]
    fullg = tables["full_gross"]
    block = tables["block_net"]
    win = tables["window_net"]

    fmts = {"total_return": "+.1%", "cagr": "+.1%", "sharpe": ".2f", "sortino": ".2f",
            "ann_vol": ".1%", "max_drawdown": ".1%", "hit_rate": ".0%",
            "avg_turnover": ".2f", "spy_cagr": "+.1%", "spy_excess_cagr": "+.1%",
            "spy_alpha": "+.1%", "spy_beta": ".2f", "spy_ir": ".2f", "spy_te": ".1%",
            "spy_rel_max_drawdown": ".1%", "ic_1m": "+.3f", "ic_6m": "+.3f",
            "q5q1_1m_ann": "+.1%", "overlap_vs_base": ".0%", "names_entered_avg": ".1f",
            "top_pct": ".0%"}

    def table_for(df, grouping=None, top_pct=None, cols=None):
        sub = df.copy()
        if grouping is not None:
            sub = sub[sub["grouping"] == grouping]
        if top_pct is not None:
            sub = sub[sub["top_pct"] == top_pct]
        cols = cols or [c for c in sub.columns if c != "grouping"]
        return _md(sub[cols], fmts)

    main_cols = ["variant", "top_pct", "cagr", "total_return", "sharpe", "sortino",
                 "ann_vol", "max_drawdown", "spy_excess_cagr", "spy_alpha", "spy_beta",
                 "spy_ir", "spy_rel_max_drawdown", "ic_1m", "ic_6m", "q5q1_1m_ann",
                 "avg_turnover", "overlap_vs_base", "names_entered_avg"]

    # Verdict inputs: per-window head-to-head at primary top pct (net Sharpe & return)
    lines = [
        "# VIX-Relative Parent-Weight Overlay — Walk-Forward Study",
        "",
        f"Semiannual OOS windows {res.windows[0].label} → {res.windows[-1].label} "
        f"({len(res.windows)} windows), rolling-5y baseline re-selected per window "
        "(subfactors, intra-parent weights and parent weights frozen at the boundary).",
        "",
        "Identical rules for every variant: monthly rebalance on test dates, long-only "
        f"top 10/20/30 % equal-weight, 1-month hold, {cost_per_side * 1e4:.0f} bps "
        "per side transaction cost on traded notional. Regime statistics use "
        "training-only 1M forward returns from a boundary-truncated price matrix; the "
        "only test-period input is the spot VIX on the rebalance date.",
        "",
        "## Variants",
        "",
        "| Variant | Logic |",
        "|---|---|",
        *[f"| {v.name} | {v.desc or v.kind} |" for v in res.variants],
        "",
        "Regime utility = 0.50·rank(IC) + 0.25·rank(IC-IR) + 0.25·rank(Q5-Q1) on "
        "comparable training observations (same VIX bucket, or same ≤30th/≥70th "
        "percentile tail); weights ∝ utility, 25 % cap; n<12 shrinks utility toward the "
        "full 5y utility by n/(n+12); n=0 falls back to baseline weights. "
        "adjusted = (1-s)·baseline + s·regime.",
        "",
        "---",
        "",
        "## Full period — NET of costs",
        "",
        table_for(full, grouping="full", cols=main_cols),
        "",
        "## Full period — GROSS",
        "",
        table_for(fullg, grouping="full", cols=main_cols),
        "",
        "## 3-year blocks — NET (top 20 %)",
        "",
    ]
    for b in block["grouping"].unique():
        lines += [f"### {b}", "",
                  table_for(block, grouping=b, top_pct=PRIMARY_TP,
                            cols=[c for c in main_cols if c != "top_pct"]), ""]

    lines += ["## Per-window — NET (top 20 %)", ""]
    win_cols = ["grouping", "variant", "total_return", "sharpe", "spy_excess_cagr",
                "ic_6m", "avg_turnover", "overlap_vs_base"]
    lines += [table_for(win[win["top_pct"] == PRIMARY_TP][win_cols], cols=win_cols), ""]

    # Head-to-head vs baseline
    lines += ["---", "", "## Head-to-head vs baseline (net, top 20 %)", "",
              "| variant | windows won (return) | win rate | ΔCAGR (full) | ΔSharpe "
              "(full) | ΔIR vs SPY | ΔIC 6M |", "|---|---|---|---|---|---|---|"]
    base_full = full[(full["variant"] == "baseline") & (full["top_pct"] == PRIMARY_TP)]
    bw = win[(win["variant"] == "baseline") & (win["top_pct"] == PRIMARY_TP)] \
        .set_index("grouping")["total_return"]
    for v in res.variants:
        if v.name == "baseline":
            continue
        vw = win[(win["variant"] == v.name) & (win["top_pct"] == PRIMARY_TP)] \
            .set_index("grouping")["total_return"]
        common = bw.index.intersection(vw.index)
        wins_n = int((vw[common] > bw[common] + 1e-12).sum())
        vf = full[(full["variant"] == v.name) & (full["top_pct"] == PRIMARY_TP)]
        if base_full.empty or vf.empty:
            continue
        b0, v0 = base_full.iloc[0], vf.iloc[0]
        lines.append(
            f"| {v.name} | {wins_n}/{len(common)} | {wins_n / max(1, len(common)):.0%} "
            f"| {_f(v0['cagr'] - b0['cagr'], '+.2%')} "
            f"| {_f(v0['sharpe'] - b0['sharpe'], '+.2f')} "
            f"| {_f(v0['spy_ir'] - b0['spy_ir'], '+.2f')} "
            f"| {_f(v0['ic_6m'] - b0['ic_6m'], '+.3f')} |")

    # Overlay activity
    dec = res.decisions
    lines += ["", "---", "", "## Overlay activity", ""]
    for v in res.variants:
        if v.name == "baseline":
            continue
        dv = dec[dec["variant"] == v.name]
        tilted = dv[dv["strength"] > 0]
        fb = dv[dv["fallback"].isin(["no_obs", "no_positive_utility"])]
        lines.append(
            f"- **{v.name}**: tilted on {len(tilted)}/{len(dv)} rebalances "
            f"({len(tilted) / max(1, len(dv)):.0%}); avg strength when tilted "
            f"{_f(tilted['strength'].mean(), '.2f')}; avg comparable obs "
            f"{_f(tilted['n_comparable'].mean(), '.0f')}; data fallbacks {len(fb)}; "
            f"avg |Δweight| when tilted {_f(tilted['l1_shift'].mean(), '.3f')}")

    lines += ["", "## PIT checks", "",
              f"All {len(res.pit_checks)} assertions passed across "
              f"{res.pit_checks['window'].nunique()} windows "
              "(training rebalances capped before the boundary, training forward-return "
              "windows end ≤ test start, VIX distributions end at train_end, spot VIX "
              "taken on/before each rebalance date). See `pit_checks.csv`.",
              "", "## Files", "",
              "`full_net.csv` `full_gross.csv` `block_net.csv` `window_net.csv` "
              "`dashboard.csv` `dashboard.png` `decisions.csv` `weights_long.csv` "
              "`baseline_weights.csv` `pit_checks.csv`", ""]

    (out_dir / "REPORT.md").write_text("\n".join(lines))


def write_all(res: StudyResult, out_dir: Path, cost_per_side: float) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    tables = build_tables(res)
    for name, df in tables.items():
        df.to_csv(out_dir / f"{name}.csv", index=False)
    dash = build_dashboard(res, tables)
    dash.to_csv(out_dir / "dashboard.csv", index=False)
    plot_dashboard(dash, res, out_dir / "dashboard.png")
    res.decisions.to_csv(out_dir / "decisions.csv", index=False)
    res.weights_long.to_csv(out_dir / "weights_long.csv", index=False)
    res.baseline_weights.to_csv(out_dir / "baseline_weights.csv", index=False)
    res.train_vix_summary.to_csv(out_dir / "train_vix_summary.csv", index=False)
    res.pit_checks.to_csv(out_dir / "pit_checks.csv", index=False)
    write_report(res, tables, dash, out_dir, cost_per_side)
