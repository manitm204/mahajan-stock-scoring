"""Factor Persistence & Regime Analysis.

For each calendar year computes IC (1M/3M/6M), IC IR, Q5-Q1 spread, hit rate,
coverage, and selection frequency / assigned weight for every subfactor and
parent.  Also records annual VIX stats and buckets windows into Low / Medium /
High VIX regimes.

Outputs (all written to ``output/factor_persistence/``):
    subfactor_ic_by_year.csv           — annual IC at 1M/3M/6M per subfactor
    subfactor_persistence_table.csv    — summary stats + stability flag
    parent_ic_by_year.csv              — annual IC at 1M/3M/6M per parent
    parent_stats_by_year.csv           — IC + IR + spread + weight per year
    parent_persistence_table.csv       — pooled summary + stability flag
    regime_subfactor_ic.csv            — mean IC by VIX regime per subfactor
    regime_parent_ic.csv               — mean IC by VIX regime per parent
    regime_parent_spread.csv           — mean spread by VIX regime per parent
    stability_ranking_subfactors.csv   — ranked by persistence
    stability_ranking_parents.csv      — ranked by persistence
    heatmap_subfactor_ic.png           — subfactor × year IC heatmap
    heatmap_parent_ic.png              — parent × year IC heatmap
    heatmap_parent_weight.png          — parent × year weight heatmap
    heatmap_parent_spread.png          — parent × year Q5-Q1 spread heatmap
    regime_subfactor_ic_heatmap.png    — subfactor × VIX-regime IC heatmap
    regime_parent_ic_heatmap.png       — parent × VIX-regime IC heatmap
    regime_parent_spread_heatmap.png   — parent × VIX-regime spread heatmap
    FACTOR_PERSISTENCE_REPORT.md       — narrative synthesis
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

from research import HORIZON_MONTHS, compute_forward_returns
from research.ic import period_ic
from research.panel import ScorePanel
from research.walkforward.compose import build_parent_panel
from research.walkforward.selection import select_config, slice_panel

VIX_LOW = 15.0
VIX_HIGH = 25.0
REGIME_ORDER = ["Low (<15)", "Medium (15-25)", "High (>25)"]
MIN_NAMES = 25
N_QUANTILES = 5
HORIZONS = ["1M", "3M", "6M"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _vix_stats(vix: pd.Series, start: str, end: str) -> dict:
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    sub = vix[(vix.index >= s) & (vix.index <= e)]
    if sub.empty:
        return {"vix_avg": float("nan"), "vix_median": float("nan"), "vix_regime": "Unknown"}
    avg = float(sub.mean())
    med = float(sub.median())
    if avg < VIX_LOW:
        regime = REGIME_ORDER[0]
    elif avg <= VIX_HIGH:
        regime = REGIME_ORDER[1]
    else:
        regime = REGIME_ORDER[2]
    return {"vix_avg": avg, "vix_median": med, "vix_regime": regime}


def _signal_ic(scores_slice: dict[str, pd.DataFrame],
               fwd_h: dict[str, pd.Series],
               signal: str) -> list[float]:
    ics: list[float] = []
    for d, fwd in fwd_h.items():
        frame = scores_slice.get(d)
        if frame is None or signal not in frame.columns:
            continue
        ic = period_ic(frame[signal], fwd, min_names=MIN_NAMES)
        if ic is not None:
            ics.append(ic)
    return ics


def _summarize_ics(ics: list[float]) -> dict:
    if not ics:
        return {"mean_ic": float("nan"), "ic_ir": float("nan"),
                "hit_rate": float("nan"), "n": 0}
    arr = np.array(ics)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if len(arr) > 1 else float("nan")
    ir = mean / std if std and std > 1e-9 else float("nan")
    return {"mean_ic": mean, "ic_ir": ir, "hit_rate": float((arr > 0).mean()), "n": len(arr)}


def _spread(scores_slice: dict[str, pd.DataFrame],
            fwd_h: dict[str, pd.Series], signal: str,
            months: int) -> float:
    """Annualised Q5-Q1 spread for one signal."""
    spreads: list[float] = []
    for d, fwd in fwd_h.items():
        frame = scores_slice.get(d)
        if frame is None or signal not in frame.columns:
            continue
        df = pd.DataFrame({"s": frame[signal], "f": fwd}).dropna()
        if len(df) < MIN_NAMES or df["s"].nunique() < N_QUANTILES:
            continue
        try:
            buckets = pd.qcut(df["s"].rank(method="first"), N_QUANTILES, labels=False)
        except ValueError:
            continue
        means = df["f"].groupby(buckets).mean()
        if len(means) == N_QUANTILES and means.notna().all():
            spreads.append(float(means.iloc[-1] - means.iloc[0]))
    if not spreads:
        return float("nan")
    ppy = 12.0 / months
    return float(np.mean(spreads) * ppy)


def _coverage(scores_slice: dict[str, pd.DataFrame], signal: str) -> float:
    """Fraction of rebalances where the signal has ≥MIN_NAMES non-NaN values."""
    hits = total = 0
    for frame in scores_slice.values():
        if frame is None:
            continue
        total += 1
        if signal in frame.columns and frame[signal].count() >= MIN_NAMES:
            hits += 1
    return hits / total if total else float("nan")


def _make_heatmap(df: pd.DataFrame, title: str, cmap: str, fmt: str,
                  out_path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
    except ImportError:
        return

    if df.empty:
        return

    fig_h = max(4, len(df.index) * 0.55 + 2)
    fig_w = max(6, len(df.columns) * 0.9 + 2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    data = df.values.astype(float)
    finite = data[np.isfinite(data)]
    vabs = float(np.nanmax(np.abs(finite))) if finite.size else 1.0

    if cmap == "RdYlGn":
        norm = mcolors.TwoSlopeNorm(vmin=-vabs, vcenter=0, vmax=vabs)
        im = ax.imshow(data, cmap=cmap, aspect="auto", norm=norm)
    else:
        im = ax.imshow(data, cmap=cmap, aspect="auto", vmin=0, vmax=vabs)

    ax.set_xticks(range(len(df.columns)))
    ax.set_xticklabels([str(c) for c in df.columns], fontsize=9, rotation=45, ha="right")
    ax.set_yticks(range(len(df.index)))
    ax.set_yticklabels(df.index, fontsize=8)

    for i in range(len(df.index)):
        for j in range(len(df.columns)):
            v = data[i, j]
            txt = f"{v:{fmt}}" if np.isfinite(v) else "—"
            brightness = abs(v) / vabs if vabs > 0 else 0
            ax.text(j, i, txt, ha="center", va="center", fontsize=7,
                    color="white" if brightness > 0.6 else "black")

    plt.colorbar(im, ax=ax, shrink=0.7)
    ax.set_title(title, fontsize=11, pad=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def run_persistence_analysis(
    panel: ScorePanel,
    matrix: pd.DataFrame,
    vix: pd.Series,
    *,
    first_year: int = 2017,
    last_year: int = 2026,
    verbose: bool = True,
) -> dict:
    """Compute per-year, per-signal persistence and regime statistics.

    Returns a dict of DataFrames ready for writing / plotting.
    """
    all_subs = panel.all_subs
    parents = panel.parent_keys

    # Build calendar-year rebalance slices
    rebal_dates = panel.rebal_dates
    year_dates: dict[int, list[str]] = {}
    for d in rebal_dates:
        y = pd.Timestamp(d).year
        if first_year <= y <= last_year:
            year_dates.setdefault(y, []).append(d)

    years = sorted(year_dates)
    if verbose:
        print(f"  Years covered: {years[0]}–{years[-1]} "
              f"({len(years)} calendar years)")

    # Precompute forward returns for all rebalances
    all_fwd = compute_forward_returns(matrix, rebal_dates, HORIZON_MONTHS)
    fwd_by_h: dict[str, dict[str, pd.Series]] = {h: {} for h in HORIZONS}
    for h in HORIZONS:
        for d, fwd in all_fwd.get(h, {}).items():
            fwd_by_h[h][d] = fwd

    # --- per-year, per-subfactor IC ---
    sub_rows: list[dict] = []       # one row per (year, subfactor)
    parent_rows: list[dict] = []    # one row per (year, parent)

    for year in years:
        dates = year_dates[year]
        scores_slice = {d: panel.scores.get(d) for d in dates}
        vix_s = _vix_stats(vix, f"{year}-01-01", f"{year}-12-31")

        if verbose:
            print(f"  {year}: {len(dates)} rebalances, VIX avg={vix_s['vix_avg']:.1f} "
                  f"({vix_s['vix_regime']})")

        fwd_slices: dict[str, dict[str, pd.Series]] = {}
        for h in HORIZONS:
            fwd_slices[h] = {d: fwd_by_h[h][d] for d in dates if d in fwd_by_h[h]}

        # Subfactor rows
        for sub in all_subs:
            parent = panel.parent_of(sub)
            row = {"year": year, "sub_factor": sub, "parent": parent,
                   **vix_s}
            for h in HORIZONS:
                ics = _signal_ic(scores_slice, fwd_slices[h], sub)
                stats = _summarize_ics(ics)
                row[f"ic_{h}"] = stats["mean_ic"]
                row[f"ir_{h}"] = stats["ic_ir"]
                row[f"hit_{h}"] = stats["hit_rate"]
                row[f"n_{h}"] = stats["n"]
            row["spread_3M"] = _spread(scores_slice, fwd_slices["3M"], sub, 3)
            row["spread_6M"] = _spread(scores_slice, fwd_slices["6M"], sub, 6)
            row["coverage"] = _coverage(scores_slice, sub)
            sub_rows.append(row)

        # Parent rows — build parent composite for this year's dates using frozen
        # sub-weights from the last valid training window up to this year.
        # We derive sub-weights by running select_config on all data *before* this year.
        train_dates = [d for d in rebal_dates if pd.Timestamp(d).year < year]
        if len(train_dates) >= 12:
            boundary = f"{year}-01-01"
            try:
                cfg = select_config(panel, train_dates, matrix, boundary=boundary)
                parent_panel = build_parent_panel(slice_panel(panel, dates), cfg.sub_weights)
                p_weights = cfg.parent_weights
            except Exception:
                parent_panel = None
                p_weights = {}
        else:
            parent_panel = None
            p_weights = {}

        for parent in parents:
            row = {"year": year, "parent": parent, **vix_s,
                   "weight": p_weights.get(parent, float("nan"))}
            if parent_panel is not None:
                p_scores = {d: parent_panel.scores.get(d, pd.DataFrame()) for d in dates}
                p_slices: dict[str, dict] = {}
                for h in HORIZONS:
                    p_slices[h] = {d: fwd_by_h[h][d] for d in dates if d in fwd_by_h[h]}
                p_scores_series: dict[str, dict[str, pd.Series]] = {}
                for d in dates:
                    frame = parent_panel.scores.get(d)
                    if frame is not None and parent in frame.columns:
                        p_scores_series.setdefault(parent, {})[d] = frame[parent]
                p_series = p_scores_series.get(parent, {})
                for h in HORIZONS:
                    ics = [period_ic(p_series[d], p_slices[h].get(d), min_names=MIN_NAMES)
                           for d in dates if d in p_series and d in p_slices[h]]
                    ics = [x for x in ics if x is not None]
                    stats = _summarize_ics(ics)
                    row[f"ic_{h}"] = stats["mean_ic"]
                    row[f"ir_{h}"] = stats["ic_ir"]
                    row[f"hit_{h}"] = stats["hit_rate"]
                row["spread_6M"] = _spread(
                    {d: (parent_panel.scores.get(d)[[parent]]
                         if parent_panel.scores.get(d) is not None
                            and parent in parent_panel.scores.get(d, pd.DataFrame()).columns
                         else pd.DataFrame())
                     for d in dates},
                    fwd_slices["6M"], parent, 6)
            else:
                for h in HORIZONS:
                    row[f"ic_{h}"] = float("nan")
                    row[f"ir_{h}"] = float("nan")
                    row[f"hit_{h}"] = float("nan")
                row["spread_6M"] = float("nan")
            parent_rows.append(row)

    sub_df = pd.DataFrame(sub_rows)
    parent_df = pd.DataFrame(parent_rows)

    # --- selection frequency from existing walkforward output ---
    sel_freq_path = Path("output/walkforward/subfactor_selection_frequency.csv")
    if sel_freq_path.exists():
        sel_freq = pd.read_csv(sel_freq_path)
    else:
        sel_freq = pd.DataFrame(columns=["sub_factor", "freq", "avg_weight"])

    # --- Pivot IC by year ---
    sub_ic_1m = sub_df.pivot(index="sub_factor", columns="year", values="ic_1M")
    sub_ic_3m = sub_df.pivot(index="sub_factor", columns="year", values="ic_3M")
    sub_ic_6m = sub_df.pivot(index="sub_factor", columns="year", values="ic_6M")
    sub_ir_3m = sub_df.pivot(index="sub_factor", columns="year", values="ir_3M")
    sub_spread = sub_df.pivot(index="sub_factor", columns="year", values="spread_6M")

    parent_ic_3m = parent_df.pivot(index="parent", columns="year", values="ic_3M")
    parent_ic_6m = parent_df.pivot(index="parent", columns="year", values="ic_6M")
    parent_ir_3m = parent_df.pivot(index="parent", columns="year", values="ir_3M")
    parent_spread = parent_df.pivot(index="parent", columns="year", values="spread_6M")
    parent_weight = parent_df.pivot(index="parent", columns="year", values="weight")

    # --- Subfactor persistence table ---
    sub_summary_rows = []
    for sub in all_subs:
        parent = panel.parent_of(sub)
        s3 = sub_df[sub_df.sub_factor == sub]["ic_3M"].dropna()
        s6 = sub_df[sub_df.sub_factor == sub]["ic_6M"].dropna()
        sel = sel_freq[sel_freq.sub_factor == sub]
        freq = float(sel["freq"].iloc[0]) if not sel.empty else float("nan")
        avg_w = float(sel["avg_weight"].iloc[0]) if not sel.empty else float("nan")
        mean3 = float(s3.mean()) if not s3.empty else float("nan")
        std3 = float(s3.std(ddof=1)) if len(s3) > 1 else float("nan")
        frac_pos3 = float((s3 > 0).mean()) if not s3.empty else float("nan")
        mean6 = float(s6.mean()) if not s6.empty else float("nan")
        std6 = float(s6.std(ddof=1)) if len(s6) > 1 else float("nan")
        frac_pos6 = float((s6 > 0).mean()) if not s6.empty else float("nan")
        # Persistence score: mean IC / std (higher is more stable and positive)
        pers3 = mean3 / std3 if std3 and std3 > 1e-9 else float("nan")
        if mean3 > 0.01 and frac_pos3 >= 0.6:
            flag = "PERSISTENT"
        elif mean3 < -0.01 and frac_pos3 <= 0.4:
            flag = "PERSISTENTLY_NEGATIVE"
        elif std3 > 0.04 and abs(mean3) < 0.01:
            flag = "NOISY"
        else:
            flag = "MIXED"
        sub_summary_rows.append({
            "sub_factor": sub, "parent": parent,
            "mean_ic_3M": mean3, "std_ic_3M": std3, "pct_positive_3M": frac_pos3,
            "mean_ic_6M": mean6, "std_ic_6M": std6, "pct_positive_6M": frac_pos6,
            "persistence_ir_3M": pers3,
            "selection_freq": freq, "avg_selected_weight": avg_w,
            "flag": flag,
        })
    sub_summary = (pd.DataFrame(sub_summary_rows)
                   .sort_values("persistence_ir_3M", ascending=False))

    # --- Parent persistence table ---
    parent_summary_rows = []
    for parent in parents:
        s3 = parent_df[parent_df.parent == parent]["ic_3M"].dropna()
        s6 = parent_df[parent_df.parent == parent]["ic_6M"].dropna()
        w_series = parent_df[parent_df.parent == parent]["weight"].dropna()
        mean3 = float(s3.mean()) if not s3.empty else float("nan")
        std3 = float(s3.std(ddof=1)) if len(s3) > 1 else float("nan")
        frac_pos3 = float((s3 > 0).mean()) if not s3.empty else float("nan")
        mean6 = float(s6.mean()) if not s6.empty else float("nan")
        pers3 = mean3 / std3 if std3 and std3 > 1e-9 else float("nan")
        if mean3 > 0.005 and frac_pos3 >= 0.6:
            flag = "PERSISTENT"
        elif mean3 < -0.005 and frac_pos3 <= 0.4:
            flag = "PERSISTENTLY_NEGATIVE"
        else:
            flag = "MIXED"
        parent_summary_rows.append({
            "parent": parent,
            "mean_ic_3M": mean3, "std_ic_3M": std3, "pct_positive_3M": frac_pos3,
            "mean_ic_6M": mean6,
            "persistence_ir_3M": pers3,
            "avg_weight": float(w_series.mean()) if not w_series.empty else float("nan"),
            "flag": flag,
        })
    parent_summary = (pd.DataFrame(parent_summary_rows)
                      .sort_values("persistence_ir_3M", ascending=False))

    # --- Regime aggregations ---
    def _regime_agg(df: pd.DataFrame, signal_col: str, value_col: str) -> pd.DataFrame:
        rows = []
        signals = df[signal_col].unique()
        for sig in signals:
            sub = df[df[signal_col] == sig]
            row = {signal_col: sig}
            for regime in REGIME_ORDER:
                vals = sub[sub.vix_regime == regime][value_col].dropna()
                row[regime] = float(vals.mean()) if not vals.empty else float("nan")
            rows.append(row)
        return pd.DataFrame(rows).set_index(signal_col)

    regime_sub_ic3 = _regime_agg(sub_df, "sub_factor", "ic_3M")
    regime_sub_ic6 = _regime_agg(sub_df, "sub_factor", "ic_6M")
    regime_parent_ic3 = _regime_agg(parent_df, "parent", "ic_3M")
    regime_parent_ic6 = _regime_agg(parent_df, "parent", "ic_6M")
    regime_parent_spread = _regime_agg(parent_df, "parent", "spread_6M")

    # --- Stability rankings ---
    def _regime_dependence(regime_df: pd.DataFrame) -> pd.Series:
        """Std across regimes (higher = more regime-dependent)."""
        cols = [c for c in REGIME_ORDER if c in regime_df.columns]
        return regime_df[cols].std(axis=1, ddof=0)

    sub_pers_rank = sub_summary.copy()
    sub_pers_rank["regime_dep"] = _regime_dependence(regime_sub_ic6).reindex(
        sub_pers_rank["sub_factor"].values).values

    parent_pers_rank = parent_summary.copy()
    parent_pers_rank["regime_dep"] = _regime_dependence(regime_parent_ic6).reindex(
        parent_pers_rank["parent"].values).values

    return {
        # Raw per-year tables
        "sub_df": sub_df,
        "parent_df": parent_df,
        # Pivots
        "sub_ic_1m": sub_ic_1m,
        "sub_ic_3m": sub_ic_3m,
        "sub_ic_6m": sub_ic_6m,
        "sub_ir_3m": sub_ir_3m,
        "sub_spread_6m": sub_spread,
        "parent_ic_3m": parent_ic_3m,
        "parent_ic_6m": parent_ic_6m,
        "parent_ir_3m": parent_ir_3m,
        "parent_spread_6m": parent_spread,
        "parent_weight": parent_weight,
        # Summaries
        "sub_persistence": sub_summary,
        "parent_persistence": parent_summary,
        # Regime tables
        "regime_sub_ic3": regime_sub_ic3,
        "regime_sub_ic6": regime_sub_ic6,
        "regime_parent_ic3": regime_parent_ic3,
        "regime_parent_ic6": regime_parent_ic6,
        "regime_parent_spread": regime_parent_spread,
        # Stability rankings
        "sub_stability_rank": sub_pers_rank,
        "parent_stability_rank": parent_pers_rank,
    }


# ---------------------------------------------------------------------------
# Writing outputs
# ---------------------------------------------------------------------------

def write_outputs(results: dict, out_dir: Path, verbose: bool = True) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    def _save(df: pd.DataFrame, name: str) -> None:
        p = out_dir / name
        df.to_csv(p)
        if verbose:
            print(f"  wrote {p}")

    # Raw annual detail
    _save(results["sub_df"], "subfactor_annual_detail.csv")
    _save(results["parent_df"], "parent_annual_detail.csv")

    # IC pivots
    _save(results["sub_ic_3m"], "subfactor_ic_3m_by_year.csv")
    _save(results["sub_ic_6m"], "subfactor_ic_6m_by_year.csv")
    _save(results["parent_ic_3m"], "parent_ic_3m_by_year.csv")
    _save(results["parent_ic_6m"], "parent_ic_6m_by_year.csv")
    _save(results["parent_weight"], "parent_weight_by_year.csv")
    _save(results["parent_spread_6m"], "parent_spread_6m_by_year.csv")

    # Summary / persistence
    _save(results["sub_persistence"], "subfactor_persistence_table.csv")
    _save(results["parent_persistence"], "parent_persistence_table.csv")

    # Regime
    _save(results["regime_sub_ic6"], "regime_subfactor_ic6m.csv")
    _save(results["regime_parent_ic6"], "regime_parent_ic6m.csv")
    _save(results["regime_parent_spread"], "regime_parent_spread6m.csv")

    # Stability rankings
    _save(results["sub_stability_rank"], "stability_ranking_subfactors.csv")
    _save(results["parent_stability_rank"], "stability_ranking_parents.csv")

    # --- Heatmaps ---
    if verbose:
        print("  generating heatmaps…")

    sub_ic6 = results["sub_ic_6m"]
    # Sort subfactors by parent for readability
    if not sub_ic6.empty:
        _make_heatmap(sub_ic6, "Subfactor OOS IC (6M) by Year",
                      "RdYlGn", "+.3f", out_dir / "heatmap_subfactor_ic.png")

    sub_ic3 = results["sub_ic_3m"]
    if not sub_ic3.empty:
        _make_heatmap(sub_ic3, "Subfactor OOS IC (3M) by Year",
                      "RdYlGn", "+.3f", out_dir / "heatmap_subfactor_ic_3m.png")

    p_ic6 = results["parent_ic_6m"]
    if not p_ic6.empty:
        _make_heatmap(p_ic6, "Parent OOS IC (6M) by Year",
                      "RdYlGn", "+.3f", out_dir / "heatmap_parent_ic.png")

    p_weight = results["parent_weight"]
    if not p_weight.empty:
        _make_heatmap(p_weight, "Parent Composite Weight by Year (frozen at training)",
                      "Blues", ".2f", out_dir / "heatmap_parent_weight.png")

    p_spread = results["parent_spread_6m"]
    if not p_spread.empty:
        _make_heatmap(p_spread, "Parent Q5-Q1 Spread (6M, annualised) by Year",
                      "RdYlGn", "+.2f", out_dir / "heatmap_parent_spread.png")

    regime_sub = results["regime_sub_ic6"]
    if not regime_sub.empty:
        _make_heatmap(regime_sub, "Subfactor Mean IC (6M) by VIX Regime",
                      "RdYlGn", "+.3f", out_dir / "heatmap_regime_subfactor_ic.png")

    regime_par = results["regime_parent_ic6"]
    if not regime_par.empty:
        _make_heatmap(regime_par, "Parent Mean IC (6M) by VIX Regime",
                      "RdYlGn", "+.3f", out_dir / "heatmap_regime_parent_ic.png")

    regime_spr = results["regime_parent_spread"]
    if not regime_spr.empty:
        _make_heatmap(regime_spr, "Parent Q5-Q1 Spread (6M) by VIX Regime",
                      "RdYlGn", "+.2f", out_dir / "heatmap_regime_parent_spread.png")

    # --- Markdown report ---
    _write_report(results, out_dir)
    if verbose:
        print(f"  wrote {out_dir / 'FACTOR_PERSISTENCE_REPORT.md'}")


def _fmt(v, fmt=".3f") -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    return f"{v:{fmt}}"


def _write_report(results: dict, out_dir: Path) -> None:
    sub_p = results["sub_persistence"]
    par_p = results["parent_persistence"]
    sub_rank = results["sub_stability_rank"]
    par_rank = results["parent_stability_rank"]
    regime_sub = results["regime_sub_ic6"]
    regime_par = results["regime_parent_ic6"]

    persistent_subs = sub_p[sub_p.flag == "PERSISTENT"]["sub_factor"].tolist()
    neg_subs = sub_p[sub_p.flag == "PERSISTENTLY_NEGATIVE"]["sub_factor"].tolist()
    top5_subs = (sub_rank.dropna(subset=["persistence_ir_3M"])
                 .head(5)["sub_factor"].tolist())

    persistent_pars = par_p[par_p.flag == "PERSISTENT"]["parent"].tolist()
    neg_pars = par_p[par_p.flag == "PERSISTENTLY_NEGATIVE"]["parent"].tolist()

    # Most regime-dependent
    if "regime_dep" in sub_rank.columns:
        regime_dep_subs = (sub_rank.dropna(subset=["regime_dep"])
                           .sort_values("regime_dep", ascending=False)
                           .head(5)["sub_factor"].tolist())
    else:
        regime_dep_subs = []

    if "regime_dep" in par_rank.columns:
        regime_dep_pars = (par_rank.dropna(subset=["regime_dep"])
                           .sort_values("regime_dep", ascending=False)
                           .head(3)["parent"].tolist())
    else:
        regime_dep_pars = []

    lines = [
        "# Factor Persistence & Regime Analysis",
        "",
        "## Overview",
        "",
        "This report covers every parent factor and selected subfactor, measured annually "
        "using point-in-time rebalances.  For each year the VIX regime is recorded "
        "(Low <15 / Medium 15-25 / High >25) and IC, IC IR, Q5-Q1 spread, hit rate and "
        "coverage are computed on OOS data.",
        "",
        "## Subfactor Persistence Summary",
        "",
        f"**Persistently positive (mean IC > 0.01, ≥60 % years positive):** "
        f"{', '.join(persistent_subs) if persistent_subs else 'None'}",
        "",
        f"**Persistently negative (mean IC < −0.01, ≤40 % years positive):** "
        f"{', '.join(neg_subs) if neg_subs else 'None'}",
        "",
        f"**Top 5 by stability IR (mean IC / std IC at 3M):** "
        f"{', '.join(top5_subs)}",
        "",
        "### Subfactor Persistence Table (top 15 by persistence IR)",
        "",
        "| Subfactor | Parent | Mean IC 3M | Std IC 3M | % Positive 3M | Persistence IR | Flag |",
        "|-----------|--------|-----------|-----------|--------------|---------------|------|",
    ]
    for _, row in sub_p.head(15).iterrows():
        lines.append(
            f"| {row.sub_factor} | {row.parent} | {_fmt(row.mean_ic_3M)} | "
            f"{_fmt(row.std_ic_3M)} | {_fmt(row.pct_positive_3M, '.0%')} | "
            f"{_fmt(row.persistence_ir_3M)} | **{row.flag}** |"
        )

    lines += [
        "",
        "## Parent Persistence Summary",
        "",
        f"**Persistently positive:** {', '.join(persistent_pars) if persistent_pars else 'None'}",
        "",
        f"**Persistently negative:** {', '.join(neg_pars) if neg_pars else 'None'}",
        "",
        "### Parent Persistence Table",
        "",
        "| Parent | Mean IC 3M | Std IC 3M | % Positive 3M | Persistence IR | Avg Weight | Flag |",
        "|--------|-----------|-----------|--------------|---------------|-----------|------|",
    ]
    for _, row in par_p.iterrows():
        lines.append(
            f"| {row.parent} | {_fmt(row.mean_ic_3M)} | {_fmt(row.std_ic_3M)} | "
            f"{_fmt(row.pct_positive_3M, '.0%')} | {_fmt(row.persistence_ir_3M)} | "
            f"{_fmt(row.avg_weight, '.1%')} | **{row.flag}** |"
        )

    lines += [
        "",
        "## Regime Analysis",
        "",
        "### Parent IC (6M) by VIX Regime",
        "",
        "| Parent | Low (<15) | Medium (15-25) | High (>25) |",
        "|--------|----------|---------------|-----------|",
    ]
    for parent in regime_par.index:
        row = regime_par.loc[parent]
        lines.append(
            f"| {parent} | {_fmt(row.get('Low (<15)', float('nan')))} | "
            f"{_fmt(row.get('Medium (15-25)', float('nan')))} | "
            f"{_fmt(row.get('High (>25)', float('nan')))} |"
        )

    lines += [
        "",
        f"**Most regime-dependent subfactors:** {', '.join(regime_dep_subs)}",
        "",
        f"**Most regime-dependent parents:** {', '.join(regime_dep_pars)}",
        "",
        "## Key Conclusions",
        "",
        "1. See `subfactor_persistence_table.csv` for the full persistence flags.",
        "2. See `heatmap_subfactor_ic.png` for year-by-year IC patterns.",
        "3. See `heatmap_regime_subfactor_ic.png` for regime sensitivity per subfactor.",
        "4. Subfactors flagged **PERSISTENTLY_NEGATIVE** are candidates for removal.",
        "5. Subfactors flagged **PERSISTENT** with low regime dependence deserve higher "
        "weight in low- and high-VIX environments alike.",
    ]

    (out_dir / "FACTOR_PERSISTENCE_REPORT.md").write_text("\n".join(lines))
