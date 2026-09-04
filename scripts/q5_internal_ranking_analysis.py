#!/usr/bin/env python3
"""Q5 internal ranking analysis: does ranking within the top quintile predict returns?

2025 OOS expanding split (trained ≤ 2024-06-28).  For each rebalance: rank universe
by composite score, take Q5 top-50, record parent scores + 3M forward return, compare
top-10 vs bottom-10.

Weekly mode: forward-fills monthly composite scores to each weekly trading date and
computes fresh 3M returns from each weekly start, so both the signal quality and
rebalance-frequency impact can be compared side-by-side.

Usage:
    python scripts/q5_internal_ranking_analysis.py              # monthly (default)
    python scripts/q5_internal_ranking_analysis.py --rebalance weekly
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as sc_stats

from backtesting import data_loader as dl
from data.db import get_db
from research import HORIZON_MONTHS, compute_forward_returns
from research.panel import ScorePanel
from research.subfactor_expansion.panel import load_cached_panel, cache_key
from research.walkforward.compose import (
    FrozenConfig, build_parent_panel, frozen_composite
)
from research.walkforward.selection import select_config, slice_panel
from research.walkforward.splits import WalkForwardSplit, SELECTION_HORIZON_MONTHS

PANEL_START = "2015-06-30"
PANEL_END   = "2026-06-30"
PRICE_END   = "2026-07-06"
CACHE_DIR   = Path("cache/subfactor_expansion")

# The two 6-month OOS periods within the 2025 test year
PERIODS = {
    "H1-2025 (Jan–Jun)": ("2025-01-01", "2025-06-30"),
    "H2-2025 (Jul–Dec)": ("2025-07-01", "2025-12-31"),
}

PARENTS = ["momentum", "value", "quality", "growth",
           "revisions", "institutional", "insider", "short"]


# ---------------------------------------------------------------------------
# Panel loading (reuse cached CandidatePanel → convert to ScorePanel)
# ---------------------------------------------------------------------------
def _adapt(cand) -> ScorePanel:
    parent_keys = [p for p, subs in cand.candidates_by_parent.items() if subs]
    return ScorePanel(
        rebal_dates=list(cand.rebal_dates),
        scores=cand.scores,
        parent_keys=parent_keys,
        sub_by_parent={p: list(cand.candidates_by_parent[p]) for p in parent_keys},
        universe=list(cand.universe),
    )


def _load_panel() -> ScorePanel:
    ckey = CACHE_DIR / cache_key(PANEL_START, PANEL_END, "monthly")
    cand = load_cached_panel(ckey)
    if cand is None:
        raise SystemExit(
            f"Candidate panel cache not found at {ckey}.\n"
            "Run:  python run_walkforward.py --rebuild-panel"
        )
    print(f"Panel loaded: {len(cand.rebal_dates)} rebalances "
          f"({cand.rebal_dates[0]} → {cand.rebal_dates[-1]}), "
          f"{len(cand.universe)} tickers.")
    return _adapt(cand)


# ---------------------------------------------------------------------------
# Per-rebalance analysis: top-50 within Q5
# ---------------------------------------------------------------------------
def analyse_rebal(
    date: str,
    composite_scores: pd.Series,
    parent_scores_frame: pd.DataFrame,
    fwd_3m: pd.Series,
    n_top: int = 50,
) -> pd.DataFrame:
    """Return a DataFrame of the top-n_top Q5 stocks with all metadata."""
    comp_rank = composite_scores.rank(ascending=False, method="first")
    n_universe = comp_rank.notna().sum()
    q5_cutoff_rank = int(np.ceil(n_universe / 5))

    top50 = (
        comp_rank[comp_rank <= q5_cutoff_rank]
        .sort_values()
        .head(n_top)
    )
    tickers = top50.index.tolist()

    rows = []
    for tk in tickers:
        row: dict = {
            "date": date,
            "ticker": tk,
            "composite_score": round(float(composite_scores.get(tk, np.nan)), 2),
            "composite_rank": int(comp_rank.get(tk, np.nan)),
            "universe_size": n_universe,
            "q5_cutoff_rank": q5_cutoff_rank,
        }
        for p in PARENTS:
            raw = float(parent_scores_frame.get(p, pd.Series()).get(tk, np.nan))
            row[f"{p}_score"] = round(raw, 2) if not np.isnan(raw) else np.nan
        row["fwd_3m_return"] = (
            round(float(fwd_3m.get(tk, np.nan)) * 100, 2)
            if tk in fwd_3m.index else np.nan
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _tier_summary(period_df: pd.DataFrame, label: str) -> pd.DataFrame:
    records = []
    for date, grp in period_df.groupby("date"):
        grp_sorted = grp.sort_values("composite_rank")
        top10 = grp_sorted.head(10)["fwd_3m_return"].dropna()
        bot10 = grp_sorted.tail(10)["fwd_3m_return"].dropna()
        if top10.empty or bot10.empty:
            continue
        records.append({
            "date": date,
            "top10_mean_3m_ret": round(float(top10.mean()), 2),
            "bot10_mean_3m_ret": round(float(bot10.mean()), 2),
            "spread_top_minus_bot": round(float(top10.mean() - bot10.mean()), 2),
            "top10_n": len(top10), "bot10_n": len(bot10),
        })
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    df.attrs["period"] = label
    return df


def _correlation_summary(period_df: pd.DataFrame) -> dict:
    sub = period_df[["composite_rank", "fwd_3m_return"]].dropna()
    if len(sub) < 10:
        return {"n": len(sub), "spearman_r": np.nan, "p_value": np.nan}
    r, p = sc_stats.spearmanr(sub["composite_rank"], sub["fwd_3m_return"])
    return {"n": len(sub), "spearman_r": round(float(r), 4), "p_value": round(float(p), 4)}


def _scatter_plot(
    df: pd.DataFrame,
    title: str,
    subtitle: str,
    label_n_obs: str,
    out_path: Path,
    *,
    x_label: str = "Composite Score (0–100, sector-relative percentile)",
    show_quintile_diamonds: bool = True,
) -> None:
    """Composite score vs 3M return scatter, coloured H1/H2."""
    sub = df[["score", "ret"]].dropna()
    if len(sub) < 10:
        print(f"  Skipping {out_path.name}: too few observations ({len(sub)}).")
        return
    x, y = sub["score"].values, sub["ret"].values
    slope, intercept, r, p, se = sc_stats.linregress(x, y)
    r_sp, p_sp = sc_stats.spearmanr(x, y)
    t_stat = slope / se if se > 0 else 0.0

    fig, ax = plt.subplots(figsize=(14, 8))
    h1 = df[df["half"] == "H1"]
    h2 = df[df["half"] == "H2"]
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
    print(f"  Saved: {out_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Q5 internal ranking analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--rebalance", default="monthly", choices=["monthly", "weekly"],
                   help="Rebalance frequency; weekly forward-fills monthly scores to weekly dates")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    rebalance = args.rebalance

    out_dir_name = "q5_internal_ranking" if rebalance == "monthly" else "q5_internal_ranking_weekly"
    OUT_DIR = Path("output") / out_dir_name
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # -- 1. Load panel & prices -----------------------------------------------
    panel = _load_panel()
    with get_db() as db:
        matrix = dl.load_price_matrix(db, panel.universe, PANEL_START, PRICE_END)
        sectors = dl.global_sectors(db)
    print(f"Price matrix: {matrix.shape[1]} tickers × {matrix.shape[0]} days.")

    # -- 2. Select frozen config for the 2025 expanding split -----------------
    split = WalkForwardSplit(
        label="2025",
        train_start=PANEL_START,
        train_end="2024-12-31",
        test_start="2025-01-01",
        test_end="2025-12-31",
        policy="expanding",
    )
    train_rebals = split.train_rebalances(panel.rebal_dates)
    test_rebals  = split.test_rebalances(panel.rebal_dates)
    print(f"\nSplit 2025 (expanding):")
    print(f"  Training: {len(train_rebals)} rebalances ({train_rebals[0]} → {train_rebals[-1]})")
    print(f"  Test:     {len(test_rebals)} rebalances ({test_rebals[0]} → {test_rebals[-1]})")

    print("\nRunning V4 selection on training data (may take 1-2 min)…")
    cfg = select_config(panel, train_rebals, matrix, boundary=split.test_start)
    print(f"  Parent weights: { {p: round(w,3) for p, w in cfg.parent_weights.items()} }")

    # -- 3. Composite scores & parent panel for monthly test rebalances -------
    print("\nScoring test rebalances…")
    comp_scores = frozen_composite(panel, test_rebals, cfg, sectors)

    test_sub_panel = slice_panel(panel, test_rebals)
    par_panel = build_parent_panel(test_sub_panel, cfg.sub_weights)

    # -- 4. Resolve effective rebalance dates & scores ------------------------
    if rebalance == "weekly":
        trading_dates = sorted(pd.to_datetime(matrix.index).strftime("%Y-%m-%d").tolist())
        weekly_rebals = dl.generate_rebalance_dates(
            trading_dates, "2025-01-01", "2025-12-31", "weekly")
        monthly_sorted = sorted(comp_scores.keys())

        eff_comp: dict[str, pd.Series] = {}
        eff_par_scores: dict[str, pd.DataFrame] = {}
        for w in weekly_rebals:
            # most-recent monthly panel date at or before this weekly date
            src = max((d for d in monthly_sorted if d <= w), default=None)
            if src is None:
                continue
            eff_comp[w] = comp_scores[src]
            if src in par_panel.scores:
                eff_par_scores[w] = par_panel.scores[src]

        # Fresh 3M forward returns from each weekly start date
        fwd_all_w = compute_forward_returns(matrix, list(eff_comp.keys()), {"3M": 3})
        eff_fwd = {w: fwd_all_w.get("3M", {}).get(w, pd.Series(dtype=float))
                   for w in eff_comp}
        eff_rebals = sorted(eff_comp.keys())
        print(f"  Weekly mode: {len(eff_rebals)} rebalances "
              f"({eff_rebals[0]} → {eff_rebals[-1]}, "
              f"forward-filled from {len(test_rebals)} monthly dates)")
    else:
        fwd_all = compute_forward_returns(matrix, test_rebals, {"3M": 3})
        eff_comp = comp_scores
        eff_par_scores = {d: par_panel.scores.get(d, pd.DataFrame()) for d in test_rebals}
        eff_fwd = {d: fwd_all.get("3M", {}).get(d, pd.Series(dtype=float))
                   for d in test_rebals}
        eff_rebals = test_rebals

    available = sum(1 for s in eff_fwd.values() if not s.empty)
    print(f"  3M forward returns available for {available}/{len(eff_rebals)} rebalances.")

    # -- 5. Per-period Q5-top-50 tables ---------------------------------------
    all_period_dfs: dict[str, pd.DataFrame] = {}

    for period_label, (p_start, p_end) in PERIODS.items():
        period_rebals = [d for d in eff_rebals if p_start <= d <= p_end]
        if not period_rebals:
            print(f"\n{period_label}: no rebalances found, skipping.")
            continue

        print(f"\n{'='*70}")
        print(f"  {period_label}  |  {len(period_rebals)} rebalances: "
              f"{period_rebals[0]} → {period_rebals[-1]}")
        print(f"{'='*70}")

        period_rows = []
        for d in period_rebals:
            sc = eff_comp.get(d)
            fwd3 = eff_fwd.get(d, pd.Series(dtype=float))
            if sc is None:
                print(f"  {d}: no composite scores, skipping.")
                continue
            par_frame = eff_par_scores.get(d, pd.DataFrame(index=sc.index))
            rdf = analyse_rebal(d, sc, par_frame, fwd3)
            period_rows.append(rdf)

            r_sorted = rdf.sort_values("composite_rank")
            t10 = r_sorted.head(10)["fwd_3m_return"].dropna()
            b10 = r_sorted.tail(10)["fwd_3m_return"].dropna()
            print(f"\n  {d}  |  Q5 universe = {rdf['q5_cutoff_rank'].iloc[0]} names  "
                  f"(showing top-{len(rdf)} by rank)")
            col_w = 8
            hdr = (f"  {'Rank':>4}  {'Ticker':<7}  {'Comp':>5}  "
                   + "  ".join(f"{p[:3]:>{col_w}}" for p in PARENTS)
                   + f"  {'3M%':>7}")
            print(hdr)
            print("  " + "-" * (len(hdr) - 2))
            for _, row in r_sorted.head(50).iterrows():
                parent_vals = "  ".join(
                    f"{row.get(f'{p}_score', np.nan):>{col_w}.1f}"
                    if not pd.isna(row.get(f"{p}_score", np.nan)) else f"{'  -':>{col_w}}"
                    for p in PARENTS
                )
                ret_str = (f"{row['fwd_3m_return']:>7.2f}"
                           if not pd.isna(row['fwd_3m_return']) else f"{'  --':>7}")
                print(f"  {int(row['composite_rank']):>4}  {row['ticker']:<7}  "
                      f"{row['composite_score']:>5.1f}  {parent_vals}  {ret_str}")

            t10_m = float(t10.mean()) if not t10.empty else np.nan
            b10_m = float(b10.mean()) if not b10.empty else np.nan
            spread = t10_m - b10_m if not (np.isnan(t10_m) or np.isnan(b10_m)) else np.nan
            print(f"\n  Top-10 mean 3M: {t10_m:+.2f}%  |  "
                  f"Bot-10 mean 3M: {b10_m:+.2f}%  |  "
                  f"Spread: {spread:+.2f}%  "
                  f"({'top beats bot' if spread > 0 else 'bot beats top' if spread < 0 else 'tied'})")

        if period_rows:
            period_df = pd.concat(period_rows, ignore_index=True)
            all_period_dfs[period_label] = period_df

    # -- 6. Cross-rebalance summary -------------------------------------------
    print(f"\n\n{'='*70}")
    print("  SUMMARY ACROSS BOTH OOS PERIODS")
    print(f"{'='*70}")

    corr_rows = []
    tier_rows = []

    for period_label, period_df in all_period_dfs.items():
        tier = _tier_summary(period_df, period_label)
        corr = _correlation_summary(period_df)

        if not tier.empty:
            pooled_t10 = tier["top10_mean_3m_ret"].mean()
            pooled_b10 = tier["bot10_mean_3m_ret"].mean()
            pooled_spr = tier["spread_top_minus_bot"].mean()
            win_rate   = (tier["spread_top_minus_bot"] > 0).mean()
            print(f"\n  {period_label}")
            print(f"    Pooled top-10 mean 3M: {pooled_t10:+.2f}%")
            print(f"    Pooled bot-10 mean 3M: {pooled_b10:+.2f}%")
            print(f"    Spread (top-10 - bot-10): {pooled_spr:+.2f}%  "
                  f"(positive in {win_rate:.0%} of rebalances)")
            print(f"    Spearman ρ (comp_rank vs 3M_ret, lower rank = better): "
                  f"{corr['spearman_r']:.4f}  (p={corr['p_value']:.3f}, n={corr['n']})")
            print(f"    Interpretation: "
                  f"{'rank predicts return (negative ρ expected)' if corr['spearman_r'] < -0.05 else 'no clear internal ordering signal'}")

        for _, r in tier.iterrows():
            tier_rows.append({"period": period_label, **r.to_dict()})
        corr_rows.append({"period": period_label, **corr})

    if len(all_period_dfs) == 2:
        combined = pd.concat(list(all_period_dfs.values()), ignore_index=True)
        combined_corr = _correlation_summary(combined)
        all_tier = pd.concat([_tier_summary(df, lbl)
                              for lbl, df in all_period_dfs.items()], ignore_index=True)
        if not all_tier.empty:
            print(f"\n  COMBINED (both periods pooled, {len(combined)} stock-observations)")
            t10_c = all_tier["top10_mean_3m_ret"].mean()
            b10_c = all_tier["bot10_mean_3m_ret"].mean()
            spr_c = all_tier["spread_top_minus_bot"].mean()
            wr_c  = (all_tier["spread_top_minus_bot"] > 0).mean()
            print(f"    Top-10 mean 3M: {t10_c:+.2f}%   "
                  f"Bot-10 mean 3M: {b10_c:+.2f}%   "
                  f"Spread: {spr_c:+.2f}%   Win-rate: {wr_c:.0%}")
            print(f"    Spearman ρ: {combined_corr['spearman_r']:.4f}  "
                  f"(p={combined_corr['p_value']:.3f}, n={combined_corr['n']})")
            neg_ρ = combined_corr["spearman_r"] < -0.05
            p_sig = combined_corr["p_value"] < 0.05
            if neg_ρ and p_sig:
                verdict = "MEANINGFUL INTERNAL ORDERING — rank within Q5 predicts 3M returns."
            elif neg_ρ:
                verdict = "WEAK SIGNAL — rank trend is correct direction but not significant."
            else:
                verdict = "NO INTERNAL ORDERING — Q5 is a flat bucket; rank-within-Q5 does not predict."
            print(f"\n  VERDICT: {verdict}")

    # -- 7. Save CSVs ----------------------------------------------------------
    for lbl, df in all_period_dfs.items():
        safe = lbl.replace(" ", "_").replace("(", "").replace(")", "").replace("–", "-")
        path = OUT_DIR / f"q5_top50_{safe}.csv"
        df.to_csv(path, index=False)

    if tier_rows:
        pd.DataFrame(tier_rows).to_csv(OUT_DIR / "tier_summary.csv", index=False)
    if corr_rows:
        pd.DataFrame(corr_rows).to_csv(OUT_DIR / "spearman_correlation.csv", index=False)

    print(f"\nCSVs saved to {OUT_DIR}/")

    # -- 8. Scatter plots ------------------------------------------------------
    model_subtitle = (
        f"2025 OOS — Expanding Model (trained ≤ 2024-06-28) | "
        f"H1 + H2 Combined  [{rebalance} rebalancing]"
    )

    # Full universe scatter
    full_rows = []
    for d in sorted(eff_comp):
        sc = eff_comp[d]
        fwd = eff_fwd.get(d, pd.Series(dtype=float))
        if fwd.empty:
            continue
        comb = pd.DataFrame({"score": sc, "ret": fwd * 100}).dropna()
        comb["half"] = "H1" if d <= "2025-06-30" else "H2"
        full_rows.append(comb)

    if full_rows:
        full_df = pd.concat(full_rows, ignore_index=True)
        _scatter_plot(
            full_df,
            title="Full Universe: Composite Score vs 3-Month Forward Return",
            subtitle=model_subtitle,
            label_n_obs=f"{len(eff_rebals)} rebalances, full universe",
            out_path=OUT_DIR / "full_universe_scatter.png",
            show_quintile_diamonds=True,
        )

    # Q5 scatter
    if all_period_dfs:
        q5_rows = []
        for lbl, df in all_period_dfs.items():
            sub = df[["composite_score", "fwd_3m_return"]].rename(
                columns={"composite_score": "score", "fwd_3m_return": "ret"}).copy()
            sub["half"] = "H1" if "H1" in lbl else "H2"
            q5_rows.append(sub)
        q5_df = pd.concat(q5_rows, ignore_index=True)
        n_q5_rebals = sum(df["date"].nunique() for df in all_period_dfs.values())
        _scatter_plot(
            q5_df,
            title="Q5 Top-50: Composite Score vs 3M Forward Return",
            subtitle=model_subtitle.replace("H1 + H2 Combined", "Both Halves Combined"),
            label_n_obs=f"{n_q5_rebals} rebalances × top-50 Q5",
            out_path=OUT_DIR / "q5_scatter.png",
            show_quintile_diamonds=False,
            x_label="Composite Score (0–100, sector-relative)",
        )

    print(f"\nPlots saved to {OUT_DIR}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
