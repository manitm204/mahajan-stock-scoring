"""Full statistical comparison: E (production) vs EQ vs EQEFF vs A (2026-09-01).

Reuses the cached derivation from scripts/full_pit_backtest_eqeff_2020.py
(output/crowding/weight_config_study/comp_eqeff_2020.pkl — composite scores
per method/date, 78 OOS months 2020-2026, weights re-derived from scratch
every 6mo on trailing 5y windows, no reuse of production weights) so nothing
is re-derived here; this is pure downstream analysis, in 3 parts:

1. IC-family stats (pooled + by-year): IC 3M/6M/blend, IR, hit rate,
   Q5-Q1 AND Q10-Q1 spread — the full picture the portfolio backtest can't
   show on its own (rank quality independent of any one book construction).

2. Rank similarity between methods: per-date Spearman correlation of the
   composite score itself, plus Jaccard overlap of the top-25% and top-10%
   name sets — answers "how much do stock ratings actually change."

3. Portfolio-construction sensitivity grid: the same cached composite scores
   re-simulated across a grid of top_pct x weighting x sector choices (not
   just the single top25/cap5/cap_match setup E happened to be tuned
   against) — tests whether E's portfolio-level edge is robust to
   construction choice or an artifact of it.

Outputs to output/crowding/weight_config_study/:
  ic_family_stats.csv          — pooled + by-year IC/IR/hit/Q5-Q1/Q10-Q1
  rank_similarity.csv          — pairwise Spearman corr + top-25%/10% Jaccard
  portfolio_sensitivity_grid.csv — CAGR/Sharpe per method x construction combo

Usage: python scripts/weight_config_analysis.py
"""
from __future__ import annotations

import pickle
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from backtesting import data_loader as dl
from data.db import get_db
from research.ablation.data import AblationData, _pit_caps
from research.ablation.engine import AblationConfig, simulate_config, benchmark_row
from research.forward_returns import realize_delistings
from research.analyst_deep_dive.common import load_price_matrix, panel_forward_returns, spearman_ic

OUT = REPO / "output" / "crowding" / "weight_config_study"
CACHE = OUT / "comp_eqeff_2020.pkl"
METHODS = ("A", "E", "EQ", "EQEFF")


def load_comp():
    if not CACHE.exists():
        raise SystemExit(f"{CACHE} missing — run scripts/full_pit_backtest_eqeff_2020.py first")
    with open(CACHE, "rb") as f:
        d = pickle.load(f)
    return d["comp"], d["universe"]


# --- 1. IC-family stats ---

def ic_family(comp, fwd):
    def series(cs, h):
        out = {}
        for d, s in cs.items():
            if d in fwd[h]:
                v = spearman_ic(s, fwd[h][d])
                if v is not None:
                    out[d] = v
        return pd.Series(out).sort_index()

    def spread(cs, h, q):
        vals = []
        for d, s in cs.items():
            if d not in fwd[h]:
                continue
            df = pd.DataFrame({"s": s, "f": fwd[h][d]}).dropna()
            if len(df) < 100:
                continue
            qbin = pd.qcut(df["s"].rank(method="first"), q, labels=False)
            m = df.groupby(qbin)["f"].mean()
            vals.append(float(m.iloc[-1] - m.iloc[0]))
        return float(np.mean(vals)) if vals else np.nan

    rows, by_year_rows = [], []
    for m in METHODS:
        cs = comp[m]
        s3, s6 = series(cs, "3M"), series(cs, "6M")
        blend = pd.concat([s3, s6], axis=1).mean(axis=1)
        rows.append({
            "method": m,
            "ic_3M": round(float(s3.mean()), 4), "ic_6M": round(float(s6.mean()), 4),
            "ic_blend": round(float(blend.mean()), 4),
            "ir_blend": round(float(blend.mean() / blend.std(ddof=1)), 3),
            "hit_rate": round(float((blend > 0).mean()), 3),
            "q5q1_3M": round(spread(cs, "3M", 5), 4), "q5q1_6M": round(spread(cs, "6M", 5), 4),
            "q10q1_3M": round(spread(cs, "3M", 10), 4), "q10q1_6M": round(spread(cs, "6M", 10), 4),
            "n_dates": int(len(blend)),
        })
        yearly = blend.groupby(blend.index.str[:4]).agg(["mean", "std", "count"])
        for yr, row in yearly.iterrows():
            ir = row["mean"] / row["std"] if row["std"] > 0 else np.nan
            by_year_rows.append({"method": m, "year": yr, "ic_blend": round(row["mean"], 4),
                                 "ir": round(ir, 3) if pd.notna(ir) else np.nan,
                                 "n": int(row["count"])})
    return pd.DataFrame(rows).set_index("method"), pd.DataFrame(by_year_rows)


# --- 2. Rank similarity ---

def rank_similarity(comp):
    rows = []
    for m1, m2 in [("E", "EQEFF"), ("E", "EQ"), ("EQ", "EQEFF"),
                   ("E", "A"), ("EQEFF", "A"), ("EQ", "A")]:
        dates = sorted(set(comp[m1]) & set(comp[m2]))
        rhos, jac25, jac10 = [], [], []
        for d in dates:
            s1, s2 = comp[m1][d], comp[m2][d]
            df = pd.DataFrame({"a": s1, "b": s2}).dropna()
            if len(df) < 50:
                continue
            rhos.append(df["a"].corr(df["b"], method="spearman"))
            top1_25 = set(df.index[df["a"] >= df["a"].quantile(0.75)])
            top2_25 = set(df.index[df["b"] >= df["b"].quantile(0.75)])
            jac25.append(len(top1_25 & top2_25) / len(top1_25 | top2_25))
            top1_10 = set(df.index[df["a"] >= df["a"].quantile(0.90)])
            top2_10 = set(df.index[df["b"] >= df["b"].quantile(0.90)])
            jac10.append(len(top1_10 & top2_10) / len(top1_10 | top2_10))
        rows.append({"pair": f"{m1} vs {m2}",
                     "mean_spearman_rho": round(float(np.mean(rhos)), 3),
                     "mean_jaccard_top25pct": round(float(np.mean(jac25)), 3),
                     "mean_jaccard_top10pct": round(float(np.mean(jac10)), 3),
                     "n_dates": len(rhos)})
    return pd.DataFrame(rows)


# --- 3. Portfolio-construction sensitivity grid ---

def sensitivity_grid(comp, universe):
    db = get_db()
    sectors = dl.global_sectors(db)
    oos_dates = sorted(comp["A"])
    matrix = realize_delistings(
        dl.load_price_matrix(db, universe, "2019-01-01", "2026-07-31"))
    rebal_dates = [d for d in oos_dates if d in matrix.index]
    caps = _pit_caps(db, matrix, rebal_dates)
    data = AblationData(matrix=matrix, sectors=sectors, run=None,
                        rebal_dates=rebal_dates, caps=caps,
                        vol=pd.DataFrame(index=rebal_dates), vix=pd.Series(dtype=float),
                        parent_ranks={}, panel=None)

    top_pcts = [0.10, 0.25, 0.40]
    weightings = ["ew", "cap5"]
    sector_modes = ["cap_match", "none"]

    rows = []
    for top_pct, weighting, sector in product(top_pcts, weightings, sector_modes):
        for m in METHODS:
            cfg = AblationConfig(name=f"{m}_{top_pct}_{weighting}_{sector}",
                                 top_pct=top_pct, weighting=weighting, sector=sector,
                                 vix_tilt=False, hold_months=1, cost_bps=10.0)
            row = simulate_config(data, cfg, scores=comp[m], keep_series=False)
            row.pop("_spy", None); row.pop("_qqq", None); row.pop("_gross", None)
            row.update({"method": m, "top_pct": top_pct, "weighting": weighting, "sector": sector})
            rows.append(row)
    grid = pd.DataFrame(rows)
    cols = ["method", "top_pct", "weighting", "sector", "net_cagr", "sharpe",
            "sortino", "calmar", "max_dd", "alpha_t", "turnover", "avg_names"]
    return grid[[c for c in cols if c in grid.columns]]


def main():
    comp, universe = load_comp()
    all_dates = sorted(set().union(*[comp[m].keys() for m in METHODS]))
    matrix_full = load_price_matrix(start="2015-01-01", end="2026-07-31")
    fwd = panel_forward_returns(matrix_full, all_dates)

    ic_df, ic_by_year = ic_family(comp, fwd)
    ic_df.to_csv(OUT / "ic_family_stats.csv")
    ic_by_year.to_csv(OUT / "ic_family_by_year.csv", index=False)
    print("=== IC-family stats (pooled, 78mo 2020-2026) ===")
    print(ic_df.to_string())

    sim_df = rank_similarity(comp)
    sim_df.to_csv(OUT / "rank_similarity.csv", index=False)
    print("\n=== Rank similarity ===")
    print(sim_df.to_string(index=False))

    print("\nrunning portfolio sensitivity grid (12 construction combos x 4 methods)...")
    grid_df = sensitivity_grid(comp, universe)
    grid_df.to_csv(OUT / "portfolio_sensitivity_grid.csv", index=False)
    print("\n=== Portfolio sensitivity grid ===")
    with pd.option_context("display.width", 220, "display.max_rows", 100,
                           "display.float_format", lambda v: f"{v:.3f}"):
        print(grid_df.to_string(index=False))

    # win-rate summary: how often does each method have the best CAGR / Sharpe across the grid?
    for metric in ("net_cagr", "sharpe"):
        best = grid_df.loc[grid_df.groupby(["top_pct", "weighting", "sector"])[metric].idxmax()]
        print(f"\nwin count by {metric} across {len(best)} construction combos:")
        print(best["method"].value_counts().to_string())

    print(f"\nwrote {OUT}/ic_family_stats.csv + ic_family_by_year.csv + "
          f"rank_similarity.csv + portfolio_sensitivity_grid.csv")


if __name__ == "__main__":
    main()
