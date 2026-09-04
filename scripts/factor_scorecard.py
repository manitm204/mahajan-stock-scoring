"""Factor scorecard: 10 portfolio-agnostic measures of composite-score quality.

User request (2026-09-01): comparing weight configs via a costed portfolio
backtest bakes in construction choices (top_pct/weighting/sector) that E
happened to be tuned against — scripts/weight_config_analysis.py's
sensitivity grid confirmed the E-vs-EQEFF ranking flips depending on those
choices. This scorecard drops the portfolio wrapper entirely and scores the
composite RANK itself against forward returns, ten complementary ways:

  1. ic_blend        Spearman IC, pooled cross-section, 3M/6M average
  2. ir_blend         IC mean / IC std (consistency)
  3. hit_rate         % of periods with IC > 0 (sign stability)
  4. q5q1             top-quintile minus bottom-quintile mean forward return
  5. monotonicity     fraction of the 10 quintile-pairs correctly ordered
                       (1.0 = returns rise monotonically Q1->Q5 every period)
  6. tent_asymmetry   (Q5-Q3) - (Q3-Q1): winner-finding vs loser-finding skill
  7. within_sector_ic IC recomputed INSIDE each sector-date group and
                       averaged — isolates stock-picking skill from sector
                       rotation bleeding through the pooled IC
  8. fm_slope_t       Fama-MacBeth: per-date OLS slope of fwd return on
                       z-scored composite score, t-stat of the time series
                       of slopes (rigorous linear-relationship test, not
                       just rank)
  9. score_stability  month-over-month Spearman corr of the composite score
                       ITSELF (not vs returns) — turnover proxy, no
                       portfolio construction needed
  10. bottom_decile_drag  Q1 mean return minus that period's cross-sectional
                       mean return — does the score also identify losers,
                       not just winners (relevant for veto-style parents)

All metrics use the SAME cached composite scores as
scripts/weight_config_analysis.py (output/crowding/weight_config_study/
comp_eqeff_2020.pkl — 78 OOS months 2020-2026, weights re-derived from
scratch every 6mo on trailing 5y windows). Metrics 4-6, 10 use 3M forward
returns as the primary trading horizon; 1-3 blend 3M/6M for continuity with
prior studies; 7-9 use 3M only (documented per-row).

Usage: python scripts/factor_scorecard.py
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from backtesting import data_loader as dl
from data.db import get_db
from research.analyst_deep_dive.common import load_price_matrix, panel_forward_returns, spearman_ic

OUT = REPO / "output" / "crowding" / "weight_config_study"
CACHE = OUT / "comp_eqeff_2020.pkl"
METHODS = ("A", "E", "EQ", "EQEFF")
MIN_OBS = 100
MIN_SECTOR_OBS = 10


def load_comp():
    if not CACHE.exists():
        raise SystemExit(f"{CACHE} missing — run scripts/full_pit_backtest_eqeff_2020.py first")
    with open(CACHE, "rb") as f:
        d = pickle.load(f)
    return d["comp"], d["universe"]


def ic_series(cs, fwd, h):
    out = {}
    for d, s in cs.items():
        if d in fwd[h]:
            v = spearman_ic(s, fwd[h][d])
            if v is not None:
                out[d] = v
    return pd.Series(out).sort_index()


def quintile_means(s: pd.Series, f: pd.Series, q: int = 5):
    df = pd.DataFrame({"s": s, "f": f}).dropna()
    if len(df) < MIN_OBS:
        return None
    qbin = pd.qcut(df["s"].rank(method="first"), q, labels=False)
    return df.groupby(qbin)["f"].mean()


def metric_q5q1_mono_tent_bottom(cs, fwd, h="3M"):
    q5q1, mono, tent, bottom_drag = [], [], [], []
    for d, s in cs.items():
        if d not in fwd[h]:
            continue
        m = quintile_means(s, fwd[h][d], 5)
        if m is None or len(m) < 5:
            continue
        arr = m.values
        q5q1.append(float(arr[4] - arr[0]))
        tent.append(float((arr[4] - arr[2]) - (arr[2] - arr[0])))
        pairs = [(i, j) for i in range(5) for j in range(i + 1, 5)]
        mono.append(sum(1 for i, j in pairs if arr[j] >= arr[i]) / len(pairs))
        df = pd.DataFrame({"s": s, "f": fwd[h][d]}).dropna()
        bottom_drag.append(float(arr[0] - df["f"].mean()))
    def avg(x):
        return float(np.mean(x)) if x else np.nan
    return avg(q5q1), avg(mono), avg(tent), avg(bottom_drag)


def metric_within_sector_ic(cs, fwd, sectors, h="3M"):
    vals = []
    for d, s in cs.items():
        if d not in fwd[h]:
            continue
        df = pd.DataFrame({"s": s, "f": fwd[h][d]}).dropna()
        if df.empty:
            continue
        sec = sectors.reindex(df.index).fillna("Unknown")
        date_ics = []
        for grp, idx in df.groupby(sec).groups.items():
            sub = df.loc[idx]
            if len(sub) < MIN_SECTOR_OBS:
                continue
            v = spearman_ic(sub["s"], sub["f"])
            if v is not None:
                date_ics.append(v)
        if date_ics:
            vals.append(float(np.mean(date_ics)))
    return float(np.mean(vals)) if vals else np.nan


def metric_fm_slope_t(cs, fwd, h="3M"):
    slopes = []
    for d, s in cs.items():
        if d not in fwd[h]:
            continue
        df = pd.DataFrame({"s": s, "f": fwd[h][d]}).dropna()
        if len(df) < MIN_OBS:
            continue
        z = (df["s"] - df["s"].mean()) / df["s"].std(ddof=0)
        slope, _intercept, _r, _p, _se = stats.linregress(z, df["f"])
        slopes.append(slope)
    slopes = np.array(slopes)
    if len(slopes) < 3:
        return np.nan, np.nan
    mean_slope = float(slopes.mean())
    t_stat = float(mean_slope / (slopes.std(ddof=1) / np.sqrt(len(slopes))))
    return mean_slope, t_stat


def metric_score_stability(cs, h_unused=None):
    dates = sorted(cs)
    rhos = []
    for d0, d1 in zip(dates[:-1], dates[1:]):
        df = pd.DataFrame({"a": cs[d0], "b": cs[d1]}).dropna()
        if len(df) < MIN_OBS:
            continue
        rhos.append(df["a"].corr(df["b"], method="spearman"))
    return float(np.mean(rhos)) if rhos else np.nan


def scorecard(comp=None, universe=None, methods=None):
    if comp is None:
        comp, universe = load_comp()
    methods = methods or METHODS
    db = get_db()
    sectors = dl.global_sectors(db)
    all_dates = sorted(set().union(*[comp[m].keys() for m in methods]))
    matrix_full = load_price_matrix(start="2015-01-01", end="2026-07-31")
    fwd = panel_forward_returns(matrix_full, all_dates)

    rows = []
    for m in methods:
        cs = comp[m]
        s3, s6 = ic_series(cs, fwd, "3M"), ic_series(cs, fwd, "6M")
        blend = pd.concat([s3, s6], axis=1).mean(axis=1)
        q5q1, mono, tent, bottom_drag = metric_q5q1_mono_tent_bottom(cs, fwd, "3M")
        wsic = metric_within_sector_ic(cs, fwd, sectors, "3M")
        fm_slope, fm_t = metric_fm_slope_t(cs, fwd, "3M")
        stability = metric_score_stability(cs)
        rows.append({
            "method": m,
            "1_ic_blend": round(float(blend.mean()), 4),
            "2_ir_blend": round(float(blend.mean() / blend.std(ddof=1)), 3),
            "3_hit_rate": round(float((blend > 0).mean()), 3),
            "4_q5q1_3M": round(q5q1, 4),
            "5_monotonicity": round(mono, 3),
            "6_tent_asymmetry": round(tent, 4),
            "7_within_sector_ic": round(wsic, 4),
            "8_fm_slope_t": round(fm_t, 2),
            "8b_fm_slope": round(fm_slope, 5),
            "9_score_stability": round(stability, 3),
            "10_bottom_decile_drag": round(bottom_drag, 4),
        })
    return pd.DataFrame(rows).set_index("method")


def main():
    df = scorecard()
    df.to_csv(OUT / "factor_scorecard.csv")
    print("=== Factor scorecard (78mo OOS, 2020-2026, portfolio-agnostic) ===\n")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(df.T.to_string())
    print(f"\nwrote {OUT}/factor_scorecard.csv")


if __name__ == "__main__":
    main()
