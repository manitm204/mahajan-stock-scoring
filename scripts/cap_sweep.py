"""Effective-cap robustness sweep for method B (pre-registered addendum 2026-08-05).

Same fully-PIT pipeline as scripts/full_pit_backtest.py — every 6 months re-derive
subfactor selection (production selector rules) + parent weights from the trailing
5 years with completed-horizon PIT truncation — but method B's effective-exposure
cap is swept over {0.15, 0.20, 0.25, 0.30, 0.35, 0.40}. Method A at its incumbent
nominal cap 0.25 is the fixed baseline. Identical portfolio construction
(top 25%, cap5, cap_match, monthly, 10 bps) vs SPY/QQQ.

Read criteria are pre-registered (PREREGISTRATION addendum): we test whether B's
edge is flat across caps (robust) or peaked at 0.25 (lucky); the argmax cap is
explicitly NOT adopted.

Outputs to output/crowding/incremental_weights/:
  cap_sweep.csv          — metrics per book/benchmark
  cap_sweep_equity.csv   — monthly net equity curves

Usage: python scripts/cap_sweep.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from backtesting import data_loader as dl
from data.db import get_db
from factors.utils import sector_percentile
from research.ablation.data import AblationData, _pit_caps
from research.ablation.engine import AblationConfig, simulate_config, benchmark_row
from research.forward_returns import realize_delistings
from research.panel import ScorePanel
from research.parent_selection import run_selection
from research.subfactor_expansion.panel import load_cached_panel
from research.analyst_deep_dive.common import (
    load_price_matrix, panel_forward_returns, spearman_ic,
)
from scripts.crowding_diagnostics import parent_score, normalize
from scripts.incremental_weight_study import PANEL_PKL, OUT, water_fill, combined_scores
from scripts.full_pit_backtest import DERIV_POINTS, WINDOW_YEARS, APPLY_MONTHS, OOS_END, H_MONTHS

B_CAPS = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
A_CAP = 0.25


def weights_A(ic, ir, cap):
    return pd.Series(water_fill(combined_scores(ic, ir).to_dict(), cap))[ic.index]


def weights_B(ic, ir, C, cap):
    s = combined_scores(ic, ir)
    if s.sum() <= 0:
        return weights_A(ic, ir, cap)
    w = (s / s.sum()).copy()
    for _ in range(50):
        eff = C.values @ w.values
        if eff.max() <= cap + 0.005:
            break
        over = eff > cap + 1e-12
        w[over] = w[over] * (cap / eff[over])
        w = w / w.sum()
    return w


def main():
    db = get_db()
    panel = load_cached_panel(PANEL_PKL)
    all_dates = panel.rebal_dates
    matrix_full = load_price_matrix(start="2015-01-01", end="2026-07-31")
    fwd = panel_forward_returns(matrix_full, all_dates)
    sectors = dl.global_sectors(db)

    books = ["A_0.25"] + [f"B_{c:.2f}" for c in B_CAPS]
    comp = {b: {} for b in books}
    for t in DERIV_POINTS:
        t_ts = pd.Timestamp(t)
        w0 = (t_ts - pd.DateOffset(years=WINDOW_YEARS)).strftime("%Y-%m-%d")
        window = [d for d in all_dates if w0 <= d < t]
        apply_end = (t_ts + pd.DateOffset(months=APPLY_MONTHS)).strftime("%Y-%m-%d")
        apply_dates = [d for d in all_dates if t <= d < apply_end and d <= OOS_END]
        if not apply_dates:
            continue
        fwd_pit = {h: {d: fwd[h][d] for d in window if d in fwd[h]
                       and (pd.Timestamp(d) + pd.DateOffset(months=n)) <= t_ts}
                   for h, n in H_MONTHS.items()}
        sp = ScorePanel(rebal_dates=window,
                        scores={d: panel.scores[d] for d in window},
                        parent_keys=[p for p, s in panel.candidates_by_parent.items() if s],
                        sub_by_parent={p: list(s) for p, s in
                                       panel.candidates_by_parent.items() if s},
                        universe=list(panel.universe))
        sel = {res.parent: res.weights for res in run_selection(sp, fwd_pit)
               if res.selected}
        parents = list(sel)
        P_by = {d: pd.DataFrame({p: parent_score(panel.scores[d], sel[p])
                                 for p in parents}) for d in window + apply_dates}
        Pn_by = {d: P_by[d].apply(normalize) for d in P_by}
        blend_rows = {}
        for d in window:
            vals = []
            for h in ("3M", "6M"):
                if d in fwd_pit[h]:
                    ics = {p: spearman_ic(P_by[d][p], fwd_pit[h][d]) for p in parents}
                    vals.append(pd.Series(ics, dtype=float))
            if vals:
                blend_rows[d] = pd.concat(vals, axis=1).mean(axis=1)
        blend = pd.DataFrame(blend_rows).T
        ic_mean, ic_ir = blend.mean()[parents], (blend.mean() / blend.std(ddof=1))[parents]
        C = (pd.concat([Pn_by[d].corr(method="pearson") for d in window])
             .groupby(level=0, sort=False).mean().loc[parents, parents])

        wv = {"A_0.25": weights_A(ic_mean, ic_ir, A_CAP)}
        for c in B_CAPS:
            wv[f"B_{c:.2f}"] = weights_B(ic_mean, ic_ir, C, c)
        print(f"{t}: {len(window)}d window, {len(apply_dates)}m apply", flush=True)

        for d in apply_dates:
            sec = sectors.reindex(Pn_by[d].index).fillna("Unknown")
            for b, w in wv.items():
                if w.sum() <= 0:
                    continue
                raw = Pn_by[d].mul(w).sum(axis=1) / w.sum()
                comp[b][d] = sector_percentile(raw, sec, higher_is_better=True, min_obs=5)

    oos_dates = sorted(comp["A_0.25"])
    matrix = realize_delistings(
        dl.load_price_matrix(db, panel.universe, "2019-01-01", "2026-07-31"))
    rebal_dates = [d for d in oos_dates if d in matrix.index]
    caps = _pit_caps(db, matrix, rebal_dates)
    data = AblationData(matrix=matrix, sectors=sectors, run=None,
                        rebal_dates=rebal_dates, caps=caps,
                        vol=pd.DataFrame(index=rebal_dates), vix=pd.Series(dtype=float),
                        parent_ranks={}, panel=None)

    rows, curves = [], {}
    for b in books:
        cfg = AblationConfig(name=b, top_pct=0.25, weighting="cap5",
                             sector="cap_match", vix_tilt=False, hold_months=1)
        row = simulate_config(data, cfg, scores=comp[b], keep_series=True)
        curves[b] = (1.0 + row.pop("_returns")).cumprod()
        spy_r, qqq_r = row.pop("_spy"), row.pop("_qqq")
        row.pop("_gross", None)
        if "SPY" not in curves:
            curves["SPY"] = (1.0 + spy_r.dropna()).cumprod()
            curves["QQQ"] = (1.0 + qqq_r.dropna()).cumprod()
        rows.append(row)
    for bench in ("SPY", "QQQ"):
        rows.append(benchmark_row(data, bench))

    cols = ["config", "net_cagr", "sharpe", "sortino", "calmar", "max_dd",
            "beta", "alpha", "alpha_t", "ex_spy", "ex_qqq", "ir",
            "turnover", "avg_names", "eff_n"]
    out = pd.DataFrame(rows)
    out = out[[c for c in cols if c in out.columns]]
    out.to_csv(OUT / "cap_sweep.csv", index=False)
    pd.DataFrame(curves).to_csv(OUT / "cap_sweep_equity.csv")
    with pd.option_context("display.width", 200, "display.float_format",
                           lambda v: f"{v:.3f}"):
        print(out.to_string(index=False))
    print(f"\nwrote {OUT}/cap_sweep.csv + cap_sweep_equity.csv")


if __name__ == "__main__":
    main()
