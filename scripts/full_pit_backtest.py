"""Fully point-in-time walk-forward portfolio backtest: A vs B vs D vs SPY/QQQ.

Addresses the residual contamination in scripts/portfolio_backtest_weight_methods.py
(user-directed 2026-08-04): that run held SELECTED_SUBS fixed at today's production
set, which was chosen on 2023-26 data. Here NOTHING from the future leaks into any
book decision:

Every 6 months (2021-01, 2021-07, … 2026-01), from a trailing 5-year window whose
forward returns have fully completed by the derivation date:
  1. re-run the production sub-factor selector (research/parent_selection.py rules
     verbatim: 0.50 IC + 0.25 IR + 0.15 spread + 0.05 hit + 0.05 coverage percentile
     ranks, greedy R²<0.60, ≤3 subs, weights ∝ IC capped 50%) over the full
     candidate library → that era's SELECTED_SUBS;
  2. build parent scores from those subs, compute parent IC stats + correlation
     matrix on the same window;
  3. derive parent weights under each method — A (standalone IC/IR, nominal cap
     0.25), B (effective-exposure cap 0.25), D (shrunk Markowitz-on-IC, λ=0.5);
  4. score the NEXT 6 months of monthly rebalances with that frozen recipe.

The stitched 2021-01→2026-06 composites then run through the ablation engine's
costed pipeline (top 25%, cap5 weighting, cap_match sector overlay, monthly,
10 bps/side) against costless SPY/QQQ buy-and-hold.

Residual limitation (cannot be removed by construction): the candidate *library
definitions* were authored 2026 with recent data in view. Selection, weights and
books are PIT; the library's existence is not.

Outputs to output/crowding/incremental_weights/:
  pit_backtest.csv        — metrics per book/benchmark
  pit_equity.csv          — monthly net equity curves
  pit_selections.csv      — the subs each window's selector picked (per parent)
  pit_parent_weights.csv  — A/B/D parent weights per window

Usage: python scripts/full_pit_backtest.py
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
from research.analyst_deep_dive.common import load_price_matrix, panel_forward_returns, spearman_ic
from scripts.crowding_diagnostics import parent_score, normalize
from scripts.incremental_weight_study import PANEL_PKL, OUT, water_fill, combined_scores, CAP, LAM

H_MONTHS = {"3M": 3, "6M": 6}
DERIV_POINTS = [f"{y}-{m:02d}-01" for y in range(2021, 2027) for m in (1, 7)
                if not (y == 2026 and m == 7)]
WINDOW_YEARS = 5
APPLY_MONTHS = 6
OOS_END = "2026-06-30"


def method_weights(ic: pd.Series, ir: pd.Series, C: pd.DataFrame) -> dict[str, pd.Series]:
    parents = list(ic.index)
    out = {}
    out["A"] = pd.Series(water_fill(combined_scores(ic, ir).to_dict(), CAP))[parents]
    s = combined_scores(ic, ir)
    if s.sum() > 0:
        w = (s / s.sum()).copy()
        for _ in range(50):
            eff = C.values @ w.values
            if eff.max() <= CAP + 0.005:
                break
            over = eff > CAP + 1e-12
            w[over] = w[over] * (CAP / eff[over])
            w = w / w.sum()
        out["B"] = w
    else:
        out["B"] = out["A"]
    M = LAM * np.eye(len(parents)) + (1 - LAM) * C.values
    wd = np.linalg.solve(M, ic.clip(lower=0.0).values)
    out["D"] = pd.Series(water_fill(dict(zip(parents, wd)), CAP))[parents]
    return out


def main():
    db = get_db()
    panel = load_cached_panel(PANEL_PKL)
    all_dates = panel.rebal_dates
    print(f"panel {len(all_dates)} dates {all_dates[0]}..{all_dates[-1]}")

    matrix_full = load_price_matrix(start="2015-01-01", end="2026-07-31")
    fwd = panel_forward_returns(matrix_full, all_dates)
    sectors = dl.global_sectors(db)

    comp = {m: {} for m in ("A", "B", "D")}
    sel_rows, w_rows = [], []
    for t in DERIV_POINTS:
        t_ts = pd.Timestamp(t)
        w0 = (t_ts - pd.DateOffset(years=WINDOW_YEARS)).strftime("%Y-%m-%d")
        window = [d for d in all_dates if w0 <= d < t]
        apply_end = (t_ts + pd.DateOffset(months=APPLY_MONTHS)).strftime("%Y-%m-%d")
        apply_dates = [d for d in all_dates if t <= d < apply_end and d <= OOS_END]
        if not apply_dates:
            continue

        # PIT truncation: a formation date's h-month return is usable only if it
        # completed before the derivation date.
        fwd_pit = {h: {d: fwd[h][d] for d in window if d in fwd[h]
                       and (pd.Timestamp(d) + pd.DateOffset(months=n)) <= t_ts}
                   for h, n in H_MONTHS.items()}

        sp = ScorePanel(rebal_dates=window,
                        scores={d: panel.scores[d] for d in window},
                        parent_keys=[p for p, s in panel.candidates_by_parent.items() if s],
                        sub_by_parent={p: list(s) for p, s in
                                       panel.candidates_by_parent.items() if s},
                        universe=list(panel.universe))
        results = run_selection(sp, fwd_pit)
        sel = {res.parent: res.weights for res in results if res.selected}
        for res in results:
            sel_rows.append({"window": t, "parent": res.parent,
                             "formula": res.formula, "flag": res.signal_flag,
                             "stop": res.stop_reason})

        parents = list(sel)
        P_by = {d: pd.DataFrame({p: parent_score(panel.scores[d], sel[p])
                                 for p in parents}) for d in window + apply_dates}
        Pn_by = {d: P_by[d].apply(normalize) for d in P_by}

        # Parent blend-IC series on the PIT-truncated window.
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
        ic_mean = blend.mean()
        ic_ir = blend.mean() / blend.std(ddof=1)
        C = (pd.concat([Pn_by[d].corr(method="pearson") for d in window])
             .groupby(level=0, sort=False).mean().loc[parents, parents])

        wv = method_weights(ic_mean[parents], ic_ir[parents], C)
        for m, w in wv.items():
            w_rows.append({"window": t, "method": m,
                           **{p: round(float(w.get(p, 0.0)), 3) for p in parents}})
        print(f"{t}: window {len(window)}d, ic-dates {len(blend)}, "
              f"apply {len(apply_dates)}m, parents {len(parents)}", flush=True)

        for d in apply_dates:
            sec = sectors.reindex(Pn_by[d].index).fillna("Unknown")
            for m, w in wv.items():
                if w.sum() <= 0:
                    continue
                raw = Pn_by[d].mul(w).sum(axis=1) / w.sum()
                comp[m][d] = sector_percentile(raw, sec, higher_is_better=True, min_obs=5)

    pd.DataFrame(sel_rows).to_csv(OUT / "pit_selections.csv", index=False)
    pd.DataFrame(w_rows).to_csv(OUT / "pit_parent_weights.csv", index=False)

    # ---- portfolio simulation (identical engine + construction as before) ----
    oos_dates = sorted(comp["A"])
    matrix = realize_delistings(
        dl.load_price_matrix(db, panel.universe, "2019-01-01", "2026-07-31"))
    rebal_dates = [d for d in oos_dates if d in matrix.index]
    caps = _pit_caps(db, matrix, rebal_dates)
    data = AblationData(matrix=matrix, sectors=sectors, run=None,
                        rebal_dates=rebal_dates, caps=caps,
                        vol=pd.DataFrame(index=rebal_dates), vix=pd.Series(dtype=float),
                        parent_ranks={}, panel=None)

    rows, curves = [], {}
    for m in ("A", "B", "D"):
        cfg = AblationConfig(name=f"pit_{m}", top_pct=0.25, weighting="cap5",
                             sector="cap_match", vix_tilt=False, hold_months=1)
        row = simulate_config(data, cfg, scores=comp[m], keep_series=True)
        curves[f"pit_{m}"] = (1.0 + row.pop("_returns")).cumprod()
        spy_r, qqq_r = row.pop("_spy"), row.pop("_qqq")
        row.pop("_gross", None)
        if "SPY" not in curves:
            curves["SPY"] = (1.0 + spy_r.dropna()).cumprod()
            curves["QQQ"] = (1.0 + qqq_r.dropna()).cumprod()
        rows.append(row)
    for b in ("SPY", "QQQ"):
        rows.append(benchmark_row(data, b))

    cols = ["config", "net_cagr", "sharpe", "sortino", "calmar", "max_dd",
            "beta", "alpha", "alpha_t", "ex_spy", "ex_qqq", "ir",
            "turnover", "avg_names", "eff_n"]
    out = pd.DataFrame(rows)
    out = out[[c for c in cols if c in out.columns]]
    out.to_csv(OUT / "pit_backtest.csv", index=False)
    pd.DataFrame(curves).to_csv(OUT / "pit_equity.csv")
    with pd.option_context("display.width", 200, "display.float_format",
                           lambda v: f"{v:.3f}"):
        print(out.to_string(index=False))
    print(f"\nwrote {OUT}/pit_backtest.csv + pit_equity.csv + pit_selections.csv "
          f"+ pit_parent_weights.csv")


if __name__ == "__main__":
    main()
