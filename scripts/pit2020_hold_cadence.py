"""Position-cadence study on the fully-PIT 2020-start B+tilt book.

Same derivation machinery as scripts/pit2020_vix_ablation.py (subs + weights
re-derived every 6 months from trailing 5y, method-B effective cap, production
VIX tilt), but instead of monthly position rebalancing the book is held on
slower cadences:

  monthly     — hold 1 month (the existing baseline)
  hold6       — reform every 6 months, single sleeve
  hold12      — reform every 12 months, single sleeve
  sleeves6    — 2 sleeves, 6-month holds, offset 3 months (half the book
                reforms every quarter... i.e. every 3 months one sleeve turns)
  sleeves12   — 2 sleeves, 12-month holds, offset 6 months

Sleeve mechanics: each sleeve enters at t0, reforms on its own grid, drifts
with returns between reforms (costs on traded volume vs the drifted book, same
as the ablation engine); the book is the equal average of sleeve returns, with
monthly-resolution metrics. Descriptive/research-only.

PIT composite scores are cached to output/crowding/incremental_weights/
pit2020_Btilt_scores.pkl on first run and reused after.

Outputs: hold_cadence.csv / hold_cadence_yearly.csv in the same directory.

Usage: python scripts/pit2020_hold_cadence.py
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from backtesting import data_loader as dl
from data.db import get_db
from factors.utils import sector_percentile
from factors.vix_tilt import apply_vix_tilt
from research.ablation.data import AblationData, _pit_caps
from research.ablation.engine import (
    AblationConfig, alpha_tstat, base_weights, benchmark_row, sector_overlay,
    select_book, simulate_config,
)
from research.forward_returns import realize_delistings
from research.panel import ScorePanel
from research.parent_selection import run_selection
from research.subfactor_expansion.panel import load_cached_panel
from research.analyst_deep_dive.common import (
    load_price_matrix, panel_forward_returns, spearman_ic,
)
from research.walkforward.portfolio import performance_metrics
from scripts.crowding_diagnostics import parent_score, normalize
from scripts.incremental_weight_study import PANEL_PKL, OUT
from scripts.cap_sweep import weights_A, weights_B
from scripts.full_pit_backtest import WINDOW_YEARS, APPLY_MONTHS, OOS_END, H_MONTHS
from scripts.pit_backtest_2020 import DERIV_POINTS

CAP = 0.25
SCORES_PKL = OUT / "pit2020_Btilt_scores.pkl"


def pit_scores(db, panel) -> dict:
    """PIT B+tilt composite per rebalance date (replica of pit2020_vix_ablation)."""
    if SCORES_PKL.exists():
        with SCORES_PKL.open("rb") as fh:
            return pickle.load(fh)
    all_dates = panel.rebal_dates
    matrix_full = load_price_matrix(start="2015-01-01", end="2026-07-31")
    fwd = panel_forward_returns(matrix_full, all_dates)
    sectors = dl.global_sectors(db)
    vixdf = db.query_df(
        "SELECT date, close FROM daily_prices WHERE ticker='VIX' ORDER BY date")
    vix = pd.Series(vixdf["close"].values, index=vixdf["date"].astype(str))

    comp = {}
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
        wB = weights_B(ic_mean, ic_ir, C, CAP)
        print(f"{t}: window {len(window)}d, apply {len(apply_dates)}m", flush=True)

        for d in apply_dates:
            sec = sectors.reindex(Pn_by[d].index).fillna("Unknown")
            s = vix[vix.index <= d]
            spot = float(s.iloc[-1]) if len(s) else None
            tw_dict, _ = apply_vix_tilt(wB.to_dict(), spot)
            # reindex, not [] — the tilt's _cap_renorm drops zero-weight parents
            tw = pd.Series(tw_dict).reindex(parents).fillna(0.0)
            raw = Pn_by[d].mul(tw).sum(axis=1) / tw.sum()
            comp[d] = sector_percentile(raw, sec, higher_is_better=True, min_obs=5)

    with SCORES_PKL.open("wb") as fh:
        pickle.dump(comp, fh)
    return comp


def sleeve_book(comp, data, cfg, hold: int, offsets: list[int]):
    """Monthly-resolution book of equal sleeves; sleeve k enters at month 0 and
    reforms when (i - offset) % hold == 0; holdings drift between reforms."""
    months = data.rebal_dates
    pxm = data.matrix.loc[[d for d in months]]
    rets, turns = [], []
    for off in offsets:
        w = pd.Series(dtype=float)
        r_s, t_s = {}, {}
        for i in range(len(months) - 1):
            d, nxt = months[i], months[i + 1]
            if i == 0 or (i - off) % hold == 0:
                sc = comp[d].dropna()
                names = select_book(sc, cfg.top_pct, None, [])
                wn = base_weights(cfg, names, sc, d, data)
                wn = sector_overlay(wn, cfg, sc.index.tolist(), d, data)
                wn = wn / wn.sum()
                traded = float((wn - w.reindex(wn.index.union(w.index))
                                .fillna(0.0)).abs().sum())
            else:
                wn, traded = w, 0.0
            ret = (pxm.loc[nxt].reindex(wn.index)
                   / pxm.loc[d].reindex(wn.index) - 1.0)
            ok = ret.notna()
            wn = wn[ok] / wn[ok].sum()
            ret = ret[ok]
            g = float((wn * ret).sum())
            r_s[nxt] = g - traded * cfg.cost_bps / 1e4
            t_s[d] = 0.5 * traded
            drift = wn * (1.0 + ret)
            w = drift / drift.sum()
        rets.append(pd.Series(r_s, dtype=float).sort_index())
        turns.append(pd.Series(t_s, dtype=float).sort_index())
    pr = pd.concat(rets, axis=1).mean(axis=1)
    turn = pd.concat(turns, axis=1).mean(axis=1)
    return pr, turn


def metrics_row(name, pr, turn, pxm):
    spy = (pxm[dl.SPY] / pxm[dl.SPY].shift(1) - 1.0).reindex(pr.index)
    qqq = (pxm[dl.QQQ] / pxm[dl.QQQ].shift(1) - 1.0).reindex(pr.index)
    m = performance_metrics(pr, 1, turn, {dl.SPY: spy, dl.QQQ: qqq})
    dd = m.get("max_drawdown")
    return {"config": name, "net_cagr": m["cagr"], "sharpe": m["sharpe"],
            "sortino": m["sortino"],
            "calmar": m["cagr"] / abs(dd) if dd and dd == dd else np.nan,
            "max_dd": dd, "beta": m.get("spy_beta"), "alpha": m.get("spy_alpha"),
            "alpha_t": alpha_tstat(pr, spy), "ex_spy": m.get("spy_excess_cagr"),
            "ex_qqq": m.get("qqq_excess_cagr"), "ir": m.get("spy_ir"),
            "turnover": m["avg_turnover"]}, pr


def main():
    db = get_db()
    panel = load_cached_panel(PANEL_PKL)
    comp = pit_scores(db, panel)
    sectors = dl.global_sectors(db)
    oos_dates = sorted(comp)
    matrix = realize_delistings(
        dl.load_price_matrix(db, panel.universe, "2018-01-01", "2026-07-31"))
    rebal_dates = [d for d in oos_dates if d in matrix.index]
    caps = _pit_caps(db, matrix, rebal_dates)
    data = AblationData(matrix=matrix, sectors=sectors, run=None,
                        rebal_dates=rebal_dates, caps=caps,
                        vol=pd.DataFrame(index=rebal_dates), vix=pd.Series(dtype=float),
                        parent_ranks={}, panel=None)
    pxm = matrix.loc[rebal_dates]
    base = AblationConfig(name="base", top_pct=0.25, weighting="cap5",
                          sector="cap_match", vix_tilt=False, hold_months=1)

    rows, series = [], {}
    for name, hold in (("monthly", 1), ("hold6", 6), ("hold12", 12)):
        cfg = AblationConfig(name=name, top_pct=0.25, weighting="cap5",
                             sector="cap_match", vix_tilt=False, hold_months=hold)
        row = simulate_config(data, cfg, scores=comp, keep_series=True)
        series[name] = row.pop("_returns")
        for k in ("_spy", "_qqq", "_gross"):
            row.pop(k, None)
        rows.append({k: row.get(k) for k in
                     ("config", "net_cagr", "sharpe", "sortino", "calmar", "max_dd",
                      "beta", "alpha", "alpha_t", "ex_spy", "ex_qqq", "ir", "turnover")})
    for name, hold, offsets in (("sleeves6", 6, [0, 3]), ("sleeves12", 12, [0, 6])):
        pr, turn = sleeve_book(comp, data, base, hold, offsets)
        row, pr = metrics_row(name, pr, turn, pxm)
        series[name] = pr
        rows.append(row)
    for bench in ("SPY", "QQQ"):
        b = benchmark_row(data, bench)
        rows.append({k: b.get(k) for k in
                     ("config", "net_cagr", "sharpe", "sortino", "calmar", "max_dd",
                      "beta", "alpha", "alpha_t", "ex_spy", "ex_qqq", "ir", "turnover")})

    out = pd.DataFrame(rows)
    out.to_csv(OUT / "hold_cadence.csv", index=False)
    yearly = {}
    for name, pr in series.items():
        yearly[name] = (1.0 + pr).groupby(pr.index.str[:4]).prod() - 1.0
    ydf = pd.DataFrame(yearly)
    ydf.to_csv(OUT / "hold_cadence_yearly.csv")
    with pd.option_context("display.width", 200, "display.float_format",
                           lambda v: f"{v:.3f}"):
        print(out.to_string(index=False))
        print("\nper-year net returns (period-grid for single-sleeve holds):")
        print(ydf.to_string())
    print(f"\nwrote {OUT}/hold_cadence.csv + hold_cadence_yearly.csv")


if __name__ == "__main__":
    main()
