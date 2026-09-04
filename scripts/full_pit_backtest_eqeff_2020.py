"""Fully-PIT backtest extended to a 2020 start, + gross-vs-net cost decomposition:
A vs E vs EQ vs EQEFF.

Two follow-ups to scripts/full_pit_backtest_eqeff.py, requested after the
62-month (2021-2026) result showed E beats EQEFF on CAGR/Sharpe with only 87%
bootstrap confidence (CI touched zero) while EQEFF's walk-forward IC edge over
the incumbent was decisive (CI90 didn't touch zero). Two honest ways to find
out whether that's really noise or a real turnover-cost effect:

  1. Extend derivation points back to 2020-01 (COVID-inclusive, 78mo instead
     of 62mo) — same protocol as scripts/pit_backtest_2020.py used for the
     A/B/D comparison — for more independent-ish information in the paired
     bootstrap.
  2. Cache the derived composite scores (comp dict) once, then simulate the
     SAME portfolios at cost_bps=10 (realistic) and cost_bps=0 (gross) — if
     EQEFF's underperformance is a turnover-cost artifact, it should close or
     flip at cost_bps=0; if it doesn't, the extra IC isn't usable regardless
     of cost model.

Outputs to output/crowding/weight_config_study/:
  pit_backtest_eqeff_2020.csv / pit_equity_eqeff_2020.csv   (net, 10bps)
  pit_backtest_eqeff_2020_gross.csv                          (gross, 0bps)
  pit_selections_eqeff_2020.csv / pit_parent_weights_eqeff_2020.csv
  comp_eqeff_2020.pkl  (cached composite scores per method/date, for reuse)

Usage: python scripts/full_pit_backtest_eqeff_2020.py
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

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
from scripts.incremental_weight_study import PANEL_PKL
from scripts.full_pit_backtest_eqeff import method_weights, WINDOW_YEARS, APPLY_MONTHS, OOS_END, H_MONTHS

OUT = REPO / "output" / "crowding" / "weight_config_study"
OUT.mkdir(parents=True, exist_ok=True)

DERIV_POINTS = [f"{y}-{m:02d}-01" for y in range(2020, 2027) for m in (1, 7)
                if not (y == 2026 and m == 7)]
METHODS = ("A", "E", "EQ", "EQEFF")


def derive():
    db = get_db()
    panel = load_cached_panel(PANEL_PKL)
    all_dates = panel.rebal_dates
    print(f"panel {len(all_dates)} dates {all_dates[0]}..{all_dates[-1]}")

    matrix_full = load_price_matrix(start="2015-01-01", end="2026-07-31")
    fwd = panel_forward_returns(matrix_full, all_dates)
    sectors = dl.global_sectors(db)

    comp = {m: {} for m in METHODS}
    sel_rows, w_rows = [], []
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

    pd.DataFrame(sel_rows).to_csv(OUT / "pit_selections_eqeff_2020.csv", index=False)
    pd.DataFrame(w_rows).to_csv(OUT / "pit_parent_weights_eqeff_2020.csv", index=False)
    with open(OUT / "comp_eqeff_2020.pkl", "wb") as f:
        pickle.dump({"comp": comp, "universe": list(panel.universe)}, f)
    return comp, list(panel.universe)


def simulate(comp, universe, cost_bps: float, tag: str):
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

    rows, curves = [], {}
    for m in METHODS:
        cfg = AblationConfig(name=f"pit_{m}", top_pct=0.25, weighting="cap5",
                             sector="cap_match", vix_tilt=False, hold_months=1,
                             cost_bps=cost_bps)
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
    out.to_csv(OUT / f"pit_backtest_eqeff_2020{tag}.csv", index=False)
    if tag == "":
        pd.DataFrame(curves).to_csv(OUT / "pit_equity_eqeff_2020.csv")
    with pd.option_context("display.width", 200, "display.float_format",
                           lambda v: f"{v:.3f}"):
        print(f"\n=== cost_bps={cost_bps} ===")
        print(out.to_string(index=False))
    return out


def main():
    cached = OUT / "comp_eqeff_2020.pkl"
    if cached.exists():
        print(f"reusing cached derivation: {cached}")
        with open(cached, "rb") as f:
            d = pickle.load(f)
        comp, universe = d["comp"], d["universe"]
    else:
        comp, universe = derive()

    simulate(comp, universe, cost_bps=10.0, tag="")
    simulate(comp, universe, cost_bps=0.0, tag="_gross")
    print(f"\nwrote {OUT}/pit_backtest_eqeff_2020*.csv")


if __name__ == "__main__":
    main()
