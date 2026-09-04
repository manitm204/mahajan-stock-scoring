"""VIX-tilt ablation on the fully-PIT 2020-start backtest (prereg addendum 2026-08-07b).

Four books through the identical pipeline: A, A+tilt, B, B+tilt. The tilt is the
production literature rule (factors/vix_tilt.py), applied at every monthly
rebalance to that window's frozen derivation vector using the PIT VIX spot
(latest close on/before the rebalance date). Question: does the tilt still add
value now that method B's effective cap trims the same momentum weight the tilt
manipulates?

Outputs to output/crowding/incremental_weights/:
  vix_ablation.csv / vix_ablation_equity.csv

Usage: python scripts/pit2020_vix_ablation.py
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
from factors.vix_tilt import apply_vix_tilt
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
from scripts.incremental_weight_study import PANEL_PKL, OUT
from scripts.cap_sweep import weights_A, weights_B
from scripts.full_pit_backtest import WINDOW_YEARS, APPLY_MONTHS, OOS_END, H_MONTHS
from scripts.pit_backtest_2020 import DERIV_POINTS

CAP = 0.25


def main():
    db = get_db()
    panel = load_cached_panel(PANEL_PKL)
    all_dates = panel.rebal_dates
    matrix_full = load_price_matrix(start="2015-01-01", end="2026-07-31")
    fwd = panel_forward_returns(matrix_full, all_dates)
    sectors = dl.global_sectors(db)
    vixdf = db.query_df(
        "SELECT date, close FROM daily_prices WHERE ticker='VIX' ORDER BY date")
    vix = pd.Series(vixdf["close"].values, index=vixdf["date"].astype(str))

    def vix_spot(d: str) -> float | None:
        s = vix[vix.index <= d]
        return float(s.iloc[-1]) if len(s) else None

    books = ["A", "A_tilt", "B", "B_tilt"]
    comp = {b: {} for b in books}
    tilt_log = []
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

        base = {"A": weights_A(ic_mean, ic_ir, CAP),
                "B": weights_B(ic_mean, ic_ir, C, CAP)}
        print(f"{t}: window {len(window)}d, apply {len(apply_dates)}m", flush=True)

        for d in apply_dates:
            sec = sectors.reindex(Pn_by[d].index).fillna("Unknown")
            spot = vix_spot(d)
            for m in ("A", "B"):
                w = base[m]
                raw = Pn_by[d].mul(w).sum(axis=1) / w.sum()
                comp[m][d] = sector_percentile(raw, sec, higher_is_better=True, min_obs=5)
                tw_dict, mom_m = apply_vix_tilt(w.to_dict(), spot)
                # reindex, not [] — the tilt's _cap_renorm drops zero-weight parents
                tw = pd.Series(tw_dict).reindex(parents).fillna(0.0)
                raw_t = Pn_by[d].mul(tw).sum(axis=1) / tw.sum()
                comp[f"{m}_tilt"][d] = sector_percentile(raw_t, sec,
                                                         higher_is_better=True, min_obs=5)
                if m == "B":
                    tilt_log.append({"date": d, "vix": spot, "mom_mult": mom_m})

    n_active = sum(1 for r in tilt_log if abs(r["mom_mult"] - 1.0) > 1e-9)
    print(f"tilt active on {n_active}/{len(tilt_log)} rebalances")
    pd.DataFrame(tilt_log).to_csv(OUT / "vix_ablation_tiltlog.csv", index=False)

    oos_dates = sorted(comp["A"])
    matrix = realize_delistings(
        dl.load_price_matrix(db, panel.universe, "2018-01-01", "2026-07-31"))
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
    out.to_csv(OUT / "vix_ablation.csv", index=False)
    pd.DataFrame(curves).to_csv(OUT / "vix_ablation_equity.csv")
    with pd.option_context("display.width", 200, "display.float_format",
                           lambda v: f"{v:.3f}"):
        print(out.to_string(index=False))
    print(f"\nwrote {OUT}/vix_ablation.csv + vix_ablation_equity.csv")


if __name__ == "__main__":
    main()
