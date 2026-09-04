"""Backtest the CURRENT production stack, held static, vs SPY/QQQ.

Exactly what is live right now: parent_selection_v4 SELECTED_SUBS (incl. the
2026-08-07 revisions re-ratification) + B-trimmed V4_PARENT_WEIGHTS + the
production VIX tilt, replayed over the 2020-01..2026-06 panel through the same
ablation pipeline as the crowding-study backtests (top25/cap5/cap_match,
monthly, 10bps). NOT point-in-time: today's subs/weights were derived on data
inside this window, so absolute levels are optimistic — the honest PIT
counterpart is scripts/pit2020_vix_ablation.py (B_tilt book).

Outputs to output/crowding/incremental_weights/: production_static.csv
(+ _equity.csv, _yearly.csv).

Usage: python scripts/production_static_backtest.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from backtesting import data_loader as dl
from data.db import get_db
from factors.parent_selection_v4 import SELECTED_SUBS, V4_PARENT_WEIGHTS
from factors.utils import sector_percentile
from factors.vix_tilt import apply_vix_tilt
from research.ablation.data import AblationData, _pit_caps
from research.ablation.engine import AblationConfig, simulate_config, benchmark_row
from research.forward_returns import realize_delistings
from research.subfactor_expansion.panel import load_cached_panel
from scripts.crowding_diagnostics import parent_score, normalize
from scripts.incremental_weight_study import PANEL_PKL, OUT

START, END = "2020-01-01", "2026-06-30"


def main():
    db = get_db()
    panel = load_cached_panel(PANEL_PKL)
    dates = [d for d in panel.rebal_dates if START <= d <= END]
    sectors = dl.global_sectors(db)
    vixdf = db.query_df(
        "SELECT date, close FROM daily_prices WHERE ticker='VIX' ORDER BY date")
    vix = pd.Series(vixdf["close"].values, index=vixdf["date"].astype(str))

    parents = list(SELECTED_SUBS)
    w0 = pd.Series(V4_PARENT_WEIGHTS)[parents]
    comp = {"prod": {}, "prod_notilt": {}}
    for d in dates:
        P = pd.DataFrame({p: parent_score(panel.scores[d], SELECTED_SUBS[p])
                          for p in parents})
        Pn = P.apply(normalize)
        sec = sectors.reindex(Pn.index).fillna("Unknown")
        s = vix[vix.index <= d]
        spot = float(s.iloc[-1]) if len(s) else None
        tw_dict, _ = apply_vix_tilt(w0.to_dict(), spot)
        # reindex, not [] — the tilt's _cap_renorm drops zero-weight parents
        tw = pd.Series(tw_dict).reindex(parents).fillna(0.0)
        for name, w in (("prod", tw), ("prod_notilt", w0)):
            raw = Pn.mul(w).sum(axis=1) / w.sum()
            comp[name][d] = sector_percentile(raw, sec, higher_is_better=True,
                                              min_obs=5)

    matrix = realize_delistings(
        dl.load_price_matrix(db, panel.universe, "2018-01-01", "2026-07-31"))
    rebal_dates = [d for d in dates if d in matrix.index]
    caps = _pit_caps(db, matrix, rebal_dates)
    data = AblationData(matrix=matrix, sectors=sectors, run=None,
                        rebal_dates=rebal_dates, caps=caps,
                        vol=pd.DataFrame(index=rebal_dates),
                        vix=pd.Series(dtype=float), parent_ranks={}, panel=None)

    rows, curves, yearly = [], {}, {}
    for b in ("prod", "prod_notilt"):
        cfg = AblationConfig(name=b, top_pct=0.25, weighting="cap5",
                             sector="cap_match", vix_tilt=False, hold_months=1)
        row = simulate_config(data, cfg, scores=comp[b], keep_series=True)
        rets = row.pop("_returns")
        curves[b] = (1.0 + rets).cumprod()
        yearly[b] = (1.0 + rets).groupby(rets.index.str[:4]).prod() - 1.0
        spy_r, qqq_r = row.pop("_spy"), row.pop("_qqq")
        row.pop("_gross", None)
        if "SPY" not in curves:
            for bench, r in (("SPY", spy_r), ("QQQ", qqq_r)):
                r = r.dropna()
                curves[bench] = (1.0 + r).cumprod()
                yearly[bench] = (1.0 + r).groupby(r.index.str[:4]).prod() - 1.0
        rows.append(row)
    for bench in ("SPY", "QQQ"):
        rows.append(benchmark_row(data, bench))

    cols = ["config", "net_cagr", "sharpe", "sortino", "calmar", "max_dd",
            "beta", "alpha", "alpha_t", "ex_spy", "ex_qqq", "ir",
            "turnover", "avg_names", "eff_n"]
    out = pd.DataFrame(rows)
    out = out[[c for c in cols if c in out.columns]]
    out.to_csv(OUT / "production_static.csv", index=False)
    pd.DataFrame(curves).to_csv(OUT / "production_static_equity.csv")
    ydf = pd.DataFrame(yearly)
    ydf.to_csv(OUT / "production_static_yearly.csv")
    with pd.option_context("display.width", 200, "display.float_format",
                           lambda v: f"{v:.3f}"):
        print(out.to_string(index=False))
        print("\nper-year net returns:")
        print(ydf.to_string())
    print(f"\nwrote {OUT}/production_static*.csv")


if __name__ == "__main__":
    main()
