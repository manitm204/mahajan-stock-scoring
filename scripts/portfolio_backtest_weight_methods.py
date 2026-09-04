"""Portfolio backtest of the weight-method study: A vs B vs D vs SPY/QQQ.

Takes the walk-forward weight vectors from the pre-registered incremental-weight
study (output/crowding/incremental_weights/weights_by_year.csv), rebuilds each
method's composite at the 2021-01→2026-06 OOS monthly rebalances, and runs each
through the ablation engine's costed portfolio pipeline with the finalist
construction: top 25% by composite, PIT-cap-weighted with a 5% position cap
(cap5), sector allocation matched to the cap-weighted universe (cap_match),
monthly rebalance, 10 bps per side. VIX tilt is deliberately OFF so the only
difference between the three books is the parent-weight method under test.

Benchmarks: SPY and QQQ buy-and-hold on the identical grid (costless).
Metrics: net CAGR, Sharpe, Sortino, Calmar, max drawdown, beta/alpha vs SPY,
excess CAGR vs both benchmarks, turnover.

Outputs to output/crowding/incremental_weights/:
  portfolio_backtest.csv   — one row per book/benchmark
  portfolio_equity.csv     — monthly equity curves (net of costs)

Usage: python scripts/portfolio_backtest_weight_methods.py
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
from factors.parent_selection_v4 import SELECTED_SUBS
from research.ablation.data import AblationData, _pit_caps
from research.ablation.engine import AblationConfig, simulate_config, benchmark_row
from research.forward_returns import realize_delistings
from research.subfactor_expansion.panel import load_cached_panel
from factors.utils import sector_percentile
from scripts.crowding_diagnostics import parent_score, normalize
from scripts.incremental_weight_study import PANEL_PKL, OUT, PARENTS

METHODS = ["A", "B", "D"]
OOS_START, OOS_END = "2021-01-01", "2026-06-30"


def main():
    db = get_db()
    panel = load_cached_panel(PANEL_PKL)
    dates = [d for d in panel.rebal_dates if OOS_START <= d <= OOS_END]
    print(f"{len(dates)} OOS rebalances {dates[0]}..{dates[-1]}")

    wdf = pd.read_csv(OUT / "weights_by_year.csv")
    wvec = {m: {str(y): pd.Series({p: float(r[p]) for p in PARENTS})
                for y, r in ((row.year, row) for _, row in
                             wdf[wdf.method == m].iterrows())}
            for m in METHODS}

    sectors = dl.global_sectors(db)
    comp = {m: {} for m in METHODS}
    for d in dates:
        frame = panel.scores[d]
        P = pd.DataFrame({p: parent_score(frame, SELECTED_SUBS[p]) for p in PARENTS})
        Pn = P.apply(normalize)
        sec = sectors.reindex(Pn.index).fillna("Unknown")
        for m in METHODS:
            w = wvec[m][d[:4]]
            raw = Pn.mul(w).sum(axis=1) / w.sum()
            comp[m][d] = sector_percentile(raw, sec, higher_is_better=True, min_obs=5)

    matrix = realize_delistings(
        dl.load_price_matrix(db, panel.universe, "2019-01-01", "2026-07-31"))
    rebal_dates = [d for d in dates if d in matrix.index]
    caps = _pit_caps(db, matrix, rebal_dates)
    data = AblationData(matrix=matrix, sectors=sectors, run=None,
                        rebal_dates=rebal_dates, caps=caps,
                        vol=pd.DataFrame(index=rebal_dates), vix=pd.Series(dtype=float),
                        parent_ranks={}, panel=None)

    rows, curves = [], {}
    for m in METHODS:
        cfg = AblationConfig(name=f"method_{m}", top_pct=0.25, weighting="cap5",
                             sector="cap_match", vix_tilt=False, hold_months=1)
        row = simulate_config(data, cfg, scores=comp[m], keep_series=True)
        curves[f"method_{m}"] = (1.0 + row.pop("_returns")).cumprod()
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
    out.to_csv(OUT / "portfolio_backtest.csv", index=False)
    pd.DataFrame(curves).to_csv(OUT / "portfolio_equity.csv")
    with pd.option_context("display.width", 200, "display.float_format",
                           lambda v: f"{v:.3f}"):
        print(out.to_string(index=False))
    print(f"\nwrote {OUT}/portfolio_backtest.csv + portfolio_equity.csv")


if __name__ == "__main__":
    main()
