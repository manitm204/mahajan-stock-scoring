"""50/50 SPY/QQQ sector allocation on the fully-PIT 2020-start B+tilt book.

Companion to scripts/pit2020_qqq_sector.py: per rebalance date the sector
target is the equal blend of the cap-weighted-universe target (SPY proxy,
exactly as the engine's cap_match computes it) and the QQQ-proxy target
(current members, PIT cap-weighted — same hindsight caveat as the QQQ study).
Books: blend_monthly, blend_sleeves6, blend_sleeves12.

Outputs: blend_sector.csv / blend_sector_yearly.csv (full side-by-side table
pulled from hold_cadence.csv + qqq_sector.csv).

Usage: python scripts/pit2020_blend_sector.py
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
from research.ablation.data import AblationData, _pit_caps
import research.ablation.engine as eng
from research.ablation.engine import (
    AblationConfig, _fill_median, benchmark_row, simulate_config,
)
from research.forward_returns import realize_delistings
from research.subfactor_expansion.panel import load_cached_panel
import scripts.pit2020_hold_cadence as cad
import scripts.pit2020_qqq_sector as qqq
from scripts.incremental_weight_study import PANEL_PKL, OUT

_orig_overlay = eng.sector_overlay


def blend_overlay(w, cfg, universe_names, d, data):
    if cfg.sector != "blend_match":
        return _orig_overlay(w, cfg, universe_names, d, data)
    if w.empty:
        return w
    sec = data.sectors.reindex(w.index).fillna("Unknown")
    uni_sec = data.sectors.reindex(universe_names).fillna("Unknown")
    ucaps = _fill_median(data.caps.loc[d].reindex(universe_names))
    spy_tgt = ucaps.groupby(uni_sec).sum()
    spy_tgt = spy_tgt / spy_tgt.sum()
    qqq_tgt = qqq.QQQ_TGT[d]
    idx = spy_tgt.index.union(qqq_tgt.index)
    tgt = 0.5 * spy_tgt.reindex(idx).fillna(0.0) + 0.5 * qqq_tgt.reindex(idx).fillna(0.0)
    present = sec.unique().tolist()
    tgt = tgt.reindex(present).fillna(0.0)
    if tgt.sum() <= 0:
        return w
    tgt = tgt / tgt.sum()
    cur = w.groupby(sec).sum().reindex(present).fillna(0.0)
    scale = (tgt / cur.replace(0.0, np.nan)).fillna(0.0)
    out = w * sec.map(scale)
    return out / out.sum() if out.sum() > 0 else w


def main():
    eng.sector_overlay = blend_overlay
    cad.sector_overlay = blend_overlay

    db = get_db()
    panel = load_cached_panel(PANEL_PKL)
    comp = cad.pit_scores(db, panel)
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

    members = qqq.fetch_members()
    in_uni = [t for t in members if t in matrix.columns]
    for d in rebal_dates:
        mc = caps.loc[d].reindex(in_uni).dropna()
        msec = sectors.reindex(mc.index).fillna("Unknown")
        tgt = mc.groupby(msec).sum()
        qqq.QQQ_TGT[d] = tgt / tgt.sum()

    bcfg = AblationConfig(name="blend_monthly", top_pct=0.25, weighting="cap5",
                          sector="blend_match", vix_tilt=False, hold_months=1)
    keep = ("config", "net_cagr", "sharpe", "sortino", "calmar", "max_dd",
            "beta", "alpha", "alpha_t", "ex_spy", "ex_qqq", "ir", "turnover")
    rows, series = [], {}

    row = simulate_config(data, bcfg, scores=comp, keep_series=True)
    series["blend_monthly"] = row.pop("_returns")
    rows.append({k: row.get(k) for k in keep})
    for name, hold, offsets in (("blend_sleeves6", 6, [0, 3]),
                                ("blend_sleeves12", 12, [0, 6])):
        pr, turn = cad.sleeve_book(comp, data, bcfg, hold, offsets)
        r, pr = cad.metrics_row(name, pr, turn, pxm)
        series[name] = pr
        rows.append(r)

    prior = pd.read_csv(OUT / "qqq_sector.csv")
    for _, r in prior.iterrows():
        rows.append({k: r.get(k) for k in keep})

    out = pd.DataFrame(rows)
    out.to_csv(OUT / "blend_sector.csv", index=False)
    yearly = {n: (1.0 + p).groupby(p.index.str[:4]).prod() - 1.0
              for n, p in series.items()}
    ydf = pd.DataFrame(yearly)
    ydf.to_csv(OUT / "blend_sector_yearly.csv")
    with pd.option_context("display.width", 200, "display.float_format",
                           lambda v: f"{v:.3f}"):
        print(out.to_string(index=False))
        print("\nper-year net returns (blend books):")
        print(ydf.to_string())
    print(f"\nwrote {OUT}/blend_sector.csv + blend_sector_yearly.csv")


if __name__ == "__main__":
    main()
