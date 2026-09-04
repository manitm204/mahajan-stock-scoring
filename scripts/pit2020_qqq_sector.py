"""QQQ-style sector allocation on the fully-PIT 2020-start B+tilt book.

Same book as scripts/pit2020_hold_cadence.py (PIT-derived subs+weights every
6 months, B effective cap, VIX tilt, top-25%/cap5), but the sector overlay
matches QQQ's sector mix instead of the cap-weighted universe (SPY proxy).

QQQ targets: current QQQ constituents (FMP etf/holdings, cached to
qqq_members.json) intersected with our universe; per rebalance date the
sector targets are those members' PIT market-cap sector shares — so the mix
drifts realistically (smaller tech share in 2020). Caveat: membership is
TODAY'S list (no historical constituents endpoint), so this is a proxy for
"allocated like QQQ", not a reconstruction of the index.

Books: qqq_monthly, qqq_sleeves6 (2x6mo offset 3), qqq_sleeves12 (2x12mo
offset 6); cap_match (SPY-style) rows are read from hold_cadence.csv for the
side-by-side. Outputs: qqq_sector.csv / qqq_sector_yearly.csv.

Usage: python scripts/pit2020_qqq_sector.py
"""
from __future__ import annotations

import json
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
from research.ablation.engine import AblationConfig, benchmark_row, simulate_config
from research.forward_returns import realize_delistings
from research.subfactor_expansion.panel import load_cached_panel
import scripts.pit2020_hold_cadence as cad
from scripts.incremental_weight_study import PANEL_PKL, OUT

MEMBERS_JSON = OUT / "qqq_members.json"

_orig_overlay = eng.sector_overlay
QQQ_TGT: dict[str, pd.Series] = {}


def qqq_overlay(w, cfg, universe_names, d, data):
    if cfg.sector != "qqq_match":
        return _orig_overlay(w, cfg, universe_names, d, data)
    if w.empty:
        return w
    sec = data.sectors.reindex(w.index).fillna("Unknown")
    present = sec.unique().tolist()
    tgt = QQQ_TGT[d].reindex(present).fillna(0.0)
    if tgt.sum() <= 0:
        return w
    tgt = tgt / tgt.sum()
    cur = w.groupby(sec).sum().reindex(present).fillna(0.0)
    scale = (tgt / cur.replace(0.0, np.nan)).fillna(0.0)
    out = w * sec.map(scale)
    return out / out.sum() if out.sum() > 0 else w


def fetch_members() -> list[str]:
    if MEMBERS_JSON.exists():
        return json.loads(MEMBERS_JSON.read_text())
    from data.config import load_config
    from data.providers import FMPProvider
    cfg = load_config()
    fmp = FMPProvider(cfg.env("FMP_API_KEY"),
                      cfg.get("transcripts", "fmp_base_url",
                              default="https://financialmodelingprep.com/stable"))
    rows = fmp._get("etf/holdings", {"symbol": "QQQ"})
    members = sorted({r["asset"] for r in rows if r.get("asset")})
    MEMBERS_JSON.write_text(json.dumps(members))
    return members


def main():
    # route both call sites through the patched overlay
    eng.sector_overlay = qqq_overlay
    cad.sector_overlay = qqq_overlay

    db = get_db()
    panel = load_cached_panel(PANEL_PKL)
    comp = cad.pit_scores(db, panel)          # cached after the cadence run
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

    members = fetch_members()
    in_uni = [t for t in members if t in matrix.columns]
    print(f"QQQ members: {len(members)}, in universe/matrix: {len(in_uni)}")
    for d in rebal_dates:
        mc = caps.loc[d].reindex(in_uni).dropna()
        msec = sectors.reindex(mc.index).fillna("Unknown")
        tgt = mc.groupby(msec).sum()
        QQQ_TGT[d] = tgt / tgt.sum()
    first, last = rebal_dates[0], rebal_dates[-1]
    print("QQQ-proxy sector targets, first vs last date:")
    print(pd.DataFrame({first: QQQ_TGT[first], last: QQQ_TGT[last]})
          .fillna(0.0).round(3).to_string())

    qcfg = AblationConfig(name="qqq_monthly", top_pct=0.25, weighting="cap5",
                          sector="qqq_match", vix_tilt=False, hold_months=1)
    keep = ("config", "net_cagr", "sharpe", "sortino", "calmar", "max_dd",
            "beta", "alpha", "alpha_t", "ex_spy", "ex_qqq", "ir", "turnover")
    rows, series = [], {}

    row = simulate_config(data, qcfg, scores=comp, keep_series=True)
    series["qqq_monthly"] = row.pop("_returns")
    rows.append({k: row.get(k) for k in keep})
    for name, hold, offsets in (("qqq_sleeves6", 6, [0, 3]),
                                ("qqq_sleeves12", 12, [0, 6])):
        pr, turn = cad.sleeve_book(comp, data, qcfg, hold, offsets)
        r, pr = cad.metrics_row(name, pr, turn, pxm)
        series[name] = pr
        rows.append(r)

    prev = pd.read_csv(OUT / "hold_cadence.csv")
    prev["config"] = "spy_" + prev["config"].astype(str)
    for _, r in prev.iterrows():
        if r["config"] in ("spy_monthly", "spy_sleeves6", "spy_sleeves12"):
            rows.append({k: r.get(k) for k in keep})
    for bench in ("SPY", "QQQ"):
        b = benchmark_row(data, bench)
        rows.append({k: b.get(k) for k in keep})

    out = pd.DataFrame(rows)
    out.to_csv(OUT / "qqq_sector.csv", index=False)
    yearly = {n: (1.0 + p).groupby(p.index.str[:4]).prod() - 1.0
              for n, p in series.items()}
    ydf = pd.DataFrame(yearly)
    ydf.to_csv(OUT / "qqq_sector_yearly.csv")
    with pd.option_context("display.width", 200, "display.float_format",
                           lambda v: f"{v:.3f}"):
        print(out.to_string(index=False))
        print("\nper-year net returns (QQQ-sector books):")
        print(ydf.to_string())
    print(f"\nwrote {OUT}/qqq_sector.csv + qqq_sector_yearly.csv")


if __name__ == "__main__":
    main()
