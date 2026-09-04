"""LLM pilot — cohort books, composite percentiles and 12M forward returns.

Builds the two pre-registered cohorts (docs/llm_pilot_design.md): the
vpos6_20 book at the 2024-06-30 and 2025-06-30 rebalance dates, each name's
composite percentile within the book at formation, and its 12-month forward
return from the deep-history price matrix (delisted names carry their return
to the last print, then flat).

Usage: python scripts/llm_pilot_cohorts.py
Output: output/llm_pilot/cohorts.csv
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from backtesting import data_loader as dl
from data.db import get_db
from loserscreen import study as st
from loserscreen.mcap import market_caps
from run_loserscreen_study import PANEL_START, PRICE_END, _load_panel
from scripts.loserscreen_ops_tax import BOOK
from vixtilt import backtest as bt
from vixtilt.baseline import load_or_build
from vixtilt.windows import semiannual_windows

OUT = Path("output/llm_pilot")
FORMATIONS = ("2024-06", "2025-06")   # resolved to that month's rebal date
FWD_MONTHS = 12


def forward_return(matrix: pd.DataFrame, name: str, d0: str, d1: str) -> float:
    px = matrix[name].dropna()
    px = px[(px.index >= d0) & (px.index <= d1)]
    if len(px) < 2 or px.index[0] != d0:
        return float("nan")
    return float(px.iloc[-1] / px.iloc[0] - 1)


def main() -> int:
    panel = _load_panel()
    rebals = list(panel.rebal_dates)
    with get_db() as db:
        matrix = dl.load_price_matrix(db, panel.universe, PANEL_START, PRICE_END)
        sectors = dl.global_sectors(db)
        mcaps = market_caps(db, rebals, panel.universe, matrix)

    rows = []
    for month in FORMATIONS:
        d0 = next(d for d in rebals if d[:7] == month)
        d1 = rebals[rebals.index(d0) + FWD_MONTHS]
        win = next(w for w in semiannual_windows(2017, "2026-06-30")
                   if d0 in w.test_rebals(rebals))
        wb = load_or_build(panel, matrix, win, verbose=False)
        frame = wb.parent_test[d0]
        score = bt.composite_from_parents(frame, wb.parent_weights, sectors)
        w = st._book_weights(BOOK, score, mcaps.get(d0), frame)
        comp_pct = score.reindex(w.index).rank(pct=True)
        for name in sorted(w.index):
            rows.append({
                "cohort": d0[:4], "formation": d0, "fwd_end": d1,
                "ticker": name, "weight": float(w[name]),
                "composite_pct": float(comp_pct[name]),
                "fwd_12m_ret": forward_return(matrix, name, d0, d1),
                "sector": sectors.get(name, "Unknown"),
            })
        held = [r for r in rows if r["formation"] == d0]
        n_nan = sum(1 for r in held if pd.isna(r["fwd_12m_ret"]))
        print(f"{d0}: {len(held)} names, fwd window → {d1}, "
              f"{n_nan} names without a valid forward return")

    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "cohorts.csv", index=False)
    both = set(df[df.cohort == "2024"].ticker) & set(df[df.cohort == "2025"].ticker)
    print(f"union {df.ticker.nunique()} tickers, overlap {len(both)}")
    print(f"wrote {OUT}/cohorts.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
