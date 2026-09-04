"""Monthly vs quarterly vs semiannual reform, PRE-TAX, across reform phases.

Does monthly rebalancing outperform slower cadences before taxes, or was the
ops-study ordering an artifact of which months the reforms landed on? Runs
the lot simulator with taxes off (costs on, 10bps/side) for every possible
reform phase: semiannual has 6 phases, quarterly 3, annual 12.

Usage: python scripts/loserscreen_cadence_phases.py   (~2min)
Outputs: output/loserscreen_final/cadence_phases.csv
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from backtesting import data_loader as dl
from data.db import get_db
from loserscreen import study as st
from loserscreen.mcap import market_caps
from run_loserscreen_study import PANEL_START, PRICE_END, _load_panel
from scripts.loserscreen_ops_tax import BOOK, OUT, START_CAPITAL, simulate

CADENCES = {"monthly": 1, "quarterly": 3, "semiannual": 6, "annual": 12}


def main() -> int:
    panel = _load_panel()
    with get_db() as db:
        matrix = dl.load_price_matrix(db, panel.universe, PANEL_START, PRICE_END)
        sectors = dl.global_sectors(db)
        mcaps = market_caps(db, list(panel.rebal_dates), panel.universe, matrix)
    res = st.run_study(panel, matrix, sectors, books=[BOOK], mcaps=mcaps,
                       verbose=False)
    w = res.weights[BOOK.name]
    dates = [d for d in sorted(w) if d in matrix.index and len(w[d])]
    years = (len(dates) - 1) / 12

    rows = []
    for name, every in CADENCES.items():
        for ph in range(every):
            r = simulate(dates, w, matrix, every, False, 0.0, 0.0, phase=ph)
            rows.append({
                "cadence": name, "phase": ph,
                "final": r["final_pre_liq"],
                "cagr": (r["final_pre_liq"] / START_CAPITAL) ** (1 / years) - 1,
                "sharpe": r["after_tax_sharpe"],   # taxes off -> pre-tax
                "ann_turnover": r["ann_turnover_1way"],
            })
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.round(4).to_csv(OUT / "cadence_phases.csv", index=False)

    agg = df.groupby("cadence").agg(
        n_phases=("phase", "count"),
        final_mean=("final", "mean"), final_min=("final", "min"),
        final_max=("final", "max"),
        cagr_mean=("cagr", "mean"),
        sharpe_mean=("sharpe", "mean"), sharpe_min=("sharpe", "min"),
        sharpe_max=("sharpe", "max"),
        turnover=("ann_turnover", "mean"),
    ).reindex(CADENCES.keys())
    pd.set_option("display.float_format", "{:,.3f}".format)
    print(agg.to_string())
    print(f"\nwrote {OUT}/cadence_phases.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
