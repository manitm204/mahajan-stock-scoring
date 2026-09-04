"""Levered versions of the chosen book (mix_screen25): margin vs futures overlay.

Two implementations of total exposure L, monthly reset on the rebalance grid:

  margin:  hold L x the model book, borrow (L-1) at fed funds + 1%:
             r = L*r_model - (L-1)*(ffr + 0.010)/12
  futures: hold 1x the model book + (L-1) SPY futures overlay (financing
           embedded at ~fed funds + 0.3%):
             r = r_model + (L-1)*(r_spy - (ffr + 0.003)/12)

Margin levers the model's alpha (expensive financing); futures add cheap pure
beta. Fed funds is a stylised annual-average table (no rate series in Layer 1);
2026 assumed 3.9%.

Usage: python scripts/loserscreen_leverage.py    (reuses cache/vixtilt/, ~30s)
Outputs: output/loserscreen_final/vs_leverage.png / yearly_vs_leverage.csv
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from backtesting import data_loader as dl
from data.db import get_db
from loserscreen import study as st
from loserscreen.mcap import market_caps
from run_loserscreen_study import PANEL_START, PRICE_END, _load_panel
from scripts.loserscreen_vs_bench import period_returns

START_CAPITAL = 10_000
LEVS = (1.25, 1.5)
MARGIN_SPREAD, FUT_SPREAD = 0.010, 0.003
FFR = {2017: 0.0100, 2018: 0.0183, 2019: 0.0216, 2020: 0.0038, 2021: 0.0008,
       2022: 0.0168, 2023: 0.0503, 2024: 0.0510, 2025: 0.0433, 2026: 0.0390}
BOOK = st.BookSpec("mix_screen25",
                   "50/50 EW-cap, minus bottom 25% by composite (chosen)",
                   weighting="mix")
OUT = Path("output/loserscreen_final")


def main() -> int:
    panel = _load_panel()
    with get_db() as db:
        matrix = dl.load_price_matrix(db, panel.universe, PANEL_START, PRICE_END)
        sectors = dl.global_sectors(db)
        mcaps = market_caps(db, list(panel.rebal_dates), panel.universe, matrix)
        qqq = db.query_df("SELECT date, adj_close FROM daily_prices "
                          "WHERE ticker='QQQ' ORDER BY date")
    qqq = qqq.set_index("date")["adj_close"].astype(float)

    res = st.run_study(panel, matrix, sectors, books=[BOOK], mcaps=mcaps,
                       verbose=False)
    pf = res.portfolios[BOOK.name]
    dates = list(pf.index) + [pf["next"].iloc[-1]]

    rets = pd.DataFrame({"model": pf["net"], "SPY": pf["spy"],
                         "QQQ": period_returns(qqq, dates)}).dropna()
    rets.index = pd.to_datetime(rets.index)
    ffr = pd.Series(rets.index.year.map(FFR), index=rets.index)
    for lev in LEVS:
        rets[f"{lev}x margin"] = (lev * rets["model"]
                                  - (lev - 1.0) * (ffr + MARGIN_SPREAD) / 12.0)
        rets[f"{lev}x futures"] = (rets["model"] + (lev - 1.0)
                                   * (rets["SPY"] - (ffr + FUT_SPREAD) / 12.0))
    order = (["model"] + [f"{lev}x {kind}" for lev in LEVS
                          for kind in ("margin", "futures")] + ["SPY", "QQQ"])
    rets = rets[order]

    eq = (1 + rets).cumprod() * START_CAPITAL
    yr = rets.groupby(rets.index.year)
    sharpe = yr.apply(lambda g: g.mean() / g.std(ddof=1) * np.sqrt(12)).round(3)
    years = len(rets) / 12
    summary = pd.DataFrame({c: {
        "final_value": float(eq[c].iloc[-1]),
        "cagr": float((eq[c].iloc[-1] / START_CAPITAL) ** (1 / years) - 1),
        "sharpe": float(rets[c].mean() / rets[c].std(ddof=1) * np.sqrt(12)),
        "ann_vol": float(rets[c].std(ddof=1) * np.sqrt(12)),
        "max_drawdown": float((eq[c] / eq[c].cummax() - 1).min()),
    } for c in rets.columns}).round(4)

    OUT.mkdir(parents=True, exist_ok=True)
    sharpe.to_csv(OUT / "yearly_vs_leverage.csv")

    colors = {"model": "tab:blue", "1.25x margin": "tab:purple",
              "1.25x futures": "tab:cyan", "1.5x margin": "tab:red",
              "1.5x futures": "tab:orange", "SPY": "tab:gray",
              "QQQ": "tab:green"}
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))
    fig.suptitle("mix_screen25 levered — margin (ffr+1%) vs SPY-futures overlay "
                 "(ffr+0.3%), monthly reset — $10,000 invested, net of costs",
                 fontsize=12)

    ax = axes[0]
    for c in eq.columns:
        ax.plot(eq.index, eq[c], label=f"{c}  (${eq[c].iloc[-1]:,.0f})",
                lw=2 if c == "model" else 1.3,
                ls="--" if c in ("SPY", "QQQ") else "-", color=colors[c])
    ax.set_yscale("log")
    ax.set_title("Growth of $10,000 (log scale)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    n = len(rets.columns)
    x = np.arange(len(sharpe.index))
    for i, c in enumerate(rets.columns):
        ax.bar(x + (i - (n - 1) / 2) * 0.8 / n, sharpe[c], width=0.75 / n,
               label=c, color=colors[c])
    ax.set_xticks(x, sharpe.index, rotation=45)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_title("Yearly Sharpe")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(OUT / "vs_leverage.png", dpi=130)

    print("Yearly Sharpe:")
    print(sharpe.to_string())
    print("\nFull period:")
    print(summary.to_string())
    print(f"\nwrote {OUT}/vs_leverage.png, yearly_vs_leverage.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
