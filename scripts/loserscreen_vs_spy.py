"""Chosen construction (mix_screen25: 50/50 EW-cap, drop bottom 25%) vs SPY.

Head-to-head of the selected loser-screen book against buying SPY: growth of an
initial $10,000, yearly returns and Sharpe ratios for both, full-period stats.

Usage: python scripts/loserscreen_vs_spy.py    (reuses cache/vixtilt/, ~30s)
Outputs: output/loserscreen_final/vs_spy.png / yearly_vs_spy.csv / summary.csv
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

START_CAPITAL = 10_000
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
    res = st.run_study(panel, matrix, sectors, books=[BOOK], mcaps=mcaps,
                       verbose=False)
    pf = res.portfolios[BOOK.name]
    rets = pd.DataFrame({"model": pf["net"], "SPY": pf["spy"]}).dropna()
    rets.index = pd.to_datetime(rets.index)

    eq = (1 + rets).cumprod() * START_CAPITAL
    yr = rets.groupby(rets.index.year)
    yearly = pd.DataFrame({
        "model_return": yr["model"].apply(lambda g: (1 + g).prod() - 1),
        "spy_return": yr["SPY"].apply(lambda g: (1 + g).prod() - 1),
        "model_sharpe": yr["model"].apply(
            lambda g: g.mean() / g.std(ddof=1) * np.sqrt(12)),
        "spy_sharpe": yr["SPY"].apply(
            lambda g: g.mean() / g.std(ddof=1) * np.sqrt(12)),
    }).round(3)
    yearly["model_wins"] = yearly["model_sharpe"] > yearly["spy_sharpe"]

    years = len(rets) / 12
    summary = {}
    for c in ("model", "SPY"):
        r = rets[c]
        e = (1 + r).cumprod()
        summary[c] = {
            "final_value": float(e.iloc[-1]) * START_CAPITAL,
            "cagr": float(e.iloc[-1] ** (1 / years) - 1),
            "sharpe": float(r.mean() / r.std(ddof=1) * np.sqrt(12)),
            "max_drawdown": float((e / e.cummax() - 1).min()),
            "ann_vol": float(r.std(ddof=1) * np.sqrt(12)),
        }
    act = rets["model"] - rets["SPY"]
    te = float(act.std(ddof=1) * np.sqrt(12))
    summary["model"].update({
        "excess_cagr": summary["model"]["cagr"] - summary["SPY"]["cagr"],
        "te_ann": te, "ir": float(act.mean() * 12 / te),
        "t_stat": float(act.mean() / (act.std(ddof=1) / np.sqrt(len(act)))),
    })
    sm = pd.DataFrame(summary).round(4)

    OUT.mkdir(parents=True, exist_ok=True)
    yearly.to_csv(OUT / "yearly_vs_spy.csv")
    sm.to_csv(OUT / "summary.csv")

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    fig.suptitle(f"mix_screen25 (50/50 EW-cap, top 75% of composite) vs SPY — "
                 f"${START_CAPITAL:,} invested, net of 10bps/side", fontsize=12)

    ax = axes[0]
    ax.plot(eq.index, eq["model"], label=f"model  (${eq['model'].iloc[-1]:,.0f})",
            lw=2, color="tab:blue")
    ax.plot(eq.index, eq["SPY"], label=f"SPY  (${eq['SPY'].iloc[-1]:,.0f})",
            lw=1.5, ls="--", color="tab:gray")
    ax.set_yscale("log")
    ax.set_title("Growth of $10,000")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    ax = axes[1]
    x = np.arange(len(yearly.index))
    ax.bar(x - 0.18, yearly["model_sharpe"], width=0.34, label="model",
           color="tab:blue")
    ax.bar(x + 0.18, yearly["spy_sharpe"], width=0.34, label="SPY",
           color="tab:gray")
    ax.set_xticks(x, yearly.index, rotation=45)
    ax.axhline(0, color="k", lw=0.5)
    wins = int(yearly["model_wins"].sum())
    ax.set_title(f"Yearly Sharpe — model wins {wins}/{len(yearly)} years")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(OUT / "vs_spy.png", dpi=130)

    print(yearly.to_string())
    print()
    print(sm.to_string())
    print(f"\nwrote {OUT}/vs_spy.png, yearly_vs_spy.csv, summary.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
