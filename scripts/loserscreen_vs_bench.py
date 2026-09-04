"""Chosen construction (mix_screen25) vs SPY, QQQ and 3x-leveraged SPY.

3x SPY is simulated the way leveraged ETFs work: 3x the *daily* SPY return with
daily reset, minus a 0.91%/yr expense ratio (UPRO-style). Financing costs are
NOT modelled, so the 3x line is slightly flattered.

Usage: python scripts/loserscreen_vs_bench.py    (reuses cache/vixtilt/, ~30s)
Outputs: output/loserscreen_final/vs_benchmarks.png / yearly_vs_benchmarks.csv
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
LEV, ER_ANN = 3.0, 0.0091
BOOK = st.BookSpec("mix_screen25",
                   "50/50 EW-cap, minus bottom 25% by composite (chosen)",
                   weighting="mix")
OUT = Path("output/loserscreen_final")


def period_returns(daily: pd.Series, dates: list[str],
                   lev: float = 1.0) -> pd.Series:
    """Formation-date grid returns from a daily close series; `lev` applies to
    each daily return (daily reset) with the ETF expense drag."""
    r = daily.pct_change()
    out = {}
    for d, nxt in zip(dates[:-1], dates[1:]):
        seg = r.loc[(r.index > d) & (r.index <= nxt)]
        if seg.empty:
            continue
        out[d] = float((1.0 + lev * seg - ER_ANN / 252 * (lev != 1.0)).prod() - 1.0)
    return pd.Series(out)


def main() -> int:
    panel = _load_panel()
    with get_db() as db:
        matrix = dl.load_price_matrix(db, panel.universe, PANEL_START, PRICE_END)
        sectors = dl.global_sectors(db)
        mcaps = market_caps(db, list(panel.rebal_dates), panel.universe, matrix)
        qqq = db.query_df("SELECT date, adj_close FROM daily_prices "
                          "WHERE ticker='QQQ' ORDER BY date")
        spy = db.query_df("SELECT date, adj_close FROM daily_prices "
                          "WHERE ticker='SPY' ORDER BY date")
    qqq = qqq.set_index("date")["adj_close"].astype(float)
    spy = spy.set_index("date")["adj_close"].astype(float)

    res = st.run_study(panel, matrix, sectors, books=[BOOK], mcaps=mcaps,
                       verbose=False)
    pf = res.portfolios[BOOK.name]
    dates = list(pf.index) + [pf["next"].iloc[-1]]

    rets = pd.DataFrame({
        "model": pf["net"],
        "SPY": pf["spy"],
        "QQQ": period_returns(qqq, dates),
        "SPY 3x (daily reset)": period_returns(spy, dates, lev=LEV),
    }).dropna()
    rets.index = pd.to_datetime(rets.index)

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
    sharpe.to_csv(OUT / "yearly_vs_benchmarks.csv")

    colors = {"model": "tab:blue", "SPY": "tab:gray", "QQQ": "tab:green",
              "SPY 3x (daily reset)": "tab:red"}
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))
    fig.suptitle("mix_screen25 vs SPY / QQQ / 3x SPY — $10,000 invested, "
                 "net of costs (3x: daily reset, ER 0.91%, no financing drag)",
                 fontsize=12)

    ax = axes[0]
    for c in eq.columns:
        ax.plot(eq.index, eq[c], label=f"{c}  (${eq[c].iloc[-1]:,.0f})",
                lw=2 if c == "model" else 1.3,
                ls="--" if c != "model" else "-", color=colors[c])
    ax.set_yscale("log")
    ax.set_title("Growth of $10,000 (log scale)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[1]
    x = np.arange(len(sharpe.index))
    for i, c in enumerate(rets.columns):
        ax.bar(x + (i - 1.5) * 0.21, sharpe[c], width=0.19, label=c,
               color=colors[c])
    ax.set_xticks(x, sharpe.index, rotation=45)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_title("Yearly Sharpe")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(OUT / "vs_benchmarks.png", dpi=130)

    print("Yearly Sharpe:")
    print(sharpe.to_string())
    print("\nFull period:")
    print(summary.to_string())
    print(f"\nwrote {OUT}/vs_benchmarks.png, yearly_vs_benchmarks.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
