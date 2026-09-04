"""Vol-target variants on mix_screen25: signal (realized vol vs VIX) x floor.

EXPLORATORY follow-up to the pre-registered scripts/loserscreen_voltarget.py
(realized-vol, clip 0.75-1.5 — PASSED). Grid: dial = trailing 63d SPY realized
vol OR spot VIX (last close <= formation date, /100); floor = 0.75 or 1.00;
cap 1.50 and target 20% everywhere. Same financing as before (borrow at
fed funds + 1%, spare cash earns fed funds).

Usage: python scripts/loserscreen_voltarget2.py    (reuses cache/vixtilt/, ~30s)
Outputs: output/loserscreen_final/voltarget2.png / voltarget2_summary.csv
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
from scripts.loserscreen_leverage import FFR, MARGIN_SPREAD

TARGET_VOL, LOOKBACK_D, E_MAX = 0.20, 63, 1.50
STATIC_LEV = 1.25
START_CAPITAL = 10_000
BOOK = st.BookSpec("mix_screen25", "chosen construction", weighting="mix")
OUT = Path("output/loserscreen_final")

VARIANTS = [  # (name, dial, floor)
    ("rv 0.75-1.5", "rv", 0.75),
    ("rv 1.0-1.5", "rv", 1.00),
    ("vix 0.75-1.5", "vix", 0.75),
    ("vix 1.0-1.5", "vix", 1.00),
]


def main() -> int:
    panel = _load_panel()
    with get_db() as db:
        matrix = dl.load_price_matrix(db, panel.universe, PANEL_START, PRICE_END)
        sectors = dl.global_sectors(db)
        mcaps = market_caps(db, list(panel.rebal_dates), panel.universe, matrix)
        spy = db.query_df("SELECT date, adj_close FROM daily_prices "
                          "WHERE ticker='SPY' ORDER BY date")
        vix = db.query_df("SELECT date, adj_close FROM daily_prices "
                          "WHERE ticker='VIX' ORDER BY date")
    spy = spy.set_index("date")["adj_close"].astype(float)
    vix = vix.set_index("date")["adj_close"].astype(float)
    rv = spy.pct_change().rolling(LOOKBACK_D).std() * np.sqrt(252)

    res = st.run_study(panel, matrix, sectors, books=[BOOK], mcaps=mcaps,
                       verbose=False)
    pf = res.portfolios[BOOK.name]
    rets = pd.DataFrame({"model": pf["net"], "SPY": pf["spy"]}).dropna()

    def pit(series: pd.Series) -> pd.Series:
        """Last value on/before each formation date."""
        return series.reindex(series.index.union(rets.index)).ffill() \
            .reindex(rets.index)

    dial = {"rv": pit(rv), "vix": pit(vix) / 100.0}
    ffr = pd.Series([FFR[pd.Timestamp(d).year] for d in rets.index],
                    index=rets.index)

    expos: dict[str, pd.Series] = {}
    for name, d, floor in VARIANTS:
        e = (TARGET_VOL / dial[d]).clip(floor, E_MAX)
        expos[name] = e
        borrow, park = (e - 1).clip(lower=0), (1 - e).clip(lower=0)
        rets[name] = (e * rets["model"] - borrow * (ffr + MARGIN_SPREAD) / 12
                      + park * ffr / 12)
    rets["static 1.25x"] = (STATIC_LEV * rets["model"]
                            - (STATIC_LEV - 1) * (ffr + MARGIN_SPREAD) / 12)
    rets.index = pd.to_datetime(rets.index)

    eq = (1 + rets).cumprod() * START_CAPITAL
    years = len(rets) / 12
    cols = ["model", "static 1.25x"] + [v[0] for v in VARIANTS]
    summary = pd.DataFrame({c: {
        "mean_exposure": float(expos[c].mean()) if c in expos else
        (STATIC_LEV if c != "model" else 1.0),
        "final_value": float(eq[c].iloc[-1]),
        "cagr": float((eq[c].iloc[-1] / START_CAPITAL) ** (1 / years) - 1),
        "sharpe": float(rets[c].mean() / rets[c].std(ddof=1) * np.sqrt(12)),
        "ann_vol": float(rets[c].std(ddof=1) * np.sqrt(12)),
        "max_drawdown": float((eq[c] / eq[c].cummax() - 1).min()),
    } for c in cols}).round(4)

    OUT.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT / "voltarget2_summary.csv")

    colors = {"model": "tab:blue", "static 1.25x": "tab:red",
              "rv 0.75-1.5": "tab:purple", "rv 1.0-1.5": "tab:cyan",
              "vix 0.75-1.5": "tab:olive", "vix 1.0-1.5": "tab:orange"}
    fig, axes = plt.subplots(2, 2, figsize=(16, 9))
    fig.suptitle("Vol-target variants — dial (63d realized vol vs VIX) × floor "
                 "(0.75 vs 1.0), target 20%, cap 1.5x (EXPLORATORY)", fontsize=12)

    ax = axes[0, 0]
    for c in cols:
        ax.plot(eq.index, eq[c], label=f"{c}  (${eq[c].iloc[-1]:,.0f})",
                lw=1.6 if c in expos else 1.1,
                ls="--" if c in ("model", "static 1.25x") else "-",
                color=colors[c])
    ax.set_yscale("log")
    ax.set_title("Growth of $10,000")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    for c in ("rv 1.0-1.5", "vix 1.0-1.5"):
        ax.step(rets.index, expos[c].values, where="post", label=c,
                color=colors[c], lw=1.4)
    ax.axhline(STATIC_LEV, color="tab:red", ls="--", lw=1, label="static 1.25x")
    ax.set_title("Exposure path (floor-1.0 variants)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    for c in ("static 1.25x", "rv 1.0-1.5", "vix 1.0-1.5", "rv 0.75-1.5"):
        e = eq[c]
        ax.plot(e.index, e / e.cummax() - 1, label=c, lw=1.2, color=colors[c])
    ax.set_title("Drawdown")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    yr = rets.groupby(rets.index.year)
    sh = yr.apply(lambda g: g.mean() / g.std(ddof=1) * np.sqrt(12))
    show = ["static 1.25x", "rv 1.0-1.5", "vix 1.0-1.5"]
    x = np.arange(len(sh.index))
    for i, c in enumerate(show):
        ax.bar(x + (i - 1) * 0.28, sh[c], width=0.26, label=c, color=colors[c])
    ax.set_xticks(x, sh.index, rotation=45)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_title("Yearly Sharpe")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(OUT / "voltarget2.png", dpi=130)

    print(summary.to_string())
    print(f"\nwrote {OUT}/voltarget2.png, voltarget2_summary.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
