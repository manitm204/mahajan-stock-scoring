"""Charts for the v4/v4b parent-veto results (loserscreen study).

Four panels: growth of $10k for the veto books vs mix_screen25 and SPY;
leave-one-out decomposition of the all-parent veto (which parent's veto helps
or hurts); the subset books against their size-matched random nulls; yearly
Sharpe. Reads LOO Sharpes and null draws from output/loserscreen_veto2/
(run `python run_loserscreen_study.py --books veto2` first).

Usage: python scripts/loserscreen_veto_charts.py    (reuses cache/vixtilt/, ~40s)
Outputs: output/loserscreen_veto2/veto_dashboard.png
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
VETO2_DIR = Path("output/loserscreen_veto2")
SHOW = ("mix_screen25", "vqs20", "vall20", "vpos4_20", "vpos6_20")
COLORS = {"mix_screen25": "tab:blue", "vqs20": "tab:purple",
          "vall20": "tab:red", "vpos4_20": "tab:cyan",
          "vpos6_20": "tab:orange", "SPY": "tab:gray"}


def main() -> int:
    full = pd.read_csv(VETO2_DIR / "full_net.csv").set_index("book")
    nulls = {b: pd.read_csv(VETO2_DIR / f"null_sharpes_{b}.csv")["null_sharpe"]
             for b in ("vall20", "vpos4_20", "vpos6_20")}

    panel = _load_panel()
    with get_db() as db:
        matrix = dl.load_price_matrix(db, panel.universe, PANEL_START, PRICE_END)
        sectors = dl.global_sectors(db)
        mcaps = market_caps(db, list(panel.rebal_dates), panel.universe, matrix)
    books = [b for b in st.BOOKS_VETO2 if b.name in SHOW]
    res = st.run_study(panel, matrix, sectors, books=books, mcaps=mcaps,
                       verbose=False)

    rets = pd.DataFrame({b.name: res.portfolios[b.name]["net"] for b in books})
    rets["SPY"] = res.portfolios["mix_screen25"]["spy"]
    rets = rets.dropna()
    rets.index = pd.to_datetime(rets.index)
    eq = (1 + rets).cumprod() * START_CAPITAL

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Parent-veto screens on mix_screen25 — vqs20 pre-registered "
                 "PASS; vall20/vpos4/vpos6 EXPLORATORY (in-sample subsets) — "
                 "net of 10bps/side", fontsize=12)

    ax = axes[0, 0]
    for c in list(SHOW) + ["SPY"]:
        n = f", ~{full.loc[c, 'avg_n_names']:.0f} names" if c != "SPY" else ""
        ax.plot(eq.index, eq[c], label=f"{c}  (${eq[c].iloc[-1]:,.0f}{n})",
                lw=2 if c in ("vpos6_20", "mix_screen25") else 1.2,
                ls="--" if c == "SPY" else "-", color=COLORS[c])
    ax.set_yscale("log")
    ax.set_title("Growth of $10,000")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    base = float(full.loc["vall20", "sharpe"])
    loo = {p: float(full.loc[f"vno_{p}20", "sharpe"]) - base
           for p in st._PARENTS}
    loo_s = pd.Series(loo).sort_values()
    ax.barh(np.arange(len(loo_s)), loo_s.values,
            color=["tab:green" if v > 0 else "tab:red" for v in loo_s.values])
    ax.set_yticks(np.arange(len(loo_s)), loo_s.index)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("Sharpe change when this parent's veto is REMOVED from vall20")
    ax.set_title(f"Leave-one-out (vall20 Sharpe {base:.3f}) — green = that "
                 "veto was hurting")
    ax.grid(alpha=0.3, axis="x")

    ax = axes[1, 0]
    bins = np.linspace(min(n.min() for n in nulls.values()) - 0.01,
                       max(full.loc[b, "sharpe"] for b in SHOW) + 0.02, 40)
    for b, col in (("vall20", "tab:red"), ("vpos6_20", "tab:orange")):
        ax.hist(nulls[b], bins=bins, alpha=0.45, color=col,
                label=f"{b} null (200 same-size random drops)")
        ax.axvline(full.loc[b, "sharpe"], color=col, lw=2,
                   label=f"{b} actual ({full.loc[b, 'sharpe']:.3f}, beats "
                         f"{(nulls[b] < full.loc[b, 'sharpe']).mean():.0%})")
    ax.axvline(full.loc["mix_screen25", "sharpe"], color="tab:blue", lw=1.5,
               ls="--", label=f"mix_screen25 "
                              f"({full.loc['mix_screen25', 'sharpe']:.3f})")
    ax.set_xlabel("full-period net Sharpe")
    ax.set_title("Veto books vs size-matched luck distribution")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    yr = rets.groupby(rets.index.year)
    sh = yr.apply(lambda g: g.mean() / g.std(ddof=1) * np.sqrt(12))
    show = ["mix_screen25", "vqs20", "vpos6_20", "SPY"]
    x = np.arange(len(sh.index))
    for i, c in enumerate(show):
        ax.bar(x + (i - (len(show) - 1) / 2) * 0.21, sh[c], width=0.19,
               label=c, color=COLORS[c])
    ax.set_xticks(x, sh.index, rotation=45)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_title("Yearly Sharpe")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(VETO2_DIR / "veto_dashboard.png", dpi=130)
    print(f"wrote {VETO2_DIR}/veto_dashboard.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
