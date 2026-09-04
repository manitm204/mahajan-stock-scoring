"""Yearly cap-vs-EW diagnostic for the loser-screen study.

Question: cap-weighted books out-Sharpe equal-weight at every screen depth — is
that concentrated in specific years (mega-cap regime) or spread evenly? Compares
per-year Sharpe/CAGR of the main books against SPY and plots a dashboard.

Usage: python scripts/loserscreen_yearly.py    (reuses cache/vixtilt/, ~30s)
Outputs: output/loserscreen_v2/yearly_sharpe.csv / yearly_cagr.csv /
         yearly_dashboard.png
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

BOOKS = [
    st.BookSpec("broad", "EW broad"),
    st.BookSpec("screen20", "EW screen20"),
    st.BookSpec("screen30", "EW screen30"),
    st.BookSpec("cap_broad", "cap broad", weighting="cap"),
    st.BookSpec("cap_screen20", "cap screen20", weighting="cap"),
    st.BookSpec("cap_screen30", "cap screen30", weighting="cap"),
    st.BookSpec("mix_broad", "50/50 broad", weighting="mix"),
    st.BookSpec("mix_screen20", "50/50 screen20", weighting="mix"),
]
OUT = Path("output/loserscreen_v2")


def yearly(df: pd.DataFrame, fn) -> pd.DataFrame:
    return df.groupby(pd.to_datetime(df.index).year).apply(fn)


def main() -> int:
    panel = _load_panel()
    with get_db() as db:
        matrix = dl.load_price_matrix(db, panel.universe, PANEL_START, PRICE_END)
        sectors = dl.global_sectors(db)
        mcaps = market_caps(db, list(panel.rebal_dates), panel.universe, matrix)
    res = st.run_study(panel, matrix, sectors, books=BOOKS, mcaps=mcaps,
                       verbose=False)

    rets = pd.DataFrame({b.name: res.portfolios[b.name]["net"] for b in BOOKS})
    rets["SPY"] = res.portfolios["broad"]["spy"]

    sharpe = yearly(rets, lambda g: g.mean() / g.std(ddof=1) * np.sqrt(12)).round(3)
    cagr = yearly(rets, lambda g: (1 + g).prod() ** (12 / len(g)) - 1).round(4)
    OUT.mkdir(parents=True, exist_ok=True)
    sharpe.to_csv(OUT / "yearly_sharpe.csv")
    cagr.to_csv(OUT / "yearly_cagr.csv")

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("Loser-screen: cap vs EW by year (net of 10bps/side)", fontsize=13)

    # 1. cumulative equity (log)
    ax = axes[0, 0]
    eq = (1 + rets[["broad", "screen20", "cap_broad", "cap_screen20",
                    "mix_screen20", "SPY"]]).cumprod()
    eq.index = pd.to_datetime(eq.index)
    for c in eq.columns:
        ax.plot(eq.index, eq[c], label=c,
                lw=2 if c in ("cap_screen20", "screen20", "mix_screen20") else 1.2,
                ls="--" if c == "SPY" else "-")
    ax.set_yscale("log")
    ax.set_title("Cumulative equity (log)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # 2. yearly Sharpe: broad books vs SPY
    ax = axes[0, 1]
    x = np.arange(len(sharpe.index))
    for i, c in enumerate(["broad", "cap_broad", "SPY"]):
        ax.bar(x + (i - 1) * 0.27, sharpe[c], width=0.25, label=c)
    ax.set_xticks(x, sharpe.index, rotation=45)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_title("Yearly Sharpe — is cap>EW market-wide? (broad books vs SPY)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    # 3. yearly Sharpe gap: cap minus EW (broad pair vs screened pair)
    ax = axes[1, 0]
    gap_broad = sharpe["cap_broad"] - sharpe["broad"]
    gap_scr = sharpe["cap_screen20"] - sharpe["screen20"]
    ax.bar(x - 0.15, gap_broad, width=0.28, label="cap_broad − broad")
    ax.bar(x + 0.15, gap_scr, width=0.28, label="cap_screen20 − screen20")
    ax.set_xticks(x, sharpe.index, rotation=45)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_title("Cap-minus-EW Sharpe gap by year")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    # 4. heatmap: yearly Sharpe, all books + SPY
    ax = axes[1, 1]
    cols = ["broad", "screen20", "screen30", "cap_broad", "cap_screen20",
            "cap_screen30", "mix_broad", "mix_screen20", "SPY"]
    m = sharpe[cols].T
    im = ax.imshow(m.values, aspect="auto", cmap="RdYlGn", vmin=-2, vmax=3)
    ax.set_xticks(range(len(m.columns)), m.columns, rotation=45)
    ax.set_yticks(range(len(cols)), cols)
    for i in range(len(cols)):
        for j in range(len(m.columns)):
            ax.text(j, i, f"{m.values[i, j]:.1f}", ha="center", va="center",
                    fontsize=7)
    ax.set_title("Yearly Sharpe heatmap")
    fig.colorbar(im, ax=ax, shrink=0.8)

    fig.tight_layout()
    fig.savefig(OUT / "yearly_dashboard.png", dpi=130)
    print(sharpe.to_string())
    print(f"\nwrote {OUT}/yearly_sharpe.csv, yearly_cagr.csv, yearly_dashboard.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
