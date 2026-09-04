"""Sector-concentration diagnostic for the veto book (roadmap item 3).

Question: does vpos6_20 (~267 names) quietly concentrate in a sector vs the
broad book? Aggregates portfolio weight by GICS sector per formation date for
mix_broad / mix_screen25 / vpos6_20. Diagnostic only — nothing is selected
from this output. Sectors are the current global mapping (not PIT), fine for
a risk description.

Usage: python scripts/loserscreen_sector_diag.py    (reuses cache/vixtilt/, ~40s)
Outputs: output/loserscreen_cands/sector_diag.png / sector_weights.csv
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
    st.BookSpec("mix_broad", "unscreened blend (reference)", weighting="mix"),
    st.BookSpec("mix_screen25", "screen only", weighting="mix"),
    st._veto_spec("vpos6_20", "ratified working spec",
                  st._VETO2_SUBSETS["vpos6_20"], 0.20),
]
OUT = Path("output/loserscreen_cands")


def main() -> int:
    panel = _load_panel()
    with get_db() as db:
        matrix = dl.load_price_matrix(db, panel.universe, PANEL_START, PRICE_END)
        sectors = dl.global_sectors(db)
        mcaps = market_caps(db, list(panel.rebal_dates), panel.universe, matrix)
    res = st.run_study(panel, matrix, sectors, books=BOOKS, mcaps=mcaps,
                       verbose=False)

    sec_w: dict[str, pd.DataFrame] = {}
    for b in BOOKS:
        rows = {}
        for d, w in res.weights[b.name].items():
            if w.empty:
                continue
            rows[d] = w.groupby(sectors.reindex(w.index).fillna("Unknown")).sum()
        sec_w[b.name] = pd.DataFrame(rows).T.fillna(0.0).sort_index()

    avg = pd.DataFrame({n: df.mean() for n, df in sec_w.items()}) \
        .sort_values("mix_broad", ascending=False)
    active = (sec_w["vpos6_20"] - sec_w["mix_broad"]
              .reindex(sec_w["vpos6_20"].index).fillna(0.0)) \
        .reindex(columns=avg.index).fillna(0.0)
    avg["active_avg"] = active.mean()
    avg["active_max"] = active.max()
    avg["active_min"] = active.min()

    hhi = pd.DataFrame({n: (df ** 2).sum(axis=1) for n, df in sec_w.items()})
    hhi.index = pd.to_datetime(hhi.index)

    OUT.mkdir(parents=True, exist_ok=True)
    avg.round(4).to_csv(OUT / "sector_weights.csv")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.suptitle("Sector diagnostic — does the veto book concentrate? "
                 "(vpos6_20 vs mix_broad / mix_screen25)", fontsize=12)

    ax = axes[0]
    x = np.arange(len(avg.index))
    for i, (n, col) in enumerate((("mix_broad", "tab:gray"),
                                  ("mix_screen25", "tab:blue"),
                                  ("vpos6_20", "tab:orange"))):
        ax.bar(x + (i - 1) * 0.27, avg[n], width=0.25, label=n, color=col)
    ax.set_xticks(x, avg.index, rotation=60, ha="right", fontsize=8)
    ax.set_title("Average sector weight")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    ax = axes[1]
    act_t = active.copy()
    act_t.index = pd.to_datetime(act_t.index)
    show = avg["active_avg"].abs().sort_values(ascending=False).index[:6]
    for s in show:
        ax.plot(act_t.index, act_t[s], lw=1.3,
                label=f"{s} (avg {avg.loc[s, 'active_avg']:+.1%})")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_title("Active sector weight: vpos6_20 − mix_broad (top 6 by |avg|)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[2]
    for n, col in (("mix_broad", "tab:gray"), ("mix_screen25", "tab:blue"),
                   ("vpos6_20", "tab:orange")):
        ax.plot(hhi.index, hhi[n], lw=1.3, label=n, color=col)
    ax.set_title("Sector concentration (HHI of sector weights)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT / "sector_diag.png", dpi=130)

    pd.set_option("display.float_format", "{:.3f}".format)
    print(avg.to_string())
    print(f"\nmax single-date active weight: "
          f"{float(active.abs().max().max()):+.1%} "
          f"({active.abs().max().idxmax()})")
    print(f"wrote {OUT}/sector_diag.png, sector_weights.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
