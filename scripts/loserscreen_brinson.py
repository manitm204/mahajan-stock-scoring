"""Brinson-Fachler decomposition: vpos6_20 active return vs mix_screen25.

Splits each month's gross active return into:
  allocation = sum_s (Wp_s - Wb_s) * (Rb_s - Rb)   — sector over/underweights
  selection  = sum_s Wp_s * (Rp_s - Rb_s)          — name picking within sector
(the two terms sum exactly to the active return). Costs excluded (turnover
difference is ~10bps/yr, immaterial to the split). The duplicate sector
taxonomies are merged so peer groups aren't fragmented.

Usage: python scripts/loserscreen_brinson.py    (reuses cache/vixtilt/, ~40s)
Outputs: output/loserscreen_veto2/brinson.png / brinson_monthly.csv /
         brinson_sectors.csv
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

PORT, BENCH = "vpos6_20", "mix_screen25"
BOOKS = [
    st.BookSpec(BENCH, "screen only (benchmark)", weighting="mix"),
    st._veto_spec(PORT, "ratified working spec", st._VETO2_SUBSETS[PORT], 0.20),
]
MERGE = {"Healthcare": "Health Care", "Financial Services": "Financials",
         "Technology": "Information Technology",
         "Consumer Cyclical": "Consumer Discretionary",
         "Consumer Defensive": "Consumer Staples",
         "Basic Materials": "Materials"}
OUT = Path("output/loserscreen_veto2")


def main() -> int:
    panel = _load_panel()
    with get_db() as db:
        matrix = dl.load_price_matrix(db, panel.universe, PANEL_START, PRICE_END)
        sectors = dl.global_sectors(db)
        mcaps = market_caps(db, list(panel.rebal_dates), panel.universe, matrix)
    sectors = sectors.replace(MERGE)
    res = st.run_study(panel, matrix, sectors, books=BOOKS, mcaps=mcaps,
                       verbose=False)

    wp_all, wb_all = res.weights[PORT], res.weights[BENCH]
    dates = [d for d in sorted(wb_all) if d in matrix.index and len(wb_all[d])]
    rows, sec_rows = [], []
    for i, d in enumerate(dates[:-1]):
        nxt = dates[i + 1]
        wp, wb = wp_all.get(d), wb_all[d]
        if wp is None or wp.empty:
            continue
        ret = matrix.loc[nxt] / matrix.loc[d] - 1.0

        def norm(w: pd.Series) -> pd.Series:
            r = ret.reindex(w.index)
            w = w[r.notna()]
            return w / w.sum()

        wwp, wwb = norm(wp), norm(wb)
        sp = sectors.reindex(wwp.index).fillna("Unknown")
        sb = sectors.reindex(wwb.index).fillna("Unknown")
        Wp, Wb = wwp.groupby(sp).sum(), wwb.groupby(sb).sum()
        Rp_s = (wwp * ret.reindex(wwp.index)).groupby(sp).sum() / Wp
        Rb_s = (wwb * ret.reindex(wwb.index)).groupby(sb).sum() / Wb
        Rb = float((wwb * ret.reindex(wwb.index)).sum())

        secs = Wp.index.union(Wb.index)
        Wp, Wb = Wp.reindex(secs, fill_value=0.0), Wb.reindex(secs, fill_value=0.0)
        Rp_s = Rp_s.reindex(secs)
        Rb_s = Rb_s.reindex(secs).fillna(Rp_s)   # sector absent from benchmark
        Rp_s = Rp_s.fillna(Rb_s)                 # sector absent from portfolio

        alloc_s = (Wp - Wb) * (Rb_s - Rb)
        sel_s = Wp * (Rp_s - Rb_s)
        act = float((wwp * ret.reindex(wwp.index)).sum()) - Rb
        assert abs(alloc_s.sum() + sel_s.sum() - act) < 1e-10
        rows.append({"date": d, "active": act,
                     "allocation": float(alloc_s.sum()),
                     "selection": float(sel_s.sum())})
        for s in secs:
            sec_rows.append({"date": d, "sector": s,
                             "allocation": float(alloc_s[s]),
                             "selection": float(sel_s[s])})

    m = pd.DataFrame(rows).set_index("date")
    m.index = pd.to_datetime(m.index)
    sec = pd.DataFrame(sec_rows).groupby("sector")[
        ["allocation", "selection"]].mean() * 12
    sec = sec.loc[sec.abs().sum(axis=1).sort_values(ascending=False).index]

    def tstat(x: pd.Series) -> float:
        return float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x))))

    summ = pd.DataFrame({c: {"ann_return": float(m[c].mean() * 12),
                             "t_stat": tstat(m[c]),
                             "share_of_active":
                                 float(m[c].mean() / m["active"].mean())}
                         for c in ("active", "allocation", "selection")}).round(4)

    OUT.mkdir(parents=True, exist_ok=True)
    m.to_csv(OUT / "brinson_monthly.csv")
    sec.round(5).to_csv(OUT / "brinson_sectors.csv")

    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))
    fig.suptitle(f"Brinson-Fachler: {PORT} active vs {BENCH} (gross) — "
                 "sector allocation vs within-sector selection", fontsize=12)

    ax = axes[0]
    for c, col in (("active", "k"), ("allocation", "tab:red"),
                   ("selection", "tab:green")):
        ax.plot(m.index, m[c].cumsum() * 100, lw=1.6, color=col,
                label=f"{c}  ({summ.loc['ann_return', c]:+.2%}/yr, "
                      f"t={summ.loc['t_stat', c]:.2f})")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel("cumulative active return (pp)")
    ax.set_title("Cumulative decomposition")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[1]
    x = np.arange(len(sec.index))
    ax.barh(x + 0.2, sec["selection"] * 100, height=0.38,
            label="selection", color="tab:green")
    ax.barh(x - 0.2, sec["allocation"] * 100, height=0.38,
            label="allocation", color="tab:red")
    ax.set_yticks(x, sec.index, fontsize=8)
    ax.invert_yaxis()
    ax.axvline(0, color="k", lw=0.6)
    ax.set_xlabel("avg contribution to active return (pp/yr)")
    ax.set_title("Per-sector contributions")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis="x")

    fig.tight_layout()
    fig.savefig(OUT / "brinson.png", dpi=130)

    print(summ.to_string())
    print()
    print((sec * 100).round(2).to_string())
    print(f"\nwrote {OUT}/brinson.png, brinson_monthly.csv, brinson_sectors.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
