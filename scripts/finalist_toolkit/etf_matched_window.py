"""Matched-window check for the ETF sweep (2026-07-20).

The ranked sweep in etf_blend.py lets each ETF use its own inception window, so
JEPQ (n=48, from May-2022) is scored on a different market than GLD (n=76).
This re-scores the leaders on JEPQ's window so the comparison is apples-to-apples.
"""
from __future__ import annotations
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/manit/Desktop/fun_projects/mahajan_hedge_fund")

import numpy as np
import pandas as pd

import options_zoo as oz
import etf_blend as eb
from ivr_sweep import get_env, run_flex, CS, IC
from combo_stats_2020 import xstats
from backtesting import data_loader as dl
from data.db import get_db
from run_walkforward import PANEL_START, PRICE_END

LEADERS = ["JEPQ", "CGDV", "CGGR", "GLD", "SLV", "PDBC", "BIL", "PULS",
           "JAAA", "SCHD", "JEPI"]


def main():
    env = get_env()
    mdates, r_book = env["mdates"], env["r_book"]
    spy_at, spy_gross = env["spy_at"], env["spy_gross"]
    tdays, spy, vixd = env["tdays"], env["spy"], env["vixd"]
    td_dt = pd.to_datetime(tdays)

    vser = pd.Series(vixd)
    lo, hi = vser.rolling(252).min(), vser.rolling(252).max()
    ivr = ((vser - lo) / (hi - lo).replace(0, np.nan) * 100).fillna(50).values
    i0 = max(252, int(np.searchsorted(np.array(tdays), "2017-01-01")))
    _, cs_d = run_flex(lambda v: (*CS, "cs"), tdays, td_dt, spy, vixd, ivr, i0)
    _, ic_d = run_flex(lambda v: (*IC, "ic") if v >= 30 else None,
                       tdays, td_dt, spy, vixd, ivr, i0)
    merged = dict(cs_d)
    for d, v in ic_d.items(): merged[d] = merged.get(d, 0.0) + v
    intr, opt = oz.monthly_sleeve(merged, mdates)
    sleeve = (intr.reindex(r_book.index).fillna(0.0) * (1 - oz.ST)
              + opt.reindex(r_book.index).fillna(0.0) * (1 - oz.BLEND))

    with get_db() as db:
        matrix_etf = dl.load_price_matrix(db, ["QQQ","IWM","GLD","TLT","QUAL","MTUM"],
                                          PANEL_START, PRICE_END)

    rets = {}
    for t in LEADERS:
        rets[t] = eb.etf_monthly_at(t, mdates, matrix_etf)

    # JEPQ's window is the binding constraint
    start = str(rets["JEPQ"].index.min())
    print(f"\nMatched window: {start} onward  (JEPQ inception binds)\n")

    def stats_on(r, lo_date):
        rr = r[r.index.astype(str) >= lo_date].dropna()
        sa = spy_at[spy_at.index.astype(str) >= lo_date].dropna()
        sg = spy_gross[spy_gross.index.astype(str) >= lo_date].dropna()
        return xstats(rr, sa, sg), len(rr)

    def blend(w, r_e, lo_date):
        idx = r_book.index.intersection(r_e.index).intersection(sleeve.index)
        idx = idx[idx.astype(str) >= lo_date]
        return ((0.70 - w) * r_book.reindex(idx) + w * r_e.reindex(idx)
                + 0.30 * sleeve.reindex(idx))

    base, n = stats_on(0.7 * r_book + 0.3 * sleeve, start)
    print(f"{'':8}{'n':>4}{'CAGR':>7}{'Shrp':>6}{'Sort':>6}{'Calm':>6}{'maxDD':>8}"
          f"{'beta':>6}{'ΔShrp':>7}")
    print(f"{'Combo':8}{n:>4}{base['cagr']*100:6.1f}%{base['sharpe']:6.2f}"
          f"{base['sortino']:6.2f}{base['calmar']:6.2f}{base['mdd']*100:7.1f}%"
          f"{base['beta']:6.2f}{0.0:+7.2f}")

    rows = []
    for t in LEADERS:
        m, nn = stats_on(blend(0.10, rets[t], start), start)
        rows.append((t, m, nn))
    rows.sort(key=lambda x: -x[1]["sharpe"])
    for t, m, nn in rows:
        print(f"{t:8}{nn:>4}{m['cagr']*100:6.1f}%{m['sharpe']:6.2f}{m['sortino']:6.2f}"
              f"{m['calmar']:6.2f}{m['mdd']*100:7.1f}%{m['beta']:6.2f}"
              f"{m['sharpe']-base['sharpe']:+7.2f}")

    # Chart on the matched window: every series rebased at the SAME date, so a
    # late-inception fund cannot appear to start above a drawdown it missed.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    combo = 0.7 * r_book + 0.3 * sleeve
    series = {"Combo 70/30 (no fund)": combo[combo.index.astype(str) >= start]}
    for t in [r[0] for r in rows]:
        series[f"+10% {t}"] = blend(0.10, rets[t], start)
    series["SPY"] = spy_at[spy_at.index.astype(str) >= start]

    fig, ax = plt.subplots(figsize=(12, 6.5))
    for name, r in series.items():
        nav = (1 + r.dropna()).cumprod()
        nav = nav / nav.iloc[0] * 100
        base_line = "no fund" in name
        ax.plot(pd.to_datetime([str(d) for d in nav.index]), nav.values, label=name,
                linewidth=3.0 if base_line else (2.0 if name == "SPY" else 1.5),
                linestyle="--" if name == "SPY" else "-",
                color="black" if base_line else None, zorder=5 if base_line else 2)
    ax.set_title(f"Matched window {start} — every series rebased at the same date "
                 f"(after-tax total return)", fontsize=12)
    ax.set_ylabel("Growth of $100 (after-tax)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9, loc="upper left", ncol=2)
    fig.tight_layout()
    png = ("/home/manit/Desktop/fun_projects/mahajan_hedge_fund/output/ablation/"
           "etf_matched_window.png")
    fig.savefig(png, dpi=140)
    print(f"\nchart -> {png}")


if __name__ == "__main__":
    main()
