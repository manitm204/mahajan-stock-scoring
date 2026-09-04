"""Tax-correct fund comparison (user request 2026-07-20).

The earlier sweeps taxed every fund at 15% long-term on a mark-to-liquidation
basis. That is wrong in three directions at once:

  * GLD / SLV are collectibles -- long-term gains taxed up to 28%, not 15%.
  * JEPI / JEPQ / QYLD distribute mostly ELN / option premium, which is
    ORDINARY income (32% here), not qualified dividends.
  * BIL and every bond fund distribute interest -- also ordinary income.
    So the "cash control" in the earlier runs was itself overstated.

This splits each fund's total return into a price leg and a distribution leg
(total return minus price return), taxes the distribution leg every month at
the fund's own rate, and taxes the price leg at its long-term rate on
liquidation. Also sweeps GLD sizing, since 10% was an arbitrary starting point.

Assumption worth naming: active mutual funds are charged 20% on distributions
to approximate their mix of qualified dividends and short-term capital gain
distributions. That is a rough number and it flatters nothing in the conclusion.
"""
from __future__ import annotations
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/manit/Desktop/fun_projects/mahajan_hedge_fund")

import numpy as np
import pandas as pd
import yfinance as yf

import options_zoo as oz
import final_stats as fs
from ivr_sweep import get_env, run_flex, CS, IC
from combo_stats_2020 import xstats

CUT = sys.argv[1] if len(sys.argv) > 1 else "2020-01-01"
ORD, LT, COLL = fs.ST, fs.LT, 0.28          # 0.32 ordinary, 0.15 LT, 0.28 collectibles

#  ticker: (label, distribution tax rate, gain tax rate)
TAX = {
    "GLD":  ("Gold",                        0.00, COLL),
    "SLV":  ("Silver",                      0.00, COLL),
    "PDBC": ("Broad commodities",           ORD,  LT),
    "BIL":  ("T-bills  *** CONTROL ***",    ORD,  LT),
    "JAAA": ("Janus AAA CLO",               ORD,  LT),
    "PULS": ("PGIM Ultra Short",            ORD,  LT),
    "TLT":  ("20yr Treasuries",             ORD,  LT),
    "JEPI": ("EQ premium income",           ORD,  LT),
    "JEPQ": ("Nasdaq premium income",       ORD,  LT),
    "QYLD": ("Nasdaq covered call",         ORD,  LT),
    "SCHD": ("Div growth / value",          LT,   LT),   # qualified dividends
    "VIG":  ("Dividend appreciation",       LT,   LT),
    "XLU":  ("Utilities",                   LT,   LT),
    "VTV":  ("Large-cap value",             LT,   LT),
    "QQQ":  ("Nasdaq-100",                  LT,   LT),
    "SPLV": ("S&P500 low volatility",       LT,   LT),
    "PRWCX":("T Rowe Cap Appreciation",     0.20, LT),   # active MF, see docstring
    "FCNTX":("Fidelity Contrafund",         0.20, LT),
    "VWIAX":("Vanguard Wellesley",          0.20, LT),
}


def legs(ticker, mdates):
    """Monthly (price return, distribution return) on the book's dates."""
    tr = yf.Ticker(ticker).history(start="2015-01-01", auto_adjust=True)["Close"]
    px = yf.Ticker(ticker).history(start="2015-01-01", auto_adjust=False)["Close"]
    out = []
    for s in (tr, px):
        m = s.resample("ME").last()
        m.index = pd.to_datetime(m.index.strftime("%Y-%m-%d"))
        idx = pd.to_datetime([str(d) for d in mdates])
        out.append(pd.Series([m.asof(d) for d in idx], index=mdates, dtype=float))
    tr_m, px_m = out
    tr_r, px_r = tr_m.pct_change(), px_m.pct_change()
    return px_r, (tr_r - px_r)


def after_tax(ticker, mdates):
    """After-tax monthly return: distributions taxed as they arrive, price
    appreciation taxed on liquidation at the instrument's own rate."""
    _, dist_rate, gain_rate = TAX[ticker]
    px_r, dist_r = legs(ticker, mdates)
    net = px_r + dist_r * (1 - dist_rate)
    nav = (1 + net.fillna(0)).cumprod()
    nav = nav / nav[net.first_valid_index()]
    # mark to liquidation on the price-driven gain only
    liq = nav - gain_rate * np.maximum(nav - 1.0, 0.0)
    r = pd.Series(liq, index=mdates).pct_change().dropna()
    return r[r.index >= net.first_valid_index()]


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
    combo = 0.7 * r_book + 0.3 * sleeve

    def stats_on(r):
        rr = r[r.index.astype(str) >= CUT].dropna()
        if len(rr) < 24: return None
        sa = spy_at[spy_at.index.astype(str) >= CUT].dropna()
        sg = spy_gross[spy_gross.index.astype(str) >= CUT].dropna()
        return xstats(rr, sa, sg), len(rr)

    def blend(w, r_e):
        idx = r_book.index.intersection(r_e.index).intersection(sleeve.index)
        idx = idx[idx.astype(str) >= CUT]
        return ((0.70 - w) * r_book.reindex(idx) + w * r_e.reindex(idx)
                + 0.30 * sleeve.reindex(idx))

    base, nb = stats_on(combo)
    print(f"\nWindow {CUT} ->  baseline Combo 70/30: CAGR {base['cagr']*100:.1f}%  "
          f"Sharpe {base['sharpe']:.2f}  maxDD {base['mdd']*100:.1f}%  n={nb}")

    print(f"\n{'='*104}")
    print("=== TAX-CORRECT: distributions taxed at their own rate, gains at theirs ===")
    print(f"{'='*104}")
    print(f"{'FUND':7}{'dist':>6}{'gain':>6}{'n':>5}  "
          f"{'--- standalone ---':^26}  {'--- 10% blended ---':^32}")
    print(f"{'':24}{'CAGR':>7}{'Shrp':>6}{'maxDD':>8}  "
          f"{'CAGR':>7}{'Shrp':>6}{'Sort':>6}{'Calm':>6}{'maxDD':>8}{'ΔShrp':>8}")

    res = {}
    for t in TAX:
        try:
            r = after_tax(t, mdates)
        except Exception as e:
            print(f"{t:7} FAILED {str(e)[:40]}"); continue
        s, b = stats_on(r), stats_on(blend(0.10, r))
        if s is None or b is None: continue
        res[t] = (s[0], b[0], b[1])

    order = sorted(res, key=lambda t: -res[t][1]["sharpe"])
    for t in order:
        s, b, n = res[t]
        lab, dr, gr = TAX[t]
        print(f"{t:7}{dr*100:5.0f}%{gr*100:5.0f}%{n:>5}  "
              f"{s['cagr']*100:6.1f}%{s['sharpe']:6.2f}{s['mdd']*100:7.1f}%  "
              f"{b['cagr']*100:6.1f}%{b['sharpe']:6.2f}{b['sortino']:6.2f}"
              f"{b['calmar']:6.2f}{b['mdd']*100:7.1f}%"
              f"{b['sharpe']-base['sharpe']:+8.2f}   {lab}")

    if "BIL" in res:
        d = res["BIL"][1]["sharpe"] - base["sharpe"]
        beat = [t for t in order if res[t][1]["sharpe"] - base["sharpe"] > d + 0.004]
        print(f"\nCash control (tax-correct) = {d:+.2f}.  Beating it: "
              f"{', '.join(beat) if beat else 'NONE'}")

    # ---- GLD sizing: 10% was arbitrary ----
    print(f"\n{'='*104}")
    print("=== GLD sizing sweep, collectibles tax (28%) ===")
    print(f"{'='*104}")
    print(f"{'weight':>8}{'CAGR':>8}{'Shrp':>7}{'Sort':>7}{'Calm':>7}{'maxDD':>8}{'beta':>7}{'ΔShrp':>8}")
    rg = after_tax("GLD", mdates)
    print(f"{'0% ':>8}{base['cagr']*100:7.1f}%{base['sharpe']:7.2f}{base['sortino']:7.2f}"
          f"{base['calmar']:7.2f}{base['mdd']*100:7.1f}%{base['beta']:7.2f}{0.0:+8.2f}")
    for w in (0.05, 0.10, 0.15, 0.20, 0.30):
        m, _ = stats_on(blend(w, rg))
        print(f"{w*100:7.0f}%{m['cagr']*100:7.1f}%{m['sharpe']:7.2f}{m['sortino']:7.2f}"
              f"{m['calmar']:7.2f}{m['mdd']*100:7.1f}%{m['beta']:7.2f}"
              f"{m['sharpe']-base['sharpe']:+8.2f}")

    # ---- Does the case survive a worse gold market? ----
    # 2020-26 gold ran at ~13% after-tax. Long-run nominal gold is closer to
    # 4-5%. Subtract a constant drag from the monthly series, keeping its
    # volatility and correlation intact, and see where the edge dies.
    print(f"\n{'='*104}")
    print("=== GLD return haircut at 10% weight -- keeps vol/correlation, cuts return ===")
    print(f"=== cash control (tax-correct) = {res['BIL'][1]['sharpe']-base['sharpe']:+.2f} ===")
    print(f"{'='*104}")
    print(f"{'haircut':>9}{'GLD CAGR':>10}{'CAGR':>8}{'Shrp':>7}{'Calm':>7}{'maxDD':>8}{'ΔShrp':>8}  verdict")
    bil_d = res["BIL"][1]["sharpe"] - base["sharpe"]
    for hc in (0.0, 0.03, 0.06, 0.09, 0.12, 0.15):
        rg_h = rg - hc / 12.0
        solo, _ = stats_on(rg_h)
        m, _ = stats_on(blend(0.10, rg_h))
        dd = m["sharpe"] - base["sharpe"]
        verdict = ("beats cash" if dd > bil_d + 0.004 else
                   "ties cash" if dd > bil_d - 0.004 else "loses to cash")
        print(f"{hc*100:8.0f}%{solo['cagr']*100:9.1f}%{m['cagr']*100:7.1f}%"
              f"{m['sharpe']:7.2f}{m['calmar']:7.2f}{m['mdd']*100:7.1f}%{dd:+8.2f}  {verdict}")


if __name__ == "__main__":
    main()
