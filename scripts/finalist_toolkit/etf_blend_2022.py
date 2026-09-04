"""ETF / active-fund diversification study (user request 2026-07-20).

Carves 10% out of the book side:  (0.70 - W)*book + W*fund + 0.30*sleeve

Start date is argv[1], default 2020-01-01 -- chosen so the window contains the
COVID crash. Funds that did not exist at the start date are reported in a
separate block and never ranked against full-history funds: a fund that skips a
drawdown looks better than one that lived through it.

Corrections vs. the first pass:
  * Total return everywhere -- yfinance auto_adjust=True, so dividends and
    distributions are reinvested. No DB adj_close (mixed conventions, fake
    break at 2022-06-21 that would sit inside this window).
  * No expense-ratio multiplier. Fund market price already nets the ER daily;
    applying it again double-counted fees. ER is reported for context only.
  * Adds actively managed ETFs and mutual funds alongside the passive set.

BIL (T-bills) is the control: any fund not beating BIL's delta is not
diversifying, it is just holding less book.
"""
from __future__ import annotations
import sys, warnings, json
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/manit/Desktop/fun_projects/mahajan_hedge_fund")

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import options_zoo as oz
import etf_blend as eb
from ivr_sweep import get_env, run_flex, CS, IC
from combo_stats_2020 import xstats

CUT = sys.argv[1] if len(sys.argv) > 1 else "2020-01-01"
TAG = CUT[:4]
OUT = "/home/manit/Desktop/fun_projects/mahajan_hedge_fund/output/ablation"

# label, expense ratio (reported, NOT applied), management style
FUNDS = {
    # --- passive, carried over ---
    "GLD":  ("Gold",                              0.0040, "passive"),
    "SLV":  ("Silver",                            0.0050, "passive"),
    "PDBC": ("Broad commodities",                 0.0059, "active"),
    "BIL":  ("1-3mo T-bills  *** CONTROL ***",    0.0014, "passive"),
    "SCHD": ("Div growth / value tilt",           0.0006, "passive"),
    "SPLV": ("S&P500 low volatility",             0.0025, "passive"),
    "USMV": ("Min-volatility US",                 0.0015, "passive"),
    "VIG":  ("Dividend appreciation",             0.0005, "passive"),
    "QQQ":  ("Nasdaq-100",                        0.0020, "passive"),
    "IWM":  ("Small-cap Russell 2000",            0.0019, "passive"),
    "TLT":  ("20yr Treasuries",                   0.0015, "passive"),
    "IEF":  ("7-10yr Treasuries",                 0.0015, "passive"),
    "LQD":  ("IG corporate credit",               0.0014, "passive"),
    "HYG":  ("High-yield credit",                 0.0049, "passive"),
    "VNQ":  ("US REITs",                          0.0013, "passive"),
    "EFA":  ("Developed intl ex-US",              0.0033, "passive"),
    "VWO":  ("Emerging markets",                  0.0008, "passive"),
    "XLU":  ("Utilities",                         0.0009, "passive"),
    "XLP":  ("Consumer staples",                  0.0009, "passive"),
    "XLV":  ("Health care",                       0.0009, "passive"),
    "RSP":  ("S&P500 equal weight",               0.0020, "passive"),
    "VTV":  ("Large-cap value",                   0.0004, "passive"),
    "BTAL": ("Anti-beta long/short",              0.0143, "active"),
    # --- active ETFs ---
    "JEPI": ("EQ premium income (covered call)",  0.0035, "active"),
    "JEPQ": ("Nasdaq premium income",             0.0035, "active"),
    "QYLD": ("Nasdaq covered call",               0.0060, "active"),
    "ARKK": ("ARK Innovation",                    0.0075, "active"),
    "ARKG": ("ARK Genomic Revolution",            0.0075, "active"),
    "CGDV": ("Capital Group Div Value",           0.0033, "active"),
    "CGGR": ("Capital Group Growth",              0.0039, "active"),
    "FBCG": ("Fidelity Blue Chip Growth",         0.0059, "active"),
    "AVUV": ("Avantis Small-cap Value",           0.0025, "active"),
    "AVUS": ("Avantis US Equity",                 0.0015, "active"),
    "DFAC": ("Dimensional US Core",               0.0018, "active"),
    "COWZ": ("Free cash flow",                    0.0049, "active"),
    "BOND": ("PIMCO Active Bond",                 0.0055, "active"),
    "FBND": ("Fidelity Total Bond",               0.0036, "active"),
    "JAAA": ("Janus AAA CLO",                     0.0021, "active"),
    "MINT": ("PIMCO Enhanced Short Maturity",     0.0035, "active"),
    "PULS": ("PGIM Ultra Short Bond",             0.0015, "active"),
    # --- active mutual funds ---
    "PRWCX": ("T Rowe Capital Appreciation",      0.0069, "active MF"),
    "DODGX": ("Dodge & Cox Stock",                0.0051, "active MF"),
    "DODBX": ("Dodge & Cox Balanced",             0.0053, "active MF"),
    "VWIAX": ("Vanguard Wellesley Income",        0.0016, "active MF"),
    "VWELX": ("Vanguard Wellington",              0.0025, "active MF"),
    "FBALX": ("Fidelity Balanced",                0.0048, "active MF"),
    "FCNTX": ("Fidelity Contrafund",              0.0039, "active MF"),
    "TRBCX": ("T Rowe Blue Chip Growth",          0.0069, "active MF"),
    "PRNHX": ("T Rowe New Horizons",              0.0075, "active MF"),
    "VDIGX": ("Vanguard Dividend Growth",         0.0029, "active MF"),
    "FLPSX": ("Fidelity Low-Priced Stock",        0.0078, "active MF"),
    "POAGX": ("PRIMECAP Odyssey Aggressive",      0.0066, "active MF"),
}


def dividend_check(tickers=("SCHD", "JEPI", "VWIAX", "TLT")):
    """Prove dividends are actually being captured: TR vs price-only CAGR."""
    print(f"\nDividend capture check (annualized, {CUT} -> now):")
    print(f"{'':8}{'price-only':>12}{'total-ret':>12}{'div contrib':>13}")
    for t in tickers:
        tr = yf.Ticker(t).history(start=CUT, auto_adjust=True)["Close"]
        po = yf.Ticker(t).history(start=CUT, auto_adjust=False)["Close"]
        yrs = len(tr) / 252
        g = lambda s: (s.iloc[-1] / s.iloc[0]) ** (1 / yrs) - 1
        print(f"{t:8}{g(po)*100:11.2f}%{g(tr)*100:11.2f}%{(g(tr)-g(po))*100:12.2f}%")


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

    dividend_check()

    print("\nLoading funds (total return, yfinance)...", flush=True)
    rets, firsts, missing = {}, {}, []
    for t in FUNDS:
        try:
            r = eb.etf_monthly_at(t, mdates)
            rr = r[r.index.astype(str) >= CUT].dropna()
            if rr.shape[0] < 24:
                missing.append((t, "under 24 months")); continue
            rets[t] = r
            firsts[t] = str(r.index.min())
        except Exception as e:
            missing.append((t, str(e)[:50]))
    print(f"  loaded {len(rets)}, dropped {len(missing)}")
    for t, why in missing:
        print(f"  DROPPED {t}: {why}")

    # Funds whose history starts after CUT cannot be ranked against the rest:
    # they skip whatever the market did before they existed.
    partial = {t: f for t, f in firsts.items() if f > CUT}
    if partial:
        print(f"\nPARTIAL HISTORY (inception after {CUT}) -- reported separately, "
              f"NOT ranked against full-history funds:")
        for t, f in sorted(partial.items(), key=lambda kv: kv[1]):
            print(f"  {t:7} starts {f}   ({FUNDS[t][0]})")

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

    combo = 0.7 * r_book + 0.3 * sleeve
    base, nb = stats_on(combo)
    spy_m = stats_on(spy_at)[0]

    recs, recs_partial = [], []
    for t in rets:
        solo, blended = stats_on(rets[t]), stats_on(blend(0.10, rets[t]))
        if solo is None or blended is None: continue
        row = (t, FUNDS[t][0], FUNDS[t][1], FUNDS[t][2],
               solo[0], blended[0], blended[1])
        (recs_partial if t in partial else recs).append(row)
    recs.sort(key=lambda x: -x[5]["sharpe"])
    recs_partial.sort(key=lambda x: -x[5]["sharpe"])

    print(f"\n{'='*118}")
    print(f"=== 10% carve-out, {CUT} forward, after-tax, TOTAL RETURN "
          f"(dividends in, ER not double-counted) ===")
    print(f"=== baseline Combo 70/30: Sharpe {base['sharpe']:.2f}  "
          f"CAGR {base['cagr']*100:.1f}%  n={nb} months ===")
    print(f"{'='*118}")
    print(f"{'FUND':7}{'style':10}{'ER':>6}{'n':>4}  "
          f"{'---------- standalone ----------':^36}  "
          f"{'---------- 10% blended ----------':^38}")
    print(f"{'':27}  {'CAGR':>6}{'Shrp':>6}{'Sort':>6}{'Calm':>6}{'maxDD':>7}{'beta':>6}  "
          f"{'CAGR':>6}{'Shrp':>6}{'Sort':>6}{'Calm':>6}{'maxDD':>7}{'beta':>6}{'ΔShrp':>7}")
    def emit(rs):
        for t, lab, er, style, s, b, n in rs:
            star = "*" if t in partial else " "
            print(f"{t:6}{star}{style:10}{er*100:5.2f}%{n:>4}  "
                  f"{s['cagr']*100:5.1f}%{s['sharpe']:6.2f}{s['sortino']:6.2f}"
                  f"{s['calmar']:6.2f}{s['mdd']*100:6.1f}%{s['beta']:6.2f}  "
                  f"{b['cagr']*100:5.1f}%{b['sharpe']:6.2f}{b['sortino']:6.2f}"
                  f"{b['calmar']:6.2f}{b['mdd']*100:6.1f}%{b['beta']:6.2f}"
                  f"{b['sharpe']-base['sharpe']:+7.2f}   {lab}")

    emit(recs)
    if recs_partial:
        print(f"\n{'-'*118}")
        print(f"PARTIAL HISTORY -- these funds did not exist for the whole window. "
              f"Their numbers are NOT comparable to the block above.")
        print(f"{'-'*118}")
        emit(recs_partial)

    bil = [r for r in recs if r[0] == "BIL"]
    if bil:
        d = bil[0][5]["sharpe"] - base["sharpe"]
        beat = [r[0] for r in recs if r[5]["sharpe"] - base["sharpe"] > d]
        print(f"\nCash control BIL delta = {d:+.2f}. Funds beating it: "
              f"{', '.join(beat) if beat else 'NONE'}")

    # ---- chart: equity curves (full-history funds only, common rebase) ----
    top = [r[0] for r in recs[:5]]
    keep = [t for t in dict.fromkeys(top + ["BIL", "GLD", "SCHD"]) if t not in partial]
    curves = {"Combo 70/30 (no fund)": combo[combo.index.astype(str) >= CUT]}
    for t in keep:
        if t in rets:
            curves[f"+10% {t}"] = blend(0.10, rets[t])
    curves["SPY"] = spy_at[spy_at.index.astype(str) >= CUT]

    fig, ax = plt.subplots(figsize=(12, 6.5))
    for name, r in curves.items():
        nav = (1 + r.dropna()).cumprod()
        nav = nav / nav.iloc[0] * 100
        lw = 2.8 if "no fund" in name else (2.2 if name == "SPY" else 1.6)
        ls = "-" if "no fund" in name else ("--" if name == "SPY" else "-")
        ax.plot(pd.to_datetime([str(d) for d in nav.index]), nav.values,
                label=name, linewidth=lw, linestyle=ls)
    ax.set_title(f"10% carve-out into a fund vs. Combo 70/30 — after-tax total return, "
                 f"{CUT} forward", fontsize=12)
    ax.set_ylabel("Growth of $100 (after-tax)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9, loc="upper left")
    fig.tight_layout()
    png = f"{OUT}/etf_blend_{TAG}.png"
    fig.savefig(png, dpi=140)
    print(f"\nchart -> {png}")

    mk = lambda rs: [
        {"ticker": t, "label": lab, "er": er, "style": st, "n": n,
         "first": firsts[t], "partial": t in partial,
         "solo": {k: float(s[k]) for k in ("cagr","sharpe","sortino","calmar","mdd","beta")},
         "blend": {k: float(b[k]) for k in ("cagr","sharpe","sortino","calmar","mdd","beta")},
         "dsharpe": float(b["sharpe"] - base["sharpe"])}
        for t, lab, er, st, s, b, n in rs]
    rows, rows_partial = mk(recs), mk(recs_partial)
    meta = {"cut": CUT, "n_base": nb, "rows_partial": rows_partial,
            "base": {k: float(base[k]) for k in ("cagr","sharpe","sortino","calmar","mdd","beta")},
            "spy": {k: float(spy_m[k]) for k in ("cagr","sharpe","sortino","calmar","mdd","beta")},
            "rows": rows}
    with open(f"{OUT}/etf_blend_{TAG}.json", "w") as f:
        json.dump(meta, f, indent=1)
    print(f"data  -> {OUT}/etf_blend_{TAG}.json")


if __name__ == "__main__":
    main()
