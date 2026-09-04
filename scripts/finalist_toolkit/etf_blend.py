"""ETF diversification study (user request 2026-07-20).

Tests carving W% out of the book side into an ETF:
    (0.70 - W) * book + W * etf + 0.30 * combo_sleeve
Sleeve stays at 30%; no extra capital needed.

ETF universe:
  From DB (full history): QQQ, IWM, GLD, TLT, QUAL, MTUM
  From yfinance (total-return adj_close): SCHD, JEPI, COWZ, AVUV, VYM, DGRW, QYLD, DFAC

After-tax on ETF: mark-to-liquidation at 15% LT (same as bench_nav).
W tested: 10% and 20%.
Shows CUT = "2020-01-01" aligned with combo study.
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
import combined_credit as cc
from ivr_sweep import get_env, run_flex, CS, IC
from combo_stats_2020 import xstats
from backtesting import data_loader as dl
from data.db import get_db
from run_walkforward import PANEL_START, PRICE_END

LT = fs.LT
CUT = "2020-01-01"

# Expense ratios (approximate) for mark-to-liquidation adjustment
EXPENSE = {"QQQ": 0.0020, "IWM": 0.0019, "GLD": 0.0040, "TLT": 0.0015,
           "QUAL": 0.0015, "MTUM": 0.0015,
           # yf tickers (adj_close already nets distributions; expense is tiny)
           "SCHD": 0.0006, "JEPI": 0.0035, "COWZ": 0.0049, "AVUV": 0.0025,
           "VYM": 0.0006, "DGRW": 0.0028, "QYLD": 0.0060, "DFAC": 0.0018,
           # expansion 2026-07-20: low-vol / defensive / real-asset / credit / intl
           "SPLV": 0.0025, "USMV": 0.0015, "VIG": 0.0005, "NOBL": 0.0035,
           "SPHD": 0.0030, "RSP": 0.0020, "VTV": 0.0004, "SCHG": 0.0004,
           "XLP": 0.0009, "XLU": 0.0009, "XLV": 0.0009,
           "IEF": 0.0015, "SHY": 0.0015, "LQD": 0.0014, "HYG": 0.0049,
           "VNQ": 0.0013, "EFA": 0.0033, "VWO": 0.0008,
           "SLV": 0.0050, "PDBC": 0.0059, "BTAL": 0.0143, "JEPQ": 0.0035,
           "BIL": 0.0014}


def etf_monthly_at(ticker, mdates, matrix=None):
    """After-tax monthly return series (mark-to-liq, LT 15%) on the book's dates.

    Total return via yfinance auto_adjust=True (splits + dividends reinvested).

    Two corrections vs. the original 2026-07-20 version:
      * No expense-ratio multiplier. An ETF's market price already nets its ER
        (fees come out of NAV daily), so applying it again double-counted fees.
      * The DB `matrix` path is no longer used. `daily_prices.adj_close` mixes
        conventions by source era -- price-only before 2022-06-21, dividend-
        adjusted after -- so it both drops dividends and plants a fake break
        mid-window. `matrix` is accepted for call compatibility and ignored.
    """
    dt_str = [str(d) for d in mdates]
    raw = yf.Ticker(ticker).history(start="2015-01-01", auto_adjust=True)
    px = raw["Close"].resample("ME").last()
    px.index = pd.to_datetime(px.index.strftime("%Y-%m-%d"))
    dt_idx = pd.to_datetime(dt_str)
    p = np.array([px.asof(d) for d in dt_idx], dtype=float)

    pretax = p / p[~np.isnan(p)][0]
    at = pretax - LT * np.maximum(pretax - 1.0, 0.0)
    s = pd.Series(at, index=mdates)
    r = s.pct_change().dropna()
    # mask dates before ETF inception
    first_valid = s.first_valid_index()
    r = r[r.index >= first_valid]
    return r


def main():
    env = get_env()
    mdates, r_book, spy_at, spy_gross = (env["mdates"], env["r_book"],
                                         env["spy_at"], env["spy_gross"])
    tdays, spy, vixd = env["tdays"], env["spy"], env["vixd"]
    td_dt = pd.to_datetime(tdays)

    # Build combo sleeve
    vser = pd.Series(vixd)
    lo, hi = vser.rolling(252).min(), vser.rolling(252).max()
    ivr = ((vser - lo) / (hi - lo).replace(0, np.nan) * 100).fillna(50).values
    i0 = max(252, int(np.searchsorted(np.array(tdays), "2017-01-01")))
    _, cs_d = run_flex(lambda v: (*CS, "cs"), tdays, td_dt, spy, vixd, ivr, i0)
    _, ic_d = run_flex(lambda v: (*IC, "ic") if v >= 30 else None, tdays, td_dt, spy, vixd, ivr, i0)
    merged = dict(cs_d)
    for d, v in ic_d.items(): merged[d] = merged.get(d, 0.0) + v
    intr, opt = oz.monthly_sleeve(merged, mdates)
    sleeve = (intr.reindex(r_book.index).fillna(0.0) * (1 - oz.ST)
              + opt.reindex(r_book.index).fillna(0.0) * (1 - oz.BLEND))

    # Load DB tickers
    with get_db() as db:
        matrix_etf = dl.load_price_matrix(db, ["QQQ","IWM","GLD","TLT","QUAL","MTUM"],
                                           PANEL_START, PRICE_END)

    # Build ETF return dict
    DB_TICKERS  = ["QQQ", "IWM", "GLD", "TLT", "QUAL", "MTUM"]
    YF_TICKERS  = ["SCHD", "JEPI", "COWZ", "AVUV", "VYM", "DGRW", "QYLD", "DFAC",
                   "SPLV", "USMV", "VIG", "NOBL", "SPHD", "RSP", "VTV", "SCHG",
                   "XLP", "XLU", "XLV", "IEF", "SHY", "LQD", "HYG",
                   "VNQ", "EFA", "VWO", "SLV", "PDBC", "BTAL", "JEPQ", "BIL"]
    ETF_LABELS  = {
        "QQQ":  "QQQ  (Nasdaq-100)",
        "IWM":  "IWM  (Small-cap Russell 2000)",
        "GLD":  "GLD  (Gold)",
        "TLT":  "TLT  (20yr Treasuries)",
        "QUAL": "QUAL (Quality factor)",
        "MTUM": "MTUM (Momentum factor)",
        "SCHD": "SCHD (Div. growth, value tilt)",
        "JEPI": "JEPI (EQ prem. income/CC, from May-2020)",
        "COWZ": "COWZ (Free cash flow)",
        "AVUV": "AVUV (Small-cap value)",
        "VYM":  "VYM  (High dividend yield)",
        "DGRW": "DGRW (Dividend growth)",
        "QYLD": "QYLD (Nasdaq CC income)",
        "DFAC": "DFAC (Dimensional US core, from Jun-2021)",
        "SPLV": "SPLV (S&P500 low volatility)",
        "USMV": "USMV (Min-volatility US)",
        "VIG":  "VIG  (Dividend appreciation)",
        "NOBL": "NOBL (Dividend aristocrats)",
        "SPHD": "SPHD (High div + low vol)",
        "RSP":  "RSP  (S&P500 equal weight)",
        "VTV":  "VTV  (Large-cap value)",
        "SCHG": "SCHG (Large-cap growth)",
        "XLP":  "XLP  (Consumer staples)",
        "XLU":  "XLU  (Utilities)",
        "XLV":  "XLV  (Health care)",
        "IEF":  "IEF  (7-10yr Treasuries)",
        "SHY":  "SHY  (1-3yr Treasuries)",
        "LQD":  "LQD  (IG corporate credit)",
        "HYG":  "HYG  (High-yield credit)",
        "VNQ":  "VNQ  (US REITs)",
        "EFA":  "EFA  (Developed intl ex-US)",
        "VWO":  "VWO  (Emerging markets)",
        "SLV":  "SLV  (Silver)",
        "PDBC": "PDBC (Broad commodities)",
        "BTAL": "BTAL (Anti-beta long/short)",
        "JEPQ": "JEPQ (Nasdaq prem. income, from May-2022)",
        "BIL":  "BIL  (1-3mo T-bills) *** CASH CONTROL ***",
    }

    print("Loading ETF prices...", flush=True)
    etf_ret = {}
    for t in DB_TICKERS:
        etf_ret[t] = etf_monthly_at(t, mdates, matrix_etf)
    for t in YF_TICKERS:
        etf_ret[t] = etf_monthly_at(t, mdates)
        print(f"  {t} loaded", flush=True)

    def blend(w_etf, r_e, r_b, r_s):
        """Carve w_etf from book; sleeve stays 30%."""
        idx = r_b.index.intersection(r_e.index).intersection(r_s.index)
        idx = idx[idx.astype(str) >= CUT]
        return ((0.70 - w_etf) * r_b.reindex(idx)
                + w_etf * r_e.reindex(idx)
                + 0.30 * r_s.reindex(idx))

    def show_block(label, rows, spy_at, spy_gross):
        print(f"\n{'':36}{'CAGR':>7}{'Vol':>6}{'Shrp':>6}{'Sort':>6}{'Calm':>6}"
              f"{'maxDD':>8}{'beta':>6}{'α/yr':>6}{'final $':>11}")
        for name, r in rows:
            rr = r[r.index.astype(str) >= CUT].dropna()
            sa = spy_at[spy_at.index.astype(str) >= CUT].dropna()
            sg = spy_gross[spy_gross.index.astype(str) >= CUT].dropna()
            if len(rr) < 6: continue
            m = xstats(rr, sa, sg)
            print(f"{name:36}{m['cagr']*100:6.1f}%{m['vol']*100:5.1f}%"
                  f"{m['sharpe']:6.2f}{m['sortino']:6.2f}{m['calmar']:6.2f}"
                  f"{m['mdd']*100:7.1f}%{m['beta']:6.2f}{m['alpha']*100:5.1f}%"
                  f"{m['final']:>11,.0f}")

    print(f"\n{'='*90}")
    print(f"=== 2020-start after-tax: carving 10% / 20% from book into ETF "
          f"(70/30 → [60%book+10%ETF+30%sleeve] or [50%+20%+30%]) ===")
    print(f"{'='*90}")

    baselines = [
        ("SPY                             ", spy_at),
        ("Book 100% (no sleeve)           ", r_book),
        ("Combo 70/30 (no ETF)            ", 0.7 * r_book + 0.3 * sleeve),
    ]

    show_block("baselines", baselines, spy_at, spy_gross)

    def row_stats(r):
        rr = r[r.index.astype(str) >= CUT].dropna()
        if len(rr) < 6: return None
        sa = spy_at[spy_at.index.astype(str) >= CUT].dropna()
        sg = spy_gross[spy_gross.index.astype(str) >= CUT].dropna()
        return xstats(rr, sa, sg), len(rr)

    base = row_stats(0.7 * r_book + 0.3 * sleeve)[0]

    recs = []
    for ticker, label in ETF_LABELS.items():
        r_e = etf_ret[ticker]
        solo = row_stats(r_e)
        bl10 = row_stats(blend(0.10, r_e, r_book, sleeve))
        if solo is None or bl10 is None: continue
        recs.append((ticker, label, solo[0], bl10[0], bl10[1]))

    recs.sort(key=lambda x: -x[3]["sharpe"])

    hdr = (f"\n{'ETF':6}{'n':>4}  {'--- ETF standalone ---':^34}   "
           f"{'--- 10% carve into book ---':^40}")
    sub = (f"{'':10}  {'CAGR':>6}{'Shrp':>6}{'Sort':>6}{'Calm':>6}{'maxDD':>7}{'beta':>6}   "
           f"{'CAGR':>6}{'Shrp':>6}{'Sort':>6}{'Calm':>6}{'maxDD':>7}{'beta':>6}{'ΔShrp':>7}")
    print(f"\n{'='*100}")
    print("=== RANKED: 10% carve-out (63% book / 27% sleeve equiv. shown as 60/10/30) ===")
    print(f"=== sorted by blended Sharpe; ΔShrp vs Combo 70/30 baseline {base['sharpe']:.2f} ===")
    print(f"{'='*100}{hdr}\n{sub}")
    for t, label, s, b, n in recs:
        print(f"{t:6}{n:>4}  {s['cagr']*100:5.1f}%{s['sharpe']:6.2f}{s['sortino']:6.2f}"
              f"{s['calmar']:6.2f}{s['mdd']*100:6.1f}%{s['beta']:6.2f}   "
              f"{b['cagr']*100:5.1f}%{b['sharpe']:6.2f}{b['sortino']:6.2f}"
              f"{b['calmar']:6.2f}{b['mdd']*100:6.1f}%{b['beta']:6.2f}"
              f"{b['sharpe']-base['sharpe']:+7.2f}")
    print(f"\nn = months in common window (varies by ETF inception; NOT comparable across rows)")
    print(f"BIL is the cash control: any ETF with ΔShrp <= BIL's adds nothing a cash "
          f"carve-out wouldn't.")


if __name__ == "__main__":
    main()
