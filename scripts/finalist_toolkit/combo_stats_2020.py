"""Full stat sheet, 2020-start after-tax: 70/30 book + combo sleeve
(CS always + IC when IVR>=30) vs SPY / QQQ / book, plus 1.25x-book leverage.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

import options_zoo as oz
import final_stats as fs
import combined_credit as cc
from ivr_sweep import get_env, run_flex, CS, IC
from research.walkforward.portfolio import _sortino
from backtesting import data_loader as dl
from data.db import get_db
from run_walkforward import PANEL_START, PRICE_END

CUT = "2020-01-01"


def xstats(r, bench_at, spy_gross):
    r = r[r.index.astype(str) >= CUT].dropna()
    b = bench_at[bench_at.index.astype(str) >= CUT].dropna()
    g = spy_gross[spy_gross.index.astype(str) >= CUT].dropna()
    eq = (1 + r).cumprod()
    cagr = eq.iloc[-1] ** (12 / len(r)) - 1
    vol = r.std(ddof=1) * np.sqrt(12)
    mdd = float((eq / eq.cummax() - 1).min())
    df = pd.concat([r.rename("p"), g.rename("g"), b.rename("b")], axis=1).dropna()
    beta = float(df.cov().loc["p", "g"] / df["g"].var(ddof=1))
    beta_at = float(df.cov().loc["p", "b"] / df["b"].var(ddof=1))
    alpha = (df["p"].mean() - beta_at * df["b"].mean()) * 12
    return dict(cagr=cagr, vol=vol, sharpe=r.mean() / r.std(ddof=1) * np.sqrt(12),
                sortino=_sortino(r, 12.0), calmar=cagr / abs(mdd) if mdd else np.nan,
                mdd=mdd, beta=beta, alpha=alpha, best=r.max(), worst=r.min(),
                pos=(r > 0).mean(), final=fs.START * float(eq.iloc[-1]))


def main():
    env = get_env()
    mdates, r_book, spy_at, spy_gross = (env["mdates"], env["r_book"],
                                         env["spy_at"], env["spy_gross"])
    tdays, spy, vixd = env["tdays"], env["spy"], env["vixd"]
    td_dt = pd.to_datetime(tdays)

    with get_db() as db:
        qqq_m = dl.load_price_matrix(db, ["QQQ"], PANEL_START, PRICE_END)
    qqq_at = pd.Series(fs.bench_nav(qqq_m, mdates, "QQQ"), index=mdates).pct_change().dropna()

    vser = pd.Series(vixd)
    lo, hi = vser.rolling(252).min(), vser.rolling(252).max()
    ivr = ((vser - lo) / (hi - lo).replace(0, np.nan) * 100).fillna(50).values
    i0 = max(252, int(np.searchsorted(np.array(tdays), "2017-01-01")))

    _, cs_daily = run_flex(lambda v: (*CS, "cs"), tdays, td_dt, spy, vixd, ivr, i0)
    _, ic_daily = run_flex(lambda v: (*IC, "ic") if v >= 30 else None,
                           tdays, td_dt, spy, vixd, ivr, i0)
    merged = dict(cs_daily)
    for d, v in ic_daily.items():
        merged[d] = merged.get(d, 0.0) + v
    sleeves = {}
    for name, daily in [("cs", cs_daily), ("combo", merged)]:
        intr, opt = oz.monthly_sleeve(daily, mdates)
        sleeves[name] = (intr.reindex(r_book.index).fillna(0.0) * (1 - oz.ST)
                         + opt.reindex(r_book.index).fillna(0.0) * (1 - oz.BLEND))

    def lever(r, L):
        ffr = pd.Series([cc.FFR[int(str(d)[:4])] for d in r.index], index=r.index)
        return L * r - (L - 1.0) * (ffr + cc.MARGIN_SPREAD) / 12.0

    book125 = lever(r_book, 1.25)
    rows = [
        ("SPY (after-tax)", spy_at),
        ("QQQ (after-tax)", qqq_at),
        ("Book 100%", r_book),
        ("70/30 + CS (incumbent)", 0.7 * r_book + 0.3 * sleeves["cs"]),
        ("70/30 + CS&IC combo", 0.7 * r_book + 0.3 * sleeves["combo"]),
        ("Book 1.25x", book125),
        ("70/30 combo, book 1.25x", 0.7 * book125 + 0.3 * sleeves["combo"]),
    ]
    print(f"\n=== 2020-01 -> 2026-06, after-tax, $100k start (alpha vs after-tax SPY) ===")
    print(f"{'':26}{'CAGR':>7}{'Vol':>7}{'Sharpe':>7}{'Sortino':>8}{'Calmar':>7}"
          f"{'maxDD':>8}{'beta':>6}{'alpha':>7}{'best_m':>7}{'worst_m':>8}{'pos_m':>6}{'final $':>11}")
    for name, r in rows:
        m = xstats(r, spy_at, spy_gross)
        print(f"{name:26}{m['cagr']*100:6.1f}%{m['vol']*100:6.1f}%{m['sharpe']:7.2f}"
              f"{m['sortino']:8.2f}{m['calmar']:7.2f}{m['mdd']*100:7.1f}%{m['beta']:6.2f}"
              f"{m['alpha']*100:6.1f}%{m['best']*100:6.1f}%{m['worst']*100:7.1f}%"
              f"{m['pos']*100:5.0f}%{m['final']:>11,.0f}")


if __name__ == "__main__":
    main()
