"""Ways to raise exposure on the book+combo portfolio, 2020-start after-tax:
   1. allocation shift (zero financing cost): book share 70->100%
   2. 1.25x book margin at three rates: 4.9% (FFR+1, idealized), 5.9% (IBKR-ish),
      10% (Fidelity retail)
"""
from __future__ import annotations
import numpy as np
import pandas as pd

import options_zoo as oz
import combined_credit as cc
from ivr_sweep import get_env, run_flex, CS, IC
from combo_stats_2020 import xstats


def main():
    env = get_env()
    mdates, r_book, spy_at, spy_gross = (env["mdates"], env["r_book"],
                                         env["spy_at"], env["spy_gross"])
    tdays, spy, vixd = env["tdays"], env["spy"], env["vixd"]
    td_dt = pd.to_datetime(tdays)
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
    intr, opt = oz.monthly_sleeve(merged, mdates)
    sleeve = (intr.reindex(r_book.index).fillna(0.0) * (1 - oz.ST)
              + opt.reindex(r_book.index).fillna(0.0) * (1 - oz.BLEND))

    def lever(r, L, rate):
        return L * r - (L - 1.0) * rate / 12.0

    rows = [("SPY", spy_at), ("Book 100%", r_book)]
    for w in (0.70, 0.80, 0.85, 0.90):
        rows.append((f"{w:.0%} book / {1-w:.0%} combo sleeve",
                     w * r_book + (1 - w) * sleeve))
    for rate, tag in [(0.049, "4.9% idealized"), (0.059, "5.9% IBKR-ish"),
                      (0.10, "10% Fidelity")]:
        rows.append((f"70/30, book 1.25x @ {tag}",
                     0.7 * lever(r_book, 1.25, rate) + 0.3 * sleeve))

    print(f"\n=== 2020-01 -> 2026-06, after-tax, $100k (alpha vs after-tax SPY) ===")
    print(f"{'':32}{'CAGR':>7}{'Vol':>7}{'Sharpe':>7}{'Sortino':>8}{'Calmar':>7}"
          f"{'maxDD':>8}{'beta':>6}{'alpha':>7}{'final $':>11}")
    for name, r in rows:
        m = xstats(r, spy_at, spy_gross)
        print(f"{name:32}{m['cagr']*100:6.1f}%{m['vol']*100:6.1f}%{m['sharpe']:7.2f}"
              f"{m['sortino']:8.2f}{m['calmar']:7.2f}{m['mdd']*100:7.1f}%{m['beta']:6.2f}"
              f"{m['alpha']*100:6.1f}%{m['final']:>11,.0f}")


if __name__ == "__main__":
    main()
