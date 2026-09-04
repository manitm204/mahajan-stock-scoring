"""Additive combo (follow-up to ivr_sweep): always-on call spread PLUS an
iron condor opened only when IVR >= t (two independent positions, F=10% each).
Contrast with the replacement hybrid, which swapped the hedge out in high vol.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

import options_zoo as oz
import combined_credit as cc
from ivr_sweep import get_env, run_flex, CS, IC


def main():
    env = get_env()
    r_book, spy_gross, mdates = env["r_book"], env["spy_gross"], env["mdates"]
    tdays, spy, vixd = env["tdays"], env["spy"], env["vixd"]
    td_dt = pd.to_datetime(tdays)
    vser = pd.Series(vixd)
    lo, hi = vser.rolling(252).min(), vser.rolling(252).max()
    ivr = ((vser - lo) / (hi - lo).replace(0, np.nan) * 100).fillna(50).values
    i0 = max(252, int(np.searchsorted(np.array(tdays), "2017-01-01")))

    cs_tr, cs_daily = run_flex(lambda v: (*CS, "cs"), tdays, td_dt, spy, vixd, ivr, i0)
    combos = {}
    for t in (30, 40):
        ic_tr, ic_daily = run_flex(lambda v: (*IC, "ic") if v >= t else None,
                                   tdays, td_dt, spy, vixd, ivr, i0)
        merged = dict(cs_daily)
        for d, v in ic_daily.items():
            merged[d] = merged.get(d, 0.0) + v
        combos[f"CS always + IC IVR>={t} (additive)"] = merged
    combos["CS always (incumbent)"] = cs_daily

    for wlabel, cut in [("FULL 2017 -> 2026", "2000"), ("2020-start", "2020-01-01")]:
        sg = spy_gross[spy_gross.index.astype(str) >= cut]
        print(f"\n--- {wlabel} (after-tax, 70/30 blend) ---")
        print(f"{'':38}{'slvCAGR':>8}{'slvShp':>7} |{'CAGR':>7}{'Sharpe':>8}"
              f"{'maxDD':>8}{'beta':>7}{'final $':>12}")
        for label, daily in combos.items():
            intr, opt = oz.monthly_sleeve(daily, mdates)
            intr2 = intr.reindex(r_book.index).fillna(0.0)
            opt2 = opt.reindex(r_book.index).fillna(0.0)
            sleeve = intr2 * (1 - oz.ST) + opt2 * (1 - oz.BLEND)
            a = cc.stats(sleeve[sleeve.index.astype(str) >= cut].dropna(), sg)
            r = 0.7 * r_book + 0.3 * sleeve
            b = cc.stats(r[r.index.astype(str) >= cut].dropna(), sg)
            print(f"{label:38}{a['cagr']*100:7.1f}%{a['sharpe']:7.2f} |{b['cagr']*100:6.1f}%"
                  f"{b['sharpe']:8.2f}{b['mdd']*100:7.1f}%{b['beta']:7.2f}{b['final']:>12,.0f}")


if __name__ == "__main__":
    main()
