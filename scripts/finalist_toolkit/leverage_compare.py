"""1.25x vs 1.5x (and beyond) on the 70/30 book (user request 2026-07-20).

Levers the BOOK leg only, as leverage_paths.py does:
    r = 0.70 * (L*r_book - (L-1)*rate/12) + 0.30 * sleeve

Reports the return metrics AND the thing that actually decides this question:
how close each leverage level runs to a forced liquidation, and what the
leverage ratio drifts up to after a drawdown (leverage is not static -- it
rises exactly when you least want it to).
"""
from __future__ import annotations
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/manit/Desktop/fun_projects/mahajan_hedge_fund")

import numpy as np
import pandas as pd

import options_zoo as oz
from ivr_sweep import get_env, run_flex, CS, IC
from combo_stats_2020 import xstats

CUT = "2020-01-01"
RATE = 0.064            # IBKR Lite, per SETUP_LEVERED_IBKR.md
MAINT = 0.25            # Reg T maintenance; IBKR house is often higher
LEVELS = (1.00, 1.25, 1.50, 1.75, 2.00)


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

    m = lambda r: xstats(r[r.index.astype(str) >= CUT].dropna(),
                         spy_at[spy_at.index.astype(str) >= CUT].dropna(),
                         spy_gross[spy_gross.index.astype(str) >= CUT].dropna())

    lever = lambda r, L, rate: L * r - (L - 1.0) * rate / 12.0

    # worst peak-to-trough on the *unlevered book*, used for the drift math
    bk = r_book[r_book.index.astype(str) >= CUT].dropna()
    nav = (1 + bk).cumprod()
    book_dd = float((nav / nav.cummax() - 1).min())

    print(f"\nWindow {CUT} onward.  Margin rate {RATE:.1%} (IBKR Lite).")
    print(f"Worst unlevered book drawdown in sample: {book_dd*100:.1f}%")

    print(f"\n{'='*104}")
    print("=== RETURNS ===")
    print(f"{'':26}{'CAGR':>7}{'Vol':>7}{'Shrp':>7}{'Sort':>7}{'Calm':>7}"
          f"{'maxDD':>8}{'beta':>6}{'α/yr':>7}{'final $15k':>12}")
    res = {}
    for L in LEVELS:
        r = 0.70 * lever(r_book, L, RATE) + 0.30 * sleeve
        s = m(r); res[L] = s
        tag = f"70/30, book {L:.2f}x"
        print(f"{tag:26}{s['cagr']*100:6.1f}%{s['vol']*100:6.1f}%{s['sharpe']:7.2f}"
              f"{s['sortino']:7.2f}{s['calmar']:7.2f}{s['mdd']*100:7.1f}%"
              f"{s['beta']:6.2f}{s['alpha']*100:6.1f}%"
              f"{15000*s['final']/100000:>12,.0f}")

    print(f"\n{'='*104}")
    print("=== RISK GEOMETRY (book leg, $10,500 of a $15k account) ===")
    print(f"{'='*104}")
    print(f"{'':10}{'exposure':>10}{'loan':>9}{'int/yr':>9}"
          f"{'fall to margin call':>21}{'lev after':>11}{'lev after':>11}")
    print(f"{'':10}{'':10}{'':9}{'':9}{f'(at {MAINT:.0%} maint)':>21}"
          f"{f'{book_dd*100:.0f}% fall':>11}{'-33% fall':>11}")
    B = 10500.0
    for L in LEVELS:
        expo, loan = L * B, (L - 1.0) * B
        # maintenance breached when equity < MAINT * remaining exposure
        # (1-x)*expo - loan < MAINT*(1-x)*expo  ->  x > 1 - loan/((1-MAINT)*expo)
        call_at = 1.0 - loan / ((1 - MAINT) * expo) if loan > 0 else 1.0
        def drift(x):
            e = (1 - x) * expo - loan
            return ((1 - x) * expo / e) if e > 0 else float("inf")
        print(f"{L:.2f}x{'':5}{expo:>10,.0f}{loan:>9,.0f}{loan*RATE:>9,.0f}"
              f"{call_at*100:>20.0f}%{drift(-book_dd):>11.2f}x{drift(0.33):>11.2f}x")

    print(f"\n{'='*104}")
    print("=== WHAT LEVERAGE COSTS AT DIFFERENT RATES (CAGR, 70/30 book leg) ===")
    print(f"{'='*104}")
    print(f"{'':10}" + "".join(f"{f'{rt:.1%}':>11}" for rt in (0.049, 0.059, 0.064, 0.075, 0.10)))
    for L in LEVELS:
        cells = ""
        for rt in (0.049, 0.059, 0.064, 0.075, 0.10):
            s = m(0.70 * lever(r_book, L, rt) + 0.30 * sleeve)
            cells += f"{s['cagr']*100:>10.1f}%"
        print(f"{L:.2f}x{'':5}{cells}")

    print(f"\n{'='*104}")
    print("=== BREAK-EVEN: how bad a book year makes leverage a net loss? ===")
    print(f"{'='*104}")
    print("Levering multiplies book return AND adds a fixed interest cost.")
    print(f"At {RATE:.1%}, the book must return more than {RATE:.1%} for leverage to add "
          f"anything at all.\nWorst 12m book return in sample, and what each level did to it:")
    roll = (1 + bk).rolling(12).apply(np.prod, raw=True) - 1
    worst = float(roll.min()); worst_end = roll.idxmin()
    print(f"\n  worst rolling 12m book return: {worst*100:.1f}% (ending {worst_end})")
    print(f"{'':10}{'levered 12m':>14}{'vs unlevered':>15}")
    for L in LEVELS:
        lev12 = L * worst - (L - 1.0) * RATE
        print(f"{L:.2f}x{'':5}{lev12*100:>13.1f}%{(lev12-worst)*100:>14.1f}pp")


if __name__ == "__main__":
    main()
