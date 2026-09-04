"""Risk-adjusted sweep of Book+CC+pairs weights (user request, 2026-07-18).

Rows: SPY, QQQ, Book, Book+CC, and Book+CC + pairs-B at 5..45% (book >= 55%).
Two views per window:
  1. unlevered stats (Sharpe/Sortino/Calmar/maxDD/beta),
  2. equal-risk: every portfolio levered (financed at FFR + 1%) to SPY's
     realized vol in that window, so CAGR compares at the same risk.
Same after-tax/monthly-mark conventions as combined_three.py.
"""
from __future__ import annotations
import pickle
import pandas as pd

from hz_experiment import load_env, make_data, PROD_CACHE
import final_stats as fs
from covered_call import overlay_unit_returns, stats, lever_ret, BLEND1256, COV
from combined_three import pairs_monthly, PAIRS
from data.db import get_db
from backtesting.data_loader import SPY

OTM = 0.02


def main():
    panel, matrix_raw, matrix, sectors, vix = load_env()
    run = pickle.load(PROD_CACHE.open("rb"))
    with get_db() as db:
        data = make_data(run, panel, matrix, sectors, vix, db)
    wts = fs.weights_for(data, fs.CFG)
    dates = [d for d in data.rebal_dates if d in wts and d in matrix.index]

    at_nav = pd.Series(fs.config_nav(dates, wts, matrix), index=dates)
    r_book = (at_nav / at_nav.shift(1) - 1.0).dropna()
    spy_gross = matrix[SPY].reindex(dates).astype(float).pct_change().dropna()
    spy_at = pd.Series(fs.bench_nav(matrix, dates, SPY), index=dates).pct_change().dropna()
    qqq_at = pd.Series(fs.bench_nav(matrix, dates, "QQQ"), index=dates).pct_change().dropna()

    unit_at = overlay_unit_returns(dates, matrix, vix, OTM)[0] * (1 - BLEND1256)
    pairs_at = {k: pairs_monthly(c, dates) * (1 - fs.ST) for k, c in PAIRS.items()}
    idx = r_book.index.intersection(pairs_at["A(half1)"].index)

    def mix(w_pairs: float, pkey: str | None, cc: bool) -> pd.Series:
        wb = 1.0 - w_pairs
        r = wb * r_book.reindex(idx).fillna(0.0)
        if pkey:
            r = r + w_pairs * pairs_at[pkey].reindex(idx).fillna(0.0)
        if cc:
            r = r + COV * wb * unit_at.reindex(idx).fillna(0.0)
        return r

    rows = [("SPY (after-tax)", spy_at.reindex(idx).dropna()),
            ("QQQ (after-tax)", qqq_at.reindex(idx).dropna()),
            ("Book alone", mix(0.0, None, False)),
            ("Book + CC", mix(0.0, None, True))]
    for wp in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45):
        rows.append((f"Book+CC + {wp*100:.0f}% B", mix(wp, "B(legs7030)", True)))
    rows.append(("Book+CC + 25% Bx", mix(0.25, "Bx(no GOOG)", True)))
    rows.append(("Book+CC + 25% A", mix(0.25, "A(half1)", True)))

    for wlabel, lo in [("FULL 2017 -> 2025-06", "2000"),
                       ("2020-start -> 2025-06", "2020-01-01")]:
        sg = spy_gross[spy_gross.index.astype(str) >= lo]
        spy_w = spy_at.reindex(idx)[lambda s: s.index.astype(str) >= lo].dropna()
        target_vol = spy_w.std(ddof=1) * (12 ** 0.5)

        print(f"\n=== {wlabel} — unlevered (after-tax) ===")
        print(f"{'':22}{'CAGR':>7}{'Vol':>7}{'Sharpe':>8}{'Sortino':>9}"
              f"{'Calmar':>8}{'maxDD':>8}{'beta':>7}")
        cut = []
        for name, r in rows:
            rr = r[r.index.astype(str) >= lo].dropna()
            m = stats(rr, sg)
            vol = rr.std(ddof=1) * (12 ** 0.5)
            cut.append((name, rr, vol))
            print(f"{name:22}{m['cagr']*100:6.1f}%{vol*100:6.1f}%{m['sharpe']:8.2f}"
                  f"{m['sortino']:9.2f}{m['calmar']:8.2f}{m['mdd']*100:7.1f}%"
                  f"{m['beta']:7.2f}")

        print(f"\n--- {wlabel} — EQUAL RISK: levered to SPY vol "
              f"({target_vol*100:.1f}%), financing FFR+1% ---")
        print(f"{'':22}{'lever':>7}{'CAGR':>7}{'Sharpe':>8}{'maxDD':>8}{'beta':>7}")
        for name, rr, vol in cut:
            L = target_vol / vol if vol > 0 else 1.0
            rl = lever_ret(rr, L)
            m = stats(rl, sg)
            print(f"{name:22}{L:6.2f}x{m['cagr']*100:6.1f}%{m['sharpe']:8.2f}"
                  f"{m['mdd']*100:7.1f}%{m['beta']:7.2f}")


if __name__ == "__main__":
    main()
