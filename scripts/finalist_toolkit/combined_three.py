"""Three-strategy combination study (2026-07-18, user request).

Combines, after-tax and at the book's monthly mark dates:
  1. the locked equity book (top25/cap5/cap_match/VIX, 13-mo hold, 4 sleeves),
  2. the 30% XSP covered-call overwrite (2% OTM monthly, 60/40 tax),
  3. the relvalue pairs finalists (s4_half1 "A", s4_legs7030 "B",
     x_legs7030 = B without GOOG/GOOGL as the honest variant).

Mixing rules:
  - capital split w_book / w_pairs, rebalanced monthly at the mark dates;
  - the covered call overwrites 30% of the BOOK portion only
    (overlay contribution = 0.30 * w_book * unit_at);
  - pairs P&L is all short-term (~47d holds) -> taxed at ST 32% via a flat
    (1-ST) scale on the monthly stream (assumes losses offset other ST gains).

Common window = book marks ∩ pairs eval (2017 -> 2025-06; pairs stop there).
Same holdout caveat as STRATEGIES.md: the pairs family failed 2025-07→2026-06.
"""
from __future__ import annotations
import pickle
import pandas as pd

from hz_experiment import load_env, make_data, PROD_CACHE
import final_stats as fs
from covered_call import overlay_unit_returns, stats, BLEND1256, COV
from data.db import get_db
from backtesting.data_loader import SPY

ROOT = "/home/manit/Desktop/fun_projects/mahajan_hedge_fund"
PAIRS = {"A(half1)": "s4_half1", "B(legs7030)": "s4_legs7030",
         "Bx(no GOOG)": "x_legs7030"}
OTM = 0.02


def pairs_monthly(config: str, dates) -> pd.Series:
    """Compound daily invested returns into the book's monthly mark periods."""
    daily = pd.read_csv(f"{ROOT}/output/relvalue/eval/{config}/daily_invested.csv",
                        index_col=0)["ret"]
    daily.index = pd.to_datetime(daily.index)
    dt = pd.to_datetime([str(d) for d in dates])
    out = {}
    for i in range(len(dates) - 1):
        chunk = daily[(daily.index > dt[i]) & (daily.index <= dt[i + 1])]
        if len(chunk):
            out[dates[i + 1]] = float((1 + chunk).prod() - 1)
    return pd.Series(out)


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

    unit, _ = overlay_unit_returns(dates, matrix, vix, OTM)
    unit_at = unit * (1 - BLEND1256)

    pairs_at = {k: pairs_monthly(c, dates) * (1 - fs.ST) for k, c in PAIRS.items()}
    # common window: book marks that have a pairs observation
    idx = r_book.index.intersection(pairs_at["A(half1)"].index)

    def mix(w_pairs: float, pkey: str | None, cc: bool) -> pd.Series:
        wb = 1.0 - w_pairs
        r = wb * r_book.reindex(idx).fillna(0.0)
        if pkey:
            r = r + w_pairs * pairs_at[pkey].reindex(idx).fillna(0.0)
        if cc:
            r = r + COV * wb * unit_at.reindex(idx).fillna(0.0)
        return r

    rows = [
        ("SPY (after-tax)", spy_at.reindex(idx)),
        ("QQQ (after-tax)", qqq_at.reindex(idx)),
        ("Pairs A alone", pairs_at["A(half1)"].reindex(idx)),
        ("Pairs B alone", pairs_at["B(legs7030)"].reindex(idx)),
        ("Book alone", mix(0.0, None, False)),
        ("Book + CC", mix(0.0, None, True)),
        ("Book 85/15 A", mix(0.15, "A(half1)", False)),
        ("Book 85/15 B", mix(0.15, "B(legs7030)", False)),
        ("Book 75/25 A", mix(0.25, "A(half1)", False)),
        ("Book 75/25 B", mix(0.25, "B(legs7030)", False)),
        ("Book+CC 85/15 A", mix(0.15, "A(half1)", True)),
        ("Book+CC 85/15 B", mix(0.15, "B(legs7030)", True)),
        ("Book+CC 75/25 A", mix(0.25, "A(half1)", True)),
        ("Book+CC 75/25 B", mix(0.25, "B(legs7030)", True)),
        ("Book+CC 85/15 Bx", mix(0.15, "Bx(no GOOG)", True)),
        ("Book+CC 75/25 Bx", mix(0.25, "Bx(no GOOG)", True)),
    ]
    for wlabel, lo in [("FULL 2017 -> 2025-06", "2000"),
                       ("2020-start -> 2025-06", "2020-01-01")]:
        print(f"\n=== {wlabel} (after-tax, monthly marks; CC = 30% of book"
              f" @2% OTM; pairs ST-taxed) ===")
        print(f"{'':22}{'CAGR':>7}{'Vol':>7}{'Sharpe':>8}{'Sortino':>9}"
              f"{'Calmar':>8}{'maxDD':>8}{'beta':>7}{'final $':>12}")
        sg = spy_gross[spy_gross.index.astype(str) >= lo]
        for name, r in rows:
            rr = r[r.index.astype(str) >= lo].dropna()
            m = stats(rr, sg)
            vol = rr.std(ddof=1) * (12 ** 0.5)
            print(f"{name:22}{m['cagr']*100:6.1f}%{vol*100:6.1f}%{m['sharpe']:8.2f}"
                  f"{m['sortino']:9.2f}{m['calmar']:8.2f}{m['mdd']*100:7.1f}%"
                  f"{m['beta']:7.2f}{m['final']:>12,.0f}")


if __name__ == "__main__":
    main()
