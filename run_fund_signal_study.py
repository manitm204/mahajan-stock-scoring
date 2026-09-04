"""Weekly regime-signal study across SPY, QQQ and IWM, 2005-2026.

Each fund is scored with its own price signals, its own volatility index
(VIX / VXN / RVX) and its rolling 60-day beta to SPY, plus the shared
market-wide regime signals. Two forward returns are tested:

  absolute  the fund's own next-month total return
  relative  that minus SPY's — the rotation question

    python run_fund_signal_study.py [--rebuild]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from spystudy import stats as st
from spystudy.multi import build_panels
from spystudy.study import IV_SPECS, SPECS

OUT = Path("output/fund_signal_study")

BETA_SPECS = [
    ("beta_100", "beta_60", lambda s: s >= 1.00, "Beta to SPY ≥ 1.00", "< 1.00",
     "Beta", "β ≥ 1.00"),
    ("beta_115", "beta_60", lambda s: s >= 1.15, "Beta to SPY ≥ 1.15", "< 1.15",
     "Beta", "β ≥ 1.15"),
    ("beta_125", "beta_60", lambda s: s >= 1.25, "Beta to SPY ≥ 1.25", "< 1.25",
     "Beta", "β ≥ 1.25"),
]

# SPECS entries that reference columns the multi panel does not carry
DROP = {"corr21_045"}
ALL_SPECS = [s for s in SPECS if s[0] not in DROP] + IV_SPECS + BETA_SPECS

LEVELS = [("rsi14", "RSI(14)"), ("sector_corr_60d", "Sector corr 60d"),
          ("trend_4m", "4-month trend"), ("ma200_dist", "Price vs MA200"),
          ("absorption_shift", "Absorption shift"), ("vix", "Own vol index"),
          ("vrp", "VRP (IV − realised)"), ("ivr", "IV Rank"),
          ("beta_60", "Beta to SPY (60d)")]

REGIMES = [("Pre-GFC 2005–07", "2005-01-01", "2007-06-30"),
           ("GFC 2007–09", "2007-07-01", "2009-12-31"),
           ("Recovery 2010–19", "2010-01-01", "2019-12-31"),
           ("Modern 2020–26", "2020-01-01", "2026-12-31")]

BASES = [("abs", "fwd_abs", "own total return"),
         ("rel", "fwd_rel", "minus SPY")]


def buckets(w: pd.DataFrame, fwd_col: str) -> pd.DataFrame:
    rows = []
    for key, col, pred, on, off, group, short in ALL_SPECS:
        if col not in w.columns:
            continue
        mask = pred(w[col]).where(w[col].notna())
        r = st.compare(mask, w[fwd_col], key, on, off)
        r["group"], r["short"] = group, short.replace("\n", " ")
        rows.append(r)
    return pd.DataFrame(rows)


def levels(w: pd.DataFrame, fwd_col: str) -> pd.DataFrame:
    rows = []
    for col, label in LEVELS:
        r = st.rank_ic(w[col], w[fwd_col])
        rows.append({"signal": col, "label": label, **r})
    return pd.DataFrame(rows)


def regimes(w: pd.DataFrame, fwd_col: str) -> pd.DataFrame:
    rows = []
    for col, label in LEVELS:
        row = {"signal": col, "label": label}
        for name, a, b in REGIMES:
            sl = w.loc[a:b]
            r = st.rank_ic(sl[col], sl[fwd_col])
            row[f"{name} rho"], row[f"{name} p"] = r["rho"], r["p"]
            row[f"{name} n"] = r["n"]
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    panels, _ = build_panels(rebuild=args.rebuild)

    for fund, w in panels.items():
        for tag, col, desc in BASES:
            if fund == "SPY" and tag == "rel":
                continue  # SPY minus SPY is identically zero
            b, l, g = buckets(w, col), levels(w, col), regimes(w, col)
            b.to_csv(OUT / f"{fund}_{tag}_buckets.csv", index=False)
            l.to_csv(OUT / f"{fund}_{tag}_levels.csv", index=False)
            g.to_csv(OUT / f"{fund}_{tag}_regimes.csv", index=False)
            y = w[col]
            print(f"\n=== {fund} · {desc} · n={y.notna().sum()} "
                  f"mean {y.mean() * 100:+.2f}% median {y.median() * 100:+.2f}%")
            sig = l[(l.lo > 0) | (l.hi < 0)]
            print("  levels clearing zero: " + (", ".join(
                f"{r.label} {r.rho:+.3f}" for r in sig.itertuples()) or "none"))
            hit = b[(b.boot_lo > 0) | (b.boot_hi < 0)]
            print("  thresholds clearing zero: " + (", ".join(
                f"{r.on_label} {r.mean_diff * 100:+.2f}pp" for r in hit.itertuples())
                or "none"))
    print(f"\nWrote {OUT}/")


if __name__ == "__main__":
    main()
