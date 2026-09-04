"""Weekly SPY regime-signal study.

Every Friday, read a set of regime signals off SPY and the 11 sector SPDRs and
ask what the next month of SPY total return looked like. Signals are computed
on price-only closes (the chart a trader sees); forward returns are total
return (what you actually earn).

    python run_spy_signal_study.py [--rebuild]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from spystudy import plots, signals as sg, stats as st
from spystudy.data import SECTOR_NAMES
from spystudy.study import build_panel, masks, run

OUT = Path("output/spy_signal_study")
OUT_DEEP = Path("output/spy_signal_study_2005")

QUINTILE_COLS = [
    ("rsi14", "RSI(14)"),
    ("sector_corr_21d", "Sector correlation, 21d"),
    ("sector_corr_60d", "Sector correlation, 60d"),
    ("trend_4m", "4-month trend"),
    ("ma200_dist", "Price vs MA200"),
    ("absorption_shift", "Absorption standardized shift"),
]

ERA_SPLIT = "2020-07-01"

# VIX-derived columns only exist on the deep (2004→) panel
IV_COLS = [("vix", "VIX level"), ("vrp", "VRP (IV − realised)"), ("ivr", "IV Rank")]

# named regimes for the 2005 sample; the short sample uses ERA_SPLIT instead
REGIMES = [("Pre-GFC 2005–07", "2005-01-01", "2007-06-30"),
           ("GFC 2007–09", "2007-07-01", "2009-12-31"),
           ("Recovery 2010–19", "2010-01-01", "2019-12-31"),
           ("Modern 2020–26", "2020-01-01", "2026-12-31")]


def regime_table(weekly: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Rank IC of each signal inside each named regime."""
    rows = []
    for col, label in cols:
        row = {"signal": col, "label": label}
        for name, a, b in REGIMES:
            sl = weekly.loc[a:b]
            r = st.rank_ic(sl[col], sl["fwd_1m"])
            row[f"{name} rho"] = r["rho"]
            row[f"{name} p"] = r["p"]
            row[f"{name} n"] = r["n"]
        rows.append(row)
    return pd.DataFrame(rows)


def continuous_tests(weekly: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rank IC of each signal level, full sample and split by era.

    The threshold buckets throw away the shape of the relation; this keeps it.
    """
    fwd = weekly["fwd_1m"]
    early, late = weekly.loc[:ERA_SPLIT], weekly.loc[ERA_SPLIT:]
    full, eras = [], []
    for col, label in QUINTILE_COLS:
        r = st.rank_ic(weekly[col], fwd)
        full.append({"signal": col, "label": label, **r})
        a = st.rank_ic(early[col], early["fwd_1m"])
        b = st.rank_ic(late[col], late["fwd_1m"])
        eras.append({"signal": col, "label": label, "n_early": a["n"],
                     "rho_early": a["rho"], "p_early": a["p"],
                     "n_late": b["n"], "rho_late": b["rho"], "p_late": b["p"]})
    return pd.DataFrame(full), pd.DataFrame(eras)


def sector_effect_table(close: pd.DataFrame, weekly: pd.DataFrame,
                        window: int = 60) -> pd.DataFrame:
    """Does any single sector's decoupling carry the predictive content?"""
    by_sector = sg.sector_correlation_by_sector(close, window, weekly.index)
    fwd = weekly["fwd_1m"]
    rows = []
    for s in by_sector.columns:
        x = by_sector[s]
        m = (x < x.quantile(1 / 3)).where(x.notna())
        r = st.compare(m, fwd, s, "bottom tercile", "rest")
        rows.append({"sector": s, "n_on": r["on_n"], "on_mean": r["on_mean"],
                     "off_mean": r["off_mean"], "diff": r["mean_diff"],
                     "nw_t": r["nw_t"], "boot_lo": r["boot_lo"],
                     "boot_hi": r["boot_hi"], "boot_p": r["boot_p"]})
    return pd.DataFrame(rows), by_sector


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true",
                    help="recompute signals instead of using the cache")
    ap.add_argument("--deep", action="store_true",
                    help="use the 2005→ FMP/FRED history and add the VIX signals")
    args = ap.parse_args()

    out = OUT_DEEP if args.deep else OUT
    globals()["OUT"] = out
    out.mkdir(parents=True, exist_ok=True)
    weekly, close = build_panel(rebuild=args.rebuild, deep=args.deep)
    span = f"{weekly.index.min():%Y}–{weekly.index.max():%Y}"
    qcols = QUINTILE_COLS + ([c for c in IV_COLS] if "vrp" in weekly.columns else [])
    globals()["QUINTILE_COLS"] = qcols
    res = run(weekly)
    m = masks(weekly)

    base = weekly["fwd_1m"]
    print(f"Weekly observations with a full forward month: {base.notna().sum()}")
    print(f"Unconditional next-month total return: mean {base.mean() * 100:+.2f}%  "
          f"median {base.median() * 100:+.2f}%  hit {(base > 0).mean() * 100:.1f}%")

    res.to_csv(out / "bucket_stats.csv", index=False)
    weekly.to_csv(out / "weekly_panel.csv")

    sec_tab, by_sector = sector_effect_table(close, weekly)
    sec_tab.to_csv(out / "sector_effect.csv", index=False)
    loo = sg.sector_correlation_leave_one_out(close, 60, weekly.index)
    low = weekly["sector_corr_60d"] < 0.45

    plots.effect_summary(res, out / "01_effect_summary.png",
                         "Does the signal change next month? Mean "
                         f"difference in SPY total return, {span}")
    plots.mean_median_bars(res, out / "02_mean_median.png",
                           "Next-month SPY total return, signal ON vs OFF")
    plots.box_grid(weekly, m, res, out / "03_boxplots.png",
                   "Distribution of next-month SPY total return, signal ON vs OFF")
    plots.hist_grid(weekly, m, res, out / "04_histograms.png",
                    "Overlapping distributions of next-month SPY total return")
    plots.quintile_grid(weekly, QUINTILE_COLS, out / "05_quintiles.png",
                        "Is the relation monotone, or does the threshold sit on "
                        "noise? Next-month return by signal quintile")
    plots.sector_attribution(by_sector, loo, low, SECTOR_NAMES,
                             out / "06_sector_attribution.png",
                             "Which sectors drive low market-wide correlation? "
                             "(60-day window)")
    plots.sector_effect(sec_tab, SECTOR_NAMES, out / "07_sector_effect.png",
                        "When one sector decouples, what does SPY do next month?")

    full, eras = continuous_tests(weekly)
    full.to_csv(out / "rank_ic.csv", index=False)
    eras.to_csv(out / "rank_ic_by_era.csv", index=False)
    plots.rank_ic_chart(full, eras, out / "08_rank_ic.png",
                        f"Treating each signal as a level rather than a threshold, {span}")

    if args.deep:
        regime_table(weekly, qcols).to_csv(out / "regime_ic.csv", index=False)

    corr = weekly[[c for c, _ in qcols]].corr(method="spearman")
    corr.columns = corr.index = [lbl for _, lbl in qcols]
    corr.to_csv(out / "signal_redundancy.csv")
    plots.redundancy(corr, out / "09_redundancy.png",
                     f"How much do these signals overlap? {span}")

    cols = ["signal", "on_label", "on_n", "on_mean", "on_median", "on_hit_rate",
            "off_n", "off_mean", "off_median", "off_hit_rate", "mean_diff",
            "nw_t", "boot_lo", "boot_hi", "boot_p"]
    print()
    print(res[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nWrote {out}/")


if __name__ == "__main__":
    main()
