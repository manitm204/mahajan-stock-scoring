"""Run the pairs-trading study and write output/pairtrading/report.md + equity chart.

Usage:
    python run_pairtrading_study.py [--pairs 20]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from pairtrading.study import perf_stats, run_grid, trade_stats

OUT = Path("output/pairtrading")


def yearly_table(daily: pd.Series) -> pd.Series:
    idx = pd.to_datetime(daily.index)
    return daily.groupby(idx.year).apply(lambda s: (1 + s).prod() - 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=20)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    print(f"Running pairs grid (top {args.pairs} pairs per window)...")
    variants = run_grid(n_pairs=args.pairs)

    rows, era_rows, yearly = [], [], {}
    for name, v in variants.items():
        v.daily = v.daily.sort_index()
        rows.append({"variant": name, **perf_stats(v.daily), **trade_stats(v.trades)})
        yearly[name] = yearly_table(v.daily)
        for era, d in [("2017-21", v.daily[v.daily.index < "2022"]),
                       ("2022-26", v.daily[v.daily.index >= "2022"])]:
            tr = [t for t in v.trades
                  if (t.open_date < "2022") == (era == "2017-21")]
            era_rows.append({"variant": name, "era": era,
                             **perf_stats(d), **trade_stats(tr)})
    summary = pd.DataFrame(rows).set_index("variant")
    eras = pd.DataFrame(era_rows).set_index(["variant", "era"])
    ytab = pd.DataFrame(yearly)

    # equity chart
    fig, ax = plt.subplots(figsize=(11, 6))
    for name, v in variants.items():
        (1 + v.daily).cumprod().pipe(
            lambda eq: ax.plot(pd.to_datetime(eq.index), eq.values, label=name))
    ax.set_title(f"Pairs trading — cumulative return on committed capital "
                 f"(top {args.pairs} pairs, 10bp/side)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "equity.png", dpi=120)

    with pd.option_context("display.float_format", "{:.3f}".format,
                           "display.width", 200):
        print("\n=== Summary ===")
        print(summary.to_string())
        print("\n=== By era (2017-21 = forensics discovery half) ===")
        print(eras[["ann_ret", "sharpe", "max_dd", "n_trades", "hit_rate"]].to_string())
        print("\n=== Yearly returns ===")
        print(ytab.to_string())

    lines = [
        "# Pairs-Trading Study — entry-discipline grid", "",
        f"Top {args.pairs} same-sector pairs per semiannual window (2017-H1..2026-H1), "
        "12M formation, one-day wait on all signals, 10bp/side, PIT universe "
        "(`members_as_of` at each test start). Classic = 2σ entry, "
        "mean-cross/window-end exit. Strict = quality pairs (≥10 crossings, "
        "half-life 5–40d, σ floor vs costs) + 2.5σ entry with reversal "
        "confirmation, 4σ broken-pair guard (window blacklist), 45-day time "
        "stop. The veto cell additionally gates entries on frozen vixtilt PIT "
        "composites (long leg not bottom-quintile, short leg not top-quintile).", "",
        "Returns are on **committed capital** (each formed pair reserves 1/N of "
        "capital for the whole window whether or not it is open).", "",
        "## Summary", "", summary.to_markdown(floatfmt=".3f"), "",
        "## By era (adjusted rules were derived on 2017-21; 2022-26 is the "
        "honest read)", "",
        eras[["ann_ret", "sharpe", "max_dd", "n_trades", "hit_rate"]]
        .to_markdown(floatfmt=".3f"), "",
        "## Yearly returns", "", ytab.to_markdown(floatfmt=".3f"), "",
        "![equity](equity.png)", "",
    ]
    (OUT / "report.md").write_text("\n".join(lines))
    summary.to_csv(OUT / "summary.csv")
    trades = pd.DataFrame([{
        "variant": name, "pair": f"{t.pair.a}/{t.pair.b}", "sector": t.pair.sector,
        "long": t.long, "short": t.short, "open": t.open_date, "close": t.close_date,
        "days": t.days, "payoff": t.payoff, "reason": t.reason,
        "long_score": t.long_score, "short_score": t.short_score,
    } for name, v in variants.items() for t in v.trades])
    trades.to_csv(OUT / "trades.csv", index=False)
    print(f"\nWrote {OUT}/report.md, summary.csv, trades.csv, equity.png")


if __name__ == "__main__":
    main()
