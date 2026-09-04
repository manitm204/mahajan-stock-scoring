"""LLM pilot — candidate portfolio constructions over the 6M window (EXPLORATORY).

Each rule selects names at formation from the 237 LLM-covered universe;
holds equal-weight for 6 months. Metrics: pooled mean 6M name return, the
averaged two-cohort growth path, and a Sharpe from the 12 stitched monthly
points (2 cohorts x 6 months — treat as descriptive, it is far too short).

These rules were chosen AFTER looking at the leaderboard — this is
curve-fitting on 237 names, not a backtest. Anything promising needs the
fresh-names confirmation.

Usage: python scripts/llm_pilot_portfolios.py
Output: output/llm_pilot/portfolio_compare.png (+ printed table)
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.llm_pilot_charts import monthly_prices
from scripts.llm_pilot_overlay_6m import WINDOWS_6M, fwd6m, growth_path, load

OUT = Path("output/llm_pilot")


def main() -> int:
    df = load()
    mpx = monthly_prices(sorted(df.ticker.unique()))
    df["fwd_6m_ret"] = [fwd6m(mpx, t, c) for t, c in zip(df.ticker, df.cohort)]
    df["comp_q"] = df.groupby("cohort").composite_pct.transform(
        lambda s: pd.qcut(s.rank(method="first"), 5, labels=False) + 1)
    df = df.dropna(subset=["fwd_6m_ret"])
    st, rv, q = df.final_research_status, df.quant_signal_review, df.comp_q

    PORTFOLIOS = {
        "whole universe":              pd.Series(True, index=df.index),
        "Q4+Q5 only":                  q >= 4,
        "Q5 only":                     q == 5,
        "all PASS (any quintile)":     st == "PASS",
        "Q4+Q5 & PASS":                (q >= 4) & (st == "PASS"),
        "Q5 & PASS":                   (q == 5) & (st == "PASS"),
        "Q4+Q5 & CONFIRMS/MIXED":      (q >= 4) & rv.isin(["CONFIRMS", "MIXED"]),
        "Q5 & MIXED/WATCHLIST":        (q == 5) & ((rv == "MIXED") | (st == "WATCHLIST")),
        "universe minus Q2":           q != 2,
        "universe minus CONFIRMS":     rv != "CONFIRMS",
        "Q4+Q5 minus CONFIRMS":        (q >= 4) & (rv != "CONFIRMS"),
    }

    rows, paths = [], {}
    for name, mask in PORTFOLIOS.items():
        sub = df[mask]
        path = growth_path({c: list(g.ticker)
                            for c, g in sub.groupby("cohort")}, mpx)
        paths[name] = path
        monthly = []
        for c, (start, end) in WINDOWS_6M.items():
            names = [t for t in sub[sub.cohort == c].ticker if t in mpx.columns]
            if names:
                monthly.append(mpx.loc[start:end, names]
                               .pct_change().iloc[1:].mean(axis=1))
        m = pd.concat(monthly)
        rows.append({"portfolio": name,
                     "n_2024": int((sub.cohort == "2024").sum()),
                     "n_2025": int((sub.cohort == "2025").sum()),
                     "mean_6m_ret": float(sub.fwd_6m_ret.mean()),
                     "path_end": float(path.iloc[-1]) - 1,
                     "sharpe_12obs": float(m.mean() / m.std(ddof=1) * np.sqrt(12))})
    res = pd.DataFrame(rows)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(19, 8))
    order = res.sort_values("mean_6m_ret")
    colors = ["purple" if p == "whole universe" else
              "seagreen" if v >= res.loc[res.portfolio == "whole universe",
                                         "mean_6m_ret"].iloc[0] else "crimson"
              for p, v in zip(order.portfolio, order.mean_6m_ret)]
    ax1.barh([f"{p}  (n={a}+{b})" for p, a, b in
              zip(order.portfolio, order.n_2024, order.n_2025)],
             order.mean_6m_ret, color=colors)
    ax1.axvline(res.loc[res.portfolio == "whole universe",
                        "mean_6m_ret"].iloc[0], color="purple", ls="--",
                lw=1.2, label="whole universe")
    ax1.axvline(0, color="gray", lw=0.8)
    ax1.set_title("mean 6M fwd return by construction "
                  "(green = beats universe; n = 2024+2025 names)")
    ax1.set_xlabel("mean 6M fwd return")
    ax1.legend(fontsize=9)

    show = ["whole universe", "Q4+Q5 only", "Q5 only", "all PASS (any quintile)",
            "Q5 & MIXED/WATCHLIST", "Q4+Q5 & CONFIRMS/MIXED", "Q5 & PASS"]
    palette = ["purple", "gray", "tab:blue", "seagreen", "crimson",
               "tab:orange", "olive"]
    for name, color in zip(show, palette):
        p = paths[name]
        ax2.plot(p.index, p.values, color=color, lw=1.8,
                 label=f"{name}  ({p.iloc[-1] - 1:+.1%})")
    ax2.axhline(1.0, color="gray", lw=0.6)
    ax2.set_title("growth of $1 over the 6M window (avg of cohort paths)")
    ax2.set_xlabel("months since formation")
    ax2.legend(fontsize=9)

    fig.suptitle("Candidate constructions on the LLM-covered universe — "
                 "EXPLORATORY (rules chosen after seeing the data)",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "portfolio_compare.png", dpi=120)

    print(res.sort_values("mean_6m_ret", ascending=False)
          .round(3).to_string(index=False))
    print(f"\nwrote {OUT}/portfolio_compare.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
