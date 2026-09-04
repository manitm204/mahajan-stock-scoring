"""Study 6 — What does the top of each analyst signal's distribution buy you?

For each key signal: group stats on the RAW signal among covered names only
(neutral-50 filler excluded), per rebalance date, 3M forward horizon (6M spread
also reported). Excess = return minus SPY over the same window.

Run:  python -m research.analyst_deep_dive.study6_percentiles
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from research.analyst_deep_dive.common import (
    OUT_DIR, load_panel, load_price_matrix, panel_forward_returns, year_of,
)

SIGNALS_PANEL = [
    "rev_pt_upgrade_ratio_30d", "rev_grade_diffusion",
    "rev_firm_skill_weighted_upside_30d", "rev_target_revision_raw_30d",
    "rev_pt_target_upside_30d",
]
GROUPS = [("top5", 0.95, 1.01), ("top10", 0.90, 1.01), ("top20", 0.80, 1.01),
          ("mid50", 0.25, 0.75), ("bottom20", 0.0, 0.20), ("bottom10", 0.0, 0.10)]
ORDERED = ["bottom10", "bottom20", "mid50", "top20", "top10", "top5"]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    panel = load_panel()
    matrix = load_price_matrix(start="2022-06-01")
    dates = panel.rebal_dates
    fwd = panel_forward_returns(matrix, dates)

    # per-date signal raw values (covered names only)
    signal_raw: dict[str, dict[str, pd.Series]] = {}
    for name in SIGNALS_PANEL:
        signal_raw[name] = {d: panel.raws[d][name].dropna() for d in dates
                            if name in panel.raws[d].columns}
    with open(OUT_DIR / "study5_scores.pkl", "rb") as f:
        s5 = pickle.load(f)
    for name in ["upgrade_x_agreement", "raise_and_bullish_90d"]:
        # study-5 scores are percentile scores incl. 50-fill; drop exact-50 filler
        signal_raw[name] = {d: s.loc[s != 50.0].dropna() for d, s in s5[name].items()}

    rows = []
    turnover_rows = []
    yearly_rows = []
    for name, by_date in signal_raw.items():
        prev_top10: set | None = None
        per_group_series: dict[str, dict[str, float]] = {g[0]: {} for g in GROUPS}
        spread_series: dict[str, float] = {}
        spread6_series: dict[str, float] = {}
        for d, sig in by_date.items():
            if d not in fwd["3M"]:
                continue
            f3 = fwd["3M"][d]
            spy3 = f3.get("SPY", np.nan)
            r = pd.DataFrame({"s": sig, "f": f3}).dropna()
            if len(r) < 50:
                continue
            r["pct"] = r["s"].rank(pct=True, method="average")
            for gname, lo, hi in GROUPS:
                sel = r[(r["pct"] > lo) & (r["pct"] <= hi)]
                if len(sel) < 5:
                    continue
                per_group_series[gname][d] = float(sel["f"].mean() - (spy3 if np.isfinite(spy3) else 0.0))
            top = r[r["pct"] > 0.90]
            bot = r[r["pct"] <= 0.10]
            if len(top) >= 5 and len(bot) >= 5:
                spread_series[d] = float(top["f"].mean() - bot["f"].mean())
            if d in fwd["6M"]:
                f6 = fwd["6M"][d]
                r6 = pd.DataFrame({"s": sig, "f": f6}).dropna()
                if len(r6) >= 50:
                    r6["pct"] = r6["s"].rank(pct=True)
                    t6, b6 = r6[r6["pct"] > 0.90], r6[r6["pct"] <= 0.10]
                    if len(t6) >= 5 and len(b6) >= 5:
                        spread6_series[d] = float(t6["f"].mean() - b6["f"].mean())
            top_set = set(top.index)
            if prev_top10 is not None and len(top_set) > 0:
                turnover_rows.append({"signal": name, "date": d,
                                      "turnover": 1 - len(top_set & prev_top10) / len(top_set)})
            prev_top10 = top_set

        group_means = {g: pd.Series(v) for g, v in per_group_series.items()}
        ordered_means = [group_means[g].mean() if len(group_means[g]) else np.nan for g in ORDERED]
        valid = [v for v in ordered_means if np.isfinite(v)]
        mono = pd.Series(valid).corr(pd.Series(range(len(valid))), method="spearman") if len(valid) >= 4 else np.nan
        sp = pd.Series(spread_series)
        sp6 = pd.Series(spread6_series)
        covs = [len(by_date[d]) / len(panel.scores[d].index) for d in by_date]
        for gname in ORDERED:
            gm = group_means[gname]
            rows.append({
                "signal": name, "group": gname,
                "mean_excess_3m": float(gm.mean()) if len(gm) else np.nan,
                "median_excess_3m": float(gm.median()) if len(gm) else np.nan,
                "hit_rate": float((gm > 0).mean()) if len(gm) else np.nan,
                "n_periods": len(gm),
            })
        rows.append({
            "signal": name, "group": "SPREAD_t10_b10",
            "mean_excess_3m": float(sp.mean()) if len(sp) else np.nan,
            "median_excess_3m": float(sp.median()) if len(sp) else np.nan,
            "hit_rate": float((sp > 0).mean()) if len(sp) else np.nan,
            "n_periods": len(sp),
            "spread_6m": float(sp6.mean()) if len(sp6) else np.nan,
            "monotonicity": float(mono) if np.isfinite(mono) else np.nan,
            "coverage": float(np.mean(covs)),
        })
        t10 = group_means["top10"]
        for yr, v in t10.groupby(t10.index.map(year_of)).mean().items():
            yearly_rows.append({"signal": name, "year": yr, "top10_mean_excess_3m": float(v)})

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "study6_percentile_economics.csv", index=False)
    tdf = pd.DataFrame(turnover_rows)
    tavg = tdf.groupby("signal")["turnover"].mean()
    tavg.to_csv(OUT_DIR / "study6_turnover.csv")
    ydf = pd.DataFrame(yearly_rows)
    ydf.to_csv(OUT_DIR / "study6_top10_by_year.csv", index=False)

    print("=== STUDY 6: percentile economics (3M excess vs SPY) ===")
    piv = df[df["group"] != "SPREAD_t10_b10"].pivot_table(
        index="signal", columns="group", values="mean_excess_3m")[ORDERED]
    print(piv.round(4).to_string())
    print("\n--- spreads / monotonicity / turnover ---")
    sp = df[df["group"] == "SPREAD_t10_b10"].set_index("signal")
    sp["turnover_top10"] = tavg
    print(sp[["mean_excess_3m", "hit_rate", "spread_6m", "monotonicity",
              "coverage", "turnover_top10"]].round(4).to_string())
    print("\n--- top10 excess by year ---")
    print(ydf.pivot_table(index="signal", columns="year",
                          values="top10_mean_excess_3m").round(4).to_string())


if __name__ == "__main__":
    main()
