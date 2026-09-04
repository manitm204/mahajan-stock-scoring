"""Study 4 — Is raw target revision an analyst signal or disguised momentum?

Measures the overlap of rev_target_revision_raw_30d with price momentum at
1/3/6/12M, the live momentum-parent composite, and the full V4 composite; then
tests a residualized version. Residualization is PIT-safe two ways:
  (a) per-date cross-sectional OLS (uses only same-date signal values — no
      returns, no future data), the primary spec;
  (b) expanding-window pooled beta (beta estimated from strictly earlier
      rebalance dates), the conservative variant.

Run:  python -m research.analyst_deep_dive.study4_momentum
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from factors.utils import sector_percentile
from factors.parent_selection_v4 import SELECTED_SUBS, V4_PARENT_WEIGHTS
from research.analyst_deep_dive.common import (
    OUT_DIR, load_panel, load_price_matrix, panel_forward_returns, sector_map,
    ic_series, summarize_ic, bootstrap_mean_ci, nearest_idx,
)

REV = "rev_target_revision_raw_30d"
MOM_HORIZONS = {"mom_1m": 1, "mom_3m": 3, "mom_6m": 6, "mom_12m": 12}


def price_momentum(matrix: pd.DataFrame, dates: list[str]) -> dict[str, dict[str, pd.Series]]:
    idx = pd.DatetimeIndex(pd.to_datetime(matrix.index))
    out: dict[str, dict[str, pd.Series]] = {k: {} for k in MOM_HORIZONS}
    for d in dates:
        if d not in matrix.index:
            continue
        p_now = matrix.loc[d]
        for name, months in MOM_HORIZONS.items():
            past = nearest_idx(idx, pd.Timestamp(d) - pd.DateOffset(months=months), 15)
            if past is None:
                continue
            p_past = matrix.loc[past.strftime("%Y-%m-%d")] if past.strftime("%Y-%m-%d") in matrix.index else None
            if p_past is None:
                lbl = matrix.index[idx.get_loc(past)]
                p_past = matrix.loc[lbl]
            out[name][d] = p_now / p_past - 1.0
    return out


def composite(scores: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    cols = [c for c in weights if c in scores.columns]
    if not cols:
        return pd.Series(dtype=float)
    wsum = sum(weights[c] for c in cols)
    return sum(scores[c] * weights[c] for c in cols) / wsum


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    panel = load_panel()
    matrix = load_price_matrix(start="2021-06-01")
    dates = panel.rebal_dates
    fwd = panel_forward_returns(matrix, dates)
    sectors = sector_map(normalized=False)
    mom = price_momentum(matrix, dates)

    mom_parent_w = SELECTED_SUBS["momentum"]
    v4_weights: dict[str, float] = {}
    for parent, subs in SELECTED_SUBS.items():
        pw = V4_PARENT_WEIGHTS.get(parent, 0.0)
        for sub, w in subs.items():
            v4_weights[sub] = pw * w

    # ------------------------------------------------------------------
    # 1. Correlation structure per date
    # ------------------------------------------------------------------
    corr_rows = []
    for d in dates:
        raws = panel.raws[d]
        sc = panel.scores[d]
        if REV not in raws.columns:
            continue
        rev = raws[REV]
        row = {"date": d, "n": int(rev.notna().sum())}
        for name in MOM_HORIZONS:
            if d in mom[name]:
                pair = pd.DataFrame({"r": rev, "m": mom[name][d]}).dropna()
                row[name] = float(pair["r"].corr(pair["m"], method="spearman")) if len(pair) >= 20 else np.nan
        mp = composite(sc, mom_parent_w)
        pair = pd.DataFrame({"r": rev, "m": mp}).dropna()
        row["mom_parent"] = float(pair["r"].corr(pair["m"], method="spearman")) if len(pair) >= 20 else np.nan
        v4 = composite(sc, v4_weights)
        pair = pd.DataFrame({"r": rev, "m": v4}).dropna()
        row["v4_composite"] = float(pair["r"].corr(pair["m"], method="spearman")) if len(pair) >= 20 else np.nan
        corr_rows.append(row)
    cdf = pd.DataFrame(corr_rows)
    cdf.to_csv(OUT_DIR / "study4_correlations_by_date.csv", index=False)
    corr_summary = cdf.drop(columns=["date", "n"]).mean().to_dict()

    # ------------------------------------------------------------------
    # 2. Residualized signals
    # ------------------------------------------------------------------
    # (a) per-date cross-sectional residual on 3M momentum
    resid_cs: dict[str, pd.Series] = {}
    betas_cs = {}
    for d in dates:
        raws = panel.raws[d]
        if REV not in raws.columns or d not in mom["mom_3m"]:
            continue
        df = pd.DataFrame({"r": raws[REV], "m": mom["mom_3m"][d]}).dropna()
        if len(df) < 30:
            continue
        beta = np.polyfit(df["m"], df["r"], 1)[0]
        alpha = df["r"].mean() - beta * df["m"].mean()
        betas_cs[d] = float(beta)
        resid = raws[REV] - (alpha + beta * mom["mom_3m"][d].reindex(raws.index))
        resid_cs[d] = resid

    # (b) expanding pooled beta from earlier dates only
    resid_exp: dict[str, pd.Series] = {}
    hist_pairs = []
    for d in dates:
        raws = panel.raws[d]
        if REV not in raws.columns or d not in mom["mom_3m"]:
            continue
        df = pd.DataFrame({"r": raws[REV], "m": mom["mom_3m"][d]}).dropna()
        if len(hist_pairs) >= 6:
            hist = pd.concat(hist_pairs, ignore_index=True)
            beta = np.polyfit(hist["m"], hist["r"], 1)[0]
            alpha = hist["r"].mean() - beta * hist["m"].mean()
            resid_exp[d] = raws[REV] - (alpha + beta * mom["mom_3m"][d].reindex(raws.index))
        hist_pairs.append(df)

    def score_dict(raw_by_date):
        out = {}
        for d, sig in raw_by_date.items():
            idx = panel.scores[d].index
            out[d] = sector_percentile(sig.reindex(idx), sectors.reindex(idx), True)
        return out

    ics = {}
    raw_scores = {d: panel.scores[d][REV] for d in dates if REV in panel.scores[d].columns}
    for label, sc_dict in [("raw_revision", raw_scores),
                           ("resid_cs", score_dict(resid_cs)),
                           ("resid_expanding", score_dict(resid_exp))]:
        for h in ["3M", "6M"]:
            s = ic_series(sc_dict, fwd[h])
            ics[(label, h)] = s

    # (c) revision combined with the momentum parent (IC-weighted 2-block)
    mom_parent_scores = {d: composite(panel.scores[d], mom_parent_w) for d in dates}
    mom_ic_3m = ic_series(mom_parent_scores, fwd["3M"]).mean()
    mom_ic_6m = ic_series(mom_parent_scores, fwd["6M"]).mean()
    rev_ic_blend = np.mean([ics[("raw_revision", "3M")].mean(), ics[("raw_revision", "6M")].mean()])
    mom_ic_blend = np.mean([mom_ic_3m, mom_ic_6m])
    w_rev = max(rev_ic_blend, 0) / (max(rev_ic_blend, 0) + max(mom_ic_blend, 0))
    w_rev = min(w_rev, 0.5)
    combo_scores = {d: (1 - w_rev) * mom_parent_scores[d] + w_rev * raw_scores[d]
                    for d in raw_scores if d in mom_parent_scores}
    for h in ["3M", "6M"]:
        ics[("mom_parent", h)] = ic_series(mom_parent_scores, fwd[h])
        ics[("mom_plus_revision", h)] = ic_series(combo_scores, fwd[h])

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    summary = {"mean_spearman_vs": {k: round(float(v), 4) for k, v in corr_summary.items()},
               "mean_cs_beta_on_mom3m": float(np.mean(list(betas_cs.values()))),
               "combo_weight_revision": float(w_rev)}
    for (label, h), s in ics.items():
        summary[f"{label}_{h}"] = summarize_ic(s)
    summary["resid_cs_3M_ci"] = bootstrap_mean_ci(ics[("resid_cs", "3M")])
    summary["raw_revision_3M_ci"] = bootstrap_mean_ci(ics[("raw_revision", "3M")])
    summary["incremental_mom_plus_rev_3M"] = float(
        ics[("mom_plus_revision", "3M")].mean() - ics[("mom_parent", "3M")].mean())
    summary["incremental_mom_plus_rev_6M"] = float(
        ics[("mom_plus_revision", "6M")].mean() - ics[("mom_parent", "6M")].mean())

    with open(OUT_DIR / "study4_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print("=== STUDY 4: momentum separation ===")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
