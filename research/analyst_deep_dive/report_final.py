"""Final comparison — uniform metrics for all 21 existing revisions candidates
plus the 6 new study-5 candidates and the firm-bias redesign, a pooled
research-side selector run (existing rules, R²<0.60, max 3 subs, weights ∝ IC
cap 50%), incremental-parent ICs, and semester walk-forward stability.

Purely research output — the production selector inputs and parent are not
touched.

Run:  python -m research.analyst_deep_dive.report_final
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from research.parent_selection import (rank_subfactors, select_subfactors,
                                       parent_weights)
from factors.parent_selection_v4 import SELECTED_SUBS
from research.analyst_deep_dive.common import (
    OUT_DIR, load_panel, load_price_matrix, panel_forward_returns,
    ic_series, summarize_ic, bootstrap_mean_ci, semester_of,
)

LIVE = ["rev_pt_upgrade_ratio_30d", "rev_grade_diffusion", "rev_pt_momentum"]
NEW = ["upgrade_ratio_decay_90d", "pct_bullish_dedup_90d", "agreement_90d",
       "raise_and_bullish_90d", "diffusion_x_upside", "upgrade_x_agreement",
       "firm_bias_adj_upside_30d"]


def main():
    panel = load_panel()
    matrix = load_price_matrix(start="2021-06-01")
    dates = panel.rebal_dates
    fwd = panel_forward_returns(matrix, dates)

    # -------- score dictionaries for the full pool --------
    pool_scores: dict[str, dict[str, pd.Series]] = {}
    existing = panel.candidates_by_parent["revisions"]
    for name in existing:
        pool_scores[name] = {d: panel.scores[d][name] for d in dates
                             if name in panel.scores[d].columns}
    with open(OUT_DIR / "study5_scores.pkl", "rb") as f:
        s5 = pickle.load(f)
    for name, sc in s5.items():
        pool_scores[name] = sc
    with open(OUT_DIR / "bias_candidate_scores.pkl", "rb") as f:
        bias = pickle.load(f)
    pool_scores["firm_bias_adj_upside_30d"] = bias["scores"]

    # raw coverage (existing: raw notna; new: score != 50 among universe)
    def coverage_of(name):
        if name in existing:
            vals = [panel.raws[d][name].notna().mean() for d in dates
                    if name in panel.raws[d].columns]
        else:
            vals = []
            for d, s in pool_scores[name].items():
                idx = panel.scores[d].index
                s = s.reindex(idx)
                vals.append(float((s.notna() & (s != 50.0)).mean()))
        return float(np.mean(vals)) if vals else np.nan

    # -------- uniform metrics --------
    def quintile_spread(sc_dict, h="3M"):
        sps = []
        for d, s in sc_dict.items():
            if d not in fwd[h]:
                continue
            df = pd.DataFrame({"s": s, "f": fwd[h][d]}).dropna()
            if len(df) < 25:
                continue
            ranks = df["s"].rank(method="first")
            buckets = pd.qcut(ranks, 5, labels=False)
            means = df["f"].groupby(buckets).mean()
            if len(means) == 5:
                sps.append(float(means.iloc[-1] - means.iloc[0]))
        return float(np.mean(sps)) if sps else np.nan

    mom_w = SELECTED_SUBS["momentum"]

    def mom_composite(d):
        sc = panel.scores[d]
        cols = [c for c in mom_w if c in sc.columns]
        wsum = sum(mom_w[c] for c in cols)
        return sum(sc[c] * mom_w[c] for c in cols) / wsum

    mom_comp = {d: mom_composite(d) for d in dates}

    rows = []
    ic_series_store: dict[str, dict[str, pd.Series]] = {}
    for name, sc in pool_scores.items():
        ics = {h: ic_series(sc, fwd[h]) for h in ["1M", "3M", "6M", "12M"]}
        ic_series_store[name] = ics
        blend = float(np.mean([ics["3M"].mean(), ics["6M"].mean()])) \
            if len(ics["3M"]) and len(ics["6M"]) else np.nan
        s3 = summarize_ic(ics["3M"])
        semesters = ics["3M"].groupby(ics["3M"].index.map(semester_of)).mean()
        wf_pos = float((semesters > 0).mean()) if len(semesters) else np.nan
        mcorr = []
        for d, s in sc.items():
            pair = pd.DataFrame({"a": s, "m": mom_comp[d]}).dropna()
            if len(pair) >= 20:
                mcorr.append(pair["a"].corr(pair["m"], method="spearman"))
        _, lo, hi = bootstrap_mean_ci(ics["3M"]) if len(ics["3M"]) else (np.nan, np.nan, np.nan)
        rows.append({
            "sub_factor": name,
            "is_live": name in LIVE, "is_new": name in NEW,
            "coverage": coverage_of(name),
            "ic_1M": float(ics["1M"].mean()) if len(ics["1M"]) else np.nan,
            "ic_3M": float(ics["3M"].mean()) if len(ics["3M"]) else np.nan,
            "ic_6M": float(ics["6M"].mean()) if len(ics["6M"]) else np.nan,
            "ic_12M": float(ics["12M"].mean()) if len(ics["12M"]) else np.nan,
            "mean_ic": blend,
            "ic_ir": s3["ic_ir"],
            "hit_rate": s3["hit_rate"],
            "ic3m_ci90": (round(lo, 4), round(hi, 4)),
            "spread_q5_q1": quintile_spread(sc),
            "corr_mom_parent": float(np.mean(mcorr)) if mcorr else np.nan,
            "wf_semesters_pos": wf_pos,
        })
    metrics = pd.DataFrame(rows)

    # -------- pooled correlation matrix (mean per-date Spearman of scores) ----
    names = list(pool_scores)
    corr_acc = {a: {b: [] for b in names} for a in names}
    for d in dates:
        frame = {}
        for name in names:
            if d in pool_scores[name]:
                frame[name] = pool_scores[name][d]
        if len(frame) < 2:
            continue
        df = pd.DataFrame(frame)
        c = df.corr(method="spearman", min_periods=30)
        for a in c.index:
            for b in c.columns:
                v = c.loc[a, b]
                if pd.notna(v):
                    corr_acc[a][b].append(float(v))
    corr = pd.DataFrame({a: {b: (np.mean(v) if v else np.nan)
                             for b, v in corr_acc[a].items()} for a in names}).T
    corr.to_csv(OUT_DIR / "final_pool_correlations.csv")

    # -------- research-side selector run on the expanded pool --------
    sub_df = metrics.rename(columns={})[
        ["sub_factor", "mean_ic", "ic_ir", "spread_q5_q1", "hit_rate", "coverage",
         "ic_1M", "ic_3M", "ic_6M", "ic_12M"]].copy()
    ranked = rank_subfactors(sub_df)
    selected, decisions, stop = select_subfactors(ranked, corr)
    ic_map = dict(zip(metrics["sub_factor"], metrics["mean_ic"]))
    weights, capped = parent_weights(selected, ic_map)

    # -------- incremental parent IC for headline candidates --------
    def composite_ic(names_and_weights: dict[str, float]):
        comp = {}
        for d in dates:
            parts, wsum = [], 0.0
            for n, w in names_and_weights.items():
                if d in pool_scores[n]:
                    parts.append(pool_scores[n][d] * w)
                    wsum += w
            if parts and wsum > 0:
                comp[d] = sum(parts) / wsum
        return {h: ic_series(comp, fwd[h]) for h in ["3M", "6M"]}

    live_w, _ = parent_weights(LIVE, ic_map)
    base = composite_ic(live_w)
    incr_rows = []
    for cand in ["rev_firm_skill_weighted_upside_30d", "rev_target_revision_raw_30d",
                 "upgrade_x_agreement", "raise_and_bullish_90d", "agreement_90d"]:
        w4, _ = parent_weights(LIVE + [cand], ic_map)
        aug = composite_ic(w4)
        incr_rows.append({
            "candidate": cand,
            "delta_ic_3M": float(aug["3M"].mean() - base["3M"].mean()),
            "delta_ic_6M": float(aug["6M"].mean() - base["6M"].mean()),
            "aug_ic_3M": float(aug["3M"].mean()), "base_ic_3M": float(base["3M"].mean()),
        })
    incr = pd.DataFrame(incr_rows)
    incr.to_csv(OUT_DIR / "final_incremental_parent_ic.csv", index=False)

    metrics = metrics.merge(incr.rename(columns={"candidate": "sub_factor"})[
        ["sub_factor", "delta_ic_3M", "delta_ic_6M"]], on="sub_factor", how="left")
    metrics = metrics.sort_values("mean_ic", ascending=False)
    metrics.to_csv(OUT_DIR / "final_decision_table.csv", index=False)

    out = {
        "selector_selected": selected,
        "selector_weights": {k: round(v, 3) for k, v in weights.items()},
        "selector_stop": stop,
        "selector_decisions": decisions[:12],
        "live_recomputed_weights": {k: round(v, 3) for k, v in live_w.items()},
        "base_parent_ic_3M": float(base["3M"].mean()),
        "base_parent_ic_6M": float(base["6M"].mean()),
    }
    with open(OUT_DIR / "final_selector_run.json", "w") as f:
        json.dump(out, f, indent=2, default=str)

    print("=== FINAL decision table (sorted by blended IC) ===")
    show = metrics[["sub_factor", "is_live", "is_new", "coverage", "ic_3M", "ic_6M",
                    "mean_ic", "ic_ir", "hit_rate", "spread_q5_q1",
                    "corr_mom_parent", "wf_semesters_pos", "delta_ic_3M"]]
    print(show.round(4).to_string(index=False))
    print("\n=== research-side selector on expanded pool ===")
    print(json.dumps({k: out[k] for k in ["selector_selected", "selector_weights",
                                          "selector_stop"]}, indent=2))
    print("\n=== incremental parent IC ===")
    print(incr.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
