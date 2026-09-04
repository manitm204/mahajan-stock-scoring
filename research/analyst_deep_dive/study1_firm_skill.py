"""Study 1 — Does firm-skill weighting add information beyond raw target upside?

Questions (per research spec):
  * Pearson / Spearman / R^2 between raw and skill-weighted signals
  * cross-sectional rank overlap + top-decile overlap
  * incremental IC and partial IC controlling for raw upside
  * parent-composite improvement/deterioration when added under existing
    selector weight rules
  * performance by calendar year and semester walk-forward periods
  * optional redesign: firm-optimism-bias-corrected upside (PIT expanding
    baseline with shrinkage) — redesign #3 of the approved list.

Run:  python -m research.analyst_deep_dive.study1_firm_skill
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from factors.utils import sector_percentile
from research.parent_selection import parent_weights
from research.analyst_deep_dive.common import (
    OUT_DIR, load_panel, load_price_matrix, load_events, panel_forward_returns,
    sector_map, spearman_ic, ic_series, summarize_ic, bootstrap_mean_ci,
    year_of, semester_of,
)

SKILL = "rev_firm_skill_weighted_upside_30d"
RAW = "rev_pt_target_upside_30d"
LIVE_FORMULA = {"rev_pt_upgrade_ratio_30d": 0.500,
                "rev_grade_diffusion": 0.283,
                "rev_pt_momentum": 0.217}


def partial_spearman(x: pd.Series, z: pd.Series, y: pd.Series):
    """Partial Spearman corr of x with y controlling for z (rank-transform)."""
    df = pd.DataFrame({"x": x, "z": z, "y": y}).dropna()
    if len(df) < 20:
        return None
    r = df.rank()
    rxy = r["x"].corr(r["y"])
    rxz = r["x"].corr(r["z"])
    rzy = r["z"].corr(r["y"])
    den = np.sqrt((1 - rxz ** 2) * (1 - rzy ** 2))
    if den == 0 or np.isnan(den):
        return None
    return float((rxy - rxz * rzy) / den)


def build_firm_bias_candidate(panel, events: pd.DataFrame, sectors: pd.Series,
                              window_days: int = 30, shrink_k: int = 25,
                              half_life_days: int = 730) -> dict[str, pd.Series]:
    """Firm-optimism-bias-corrected upside, strictly PIT.

    For each rebalance date d: signal(ticker) = mean over events in
    (d-window, d] of (upside_i - firm_bias(firm_i, d)), where firm_bias is the
    recency-weighted mean upside of that firm's events strictly BEFORE d,
    shrunk toward the global (all-firm) recency-weighted mean by effective
    sample size n_eff/(n_eff+k). A firm that always projects +40% no longer
    dominates: only upside unusual FOR THAT FIRM counts.
    """
    ev = events.sort_values("date").reset_index(drop=True)
    ev_dates = pd.to_datetime(ev["date"])
    out: dict[str, pd.Series] = {}
    for d in panel.rebal_dates:
        cutoff = pd.Timestamp(d)
        hist_mask = ev_dates < cutoff
        hist = ev[hist_mask]
        if hist.empty:
            continue
        age = (cutoff - ev_dates[hist_mask]).dt.days.values
        w = np.power(2.0, -age / half_life_days)
        hw = pd.DataFrame({"firm": hist["analyst_company"].values,
                           "u": hist["upside"].values, "w": w})
        g = hw.groupby("firm")
        wsum = g["w"].sum()
        wusum = g.apply(lambda x: (x["u"] * x["w"]).sum(), include_groups=False)
        w2sum = g.apply(lambda x: (x["w"] ** 2).sum(), include_groups=False)
        firm_mean = wusum / wsum
        n_eff = wsum ** 2 / w2sum
        global_mean = float((hw["u"] * hw["w"]).sum() / hw["w"].sum())
        shrink = n_eff / (n_eff + shrink_k)
        firm_bias = shrink * firm_mean + (1 - shrink) * global_mean
        win_mask = hist_mask & (ev_dates > cutoff - pd.Timedelta(days=window_days))
        win = ev[win_mask]
        if win.empty:
            continue
        adj = win["upside"].values - win["analyst_company"].map(firm_bias).fillna(global_mean).values
        sig = pd.DataFrame({"ticker": win["ticker"].values, "adj": adj}) \
            .groupby("ticker")["adj"].mean()
        out[d] = sig
    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    panel = load_panel()
    matrix = load_price_matrix(start="2022-06-01")
    fwd = panel_forward_returns(matrix, panel.rebal_dates)
    sectors = sector_map(normalized=False)          # battery parity
    sectors_norm = sector_map(normalized=True)      # robustness variant
    events = load_events(clean=True)

    dates = panel.rebal_dates
    rows = []
    # ------------------------------------------------------------------
    # 1. Per-date cross-sectional relationship between skill and raw
    # ------------------------------------------------------------------
    per_date = []
    for d in dates:
        raws = panel.raws[d]
        if SKILL not in raws.columns or RAW not in raws.columns:
            continue
        x, z = raws[SKILL], raws[RAW]
        both = pd.DataFrame({"x": x, "z": z}).dropna()
        if len(both) < 20:
            continue
        pear = both["x"].corr(both["z"])
        spear = both["x"].corr(both["z"], method="spearman")
        n10 = max(1, int(len(both) * 0.10))
        top_x = set(both["x"].nlargest(n10).index)
        top_z = set(both["z"].nlargest(n10).index)
        per_date.append({
            "date": d, "n_both": len(both),
            "pearson": pear, "spearman": spear, "r2": pear ** 2,
            "rank_overlap_top10": len(top_x & top_z) / n10,
        })
    rel = pd.DataFrame(per_date)
    rel.to_csv(OUT_DIR / "study1_skill_vs_raw_by_date.csv", index=False)

    # ------------------------------------------------------------------
    # 2. IC framework: 3M/6M ICs, incremental and partial
    # ------------------------------------------------------------------
    def scores_of(name):
        return {d: panel.scores[d][name] for d in dates if name in panel.scores[d].columns}

    ic_results = {}
    for name in [SKILL, RAW, "rev_pt_bull_decile_upside_30d"]:
        for h in ["3M", "6M"]:
            ics = ic_series(scores_of(name), fwd[h])
            ic_results[(name, h)] = ics

    # partial IC of skill given raw (rank-based, per date, on raw values)
    partial_ics = {"3M": {}, "6M": {}}
    for h in ["3M", "6M"]:
        for d in dates:
            if d not in fwd[h]:
                continue
            raws = panel.raws[d]
            p = partial_spearman(raws[SKILL], raws[RAW], fwd[h][d])
            if p is not None:
                partial_ics[h][d] = p
    partial_3m = pd.Series(partial_ics["3M"]).sort_index()
    partial_6m = pd.Series(partial_ics["6M"]).sort_index()

    # ------------------------------------------------------------------
    # 3. Parent-composite variants under existing selector weight rules
    # ------------------------------------------------------------------
    def blended_ic(name):
        i3 = ic_results.get((name, "3M"))
        i6 = ic_results.get((name, "6M"))
        if i3 is None or i6 is None:
            i3 = ic_series(scores_of(name), fwd["3M"])
            i6 = ic_series(scores_of(name), fwd["6M"])
        return float(np.mean([i3.mean(), i6.mean()]))

    ic_map = {}
    for name in list(LIVE_FORMULA) + [SKILL, "rev_target_revision_raw_30d"]:
        ic_map[name] = blended_ic(name)

    variants = {
        "P0_live": list(LIVE_FORMULA),
        "P1_live_plus_rawrev": list(LIVE_FORMULA) + ["rev_target_revision_raw_30d"],
        "P2_live_plus_skill": list(LIVE_FORMULA) + [SKILL],
        "P3_live_plus_both": list(LIVE_FORMULA) + ["rev_target_revision_raw_30d", SKILL],
    }
    variant_rows = []
    for vname, subs in variants.items():
        weights, _ = parent_weights(subs, ic_map)
        comp_by_date = {}
        for d in dates:
            sc = panel.scores[d]
            cols = [c for c in weights if c in sc.columns]
            if not cols:
                continue
            wsum = sum(weights[c] for c in cols)
            comp_by_date[d] = sum(sc[c] * weights[c] for c in cols) / wsum
        for h in ["3M", "6M"]:
            ics = ic_series(comp_by_date, fwd[h])
            s = summarize_ic(ics)
            variant_rows.append({"variant": vname, "horizon": h,
                                 "weights": {k: round(v, 3) for k, v in weights.items()},
                                 **s})
    vdf = pd.DataFrame(variant_rows)
    vdf.to_csv(OUT_DIR / "study1_parent_variants.csv", index=False)

    # ------------------------------------------------------------------
    # 4. By-year / semester stability for skill, raw, partial
    # ------------------------------------------------------------------
    stab_rows = []
    for label, ics in [("skill_3M", ic_results[(SKILL, "3M")]),
                       ("skill_6M", ic_results[(SKILL, "6M")]),
                       ("raw_3M", ic_results[(RAW, "3M")]),
                       ("partial_skill_given_raw_3M", partial_3m),
                       ("partial_skill_given_raw_6M", partial_6m)]:
        for grp_fn, gname in [(year_of, "year"), (semester_of, "semester")]:
            g = ics.groupby(ics.index.map(grp_fn)).mean()
            for k, v in g.items():
                stab_rows.append({"signal": label, "period_type": gname,
                                  "period": k, "mean_ic": float(v)})
    pd.DataFrame(stab_rows).to_csv(OUT_DIR / "study1_stability.csv", index=False)

    # ------------------------------------------------------------------
    # 5. Firm-optimism-bias-corrected candidate (redesign #3)
    # ------------------------------------------------------------------
    bias_raw = build_firm_bias_candidate(panel, events, sectors)
    bias_scores = {}
    bias_scores_norm = {}
    for d, sig in bias_raw.items():
        idx = panel.scores[d].index
        sec = sectors.reindex(idx)
        bias_scores[d] = sector_percentile(sig.reindex(idx), sec, higher_is_better=True)
        sec_n = sectors_norm.reindex(idx)
        bias_scores_norm[d] = sector_percentile(sig.reindex(idx), sec_n, higher_is_better=True)

    bias_ic = {}
    for h in ["3M", "6M"]:
        bias_ic[h] = ic_series(bias_scores, fwd[h])
    bias_ic_norm3 = ic_series(bias_scores_norm, fwd["3M"])
    # coverage + corr vs raw & skill
    covs, corr_raw, corr_skill = [], [], []
    for d, sig in bias_raw.items():
        idx = panel.scores[d].index
        covs.append(sig.reindex(idx).notna().mean())
        raws = panel.raws[d]
        cr = pd.DataFrame({"b": sig.reindex(idx), "r": raws[RAW]}).dropna()
        cs = pd.DataFrame({"b": sig.reindex(idx), "s": raws[SKILL]}).dropna()
        if len(cr) >= 20:
            corr_raw.append(cr["b"].corr(cr["r"], method="spearman"))
        if len(cs) >= 20:
            corr_skill.append(cs["b"].corr(cs["s"], method="spearman"))

    # partial IC of bias-candidate given raw upside
    bias_partial = {}
    for d, sig in bias_raw.items():
        if d not in fwd["3M"]:
            continue
        raws = panel.raws[d]
        p = partial_spearman(sig.reindex(raws.index), raws[RAW], fwd["3M"][d])
        if p is not None:
            bias_partial[d] = p
    bias_partial = pd.Series(bias_partial).sort_index()

    # ------------------------------------------------------------------
    # Summary print + save
    # ------------------------------------------------------------------
    summary = {
        "n_dates_with_both": len(rel),
        "mean_pearson": float(rel["pearson"].mean()),
        "mean_spearman": float(rel["spearman"].mean()),
        "mean_r2": float(rel["r2"].mean()),
        "mean_top10_overlap": float(rel["rank_overlap_top10"].mean()),
        "skill_ic_3m": summarize_ic(ic_results[(SKILL, "3M")]),
        "skill_ic_6m": summarize_ic(ic_results[(SKILL, "6M")]),
        "raw_ic_3m": summarize_ic(ic_results[(RAW, "3M")]),
        "raw_ic_6m": summarize_ic(ic_results[(RAW, "6M")]),
        "incremental_ic_3m_skill_minus_raw":
            float(ic_results[(SKILL, "3M")].mean() - ic_results[(RAW, "3M")].mean()),
        "partial_ic_3m": summarize_ic(partial_3m),
        "partial_ic_3m_ci": bootstrap_mean_ci(partial_3m),
        "partial_ic_6m": summarize_ic(partial_6m),
        "partial_ic_6m_ci": bootstrap_mean_ci(partial_6m),
        "bias_candidate_ic_3m": summarize_ic(bias_ic["3M"]),
        "bias_candidate_ic_3m_ci": bootstrap_mean_ci(bias_ic["3M"]),
        "bias_candidate_ic_6m": summarize_ic(bias_ic["6M"]),
        "bias_candidate_ic_3m_normalized_sectors": summarize_ic(bias_ic_norm3),
        "bias_candidate_coverage": float(np.mean(covs)),
        "bias_candidate_spearman_vs_raw": float(np.mean(corr_raw)),
        "bias_candidate_spearman_vs_skill": float(np.mean(corr_skill)),
        "bias_candidate_partial_ic_given_raw_3m": summarize_ic(bias_partial),
        "bias_candidate_partial_ci": bootstrap_mean_ci(bias_partial),
    }
    import json
    with open(OUT_DIR / "study1_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # persist bias candidate scores for later studies
    import pickle
    with open(OUT_DIR / "bias_candidate_scores.pkl", "wb") as f:
        pickle.dump({"raw": bias_raw, "scores": bias_scores}, f)

    print("=== STUDY 1: firm-skill audit ===")
    for k, v in summary.items():
        print(f"{k}: {v}")
    print("\n--- parent variants ---")
    print(vdf.to_string(index=False))


if __name__ == "__main__":
    main()
