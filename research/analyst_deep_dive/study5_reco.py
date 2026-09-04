"""Study 5 — Improvements to the recommendation-style signals.

Baselines kept as-is: rev_pt_upgrade_ratio_30d (share of trailing-30d target
events above spot) and rev_grade_diffusion (net buy/sell level). New variants
(deliberately few — no hyperparameter sweeps):

  A. upgrade_ratio_decay_90d  — 90d window, 30d-half-life recency weights.
  B. pct_bullish_dedup_90d    — latest event per firm in 90d, share bullish
                                (breadth-clean: one vote per firm).
  C. agreement_90d            — |weighted mean sign| with per-firm dedup:
                                do the covering firms agree on direction?
  D. raise_and_bullish_90d    — share of firms whose latest event is BOTH a
                                raise vs their own prior target AND above spot
                                (confirmation interaction).
  E. diffusion_x_upside       — grade diffusion percentile × target-upside
                                percentile ("Strong Buy + high upside").
  F. upgrade_x_agreement      — upgrade-ratio percentile × agreement
                                percentile ("high upside + broad agreement").

Rating-level "NetRevision = upgrades - downgrades" from the grades table is
month-over-month diffusion change — already tested and failed in the battery
(rev_grade_diffusion_change_90d, rev_rating_change_30d/90d); not retested per
the do-not-retest instruction. Firm-selectivity on RATINGS is not feasible:
analyst_grades has no firm identity (counts only) — documented limitation.

Run:  python -m research.analyst_deep_dive.study5_reco
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from factors.utils import sector_percentile
from research.analyst_deep_dive.common import (
    OUT_DIR, load_panel, load_price_matrix, load_events, panel_forward_returns,
    sector_map, ic_series, summarize_ic, bootstrap_mean_ci,
)

HALF_LIFE = 30.0
WINDOW = 90


def build_signals(panel, events: pd.DataFrame) -> dict[str, dict[str, pd.Series]]:
    ev = events.sort_values(["ticker", "analyst_company", "date"]).reset_index(drop=True)
    grp = ev.groupby(["ticker", "analyst_company"])
    ev["prev_target"] = grp["price_target"].shift(1)
    ev["prev_date"] = grp["date"].shift(1)
    ev_dates = pd.to_datetime(ev["date"])
    prior_age = (ev_dates - pd.to_datetime(ev["prev_date"])).dt.days
    ev["is_raise"] = np.where(ev["prev_target"].notna() & (prior_age <= 365),
                              (ev["price_target"] > ev["prev_target"]).astype(float), np.nan)
    ev["bullish"] = (ev["upside"] > 0).astype(float)

    out: dict[str, dict[str, pd.Series]] = {
        "upgrade_ratio_decay_90d": {}, "pct_bullish_dedup_90d": {},
        "agreement_90d": {}, "raise_and_bullish_90d": {},
    }
    for d in panel.rebal_dates:
        cutoff = pd.Timestamp(d)
        m = (ev_dates <= cutoff) & (ev_dates > cutoff - pd.Timedelta(days=WINDOW))
        win = ev[m].copy()
        if win.empty:
            continue
        win["age"] = (cutoff - ev_dates[m]).dt.days
        win["w"] = np.power(2.0, -win["age"] / HALF_LIFE)

        # A: recency-weighted bullish share over ALL events in window
        g = win.groupby("ticker")
        out["upgrade_ratio_decay_90d"][d] = g.apply(
            lambda x: (x["bullish"] * x["w"]).sum() / x["w"].sum(), include_groups=False)

        # dedup to latest event per (ticker, firm)
        latest = win.sort_values("date").drop_duplicates(["ticker", "analyst_company"], keep="last")
        gl = latest.groupby("ticker")
        out["pct_bullish_dedup_90d"][d] = gl["bullish"].mean()
        out["agreement_90d"][d] = gl.apply(
            lambda x: abs((np.sign(x["upside"]) * x["w"]).sum()) / x["w"].sum(),
            include_groups=False)
        both = latest[latest["is_raise"].notna()]
        gb = both.groupby("ticker")
        raise_bull = gb.apply(lambda x: ((x["is_raise"] > 0) & (x["bullish"] > 0)).mean(),
                              include_groups=False)
        out["raise_and_bullish_90d"][d] = raise_bull
    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    panel = load_panel()
    matrix = load_price_matrix(start="2022-06-01")
    fwd = panel_forward_returns(matrix, panel.rebal_dates)
    sectors = sector_map(normalized=False)
    events = load_events(clean=True)
    dates = panel.rebal_dates

    new_raw = build_signals(panel, events)

    # interaction candidates from within-date percentile ranks
    inter_raw: dict[str, dict[str, pd.Series]] = {"diffusion_x_upside": {}, "upgrade_x_agreement": {}}
    for d in dates:
        raws = panel.raws[d]
        idx = panel.scores[d].index
        diff_p = raws["rev_grade_diffusion"].rank(pct=True)
        ups_p = raws["rev_pt_target_upside_30d"].rank(pct=True)
        inter_raw["diffusion_x_upside"][d] = (diff_p * ups_p).reindex(idx)
        if d in new_raw["agreement_90d"]:
            ur_p = raws["rev_pt_upgrade_ratio_30d"].rank(pct=True)
            ag_p = new_raw["agreement_90d"][d].reindex(idx).rank(pct=True)
            inter_raw["upgrade_x_agreement"][d] = (ur_p * ag_p).reindex(idx)

    all_new = {**new_raw, **inter_raw}

    def score_dict(raw_by_date):
        out = {}
        for d, sig in raw_by_date.items():
            idx = panel.scores[d].index
            out[d] = sector_percentile(sig.reindex(idx), sectors.reindex(idx), True)
        return out

    baselines = {
        "BASE_upgrade_ratio_30d": {d: panel.scores[d]["rev_pt_upgrade_ratio_30d"] for d in dates},
        "BASE_grade_diffusion": {d: panel.scores[d]["rev_grade_diffusion"] for d in dates},
    }
    candidates = {name: score_dict(raw) for name, raw in all_new.items()}

    rows = []
    ic_store = {}
    for name, sc in {**baselines, **candidates}.items():
        ic3 = ic_series(sc, fwd["3M"])
        ic6 = ic_series(sc, fwd["6M"])
        ic_store[name] = {"3M": ic3, "6M": ic6}
        cov = np.nan
        if name in all_new:
            covs = [all_new[name][d].reindex(panel.scores[d].index).notna().mean()
                    for d in all_new[name]]
            cov = float(np.mean(covs))
        mean_blend = float(np.mean([ic3.mean(), ic6.mean()]))
        _, lo, hi = bootstrap_mean_ci(ic3)
        rows.append({
            "signal": name, "coverage": cov,
            "ic_3m": float(ic3.mean()), "ic_6m": float(ic6.mean()),
            "ic_3m6m": mean_blend,
            "ic_ir_3m": summarize_ic(ic3)["ic_ir"],
            "hit_rate_3m": summarize_ic(ic3)["hit_rate"],
            "ic3m_ci_lo": lo, "ic3m_ci_hi": hi,
            "n_periods": len(ic3),
        })
    res = pd.DataFrame(rows).sort_values("ic_3m6m", ascending=False)

    # correlations of each new candidate vs the two baselines + skill + raw upside
    corr_rows = []
    ref_cols = ["rev_pt_upgrade_ratio_30d", "rev_grade_diffusion",
                "rev_firm_skill_weighted_upside_30d", "rev_pt_target_upside_30d"]
    for name, raw in all_new.items():
        accum = {r: [] for r in ref_cols}
        for d, sig in raw.items():
            raws = panel.raws[d]
            for r in ref_cols:
                pair = pd.DataFrame({"a": sig.reindex(raws.index), "b": raws[r]}).dropna()
                if len(pair) >= 20:
                    accum[r].append(pair["a"].corr(pair["b"], method="spearman"))
        corr_rows.append({"signal": name,
                          **{f"corr_{r}": float(np.mean(v)) if v else np.nan
                             for r, v in accum.items()}})
    cdf = pd.DataFrame(corr_rows)

    res.to_csv(OUT_DIR / "study5_candidates.csv", index=False)
    cdf.to_csv(OUT_DIR / "study5_correlations.csv", index=False)
    with open(OUT_DIR / "study5_scores.pkl", "wb") as f:
        pickle.dump({name: sc for name, sc in candidates.items()}, f)

    print("=== STUDY 5: recommendation-signal candidates ===")
    print(res.round(4).to_string(index=False))
    print("\n--- mean Spearman vs existing signals ---")
    print(cdf.round(3).to_string(index=False))

    # by-year stability of anything that beats both baselines on ic_3m6m
    base_best = res[res["signal"].str.startswith("BASE")]["ic_3m6m"].max()
    winners = res[(~res["signal"].str.startswith("BASE")) & (res["ic_3m6m"] > base_best)]
    stab_rows = []
    for name in winners["signal"]:
        for h in ["3M", "6M"]:
            s = ic_store[name][h]
            g = s.groupby(s.index.map(lambda x: x[:4])).mean()
            for yr, v in g.items():
                stab_rows.append({"signal": name, "horizon": h, "year": yr, "mean_ic": float(v)})
    if stab_rows:
        pd.DataFrame(stab_rows).to_csv(OUT_DIR / "study5_winner_stability.csv", index=False)
        print("\n--- winners by year ---")
        print(pd.DataFrame(stab_rows).pivot_table(index=["signal", "horizon"],
                                                  columns="year", values="mean_ic").round(4))


if __name__ == "__main__":
    main()
