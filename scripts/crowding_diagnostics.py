"""Cross-parent correlation / double-counting ("crowding") diagnostics.

Question (user, 2026-08-04): how correlated are the production subfactors with
OTHER parents, and the parents with each other? Since the composite is a
weighted blend of dispersion-equalized parents, two correlated parents both
carrying weight is an implicit bet-size increase on their shared driver — the
25% parent cap constrains nominal weights, not the effective exposure.

DIAGNOSTIC ONLY — descriptive statistics of the live model's structure, no
signal search, no production changes. Replicates the production Layer-2 stack
exactly on the cached battery panel (42 monthly PIT rebalances 2023-01→2026-06):

  sub sector-percentiles (panel)             — factors.utils.sector_percentile
  parent = renormalized Σ w_s · sub_s        — factors.base.score_factor
  parent_norm = 50 + (parent−50)·(20/std)    — factors.composite._normalize_parents
  composite_raw = Σ W_p · parent_norm        — factors.composite.build_composite

Static engine weights (config factors.engine_weights); the VIX momentum tilt is
ignored here (noted as a caveat in the report).

Outputs to output/crowding/:
  parent_corr_spearman.csv    — 8×8 mean per-date parent correlation
  sub_x_parent_corr.csv       — 24 production subs × 8 parents
  sub_pairs_cross_parent.csv  — cross-parent sub pairs with mean |rho| ≥ 0.40
  effective_weights.csv       — nominal vs correlation-adjusted effective weight
  parent_ic_context.csv       — in-window parent IC 3M/6M (context only)
  REPORT.md                   — full write-up

Usage: python scripts/crowding_diagnostics.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from factors.parent_selection_v4 import SELECTED_SUBS, V4_PARENT_WEIGHTS
from research.analyst_deep_dive.common import (
    load_panel, load_price_matrix, panel_forward_returns, ic_series, summarize_ic,
)

OUT = REPO / "output" / "crowding"
NEUTRAL, TARGET_STD, MIN_STD = 50.0, 20.0, 1e-6

# The 2026-08-03-morning revisions formula, kept for a before/after crowding
# read on the one slot the 08-04 re-ratification removed (momentum-tainted).
OLD_REVISIONS = {"rev_pt_upgrade_ratio_30d": 0.425,
                 "rev_raise_and_bullish_90d": 0.365,
                 "rev_target_revision_raw_30d": 0.210}


def parent_score(frame: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    cols = [c for c in weights if c in frame.columns]
    if not cols:
        raise KeyError(f"none of {list(weights)} in panel")
    w = np.array([weights[c] for c in cols])
    w = w / w.sum()
    return frame[cols].fillna(NEUTRAL).mul(w, axis=1).sum(axis=1)


def normalize(p: pd.Series) -> pd.Series:
    std = float(p.std())
    if std < MIN_STD:
        return p
    return NEUTRAL + (p - NEUTRAL) * (TARGET_STD / std)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    panel = load_panel()
    dates = panel.rebal_dates
    W = dict(V4_PARENT_WEIGHTS)
    parents = list(SELECTED_SUBS)
    subs = [(p, s) for p in parents for s in SELECTED_SUBS[p]]
    sub_names = [s for _, s in subs]
    print(f"{len(dates)} dates, {len(parents)} parents, {len(subs)} production subs")

    missing = [s for s in sub_names if s not in panel.scores[dates[-1]].columns]
    if missing:
        raise SystemExit(f"panel missing production subs: {missing}")

    sp_mats, pe_mats, sxp_mats, sxs_mats = [], [], [], []
    comp_rho, rev_old_new = [], []
    parent_by_date = {p: {} for p in parents}
    for d in dates:
        frame = panel.scores[d]
        P = pd.DataFrame({p: parent_score(frame, SELECTED_SUBS[p]) for p in parents})
        for p in parents:
            parent_by_date[p][d] = P[p]
        Pn = P.apply(normalize)
        comp = Pn.mul(pd.Series(W)).sum(axis=1) / sum(W.values())

        S = frame[sub_names].fillna(NEUTRAL)
        both = pd.concat([S, P], axis=1)
        sp = P.corr(method="spearman")
        sp_mats.append(sp)
        pe_mats.append(Pn.corr(method="pearson"))
        c = both.corr(method="spearman")
        sxp_mats.append(c.loc[sub_names, parents])
        sxs_mats.append(c.loc[sub_names, sub_names])
        comp_rho.append(P.corrwith(comp, method="spearman"))

        rev_old = parent_score(frame, OLD_REVISIONS)
        rev_old_new.append({
            "old_vs_momentum": rev_old.corr(P["momentum"], method="spearman"),
            "new_vs_momentum": P["revisions"].corr(P["momentum"], method="spearman"),
        })

    def nanmean_mats(mats):
        return pd.concat(mats).groupby(level=0, sort=False).mean()

    sp_mean = nanmean_mats(sp_mats)
    pe_mean = nanmean_mats(pe_mats)
    sxp_mean = nanmean_mats(sxp_mats)
    sxs_mean = nanmean_mats(sxs_mats)
    comp_rho = pd.DataFrame(comp_rho).mean()
    rev_shift = pd.DataFrame(rev_old_new).mean()

    sp_mean.round(3).to_csv(OUT / "parent_corr_spearman.csv")

    sxp = sxp_mean.round(3).copy()
    own = pd.Series({s: p for p, s in subs})
    sxp.insert(0, "own_parent", own)
    off = sxp_mean.copy()
    for s, p in own.items():
        off.loc[s, p] = np.nan
    sxp["max_other_parent"] = off.abs().idxmax(axis=1)
    sxp["max_other_rho"] = [off.loc[s, sxp.loc[s, "max_other_parent"]].round(3)
                            for s in sxp.index]
    sxp.to_csv(OUT / "sub_x_parent_corr.csv")

    pairs = []
    for i, (pi, si) in enumerate(subs):
        for pj, sj in subs[i + 1:]:
            if pi == pj:
                continue
            r = sxs_mean.loc[si, sj]
            if abs(r) >= 0.40:
                pairs.append({"sub_a": si, "parent_a": pi, "sub_b": sj,
                              "parent_b": pj, "mean_rho": round(float(r), 3)})
    pairs_df = pd.DataFrame(pairs).sort_values("mean_rho", key=abs,
                                               ascending=False) if pairs else pd.DataFrame()
    pairs_df.to_csv(OUT / "sub_pairs_cross_parent.csv", index=False)

    # Effective weight: composite = Σ W_p · z_p with equal stds, so
    # Var = Σ_ij W_i W_j C_ij (Pearson C on normalized parents). Parent i's
    # effective exposure eff_i = Σ_j W_j C_ij (= beta of composite on parent i,
    # in units of a stand-alone parent); variance share = W_i·eff_i / Var.
    w = pd.Series(W)[parents]
    w = w / w.sum()
    eff = pe_mean.loc[parents, parents].mul(w, axis=1).sum(axis=1)
    var_share = (w * eff) / float(w @ pe_mean.loc[parents, parents] @ w)
    eff_df = pd.DataFrame({
        "nominal_weight": w.round(3),
        "effective_exposure": eff.round(3),
        "amplification": (eff / w).round(2),
        "variance_share": var_share.round(3),
        "rho_with_composite": comp_rho.round(3),
    }).sort_values("variance_share", ascending=False)
    eff_df.to_csv(OUT / "effective_weights.csv")

    # Portfolio-level redundancy: composite variance w'Cw (equal parent stds)
    # vs the Σw² it would be with independent parents. DR² = 1/(w'Cw) is the
    # effective number of independent equal-risk bets these weights buy.
    C = pe_mean.loc[parents, parents]
    var_actual = float(w @ C @ w)
    var_indep = float((w ** 2).sum())
    enb = {"var_actual": round(var_actual, 4), "var_if_independent": round(var_indep, 4),
           "variance_inflation": round(var_actual / var_indep, 2),
           "effective_n_bets": round(1.0 / var_actual, 2),
           "n_bets_if_independent": round(1.0 / var_indep, 2)}

    # In-window parent ICs for the "should correlated-and-good get MORE weight"
    # discussion (context only — same window the weights were derived on).
    matrix = load_price_matrix(start="2022-06-01", end="2026-07-29")
    fwd = panel_forward_returns(matrix, dates)
    ic_rows = {}
    for p in parents:
        row = {}
        for h in ("3M", "6M"):
            s = summarize_ic(ic_series(parent_by_date[p], fwd[h]))
            row[f"ic_{h}"] = round(s["mean_ic"], 4)
            row[f"ir_{h}"] = round(s["ic_ir"], 3)
        ic_rows[p] = row
    ic_df = pd.DataFrame(ic_rows).T
    ic_df.to_csv(OUT / "parent_ic_context.csv")

    print("\n=== effective number of bets ===")
    print(enb)
    print("\n=== parent corr (mean per-date Spearman) ===")
    print(sp_mean.round(2).to_string())
    print("\n=== effective weights ===")
    print(eff_df.to_string())
    print("\n=== revisions-vs-momentum, old vs new formula ===")
    print(rev_shift.round(3).to_string())
    print("\n=== parent IC context (in-window) ===")
    print(ic_df.to_string())
    print(f"\n=== cross-parent sub pairs |rho|>=0.40: {len(pairs_df)} ===")
    if len(pairs_df):
        print(pairs_df.head(15).to_string(index=False))

    with open(OUT / "raw_results.txt", "w") as f:
        f.write("parent_corr_spearman\n" + sp_mean.round(3).to_string())
        f.write("\n\nparent_corr_pearson_normalized\n" + pe_mean.round(3).to_string())
        f.write("\n\neffective_weights\n" + eff_df.to_string())
        f.write("\n\neffective_n_bets\n" + str(enb))
        f.write("\n\nrevisions_vs_momentum_old_new\n" + rev_shift.round(3).to_string())
        f.write("\n\nparent_ic_context\n" + ic_df.to_string())
        f.write(f"\n\ncross_parent_pairs\n{pairs_df.to_string(index=False) if len(pairs_df) else 'none'}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
