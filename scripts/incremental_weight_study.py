"""Incremental-objective parent-weight study (walk-forward, pre-registered).

Implements output/crowding/PREREGISTRATION_incremental_weights_2026-08-04.md
verbatim: annual walk-forward 2019-2026H1 on the deep candidate panel
(2015-06-30→2026-06-30, post-fix rebuild), current SELECTED_SUBS held fixed,
production-replica parent/composite construction. Methods:

  A  — incumbent derivation walk-forward (standalone IC/IR, nominal cap 0.25)
  A0 — live static weight vector (look-ahead reference row, not the comparator)
  B  — effective-exposure cap 0.25 (cap binds C·w, not w)
  C  — uniqueness discount: w ∝ s_i·(1−R²_i), nominal cap 0.25
  D  — shrunk Markowitz-on-IC: (0.5·I + 0.5·C)⁻¹·max(IC,0), clip, cap 0.25

Success bar and inference exactly as pre-registered. Outputs to
output/crowding/incremental_weights/.

Usage: python scripts/incremental_weight_study.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from factors.parent_selection_v4 import SELECTED_SUBS, V4_PARENT_WEIGHTS
from factors.utils import sector_percentile
from research.subfactor_expansion.panel import load_cached_panel
from research.analyst_deep_dive.common import (
    load_price_matrix, panel_forward_returns, spearman_ic, sector_map,
)
from scripts.crowding_diagnostics import parent_score, normalize

PANEL_PKL = REPO / "cache" / "subfactor_expansion" / "cand_panel_2015-06-30_2026-06-30_monthly_v2.pkl"
OUT = REPO / "output" / "crowding" / "incremental_weights"
PARENTS = list(SELECTED_SUBS)
CAP = 0.25
LAM = 0.5
EVAL_YEARS = list(range(2019, 2027))
MIN_IC_DATES = 24
SEED, N_BOOT = 42, 2000


def water_fill(w: dict[str, float], cap: float = CAP) -> dict[str, float]:
    w = {p: max(x, 0.0) for p, x in w.items()}
    tot = sum(w.values())
    if tot <= 0:
        return {p: 0.0 for p in w}
    w = {p: x / tot for p, x in w.items()}
    for _ in range(20):
        over = [p for p, x in w.items() if x > cap + 1e-12]
        if not over:
            break
        excess = sum(w[p] - cap for p in over)
        for p in over:
            w[p] = cap
        under = [p for p, x in w.items() if x < cap - 1e-12 and x > 0]
        pool = sum(w[p] for p in under)
        if not under or pool <= 1e-12:
            break
        for p in under:
            w[p] += excess * w[p] / pool
    tot = sum(w.values())
    return {p: x / tot for p, x in w.items()}


def combined_scores(ic: pd.Series, ir: pd.Series) -> pd.Series:
    ic, ir = ic.clip(lower=0.0), ir.clip(lower=0.0)
    ic_n = ic / ic.max() if ic.max() > 0 else ic * 0.0
    ir_n = ir / ir.max() if ir.max() > 0 else ir * 0.0
    return 0.5 * ic_n + 0.5 * ir_n


def method_A(ic, ir, C):
    return water_fill(combined_scores(ic, ir).to_dict())


def method_B(ic, ir, C):
    s = combined_scores(ic, ir)
    tot = s.sum()
    if tot <= 0:
        return {p: 0.0 for p in PARENTS}
    w = (s / tot).copy()
    for _ in range(50):
        eff = C.values @ w.values
        over = eff > CAP + 1e-12
        if eff.max() <= 0.255:
            break
        w[over] = w[over] * (CAP / eff[over])
        w = w / w.sum()
    return w.to_dict()


def method_C(ic, ir, C):
    Cinv = np.linalg.pinv(C.values)
    r2 = 1.0 - 1.0 / np.clip(np.diag(Cinv), 1.0, None)
    s = combined_scores(ic, ir) * (1.0 - r2)
    return water_fill(s.to_dict())


def method_D(ic, ir, C):
    icv = ic.clip(lower=0.0).values
    M = LAM * np.eye(len(PARENTS)) + (1 - LAM) * C.values
    w = np.linalg.solve(M, icv)
    return water_fill(dict(zip(PARENTS, w)))


METHODS = {"A": method_A, "B": method_B, "C": method_C, "D": method_D}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    panel = load_cached_panel(PANEL_PKL)
    if panel is None:
        raise SystemExit(f"no panel at {PANEL_PKL} — build it first")
    dates = panel.rebal_dates
    print(f"panel: {len(dates)} dates {dates[0]}..{dates[-1]}")
    sectors = sector_map(normalized=False)

    # Parent scores (production replica) once for every date.
    P_by, Pn_by = {}, {}
    for d in dates:
        frame = panel.scores[d]
        P = pd.DataFrame({p: parent_score(frame, SELECTED_SUBS[p]) for p in PARENTS})
        P_by[d] = P
        Pn_by[d] = P.apply(normalize)

    matrix = load_price_matrix(start="2015-01-01", end="2026-07-29")
    fwd = panel_forward_returns(matrix, dates)

    # Per-date parent ICs (3M/6M) for derivation stats.
    par_ic = {h: pd.DataFrame(index=dates, columns=PARENTS, dtype=float)
              for h in ("3M", "6M")}
    for d in dates:
        for h in ("3M", "6M"):
            if d not in fwd[h]:
                continue
            for p in PARENTS:
                v = spearman_ic(P_by[d][p], fwd[h][d])
                if v is not None:
                    par_ic[h].loc[d, p] = v

    # Per-date parent correlation (Pearson, normalized) for derivation C.
    corr_by = {d: Pn_by[d].corr(method="pearson") for d in dates}

    static_w = pd.Series(V4_PARENT_WEIGHTS)[PARENTS]
    static_w = static_w / static_w.sum()

    weights_rows, skipped = [], []
    comp = {m: {} for m in list(METHODS) + ["A0"]}
    struct_rows = []
    for Y in EVAL_YEARS:
        d0, d1 = f"{Y-5}-01-01", f"{Y-1}-12-31"
        deriv = [d for d in dates if d0 <= d <= d1]
        lim3, lim6 = f"{Y-1}-09-30", f"{Y-1}-06-30"
        blend_rows = {}
        for d in deriv:
            vals = []
            if d <= lim3:
                vals.append(par_ic["3M"].loc[d])
            if d <= lim6:
                vals.append(par_ic["6M"].loc[d])
            if vals:
                blend_rows[d] = pd.concat(vals, axis=1).mean(axis=1)
        blend = pd.DataFrame(blend_rows).T
        n_ok = int(blend.notna().all(axis=1).sum())
        if n_ok < MIN_IC_DATES:
            skipped.append({"year": Y, "usable_ic_dates": n_ok})
            continue
        ic_mean = blend.mean()
        ic_ir = blend.mean() / blend.std(ddof=1)
        C = (pd.concat([corr_by[d] for d in deriv])
             .groupby(level=0, sort=False).mean().loc[PARENTS, PARENTS])

        year_dates = [d for d in dates if d.startswith(str(Y))]
        year_w = {}
        for m, fn in METHODS.items():
            w = pd.Series(fn(ic_mean, ic_ir, C))[PARENTS].fillna(0.0)
            year_w[m] = w
        year_w["A0"] = static_w

        for m, w in year_w.items():
            eff = C.values @ w.values
            struct_rows.append({
                "year": Y, "method": m,
                "eff_n_bets": round(1.0 / float(w.values @ C.values @ w.values), 2),
                "max_eff_exposure": round(float(eff.max()), 3),
                "n_active_parents": int((w > 0.01).sum()),
            })
            weights_rows.append({"year": Y, "method": m,
                                 **{p: round(float(w[p]), 3) for p in PARENTS}})
            for d in year_dates:
                raw = Pn_by[d].mul(w).sum(axis=1)
                sec = sectors.reindex(raw.index)
                comp[m][d] = sector_percentile(raw, sec, higher_is_better=True)

    pd.DataFrame(weights_rows).to_csv(OUT / "weights_by_year.csv", index=False)
    pd.DataFrame(struct_rows).to_csv(OUT / "structure_by_year.csv", index=False)
    if skipped:
        print("skipped years:", skipped)

    # Pooled OOS evaluation.
    def comp_ic_series(cs, h):
        out = {}
        for d, s in cs.items():
            if d in fwd[h]:
                v = spearman_ic(s, fwd[h][d])
                if v is not None:
                    out[d] = v
        return pd.Series(out).sort_index()

    def spread(cs, h):
        vals = []
        for d, s in cs.items():
            if d not in fwd[h]:
                continue
            df = pd.DataFrame({"s": s, "f": fwd[h][d]}).dropna()
            if len(df) < 100:
                continue
            q = pd.qcut(df["s"].rank(method="first"), 5, labels=False)
            m = df.groupby(q)["f"].mean()
            vals.append(float(m.iloc[-1] - m.iloc[0]))
        return float(np.mean(vals)) if vals else np.nan

    results, blend_by_method = {}, {}
    for m, cs in comp.items():
        s3, s6 = comp_ic_series(cs, "3M"), comp_ic_series(cs, "6M")
        blend = pd.concat([s3, s6], axis=1).mean(axis=1)
        blend_by_method[m] = blend
        yearly = blend.groupby(blend.index.str[:4]).mean().round(4)
        results[m] = {
            "ic_3M": round(float(s3.mean()), 4), "ic_6M": round(float(s6.mean()), 4),
            "ic_blend": round(float(blend.mean()), 4),
            "ir_blend": round(float(blend.mean() / blend.std(ddof=1)), 3),
            "hit_blend": round(float((blend > 0).mean()), 3),
            "n_dates": int(len(blend)),
            "q5q1_3M": round(spread(cs, "3M"), 4), "q5q1_6M": round(spread(cs, "6M"), 4),
            "by_year": yearly.to_dict(),
        }

    # Paired diffs vs baseline A (month bootstrap, pre-registered).
    rng = np.random.default_rng(SEED)
    diffs = {}
    base = blend_by_method["A"]
    for m in METHODS:
        if m == "A":
            continue
        d = (blend_by_method[m] - base).dropna()
        boots = np.array([rng.choice(d.values, size=len(d), replace=True).mean()
                          for _ in range(N_BOOT)])
        diffs[m] = {"mean_diff": round(float(d.mean()), 4),
                    "ci90": [round(float(np.quantile(boots, 0.05)), 4),
                             round(float(np.quantile(boots, 0.95)), 4)],
                    "n": int(len(d))}

    out = {"results": results, "paired_vs_A": diffs,
           "skipped_years": skipped}
    with open(OUT / "results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
