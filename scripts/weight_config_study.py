"""Weight-configuration walk-forward comparison (2026-09-01).

User question: how much does the effective-exposure cap / floor actually
matter vs. just targeting R² directly, vs. doing nothing (equal weight)?
Extends the pre-registered walk-forward harness
(scripts/incremental_weight_study.py — 5y lookback derivation, re-derived
annually, evaluated OOS 2019-2026, current SELECTED_SUBS held fixed, no
look-ahead) with 8 configurations:

  A            incumbent IC/IR-blend, nominal cap 0.25 (pre-registered)
  B            effective-exposure cap 0.25 (pre-registered, shipped 2026-08-07)
  EQ           equal nominal weight (1/8 each), no cap
  EQ_CAP       equal weight + effective-exposure cap 0.25
  EQ_CAP_FLOOR equal weight + eff-exposure cap 0.25 + diversifier floor
               (parents below 0.05 effective exposure floored to 0.08 nominal)
  E            A's IC/IR-blend + eff-exposure cap + diversifier floor
               (reproduces the shipped production method inside this harness)
  EQEFF        solve for w that equalizes EFFECTIVE exposure across parents
               (w ∝ pinv(C)·1, clipped ≥0, cap 0.25) — the "risk-parity on
               correlation" solution, no IC information used at all
  R2FLOOR      A's IC/IR-blend + cap 0.25, then any parent whose R² vs the
               composite (closed-form: R²_i = (C·w)_i² / w'Cw) falls below
               0.05 is floored to 0.08 nominal weight

R² per parent is the same closed-form used for effective exposure  (R²_i =
corr(P_i, composite)² since parent scores are unit-variance after
normalization) — consistent with the dashboard's realized-output R², just
computed analytically from the derivation-window correlation matrix instead
of empirically from one live date, so it can be tracked walk-forward.

For each config, per eval year: weights, effective exposure per parent, R²
per parent, eff # of bets (1/w'Cw), max effective exposure, n active
parents. Pooled OOS: IC (3M/6M/blend), IR, Q5-Q1 spread, hit rate, 90% CI of
the mean IC difference vs A (month-block bootstrap, same as the
pre-registered study).

Outputs to output/crowding/weight_config_study/:
  weights_by_year.csv       — nominal weight per parent/method/year
  structure_by_year.csv     — eff exposure + R² per parent/method/year
  config_summary.csv        — one row per config: pooled IC/IR/CI/eff bets/spread
  results.json              — full results dict
  REPORT.md                 — write-up

Usage: python scripts/weight_config_study.py
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
OUT = REPO / "output" / "crowding" / "weight_config_study"
PARENTS = list(SELECTED_SUBS)
CAP = 0.25
FLOOR_EFF = 0.05
FLOOR_R2 = 0.05
FLOOR_W = 0.08
EVAL_YEARS = list(range(2019, 2027))
MIN_IC_DATES = 24
SEED, N_BOOT = 42, 2000


# --- weight-derivation primitives (mirror scripts/incremental_weight_study.py) ---

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


def cap_trim(w: pd.Series, C: pd.DataFrame, cap: float = CAP, iters: int = 50) -> pd.Series:
    """Method-B style effective-exposure cap trim (in place on a copy)."""
    w = w.copy()
    for _ in range(iters):
        eff = C.values @ w.values
        over = eff > cap + 1e-12
        if eff.max() <= cap * 1.02:
            break
        w[over] = w[over] * (cap / eff[over])
        w = w / w.sum()
    return w


def eff_exposure_metric(w: pd.Series, C: pd.DataFrame) -> pd.Series:
    return pd.Series(C.values @ w.values, index=PARENTS)


def r2_metric(w: pd.Series, C: pd.DataFrame) -> pd.Series:
    var = float(w.values @ C.values @ w.values)
    if var <= 1e-12:
        return pd.Series(0.0, index=PARENTS)
    eff = C.values @ w.values
    return pd.Series((eff ** 2) / var, index=PARENTS)


def cap_and_floor(w0: dict[str, float], C: pd.DataFrame, metric_fn, floor_thresh: float,
                   floor_w: float = FLOOR_W, cap: float = CAP, iters: int = 300) -> dict[str, float]:
    w = pd.Series(w0)[PARENTS].astype(float)
    w = water_fill(w.to_dict(), cap)
    w = pd.Series(w)[PARENTS]
    for _ in range(iters):
        w = cap_trim(w, C, cap)
        metric = metric_fn(w, C)
        under = [p for p in PARENTS if metric[p] < floor_thresh and w[p] < floor_w - 1e-9]
        if not under:
            break
        above = [p for p in PARENTS if p not in under and w[p] > floor_w]
        pool = sum(w[p] for p in above)
        if pool <= 1e-9:
            break
        deficit = sum(floor_w - w[p] for p in under)
        for p in under:
            w[p] = floor_w
        for p in above:
            w[p] -= deficit * w[p] / pool
        w = w.clip(lower=0.0)
        w = w / w.sum()
    return w.to_dict()


# --- configurations ---

def method_A(ic, ir, C):
    return water_fill(combined_scores(ic, ir).to_dict())


def method_B(ic, ir, C):
    s = combined_scores(ic, ir)
    tot = s.sum()
    if tot <= 0:
        w = pd.Series({p: 0.0 for p in PARENTS})
    else:
        w = (s / tot).copy()
    return cap_trim(w, C).to_dict()


def method_EQ(ic, ir, C):
    return {p: 1.0 / len(PARENTS) for p in PARENTS}


def method_EQ_CAP(ic, ir, C):
    w = pd.Series({p: 1.0 / len(PARENTS) for p in PARENTS})
    return cap_trim(w, C).to_dict()


def method_EQ_CAP_FLOOR(ic, ir, C):
    w0 = {p: 1.0 / len(PARENTS) for p in PARENTS}
    return cap_and_floor(w0, C, eff_exposure_metric, FLOOR_EFF)


def method_E(ic, ir, C):
    w0 = method_A(ic, ir, C)
    return cap_and_floor(w0, C, eff_exposure_metric, FLOOR_EFF)


def method_EQEFF(ic, ir, C):
    ones = np.ones(len(PARENTS))
    raw = np.linalg.pinv(C.values) @ ones
    raw = np.clip(raw, 0.0, None)
    if raw.sum() <= 1e-9:
        raw = ones
    w = pd.Series(raw, index=PARENTS)
    return cap_trim(w / w.sum(), C).to_dict()


def method_R2FLOOR(ic, ir, C):
    w0 = method_A(ic, ir, C)
    return cap_and_floor(w0, C, r2_metric, FLOOR_R2)


METHODS = {
    "A": method_A, "B": method_B,
    "EQ": method_EQ, "EQ_CAP": method_EQ_CAP, "EQ_CAP_FLOOR": method_EQ_CAP_FLOOR,
    "E": method_E, "EQEFF": method_EQEFF, "R2FLOOR": method_R2FLOOR,
}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    panel = load_cached_panel(PANEL_PKL)
    if panel is None:
        raise SystemExit(f"no panel at {PANEL_PKL} — build it first")
    dates = panel.rebal_dates
    print(f"panel: {len(dates)} dates {dates[0]}..{dates[-1]}")
    sectors = sector_map(normalized=False)

    P_by, Pn_by = {}, {}
    for d in dates:
        frame = panel.scores[d]
        P = pd.DataFrame({p: parent_score(frame, SELECTED_SUBS[p]) for p in PARENTS})
        P_by[d] = P
        Pn_by[d] = P.apply(normalize)

    matrix = load_price_matrix(start="2015-01-01", end="2026-07-29")
    fwd = panel_forward_returns(matrix, dates)

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

    corr_by = {d: Pn_by[d].corr(method="pearson") for d in dates}

    static_w = pd.Series(V4_PARENT_WEIGHTS)[PARENTS]
    static_w = static_w / static_w.sum()

    weights_rows, struct_rows, skipped = [], [], []
    comp = {m: {} for m in list(METHODS) + ["A0"]}
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
            eff = eff_exposure_metric(w, C)
            r2 = r2_metric(w, C)
            struct_rows.append({
                "year": Y, "method": m,
                "eff_n_bets": round(1.0 / float(w.values @ C.values @ w.values), 2),
                "max_eff_exposure": round(float(eff.max()), 3),
                "min_eff_exposure": round(float(eff.min()), 3),
                "n_active_parents": int((w > 0.01).sum()),
                **{f"w_{p}": round(float(w[p]), 4) for p in PARENTS},
                **{f"eff_{p}": round(float(eff[p]), 4) for p in PARENTS},
                **{f"r2_{p}": round(float(r2[p]), 4) for p in PARENTS},
            })
            weights_rows.append({"year": Y, "method": m,
                                 **{p: round(float(w[p]), 4) for p in PARENTS}})
            for d in year_dates:
                raw = Pn_by[d].mul(w).sum(axis=1)
                sec = sectors.reindex(raw.index)
                comp[m][d] = sector_percentile(raw, sec, higher_is_better=True)

    pd.DataFrame(weights_rows).to_csv(OUT / "weights_by_year.csv", index=False)
    pd.DataFrame(struct_rows).to_csv(OUT / "structure_by_year.csv", index=False)
    if skipped:
        print("skipped years:", skipped)

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

    struct_df = pd.DataFrame(struct_rows)
    avg_struct = struct_df.groupby("method")[
        ["eff_n_bets", "max_eff_exposure", "min_eff_exposure", "n_active_parents"]
    ].mean().round(3)
    avg_r2 = struct_df.groupby("method")[[f"r2_{p}" for p in PARENTS]].mean().round(4)
    avg_w = struct_df.groupby("method")[[f"w_{p}" for p in PARENTS]].mean().round(4)
    avg_eff = struct_df.groupby("method")[[f"eff_{p}" for p in PARENTS]].mean().round(4)

    summary_rows = []
    for m in list(METHODS) + ["A0"]:
        r = results[m]
        row = {"method": m, "ic_blend": r["ic_blend"], "ir_blend": r["ir_blend"],
               "hit_blend": r["hit_blend"], "q5q1_3M": r["q5q1_3M"], "q5q1_6M": r["q5q1_6M"],
               "n_dates": r["n_dates"]}
        if m in avg_struct.index:
            row |= avg_struct.loc[m].to_dict()
        if m in diffs:
            row["ic_diff_vs_A"] = diffs[m]["mean_diff"]
            row["ci90_lo"], row["ci90_hi"] = diffs[m]["ci90"]
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows).set_index("method")
    summary_df.to_csv(OUT / "config_summary.csv")

    out = {"results": results, "paired_vs_A": diffs, "skipped_years": skipped,
           "avg_r2_by_parent": avg_r2.to_dict(orient="index"),
           "avg_weight_by_parent": avg_w.to_dict(orient="index"),
           "avg_eff_exposure_by_parent": avg_eff.to_dict(orient="index")}
    with open(OUT / "results.json", "w") as f:
        json.dump(out, f, indent=2)

    print("\n=== config summary (pooled OOS 2019-2026) ===")
    print(summary_df.to_string())
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
