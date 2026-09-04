"""10-configuration walk-forward comparison (2026-09-01), scored on the
portfolio-agnostic factor scorecard (scripts/factor_scorecard.py), not a
costed portfolio backtest — user explicitly wants to decouple "is this a
good ranking" from "what top_pct/weighting/sector construction was E tuned
against" (scripts/weight_config_analysis.py's sensitivity grid showed the
E-vs-EQEFF portfolio ranking flips with construction choice).

Same protocol as scripts/full_pit_backtest_eqeff_2020.py: weights (and
subfactor selection) re-derived FROM SCRATCH every 6 months on a trailing 5y
window, 2020-01 through 2026-01 derivation points, no reuse of production
weights. 10 new configurations spanning the space between EQEFF (pure
correlation-parity, no IC) and A (pure IC/IR blend, no correlation
adjustment), using both constructions the user proposed:

Structural solves  w \\propto (\\lambda I + (1-\\lambda) C)^{-1} \\cdot target, cap 0.25:
  D0        target=IC, \\lambda=0   (pure C^{-1}\\cdotIC, no shrinkage)
  D25       target=IC, \\lambda=0.25
  D75       target=IC, \\lambda=0.75
  D0_WINSOR target=IC winsorized to [0, 2x cross-parent median], \\lambda=0
  IREFF     target=IR instead of IC (consistency-weighted, not raw strength)

Linear blends of the two solved vectors (EQEFF, A):
  BLEND25   0.25*EQEFF + 0.75*A
  BLEND50   0.50*EQEFF + 0.50*A
  BLEND75   0.75*EQEFF + 0.25*A

Sequential constructions:
  EQ_IC_FLOOR    equal(1/8) -> tilt by IC/mean(IC) -> cap+floor squeeze
                 (same squeeze as production E, applied to an IC-tilted
                 equal-weight start instead of A's derivation)
  EQEFF_IC_TILT  EQEFF -> multiply by IC-rank tilt (top parent ~1.3x,
                 bottom ~0.7x) -> cap-trim effective exposure back to 0.25

A, E, EQEFF kept as reference columns.

Outputs to output/crowding/weight_config_study/:
  comp_10configs.pkl        — cached composite scores per method/date
  pit_parent_weights_10configs.csv
  factor_scorecard_10configs.csv

Usage: python scripts/full_pit_backtest_10configs.py
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from backtesting import data_loader as dl
from data.db import get_db
from factors.utils import sector_percentile
from research.panel import ScorePanel
from research.parent_selection import run_selection
from research.subfactor_expansion.panel import load_cached_panel
from research.analyst_deep_dive.common import load_price_matrix, panel_forward_returns, spearman_ic
from scripts.crowding_diagnostics import parent_score, normalize
from scripts.incremental_weight_study import PANEL_PKL, water_fill, combined_scores, CAP
from scripts.full_pit_backtest_eqeff import cap_trim, WINDOW_YEARS, APPLY_MONTHS, OOS_END, H_MONTHS
from scripts.weight_config_study import cap_and_floor, eff_exposure_metric, FLOOR_EFF, FLOOR_W
from scripts.factor_scorecard import scorecard

OUT = REPO / "output" / "crowding" / "weight_config_study"
OUT.mkdir(parents=True, exist_ok=True)

DERIV_POINTS = [f"{y}-{m:02d}-01" for y in range(2020, 2027) for m in (1, 7)
                if not (y == 2026 and m == 7)]

BASELINES = ("A", "E", "EQ", "EQEFF")
NEW_10 = ("D0", "D25", "D75", "D0_WINSOR", "IREFF",
          "BLEND25", "BLEND50", "BLEND75", "EQ_IC_FLOOR", "EQEFF_IC_TILT")
ALL_METHODS = BASELINES + NEW_10


def solve_c_inv_target(target: pd.Series, C: pd.DataFrame, cap: float = CAP) -> pd.Series:
    Cinv = np.linalg.pinv(C.values)
    raw = np.clip(Cinv @ target.values, 0.0, None)
    if raw.sum() <= 1e-9:
        raw = np.ones(len(target))
    w = pd.Series(raw, index=target.index)
    return cap_trim(w / w.sum(), C, cap)


def method_D_lam(ic: pd.Series, C: pd.DataFrame, lam: float, cap: float = CAP) -> pd.Series:
    icv = ic.clip(lower=0.0).values
    M = lam * np.eye(len(ic)) + (1 - lam) * C.values
    w = np.clip(np.linalg.solve(M, icv), 0.0, None)
    return pd.Series(water_fill(dict(zip(ic.index, w)), cap))[ic.index]


def method_weights_10(ic: pd.Series, ir: pd.Series, C: pd.DataFrame) -> dict[str, pd.Series]:
    parents = list(ic.index)

    out: dict[str, pd.Series] = {}
    out["A"] = pd.Series(water_fill(combined_scores(ic, ir).to_dict(), CAP))[parents]

    s = combined_scores(ic, ir)
    wb = (s / s.sum()).copy() if s.sum() > 0 else out["A"].copy()
    wb = cap_trim(wb[parents], C)

    we = wb.copy()
    for _ in range(50):
        eff = C.values @ we.values
        over = eff > CAP + 1e-12
        if over.any():
            we[over] = we[over] * (CAP / eff[over])
            we = we / we.sum()
            continue
        under_mask = eff < FLOOR_EFF - 1e-12
        under = we.index[under_mask]
        deficit = (FLOOR_W - we[under]).clip(lower=0.0)
        if deficit.sum() < 1e-6:
            break
        donors = we.index.difference(under)
        pool = we[donors].sum()
        if pool <= 1e-9:
            break
        take = deficit.sum() * (we[donors] / pool)
        we[donors] = we[donors] - take
        we[under] = we[under] + deficit
        we = we.clip(lower=0.0)
        we = we / we.sum()
    out["E"] = we

    out["EQ"] = pd.Series(1.0 / len(parents), index=parents)

    ones = pd.Series(1.0, index=parents)
    out["EQEFF"] = solve_c_inv_target(ones, C)

    # --- structural solves ---
    out["D0"] = method_D_lam(ic, C, lam=0.0)
    out["D25"] = method_D_lam(ic, C, lam=0.25)
    out["D75"] = method_D_lam(ic, C, lam=0.75)

    ic_pos = ic.clip(lower=0.0)
    med = ic_pos.median()
    cap_val = 2 * med if med > 0 else (ic_pos.max() or 1.0)
    ic_winsor = ic_pos.clip(upper=cap_val)
    out["D0_WINSOR"] = solve_c_inv_target(ic_winsor, C)

    out["IREFF"] = solve_c_inv_target(ir.clip(lower=0.0), C)

    # --- linear blends of EQEFF and A ---
    for alpha, name in ((0.25, "BLEND25"), (0.50, "BLEND50"), (0.75, "BLEND75")):
        w = alpha * out["EQEFF"] + (1 - alpha) * out["A"]
        w = w.clip(lower=0.0)
        w = w / w.sum()
        out[name] = cap_trim(w, C)

    # --- sequential: equal -> IC tilt -> cap+floor squeeze ---
    tilt = (ic_pos / ic_pos.mean()).clip(lower=0.3, upper=3.0) if ic_pos.mean() > 0 else pd.Series(1.0, index=parents)
    w1 = (out["EQ"] * tilt)
    w1 = w1 / w1.sum()
    out["EQ_IC_FLOOR"] = pd.Series(
        cap_and_floor(w1.to_dict(), C, eff_exposure_metric, FLOOR_EFF, FLOOR_W))[parents]

    # --- sequential: EQEFF -> IC-rank tilt -> cap-trim back ---
    rank_pct = ic.rank(pct=True)
    tilt2 = 0.7 + 0.6 * rank_pct
    w2 = out["EQEFF"] * tilt2
    w2 = w2.clip(lower=0.0)
    w2 = w2 / w2.sum()
    out["EQEFF_IC_TILT"] = cap_trim(w2, C)

    return {m: out[m][parents] for m in ALL_METHODS}


def derive():
    db = get_db()
    panel = load_cached_panel(PANEL_PKL)
    all_dates = panel.rebal_dates
    print(f"panel {len(all_dates)} dates {all_dates[0]}..{all_dates[-1]}")

    matrix_full = load_price_matrix(start="2015-01-01", end="2026-07-31")
    fwd = panel_forward_returns(matrix_full, all_dates)
    sectors = dl.global_sectors(db)

    comp = {m: {} for m in ALL_METHODS}
    w_rows = []
    for t in DERIV_POINTS:
        t_ts = pd.Timestamp(t)
        w0 = (t_ts - pd.DateOffset(years=WINDOW_YEARS)).strftime("%Y-%m-%d")
        window = [d for d in all_dates if w0 <= d < t]
        apply_end = (t_ts + pd.DateOffset(months=APPLY_MONTHS)).strftime("%Y-%m-%d")
        apply_dates = [d for d in all_dates if t <= d < apply_end and d <= OOS_END]
        if not apply_dates:
            continue

        fwd_pit = {h: {d: fwd[h][d] for d in window if d in fwd[h]
                       and (pd.Timestamp(d) + pd.DateOffset(months=n)) <= t_ts}
                   for h, n in H_MONTHS.items()}

        sp = ScorePanel(rebal_dates=window,
                        scores={d: panel.scores[d] for d in window},
                        parent_keys=[p for p, s in panel.candidates_by_parent.items() if s],
                        sub_by_parent={p: list(s) for p, s in
                                       panel.candidates_by_parent.items() if s},
                        universe=list(panel.universe))
        results = run_selection(sp, fwd_pit)
        sel = {res.parent: res.weights for res in results if res.selected}

        parents = list(sel)
        P_by = {d: pd.DataFrame({p: parent_score(panel.scores[d], sel[p])
                                 for p in parents}) for d in window + apply_dates}
        Pn_by = {d: P_by[d].apply(normalize) for d in P_by}

        blend_rows = {}
        for d in window:
            vals = []
            for h in ("3M", "6M"):
                if d in fwd_pit[h]:
                    ics = {p: spearman_ic(P_by[d][p], fwd_pit[h][d]) for p in parents}
                    vals.append(pd.Series(ics, dtype=float))
            if vals:
                blend_rows[d] = pd.concat(vals, axis=1).mean(axis=1)
        blend = pd.DataFrame(blend_rows).T
        ic_mean = blend.mean()
        ic_ir = blend.mean() / blend.std(ddof=1)
        C = (pd.concat([Pn_by[d].corr(method="pearson") for d in window])
             .groupby(level=0, sort=False).mean().loc[parents, parents])

        wv = method_weights_10(ic_mean[parents], ic_ir[parents], C)
        for m, w in wv.items():
            w_rows.append({"window": t, "method": m,
                           **{p: round(float(w.get(p, 0.0)), 3) for p in parents}})
        print(f"{t}: window {len(window)}d, ic-dates {len(blend)}, "
              f"apply {len(apply_dates)}m, parents {len(parents)}", flush=True)

        for d in apply_dates:
            sec = sectors.reindex(Pn_by[d].index).fillna("Unknown")
            for m, w in wv.items():
                if w.sum() <= 0:
                    continue
                raw = Pn_by[d].mul(w).sum(axis=1) / w.sum()
                comp[m][d] = sector_percentile(raw, sec, higher_is_better=True, min_obs=5)

    pd.DataFrame(w_rows).to_csv(OUT / "pit_parent_weights_10configs.csv", index=False)
    with open(OUT / "comp_10configs.pkl", "wb") as f:
        pickle.dump({"comp": comp, "universe": list(panel.universe)}, f)
    return comp, list(panel.universe)


def main():
    cached = OUT / "comp_10configs.pkl"
    if cached.exists():
        print(f"reusing cached derivation: {cached}")
        with open(cached, "rb") as f:
            d = pickle.load(f)
        comp, universe = d["comp"], d["universe"]
    else:
        comp, universe = derive()

    df = scorecard(comp=comp, universe=universe, methods=list(ALL_METHODS))
    df.to_csv(OUT / "factor_scorecard_10configs.csv")
    print("\n=== 10-config factor scorecard (78mo OOS, 2020-2026) ===\n")
    with pd.option_context("display.width", 220, "display.max_columns", 20):
        print(df.T.to_string())
    print(f"\nwrote {OUT}/factor_scorecard_10configs.csv")


if __name__ == "__main__":
    main()
