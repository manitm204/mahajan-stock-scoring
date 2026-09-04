"""Fully point-in-time walk-forward portfolio backtest: A vs E vs EQ vs EQEFF vs SPY/QQQ.

Same protocol as scripts/full_pit_backtest_e.py (identical to
scripts/full_pit_backtest.py): at each 6-month derivation point, weights are
derived FRESH from only the trailing 5-year window (no reuse of current
production weights), then frozen and applied forward for the next 6 months,
then re-derived. This is the discriminating test — scripts/weight_config_study.py
already showed EQ and (especially) EQEFF decisively beat every IC-informed
method on walk-forward IC, but method D had that exact profile too (best IC,
worst portfolio Sharpe/CAGR) and lost the portfolio test, so IC alone doesn't
settle whether EQ/EQEFF should ship.

  A      incumbent IC/IR-blend, nominal cap 0.25 (reference baseline)
  E      current production: cap + diversifier floor
  EQ     equal nominal weight (1/8 each), no cap/floor needed
  EQEFF  equalizes effective exposure C·w across all parents (w ∝ pinv(C)·1,
         clipped ≥0, cap 0.25) — no IC information used at all

Usage: python scripts/full_pit_backtest_eqeff.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from backtesting import data_loader as dl
from data.db import get_db
from factors.utils import sector_percentile
from research.ablation.data import AblationData, _pit_caps
from research.ablation.engine import AblationConfig, simulate_config, benchmark_row
from research.forward_returns import realize_delistings
from research.panel import ScorePanel
from research.parent_selection import run_selection
from research.subfactor_expansion.panel import load_cached_panel
from research.analyst_deep_dive.common import load_price_matrix, panel_forward_returns, spearman_ic
from scripts.crowding_diagnostics import parent_score, normalize
from scripts.incremental_weight_study import PANEL_PKL, water_fill, combined_scores, CAP

OUT = REPO / "output" / "crowding" / "weight_config_study"
OUT.mkdir(parents=True, exist_ok=True)

FLOOR_EFF = 0.05
FLOOR_W = 0.08

H_MONTHS = {"3M": 3, "6M": 6}
DERIV_POINTS = [f"{y}-{m:02d}-01" for y in range(2021, 2027) for m in (1, 7)
                if not (y == 2026 and m == 7)]
WINDOW_YEARS = 5
APPLY_MONTHS = 6
OOS_END = "2026-06-30"


def cap_trim(w: pd.Series, C: pd.DataFrame, cap: float = CAP, iters: int = 50) -> pd.Series:
    w = w.copy()
    for _ in range(iters):
        eff = C.values @ w.values
        if eff.max() <= cap + 0.005:
            break
        over = eff > cap + 1e-12
        w[over] = w[over] * (cap / eff[over])
        w = w / w.sum()
    return w


def method_weights(ic: pd.Series, ir: pd.Series, C: pd.DataFrame) -> dict[str, pd.Series]:
    parents = list(ic.index)
    out = {}
    out["A"] = pd.Series(water_fill(combined_scores(ic, ir).to_dict(), CAP))[parents]

    s = combined_scores(ic, ir)
    wb = (s / s.sum()).copy() if s.sum() > 0 else out["A"].copy()
    wb = cap_trim(wb[parents], C)

    # E: B's cap-trim to convergence, then float diversifiers.
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

    # EQ: flat 1/n, no IC/correlation information used.
    out["EQ"] = pd.Series(1.0 / len(parents), index=parents)

    # EQEFF: equalize effective exposure C·w (risk-parity on correlation).
    ones = np.ones(len(parents))
    raw = np.linalg.pinv(C.values) @ ones
    raw = np.clip(raw, 0.0, None)
    if raw.sum() <= 1e-9:
        raw = ones
    weq = pd.Series(raw, index=parents)
    out["EQEFF"] = cap_trim(weq / weq.sum(), C)

    return out


def main():
    db = get_db()
    panel = load_cached_panel(PANEL_PKL)
    all_dates = panel.rebal_dates
    print(f"panel {len(all_dates)} dates {all_dates[0]}..{all_dates[-1]}")

    matrix_full = load_price_matrix(start="2015-01-01", end="2026-07-31")
    fwd = panel_forward_returns(matrix_full, all_dates)
    sectors = dl.global_sectors(db)

    METHODS = ("A", "E", "EQ", "EQEFF")
    comp = {m: {} for m in METHODS}
    sel_rows, w_rows = [], []
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
        for res in results:
            sel_rows.append({"window": t, "parent": res.parent,
                             "formula": res.formula, "flag": res.signal_flag,
                             "stop": res.stop_reason})

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

        wv = method_weights(ic_mean[parents], ic_ir[parents], C)
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

    pd.DataFrame(sel_rows).to_csv(OUT / "pit_selections_eqeff.csv", index=False)
    pd.DataFrame(w_rows).to_csv(OUT / "pit_parent_weights_eqeff.csv", index=False)

    oos_dates = sorted(comp["A"])
    matrix = realize_delistings(
        dl.load_price_matrix(db, panel.universe, "2019-01-01", "2026-07-31"))
    rebal_dates = [d for d in oos_dates if d in matrix.index]
    caps = _pit_caps(db, matrix, rebal_dates)
    data = AblationData(matrix=matrix, sectors=sectors, run=None,
                        rebal_dates=rebal_dates, caps=caps,
                        vol=pd.DataFrame(index=rebal_dates), vix=pd.Series(dtype=float),
                        parent_ranks={}, panel=None)

    rows, curves = [], {}
    for m in METHODS:
        cfg = AblationConfig(name=f"pit_{m}", top_pct=0.25, weighting="cap5",
                             sector="cap_match", vix_tilt=False, hold_months=1)
        row = simulate_config(data, cfg, scores=comp[m], keep_series=True)
        curves[f"pit_{m}"] = (1.0 + row.pop("_returns")).cumprod()
        spy_r, qqq_r = row.pop("_spy"), row.pop("_qqq")
        row.pop("_gross", None)
        if "SPY" not in curves:
            curves["SPY"] = (1.0 + spy_r.dropna()).cumprod()
            curves["QQQ"] = (1.0 + qqq_r.dropna()).cumprod()
        rows.append(row)
    for b in ("SPY", "QQQ"):
        rows.append(benchmark_row(data, b))

    cols = ["config", "net_cagr", "sharpe", "sortino", "calmar", "max_dd",
            "beta", "alpha", "alpha_t", "ex_spy", "ex_qqq", "ir",
            "turnover", "avg_names", "eff_n"]
    out = pd.DataFrame(rows)
    out = out[[c for c in cols if c in out.columns]]
    out.to_csv(OUT / "pit_backtest_eqeff.csv", index=False)
    pd.DataFrame(curves).to_csv(OUT / "pit_equity_eqeff.csv")
    with pd.option_context("display.width", 200, "display.float_format",
                           lambda v: f"{v:.3f}"):
        print(out.to_string(index=False))
    print(f"\nwrote {OUT}/pit_backtest_eqeff.csv + pit_equity_eqeff.csv "
          f"+ pit_selections_eqeff.csv + pit_parent_weights_eqeff.csv")


if __name__ == "__main__":
    main()
