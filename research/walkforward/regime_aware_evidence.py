"""Monthly point-in-time evidence cache + as-of(cutoff) slicing.

One row per (sub_factor, rebalance date) -- unlike factor_persistence.py, which
collapses to one row per (sub_factor, calendar year), this keeps every month's
raw observation so a monthly walk-forward can ask "what did we know as of an
arbitrary cutoff" via expanding/rolling slices instead of only per-year reads.
See docs/superpowers/specs/2026-07-11-hierarchical-regime-aware-factor-model-design.md
Section 1.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from research import compute_forward_returns
from research.ic import period_ic
from research.panel import ScorePanel
from research.quintiles import quintile_profile
from research.walkforward.regime_probability import (
    REGIME_ORDER, SHRINKAGE_K, effective_regime_stats, expected_ic,
)
from research.walkforward.splits import DATA_START
from research.walkforward.vix_regime_study import _vix_spot

RECENT_MONTHS = 24
MIN_NAMES = 20
SPREAD_ANNUALIZATION_6M = 2.0  # 12 months / 6-month horizon
STD_EPSILON = 1e-9             # guards persistence_ir against a near-zero-std blowup

# REGIME_ORDER labels are "<Name> (<range>)" -- this maps each to its lowercase
# key used in column names (regime_ic_low, regime_n_eff_medium, ...). Defined once
# so the empty-window column schema and the populated-row builder can't drift apart.
REGIME_KEYS = {r: r.split(" ")[0].lower() for r in REGIME_ORDER}

# Single source of truth for as_of()'s output schema: the empty-window fallback
# and the populated row-builder both key off these two tuples, so a field added
# to one can't silently go missing from the other.
_BASE_COLUMNS = (
    "sub_factor", "parent", "long_run_mean_ic", "long_run_std_ic",
    "long_run_hit_rate", "long_run_spread_ann", "recent_24m_ic",
    "pct_positive_years", "persistence_ir", "spread_consistency",
    "coverage", "n_months", "n_years", "regime_dependence", "expected_ic",
)
_REGIME_COLUMNS = tuple(
    f"regime_{field}_{key}" for key in REGIME_KEYS.values() for field in ("ic", "n_eff")
)
_ALL_COLUMNS = _BASE_COLUMNS + _REGIME_COLUMNS


def build_monthly_cache(
    panel: ScorePanel, matrix: pd.DataFrame, vix: pd.Series, *, min_names: int = MIN_NAMES,
) -> pd.DataFrame:
    """One row per (sub_factor, date): ic_3M, ic_6M, spread_3M_raw, spread_6M_raw,
    coverage (fraction of universe scored that date), vix_level (spot VIX at that date).

    Spreads are the *raw* one-period Q5-Q1 gap (not annualised) -- as_of() annualises
    after aggregating, matching the convention factor_persistence.py already uses
    (annualise the mean, not each observation).
    """
    fwd_by_h = compute_forward_returns(matrix, panel.rebal_dates, {"3M": 3, "6M": 6})
    n_uni = len(panel.universe)
    rows: list[dict] = []
    for sub in panel.all_subs:
        parent = panel.parent_of(sub)
        for d in panel.rebal_dates:
            frame = panel.scores.get(d)
            if frame is None or sub not in frame.columns:
                continue
            row: dict = {
                "sub_factor": sub, "parent": parent, "date": d,
                "vix_level": _vix_spot(vix, d),
                "coverage": float(frame[sub].count()) / n_uni if n_uni else float("nan"),
            }
            for h in ("3M", "6M"):
                fwd = fwd_by_h.get(h, {}).get(d)
                if fwd is None:
                    row[f"ic_{h}"] = float("nan")
                    row[f"spread_{h}_raw"] = float("nan")
                    continue
                ic = period_ic(frame[sub], fwd, min_names=min_names)
                row[f"ic_{h}"] = ic if ic is not None else float("nan")
                prof = quintile_profile(frame[sub], fwd, min_names=min_names)
                row[f"spread_{h}_raw"] = float(prof[-1] - prof[0]) if prof is not None else float("nan")
            rows.append(row)
    return pd.DataFrame(rows)


def as_of(
    cache: pd.DataFrame, cutoff: str, vix: pd.Series, *,
    panel_inception: str = DATA_START, recent_months: int = RECENT_MONTHS,
    k: float = SHRINKAGE_K,
) -> pd.DataFrame:
    """Everything Section 1/2 of the spec need, one row per sub_factor, PIT-safe.

    Only cache rows with panel_inception <= date <= cutoff are used. Long-run stats
    expand from panel_inception; recent_24m_ic is a trailing window (shorter than 24
    months, degrading to ~long_run, in the first two years -- see spec Section 1).
    """
    hist = cache[(cache["date"] <= cutoff) & (cache["date"] >= panel_inception)].copy()
    if hist.empty:
        return pd.DataFrame(columns=_ALL_COLUMNS)

    hist["mean_ic"] = hist[["ic_3M", "ic_6M"]].mean(axis=1)
    hist["year"] = hist["date"].str.slice(0, 4)
    recent_start = (pd.Timestamp(cutoff) - pd.DateOffset(months=recent_months)).date().isoformat()
    vix_now = _vix_spot(vix, cutoff)

    rows: list[dict] = []
    for sub, g in hist.groupby("sub_factor"):
        g = g.sort_values("date")
        mean_ic_series = g.set_index("date")["mean_ic"].dropna()
        long_run_mean = float(mean_ic_series.mean()) if not mean_ic_series.empty else float("nan")
        long_run_std = (float(mean_ic_series.std(ddof=1))
                        if len(mean_ic_series) > 1 else float("nan"))
        long_run_hit = (float((mean_ic_series > 0).mean())
                        if not mean_ic_series.empty else float("nan"))

        recent = mean_ic_series[mean_ic_series.index >= recent_start]
        recent_ic = float(recent.mean()) if not recent.empty else float("nan")

        spread6 = g.set_index("date")["spread_6M_raw"].dropna()
        long_run_spread_ann = (float(spread6.mean() * SPREAD_ANNUALIZATION_6M)
                               if not spread6.empty else float("nan"))

        annual = g.groupby("year")["mean_ic"].mean().dropna()
        pct_pos_years = float((annual > 0).mean()) if not annual.empty else float("nan")
        pers_ir = (float(annual.mean() / annual.std(ddof=1))
                  if len(annual) > 1 and annual.std(ddof=1) > STD_EPSILON else float("nan"))
        annual_spread = (g.assign(spread_ann=g["spread_6M_raw"] * SPREAD_ANNUALIZATION_6M)
                         .groupby("year")["spread_ann"].mean().dropna())
        spread_consistency = (
            float((np.sign(annual_spread) == np.sign(long_run_spread_ann)).mean())
            if not annual_spread.empty and pd.notna(long_run_spread_ann)
            else float("nan"))

        coverage = float(g["coverage"].mean()) if not g["coverage"].empty else float("nan")
        n_months = int(len(mean_ic_series))
        n_years = int(g["year"].nunique())

        vix_by_date = g.set_index("date")["vix_level"]
        regime_stats = effective_regime_stats(vix_by_date, mean_ic_series)
        raw_ics = np.array([regime_stats[r]["regime_ic"] for r in REGIME_ORDER], dtype=float)
        regime_dep = float(np.nanstd(raw_ics)) if np.isfinite(raw_ics).any() else float("nan")

        exp_ic = expected_ic(long_run_mean, recent_ic, vix_now, regime_stats, k=k)

        row = dict(zip(_BASE_COLUMNS, (
            sub, g["parent"].iloc[0], long_run_mean, long_run_std,
            long_run_hit, long_run_spread_ann, recent_ic, pct_pos_years,
            pers_ir, spread_consistency, coverage, n_months, n_years,
            regime_dep, exp_ic,
        )))
        for r in REGIME_ORDER:
            key = REGIME_KEYS[r]
            row[f"regime_ic_{key}"] = regime_stats[r]["regime_ic"]
            row[f"regime_n_eff_{key}"] = regime_stats[r]["n_eff"]
        rows.append(row)
    return pd.DataFrame(rows)
