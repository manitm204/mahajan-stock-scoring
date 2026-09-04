"""Out-of-sample analytics on a frozen composite: IC, quantiles, holding, long-short.

Everything here consumes a ``scores`` dict (``{date: Series[ticker → composite]}`` — the
frozen composite applied to test-period rebalances) plus the price matrix, so the same
functions serve the pooled cross-split read and each per-split read. Signal-strength reads
(IC, quantile spreads) reuse the existing point-in-time machinery in
:mod:`research.ic` / :mod:`research.quintiles`; the implementable reads (holding period,
long vs long-short) go through :mod:`research.walkforward.portfolio`.

Overlapping multi-month horizons understate return volatility, so 3M/6M/12M IRs and
Sharpes are indicative rather than independent-sample t-stats — flagged in the reports.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from research.forward_returns import HORIZON_MONTHS
from research.ic import period_ic
from . import portfolio as pf

MIN_NAMES = 25
N_QUANTILES = 5


# --------------------------------------------------------------------------- #
# Q1 — does the composite rank future returns?
# --------------------------------------------------------------------------- #
def composite_ic(scores: dict[str, pd.Series],
                 fwd_by_h: dict[str, dict[str, pd.Series]]) -> pd.DataFrame:
    """Per-horizon composite IC summary: mean/median IC, IR, hit rate, t-stat, n."""
    rows: list[dict] = []
    for h, fwd_h in fwd_by_h.items():
        ics: list[float] = []
        for d, fwd in fwd_h.items():
            sc = scores.get(d)
            if sc is None:
                continue
            ic = period_ic(sc, fwd, min_names=20)
            if ic is not None:
                ics.append(ic)
        arr = np.array(ics, dtype=float)
        if arr.size == 0:
            rows.append({"horizon": h, "n_periods": 0})
            continue
        mean, std = float(arr.mean()), float(arr.std(ddof=1)) if arr.size > 1 else np.nan
        rows.append({
            "horizon": h, "n_periods": int(arr.size),
            "mean_ic": mean, "median_ic": float(np.median(arr)),
            "ic_std": std,
            "information_ratio": (mean / std) if std and std > 1e-9 else np.nan,
            "t_stat": (mean / (std / np.sqrt(arr.size))) if std and std > 1e-9 else np.nan,
            "hit_rate": float((arr > 0).mean()),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Q2 — quantile ranking power
# --------------------------------------------------------------------------- #
def _bucket_period_returns(scores: dict[str, pd.Series], fwd_h: dict[str, pd.Series],
                           n: int = N_QUANTILES) -> pd.DataFrame:
    """Per-period mean forward return of each score quantile (Q1=lowest … Qn=highest).

    Buckets are cut on the *rank* of the score (robust to the heavy tie-mass at the
    neutral 50). Returns a frame indexed by rebalance date with columns ``q1..qn``.
    """
    rows: dict[str, np.ndarray] = {}
    for d, fwd in fwd_h.items():
        sc = scores.get(d)
        if sc is None:
            continue
        df = pd.DataFrame({"s": sc, "f": fwd}).dropna()
        if len(df) < MIN_NAMES or df["s"].nunique() < n:
            continue
        try:
            buckets = pd.qcut(df["s"].rank(method="first"), n, labels=False)
        except ValueError:
            continue
        means = df["f"].groupby(buckets).mean().reindex(range(n))
        if means.isna().any():
            continue
        rows[d] = means.to_numpy(dtype=float)
    out = pd.DataFrame(rows).T
    if not out.empty:
        out.columns = [f"q{i + 1}" for i in range(n)]
    return out.sort_index()


def _summarize_series(s: pd.Series, months: int) -> dict:
    """Level/annualised stats for a horizon-``months`` per-period return series."""
    r = s.dropna()
    if r.empty:
        return {k: np.nan for k in ("avg", "median", "annualized", "volatility",
                                    "sharpe", "hit_rate")}
    ppy = 12.0 / months
    mean, std = float(r.mean()), float(r.std(ddof=1)) if len(r) > 1 else np.nan
    return {
        "avg": mean,
        "median": float(r.median()),
        "annualized": float((1.0 + mean) ** ppy - 1.0),
        "volatility": float(std * np.sqrt(ppy)) if std == std else np.nan,
        "sharpe": float(mean / std * np.sqrt(ppy)) if std and std > 0 else np.nan,
        "hit_rate": float((r > 0).mean()),
    }


def quantile_analysis(scores: dict[str, pd.Series],
                      fwd_by_h: dict[str, dict[str, pd.Series]],
                      n: int = N_QUANTILES) -> dict[str, dict]:
    """For each horizon: per-bucket summary table, Q5−Q1 spread stats, monotonicity.

    Returns ``{horizon: {"table": DataFrame(q1..qn × stats), "spread": dict,
    "monotonic_rate": float, "profile_spearman": float, "n_periods": int}}``.
    ``monotonic_rate`` is the share of periods where returns rise strictly Q1<…<Qn;
    ``profile_spearman`` is the rank correlation of the averaged bucket profile vs order.
    """
    out: dict[str, dict] = {}
    for h, fwd_h in fwd_by_h.items():
        months = HORIZON_MONTHS[h]
        bp = _bucket_period_returns(scores, fwd_h, n)
        if bp.empty:
            out[h] = {"table": pd.DataFrame(), "spread": {}, "monotonic_rate": np.nan,
                      "profile_spearman": np.nan, "n_periods": 0}
            continue
        table = pd.DataFrame({col: _summarize_series(bp[col], months)
                              for col in bp.columns}).T
        spread_series = bp[f"q{n}"] - bp["q1"]
        spread = _summarize_series(spread_series, months)
        mono_rate = float((bp.apply(lambda row: bool(np.all(np.diff(row.values) > 0)),
                                    axis=1)).mean())
        profile = bp.mean(axis=0).to_numpy()
        prof_sp = float(pd.Series(np.arange(n)).corr(pd.Series(profile), method="spearman")) \
            if np.std(profile) > 1e-12 else np.nan
        out[h] = {"table": table, "spread": spread, "monotonic_rate": mono_rate,
                  "profile_spearman": prof_sp, "n_periods": int(len(bp))}
    return out


# --------------------------------------------------------------------------- #
# Q3 — portfolio construction sweep
# --------------------------------------------------------------------------- #
def portfolio_sweep(scores: dict[str, pd.Series], price_matrix: pd.DataFrame,
                    sectors: pd.Series, *,
                    top_pcts=(0.10, 0.20, 0.30),
                    modes=("equal", "sector_neutral")) -> pd.DataFrame:
    """Top-10/20/30 % × {equal, sector-neutral}, monthly hold → one metrics row each."""
    rows: list[dict] = []
    for tp in top_pcts:
        for mode in modes:
            res = pf.simulate(scores, price_matrix, sectors, top_pct=tp, mode=mode,
                              hold_months=1)
            rows.append({"top_pct": tp, "mode": mode, **res.metrics})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Q4 — holding period
# --------------------------------------------------------------------------- #
def holding_period_sweep(scores: dict[str, pd.Series], price_matrix: pd.DataFrame,
                         sectors: pd.Series, *,
                         holds=(1, 3, 6, 12), top_pct: float = 0.20,
                         mode: str = "equal") -> pd.DataFrame:
    """Top-``top_pct`` long portfolio at 1/3/6/12-month non-overlapping holds."""
    rows: list[dict] = []
    for h in holds:
        res = pf.simulate(scores, price_matrix, sectors, top_pct=top_pct, mode=mode,
                          hold_months=h)
        rows.append({"hold_months": h, **res.metrics})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Q5 — long-only vs long-short
# --------------------------------------------------------------------------- #
def long_vs_longshort(scores: dict[str, pd.Series], price_matrix: pd.DataFrame,
                      sectors: pd.Series, *, top_pct: float = 0.20,
                      mode: str = "equal", hold_months: int = 1
                      ) -> tuple[pf.SimResult, pf.SimResult]:
    """(long-only top-``top_pct``, long-short top−bottom) simulations."""
    lo = pf.simulate(scores, price_matrix, sectors, top_pct=top_pct, mode=mode,
                     short=False, hold_months=hold_months, label="long_only")
    ls = pf.simulate(scores, price_matrix, sectors, top_pct=top_pct, mode=mode,
                     short=True, hold_months=hold_months, label="long_short")
    return lo, ls


# --------------------------------------------------------------------------- #
# Section 3 — parent-factor out-of-sample analysis
# --------------------------------------------------------------------------- #
def parent_oos_ic(parent_scores: dict[str, dict[str, pd.Series]],
                  fwd_by_h: dict[str, dict[str, pd.Series]]) -> pd.DataFrame:
    """Per-parent, per-horizon OOS IC / IR / hit rate.

    ``parent_scores`` is ``{parent: {date: Series[ticker→parent score]}}`` pooled across
    every split's test period (each split contributes its own frozen sub-weighted parent
    scores). One row per (parent, horizon)."""
    rows: list[dict] = []
    for parent, by_date in parent_scores.items():
        for h, fwd_h in fwd_by_h.items():
            ics = [period_ic(by_date[d], fwd, min_names=20)
                   for d, fwd in fwd_h.items() if d in by_date]
            ics = [x for x in ics if x is not None]
            arr = np.array(ics, dtype=float)
            if arr.size == 0:
                continue
            std = float(arr.std(ddof=1)) if arr.size > 1 else np.nan
            rows.append({
                "parent": parent, "horizon": h, "n_periods": int(arr.size),
                "mean_ic": float(arr.mean()), "median_ic": float(np.median(arr)),
                "information_ratio": (float(arr.mean()) / std) if std and std > 1e-9 else np.nan,
                "hit_rate": float((arr > 0).mean()),
            })
    return pd.DataFrame(rows)


def parent_correlation(parent_scores: dict[str, dict[str, pd.Series]]) -> pd.DataFrame:
    """Average cross-sectional Spearman correlation between parent scores.

    For each pooled test date the parents' scores are rank-correlated across names; the
    mean over dates gives a parent×parent redundancy matrix (how much the factors overlap
    out of sample)."""
    parents = list(parent_scores)
    dates = sorted({d for by in parent_scores.values() for d in by})
    mats: list[pd.DataFrame] = []
    for d in dates:
        cols = {p: parent_scores[p][d] for p in parents if d in parent_scores[p]}
        if len(cols) < 2:
            continue
        frame = pd.DataFrame(cols).dropna()
        if len(frame) < MIN_NAMES:
            continue
        mats.append(frame.corr(method="spearman"))
    if not mats:
        return pd.DataFrame(index=parents, columns=parents, dtype=float)
    return sum(mats) / len(mats)
