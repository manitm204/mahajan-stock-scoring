"""Uncertainty & fragility diagnostics for a candidate's period-return series.

Complements engine.simulate_config's point estimates with:
- stationary block bootstrap CIs for Sharpe and CAPM alpha (paired resampling of
  (portfolio, benchmark) so beta structure survives resampling);
- drop-best fragility (metrics after removing the best calendar year / best k months);
- rolling alpha vs a benchmark.

All functions take per-period return Series indexed by date-strings (the engine's
native format) and are pure — no data bundle needed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from research.ablation.engine import _alpha_on
from research.walkforward.portfolio import _cagr


def _block_bootstrap_idx(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    """Circular block bootstrap index vector of length n."""
    starts = rng.integers(0, n, size=int(np.ceil(n / block)))
    idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
    return idx % n


def bootstrap_ci(p: pd.Series, b: pd.Series, ppy: float, n_boot: int = 2000,
                 block: int = 6, seed: int = 7) -> dict:
    """Bootstrap CIs for Sharpe, CAGR, and alpha-vs-benchmark of ``p``.

    Paired circular block bootstrap (default 6-period blocks) preserves both
    autocorrelation and the portfolio/benchmark dependence.
    """
    df = pd.concat([p.rename("p"), b.rename("b")], axis=1).dropna()
    n = len(df)
    rng = np.random.default_rng(seed)
    sh, cg, al = [], [], []
    pv, bv = df["p"].values, df["b"].values
    for _ in range(n_boot):
        idx = _block_bootstrap_idx(n, block, rng)
        ps, bs = pd.Series(pv[idx]), pd.Series(bv[idx])
        s = ps.std(ddof=1)
        sh.append(float(ps.mean() / s * np.sqrt(ppy)) if s > 0 else np.nan)
        cg.append(_cagr(ps, ppy))
        al.append(_alpha_on(ps, bs, ppy))
    q = lambda v, lo, hi: (float(np.nanpercentile(v, lo)),
                           float(np.nanpercentile(v, hi)))
    return {
        "sharpe_ci90": q(sh, 5, 95), "cagr_ci90": q(cg, 5, 95),
        "alpha_ci90": q(al, 5, 95),
        "p_alpha_neg": float(np.nanmean(np.array(al) < 0.0)),
        "p_sharpe_below_bench": float(np.nanmean(
            np.array(sh) < (df["b"].mean() / df["b"].std(ddof=1) * np.sqrt(ppy)))),
    }


def drop_best(p: pd.Series, b: pd.Series, ppy: float) -> dict:
    """Fragility: excess CAGR vs benchmark after removing the best calendar year,
    and after removing the k best months (k = 3), from *both* series."""
    df = pd.concat([p.rename("p"), b.rename("b")], axis=1).dropna()
    years = df.index.astype(str).str[:4]
    ex_by_year = {y: _cagr(df.loc[years == y, "p"], ppy)
                  - _cagr(df.loc[years == y, "b"], ppy)
                  for y in sorted(set(years))}
    best_year = max(ex_by_year, key=ex_by_year.get)
    keep = years != best_year
    out = {
        "best_year": best_year,
        "excess_full": _cagr(df["p"], ppy) - _cagr(df["b"], ppy),
        "excess_wo_best_year": _cagr(df.loc[keep, "p"], ppy)
        - _cagr(df.loc[keep, "b"], ppy),
        "excess_by_year": ex_by_year,
    }
    ex = df["p"] - df["b"]
    worst_k = ex.nlargest(3).index
    keep3 = ~df.index.isin(worst_k)
    out["excess_wo_best3_periods"] = (_cagr(df.loc[keep3, "p"], ppy)
                                      - _cagr(df.loc[keep3, "b"], ppy))
    return out


def rolling_alpha(p: pd.Series, b: pd.Series, ppy: float,
                  window: int = 36) -> pd.Series:
    """Rolling annualized CAPM alpha of ``p`` on ``b`` over ``window`` periods."""
    df = pd.concat([p.rename("p"), b.rename("b")], axis=1).dropna()
    vals = {}
    for i in range(window, len(df) + 1):
        w = df.iloc[i - window:i]
        vals[df.index[i - 1]] = _alpha_on(w["p"], w["b"], ppy)
    return pd.Series(vals, dtype=float)
