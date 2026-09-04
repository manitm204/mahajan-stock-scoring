"""Forward returns, IC / quintile statistics and performance metrics — own
implementations following the repo's measurement conventions (Spearman IC with
a 20-name floor, rank-quintile Q5-Q1 spread, monthly annualisation, CAPM
alpha/beta and information ratio versus the benchmark).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

HORIZON_MONTHS: dict[str, int] = {"1M": 1, "3M": 3, "6M": 6}
MAX_GAP_DAYS = 25
MIN_NAMES = 20
N_QUANTILES = 5


# --------------------------------------------------------------------------- #
# Forward returns
# --------------------------------------------------------------------------- #
def forward_returns(matrix: pd.DataFrame, dates: list[str],
                    horizons: dict[str, int] | None = None,
                    ) -> dict[str, dict[str, pd.Series]]:
    """``{horizon: {start_date: per-ticker forward return}}``.

    A window whose end would fall past the last available price is dropped —
    no look-ahead, just missing. Ends snap to the nearest trading date within
    ``MAX_GAP_DAYS``.
    """
    horizons = horizons or HORIZON_MONTHS
    index_ts = pd.DatetimeIndex(pd.to_datetime(matrix.index))
    label_by_ts = {ts: lbl for ts, lbl in zip(index_ts, matrix.index)}
    out: dict[str, dict[str, pd.Series]] = {h: {} for h in horizons}
    for d in dates:
        if d not in matrix.index:
            continue
        start_ts = pd.Timestamp(d)
        start_px = matrix.loc[d]
        for label, months in horizons.items():
            target = start_ts + pd.DateOffset(months=months)
            pos = index_ts.searchsorted(target)
            cands = [index_ts[i] for i in (pos, pos - 1)
                     if 0 <= i < len(index_ts)]
            end_ts = min(cands, key=lambda t: abs((t - target).days), default=None)
            if (end_ts is None or end_ts <= start_ts
                    or abs((end_ts - target).days) > MAX_GAP_DAYS):
                continue
            fwd = (matrix.loc[label_by_ts[end_ts]] / start_px) - 1.0
            out[label][d] = fwd.dropna()
    return out


# --------------------------------------------------------------------------- #
# Cross-sectional statistics
# --------------------------------------------------------------------------- #
def spearman_ic(scores: pd.Series, fwd: pd.Series,
                min_names: int = MIN_NAMES) -> float | None:
    df = pd.DataFrame({"s": scores, "f": fwd}).dropna()
    if len(df) < min_names or df["s"].nunique() < 2:
        return None
    ic = df["s"].corr(df["f"], method="spearman")
    return None if pd.isna(ic) else float(ic)


def quintile_spread(scores: pd.Series, fwd: pd.Series,
                    min_names: int = MIN_NAMES) -> float | None:
    """Q5-Q1: mean forward return of the top score quintile minus the bottom.

    Buckets cut on the score *rank* (robust to the tie-mass at the neutral 50).
    """
    df = pd.DataFrame({"s": scores, "f": fwd}).dropna()
    if len(df) < min_names:
        return None
    ranks = df["s"].rank(method="first")
    q = np.ceil(ranks / len(df) * N_QUANTILES).clip(1, N_QUANTILES)
    top, bot = df["f"][q == N_QUANTILES], df["f"][q == 1]
    if top.empty or bot.empty:
        return None
    return float(top.mean() - bot.mean())


# --------------------------------------------------------------------------- #
# Portfolio performance
# --------------------------------------------------------------------------- #
def _cagr(r: pd.Series, ppy: float) -> float:
    r = r.dropna()
    if r.empty:
        return float("nan")
    eq = float((1.0 + r).prod())
    years = len(r) / ppy
    return eq ** (1.0 / years) - 1.0 if years > 0 and eq > 0 else float("nan")


def max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return float("nan")
    return float((equity / equity.cummax() - 1.0).min())


def perf_metrics(returns: pd.Series, turnover: pd.Series | None = None,
                 ppy: float = 12.0) -> dict:
    """CAGR / total return / Sharpe / Sortino / vol / max-DD / hit / turnover."""
    r = returns.dropna()
    out: dict = {"n_periods": int(len(r))}
    if r.empty:
        return out
    equity = (1.0 + r).cumprod()
    std = float(r.std(ddof=1)) if len(r) > 1 else float("nan")
    downside = r[r < 0]
    dd = float(np.sqrt((downside ** 2).mean())) if not downside.empty else float("nan")
    out.update({
        "total_return": float(equity.iloc[-1] - 1.0),
        "cagr": _cagr(r, ppy),
        "sharpe": float(r.mean() / std * np.sqrt(ppy)) if std and std > 0 else float("nan"),
        "sortino": float(r.mean() / dd * np.sqrt(ppy)) if dd and dd > 0 else float("nan"),
        "ann_vol": std * np.sqrt(ppy) if std == std else float("nan"),
        "max_drawdown": max_drawdown(equity),
        "hit_rate": float((r > 0).mean()),
        "avg_turnover": (float(turnover.dropna().mean())
                         if turnover is not None and not turnover.dropna().empty
                         else float("nan")),
    })
    return out


def benchmark_stats(returns: pd.Series, bench: pd.Series,
                    ppy: float = 12.0) -> dict:
    """Excess CAGR, CAPM alpha/beta, information ratio, tracking error and the
    worst drawdown of the portfolio/benchmark equity ratio."""
    df = pd.concat([returns.rename("p"), bench.rename("b")], axis=1).dropna()
    out = {k: float("nan") for k in
           ("bench_cagr", "excess_cagr", "alpha", "beta", "info_ratio",
            "tracking_error", "rel_max_drawdown")}
    if df.empty:
        return out
    p, b = df["p"], df["b"]
    out["bench_cagr"] = _cagr(b, ppy)
    pc = _cagr(p, ppy)
    out["excess_cagr"] = pc - out["bench_cagr"] if pc == pc else float("nan")
    bvar = float(((b - b.mean()) ** 2).sum())
    if bvar > 0 and len(b) > 1:
        beta = float(((p - p.mean()) * (b - b.mean())).sum() / bvar)
        out["beta"] = beta
        out["alpha"] = float((p.mean() - beta * b.mean()) * ppy)
    active = p - b
    te = float(active.std(ddof=1)) if len(active) > 1 else float("nan")
    out["tracking_error"] = te * np.sqrt(ppy) if te == te else float("nan")
    out["info_ratio"] = (float(active.mean() / te * np.sqrt(ppy))
                         if te and te > 0 else float("nan"))
    rel_eq = (1.0 + p).cumprod() / (1.0 + b).cumprod()
    out["rel_max_drawdown"] = max_drawdown(rel_eq)
    return out
