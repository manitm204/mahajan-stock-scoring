"""Composite scoring, portfolio simulation and OOS statistics for the overlay study.

The composite blend mirrors production (``factors.composite.build_composite``): parents
dispersion-equalised (zscore to std 20 around the neutral 50), missing parents held at
50, weighted, renormalised by weight used, re-ranked within GICS sector to 0-100. The
two production helpers are imported from ``factors`` (production code, not research/).

The simulator applies *identical* rules to every variant: monthly rebalance on the test
grid, top-percentile long book, equal weight, 1-month hold, and an explicit transaction
cost of ``cost_per_side`` per unit of traded notional (Σ|Δw|). SPY on the same grid is
the benchmark.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from factors.composite import NEUTRAL, _normalize_parents
from factors.utils import sector_percentile

SPY = "SPY"


# --------------------------------------------------------------------------- #
# Composite from parent frame + weights (production-faithful)
# --------------------------------------------------------------------------- #
def composite_from_parents(frame: pd.DataFrame, weights: dict[str, float],
                           sectors: pd.Series, *, norm_mode: str = "zscore",
                           target_std: float = 20.0, min_obs: int = 5) -> pd.Series:
    cols = sorted(p for p in weights if p in frame.columns and weights[p] > 0)
    if not cols:
        return pd.Series(dtype=float)
    blend = _normalize_parents(frame[cols], norm_mode, target_std)
    comp = pd.Series(0.0, index=frame.index)
    used = 0.0
    for p in cols:
        comp += weights[p] * blend[p].fillna(NEUTRAL)
        used += weights[p]
    if used > 0:
        comp /= used
    secs = sectors.reindex(frame.index).fillna("Unknown")
    out = sector_percentile(comp, secs, higher_is_better=True, min_obs=min_obs)
    # a name with NO scored parent must not get a (neutral) composite —
    # sector_percentile fills missing with 50, so mask AFTER ranking; NaN
    # keeps the name out of every book rule
    out[frame[cols].isna().all(axis=1)] = float("nan")
    return out


# --------------------------------------------------------------------------- #
# Portfolio simulation (identical rules per variant, explicit costs)
# --------------------------------------------------------------------------- #
def top_names(score: pd.Series, top_pct: float) -> list[str]:
    s = score.dropna()
    if s.empty:
        return []
    k = max(1, int(round(len(s) * top_pct)))
    # sort index first + stable sort → deterministic tie-break at the book boundary
    return s.sort_index().sort_values(ascending=False, kind="mergesort").index[:k].tolist()


def simulate(scores: dict[str, pd.Series], matrix: pd.DataFrame, top_pct: float,
             cost_per_side: float) -> pd.DataFrame:
    """Simulate the top-``top_pct`` equal-weight book over the full rebalance sequence.

    Returns a frame indexed by *formation date* with columns: ``gross`` / ``net`` period
    return (formation → next rebalance), ``turnover`` (one-way, 0.5·Σ|Δw|), ``cost``,
    ``spy`` period return, ``n_names``. The last formation date has no realised period
    and is dropped.
    """
    dates = [d for d in sorted(scores) if d in matrix.index]
    rows: list[dict] = []
    prev = pd.Series(dtype=float)
    for i, d in enumerate(dates[:-1]):
        nxt = dates[i + 1]
        names = top_names(scores[d], top_pct)
        if not names:
            continue
        w = pd.Series(1.0 / len(names), index=names)
        px0, px1 = matrix.loc[d], matrix.loc[nxt]
        ret = (px1.reindex(names) / px0.reindex(names)) - 1.0
        ok = ret.notna()
        if not ok.any():
            continue
        ww = w[ok] / w[ok].sum()
        gross = float((ww * ret[ok]).sum())
        idx = prev.index.union(w.index)
        traded = float((w.reindex(idx).fillna(0.0) - prev.reindex(idx).fillna(0.0))
                       .abs().sum())
        cost = cost_per_side * traded
        spy = float(px1[SPY] / px0[SPY] - 1.0) if SPY in px0.index and px0[SPY] > 0 \
            else float("nan")
        rows.append({"date": d, "next": nxt, "gross": gross, "net": gross - cost,
                     "turnover": 0.5 * traded, "cost": cost, "spy": spy,
                     "n_names": int(ok.sum())})
        prev = w
    return pd.DataFrame(rows).set_index("date") if rows else pd.DataFrame(
        columns=["next", "gross", "net", "turnover", "cost", "spy", "n_names"])


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return float("nan")
    dd = equity / equity.cummax() - 1.0
    return float(dd.min())


def _cagr(r: pd.Series, ppy: float = 12.0) -> float:
    r = r.dropna()
    if r.empty:
        return float("nan")
    eq = float((1.0 + r).prod())
    years = len(r) / ppy
    return eq ** (1.0 / years) - 1.0 if years > 0 and eq > 0 else float("nan")


def perf_metrics(returns: pd.Series, spy: pd.Series,
                 turnover: pd.Series | None = None, ppy: float = 12.0) -> dict:
    """CAGR/total/Sharpe/Sortino/vol/maxDD/hit + SPY excess/alpha/beta/IR/TE/rel-DD."""
    r = returns.dropna()
    out: dict = {"n_periods": int(len(r))}
    if r.empty:
        return out
    eq = (1.0 + r).cumprod()
    std = float(r.std(ddof=1)) if len(r) > 1 else float("nan")
    downside = r[r < 0]
    dd = float(np.sqrt((downside ** 2).mean())) if not downside.empty else float("nan")
    out.update({
        "total_return": float(eq.iloc[-1] - 1.0),
        "cagr": _cagr(r, ppy),
        "sharpe": float(r.mean() / std * np.sqrt(ppy)) if std and std > 0 else float("nan"),
        "sortino": float(r.mean() / dd * np.sqrt(ppy)) if dd and dd > 0 else float("nan"),
        "ann_vol": std * np.sqrt(ppy) if std == std else float("nan"),
        "max_drawdown": max_drawdown(eq),
        "hit_rate": float((r > 0).mean()),
        "avg_turnover": float(turnover.dropna().mean())
        if turnover is not None and not turnover.dropna().empty else float("nan"),
    })
    df = pd.concat([r.rename("p"), spy.rename("b")], axis=1).dropna()
    if not df.empty:
        p, b = df["p"], df["b"]
        out["spy_cagr"] = _cagr(b, ppy)
        out["spy_excess_cagr"] = out["cagr"] - out["spy_cagr"]
        varb = float(((b - b.mean()) ** 2).sum())
        if varb > 0:
            beta = float(((p - p.mean()) * (b - b.mean())).sum() / varb)
            out["spy_beta"] = beta
            out["spy_alpha"] = float((p.mean() - beta * b.mean()) * ppy)
        active = p - b
        te = float(active.std(ddof=1)) if len(active) > 1 else float("nan")
        out["spy_te"] = te * np.sqrt(ppy) if te == te else float("nan")
        out["spy_ir"] = float(active.mean() / te * np.sqrt(ppy)) if te and te > 0 \
            else float("nan")
        rel = (1.0 + p).cumprod() / (1.0 + b).cumprod()
        out["spy_rel_max_drawdown"] = max_drawdown(rel)
    return out


# --------------------------------------------------------------------------- #
# OOS composite IC / Q5-Q1 and holdings overlap
# --------------------------------------------------------------------------- #
def per_date_ic(scores: dict[str, pd.Series],
                fwd_by_h: dict[str, dict[str, pd.Series]],
                min_names: int = 20) -> pd.DataFrame:
    """Long frame: date, horizon, ic (Spearman) for every scored test date."""
    rows = []
    for h, by_date in fwd_by_h.items():
        for d, sc in scores.items():
            fwd = by_date.get(d)
            if fwd is None:
                continue
            df = pd.concat([sc.rename("s"), fwd.rename("f")], axis=1).dropna()
            if len(df) < min_names or df["s"].nunique() < 2:
                continue
            rows.append({"date": d, "horizon": h,
                         "ic": float(df["s"].corr(df["f"], method="spearman"))})
    return pd.DataFrame(rows)


def per_date_q5q1(scores: dict[str, pd.Series],
                  fwd1m: dict[str, pd.Series], min_names: int = 20) -> pd.Series:
    out = {}
    for d, sc in scores.items():
        fwd = fwd1m.get(d)
        if fwd is None:
            continue
        df = pd.concat([sc.rename("s"), fwd.rename("f")], axis=1).dropna()
        if len(df) < min_names:
            continue
        q = pd.qcut(df["s"].rank(method="first"), 5, labels=False)
        out[d] = float(df["f"][q == 4].mean() - df["f"][q == 0].mean())
    return pd.Series(out).sort_index()


def overlap_stats(var_scores: dict[str, pd.Series], base_scores: dict[str, pd.Series],
                  top_pct: float) -> pd.DataFrame:
    """Per-date holdings overlap of variant vs baseline top book."""
    rows = []
    for d in sorted(var_scores):
        if d not in base_scores:
            continue
        tv, tb = set(top_names(var_scores[d], top_pct)), \
            set(top_names(base_scores[d], top_pct))
        if not tb:
            continue
        rows.append({"date": d, "overlap": len(tv & tb) / len(tb),
                     "entered": len(tv - tb), "left": len(tb - tv),
                     "n_base": len(tb)})
    return pd.DataFrame(rows).set_index("date") if rows else pd.DataFrame(
        columns=["overlap", "entered", "left", "n_base"])
