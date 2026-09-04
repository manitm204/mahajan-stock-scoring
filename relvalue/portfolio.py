"""Capital accounting, performance statistics, benchmarks and blends."""
from __future__ import annotations

import numpy as np
import pandas as pd


def committed_returns(payoff: pd.DataFrame, n_slots: int) -> pd.Series:
    """Capital committed to the full book whether deployed or not."""
    return payoff.sum(axis=1) / n_slots


def invested_returns(payoff: pd.DataFrame, open_units: pd.DataFrame,
                     floor: int = 8) -> pd.Series:
    """Capital spread across open positions only, floored at ``floor`` slots
    (reserve assumed in cash, uncredited) — the accounting adopted by the
    2026-07 pairs study round 7."""
    n_open = open_units.gt(0).sum(axis=1).clip(lower=floor)
    return payoff.sum(axis=1) / n_open


def perf_stats(daily: pd.Series, bench: dict[str, pd.Series] | None = None) -> dict:
    daily = daily.dropna()
    if daily.empty:
        return {}
    mu, sd = daily.mean(), daily.std()
    ann_ret, ann_vol = mu * 252, sd * np.sqrt(252)
    downside = daily[daily < 0].std() * np.sqrt(252)
    eq = (1 + daily).cumprod()
    dd_series = eq / eq.cummax() - 1
    max_dd = float(dd_series.min())
    # longest drawdown spell in trading days
    under = dd_series < 0
    spells = (~under).cumsum()[under]
    dd_dur = int(spells.value_counts().max()) if len(spells) else 0
    years = len(daily) / 252
    cagr = eq.iloc[-1] ** (1 / years) - 1 if years > 0 else np.nan
    out = {
        "cagr": float(cagr), "ann_ret": float(ann_ret), "ann_vol": float(ann_vol),
        "sharpe": float(ann_ret / ann_vol) if ann_vol > 0 else np.nan,
        "sortino": float(ann_ret / downside) if downside and downside > 0 else np.nan,
        "calmar": float(cagr / abs(max_dd)) if max_dd < 0 else np.nan,
        "max_dd": max_dd, "dd_days": dd_dur,
        "t_stat": float(ann_ret / ann_vol * np.sqrt(years)) if ann_vol > 0 else np.nan,
        "years": float(years),
    }
    for name, b in (bench or {}).items():
        both = pd.concat([daily, b], axis=1, keys=["s", "b"]).dropna()
        if len(both) < 60:
            continue
        cov = both["s"].cov(both["b"])
        var = both["b"].var()
        beta = cov / var if var > 0 else np.nan
        alpha = (both["s"].mean() - beta * both["b"].mean()) * 252
        out[f"beta_{name}"] = float(beta)
        out[f"alpha_{name}"] = float(alpha)
        out[f"corr_{name}"] = float(both["s"].corr(both["b"]))
    return out


def trade_stats(trades: list) -> dict:
    if not trades:
        return {"n_trades": 0}
    pay = np.array([t.payoff for t in trades])
    days = np.array([t.days for t in trades])
    units = np.array([getattr(t, "units", 1.0) for t in trades])
    per = pay / np.maximum(units * 2, 1e-9)          # per $ gross
    wins, losses = pay[pay > 0], pay[pay <= 0]
    reasons = pd.Series([t.reason for t in trades]).value_counts()
    return {
        "n_trades": len(trades),
        "hit_rate": float((pay > 0).mean()),
        "avg_payoff": float(pay.mean()), "med_payoff": float(np.median(pay)),
        "avg_win": float(wins.mean()) if len(wins) else np.nan,
        "avg_loss": float(losses.mean()) if len(losses) else np.nan,
        "profit_factor": float(wins.sum() / -losses.sum()) if losses.sum() < 0 else np.inf,
        "avg_days": float(days.mean()),
        "avg_ret_gross": float(per.mean()),
        "pct_converged": float(reasons.get("converged", 0) / len(trades)),
    }


def bench_returns(tr: pd.DataFrame, tickers=("SPY", "QQQ")) -> dict[str, pd.Series]:
    return {t: tr[t].pct_change(fill_method=None).dropna()
            for t in tickers if t in tr.columns}


def blend(strat: pd.Series, bench: pd.Series, w_strat: float) -> pd.Series:
    """Daily-rebalanced blend of benchmark and strategy."""
    both = pd.concat([strat, bench], axis=1, keys=["s", "b"]).dropna()
    return w_strat * both["s"] + (1 - w_strat) * both["b"]


def yearly_table(daily: pd.Series) -> pd.Series:
    eq = (1 + daily).groupby(daily.index.str[:4]).prod() - 1
    return eq
