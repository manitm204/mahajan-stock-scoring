"""Performance and predictive-power metrics derived from a :class:`BacktestResult`.

Everything is computed from the engine's outputs (period return series, equity
curve, name-level forward returns, trade ledger) so the numbers are internally
consistent with the equity curve the plots draw. Returns are annualized using the
rebalance frequency (weekly=52, monthly=12, quarterly=4); month-based stats are
resampled from the equity curve so they are well-defined at any frequency.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

PERIODS_PER_YEAR = {"weekly": 52.0, "monthly": 12.0, "quarterly": 4.0}


def max_drawdown(equity: pd.Series) -> float:
    """Most negative peak-to-trough decline of an equity curve (e.g. -0.23)."""
    eq = equity.dropna()
    if eq.empty:
        return float("nan")
    return float((eq / eq.cummax() - 1.0).min())


def hit_rates(forward: pd.DataFrame) -> dict[str, float]:
    """The four directional hit rates the strategy is judged on."""
    out = {
        "long_hit_rate": np.nan,
        "short_hit_rate": np.nan,
        "long_vs_spy_hit_rate": np.nan,
        "short_underperf_hit_rate": np.nan,
    }
    if forward is None or forward.empty:
        return out
    longs = forward[forward["side"] == "long"]
    shorts = forward[forward["side"] == "short"]
    if not longs.empty:
        out["long_hit_rate"] = float((longs["stock_fwd"] > 0).mean())
        out["long_vs_spy_hit_rate"] = float((longs["stock_fwd"] > longs["spy_fwd"]).mean())
    if not shorts.empty:
        out["short_hit_rate"] = float((shorts["stock_fwd"] < 0).mean())
        out["short_underperf_hit_rate"] = float((shorts["stock_fwd"] < shorts["spy_fwd"]).mean())
    return out


def _monthly_stats(equity: pd.Series) -> tuple[float, float, float]:
    eq = equity.dropna().copy()
    if len(eq) < 2:
        return (np.nan, np.nan, np.nan)
    eq.index = pd.to_datetime(eq.index)
    monthly = eq.resample("ME").last().pct_change().dropna()
    if monthly.empty:
        return (np.nan, np.nan, np.nan)
    return float(monthly.mean()), float(monthly.max()), float(monthly.min())


def compute_metrics(result, risk_free: float = 0.0) -> dict[str, float]:
    """Full metric set for the strategy's primary equity series."""
    ppy = PERIODS_PER_YEAR.get(result.config.rebalance, 12.0)
    primary = result.primary_series

    rets = (result.period_returns[primary].dropna()
            if not result.period_returns.empty else pd.Series(dtype=float))
    equity = (result.equity_curve[primary].dropna()
              if primary in result.equity_curve else pd.Series(dtype=float))

    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1.0) if len(equity) >= 2 else np.nan
    n = len(rets)
    years = n / ppy if ppy else np.nan
    ann_return = (
        float((equity.iloc[-1] / equity.iloc[0]) ** (1.0 / years) - 1.0)
        if len(equity) >= 2 and years and years > 0 else np.nan)
    std = float(rets.std(ddof=1)) if n > 1 else np.nan
    ann_vol = std * np.sqrt(ppy) if not np.isnan(std) else np.nan
    sharpe = (
        float((rets.mean() - risk_free / ppy) / std * np.sqrt(ppy))
        if n > 1 and std and std > 0 else np.nan)
    win_rate = float((rets > 0).mean()) if n else np.nan
    avg_month, best_month, worst_month = _monthly_stats(equity)

    rates = hit_rates(result.forward_returns)

    fwd = result.forward_returns
    longs = fwd[fwd["side"] == "long"] if not fwd.empty else fwd
    shorts = fwd[fwd["side"] == "short"] if not fwd.empty else fwd
    avg_long_fwd = float(longs["stock_fwd"].mean()) if len(longs) else np.nan
    avg_short_fwd = float(shorts["profit"].mean()) if len(shorts) else np.nan

    if self_short_enabled(result) and not result.period_returns.empty:
        spread = (result.period_returns["long_basket"] + result.period_returns["short_basket"])
        avg_spread = float(spread.dropna().mean()) if not spread.dropna().empty else np.nan
    else:
        avg_spread = np.nan

    log = result.rebalance_log
    turnover = float(log["turnover"].mean()) if not log.empty else np.nan
    trades = result.trades
    n_trades = int(len(trades))
    avg_hold = float(trades["holding_days"].mean()) if n_trades else np.nan

    return {
        "total_return": total_return,
        "annualized_return": ann_return,
        "annualized_volatility": ann_vol,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_drawdown(equity),
        "win_rate": win_rate,
        "avg_monthly_return": avg_month,
        "best_month": best_month,
        "worst_month": worst_month,
        "long_hit_rate": rates["long_hit_rate"],
        "short_hit_rate": rates["short_hit_rate"],
        "long_vs_spy_hit_rate": rates["long_vs_spy_hit_rate"],
        "short_underperf_hit_rate": rates["short_underperf_hit_rate"],
        "avg_long_forward_return": avg_long_fwd,
        "avg_short_forward_return": avg_short_fwd,
        "avg_spread_return": avg_spread,
        "turnover": turnover,
        "avg_holding_period_days": avg_hold,
        "number_of_trades": n_trades,
    }


def basket_totals(result) -> dict[str, float]:
    """Total return of each tracked basket (for the console comparison block)."""
    eq = result.equity_curve
    out = {}
    for col in ("long_basket", "short_basket", "long_short", "spy"):
        s = eq[col].dropna() if col in eq else pd.Series(dtype=float)
        out[col] = float(s.iloc[-1] / s.iloc[0] - 1.0) if len(s) >= 2 else np.nan
    return out


def self_short_enabled(result) -> bool:
    return bool(result.spec.short_enabled)


# ---------------------------------------------------------------------------
# IS / OOS segment metrics
# ---------------------------------------------------------------------------
def compute_segment_metrics(result, phase: str) -> dict[str, float]:
    """Recompute the full metric set on a sub-segment of the backtest.

    ``phase`` is either ``"is"`` or ``"oos"``. The function rebuilds the equity
    curve from the period returns inside that phase (so each segment starts
    at 1.0 and is independently interpretable), filters the per-period
    rebalance log and per-name forward returns, and restricts the trade
    ledger to trades whose entry date falls inside the segment.

    Returns ``{}`` when the phase has no rebalances (e.g. holdout disabled).
    """
    pr = result.period_returns
    if pr is None or pr.empty or "phase" not in pr.columns:
        return {}
    mask = (pr["phase"].astype(str) == phase)
    if not mask.any():
        return {}

    seg_periods = pr[mask].copy()
    seg_log = (result.rebalance_log[result.rebalance_log["phase"] == phase].copy()
               if "phase" in result.rebalance_log.columns else result.rebalance_log.copy())
    if not seg_log.empty:
        seg_rebal_dates = set(seg_log["rebalance_date"].astype(str).tolist())
    else:
        seg_rebal_dates = set()

    fwd = result.forward_returns
    if not fwd.empty and seg_rebal_dates:
        seg_fwd = fwd[fwd["rebalance_date"].astype(str).isin(seg_rebal_dates)].copy()
    else:
        seg_fwd = fwd.iloc[0:0].copy()

    trades = result.trades
    if not trades.empty and seg_rebal_dates:
        seg_trades = trades[trades["entry_date"].astype(str).isin(seg_rebal_dates)].copy()
    else:
        seg_trades = trades.iloc[0:0].copy()

    anchor = (str(seg_log["rebalance_date"].iloc[0])
              if not seg_log.empty else str(seg_periods.index[0]))
    seg_equity = _rebuild_equity(seg_periods, result.spec.short_enabled, anchor)

    seg_result = SimpleNamespace(
        config=result.config, spec=result.spec,
        equity_curve=seg_equity,
        period_returns=seg_periods.drop(columns=["phase"], errors="ignore"),
        forward_returns=seg_fwd,
        rebalance_log=seg_log,
        trades=seg_trades,
    )
    seg_result.primary_series = result.primary_series
    return compute_metrics(seg_result)


def _rebuild_equity(periods: pd.DataFrame, short_enabled: bool,
                    anchor: str) -> pd.DataFrame:
    """Compound a segment's period returns to a stand-alone equity curve."""
    active = {"long_basket": True, "long_short": True, "spy": True,
              "short_basket": bool(short_enabled)}
    cols = ["long_basket", "short_basket", "long_short", "spy"]
    if periods.empty:
        return pd.DataFrame(columns=cols)
    rows = [{"date": anchor,
             **{k: (1.0 if active[k] else np.nan) for k in cols}}]
    cum = {k: 1.0 for k in cols}
    for end, p in periods.iterrows():
        row = {"date": end}
        for k in cols:
            r = p.get(k, np.nan)
            if not active[k] or pd.isna(r):
                row[k] = np.nan
            else:
                cum[k] *= (1.0 + r)
                row[k] = cum[k]
        rows.append(row)
    return pd.DataFrame(rows).set_index("date")
