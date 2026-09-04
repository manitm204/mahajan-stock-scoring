"""CSV exports and the end-of-run console summary.

Writes the six required artifacts to the run's output folder and renders the
human-readable summary block. Formatting helpers map NaN to ``n/a`` so a run with
a missing series (e.g. a long-only strategy that has no short basket) prints
cleanly instead of showing ``nan``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import basket_totals, compute_segment_metrics

_FREQ_LABEL = {"weekly": "Weekly", "monthly": "Monthly", "quarterly": "Quarterly"}

_FACTOR_COLS = [
    "rebalance_date",
    "q1_forward_return", "q2_forward_return", "q3_forward_return",
    "q4_forward_return", "q5_forward_return", "q5_minus_q1_spread",
]


def save_csvs(result, metrics: dict, output_dir: str | Path) -> dict[str, Path]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    eq = result.equity_curve.copy()
    eq.index.name = "date"
    paths["equity_curve"] = _write(eq, out / "equity_curve.csv", index=True)
    paths["positions"] = _write(result.positions, out / "positions.csv")
    paths["trades"] = _write(_sorted_trades(result.trades), out / "trades.csv")
    paths["rebalance_log"] = _write(result.rebalance_log, out / "rebalance_log.csv")
    paths["performance_summary"] = _write(
        _summary_frame(result, metrics), out / "performance_summary.csv")

    q = result.factor_quintiles
    if not q.empty:
        q = q.reindex(columns=[c for c in _FACTOR_COLS if c in q.columns])
    paths["factor_forward_returns"] = _write(q, out / "factor_forward_returns.csv")

    if not result.weights_log.empty:
        paths["factor_weights"] = _write(result.weights_log, out / "factor_weights.csv")

    holdout_start = getattr(result, "holdout_start", None)
    if holdout_start:
        is_metrics = compute_segment_metrics(result, "is")
        oos_metrics = compute_segment_metrics(result, "oos")
        if is_metrics:
            paths["performance_summary_is"] = _write(
                _segment_summary_frame(result, is_metrics, "is", holdout_start),
                out / "performance_summary_is.csv")
        if oos_metrics:
            paths["performance_summary_oos"] = _write(
                _segment_summary_frame(result, oos_metrics, "oos", holdout_start),
                out / "performance_summary_oos.csv")
    return paths


def console_summary(result, metrics: dict, output_dir: str | Path) -> str:
    cfg = result.config
    totals = basket_totals(result)
    L: list[str] = []
    add = L.append

    add("")
    add("=" * 64)
    add(" BACKTEST SUMMARY ".center(64, "="))
    add("=" * 64)
    add(f" Strategy            : {result.spec.label}")
    add(f" Date Range          : {cfg.start_date}  ->  {cfg.end_date}")
    add(f" Rebalance Frequency : {_FREQ_LABEL.get(cfg.rebalance, cfg.rebalance)} "
        f"({len(result.rebalance_dates)} rebalances)")
    add(f" Transaction Costs   : {'ON' if cfg.transaction_costs else 'OFF'} "
        f"({cfg.transaction_cost_bps:.0f} bps trade, {cfg.short_borrow_cost_annual_bps:.0f} bps/yr borrow)")
    add("-" * 64)
    add(f" Total Return        : {_pct(metrics['total_return'])}")
    add(f" Annualized Return   : {_pct(metrics['annualized_return'])}")
    add(f" Annualized Vol      : {_pct(metrics['annualized_volatility'])}")
    add(f" Sharpe              : {_num(metrics['sharpe_ratio'])}")
    add(f" Max Drawdown        : {_pct(metrics['max_drawdown'])}")
    add("-" * 64)
    add(f" Long Basket Return  : {_pct(totals['long_basket'])}")
    add(f" Short Basket Return : {_pct(totals['short_basket'])}")
    add(f" Long/Short Return   : {_pct(totals['long_short'])}")
    add(f" SPY Return          : {_pct(totals['spy'])}")
    add("-" * 64)
    add(f" Long Hit Rate       : {_pct(metrics['long_hit_rate'])}")
    add(f" Short Hit Rate      : {_pct(metrics['short_hit_rate'])}")
    add(f" Long vs SPY Hit Rate: {_pct(metrics['long_vs_spy_hit_rate'])}")
    add(f" Short Underperf Rate: {_pct(metrics['short_underperf_hit_rate'])}")
    add(f" Avg Holding Period  : {_days(metrics['avg_holding_period_days'])}")
    add(f" Turnover (1-way/reb): {_pct(metrics['turnover'])}")
    add(f" Number of Trades    : {metrics['number_of_trades']}")

    holdout_start = getattr(result, "holdout_start", None)
    if holdout_start:
        add("-" * 64)
        add(f" IN-SAMPLE / OUT-OF-SAMPLE SPLIT  (OOS starts {holdout_start})")
        add(_segment_block(result))

    add("-" * 64)
    add(" AVERAGE FACTOR WEIGHTS (mean over rebalances)")
    add(_weights_block(result))
    add("-" * 64)
    add(" TOP 5 BEST TRADES")
    add(_trade_block(result.trades, best=True))
    add(" TOP 5 WORST TRADES")
    add(_trade_block(result.trades, best=False))
    add("-" * 64)
    add(" SAVED OUTPUTS (-> {}/ )".format(str(output_dir).rstrip("/")))
    csvs = ["equity_curve", "positions", "trades", "rebalance_log",
            "performance_summary", "factor_forward_returns"]
    if not result.weights_log.empty:
        csvs.append("factor_weights")
    if getattr(result, "holdout_start", None):
        csvs += ["performance_summary_is", "performance_summary_oos"]
    for name in csvs:
        add(f"   - {name}.csv")
    pngs = ["equity_curve", "drawdowns", "rolling_returns",
            "long_forward_return_distribution", "short_forward_return_distribution",
            "hit_rates", "factor_quintile_returns"]
    if not result.weights_log.empty:
        pngs.append("factor_weights")
    for name in pngs:
        add(f"   - {name}.png")
    if result.pit_notes:
        add("-" * 64)
        add(" POINT-IN-TIME ASSUMPTIONS")
        for note in result.pit_notes:
            add(f"   - {note}")
    add("=" * 64)
    return "\n".join(L)


# ---------------------------------------------------------------------------
def _summary_frame(result, metrics: dict) -> pd.DataFrame:
    cfg = result.config
    rows = [
        ("strategy", result.spec.name),
        ("strategy_label", result.spec.label),
        ("start_date", cfg.start_date),
        ("end_date", cfg.end_date),
        ("rebalance", cfg.rebalance),
        ("transaction_costs", "on" if cfg.transaction_costs else "off"),
        ("transaction_cost_bps", cfg.transaction_cost_bps),
        ("short_borrow_cost_annual_bps", cfg.short_borrow_cost_annual_bps),
        ("n_rebalances", len(result.rebalance_dates)),
        ("primary_series", result.primary_series),
    ]
    rows += [(k, _round(v)) for k, v in metrics.items()]
    totals = basket_totals(result)
    rows += [(f"{k}_total_return", _round(v)) for k, v in totals.items()]
    return pd.DataFrame(rows, columns=["metric", "value"])


def _segment_summary_frame(result, metrics: dict, phase: str,
                           holdout_start: str) -> pd.DataFrame:
    cfg = result.config
    log = result.rebalance_log
    seg_log = (log[log["phase"] == phase] if "phase" in log.columns else log)
    if not seg_log.empty:
        start = seg_log["rebalance_date"].iloc[0]
        end = seg_log["period_end"].iloc[-1]
        n = int(len(seg_log))
    else:
        start = end = ""
        n = 0
    rows = [
        ("segment", phase),
        ("strategy", result.spec.name),
        ("strategy_label", result.spec.label),
        ("holdout_start", holdout_start),
        ("segment_start", start),
        ("segment_end", end),
        ("n_rebalances", n),
        ("primary_series", result.primary_series),
        ("rebalance", cfg.rebalance),
    ]
    rows += [(k, _round(v)) for k, v in metrics.items()]
    return pd.DataFrame(rows, columns=["metric", "value"])


def _segment_block(result) -> str:
    """Side-by-side IS / OOS headline metrics for the console summary."""
    is_m = compute_segment_metrics(result, "is")
    oos_m = compute_segment_metrics(result, "oos")
    rows = [
        ("Ann. Return",    "annualized_return",    _pct),
        ("Ann. Vol",       "annualized_volatility", _pct),
        ("Sharpe",         "sharpe_ratio",          _num),
        ("Max Drawdown",   "max_drawdown",          _pct),
        ("Win Rate",       "win_rate",              _pct),
        ("Long Hit Rate",  "long_hit_rate",         _pct),
        ("Long vs SPY",    "long_vs_spy_hit_rate",  _pct),
        ("Turnover",       "turnover",              _pct),
    ]
    log = result.rebalance_log
    n_is = int((log["phase"] == "is").sum()) if "phase" in log.columns else 0
    n_oos = int((log["phase"] == "oos").sum()) if "phase" in log.columns else 0
    L = [f"   {'':<18}  {'IS':>10}  {'OOS':>10}",
         f"   {'rebalances':<18}  {n_is:>10d}  {n_oos:>10d}"]
    for label, key, fmt in rows:
        L.append(f"   {label:<18}  {fmt(is_m.get(key, np.nan)):>10}  "
                 f"{fmt(oos_m.get(key, np.nan)):>10}")
    return "\n".join(L)


def _weights_block(result) -> str:
    """Mean composite weight per factor over the run, strongest first."""
    wl = result.weights_log
    if wl is None or wl.empty:
        return "   (n/a)"
    wcols = [c for c in wl.columns if c.endswith("_w")]
    means = wl[wcols].mean().sort_values(ascending=False)
    lines = [f"   {col[:-2]:<14}: {_pct(v)}" for col, v in means.items()]
    if "applied" in wl.columns:
        tag = ", ".join(f"{k}×{v}" for k, v in wl["applied"].value_counts().items())
        lines.append(f"   modes          : {tag}")
    return "\n".join(lines)


def _sorted_trades(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return trades
    return trades.sort_values(["entry_date", "side", "ticker"]).reset_index(drop=True)


def _trade_block(trades: pd.DataFrame, best: bool) -> str:
    if trades.empty or trades["return"].notna().sum() == 0:
        return "   (none)"
    valid = trades.dropna(subset=["return"]).sort_values("return", ascending=not best)
    rows = valid.head(5)
    lines = []
    for _, t in rows.iterrows():
        lines.append(
            f"   {t['ticker']:<6} {t['side']:<5} "
            f"{t['entry_date']} -> {t['exit_date']}  "
            f"{_pct(t['return']):>9}  ({int(t['holding_periods'])}p/{int(t['holding_days'])}d)")
    return "\n".join(lines)


def _write(df: pd.DataFrame, path: Path, index: bool = False) -> Path:
    df.to_csv(path, index=index)
    return path


def _round(v):
    if isinstance(v, (int, np.integer)):
        return int(v)
    try:
        f = float(v)
    except (TypeError, ValueError):
        return v
    return None if np.isnan(f) else round(f, 6)


def _pct(v) -> str:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "n/a"
    return "n/a" if np.isnan(f) else f"{f * 100:.2f}%"


def _num(v) -> str:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "n/a"
    return "n/a" if np.isnan(f) else f"{f:.2f}"


def _days(v) -> str:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "n/a"
    return "n/a" if np.isnan(f) else f"{f:.0f} days"
