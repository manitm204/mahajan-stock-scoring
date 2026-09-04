"""Regime-by-regime performance analysis focused on the rolling-5y model.

Re-runs the semiannual walk-forward for the ``rolling5y`` policy only (half the work of
the full regime study) and captures a *rich* per-window metric bundle: portfolio CAGR /
Sharpe / max-DD, SPY-standalone CAGR / Sharpe / max-DD on the identical window, portfolio
beta / IR / excess vs SPY, composite IC (per horizon), and the Q5–Q1 spread. Each 6-month
test window is a monthly-rebalanced top-20 % book; because the last-month formation's
1-month forward return settles inside the next window, the effective evaluation coverage
of a nominal ``H1``/``H2`` window is ≈9 months once holdings are counted — flagged in the
report but computed on the strict window-return series.

Downstream: :func:`build_focused_summary` produces per-window, 3-year-bucket, and full-
period aggregate tables ready for reporting; :mod:`research.walkforward.regime5y_report`
turns them into markdown/CSV/PNG under ``output/regime5y_focus/``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from backtesting.data_loader import SPY
from backtesting.metrics import max_drawdown
from research import HORIZON_MONTHS, compute_forward_returns
from research.panel import ScorePanel

from . import analysis
from . import portfolio as pf
from .compose import FrozenConfig, frozen_composite
from .selection import select_config
from .splits import WalkForwardSplit, semiannual_labels, semiannual_policy_splits


HORIZON_KEY_IC = "6M"     # headline composite IC per window
HORIZON_KEY_SPREAD = "6M" # headline Q5-Q1 spread per window
TOP_PCT = 0.20            # focus book


@dataclass
class FocusWindow:
    """Per-window artefacts for the focused rolling-5y analysis."""

    label: str
    split: WalkForwardSplit
    config: FrozenConfig
    scores: dict[str, pd.Series]
    portfolio_metrics: dict           # from pf.performance_metrics
    spy_metrics: dict                 # {spy_cagr, spy_sharpe, spy_max_drawdown, spy_ann_vol}
    ic_by_horizon: dict[str, float]   # {horizon: mean_ic}
    q5_q1_spread: dict                # {annualized, sharpe, hit_rate, monotonic_rate}
    n_train: int
    n_test: int


@dataclass
class FocusRun:
    windows: list[FocusWindow] = field(default_factory=list)
    matrix: pd.DataFrame = field(default_factory=pd.DataFrame)
    sectors: pd.Series = field(default_factory=lambda: pd.Series(dtype=object))
    labels: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Per-window computation
# --------------------------------------------------------------------------- #
def _spy_standalone(spy_returns: pd.Series, hold_months: int = 1) -> dict:
    """Level/annualised SPY stats on the aligned per-period return series."""
    r = spy_returns.dropna()
    if r.empty:
        return {k: float("nan") for k in
                ("spy_cagr", "spy_sharpe", "spy_sortino", "spy_max_drawdown", "spy_ann_vol")}
    ppy = 12.0 / hold_months
    equity = (1.0 + r).cumprod()
    years = len(r) / ppy
    std = float(r.std(ddof=1)) if len(r) > 1 else float("nan")
    downside = r.clip(upper=0.0)
    dd_std = float(np.sqrt((downside ** 2).mean())) if len(r) > 0 else float("nan")
    return {
        "spy_cagr": float(equity.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 else float("nan"),
        "spy_sharpe": float(r.mean() / std * np.sqrt(ppy)) if std and std > 0 else float("nan"),
        "spy_sortino": float(r.mean() / dd_std * np.sqrt(ppy)) if dd_std and dd_std > 0 else float("nan"),
        "spy_max_drawdown": float(max_drawdown(equity)),
        "spy_ann_vol": std * np.sqrt(ppy) if std == std else float("nan"),
    }


def _q5_q1_for_window(scores: dict[str, pd.Series], fwd_by_h: dict[str, dict[str, pd.Series]],
                      horizon: str) -> dict:
    """Q5-Q1 spread stats for the single test window at ``horizon``."""
    qres = analysis.quantile_analysis(scores, fwd_by_h)
    q = qres.get(horizon)
    if not q:
        return {"annualized": np.nan, "sharpe": np.nan, "hit_rate": np.nan,
                "monotonic_rate": np.nan, "n_periods": 0}
    spr = q.get("spread") or {}
    return {
        "annualized": spr.get("annualized"),
        "sharpe": spr.get("sharpe"),
        "hit_rate": spr.get("hit_rate"),
        "monotonic_rate": q.get("monotonic_rate"),
        "n_periods": q.get("n_periods"),
    }


def _ic_by_horizon(scores: dict[str, pd.Series],
                   fwd_by_h: dict[str, dict[str, pd.Series]]) -> dict[str, float]:
    df = analysis.composite_ic(scores, fwd_by_h)
    if df.empty:
        return {}
    return {row["horizon"]: float(row.get("mean_ic", np.nan))
            for _, row in df.iterrows()}


def run_focused_5y(panel: ScorePanel, matrix: pd.DataFrame, sectors: pd.Series,
                   *, first_test_year: int = 2017,
                   last_end: str | None = None, verbose: bool = True) -> FocusRun:
    """Loop the rolling-5y semiannual grid and build a :class:`FocusRun`."""
    kwargs = {"first_test_year": first_test_year}
    if last_end is not None:
        kwargs["last_end"] = last_end

    splits = semiannual_policy_splits("rolling5y", **kwargs)
    labels = semiannual_labels(first_test_year=first_test_year,
                               **({"last_end": last_end} if last_end else {}))
    run = FocusRun(matrix=matrix, sectors=sectors, labels=labels)

    for sp in splits:
        train_rebals = sp.train_rebalances(panel.rebal_dates)
        test_rebals = sp.test_rebalances(panel.rebal_dates)
        if not train_rebals or not test_rebals:
            if verbose:
                print(f"  skip {sp.label}")
            continue
        if verbose:
            print(f"  {sp.label}: select on {len(train_rebals)} "
                  f"({train_rebals[0]}→{train_rebals[-1]}), test {len(test_rebals)} "
                  f"({test_rebals[0]}→{test_rebals[-1]})")
        cfg = select_config(panel, train_rebals, matrix, boundary=sp.test_start)
        scores = frozen_composite(panel, test_rebals, cfg, sectors)

        book = pf.simulate(scores, matrix, sectors, top_pct=TOP_PCT, mode="equal",
                           hold_months=1)
        spy_ret = book.bench_period_returns.get(SPY, pd.Series(dtype=float))
        spy_stats = _spy_standalone(spy_ret, hold_months=1)

        fwd = compute_forward_returns(matrix, list(scores), HORIZON_MONTHS)
        ic_by_h = _ic_by_horizon(scores, fwd)
        q5q1 = _q5_q1_for_window(scores, fwd, horizon=HORIZON_KEY_SPREAD)

        label_root = sp.label.split(":", 1)[1]
        run.windows.append(FocusWindow(
            label=label_root, split=sp, config=cfg, scores=scores,
            portfolio_metrics=book.metrics, spy_metrics=spy_stats,
            ic_by_horizon=ic_by_h, q5_q1_spread=q5q1,
            n_train=len(train_rebals), n_test=len(test_rebals),
        ))
    return run


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
BUCKET_DEFS = [("2017-2019", 2017, 2019), ("2020-2022", 2020, 2022),
               ("2023-2025", 2023, 2025), ("2026+", 2026, 2099)]


def _year(label: str) -> int:
    return int(label.split("-", 1)[0])


def _window_row(w: FocusWindow) -> dict:
    m = w.portfolio_metrics
    return {
        "window": w.label, "n_periods": m.get("n_periods"),
        "port_cagr": m.get("cagr"), "port_sharpe": m.get("sharpe"),
        "port_sortino": m.get("sortino"),
        "port_max_drawdown": m.get("max_drawdown"), "port_ann_vol": m.get("ann_vol"),
        "spy_cagr": w.spy_metrics.get("spy_cagr"),
        "spy_sharpe": w.spy_metrics.get("spy_sharpe"),
        "spy_sortino": w.spy_metrics.get("spy_sortino"),
        "spy_max_drawdown": w.spy_metrics.get("spy_max_drawdown"),
        "spy_ann_vol": w.spy_metrics.get("spy_ann_vol"),
        "excess_cagr": m.get("spy_excess_cagr"),
        "alpha": m.get("spy_alpha"),
        "beta": m.get("spy_beta"),
        "info_ratio": m.get("spy_ir"),
        "tracking_error": m.get("spy_te"),
        "rel_max_drawdown": m.get("spy_rel_max_drawdown"),
        "composite_ic_1M": w.ic_by_horizon.get("1M"),
        "composite_ic_3M": w.ic_by_horizon.get("3M"),
        "composite_ic_6M": w.ic_by_horizon.get("6M"),
        "composite_ic_12M": w.ic_by_horizon.get("12M"),
        "q5_q1_spread_annualized": w.q5_q1_spread.get("annualized"),
        "q5_q1_spread_sharpe": w.q5_q1_spread.get("sharpe"),
        "q5_q1_spread_hit_rate": w.q5_q1_spread.get("hit_rate"),
        "q5_q1_monotonic_rate": w.q5_q1_spread.get("monotonic_rate"),
    }


def per_window_table(run: FocusRun) -> pd.DataFrame:
    """One row per test window with the full metric bundle."""
    return pd.DataFrame([_window_row(w) for w in run.windows])


def pooled_returns(run: FocusRun,
                   windows: list[FocusWindow]) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Concatenate per-window portfolio / SPY / turnover returns across ``windows``.

    Recomputes each window's monthly book from its frozen scores + the shared price matrix
    so we don't have to persist the potentially-large per-window Series on FocusWindow.
    """
    ret_parts, spy_parts, turn_parts = [], [], []
    for w in windows:
        book = pf.simulate(w.scores, run.matrix, run.sectors, top_pct=TOP_PCT,
                           mode="equal", hold_months=1)
        ret_parts.append(book.period_returns)
        spy_parts.append(book.bench_period_returns.get(SPY, pd.Series(dtype=float)))
        turn_parts.append(book.turnover)
    if not ret_parts:
        return (pd.Series(dtype=float),) * 3
    dedup = lambda s: s[~s.index.duplicated(keep="first")].sort_index()
    return (dedup(pd.concat(ret_parts)),
            dedup(pd.concat(spy_parts)),
            dedup(pd.concat(turn_parts)))


def _grouped_metrics(run: FocusRun, windows: list[FocusWindow],
                     label: str) -> dict:
    """Aggregate metrics across ``windows``: pool returns + compute portfolio/SPY stats."""
    ret, spy, turn = pooled_returns(run, windows)
    if ret.empty:
        return {"grouping": label, "n_periods": 0}
    metrics = pf.performance_metrics(ret, hold_months=1, turnover=turn,
                                     benchmarks={SPY: spy})
    spy_stats = _spy_standalone(spy, hold_months=1)
    # pool composite IC + Q5-Q1 across windows' scores
    pooled_scores: dict[str, pd.Series] = {}
    for w in windows:
        pooled_scores.update(w.scores)
    fwd = compute_forward_returns(run.matrix, list(pooled_scores), HORIZON_MONTHS)
    ic = analysis.composite_ic(pooled_scores, fwd)
    ic_by_h = {row["horizon"]: float(row.get("mean_ic", np.nan))
               for _, row in ic.iterrows()}
    q5q1 = _q5_q1_for_window(pooled_scores, fwd, HORIZON_KEY_SPREAD)
    return {
        "grouping": label,
        "n_windows": len(windows),
        "n_periods": int(ret.dropna().size),
        "port_cagr": metrics.get("cagr"),
        "port_sharpe": metrics.get("sharpe"),
        "port_max_drawdown": metrics.get("max_drawdown"),
        "port_ann_vol": metrics.get("ann_vol"),
        "spy_cagr": spy_stats.get("spy_cagr"),
        "spy_sharpe": spy_stats.get("spy_sharpe"),
        "spy_max_drawdown": spy_stats.get("spy_max_drawdown"),
        "spy_ann_vol": spy_stats.get("spy_ann_vol"),
        "excess_cagr": metrics.get("spy_excess_cagr"),
        "alpha": metrics.get("spy_alpha"),
        "beta": metrics.get("spy_beta"),
        "info_ratio": metrics.get("spy_ir"),
        "tracking_error": metrics.get("spy_te"),
        "rel_max_drawdown": metrics.get("spy_rel_max_drawdown"),
        "composite_ic_1M": ic_by_h.get("1M"),
        "composite_ic_3M": ic_by_h.get("3M"),
        "composite_ic_6M": ic_by_h.get("6M"),
        "composite_ic_12M": ic_by_h.get("12M"),
        "q5_q1_spread_annualized": q5q1.get("annualized"),
        "q5_q1_spread_sharpe": q5q1.get("sharpe"),
        "q5_q1_monotonic_rate": q5q1.get("monotonic_rate"),
    }


def bucket_table(run: FocusRun) -> pd.DataFrame:
    """3-year rolling buckets."""
    rows: list[dict] = []
    for name, lo, hi in BUCKET_DEFS:
        wsub = [w for w in run.windows if lo <= _year(w.label) <= hi]
        if not wsub:
            continue
        rows.append(_grouped_metrics(run, wsub, name))
    return pd.DataFrame(rows)


def full_table(run: FocusRun) -> pd.DataFrame:
    """Full-period single-row aggregate."""
    if not run.windows:
        return pd.DataFrame()
    return pd.DataFrame([_grouped_metrics(run, run.windows, "2017–present")])


def build_focused_summary(run: FocusRun) -> dict[str, pd.DataFrame]:
    return {
        "per_window": per_window_table(run),
        "bucket": bucket_table(run),
        "full": full_table(run),
    }
