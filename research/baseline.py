"""Corrected-baseline factor model: predictive-power weighting, evaluated OOF.

This module turns the captured :class:`~research.panel.ScorePanel` (every parent
and sub-factor already scored 0-100 in its *audited* economic direction — see
``output/factor_research/FACTOR_DIRECTION_AUDIT.md``) into a single composite
signal whose factor and sub-factor weights are set by **predictive power, not
trailing return**.

Two ideas keep it honest:

* **Weight by Information Coefficient, shrunk toward equal weight.** A signal's
  weight is proportional to its (floored-at-zero) mean IC over the training
  window, then blended with the equal-weight prior by ``shrink``. Floored at zero
  because the Factor Direction Audit is explicit that a directionally-correct but
  regime-weak factor (e.g. quality 2022-24) must be *down-weighted, never
  sign-flipped* to chase IC — flipping signs to maximise in-sample IC is
  data-snooping. Shrinkage stops one lucky window from zeroing a factor.

* **Walk-forward, so the headline read is out-of-fold.** At each rebalance the
  weights come only from rebalances whose forward-return outcome was already
  realised by that date (a ``settle_lag`` guard drops the still-in-flight tail).
  The composite at date *d* therefore never sees a return from on/after *d* —
  the same point-in-time discipline the rest of Layer 2 holds to.

The naive equal-weight composite and the current config-weighted composite are
built the same way as benchmarks, so "did the corrected, IC-weighted baseline
actually help?" is a like-for-like comparison.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np
import pandas as pd

from .ic import period_ic
from .panel import ScorePanel

NEUTRAL = 50.0


@dataclass
class BaselineConfig:
    """Tunables for the OOF baseline. Defaults chosen to resist overfitting."""

    ic_horizon: str = "1M"          # forward horizon used to learn weights
    shrink: float = 0.5             # blend IC weights toward equal weight [0,1]
    ic_floor: float = 0.0           # clip training IC below this before weighting
    min_train_periods: int = 12     # warm-up: fewer settled periods -> equal weight
    settle_lag: int = 1             # rebalances near d whose outcome isn't realised yet
    min_names: int = 20             # min cross-section to trust a period's IC


# ---------------------------------------------------------------------------
# Weight derivation
# ---------------------------------------------------------------------------
def equal_weights(keys: list[str]) -> dict[str, float]:
    if not keys:
        return {}
    w = 1.0 / len(keys)
    return {k: w for k in keys}


def weights_from_ic(
    ic_map: dict[str, float], keys: list[str], shrink: float, ic_floor: float
) -> dict[str, float]:
    """IC-proportional weights (floored, normalised) shrunk to equal weight.

    A key with non-finite or below-floor IC contributes zero raw weight; if every
    key floors out, the result is the equal-weight prior (we never invert a sign
    to manufacture a positive weight). ``shrink`` then pulls the IC weights toward
    equal weight so thin evidence cannot fully concentrate the model.
    """
    if not keys:
        return {}
    eq = equal_weights(keys)
    raw = {}
    for k in keys:
        ic = ic_map.get(k, float("nan"))
        raw[k] = max(float(ic) - ic_floor, 0.0) if np.isfinite(ic) else 0.0
    total = sum(raw.values())
    base = {k: raw[k] / total for k in keys} if total > 1e-12 else dict(eq)
    s = min(max(shrink, 0.0), 1.0)
    return {k: (1.0 - s) * base[k] + s * eq[k] for k in keys}


def mean_ic(
    panel: ScorePanel,
    fwd_h: dict[str, pd.Series],
    signal: str,
    dates: list[str],
    min_names: int,
) -> float:
    """Average per-period Spearman IC of ``signal`` over ``dates`` (NaN if none)."""
    ics: list[float] = []
    for d in dates:
        if d not in panel.scores or d not in fwd_h:
            continue
        frame = panel.scores[d]
        if signal not in frame.columns:
            continue
        ic = period_ic(frame[signal], fwd_h[d], min_names=min_names)
        if ic is not None:
            ics.append(ic)
    return float(np.mean(ics)) if ics else float("nan")


@dataclass
class WeightSet:
    """A full set of model weights: parent-level and sub-within-parent."""

    parent: dict[str, float]
    sub: dict[str, dict[str, float]]          # parent -> {sub -> weight}
    n_train: int = 0
    is_warmup: bool = False

    def sub_flat(self) -> dict[str, float]:
        """Effective standalone weight of each sub in the composite (parent x sub)."""
        out: dict[str, float] = {}
        for p, subs in self.sub.items():
            pw = self.parent.get(p, 0.0)
            for s, w in subs.items():
                out[s] = pw * w
        return out


def train_weights(
    panel: ScorePanel,
    fwd_h: dict[str, pd.Series],
    train_dates: list[str],
    cfg: BaselineConfig,
) -> WeightSet:
    """Learn IC-shrunk parent and sub weights from ``train_dates`` only."""
    parent_ic = {
        p: mean_ic(panel, fwd_h, p, train_dates, cfg.min_names)
        for p in panel.parent_keys
    }
    parent_w = weights_from_ic(parent_ic, panel.parent_keys, cfg.shrink, cfg.ic_floor)
    sub_w: dict[str, dict[str, float]] = {}
    for p in panel.parent_keys:
        subs = panel.sub_by_parent.get(p, [])
        sub_ic = {s: mean_ic(panel, fwd_h, s, train_dates, cfg.min_names) for s in subs}
        sub_w[p] = weights_from_ic(sub_ic, subs, cfg.shrink, cfg.ic_floor)
    return WeightSet(parent=parent_w, sub=sub_w, n_train=len(train_dates))


# ---------------------------------------------------------------------------
# Composite construction
# ---------------------------------------------------------------------------
def compose_one(panel: ScorePanel, date: str, weights: WeightSet) -> pd.Series:
    """Composite score (0-100-ish) for one date under ``weights``.

    Parent score = weighted mean of its sub-factor scores; composite = weighted
    mean of parent scores. Missing sub scores fall back to neutral 50, matching
    the rest of Layer 2 (a missing input is never a penalty).
    """
    frame = panel.scores[date]
    parent_vals: dict[str, pd.Series] = {}
    for p in panel.parent_keys:
        subs = panel.sub_by_parent.get(p, [])
        sw = weights.sub.get(p, {})
        present = [s for s in subs if s in frame.columns and sw.get(s, 0.0) > 0]
        if not present:
            # Fall back to the panel's own equal-weight parent score if available.
            parent_vals[p] = frame[p].reindex(panel.universe) if p in frame.columns \
                else pd.Series(NEUTRAL, index=panel.universe)
            continue
        wsum = sum(sw[s] for s in present)
        acc = pd.Series(0.0, index=panel.universe)
        for s in present:
            acc += (sw[s] / wsum) * frame[s].reindex(panel.universe).fillna(NEUTRAL)
        parent_vals[p] = acc
    comp = pd.Series(0.0, index=panel.universe)
    pw = weights.parent
    pwsum = sum(pw.get(p, 0.0) for p in panel.parent_keys) or 1.0
    for p in panel.parent_keys:
        comp += (pw.get(p, 0.0) / pwsum) * parent_vals[p].fillna(NEUTRAL)
    return comp


def static_composite(
    panel: ScorePanel, weights: WeightSet, dates: list[str] | None = None
) -> dict[str, pd.Series]:
    """One fixed weight set applied to every date (benchmarks, deploy preview)."""
    dates = dates or panel.rebal_dates
    return {d: compose_one(panel, d, weights) for d in dates}


def walk_forward_composite(
    panel: ScorePanel, fwd_h: dict[str, pd.Series], cfg: BaselineConfig
) -> tuple[dict[str, pd.Series], dict[str, WeightSet]]:
    """Out-of-fold composite: each date scored with weights from settled history.

    ``settle_lag`` excludes the most recent ``settle_lag`` rebalances before *d*
    whose ``ic_horizon`` forward window has not finished by *d* (no look-ahead).
    Until ``min_train_periods`` settled periods exist the date uses equal weights.
    """
    dates = panel.rebal_dates
    comp: dict[str, pd.Series] = {}
    hist: dict[str, WeightSet] = {}
    eq = WeightSet(
        parent=equal_weights(panel.parent_keys),
        sub={p: equal_weights(panel.sub_by_parent.get(p, [])) for p in panel.parent_keys},
        is_warmup=True,
    )
    for i, d in enumerate(dates):
        train = dates[: max(0, i - cfg.settle_lag)]
        if len(train) < cfg.min_train_periods:
            ws = WeightSet(parent=eq.parent, sub=eq.sub, n_train=len(train),
                           is_warmup=True)
        else:
            ws = train_weights(panel, fwd_h, train, cfg)
        hist[d] = ws
        comp[d] = compose_one(panel, d, ws)
    return comp, hist


# ---------------------------------------------------------------------------
# Evaluation (reuses the research quintile / IC machinery)
# ---------------------------------------------------------------------------
def avg_cross_sectional_corr(
    a: dict[str, pd.Series], b: dict[str, pd.Series], dates: list[str]
) -> float:
    """Mean per-date Spearman correlation between two signals (NaN if none)."""
    corrs: list[float] = []
    for d in dates:
        if d not in a or d not in b:
            continue
        df = pd.DataFrame({"a": a[d], "b": b[d]}).dropna()
        if len(df) < 3 or df["a"].nunique() < 2 or df["b"].nunique() < 2:
            continue
        c = df["a"].corr(df["b"], method="spearman")
        if pd.notna(c):
            corrs.append(float(c))
    return float(np.mean(corrs)) if corrs else float("nan")


def factor_contribution(
    panel: ScorePanel,
    fwd_h: dict[str, pd.Series],
    full_weights: WeightSet,
    oof_history: dict[str, WeightSet],
    oof_composite: dict[str, pd.Series],
    cfg: BaselineConfig,
) -> pd.DataFrame:
    """Per-parent contribution: weight, standalone IC, weight x IC, corr to composite.

    ``weight`` is the deployable full-sample weight; ``avg_oof_weight`` is the mean
    weight the walk-forward actually used (shows how much warm-up/shrinkage moved
    it). ``contribution`` = weight x standalone IC is a first-order read of how much
    predictive power each factor brings to the blend.
    """
    dates = panel.rebal_dates
    parent_series = {
        p: {d: panel.scores[d][p] for d in dates if p in panel.scores[d].columns}
        for p in panel.parent_keys
    }
    rows = []
    for p in panel.parent_keys:
        ic = mean_ic(panel, fwd_h, p, dates, cfg.min_names)
        w = full_weights.parent.get(p, float("nan"))
        avg_oof = float(np.mean([h.parent.get(p, np.nan) for h in oof_history.values()]))
        rows.append({
            "parent": p,
            "weight": w,
            "avg_oof_weight": avg_oof,
            "standalone_ic": ic,
            "contribution": (w * ic) if np.isfinite(ic) and np.isfinite(w) else float("nan"),
            "corr_to_composite": avg_cross_sectional_corr(
                parent_series[p], oof_composite, dates),
        })
    df = pd.DataFrame(rows)
    tot = df["contribution"].sum(skipna=True)
    df["contribution_share"] = df["contribution"] / tot if tot else float("nan")
    return df.sort_values("contribution", ascending=False, na_position="last").reset_index(drop=True)


def subfactor_weight_table(
    panel: ScorePanel,
    fwd_h: dict[str, pd.Series],
    full_weights: WeightSet,
    cfg: BaselineConfig,
) -> pd.DataFrame:
    """Per-sub: within-parent weight, effective weight, and standalone IC."""
    flat = full_weights.sub_flat()
    rows = []
    for p in panel.parent_keys:
        for s in panel.sub_by_parent.get(p, []):
            rows.append({
                "sub_factor": s,
                "parent": p,
                "within_parent_weight": full_weights.sub.get(p, {}).get(s, float("nan")),
                "effective_weight": flat.get(s, float("nan")),
                "standalone_ic": mean_ic(panel, fwd_h, s, panel.rebal_dates, cfg.min_names),
            })
    return pd.DataFrame(rows).sort_values(
        ["parent", "within_parent_weight"], ascending=[True, False]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Simple quintile backtest (portfolio-level stats: CAGR, Sharpe, drawdown)
# ---------------------------------------------------------------------------
PERIODS_PER_YEAR = 12.0   # monthly rebalance, matching backtesting.metrics


def portfolio_stats(returns: pd.Series, ppy: float = PERIODS_PER_YEAR,
                    risk_free: float = 0.0) -> dict[str, float]:
    """Annualised stats for a monthly return series (same conventions as
    :mod:`backtesting.metrics`: ddof=1 std, sqrt(ppy) annualisation, CAGR off
    the compounded equity curve)."""
    r = returns.dropna()
    n = len(r)
    if n == 0:
        return {k: float("nan") for k in
                ("total_return", "cagr", "ann_vol", "sharpe", "max_drawdown",
                 "win_rate", "best_month", "worst_month", "avg_month", "n_periods")}
    equity = (1.0 + r).cumprod()
    years = n / ppy
    cagr = float(equity.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 else float("nan")
    std = float(r.std(ddof=1)) if n > 1 else float("nan")
    sharpe = (float((r.mean() - risk_free / ppy) / std * np.sqrt(ppy))
              if n > 1 and std and std > 0 else float("nan"))
    eq_with_base = pd.concat([pd.Series([1.0]), equity], ignore_index=True)
    mdd = float((eq_with_base / eq_with_base.cummax() - 1.0).min())
    return {
        "total_return": float(equity.iloc[-1] - 1.0),
        "cagr": cagr,
        "ann_vol": std * np.sqrt(ppy) if np.isfinite(std) else float("nan"),
        "sharpe": sharpe,
        "max_drawdown": mdd,
        "win_rate": float((r > 0).mean()),
        "best_month": float(r.max()),
        "worst_month": float(r.min()),
        "avg_month": float(r.mean()),
        "n_periods": n,
    }


def quintile_returns(
    comp_scores: dict[str, pd.Series],
    fwd_1m: dict[str, pd.Series],
    dates: list[str],
    n_q: int = 5,
    min_names: int = 25,
) -> pd.DataFrame:
    """Per-date mean forward return of each composite-score quintile.

    Equal-weight buckets cut on score rank (Q1 = lowest score, Qn = highest).
    Returns a date-indexed frame with columns ``q1..qn`` plus ``long_short``
    (Qn − Q1) — the building block for both the quintile profile and the
    long/long-short portfolio stats.
    """
    rows: dict[str, dict] = {}
    for d in sorted(dates):
        if d not in comp_scores or d not in fwd_1m:
            continue
        df = pd.DataFrame({"s": comp_scores[d], "f": fwd_1m[d]}).dropna()
        if len(df) < max(min_names, n_q * 2) or df["s"].nunique() < n_q:
            continue
        ranks = df["s"].rank(method="first")
        try:
            buckets = pd.qcut(ranks, n_q, labels=False)
        except ValueError:
            continue
        means = df["f"].groupby(buckets).mean().reindex(range(n_q))
        if means.isna().any():
            continue
        row = {f"q{i + 1}": float(means.iloc[i]) for i in range(n_q)}
        row["long_short"] = float(means.iloc[-1] - means.iloc[0])
        rows[d] = row
    out = pd.DataFrame.from_dict(rows, orient="index")
    return out.sort_index()


def backtest_composite(
    comp_scores: dict[str, pd.Series],
    fwd_1m: dict[str, pd.Series],
    dates: list[str],
    n_q: int = 5,
) -> dict[str, object]:
    """Quintile profile + top-quintile-long and long-short portfolio stats."""
    qr = quintile_returns(comp_scores, fwd_1m, dates, n_q=n_q)
    if qr.empty:
        return {"quintile_means": pd.Series(dtype=float), "long": {}, "long_short": {},
                "long_series": pd.Series(dtype=float), "ls_series": pd.Series(dtype=float)}
    qcols = [f"q{i + 1}" for i in range(n_q)]
    long_series = qr[f"q{n_q}"]               # hold the top quintile each month
    ls_series = qr["long_short"]              # long top / short bottom
    return {
        "quintile_means": qr[qcols].mean(),   # avg per-bucket monthly return
        "spread_q5_q1": float(qr[qcols].mean().iloc[-1] - qr[qcols].mean().iloc[0]),
        "long": portfolio_stats(long_series),
        "long_short": portfolio_stats(ls_series),
        "long_series": long_series,
        "ls_series": ls_series,
    }


def as_signal_panel(
    comp_by_name: dict[str, dict[str, pd.Series]], dates: list[str]
) -> SimpleNamespace:
    """Wrap composite score dicts as a panel-like object for quintiles/IC.

    ``comp_by_name`` maps a composite label (e.g. "oof") to its ``{date: Series}``.
    Returns an object exposing ``.scores[date]`` (ticker x label frame) and
    ``.rebal_dates`` — the only surface :mod:`research.quintiles`/:mod:`research.ic`
    touch.
    """
    names = list(comp_by_name)
    scores: dict[str, pd.DataFrame] = {}
    for d in dates:
        cols = {n: comp_by_name[n][d] for n in names if d in comp_by_name[n]}
        if cols:
            scores[d] = pd.DataFrame(cols)
    return SimpleNamespace(scores=scores, rebal_dates=list(scores.keys()))
