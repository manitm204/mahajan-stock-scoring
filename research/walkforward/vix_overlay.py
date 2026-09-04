"""VIX-aware parent-weighting overlay study.

Three overlay variants tested against the rolling-5y baseline in a strict PIT
semiannual walk-forward (2017–2026):

  baseline       — standard rolling-5y V4 parent weights (no VIX awareness)
  overlay_full   — weights ∝ max(0, IC_in_regime); zeros negative-IC parents
  overlay_cons   — baseline + conservative ±5% tilt toward overlay_full weights
  overlay_pos    — only parents with positive IC AND positive Q5-Q1 in the regime;
                   redistributes weights by IC magnitude, capped at PARENT_CAP

At each test rebalance date the spot VIX selects the regime weight vector.
All regime characteristics are derived exclusively from training data — no look-ahead.
If a VIX regime has < MIN_REGIME_PERIODS training dates, that overlay falls back
to the baseline weights for that regime.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from backtesting.data_loader import SPY
from factors.composite import NEUTRAL, _normalize_parents
from factors.utils import sector_percentile
from research import HORIZON_MONTHS, compute_forward_returns
from research.ic import period_ic
from research.panel import ScorePanel

from . import analysis
from . import portfolio as pf
from .compose import FrozenConfig, build_parent_panel, ic_ir_weights
from .selection import select_config, slice_panel
from .splits import semiannual_policy_splits, WalkForwardSplit
from .vix_regime_study import REGIME_ORDER, _vix_spot, vix_regime_label

OVERLAY_TYPES = ["baseline", "overlay_full", "overlay_cons", "overlay_pos"]
CONSERVATIVE_CAP = 0.05   # max absolute tilt from baseline per parent
PARENT_CAP = 0.25
MIN_REGIME_PERIODS = 2    # fall back to baseline if fewer training dates in regime
TOP_PCT = 0.20
IC_HORIZON = "6M"


# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #
@dataclass
class RegimeWeightSet:
    """Regime-specific weight vectors for all overlay types (one per training window)."""
    regime_ic: dict[str, dict[str, float]]    # {regime: {parent: mean_ic}}
    regime_q5q1: dict[str, dict[str, float]]  # {regime: {parent: spread_ann}}
    regime_count: dict[str, int]              # {regime: n_training_dates}
    baseline: dict[str, float]
    full_tilt: dict[str, dict[str, float]]    # {regime: parent_weights}
    conservative: dict[str, dict[str, float]]
    positive_only: dict[str, dict[str, float]]

    def weights_for(self, overlay: str, regime: str) -> dict[str, float]:
        if overlay == "baseline":
            return self.baseline
        if overlay == "overlay_full":
            return self.full_tilt.get(regime, self.baseline)
        if overlay == "overlay_cons":
            return self.conservative.get(regime, self.baseline)
        if overlay == "overlay_pos":
            return self.positive_only.get(regime, self.baseline)
        return self.baseline


@dataclass
class OverlayWindowResult:
    label: str
    split: WalkForwardSplit
    rws: RegimeWeightSet
    # {overlay: mean_ic_6m}
    ic_6m: dict[str, float] = field(default_factory=dict)
    # {overlay: q5q1_spread_ann}
    q5q1_ann: dict[str, float] = field(default_factory=dict)
    # {overlay: portfolio_metrics_dict}
    port_metrics: dict[str, dict] = field(default_factory=dict)
    # {date: regime} for each test rebalance
    regime_by_date: dict[str, str] = field(default_factory=dict)
    # {overlay: {parent: mean_weight over test rebals}}
    avg_weights: dict[str, dict[str, float]] = field(default_factory=dict)


@dataclass
class OverlayRun:
    windows: list[OverlayWindowResult] = field(default_factory=list)
    matrix: pd.DataFrame = field(default_factory=pd.DataFrame)
    sectors: pd.Series = field(default_factory=lambda: pd.Series(dtype=object))
    vix_series: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))


# --------------------------------------------------------------------------- #
# Weight-derivation helpers
# --------------------------------------------------------------------------- #
def _water_fill(w: dict[str, float], cap: float) -> dict[str, float]:
    w = {p: max(0.0, v) for p, v in w.items()}
    total = sum(w.values())
    if total <= 0:
        return w
    w = {p: v / total for p, v in w.items()}
    for _ in range(20):
        over = [p for p, x in w.items() if x > cap + 1e-12]
        if not over:
            break
        excess = sum(w[p] - cap for p in over)
        for p in over:
            w[p] = cap
        under = [p for p, x in w.items() if x < cap - 1e-12 and x > 0]
        pool = sum(w[p] for p in under)
        if not under or pool <= 1e-12:
            break
        for p in under:
            w[p] += excess * w[p] / pool
    tot = sum(w.values())
    return {p: w[p] / tot for p in w if w[p] > 1e-12} if tot > 0 else {}


def _full_tilt(regime_ic: dict[str, float], baseline: dict[str, float]) -> dict[str, float]:
    """Weights ∝ max(0, IC); negative-IC parents get zero.  Falls back to baseline if flat."""
    pos = {p: max(0.0, ic) for p, ic in regime_ic.items()}
    if sum(pos.values()) <= 0:
        return baseline
    all_p = set(baseline) | set(pos)
    raw = {p: pos.get(p, 0.0) for p in all_p}
    return _water_fill(raw, PARENT_CAP)


def _conservative(full: dict[str, float], baseline: dict[str, float]) -> dict[str, float]:
    """Clip each parent's change to CONSERVATIVE_CAP; renormalize."""
    all_p = set(baseline) | set(full)
    w = {}
    for p in all_p:
        t = full.get(p, 0.0)
        b = baseline.get(p, 0.0)
        w[p] = float(np.clip(t, b - CONSERVATIVE_CAP, b + CONSERVATIVE_CAP))
        w[p] = max(0.0, min(PARENT_CAP, w[p]))
    tot = sum(w.values())
    if tot <= 0:
        return baseline
    return {p: v / tot for p, v in w.items() if v > 1e-12}


def _positive_only(regime_ic: dict[str, float], regime_q5q1: dict[str, float],
                   baseline: dict[str, float]) -> dict[str, float]:
    """Only parents with positive IC AND positive Q5-Q1; redistribute by IC."""
    eligible = {p: ic for p, ic in regime_ic.items()
                if ic > 0 and regime_q5q1.get(p, float("nan")) > 0}
    if not eligible:
        return baseline
    return _water_fill(eligible, PARENT_CAP)


# --------------------------------------------------------------------------- #
# Training-regime analytics
# --------------------------------------------------------------------------- #
def _train_regime_analytics(
    parent_panel: ScorePanel,
    train_rebals: list[str],
    train_fwd: dict[str, dict[str, pd.Series]],
    vix_series: pd.Series,
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]], dict[str, int]]:
    """Per-parent IC and Q5-Q1 spread in each VIX regime, using training data only.

    Returns (regime_ic, regime_q5q1, regime_count) where each is keyed by regime label.
    """
    fwd_h = train_fwd.get(IC_HORIZON, {})
    # Collect per-parent IC observations grouped by regime
    regime_ic_lists: dict[str, dict[str, list[float]]] = {r: {} for r in REGIME_ORDER}
    # Collect per-parent score dicts grouped by regime (for Q5-Q1)
    regime_scores: dict[str, dict[str, dict[str, pd.Series]]] = {r: {} for r in REGIME_ORDER}
    regime_count: dict[str, int] = {r: 0 for r in REGIME_ORDER}

    for d in train_rebals:
        if d not in fwd_h:
            continue
        vix_val = _vix_spot(vix_series, d)
        regime = vix_regime_label(vix_val)
        regime_count[regime] = regime_count.get(regime, 0) + 1
        frame = parent_panel.scores.get(d)
        if frame is None:
            continue
        for parent in parent_panel.parent_keys:
            pscore = frame.get(parent)
            if pscore is None:
                continue
            ic = period_ic(pscore, fwd_h[d], min_names=20)
            if ic is not None:
                regime_ic_lists[regime].setdefault(parent, []).append(ic)
            regime_scores[regime].setdefault(parent, {})[d] = pscore

    # Average IC per regime
    regime_ic: dict[str, dict[str, float]] = {}
    for regime, by_parent in regime_ic_lists.items():
        if by_parent:
            regime_ic[regime] = {p: float(np.mean(ics)) for p, ics in by_parent.items()}

    # Q5-Q1 per regime (only if >= MIN_REGIME_PERIODS)
    regime_q5q1: dict[str, dict[str, float]] = {}
    for regime, by_parent in regime_scores.items():
        if regime_count.get(regime, 0) < MIN_REGIME_PERIODS:
            continue
        q5q1_row: dict[str, float] = {}
        for parent, by_date in by_parent.items():
            qres = analysis.quantile_analysis(by_date, train_fwd)
            spr = qres.get(IC_HORIZON, {}).get("spread", {})
            q5q1_row[parent] = float(spr.get("annualized", float("nan")))
        regime_q5q1[regime] = q5q1_row

    return regime_ic, regime_q5q1, regime_count


def _build_regime_weight_set(
    regime_ic: dict[str, dict[str, float]],
    regime_q5q1: dict[str, dict[str, float]],
    regime_count: dict[str, int],
    baseline: dict[str, float],
) -> RegimeWeightSet:
    """Derive all 3 overlay weight dicts for all regimes from training analytics."""
    full_tilt: dict[str, dict[str, float]] = {}
    conservative: dict[str, dict[str, float]] = {}
    positive_only: dict[str, dict[str, float]] = {}

    for regime in REGIME_ORDER:
        ric = regime_ic.get(regime)
        rq5 = regime_q5q1.get(regime, {})
        n = regime_count.get(regime, 0)
        if ric is None or n < MIN_REGIME_PERIODS:
            # Fall back to baseline for thin regimes
            full_tilt[regime] = baseline
            conservative[regime] = baseline
            positive_only[regime] = baseline
        else:
            ft = _full_tilt(ric, baseline)
            full_tilt[regime] = ft
            conservative[regime] = _conservative(ft, baseline)
            positive_only[regime] = _positive_only(ric, rq5, baseline)

    return RegimeWeightSet(
        regime_ic=regime_ic, regime_q5q1=regime_q5q1, regime_count=regime_count,
        baseline=baseline, full_tilt=full_tilt, conservative=conservative,
        positive_only=positive_only,
    )


# --------------------------------------------------------------------------- #
# Composite construction with per-date weights
# --------------------------------------------------------------------------- #
def _composite_at(frame: pd.DataFrame, pweights: dict[str, float],
                  sectors: pd.Series) -> pd.Series:
    cols = [p for p in pweights if p in frame.columns]
    if not cols:
        return pd.Series(dtype=float)
    blend = _normalize_parents(frame[cols], "zscore", 20.0)
    comp_raw = pd.Series(0.0, index=frame.index)
    used = 0.0
    for key in cols:
        comp_raw += pweights[key] * blend[key].fillna(NEUTRAL)
        used += pweights[key]
    if used > 0:
        comp_raw /= used
    secs = sectors.reindex(frame.index).fillna("Unknown")
    return sector_percentile(comp_raw, secs, higher_is_better=True, min_obs=5)


def _build_overlay_scores(
    parent_panel: ScorePanel,
    test_rebals: list[str],
    rws: RegimeWeightSet,
    vix_series: pd.Series,
    sectors: pd.Series,
) -> tuple[dict[str, dict[str, pd.Series]], dict[str, str], dict[str, dict[str, list[float]]]]:
    """Build composite scores for all overlays; return (scores_by_overlay, regime_by_date,
    weight_accumulator[overlay][parent] = list of per-date weights for averaging later)."""
    scores: dict[str, dict[str, pd.Series]] = {ot: {} for ot in OVERLAY_TYPES}
    regime_by_date: dict[str, str] = {}
    w_acc: dict[str, dict[str, list[float]]] = {ot: {} for ot in OVERLAY_TYPES}

    for d in test_rebals:
        frame = parent_panel.scores.get(d)
        if frame is None:
            continue
        vix_val = _vix_spot(vix_series, d)
        regime = vix_regime_label(vix_val)
        regime_by_date[d] = regime

        for ot in OVERLAY_TYPES:
            pw = rws.weights_for(ot, regime)
            scores[ot][d] = _composite_at(frame, pw, sectors)
            for p, wv in pw.items():
                w_acc[ot].setdefault(p, []).append(wv)

    return scores, regime_by_date, w_acc


# --------------------------------------------------------------------------- #
# Main runner
# --------------------------------------------------------------------------- #
def run_overlay_study(
    panel: ScorePanel, matrix: pd.DataFrame, sectors: pd.Series,
    vix_series: pd.Series, *,
    first_test_year: int = 2017, last_end: str | None = None,
    verbose: bool = True,
) -> OverlayRun:
    """Rolling-5y semiannual walk-forward with 4 composite variants per window."""
    kwargs: dict = {"first_test_year": first_test_year}
    if last_end is not None:
        kwargs["last_end"] = last_end
    splits = semiannual_policy_splits("rolling5y", **kwargs)
    run = OverlayRun(matrix=matrix, sectors=sectors, vix_series=vix_series)

    for sp in splits:
        train_rebals = sp.train_rebalances(panel.rebal_dates)
        test_rebals = sp.test_rebalances(panel.rebal_dates)
        if not train_rebals or not test_rebals:
            if verbose:
                print(f"  skip {sp.label}: train={len(train_rebals)} test={len(test_rebals)}")
            continue
        if verbose:
            print(f"  {sp.label}: train {len(train_rebals)} ({train_rebals[0]}→{train_rebals[-1]})"
                  f", test {len(test_rebals)} ({test_rebals[0]}→{test_rebals[-1]})")

        # Baseline config (sub-weights + baseline parent weights)
        cfg = select_config(panel, train_rebals, matrix, boundary=sp.test_start)
        baseline_w = dict(cfg.parent_weights)

        # Training forward returns (price matrix truncated at test boundary)
        train_px = matrix.loc[matrix.index <= sp.test_start]
        train_fwd = compute_forward_returns(train_px, train_rebals, HORIZON_MONTHS)

        # Parent panel on training dates → regime analytics
        train_sub = slice_panel(panel, train_rebals)
        parent_panel_train = build_parent_panel(train_sub, cfg.sub_weights)
        regime_ic, regime_q5q1, regime_count = _train_regime_analytics(
            parent_panel_train, train_rebals, train_fwd, vix_series)
        rws = _build_regime_weight_set(regime_ic, regime_q5q1, regime_count, baseline_w)

        if verbose:
            counts = {r: regime_count.get(r, 0) for r in REGIME_ORDER}
            print(f"    regime counts in train: {counts}")

        # Parent panel on test dates → overlay composites
        test_sub = slice_panel(panel, test_rebals)
        parent_panel_test = build_parent_panel(test_sub, cfg.sub_weights)
        ov_scores, regime_by_date, w_acc = _build_overlay_scores(
            parent_panel_test, test_rebals, rws, vix_series, sectors)

        # Metrics for each overlay
        test_fwd = compute_forward_returns(matrix, list(test_rebals), HORIZON_MONTHS)
        ic_6m: dict[str, float] = {}
        q5q1_ann: dict[str, float] = {}
        port_metrics: dict[str, dict] = {}

        for ot in OVERLAY_TYPES:
            sc = ov_scores[ot]
            if not sc:
                continue
            ic_df = analysis.composite_ic(sc, test_fwd)
            r = ic_df[ic_df["horizon"] == IC_HORIZON]
            ic_6m[ot] = float(r.iloc[0]["mean_ic"]) if not r.empty else float("nan")
            qres = analysis.quantile_analysis(sc, test_fwd)
            q6 = qres.get(IC_HORIZON, {})
            q5q1_ann[ot] = float(q6.get("spread", {}).get("annualized", float("nan")))
            book = pf.simulate(sc, matrix, sectors, top_pct=TOP_PCT,
                               mode="equal", hold_months=1)
            port_metrics[ot] = book.metrics

        avg_weights = {ot: {p: float(np.mean(vs)) for p, vs in w_acc[ot].items()}
                       for ot in OVERLAY_TYPES}
        label_root = sp.label.split(":", 1)[1]
        run.windows.append(OverlayWindowResult(
            label=label_root, split=sp, rws=rws,
            ic_6m=ic_6m, q5q1_ann=q5q1_ann, port_metrics=port_metrics,
            regime_by_date=regime_by_date, avg_weights=avg_weights,
        ))
    return run


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
METRIC_COLS = ["cagr", "sharpe", "sortino", "max_drawdown", "spy_excess_cagr",
               "spy_ir", "spy_beta", "spy_alpha", "avg_turnover"]


def build_overlay_summary(run: OverlayRun) -> dict[str, pd.DataFrame]:
    """Build per-window, full-pooled, and regime-weight tables.

    Keys: ``per_window``, ``full_summary``, ``regime_weights``
    """
    per_window_rows: list[dict] = []
    all_scores: dict[str, dict[str, pd.Series]] = {ot: {} for ot in OVERLAY_TYPES}
    # For regime weight heatmap: {overlay: {parent: {regime: [weights]}}}
    regime_w_acc: dict[str, dict[str, dict[str, list[float]]]] = {ot: {} for ot in OVERLAY_TYPES}

    for w in run.windows:
        for ot in OVERLAY_TYPES:
            m = w.port_metrics.get(ot, {})
            per_window_rows.append({
                "window": w.label,
                "overlay": ot,
                "ic_6m": w.ic_6m.get(ot, float("nan")),
                "q5q1_ann": w.q5q1_ann.get(ot, float("nan")),
                **{k: m.get(k) for k in METRIC_COLS},
            })
            # Accumulate composite scores for full-period pooling
            # (we no longer have raw scores, but can note per-window IC)

        # Regime-specific weight accumulation
        for d, regime in w.regime_by_date.items():
            for ot in OVERLAY_TYPES:
                pw = w.rws.weights_for(ot, regime)
                for parent, wv in pw.items():
                    (regime_w_acc[ot]
                     .setdefault(parent, {})
                     .setdefault(regime, [])
                     .append(wv))

    per_window_df = pd.DataFrame(per_window_rows)

    # Full-period summary (mean of per-window means)
    full_rows: list[dict] = []
    for ot in OVERLAY_TYPES:
        sub = per_window_df[per_window_df["overlay"] == ot]
        row: dict = {"overlay": ot, "n_windows": len(sub)}
        for col in ["ic_6m", "q5q1_ann"] + METRIC_COLS:
            vals = sub[col].dropna() if col in sub.columns else pd.Series(dtype=float)
            row[col] = float(vals.mean()) if not vals.empty else float("nan")
        full_rows.append(row)
    full_summary = pd.DataFrame(full_rows)

    # Regime weight heatmap DataFrames: one per overlay
    regime_weight_frames: dict[str, pd.DataFrame] = {}
    for ot in OVERLAY_TYPES:
        parents = sorted(regime_w_acc.get(ot, {}))
        data: dict[str, dict[str, float]] = {}
        for p in parents:
            data[p] = {r: float(np.mean(regime_w_acc[ot][p].get(r, [float("nan")])))
                       for r in REGIME_ORDER}
        df = pd.DataFrame(data).T
        df.index.name = "parent"
        regime_weight_frames[ot] = df[[r for r in REGIME_ORDER if r in df.columns]]

    return {
        "per_window": per_window_df,
        "full_summary": full_summary,
        "regime_weights": regime_weight_frames,
    }
