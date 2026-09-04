"""Semiannual (6-month) walk-forward regime study — 5y vs 2y vs hybrid training.

Runs a strict PIT walk-forward with **6-month test windows** (H1/H2 each year) from 2017-H1
through the latest available half-year, rebuilding the entire Parent-Selection-V4
construction on train-only data for each window under four policies:

* ``rolling5y`` — trailing 5-year training window
* ``rolling2y`` — trailing 2-year training window
* ``blend_comp`` — 50/50 average of the 5y and 2y composite scores on each test date
* ``blend_parent`` — 50/50 average of the 5y and 2y frozen configs (parent+sub weights)

For each policy each window yields: frozen config (5y/2y) or blended, composite scores on
test dates, per-window OOS composite IC, portfolio simulations (top-10/20/30 %, monthly),
and per-parent frozen scores (used for parent-weight and parent-IC evolution charts).

The study surfaces results at three levels: per-window, 3-year buckets (2017-2019,
2020-2022, 2023-2025, 2026-partial), and full-sample pooled — plus per-policy summary
tables ready for reporting. Nothing here mutates production; it is a read-only research
extension of :mod:`research.walkforward`.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from research import HORIZON_MONTHS, compute_forward_returns
from research.panel import ScorePanel

from . import analysis
from . import portfolio as pf
from .compose import FrozenConfig, build_parent_panel, frozen_composite
from .hybrid import average_composites, blend_parent_weights
from .selection import select_config, slice_panel
from .splits import (SEMI_TRAIN_POLICIES, WalkForwardSplit, semiannual_labels,
                     semiannual_policy_splits)

POLICIES = ["rolling5y", "rolling2y", "blend_comp", "blend_parent"]
STUDY_HORIZONS = ("1M", "3M", "6M")


# --------------------------------------------------------------------------- #
# Per-window artefacts
# --------------------------------------------------------------------------- #
@dataclass
class WindowResult:
    """One (policy, semiannual test window) row of frozen artefacts + measured OOS.

    ``label`` is the H1/H2 window key ("2020-H1"); ``policy`` is one of :data:`POLICIES`.
    ``config`` is the frozen (or blended) FrozenConfig used; ``scores`` is the composite
    score dict on the test rebalances. ``ic`` is per-horizon composite IC; ``portfolios``
    holds a top-20 % equal-weight :class:`SimResult` (used for equity curves).
    """

    label: str
    policy: str
    split: WalkForwardSplit
    config: FrozenConfig
    scores: dict[str, pd.Series]
    ic: pd.DataFrame
    portfolios: dict[float, pf.SimResult]           # {top_pct: SimResult}
    n_train: int
    n_test: int


@dataclass
class RegimeRun:
    """Full study output: per-policy artefacts + shared inputs for downstream reports."""

    windows: dict[str, list[WindowResult]] = field(default_factory=dict)  # policy → list
    parent_scores: dict[str, dict[str, dict[str, pd.Series]]] = field(default_factory=dict)
    #     policy → parent → date → Series[ticker → parent score]
    matrix: pd.DataFrame = field(default_factory=pd.DataFrame)
    sectors: pd.Series = field(default_factory=lambda: pd.Series(dtype=object))
    labels: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #
def _apply_config(cfg: FrozenConfig, panel: ScorePanel, test_rebals: list[str],
                  sectors: pd.Series) -> dict[str, pd.Series]:
    return frozen_composite(panel, test_rebals, cfg, sectors)


def _simulate_book(scores: dict[str, pd.Series], matrix: pd.DataFrame,
                   sectors: pd.Series, top_pcts=(0.10, 0.20, 0.30)) -> dict[float, pf.SimResult]:
    return {tp: pf.simulate(scores, matrix, sectors, top_pct=tp, mode="equal",
                            hold_months=1) for tp in top_pcts}


def _stash_parent_scores(parent_panel: ScorePanel,
                         bucket: dict[str, dict[str, pd.Series]]) -> None:
    for parent in parent_panel.parent_keys:
        pbucket = bucket.setdefault(parent, {})
        for d in parent_panel.rebal_dates:
            col = parent_panel.scores[d].get(parent)
            if col is not None:
                pbucket[d] = col


def run_regime_study(panel: ScorePanel, matrix: pd.DataFrame, sectors: pd.Series,
                     *, first_test_year: int = 2017,
                     last_end: str | None = None,
                     top_pcts=(0.10, 0.20, 0.30), verbose: bool = True) -> RegimeRun:
    """Loop the four policies × semiannual grid; return a :class:`RegimeRun`."""
    kwargs = {"first_test_year": first_test_year}
    if last_end is not None:
        kwargs["last_end"] = last_end

    splits_by_policy = {p: semiannual_policy_splits(p, **kwargs)
                        for p in SEMI_TRAIN_POLICIES}
    labels = semiannual_labels(first_test_year=first_test_year,
                               **({"last_end": last_end} if last_end else {}))
    by_label: dict[str, dict[str, WalkForwardSplit]] = {lbl: {} for lbl in labels}
    for pol, splits in splits_by_policy.items():
        for sp in splits:
            root = sp.label.split(":", 1)[1]
            by_label[root][pol] = sp

    run = RegimeRun(matrix=matrix, sectors=sectors, labels=labels,
                    windows={p: [] for p in POLICIES},
                    parent_scores={p: {} for p in POLICIES})

    for lbl in labels:
        pols = by_label.get(lbl, {})
        if not pols:
            continue
        # Both trainings must fit (they share the test window definition).
        cfgs: dict[str, FrozenConfig] = {}
        scores: dict[str, dict[str, pd.Series]] = {}
        splits_for_lbl: dict[str, WalkForwardSplit] = {}
        n_train: dict[str, int] = {}
        n_test = 0
        skip = False
        test_rebals: list[str] = []
        for pol, sp in pols.items():
            train_rebals = sp.train_rebalances(panel.rebal_dates)
            test_rebals = sp.test_rebalances(panel.rebal_dates)
            if not train_rebals or not test_rebals:
                if verbose:
                    print(f"  skip {pol}:{lbl}: train={len(train_rebals)} "
                          f"test={len(test_rebals)}")
                skip = True
                break
            if verbose:
                print(f"  {pol}:{lbl}: select on {len(train_rebals)} "
                      f"({train_rebals[0]}→{train_rebals[-1]}), test {len(test_rebals)} "
                      f"({test_rebals[0]}→{test_rebals[-1]})")
            cfg = select_config(panel, train_rebals, matrix, boundary=sp.test_start)
            cfgs[pol] = cfg
            scores[pol] = _apply_config(cfg, panel, test_rebals, sectors)
            splits_for_lbl[pol] = sp
            n_train[pol] = len(train_rebals)
            n_test = len(test_rebals)
        if skip:
            continue

        # Blended composites — only need one of the splits (test window is identical).
        base_split = splits_for_lbl["rolling5y"]
        blend_cfg = blend_parent_weights(cfgs["rolling5y"], cfgs["rolling2y"], w=0.5)
        scores_blend_parent = _apply_config(blend_cfg, panel, test_rebals, sectors)
        scores_blend_comp = average_composites(scores["rolling5y"], scores["rolling2y"],
                                               w=0.5)

        emit = {
            "rolling5y": (cfgs["rolling5y"], scores["rolling5y"], splits_for_lbl["rolling5y"],
                          n_train["rolling5y"]),
            "rolling2y": (cfgs["rolling2y"], scores["rolling2y"], splits_for_lbl["rolling2y"],
                          n_train["rolling2y"]),
            "blend_comp": (blend_cfg, scores_blend_comp, base_split, n_train["rolling5y"]),
            "blend_parent": (blend_cfg, scores_blend_parent, base_split,
                             n_train["rolling5y"]),
        }
        for pol, (cfg, sc, sp, ntr) in emit.items():
            fwd = compute_forward_returns(matrix, list(sc), HORIZON_MONTHS)
            ic = analysis.composite_ic(sc, fwd)
            books = _simulate_book(sc, matrix, sectors, top_pcts=top_pcts)
            run.windows[pol].append(WindowResult(
                label=lbl, policy=pol, split=sp, config=cfg, scores=sc, ic=ic,
                portfolios=books, n_train=ntr, n_test=n_test,
            ))
            parent_panel = build_parent_panel(slice_panel(panel, list(sc)),
                                              cfg.sub_weights)
            _stash_parent_scores(parent_panel, run.parent_scores[pol])
    return run


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
BUCKET_DEFS: list[tuple[str, int, int]] = [
    ("2017-2019", 2017, 2019),
    ("2020-2022", 2020, 2022),
    ("2023-2025", 2023, 2025),
    ("2026+",     2026, 2099),
]


def _label_year(label: str) -> int:
    return int(label.split("-", 1)[0])


def _bucket_for(label: str) -> str:
    y = _label_year(label)
    for name, lo, hi in BUCKET_DEFS:
        if lo <= y <= hi:
            return name
    return "other"


def _pool_scores(windows: list[WindowResult]) -> dict[str, pd.Series]:
    pooled: dict[str, pd.Series] = {}
    for w in windows:
        pooled.update(w.scores)
    return pooled


def _pool_portfolio_returns(windows: list[WindowResult],
                            top_pct: float) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Concatenate per-window portfolio period-return / SPY-return / turnover series."""
    from backtesting.data_loader import SPY
    ret_parts, spy_parts, turn_parts = [], [], []
    for w in windows:
        book = w.portfolios.get(top_pct)
        if book is None:
            continue
        ret_parts.append(book.period_returns)
        spy_parts.append(book.bench_period_returns.get(SPY, pd.Series(dtype=float)))
        turn_parts.append(book.turnover)
    if not ret_parts:
        return (pd.Series(dtype=float), pd.Series(dtype=float), pd.Series(dtype=float))
    return (pd.concat(ret_parts).sort_index(),
            pd.concat(spy_parts).sort_index(),
            pd.concat(turn_parts).sort_index())


def _ic_summary_from_windows(windows: list[WindowResult]) -> pd.DataFrame:
    """Combine per-window per-horizon IC rows into pooled means (across window-mean ICs)."""
    if not windows:
        return pd.DataFrame()
    frames = []
    for w in windows:
        if w.ic is not None and not w.ic.empty:
            f = w.ic.copy()
            f["label"] = w.label
            frames.append(f)
    if not frames:
        return pd.DataFrame()
    cat = pd.concat(frames, ignore_index=True)
    rows: list[dict] = []
    for h, g in cat.groupby("horizon"):
        g = g.dropna(subset=["mean_ic"])
        if g.empty:
            continue
        m = float(g["mean_ic"].mean())
        std = float(g["mean_ic"].std(ddof=1)) if len(g) > 1 else np.nan
        rows.append({
            "horizon": h, "n_windows": int(len(g)),
            "mean_ic": m,
            "ic_std_across_windows": std,
            "ir_across_windows": (m / std) if std and std > 1e-9 else np.nan,
            "hit_rate_windows": float((g["mean_ic"] > 0).mean()),
        })
    return pd.DataFrame(rows).sort_values("horizon").reset_index(drop=True)


def _quantile_summary(pool: dict[str, pd.Series], matrix: pd.DataFrame) -> dict[str, dict]:
    if not pool:
        return {}
    fwd = compute_forward_returns(matrix, list(pool), HORIZON_MONTHS)
    return analysis.quantile_analysis(pool, fwd)


def _portfolio_metrics_row(returns: pd.Series, spy: pd.Series, turnover: pd.Series,
                           label: str, policy: str, top_pct: float,
                           hold_months: int = 1) -> dict:
    from backtesting.data_loader import SPY
    metrics = pf.performance_metrics(returns, hold_months, turnover,
                                     benchmarks={SPY: spy})
    metrics.update({"grouping": label, "policy": policy,
                    "top_pct": top_pct, "n_periods": int(returns.dropna().size)})
    return metrics


def _windows_in(windows: list[WindowResult], predicate) -> list[WindowResult]:
    return [w for w in windows if predicate(w)]


def build_summaries(run: RegimeRun,
                    top_pcts=(0.10, 0.20, 0.30)) -> dict[str, pd.DataFrame]:
    """Return the three-level aggregate tables plus per-window portfolio rows.

    Keys of the return dict:
    * ``per_window_ic`` — per (policy, window, horizon) composite IC row
    * ``per_window_portfolio`` — per (policy, window, top_pct) portfolio metrics
    * ``bucket_ic`` — per (policy, bucket, horizon) pooled IC (pool by window means)
    * ``bucket_portfolio`` — per (policy, bucket, top_pct) portfolio metrics
    * ``bucket_quantile`` — per (policy, bucket, horizon) Q5-Q1 spread + monotonicity
    * ``full_ic`` — per (policy, horizon) pooled IC across all windows
    * ``full_portfolio`` — per (policy, top_pct) full-sample portfolio metrics
    * ``full_quantile`` — per (policy, horizon) full-sample Q5-Q1 spread + monotonicity
    """
    from backtesting.data_loader import SPY

    per_window_ic: list[dict] = []
    per_window_portfolio: list[dict] = []
    bucket_ic: list[dict] = []
    bucket_portfolio: list[dict] = []
    bucket_quantile: list[dict] = []
    full_ic: list[dict] = []
    full_portfolio: list[dict] = []
    full_quantile: list[dict] = []

    for policy, windows in run.windows.items():
        if not windows:
            continue

        for w in windows:
            if w.ic is None or w.ic.empty:
                continue
            for _, row in w.ic.iterrows():
                per_window_ic.append({
                    "policy": policy, "window": w.label,
                    "horizon": row["horizon"],
                    "n_periods": int(row.get("n_periods", 0) or 0),
                    "mean_ic": float(row.get("mean_ic", np.nan)),
                    "ir": float(row.get("information_ratio", np.nan)),
                    "hit_rate": float(row.get("hit_rate", np.nan)),
                    "t_stat": float(row.get("t_stat", np.nan)),
                })
            for tp, book in w.portfolios.items():
                m = book.metrics
                per_window_portfolio.append({
                    "policy": policy, "window": w.label,
                    "top_pct": tp, "hold_months": 1,
                    **{k: m.get(k) for k in (
                        "n_periods", "total_return", "cagr", "sharpe", "sortino",
                        "ann_vol", "max_drawdown", "avg_turnover", "hit_rate",
                        "spy_cagr", "spy_excess_cagr", "spy_alpha", "spy_beta",
                        "spy_ir", "spy_te", "spy_rel_max_drawdown",
                    )},
                })

        # 3-year buckets
        for bucket_name, lo, hi in BUCKET_DEFS:
            wsub = _windows_in(windows, lambda w: lo <= _label_year(w.label) <= hi)
            if not wsub:
                continue
            ic_tbl = _ic_summary_from_windows(wsub)
            for _, r in ic_tbl.iterrows():
                bucket_ic.append({"policy": policy, "bucket": bucket_name, **r.to_dict()})
            for tp in top_pcts:
                ret, spy, turn = _pool_portfolio_returns(wsub, tp)
                if ret.empty:
                    continue
                bucket_portfolio.append(_portfolio_metrics_row(
                    ret, spy, turn, bucket_name, policy, tp))
            pool = _pool_scores(wsub)
            qres = _quantile_summary(pool, run.matrix)
            for h, q in qres.items():
                sp = q.get("spread") or {}
                bucket_quantile.append({
                    "policy": policy, "bucket": bucket_name, "horizon": h,
                    "n_periods": q.get("n_periods"),
                    "monotonic_rate": q.get("monotonic_rate"),
                    "profile_spearman": q.get("profile_spearman"),
                    "spread_avg": sp.get("avg"),
                    "spread_annualized": sp.get("annualized"),
                    "spread_sharpe": sp.get("sharpe"),
                    "spread_hit_rate": sp.get("hit_rate"),
                })

        # Full pooled
        pool = _pool_scores(windows)
        fwd = compute_forward_returns(run.matrix, list(pool), HORIZON_MONTHS)
        full_ic_df = analysis.composite_ic(pool, fwd)
        for _, r in full_ic_df.iterrows():
            full_ic.append({"policy": policy, **r.to_dict()})
        for tp in top_pcts:
            ret, spy, turn = _pool_portfolio_returns(windows, tp)
            if ret.empty:
                continue
            full_portfolio.append(_portfolio_metrics_row(ret, spy, turn,
                                                        "full", policy, tp))
        qres = _quantile_summary(pool, run.matrix)
        for h, q in qres.items():
            sp = q.get("spread") or {}
            full_quantile.append({
                "policy": policy, "horizon": h,
                "n_periods": q.get("n_periods"),
                "monotonic_rate": q.get("monotonic_rate"),
                "profile_spearman": q.get("profile_spearman"),
                "spread_avg": sp.get("avg"),
                "spread_annualized": sp.get("annualized"),
                "spread_sharpe": sp.get("sharpe"),
                "spread_hit_rate": sp.get("hit_rate"),
            })

    return {
        "per_window_ic": pd.DataFrame(per_window_ic),
        "per_window_portfolio": pd.DataFrame(per_window_portfolio),
        "bucket_ic": pd.DataFrame(bucket_ic),
        "bucket_portfolio": pd.DataFrame(bucket_portfolio),
        "bucket_quantile": pd.DataFrame(bucket_quantile),
        "full_ic": pd.DataFrame(full_ic),
        "full_portfolio": pd.DataFrame(full_portfolio),
        "full_quantile": pd.DataFrame(full_quantile),
    }


# --------------------------------------------------------------------------- #
# Diagnostic tables — parent-weight and parent-IC evolution
# --------------------------------------------------------------------------- #
def parent_weight_matrix(run: RegimeRun, policy: str) -> pd.DataFrame:
    """DataFrame: rows = parents, columns = window labels, values = parent weights."""
    windows = run.windows.get(policy, [])
    parents = sorted({p for w in windows for p in w.config.parent_weights})
    data: dict[str, dict[str, float]] = {p: {} for p in parents}
    for w in windows:
        for p in parents:
            data[p][w.label] = float(w.config.parent_weights.get(p, 0.0))
    frame = pd.DataFrame(data).T
    frame.index.name = "parent"
    return frame.reindex(columns=[w.label for w in windows])


def parent_ic_matrix(run: RegimeRun, policy: str,
                     horizon: str = "6M") -> pd.DataFrame:
    """DataFrame: rows = parents, cols = window labels, values = per-window OOS parent IC."""
    from research.ic import period_ic
    windows = run.windows.get(policy, [])
    parents = sorted(run.parent_scores.get(policy, {}))
    out = pd.DataFrame(index=parents, columns=[w.label for w in windows], dtype=float)
    for w in windows:
        fwd_h = compute_forward_returns(run.matrix, list(w.scores),
                                        HORIZON_MONTHS).get(horizon, {})
        if not fwd_h:
            continue
        for p in parents:
            pscore = run.parent_scores[policy].get(p, {})
            ics = [period_ic(pscore[d], fwd, min_names=20)
                   for d, fwd in fwd_h.items() if d in pscore]
            ics = [x for x in ics if x is not None]
            if ics:
                out.loc[p, w.label] = float(np.mean(ics))
    out.index.name = "parent"
    return out


def divergence_table(run: RegimeRun) -> pd.DataFrame:
    """|Δparent_weight(5y − 2y)| L1 turnover + 6M IC gap per window (regime signal)."""
    w5 = {w.label: w for w in run.windows.get("rolling5y", [])}
    w2 = {w.label: w for w in run.windows.get("rolling2y", [])}
    labels = [lbl for lbl in run.labels if lbl in w5 and lbl in w2]
    rows: list[dict] = []
    for lbl in labels:
        a = w5[lbl].config.parent_weights
        b = w2[lbl].config.parent_weights
        keys = set(a) | set(b)
        l1 = 0.5 * sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in keys)
        ic5 = w5[lbl].ic.set_index("horizon")["mean_ic"].get("6M", np.nan) \
            if not w5[lbl].ic.empty else np.nan
        ic2 = w2[lbl].ic.set_index("horizon")["mean_ic"].get("6M", np.nan) \
            if not w2[lbl].ic.empty else np.nan
        rows.append({"window": lbl, "parent_weight_l1": l1,
                     "ic_6M_5y": float(ic5) if pd.notna(ic5) else np.nan,
                     "ic_6M_2y": float(ic2) if pd.notna(ic2) else np.nan,
                     "ic_6M_gap": (float(ic5) - float(ic2))
                                  if pd.notna(ic5) and pd.notna(ic2) else np.nan})
    return pd.DataFrame(rows)
