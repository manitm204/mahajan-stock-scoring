"""Per-window frozen baseline — built once, cached on disk, shared by every
overlay variant.

The baseline model is the question's fixed point: "does a VIX overlay improve
the *existing* rolling-5Y model?". To guarantee the baseline tested here is the
existing model (not a reimplementation that could silently differ), the
per-window sub-factor selection, intra-parent weights and baseline parent
weights are produced by ``research.walkforward.selection.select_config`` — the
validated V4 chain. That call is the single deliberate reuse of ``research/``
code in this study, confined to this module; every downstream computation
(parent scores, regime statistics, composites, portfolios, metrics) is
implemented fresh in this package.

Each :class:`FrozenWindow` carries everything an overlay variant needs, so the
expensive selection runs once per window and the variants replay from cache.
"""
from __future__ import annotations

import pickle
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .composite import composite_scores
from .data_io import CACHE_DIR, StudyPanel
from .metrics import forward_returns, quintile_spread, spearman_ic
from .windows import StudyWindow

STAT_HORIZONS = ("3M", "6M")   # the horizons the V4 model itself is built on
CACHE_VERSION = 1


@dataclass
class FrozenWindow:
    """One window's frozen baseline + the training-only inputs for overlays."""

    label: str
    train_start: str
    test_start: str
    test_end: str
    train_rebals: list[str]
    test_rebals: list[str]
    sub_weights: dict[str, dict[str, float]]
    parent_weights: dict[str, float]
    parent_scores: dict[str, pd.DataFrame]   # date → (PIT ticker × parent)
    train_stats: pd.DataFrame                # date, parent, ic, spread (train only)
    vix_train: pd.Series                     # daily VIX, dates < test_start
    vix_at_rebal: dict[str, float]           # rebal date → spot VIX (last ≤ date)


# --------------------------------------------------------------------------- #
# Point-in-time guarantees
# --------------------------------------------------------------------------- #
def spot_vix(vix: pd.Series, date: str) -> float:
    """Last VIX close on/before ``date`` — asserts the observation is not from
    the future."""
    ts = pd.Timestamp(date)
    sub = vix.loc[:ts]
    assert not sub.empty, f"no VIX observation on/before {date}"
    assert sub.index[-1] <= ts
    return float(sub.iloc[-1])


def assert_pit(win: StudyWindow, train_rebals: list[str], test_rebals: list[str],
               px_train: pd.DataFrame, vix_train: pd.Series,
               train_stats: pd.DataFrame) -> None:
    """Prove no test-period dates enter the training statistics."""
    assert train_rebals, f"{win.label}: empty training rebalances"
    assert test_rebals, f"{win.label}: empty test rebalances"
    assert max(train_rebals) <= win.rebal_cap < win.test_start, \
        f"{win.label}: training rebalance past the 6M-forward cap"
    assert min(test_rebals) >= win.test_start
    # Training forward returns can never see past the boundary: the price
    # matrix they are computed from is truncated at test_start.
    assert str(px_train.index.max()) <= win.test_start, \
        f"{win.label}: training price matrix crosses the test boundary"
    # The VIX distribution the overlay compares against is train-only.
    assert vix_train.index.max() < pd.Timestamp(win.test_start), \
        f"{win.label}: training VIX distribution crosses the test boundary"
    assert set(train_stats["date"]) <= set(train_rebals)


# --------------------------------------------------------------------------- #
# Parent scores + training regime statistics (own implementations)
# --------------------------------------------------------------------------- #
def parent_frame(frame: pd.DataFrame,
                 sub_weights: dict[str, dict[str, float]]) -> pd.DataFrame:
    """(ticker × parent) coverage-aware weighted mean of each parent's frozen
    sub-factors, on the date's PIT members."""
    cols: dict[str, pd.Series] = {}
    for parent, wmap in sub_weights.items():
        if not wmap:
            continue
        present = [s for s in wmap if s in frame.columns]
        if not present:
            cols[parent] = pd.Series(np.nan, index=frame.index)
            continue
        mat = frame[present].to_numpy(dtype=float)
        w = np.array([wmap[s] for s in present], dtype=float)
        mask = ~np.isnan(mat)
        wrow = mask * w[None, :]
        wsum = wrow.sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            comp = (np.where(mask, mat, 0.0) * wrow).sum(axis=1) / wsum
        cols[parent] = pd.Series(np.where(wsum > 0, comp, np.nan), index=frame.index)
    return pd.DataFrame(cols)


def _training_stats(parent_scores: dict[str, pd.DataFrame], train_rebals: list[str],
                    px_train: pd.DataFrame) -> pd.DataFrame:
    """Per (training rebalance, parent): mean-of-3M/6M Spearman IC and Q5-Q1
    spread — the raw material for the regime utilities. Training prices only."""
    fwd = forward_returns(px_train, train_rebals,
                          {h: {"3M": 3, "6M": 6}[h] for h in STAT_HORIZONS})
    rows: list[dict] = []
    for d in train_rebals:
        pframe = parent_scores.get(d)
        if pframe is None:
            continue
        for parent in pframe.columns:
            ics, spreads = [], []
            for h in STAT_HORIZONS:
                f = fwd[h].get(d)
                if f is None:
                    continue
                ic = spearman_ic(pframe[parent], f)
                sp = quintile_spread(pframe[parent], f)
                if ic is not None:
                    ics.append(ic)
                if sp is not None:
                    spreads.append(sp)
            if ics:
                rows.append({"date": d, "parent": parent,
                             "ic": float(np.mean(ics)),
                             "spread": float(np.mean(spreads)) if spreads else np.nan})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Build / cache
# --------------------------------------------------------------------------- #
def _cache_path(label: str):
    return CACHE_DIR / f"win_{label}_v{CACHE_VERSION}.pkl"


def build_window(win: StudyWindow, panel: StudyPanel, matrix: pd.DataFrame,
                 vix: pd.Series, *, verbose: bool = True) -> FrozenWindow:
    train_rebals = win.train_rebalances(panel.rebal_dates)
    test_rebals = win.test_rebalances(panel.rebal_dates)

    # --- the existing rolling-5Y model, frozen (single research/ reuse) ---
    from research.panel import ScorePanel
    from research.walkforward.selection import select_config
    sp = ScorePanel(rebal_dates=list(panel.rebal_dates), scores=panel.scores,
                    parent_keys=list(panel.parent_keys),
                    sub_by_parent={p: list(s) for p, s in panel.sub_by_parent.items()},
                    universe=list(panel.universe))
    t0 = time.time()
    cfg = select_config(sp, train_rebals, matrix, boundary=win.test_start)
    if verbose:
        print(f"  {win.label}: select_config on {len(train_rebals)} train rebals "
              f"({train_rebals[0]}→{train_rebals[-1]}) in {time.time() - t0:.1f}s")

    parent_scores = {d: parent_frame(panel.scores[d], cfg.sub_weights)
                     for d in [*train_rebals, *test_rebals] if d in panel.scores}

    px_train = matrix.loc[matrix.index <= win.test_start]
    train_stats = _training_stats(parent_scores, train_rebals, px_train)

    vix_train = vix[(vix.index >= pd.Timestamp(win.train_start))
                    & (vix.index < pd.Timestamp(win.test_start))]
    vix_at_rebal = {d: spot_vix(vix, d) for d in [*train_rebals, *test_rebals]}

    assert_pit(win, train_rebals, test_rebals, px_train, vix_train, train_stats)

    return FrozenWindow(
        label=win.label, train_start=win.train_start, test_start=win.test_start,
        test_end=win.test_end, train_rebals=train_rebals, test_rebals=test_rebals,
        sub_weights={p: dict(w) for p, w in cfg.sub_weights.items()},
        parent_weights=dict(cfg.parent_weights),
        parent_scores=parent_scores, train_stats=train_stats,
        vix_train=vix_train, vix_at_rebal=vix_at_rebal,
    )


def load_or_build(win: StudyWindow, panel: StudyPanel, matrix: pd.DataFrame,
                  vix: pd.Series, *, rebuild: bool = False,
                  verbose: bool = True) -> FrozenWindow:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _cache_path(win.label)
    if path.exists() and not rebuild:
        with path.open("rb") as fh:
            fw = pickle.load(fh)
        if isinstance(fw, FrozenWindow) and fw.test_end == win.test_end:
            return fw
    fw = build_window(win, panel, matrix, vix, verbose=verbose)
    with path.open("wb") as fh:
        pickle.dump(fw, fh)
    return fw


# --------------------------------------------------------------------------- #
# One-time fidelity cross-check (validation only, not part of the pipeline)
# --------------------------------------------------------------------------- #
def verify_composite_fidelity(fw: FrozenWindow, panel: StudyPanel,
                              sectors: pd.Series) -> float:
    """Assert this package's composite reproduces the research composite
    exactly under identical inputs (all-time-union convention). Returns the
    max abs score difference observed."""
    from research.panel import ScorePanel
    from research.walkforward.compose import FrozenConfig, frozen_composite
    dates = fw.test_rebals[:2]
    mini = ScorePanel(rebal_dates=dates, scores={d: panel.scores[d] for d in dates},
                      parent_keys=list(panel.parent_keys),
                      sub_by_parent={p: list(s) for p, s in panel.sub_by_parent.items()},
                      universe=list(panel.universe))
    cfg = FrozenConfig(sub_weights=fw.sub_weights,
                       parent_weights=fw.parent_weights, meta={})
    theirs = frozen_composite(mini, dates, cfg, sectors)
    worst = 0.0
    for d in dates:
        pf = fw.parent_scores[d].reindex(panel.universe)
        mine = composite_scores(pf, fw.parent_weights, sectors)
        diff = float((mine - theirs[d]).abs().max())
        worst = max(worst, diff)
        assert diff < 1e-6, f"composite fidelity broke on {d}: max diff {diff}"
    return worst
