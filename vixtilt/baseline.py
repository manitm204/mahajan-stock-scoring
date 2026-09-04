"""Stage A — per-window frozen rolling-5y baseline, cached to disk once.

For each semiannual window this builds the *existing* baseline model exactly as the
production walk-forward does (``research.walkforward.selection.select_config``: sub-factor
selection, intra-parent weights, IC/IR-capped parent weights) on the capped training
rebalances, then materialises everything the overlay study needs:

* frozen ``sub_weights`` / ``parent_weights``;
* PIT parent score frames (ticker × parent) on the training *stat* rebalances and the
  test rebalances;
* 1M forward returns on the stat rebalances computed from a price matrix truncated at
  the test boundary — with assertions that no realised window end crosses the boundary.

The result pickles to ``cache/vixtilt/window_<label>_v<V>.pkl`` and is reused across all
overlay variants and re-runs (this is the expensive step; everything downstream is fast).
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from research.panel import ScorePanel
from research.walkforward.compose import build_parent_panel
from research.walkforward.selection import select_config

from .windows import Window

CACHE_VERSION = 1
CACHE_DIR = Path("cache/vixtilt")

_MAX_GAP_DAYS = 25   # tolerance snapping d + H months to a real trading date


# --------------------------------------------------------------------------- #
# Forward returns (fresh implementation that also reports the realised end date
# of every window, so PIT assertions can be stated directly).
# --------------------------------------------------------------------------- #
def forward_returns(matrix: pd.DataFrame, dates: list[str], months: int,
                    boundary: str | None = None,
                    ) -> dict[str, tuple[str, pd.Series]]:
    """``{start_date: (end_date, per-ticker fwd return Series)}``.

    If ``boundary`` is given the price matrix is truncated there first, so no realised
    window can end after it (windows whose target lands past the boundary snap back to
    the last pre-boundary trading day within tolerance, or are dropped).
    """
    px = matrix.loc[matrix.index <= boundary] if boundary else matrix
    idx = pd.DatetimeIndex(pd.to_datetime(px.index))
    out: dict[str, tuple[str, pd.Series]] = {}
    for d in dates:
        if d not in px.index:
            continue
        target = pd.Timestamp(d) + pd.DateOffset(months=months)
        pos = idx.searchsorted(target)
        cands = [idx[p] for p in (pos, pos - 1) if 0 <= p < len(idx)]
        cands = [c for c in cands if abs((c - target).days) <= _MAX_GAP_DAYS]
        if not cands:
            continue
        end_ts = min(cands, key=lambda c: abs((c - target).days))
        if end_ts <= pd.Timestamp(d):
            continue
        end = end_ts.date().isoformat()
        fwd = (px.loc[end] / px.loc[d]) - 1.0
        out[d] = (end, fwd.dropna())
    return out


# --------------------------------------------------------------------------- #
# Cached per-window artefacts
# --------------------------------------------------------------------------- #
@dataclass
class WindowBaseline:
    label: str
    window: Window
    sub_weights: dict[str, dict[str, float]]        # parent -> {sub: weight}
    parent_weights: dict[str, float]                # frozen baseline weights
    selection_rebals: list[str]
    stat_rebals: list[str]
    test_rebals: list[str]
    parent_train: dict[str, pd.DataFrame] = field(default_factory=dict)  # stat date -> frame
    parent_test: dict[str, pd.DataFrame] = field(default_factory=dict)   # test date -> frame
    fwd1m_train: dict[str, pd.Series] = field(default_factory=dict)      # stat date -> fwd 1M
    fwd1m_ends: dict[str, str] = field(default_factory=dict)             # stat date -> end date
    meta: dict = field(default_factory=dict)
    version: int = CACHE_VERSION

    @property
    def parents(self) -> list[str]:
        frames = self.parent_test or self.parent_train
        first = next(iter(frames.values()), pd.DataFrame())
        return list(first.columns)


def _slice(panel: ScorePanel, dates: list[str]) -> ScorePanel:
    keep = [d for d in dates if d in panel.scores]
    return ScorePanel(rebal_dates=keep, scores={d: panel.scores[d] for d in keep},
                      parent_keys=list(panel.parent_keys),
                      sub_by_parent={p: list(s) for p, s in panel.sub_by_parent.items()},
                      universe=list(panel.universe))


def assert_window_pit(wb: WindowBaseline) -> list[str]:
    """Hard assertions that no test-period information entered training artefacts.

    Returns the list of human-readable checks performed (for the pit_checks log)."""
    w = wb.window
    sel_cap = w._cap(6)
    checks: list[str] = []
    assert all(d <= w.train_end for d in wb.selection_rebals + wb.stat_rebals), \
        f"{wb.label}: training rebalance after train_end"
    checks.append(f"all train rebals <= train_end ({w.train_end})")
    assert all(d <= sel_cap for d in wb.selection_rebals), \
        f"{wb.label}: selection rebalance inside 6M horizon cap"
    checks.append(f"selection rebals <= test_start - 6M ({sel_cap})")
    assert all(end <= w.test_start for end in wb.fwd1m_ends.values()), \
        f"{wb.label}: training 1M fwd window ends after test_start"
    checks.append(f"train fwd-1M window ends <= test_start ({w.test_start})")
    assert all(w.test_start <= d <= w.test_end for d in wb.test_rebals), \
        f"{wb.label}: test rebalance outside test window"
    checks.append("test rebals inside test window")
    mfe = wb.meta.get("max_fwd_end")
    if mfe:
        assert mfe <= w.test_start, f"{wb.label}: selection fwd start beyond boundary"
        checks.append(f"selection max fwd start {mfe} <= boundary")
    return checks


def build_window_baseline(panel: ScorePanel, matrix: pd.DataFrame,
                          window: Window, verbose: bool = True) -> WindowBaseline:
    sel = window.selection_rebals(panel.rebal_dates)
    stat = window.stat_rebals(panel.rebal_dates)
    test = window.test_rebals(panel.rebal_dates)
    if not sel or not test:
        raise ValueError(f"{window.label}: unusable window (sel={len(sel)} test={len(test)})")
    if verbose:
        print(f"  {window.label}: select on {len(sel)} ({sel[0]}→{sel[-1]}), "
              f"stat {len(stat)}, test {len(test)} ({test[0]}→{test[-1]})")

    cfg = select_config(panel, sel, matrix, boundary=window.test_start)

    # Parent score frames from the frozen sub weights, on stat + test dates.
    pp = build_parent_panel(_slice(panel, stat + test), cfg.sub_weights)
    parent_train = {d: pp.scores[d] for d in stat if d in pp.scores}
    parent_test = {d: pp.scores[d] for d in test if d in pp.scores}

    fwd = forward_returns(matrix, stat, 1, boundary=window.test_start)
    fwd1m_train = {d: s for d, (_, s) in fwd.items()}
    fwd1m_ends = {d: e for d, (e, _) in fwd.items()}

    wb = WindowBaseline(
        label=window.label, window=window,
        sub_weights={p: dict(w) for p, w in cfg.sub_weights.items()},
        parent_weights=dict(cfg.parent_weights),
        selection_rebals=sel, stat_rebals=stat, test_rebals=test,
        parent_train=parent_train, parent_test=parent_test,
        fwd1m_train=fwd1m_train, fwd1m_ends=fwd1m_ends,
        meta={"boundary": window.test_start,
              "max_fwd_end": cfg.meta.get("max_fwd_end"),
              "n_selection": len(sel), "n_stat": len(stat), "n_test": len(test)},
    )
    assert_window_pit(wb)
    return wb


# --------------------------------------------------------------------------- #
# Disk cache
# --------------------------------------------------------------------------- #
def _cache_path(label: str, cache_dir: Path) -> Path:
    return cache_dir / f"window_{label}_v{CACHE_VERSION}.pkl"


def load_or_build(panel: ScorePanel, matrix: pd.DataFrame, window: Window,
                  cache_dir: Path = CACHE_DIR, rebuild: bool = False,
                  verbose: bool = True) -> WindowBaseline:
    path = _cache_path(window.label, cache_dir)
    if not rebuild and path.exists():
        try:
            with path.open("rb") as fh:
                wb = pickle.load(fh)
            if isinstance(wb, WindowBaseline) and wb.version == CACHE_VERSION \
                    and wb.label == window.label:
                assert_window_pit(wb)
                if verbose:
                    print(f"  {window.label}: loaded from cache")
                return wb
        except Exception as exc:      # corrupt cache → rebuild
            if verbose:
                print(f"  {window.label}: cache unreadable ({exc}); rebuilding")
    wb = build_window_baseline(panel, matrix, window, verbose=verbose)
    cache_dir.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        pickle.dump(wb, fh)
    return wb
