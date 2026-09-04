# Hierarchical, Regime-Aware Production Factor Model — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a research-only, PIT-safe prototype that scores subfactors/parents with long-run + recent + probabilistic-VIX-regime evidence (shrunk toward long-run), selects/weights them with hysteresis, and compares it against the existing rolling-5Y baseline in a walk-forward harness — without touching any production file.

**Architecture:** Seven new files under `research/walkforward/` (one per concern: probability math, evidence cache, scoring/classification, subfactor selection, parent weighting/hysteresis, the walk-forward loop) plus one entry-point script, each built bottom-up with its own unit tests against synthetic `ScorePanel`s before the final task wires them together against the real cached candidate panel. Every new file only *adds* — nothing in `research/parent_selection.py`, `research/walkforward/{selection,compose,splits,vix_overlay,vix_regime_study,portfolio,analysis}.py`, `factors/`, or `config.yaml` is modified; those are imported and reused as-is.

**Tech Stack:** Python, pandas/numpy, pytest. Reuses this repo's existing `ScorePanel` (`research/panel.py`), `select_config`/`FrozenConfig` (`research/walkforward/selection.py`/`compose.py`), `select_subfactors`/`_pct_rank` (`research/parent_selection.py`), `semiannual_policy_splits` (`research/walkforward/splits.py`), and `pf.simulate`/`analysis.composite_ic`/`analysis.quantile_analysis` (`research/walkforward/portfolio.py`/`analysis.py`).

**Spec:** `docs/superpowers/specs/2026-07-11-hierarchical-regime-aware-factor-model-design.md` — read this first for the full formulas and rationale; this plan implements it task-by-task.

---

## File Structure

| File | New/Modified | Responsibility |
|---|---|---|
| `research/walkforward/regime_probability.py` | New | Smooth VIX regime probabilities, shrinkage, `expected_ic` blend. Pure functions, no panel/DB dependency. |
| `research/walkforward/regime_aware_evidence.py` | New | Monthly PIT evidence cache (`build_monthly_cache`) + `as_of(cutoff)` slicing (long-run/recent-24M/annual rollup). |
| `research/walkforward/regime_aware_scoring.py` | New | Predictive/Reliability/Production scores + CORE/REGIME_DEPENDENT/WATCHLIST/EXCLUDED classification. |
| `research/walkforward/regime_aware_selection.py` | New | Per-parent subfactor selection (R² gate reused from `parent_selection.py`, new "materially lowers parent" stop rule, weight floor). |
| `research/walkforward/regime_aware_parents.py` | New | Parent utility score, 70/30 base/adaptive blend, monthly/quarterly change caps, hysteresis state machine. |
| `research/walkforward/regime_aware_walkforward.py` | New | Continuous monthly-stepping loop; builds variants A/B/C/D/+floor; snapshots `FrozenConfig` at each semiannual boundary. |
| `run_regime_aware_study.py` | New | Entry point: loads panel/matrix/VIX, runs the walk-forward, writes `output/regime_aware/` CSVs + `REGIME_AWARE_REPORT.md`. |
| `tests/test_regime_probability.py` | New | Unit tests for Task 1. |
| `tests/test_regime_aware_evidence.py` | New | Unit tests for Task 2. |
| `tests/test_regime_aware_scoring.py` | New | Unit tests for Task 3. |
| `tests/test_regime_aware_selection.py` | New | Unit tests for Task 4. |
| `tests/test_regime_aware_parents.py` | New | Unit tests for Task 5. |
| `tests/test_regime_aware_walkforward.py` | New | Unit + DB-guarded integration tests for Tasks 6 and 8. |
| `tests/test_run_regime_aware_study.py` | New | Unit test for Task 7's `_recommendation`. |

Run the full new test suite anytime with:
```bash
pytest tests/test_regime_probability.py tests/test_regime_aware_evidence.py tests/test_regime_aware_scoring.py tests/test_regime_aware_selection.py tests/test_regime_aware_parents.py tests/test_regime_aware_walkforward.py tests/test_run_regime_aware_study.py -v
```

---

### Task 1: Smooth VIX Regime Probabilities + Shrinkage (`regime_probability.py`)

**Files:**
- Create: `research/walkforward/regime_probability.py`
- Test: `tests/test_regime_probability.py`

This is pure math (Section 1 of the spec: smooth regime membership, shrinkage, the `expected_ic` blend) with no `ScorePanel`/DB dependency, so it's built and tested first, in isolation.

- [ ] **Step 1: Write the failing tests for `regime_probabilities`**

```python
# tests/test_regime_probability.py
from __future__ import annotations

import numpy as np
import pytest

from research.walkforward.regime_probability import (
    REGIME_ORDER, regime_probabilities, shrink_regime_ic,
    effective_regime_stats, expected_ic,
)


def test_regime_probabilities_sum_to_one_and_nonnegative():
    for vix in [5.0, 15.0, 20.0, 25.0, 40.0]:
        probs = regime_probabilities(vix)
        assert abs(sum(probs.values()) - 1.0) < 1e-9
        assert all(p >= 0.0 for p in probs.values())


def test_regime_probabilities_boundary_is_half_split():
    # At the boundary itself, Low/Medium split exactly 50/50 regardless of width
    # (sigmoid(0)==0.5 always); with width=4 the *other* boundary (High, 10 points
    # away) still bleeds in a little (~0.076) -- that bleed is the point of a smooth
    # transition, not a bug, so this checks the exact values at width=4 rather than
    # assuming High is ~0.
    probs = regime_probabilities(15.0)
    assert probs["Low (<15)"] == pytest.approx(0.5, abs=1e-9)
    assert probs["Medium (15-25)"] == pytest.approx(0.424142, abs=1e-4)
    assert probs["High (>25)"] == pytest.approx(0.075858, abs=1e-4)


def test_regime_probabilities_center_dominates():
    # At the exact bucket center (vix=20, halfway between the 15/25 boundaries),
    # Medium is the plurality but width=4 is wide enough that it isn't overwhelming
    # (~0.55, not ~1.0) -- that's the smooth-transition tradeoff the spec's width=4
    # explicitly chooses over a near-hard cutoff.
    probs = regime_probabilities(20.0)
    assert probs["Medium (15-25)"] > 0.5
    assert probs["Medium (15-25)"] > probs["Low (<15)"]
    assert probs["Medium (15-25)"] > probs["High (>25)"]


def test_regime_probabilities_nan_vix_returns_nan():
    probs = regime_probabilities(float("nan"))
    assert all(np.isnan(p) for p in probs.values())
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_regime_probability.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'research.walkforward.regime_probability'`

- [ ] **Step 3: Implement `regime_probabilities`**

```python
# research/walkforward/regime_probability.py
"""Smooth VIX regime probabilities, shrinkage, and the Expected IC blend.

Pure functions only — no ScorePanel/DB dependency, so every function here takes
plain floats/Series and is testable in isolation. See
docs/superpowers/specs/2026-07-11-hierarchical-regime-aware-factor-model-design.md
Section 1 for the derivation.

Replaces the hard vix < 15 / 15-25 / > 25 cutoffs in vix_regime_study.py with a
soft 3-way split: two logistic sigmoids centered at the same boundaries (15, 25)
give a probability vector that always sums to 1, instead of one hard label.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from research.walkforward.vix_regime_study import REGIME_ORDER

REGIME_LOW = 15.0
REGIME_HIGH = 25.0
TRANSITION_WIDTH = 4.0     # sigmoid width; larger = smoother regime transition
SHRINKAGE_K = 24.0         # default shrinkage constant (spec Section 2)


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def regime_probabilities(
    vix: float, *, low: float = REGIME_LOW, high: float = REGIME_HIGH,
    width: float = TRANSITION_WIDTH,
) -> dict[str, float]:
    """Soft (Low, Medium, High) membership for one VIX level. Always sums to 1.

    p_low = 1 - sigmoid((vix-low)/width); p_high = sigmoid((vix-high)/width);
    p_med = the remainder. p_med >= 0 always because sigmoid is monotonic and
    low < high, so sigmoid((vix-low)/width) >= sigmoid((vix-high)/width).
    """
    if not np.isfinite(vix):
        return {r: float("nan") for r in REGIME_ORDER}
    s1 = sigmoid((vix - low) / width)
    s2 = sigmoid((vix - high) / width)
    return {REGIME_ORDER[0]: 1.0 - s1, REGIME_ORDER[1]: s1 - s2, REGIME_ORDER[2]: s2}
```

- [ ] **Step 4: Run tests to verify the new ones pass**

Run: `pytest tests/test_regime_probability.py -v -k regime_probabilities`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add research/walkforward/regime_probability.py tests/test_regime_probability.py
git commit -m "feat: add smooth VIX regime probability function"
```

- [ ] **Step 6: Write the failing tests for `shrink_regime_ic`**

```python
# append to tests/test_regime_probability.py
def test_shrink_pulls_toward_long_run_with_thin_sample():
    shrunk = shrink_regime_ic(regime_ic=0.10, long_run_ic=0.01, n_eff=1.0, k=24.0)
    lam = 1.0 / (1.0 + 24.0)
    assert shrunk == pytest.approx(lam * 0.10 + (1 - lam) * 0.01, abs=1e-9)


def test_shrink_trusts_observed_with_large_sample():
    shrunk = shrink_regime_ic(regime_ic=0.10, long_run_ic=0.01, n_eff=1000.0, k=24.0)
    assert shrunk == pytest.approx(0.10, abs=0.01)


def test_shrink_falls_back_to_long_run_when_no_observations():
    shrunk = shrink_regime_ic(regime_ic=float("nan"), long_run_ic=0.02, n_eff=0.0, k=24.0)
    assert shrunk == pytest.approx(0.02, abs=1e-9)
```

- [ ] **Step 7: Run to verify failure, then implement `shrink_regime_ic`**

Run: `pytest tests/test_regime_probability.py -v -k shrink` → FAIL (`NameError`/`ImportError`)

```python
# append to research/walkforward/regime_probability.py
def shrink_regime_ic(
    regime_ic: float, long_run_ic: float, n_eff: float, k: float = SHRINKAGE_K,
) -> float:
    """lambda = n_eff/(n_eff+k); shrunk = lambda*regime_ic + (1-lambda)*long_run_ic.

    Falls back to long_run_ic when there's no usable regime observation (n_eff<=0
    or regime_ic is NaN) — an empty/thin regime should never inject noise.
    """
    if n_eff <= 0 or not np.isfinite(regime_ic):
        return long_run_ic
    lam = n_eff / (n_eff + k)
    return lam * regime_ic + (1.0 - lam) * long_run_ic
```

- [ ] **Step 8: Run tests to verify they pass**

Run: `pytest tests/test_regime_probability.py -v -k shrink`
Expected: PASS (3 tests)

- [ ] **Step 9: Commit**

```bash
git add research/walkforward/regime_probability.py tests/test_regime_probability.py
git commit -m "feat: add regime-IC shrinkage toward long-run mean"
```

- [ ] **Step 10: Write the failing tests for `effective_regime_stats` and `expected_ic`**

```python
# append to tests/test_regime_probability.py
def test_effective_regime_stats_weights_by_probability():
    # Two dates: VIX=10 (heavily Low) and VIX=30 (heavily High), IC=0.10 and IC=0.02.
    vix = pd.Series({"2020-01-31": 10.0, "2020-02-29": 30.0})
    ic = pd.Series({"2020-01-31": 0.10, "2020-02-29": 0.02})
    stats = effective_regime_stats(vix, ic)
    assert set(stats) == set(REGIME_ORDER)
    # Low regime's n_eff should come almost entirely from the VIX=10 date.
    assert stats["Low (<15)"]["n_eff"] == pytest.approx(
        regime_probabilities(10.0)["Low (<15)"] + regime_probabilities(30.0)["Low (<15)"],
        abs=1e-9)
    assert stats["Low (<15)"]["regime_ic"] == pytest.approx(0.10, abs=0.02)
    assert stats["High (>25)"]["regime_ic"] == pytest.approx(0.02, abs=0.02)


def test_effective_regime_stats_empty_series_returns_nan_stats():
    stats = effective_regime_stats(pd.Series(dtype=float), pd.Series(dtype=float))
    for r in REGIME_ORDER:
        assert stats[r]["n_eff"] == 0.0
        assert np.isnan(stats[r]["regime_ic"])


def test_expected_ic_blends_50_25_25():
    # Flat regime evidence equal to long_run_ic → regime component collapses to
    # long_run_ic too, so expected_ic == 0.75*long_run + 0.25*recent exactly.
    regime_stats = {r: {"n_eff": 1000.0, "regime_ic": 0.05} for r in REGIME_ORDER}
    result = expected_ic(long_run_ic=0.05, recent_ic=0.09, vix_now=20.0,
                         regime_stats=regime_stats)
    assert result == pytest.approx(0.75 * 0.05 + 0.25 * 0.09, abs=1e-6)


def test_expected_ic_falls_back_to_long_run_when_recent_missing():
    regime_stats = {r: {"n_eff": 0.0, "regime_ic": float("nan")} for r in REGIME_ORDER}
    result = expected_ic(long_run_ic=0.03, recent_ic=float("nan"), vix_now=20.0,
                         regime_stats=regime_stats)
    assert result == pytest.approx(0.03, abs=1e-9)


def test_expected_ic_nan_long_run_is_nan():
    regime_stats = {r: {"n_eff": 0.0, "regime_ic": float("nan")} for r in REGIME_ORDER}
    result = expected_ic(long_run_ic=float("nan"), recent_ic=0.05, vix_now=20.0,
                         regime_stats=regime_stats)
    assert np.isnan(result)
```

- [ ] **Step 11: Run to verify failure, then implement both functions**

Run: `pytest tests/test_regime_probability.py -v -k "effective_regime_stats or expected_ic"` → FAIL

```python
# append to research/walkforward/regime_probability.py
def effective_regime_stats(
    vix_by_date: pd.Series, ic_by_date: pd.Series, *, width: float = TRANSITION_WIDTH,
) -> dict[str, dict[str, float]]:
    """Probability-weighted (n_eff, regime_ic) per regime over the dates given.

    Both Series are indexed by date and must already be filtered by the caller to
    date <= cutoff (this function has no notion of "now" — it aggregates whatever
    it's handed). n_eff is the probability-weighted effective sample size; regime_ic
    is the probability-weighted mean IC. NaN ICs are skipped.
    """
    out = {r: {"n_eff": 0.0, "regime_ic": float("nan")} for r in REGIME_ORDER}
    idx = vix_by_date.index.intersection(ic_by_date.index)
    if idx.empty:
        return out
    weighted_ic = {r: 0.0 for r in REGIME_ORDER}
    n_eff = {r: 0.0 for r in REGIME_ORDER}
    for d in idx:
        ic = ic_by_date.loc[d]
        if pd.isna(ic):
            continue
        probs = regime_probabilities(float(vix_by_date.loc[d]), width=width)
        for r in REGIME_ORDER:
            p = probs[r]
            if not np.isfinite(p):
                continue
            n_eff[r] += p
            weighted_ic[r] += p * float(ic)
    for r in REGIME_ORDER:
        out[r]["n_eff"] = n_eff[r]
        out[r]["regime_ic"] = weighted_ic[r] / n_eff[r] if n_eff[r] > 1e-9 else float("nan")
    return out


def expected_ic(
    long_run_ic: float, recent_ic: float, vix_now: float,
    regime_stats: dict[str, dict[str, float]], *,
    k: float = SHRINKAGE_K, width: float = TRANSITION_WIDTH,
) -> float:
    """expected_ic = 0.50*long_run + 0.25*recent + 0.25*prob-weighted shrunk regime IC.

    ``regime_stats`` is the output of :func:`effective_regime_stats` (already
    computed over date <= cutoff). ``recent_ic`` falls back to ``long_run_ic`` when
    NaN (e.g. fewer than 24 months of history so far — see spec Section 1).
    """
    if not np.isfinite(long_run_ic):
        return float("nan")
    recent = recent_ic if np.isfinite(recent_ic) else long_run_ic
    probs_now = regime_probabilities(vix_now, width=width)
    regime_component = 0.0
    total_p = 0.0
    for r in REGIME_ORDER:
        p = probs_now[r]
        if not np.isfinite(p):
            continue
        stats = regime_stats.get(r, {"n_eff": 0.0, "regime_ic": float("nan")})
        shrunk = shrink_regime_ic(stats["regime_ic"], long_run_ic, stats["n_eff"], k=k)
        regime_component += p * shrunk
        total_p += p
    if total_p <= 1e-9:
        regime_component = long_run_ic
    else:
        regime_component /= total_p
    return 0.50 * long_run_ic + 0.25 * recent + 0.25 * regime_component
```

- [ ] **Step 12: Run tests to verify all pass**

Run: `pytest tests/test_regime_probability.py -v`
Expected: PASS (12 tests)

- [ ] **Step 13: Commit**

```bash
git add research/walkforward/regime_probability.py tests/test_regime_probability.py
git commit -m "feat: add effective regime stats and expected_ic blend"
```

---

### Task 2: Monthly PIT Evidence Cache (`regime_aware_evidence.py`)

**Files:**
- Create: `research/walkforward/regime_aware_evidence.py`
- Test: `tests/test_regime_aware_evidence.py`

Builds the per-`(sub_factor, date)` cache (one row per monthly rebalance, not
collapsed to a calendar year like `factor_persistence.py`), then slices it
PIT-safe at an arbitrary cutoff into everything Section 1/2 of the spec need:
long-run, recent-24M, and (via `regime_probability.py`, Task 1) probability-
weighted shrunk regime evidence and `expected_ic`.

- [ ] **Step 1: Write the failing test for `build_monthly_cache`**

```python
# tests/test_regime_aware_evidence.py
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.panel import ScorePanel
from research.walkforward.regime_aware_evidence import build_monthly_cache, as_of


def _toy_panel(dates: list[str], universe: list[str]) -> ScorePanel:
    """One parent 'p1' with two subs whose scores rank the universe deterministically
    (s1 ascending with ticker order, s2 descending), so IC direction is predictable."""
    scores = {}
    for k, d in enumerate(dates):
        base = np.linspace(10, 90, len(universe)) + k
        scores[d] = pd.DataFrame({"s1": base, "s2": base[::-1]}, index=universe)
    return ScorePanel(rebal_dates=list(dates), scores=scores,
                      parent_keys=["p1"], sub_by_parent={"p1": ["s1", "s2"]},
                      universe=list(universe))


def _toy_matrix(dates: list[str], universe: list[str], trend: dict[str, float]) -> pd.DataFrame:
    """Monotone price paths so forward returns are deterministic: ticker T{i} grows at
    (trend intercept + i * trend slope) per period, giving s1 a positive IC and s2 a
    negative IC by construction (s1 ranks tickers ascending, prices grow ascending too)."""
    idx = pd.date_range(dates[0], periods=len(dates) + 8, freq="ME").strftime("%Y-%m-%d")
    data = {}
    for i, t in enumerate(universe):
        growth = 1.0 + trend["intercept"] + i * trend["slope"]
        data[t] = growth ** np.arange(len(idx))
    return pd.DataFrame(data, index=idx)


UNIVERSE = [f"T{i}" for i in range(30)]
DATES = pd.date_range("2018-01-31", periods=10, freq="ME").strftime("%Y-%m-%d").tolist()


def test_build_monthly_cache_shape_and_columns():
    panel = _toy_panel(DATES, UNIVERSE)
    matrix = _toy_matrix(DATES, UNIVERSE, {"intercept": 0.01, "slope": 0.002})
    vix = pd.Series(20.0, index=pd.to_datetime(matrix.index))
    cache = build_monthly_cache(panel, matrix, vix)
    assert set(cache.columns) == {
        "sub_factor", "parent", "date", "vix_level", "coverage",
        "ic_3M", "spread_3M_raw", "ic_6M", "spread_6M_raw",
    }
    assert set(cache["sub_factor"].unique()) == {"s1", "s2"}
    # s1 ranks tickers ascending and prices grow ascending with ticker index -> positive IC.
    s1_ic = cache[cache.sub_factor == "s1"]["ic_6M"].dropna()
    assert (s1_ic > 0).all()
    s2_ic = cache[cache.sub_factor == "s2"]["ic_6M"].dropna()
    assert (s2_ic < 0).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_regime_aware_evidence.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'research.walkforward.regime_aware_evidence'`

- [ ] **Step 3: Implement `build_monthly_cache`**

```python
# research/walkforward/regime_aware_evidence.py
"""Monthly point-in-time evidence cache + as-of(cutoff) slicing.

One row per (sub_factor, rebalance date) -- unlike factor_persistence.py, which
collapses to one row per (sub_factor, calendar year), this keeps every month's
raw observation so a monthly walk-forward can ask "what did we know as of an
arbitrary cutoff" via expanding/rolling slices instead of only per-year reads.
See docs/superpowers/specs/2026-07-11-hierarchical-regime-aware-factor-model-design.md
Section 1.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from research import compute_forward_returns
from research.ic import period_ic
from research.panel import ScorePanel
from research.quintiles import quintile_profile
from research.walkforward.regime_probability import (
    REGIME_ORDER, SHRINKAGE_K, effective_regime_stats, expected_ic,
)
from research.walkforward.splits import DATA_START
from research.walkforward.vix_regime_study import _vix_spot

RECENT_MONTHS = 24
MIN_NAMES = 20


def build_monthly_cache(
    panel: ScorePanel, matrix: pd.DataFrame, vix: pd.Series, *, min_names: int = MIN_NAMES,
) -> pd.DataFrame:
    """One row per (sub_factor, date): ic_3M, ic_6M, spread_3M_raw, spread_6M_raw,
    coverage (fraction of universe scored that date), vix_level (spot VIX at that date).

    Spreads are the *raw* one-period Q5-Q1 gap (not annualised) -- as_of() annualises
    after aggregating, matching the convention factor_persistence.py already uses
    (annualise the mean, not each observation).
    """
    fwd_by_h = compute_forward_returns(matrix, panel.rebal_dates, {"3M": 3, "6M": 6})
    n_uni = len(panel.universe)
    rows: list[dict] = []
    for sub in panel.all_subs:
        parent = panel.parent_of(sub)
        for d in panel.rebal_dates:
            frame = panel.scores.get(d)
            if frame is None or sub not in frame.columns:
                continue
            row: dict = {
                "sub_factor": sub, "parent": parent, "date": d,
                "vix_level": _vix_spot(vix, d),
                "coverage": float(frame[sub].count()) / n_uni if n_uni else float("nan"),
            }
            for h in ("3M", "6M"):
                fwd = fwd_by_h.get(h, {}).get(d)
                if fwd is None:
                    row[f"ic_{h}"] = float("nan")
                    row[f"spread_{h}_raw"] = float("nan")
                    continue
                ic = period_ic(frame[sub], fwd, min_names=min_names)
                row[f"ic_{h}"] = ic if ic is not None else float("nan")
                prof = quintile_profile(frame[sub], fwd, min_names=min_names)
                row[f"spread_{h}_raw"] = float(prof[-1] - prof[0]) if prof is not None else float("nan")
            rows.append(row)
    return pd.DataFrame(rows)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_regime_aware_evidence.py -v -k build_monthly_cache`
Expected: PASS (1 test)

- [ ] **Step 5: Commit**

```bash
git add research/walkforward/regime_aware_evidence.py tests/test_regime_aware_evidence.py
git commit -m "feat: add monthly PIT evidence cache builder"
```

- [ ] **Step 6: Write the failing test for `as_of`**

```python
# append to tests/test_regime_aware_evidence.py
def _hand_cache(sub: str, parent: str, dates: list[str], ics: list[float],
                spreads6: list[float], vix_levels: list[float],
                coverage: float = 1.0) -> pd.DataFrame:
    """A hand-built single-sub cache: ic_3M==ic_6M==ics[i] for simplicity."""
    return pd.DataFrame({
        "sub_factor": sub, "parent": parent, "date": dates,
        "vix_level": vix_levels, "coverage": coverage,
        "ic_3M": ics, "ic_6M": ics,
        "spread_3M_raw": spreads6, "spread_6M_raw": spreads6,
    })


def test_as_of_long_run_and_recent_match_hand_computation():
    # 36 months, Jan-2015..Dec-2017 (panel_inception = 2015-06-30 per DATA_START, so the
    # first 5 months fall before inception and must be excluded from long_run).
    dates = pd.date_range("2015-01-31", periods=36, freq="ME").strftime("%Y-%m-%d").tolist()
    ics = [0.01 + 0.001 * i for i in range(36)]        # trending up over time
    spreads = [0.02] * 36
    vix_levels = [20.0] * 36                            # always Medium regime
    cache = _hand_cache("s1", "p1", dates, ics, spreads, vix_levels)
    cutoff = dates[-1]
    vix = pd.Series(vix_levels, index=pd.to_datetime(dates))

    out = as_of(cache, cutoff, vix, panel_inception="2015-06-30")
    row = out[out.sub_factor == "s1"].iloc[0]

    in_window = [ic for d, ic in zip(dates, ics) if d >= "2015-06-30"]
    assert row["long_run_mean_ic"] == pytest.approx(np.mean(in_window), abs=1e-9)
    assert row["n_months"] == len(in_window)

    recent_start = (pd.Timestamp(cutoff) - pd.DateOffset(months=24)).date().isoformat()
    recent = [ic for d, ic in zip(dates, ics) if d >= recent_start]
    assert row["recent_24m_ic"] == pytest.approx(np.mean(recent), abs=1e-9)

    # Constant VIX=20 (pure Medium) -> almost all regime weight in Medium.
    assert row["regime_n_eff_medium"] > row["regime_n_eff_low"]
    assert row["regime_n_eff_medium"] > row["regime_n_eff_high"]

    # spread always +0.02 raw at 6M -> annualised = 0.02 * (12/6) = 0.04
    assert row["long_run_spread_ann"] == pytest.approx(0.04, abs=1e-9)


def test_as_of_excludes_dates_after_cutoff():
    dates = pd.date_range("2016-01-31", periods=24, freq="ME").strftime("%Y-%m-%d").tolist()
    ics = [0.05] * 12 + [-0.05] * 12          # sign flips halfway through
    spreads = [0.01] * 24
    vix_levels = [20.0] * 24
    cache = _hand_cache("s1", "p1", dates, ics, spreads, vix_levels)
    vix = pd.Series(vix_levels, index=pd.to_datetime(dates))

    cutoff = dates[11]                         # exactly at the sign flip boundary
    out = as_of(cache, cutoff, vix, panel_inception="2015-06-30")
    row = out[out.sub_factor == "s1"].iloc[0]
    # Only the first 12 (all +0.05) months should be visible -- no look-ahead into the
    # -0.05 months that come after cutoff.
    assert row["long_run_mean_ic"] == pytest.approx(0.05, abs=1e-9)
    assert row["pct_positive_years"] == pytest.approx(1.0, abs=1e-9)


def test_as_of_empty_window_returns_all_expected_columns():
    # cutoff before panel_inception -> the PIT filter empties the window entirely;
    # the empty-path DataFrame must still carry every column the populated path
    # produces (including the 6 regime_* columns), so a caller doing df["regime_ic_low"]
    # on either path never KeyErrors.
    cache = _hand_cache("s1", "p1", ["2020-01-31"], [0.05], [0.01], [20.0])
    vix = pd.Series([20.0], index=pd.to_datetime(["2020-01-31"]))
    out = as_of(cache, "2015-01-01", vix, panel_inception="2015-06-30")
    assert out.empty
    expected_cols = {
        "sub_factor", "parent", "long_run_mean_ic", "long_run_std_ic",
        "long_run_hit_rate", "long_run_spread_ann", "recent_24m_ic",
        "pct_positive_years", "persistence_ir", "spread_consistency",
        "coverage", "n_months", "n_years", "regime_dependence", "expected_ic",
        "regime_ic_low", "regime_n_eff_low", "regime_ic_medium", "regime_n_eff_medium",
        "regime_ic_high", "regime_n_eff_high",
    }
    assert set(out.columns) == expected_cols
```

- [ ] **Step 7: Run to verify failure, then implement `as_of`**

Run: `pytest tests/test_regime_aware_evidence.py -v -k as_of` → FAIL

```python
# append to research/walkforward/regime_aware_evidence.py
def as_of(
    cache: pd.DataFrame, cutoff: str, vix: pd.Series, *,
    panel_inception: str = DATA_START, recent_months: int = RECENT_MONTHS,
    k: float = SHRINKAGE_K,
) -> pd.DataFrame:
    """Everything Section 1/2 of the spec need, one row per sub_factor, PIT-safe.

    Only cache rows with panel_inception <= date <= cutoff are used. Long-run stats
    expand from panel_inception; recent_24m_ic is a trailing window (shorter than 24
    months, degrading to ~long_run, in the first two years -- see spec Section 1).
    """
    hist = cache[(cache["date"] <= cutoff) & (cache["date"] >= panel_inception)].copy()
    regime_cols = [f"regime_{field}_{r.split(' ')[0].lower()}"
                  for r in REGIME_ORDER for field in ("ic", "n_eff")]
    cols = ["sub_factor", "parent", "long_run_mean_ic", "long_run_std_ic",
            "long_run_hit_rate", "long_run_spread_ann", "recent_24m_ic",
            "pct_positive_years", "persistence_ir", "spread_consistency",
            "coverage", "n_months", "n_years", "regime_dependence", "expected_ic",
            *regime_cols]
    if hist.empty:
        return pd.DataFrame(columns=cols)

    hist["mean_ic"] = hist[["ic_3M", "ic_6M"]].mean(axis=1)
    hist["year"] = hist["date"].str.slice(0, 4)
    recent_start = (pd.Timestamp(cutoff) - pd.DateOffset(months=recent_months)).date().isoformat()
    vix_now = _vix_spot(vix, cutoff)

    rows: list[dict] = []
    for sub, g in hist.groupby("sub_factor"):
        g = g.sort_values("date")
        mean_ic_series = g.set_index("date")["mean_ic"].dropna()
        long_run_mean = float(mean_ic_series.mean()) if not mean_ic_series.empty else float("nan")
        long_run_std = (float(mean_ic_series.std(ddof=1))
                        if len(mean_ic_series) > 1 else float("nan"))
        long_run_hit = (float((mean_ic_series > 0).mean())
                        if not mean_ic_series.empty else float("nan"))

        recent = mean_ic_series[mean_ic_series.index >= recent_start]
        recent_ic = float(recent.mean()) if not recent.empty else float("nan")

        spread6 = g.set_index("date")["spread_6M_raw"].dropna()
        long_run_spread_ann = float(spread6.mean() * 2.0) if not spread6.empty else float("nan")

        annual = g.groupby("year")["mean_ic"].mean().dropna()
        pct_pos_years = float((annual > 0).mean()) if not annual.empty else float("nan")
        pers_ir = (float(annual.mean() / annual.std(ddof=1))
                  if len(annual) > 1 and annual.std(ddof=1) > 1e-9 else float("nan"))
        annual_spread = (g.assign(spread_ann=g["spread_6M_raw"] * 2.0)
                         .groupby("year")["spread_ann"].mean().dropna())
        spread_consistency = (
            float((np.sign(annual_spread) == np.sign(long_run_spread_ann)).mean())
            if not annual_spread.empty and long_run_spread_ann == long_run_spread_ann
            else float("nan"))

        coverage = float(g["coverage"].mean()) if not g["coverage"].empty else float("nan")
        n_months = int(len(mean_ic_series))
        n_years = int(g["year"].nunique())

        vix_by_date = g.set_index("date")["vix_level"]
        regime_stats = effective_regime_stats(vix_by_date, mean_ic_series)
        raw_ics = np.array([regime_stats[r]["regime_ic"] for r in REGIME_ORDER], dtype=float)
        regime_dep = float(np.nanstd(raw_ics)) if np.isfinite(raw_ics).any() else float("nan")

        exp_ic = expected_ic(long_run_mean, recent_ic, vix_now, regime_stats, k=k)

        row = {
            "sub_factor": sub, "parent": g["parent"].iloc[0],
            "long_run_mean_ic": long_run_mean, "long_run_std_ic": long_run_std,
            "long_run_hit_rate": long_run_hit, "long_run_spread_ann": long_run_spread_ann,
            "recent_24m_ic": recent_ic, "pct_positive_years": pct_pos_years,
            "persistence_ir": pers_ir, "spread_consistency": spread_consistency,
            "coverage": coverage, "n_months": n_months, "n_years": n_years,
            "regime_dependence": regime_dep, "expected_ic": exp_ic,
        }
        for r in REGIME_ORDER:
            key = r.split(" ")[0].lower()
            row[f"regime_ic_{key}"] = regime_stats[r]["regime_ic"]
            row[f"regime_n_eff_{key}"] = regime_stats[r]["n_eff"]
        rows.append(row)
    return pd.DataFrame(rows)
```

- [ ] **Step 8: Run tests to verify all pass**

Run: `pytest tests/test_regime_aware_evidence.py -v`
Expected: PASS (4 tests)

- [ ] **Step 9: Commit**

```bash
git add research/walkforward/regime_aware_evidence.py tests/test_regime_aware_evidence.py
git commit -m "feat: add as_of(cutoff) PIT evidence slicing"
```

---

### Task 3: Scoring & Classification (`regime_aware_scoring.py`)

**Files:**
- Create: `research/walkforward/regime_aware_scoring.py`
- Test: `tests/test_regime_aware_scoring.py`

Consumes one `as_of()` evidence table (Task 2) and adds: `ic_ir`, absolute
`classification` (CORE/REGIME_DEPENDENT/WATCHLIST/EXCLUDED), per-parent
percentile-ranked Predictive/Reliability/Production scores, and an `eligible`
flag. See spec Section 2.

- [ ] **Step 1: Write the failing test for `classify_subfactors`**

```python
# tests/test_regime_aware_scoring.py
from __future__ import annotations

import pandas as pd
import pytest

from research.walkforward.regime_aware_scoring import classify_subfactors, score_table


def test_classify_core_regime_dependent_excluded_watchlist():
    evidence = pd.DataFrame([
        # CORE: strong IC, mostly positive years, low regime swing.
        {"sub_factor": "core1", "long_run_mean_ic": 0.02, "pct_positive_years": 0.80,
         "regime_dependence": 0.01, "n_years": 5},
        # EXCLUDED: negative IC, mostly negative years, enough history to trust it.
        {"sub_factor": "bad1", "long_run_mean_ic": -0.02, "pct_positive_years": 0.20,
         "regime_dependence": 0.02, "n_years": 5},
        # REGIME_DEPENDENT: non-negative IC but a big regime swing.
        {"sub_factor": "regime1", "long_run_mean_ic": 0.005, "pct_positive_years": 0.50,
         "regime_dependence": 0.08, "n_years": 5},
        # WATCHLIST: negative IC but too little history to call it EXCLUDED.
        {"sub_factor": "thin1", "long_run_mean_ic": -0.02, "pct_positive_years": 0.20,
         "regime_dependence": 0.01, "n_years": 1},
    ])
    flags = classify_subfactors(evidence)
    assert flags.tolist() == ["CORE", "EXCLUDED", "REGIME_DEPENDENT", "WATCHLIST"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_regime_aware_scoring.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'research.walkforward.regime_aware_scoring'`

- [ ] **Step 3: Implement `classify_subfactors`**

```python
# research/walkforward/regime_aware_scoring.py
"""Predictive/Reliability/Production scores + CORE/REGIME_DEPENDENT/WATCHLIST/
EXCLUDED classification, consuming one regime_aware_evidence.as_of() table.
See docs/superpowers/specs/2026-07-11-hierarchical-regime-aware-factor-model-design.md
Section 2.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from research.parent_selection import _pct_rank

CORE_MIN_IC = 0.01
CORE_MIN_POS_YEARS = 0.60
CORE_MAX_REGIME_DEP = 0.03
EXCLUDED_MAX_POS_YEARS = 0.40
MIN_YEARS_FOR_VERDICT = 3

PREDICTIVE_WEIGHTS = {"expected_ic": 0.40, "ic_ir": 0.25,
                      "long_run_spread_ann": 0.25, "long_run_hit_rate": 0.10}
RELIABILITY_WEIGHTS = {"pct_positive_years": 0.35, "persistence_ir": 0.25,
                       "spread_consistency": 0.20, "coverage": 0.10, "n_months": 0.10}


def _classify_row(row: pd.Series) -> str:
    ic = row["long_run_mean_ic"]
    pos_years = row["pct_positive_years"]
    n_years = row["n_years"]
    regime_dep = row["regime_dependence"]
    if pd.isna(ic) or pd.isna(pos_years):
        return "WATCHLIST"
    if ic < 0 and pos_years <= EXCLUDED_MAX_POS_YEARS and n_years >= MIN_YEARS_FOR_VERDICT:
        return "EXCLUDED"
    if (ic >= CORE_MIN_IC and pos_years >= CORE_MIN_POS_YEARS
            and pd.notna(regime_dep) and regime_dep <= CORE_MAX_REGIME_DEP):
        return "CORE"
    if ic >= 0 and pd.notna(regime_dep) and regime_dep > CORE_MAX_REGIME_DEP:
        return "REGIME_DEPENDENT"
    return "WATCHLIST"


def classify_subfactors(evidence: pd.DataFrame) -> pd.Series:
    """CORE / REGIME_DEPENDENT / WATCHLIST / EXCLUDED per row, absolute thresholds
    (not parent-relative) -- see spec Section 2 for the rule table."""
    return evidence.apply(_classify_row, axis=1)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_regime_aware_scoring.py -v -k classify`
Expected: PASS (1 test)

- [ ] **Step 5: Commit**

```bash
git add research/walkforward/regime_aware_scoring.py tests/test_regime_aware_scoring.py
git commit -m "feat: add CORE/REGIME_DEPENDENT/WATCHLIST/EXCLUDED classification"
```

- [ ] **Step 6: Write the failing tests for `score_table`**

```python
# append to tests/test_regime_aware_scoring.py
def test_score_table_ranks_within_parent_only():
    evidence = pd.DataFrame([
        # parent A: a1 dominates every metric, a2 is worse on every metric.
        {"sub_factor": "a1", "parent": "A", "expected_ic": 0.05, "long_run_mean_ic": 0.05,
         "long_run_std_ic": 0.02, "long_run_spread_ann": 0.08, "long_run_hit_rate": 0.7,
         "pct_positive_years": 0.8, "persistence_ir": 1.0, "spread_consistency": 0.9,
         "coverage": 0.95, "n_months": 60, "n_years": 5, "regime_dependence": 0.01},
        {"sub_factor": "a2", "parent": "A", "expected_ic": 0.01, "long_run_mean_ic": 0.01,
         "long_run_std_ic": 0.03, "long_run_spread_ann": 0.02, "long_run_hit_rate": 0.5,
         "pct_positive_years": 0.5, "persistence_ir": 0.2, "spread_consistency": 0.5,
         "coverage": 0.80, "n_months": 40, "n_years": 4, "regime_dependence": 0.02},
        # parent B: single, absolutely weaker sub -- should still rank 1.0 within its
        # own bucket, proving ranking is per-parent, not global.
        {"sub_factor": "b1", "parent": "B", "expected_ic": 0.001, "long_run_mean_ic": 0.001,
         "long_run_std_ic": 0.05, "long_run_spread_ann": 0.001, "long_run_hit_rate": 0.51,
         "pct_positive_years": 0.51, "persistence_ir": 0.1, "spread_consistency": 0.51,
         "coverage": 0.60, "n_months": 36, "n_years": 3, "regime_dependence": 0.01},
    ])
    out = score_table(evidence)
    a1 = out[out.sub_factor == "a1"].iloc[0]
    a2 = out[out.sub_factor == "a2"].iloc[0]
    b1 = out[out.sub_factor == "b1"].iloc[0]

    assert a1["production_score"] > a2["production_score"]
    assert b1["predictive_score"] == pytest.approx(1.0, abs=1e-9)
    assert b1["reliability_score"] == pytest.approx(1.0, abs=1e-9)
    assert a1["eligible"] and a2["eligible"] and b1["eligible"]
    assert a1["ic_ir"] == pytest.approx(0.05 / 0.02, abs=1e-9)


def test_score_table_ineligible_when_expected_ic_not_positive():
    evidence = pd.DataFrame([
        {"sub_factor": "neg1", "parent": "A", "expected_ic": -0.01, "long_run_mean_ic": -0.01,
         "long_run_std_ic": 0.02, "long_run_spread_ann": -0.01, "long_run_hit_rate": 0.4,
         "pct_positive_years": 0.3, "persistence_ir": -0.5, "spread_consistency": 0.3,
         "coverage": 0.9, "n_months": 50, "n_years": 4, "regime_dependence": 0.01},
    ])
    out = score_table(evidence)
    assert not out.iloc[0]["eligible"]
```

- [ ] **Step 7: Run to verify failure, then implement `score_table`**

Run: `pytest tests/test_regime_aware_scoring.py -v -k score_table` → FAIL

```python
# append to research/walkforward/regime_aware_scoring.py
def score_table(evidence: pd.DataFrame) -> pd.DataFrame:
    """Adds ic_ir, classification, an eligible flag (expected_ic > 0 and not
    EXCLUDED), and per-parent percentile-ranked Predictive/Reliability/Production
    scores. Ranking is grouped by ``parent`` -- a candidate's score reflects its
    standing among its own parent's siblings, never the global pool.
    """
    df = evidence.copy()
    df["ic_ir"] = (df["long_run_mean_ic"] / df["long_run_std_ic"]).replace(
        [np.inf, -np.inf], np.nan)
    df["classification"] = classify_subfactors(df)
    df["eligible"] = (df["expected_ic"] > 0) & (df["classification"] != "EXCLUDED")

    parts = []
    for _, g in df.groupby("parent"):
        g = g.copy()
        for metric in PREDICTIVE_WEIGHTS:
            g[f"rank_{metric}"] = _pct_rank(g[metric])
        for metric in RELIABILITY_WEIGHTS:
            g[f"rank_{metric}"] = _pct_rank(g[metric])
        g["predictive_score"] = sum(
            w * g[f"rank_{m}"] for m, w in PREDICTIVE_WEIGHTS.items())
        g["reliability_score"] = sum(
            w * g[f"rank_{m}"] for m, w in RELIABILITY_WEIGHTS.items())
        g["production_score"] = g["predictive_score"] * g["reliability_score"]
        parts.append(g)
    return pd.concat(parts, ignore_index=True) if parts else df
```

- [ ] **Step 8: Run tests to verify all pass**

Run: `pytest tests/test_regime_aware_scoring.py -v`
Expected: PASS (3 tests)

- [ ] **Step 9: Commit**

```bash
git add research/walkforward/regime_aware_scoring.py tests/test_regime_aware_scoring.py
git commit -m "feat: add Predictive/Reliability/Production scoring"
```

---

### Task 4: Subfactor Selection & Weighting (`regime_aware_selection.py`)

**Files:**
- Create: `research/walkforward/regime_aware_selection.py`
- Test: `tests/test_regime_aware_selection.py`

Reuses `parent_selection.py::select_subfactors()`'s R²-diversification loop
verbatim (fed Production Score, gated on `expected_ic` via the eligibility
pre-filter), then adds the new "materially lowers the parent" degradation
check and a 10% minimum-weight floor. See spec Section 3.

A note on the degradation check: the spec says to "rebuild the parent
composite including [a candidate] and recompute its Expected IC/IC-IR/Q5-Q1."
A naive weighted-average of the *individual* subs' own evidence metrics is
**not** a valid proxy for this -- blending a strong signal with a weaker-but-
diversifying one nearly always lowers a naive weighted-average IC even when
the real composite benefits from the lower correlation (that's the entire
point of diversification). So this task actually rebuilds the composite via
the existing `build_parent_panel` and recomputes its real IC from scores +
forward returns, not an approximation. The degradation-comparison *logic*
(`_apply_degradation_check`) takes that computation as an injected function so
it stays unit-testable without a real panel; `_composite_ic_stats` is the real
implementation wired in for actual use.

- [ ] **Step 1: Write the failing tests for `parent_subfactor_weights`**

```python
# tests/test_regime_aware_selection.py
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research import compute_forward_returns
from research.panel import ScorePanel
from research.walkforward.regime_aware_selection import (
    _apply_degradation_check, _composite_ic_stats, parent_subfactor_weights,
    select_parent_subfactors,
)


def test_parent_subfactor_weights_proportional_cap_and_floor():
    scores = pd.Series({"s1": 0.9, "s2": 0.08, "s3": 0.02})
    w = parent_subfactor_weights(["s1", "s2", "s3"], scores, cap=0.50, min_weight=0.10)
    assert abs(sum(w.values()) - 1.0) < 1e-9
    assert max(w.values()) <= 0.50 + 1e-9
    assert all(v >= 0.10 - 1e-9 for v in w.values())


def test_parent_subfactor_weights_single_sub_gets_full_weight():
    w = parent_subfactor_weights(["s1"], pd.Series({"s1": 0.5}))
    assert w == {"s1": pytest.approx(1.0)}


def test_parent_subfactor_weights_empty_selection():
    assert parent_subfactor_weights([], pd.Series(dtype=float)) == {}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_regime_aware_selection.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'research.walkforward.regime_aware_selection'`

- [ ] **Step 3: Implement `parent_subfactor_weights`**

```python
# research/walkforward/regime_aware_selection.py
"""Per-parent subfactor selection: reuses parent_selection.py's R²-diversification
loop (fed Production Score instead of the old blended Sub-factor Score, gated on
Expected IC via an eligibility pre-filter), adds a "materially lowers the parent"
post-hoc degradation check against a real recomputed composite, and reweights with
a 10% minimum-weight floor. See
docs/superpowers/specs/2026-07-11-hierarchical-regime-aware-factor-model-design.md
Section 3.
"""
from __future__ import annotations

from typing import Callable

import pandas as pd

from research.ic import period_ic
from research.panel import ScorePanel
from research.parent_selection import R2_MAX, SINGLE_CAP, select_subfactors
from research.quintiles import quintile_profile
from research.walkforward.compose import build_parent_panel

MAX_SUBS = 3
DEGRADATION_TOL = 0.10   # reject a candidate if it drops any parent metric >10% relative
MIN_WEIGHT = 0.10        # floor for any selected sub's final weight
MIN_NAMES = 20


def parent_subfactor_weights(
    selected: list[str], production_score: pd.Series, *,
    cap: float = SINGLE_CAP, min_weight: float = MIN_WEIGHT,
) -> dict[str, float]:
    """Weight ∝ positive Production Score, water-filled to ``cap``, then any share
    below ``min_weight`` is floored and the set renormalises. Flooring can only
    ever affect a strict subset (never all) when max_subs * min_weight <= 1 (true
    for the defaults 3 * 0.10 = 0.30) since weights summing to 1 can't all sit
    below 1/max_subs.
    """
    if not selected:
        return {}
    raw = {s: max(float(production_score.get(s, 0.0)), 0.0) for s in selected}
    total = sum(raw.values())
    w = ({s: 1.0 / len(selected) for s in selected} if total <= 1e-12
        else {s: raw[s] / total for s in selected})
    for _ in range(20):
        over = [s for s in w if w[s] > cap + 1e-12]
        if not over:
            break
        excess = sum(w[s] - cap for s in over)
        for s in over:
            w[s] = cap
        under = [s for s in w if w[s] < cap - 1e-12]
        pool = sum(w[s] for s in under)
        if not under or pool <= 1e-12:
            break
        for s in under:
            w[s] += excess * w[s] / pool
    below = [s for s in w if w[s] < min_weight]
    if below:
        for s in below:
            w[s] = min_weight
        total = sum(w.values())
        w = {s: v / total for s, v in w.items()}
    return w
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_regime_aware_selection.py -v -k parent_subfactor_weights`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add research/walkforward/regime_aware_selection.py tests/test_regime_aware_selection.py
git commit -m "feat: add proportional-weight-with-floor helper"
```

- [ ] **Step 6: Write the failing tests for `_apply_degradation_check`**

```python
# append to tests/test_regime_aware_selection.py
def test_apply_degradation_check_stops_on_material_drop():
    metric_map = pd.DataFrame({"production_score": [0.9, 0.5]}, index=["s1", "s2"])
    # Adding s2 drops mean_ic from 0.05 to 0.03 -- a 40% relative drop, breaching 10%.
    fake_stats = {
        frozenset({"s1"}): {"mean_ic": 0.05, "ic_ir": 1.0, "spread": 0.05},
        frozenset({"s1", "s2"}): {"mean_ic": 0.03, "ic_ir": 1.0, "spread": 0.05},
    }
    kept, stop = _apply_degradation_check(
        ["s1", "s2"], metric_map, lambda w: fake_stats[frozenset(w)],
        cap=0.5, min_weight=0.10, tol=0.10)
    assert kept == ["s1"]
    assert stop is not None and "degraded" in stop


def test_apply_degradation_check_keeps_non_degrading_addition():
    metric_map = pd.DataFrame({"production_score": [0.9, 0.5]}, index=["s1", "s2"])
    fake_stats = {
        frozenset({"s1"}): {"mean_ic": 0.05, "ic_ir": 1.0, "spread": 0.05},
        frozenset({"s1", "s2"}): {"mean_ic": 0.048, "ic_ir": 1.1, "spread": 0.06},
    }
    kept, stop = _apply_degradation_check(
        ["s1", "s2"], metric_map, lambda w: fake_stats[frozenset(w)],
        cap=0.5, min_weight=0.10, tol=0.10)
    assert kept == ["s1", "s2"]
    assert stop is None
```

- [ ] **Step 7: Run to verify failure, then implement `_apply_degradation_check`**

Run: `pytest tests/test_regime_aware_selection.py -v -k degradation_check` → FAIL

```python
# append to research/walkforward/regime_aware_selection.py
def _apply_degradation_check(
    selected: list[str], metric_map: pd.DataFrame,
    stats_fn: Callable[[dict[str, float]], dict[str, float]], *,
    cap: float, min_weight: float, tol: float,
) -> tuple[list[str], str | None]:
    """Walks ``selected`` in order, truncating at the first candidate whose addition
    drops any of (mean_ic, ic_ir, spread) -- as read from ``stats_fn(weights)`` --
    by more than ``tol`` relative to the metrics without it. ``stats_fn`` maps a
    {sub: weight} dict to {"mean_ic", "ic_ir", "spread"}; injected so this stays
    testable without a real ScorePanel (see ``_composite_ic_stats`` for the real one).
    """
    kept: list[str] = []
    for sub in selected:
        trial = kept + [sub]
        trial_w = parent_subfactor_weights(trial, metric_map["production_score"],
                                           cap=cap, min_weight=min_weight)
        if kept:
            base_w = parent_subfactor_weights(kept, metric_map["production_score"],
                                              cap=cap, min_weight=min_weight)
            base_stats, trial_stats = stats_fn(base_w), stats_fn(trial_w)
            degraded = any(
                pd.notna(base_stats[m]) and pd.notna(trial_stats[m]) and base_stats[m] != 0
                and (trial_stats[m] - base_stats[m]) / abs(base_stats[m]) < -tol
                for m in ("mean_ic", "ic_ir", "spread"))
            if degraded:
                return kept, f"adding `{sub}` degraded a parent metric > {tol:.0%}"
        kept.append(sub)
    return kept, None
```

- [ ] **Step 8: Run tests to verify they pass**

Run: `pytest tests/test_regime_aware_selection.py -v -k degradation_check`
Expected: PASS (2 tests)

- [ ] **Step 9: Commit**

```bash
git add research/walkforward/regime_aware_selection.py tests/test_regime_aware_selection.py
git commit -m "feat: add parent-degradation stop-check (injectable stats)"
```

- [ ] **Step 10: Write the failing test for `_composite_ic_stats`**

```python
# append to tests/test_regime_aware_selection.py
UNIVERSE = [f"T{i}" for i in range(30)]
DATES = pd.date_range("2018-01-31", periods=10, freq="ME").strftime("%Y-%m-%d").tolist()


def _toy_panel(dates: list[str], universe: list[str]) -> ScorePanel:
    """One parent 'p1' with two subs: s1 ranks ascending with ticker order, s2 ranks
    descending -- same construction as tests/test_regime_aware_evidence.py."""
    scores = {}
    for k, d in enumerate(dates):
        base = np.linspace(10, 90, len(universe)) + k
        scores[d] = pd.DataFrame({"s1": base, "s2": base[::-1]}, index=universe)
    return ScorePanel(rebal_dates=list(dates), scores=scores,
                      parent_keys=["p1"], sub_by_parent={"p1": ["s1", "s2"]},
                      universe=list(universe))


def _toy_matrix(dates: list[str], universe: list[str]) -> pd.DataFrame:
    """Prices grow faster for higher-index tickers -> a signal ranking tickers
    ascending (s1) has positive IC; descending (s2) has negative IC."""
    idx = pd.date_range(dates[0], periods=len(dates) + 8, freq="ME").strftime("%Y-%m-%d")
    data = {t: (1.01 + i * 0.002) ** np.arange(len(idx)) for i, t in enumerate(universe)}
    return pd.DataFrame(data, index=idx)


def test_composite_ic_stats_computes_real_composite_metrics():
    panel = _toy_panel(DATES, UNIVERSE)
    matrix = _toy_matrix(DATES, UNIVERSE)
    fwd_by_h = compute_forward_returns(matrix, DATES, {"3M": 3, "6M": 6})
    stats_s1 = _composite_ic_stats(panel, {"s1": 1.0}, fwd_by_h)
    stats_s2 = _composite_ic_stats(panel, {"s2": 1.0}, fwd_by_h)
    assert stats_s1["mean_ic"] > 0
    assert stats_s2["mean_ic"] < 0


def test_composite_ic_stats_empty_weights_returns_nan():
    panel = _toy_panel(DATES, UNIVERSE)
    matrix = _toy_matrix(DATES, UNIVERSE)
    fwd_by_h = compute_forward_returns(matrix, DATES, {"3M": 3, "6M": 6})
    stats = _composite_ic_stats(panel, {}, fwd_by_h)
    assert all(v != v for v in stats.values())   # all NaN
```

- [ ] **Step 11: Run to verify failure, then implement `_composite_ic_stats`**

Run: `pytest tests/test_regime_aware_selection.py -v -k composite_ic_stats` → FAIL

```python
# append to research/walkforward/regime_aware_selection.py
def _composite_ic_stats(
    sub_panel: ScorePanel, weights: dict[str, float],
    fwd_by_h: dict[str, dict[str, pd.Series]], *, min_names: int = MIN_NAMES,
) -> dict[str, float]:
    """Real long-run mean IC / IC-IR / Q5-Q1 spread of the composite built from
    ``weights`` over every date in ``sub_panel`` -- used only for the
    materially-lowers-the-parent degradation check, so it stays a plain long-run
    read (no recent/regime blend needed for this local comparison).
    """
    if not weights:
        return {"mean_ic": float("nan"), "ic_ir": float("nan"), "spread": float("nan")}
    trial_panel = build_parent_panel(sub_panel, {"__trial__": weights})
    ics: list[float] = []
    spreads: list[float] = []
    for d in trial_panel.rebal_dates:
        frame = trial_panel.scores.get(d)
        if frame is None or "__trial__" not in frame.columns:
            continue
        score = frame["__trial__"]
        for h in ("3M", "6M"):
            fwd = fwd_by_h.get(h, {}).get(d)
            if fwd is None:
                continue
            ic = period_ic(score, fwd, min_names=min_names)
            if ic is not None:
                ics.append(ic)
        fwd6 = fwd_by_h.get("6M", {}).get(d)
        if fwd6 is not None:
            prof = quintile_profile(score, fwd6, min_names=min_names)
            if prof is not None:
                spreads.append(float(prof[-1] - prof[0]))
    ic_s = pd.Series(ics, dtype=float)
    mean_ic = float(ic_s.mean()) if not ic_s.empty else float("nan")
    std_ic = float(ic_s.std(ddof=1)) if len(ic_s) > 1 else float("nan")
    ic_ir = mean_ic / std_ic if std_ic and std_ic > 1e-9 else float("nan")
    spread = float(pd.Series(spreads).mean() * 2.0) if spreads else float("nan")
    return {"mean_ic": mean_ic, "ic_ir": ic_ir, "spread": spread}
```

- [ ] **Step 12: Run tests to verify they pass**

Run: `pytest tests/test_regime_aware_selection.py -v -k composite_ic_stats`
Expected: PASS (2 tests)

- [ ] **Step 13: Commit**

```bash
git add research/walkforward/regime_aware_selection.py tests/test_regime_aware_selection.py
git commit -m "feat: add real composite IC recomputation for degradation checks"
```

- [ ] **Step 14: Write the failing tests for `select_parent_subfactors`**

```python
# append to tests/test_regime_aware_selection.py
def test_select_parent_subfactors_end_to_end_wiring():
    """s2 is ineligible (negative expected_ic) so it must never be considered, and
    s1 alone should be selected and weighted 100%."""
    panel = _toy_panel(DATES, UNIVERSE)
    matrix = _toy_matrix(DATES, UNIVERSE)
    fwd_by_h = compute_forward_returns(matrix, DATES, {"3M": 3, "6M": 6})
    corr = pd.DataFrame({"s1": [1.0, -1.0], "s2": [-1.0, 1.0]}, index=["s1", "s2"])
    scored = pd.DataFrame([
        {"sub_factor": "s1", "parent": "p1", "expected_ic": 0.05, "production_score": 0.9,
         "eligible": True},
        {"sub_factor": "s2", "parent": "p1", "expected_ic": -0.05, "production_score": 0.1,
         "eligible": False},
    ])
    result = select_parent_subfactors(scored, corr, panel, fwd_by_h)
    assert result["selected"] == ["s1"]
    assert result["weights"] == {"s1": pytest.approx(1.0)}


def test_select_parent_subfactors_rejects_redundant_sub():
    """s1 and s1_dup are perfectly correlated (R²=1.0 >= the 0.60 gate); only the
    higher-ranked one should survive, even though both are eligible."""
    panel = _toy_panel(DATES, UNIVERSE)
    matrix = _toy_matrix(DATES, UNIVERSE)
    fwd_by_h = compute_forward_returns(matrix, DATES, {"3M": 3, "6M": 6})
    corr = pd.DataFrame({"s1": [1.0, 1.0], "s1_dup": [1.0, 1.0]}, index=["s1", "s1_dup"])
    scored = pd.DataFrame([
        {"sub_factor": "s1", "parent": "p1", "expected_ic": 0.05, "production_score": 0.9,
         "eligible": True},
        {"sub_factor": "s1_dup", "parent": "p1", "expected_ic": 0.04, "production_score": 0.7,
         "eligible": True},
    ])
    result = select_parent_subfactors(scored, corr, panel, fwd_by_h)
    assert result["selected"] == ["s1"]
    assert "s1_dup" not in result["weights"]


def test_select_parent_subfactors_no_eligible_candidates():
    scored = pd.DataFrame([
        {"sub_factor": "s1", "parent": "p1", "expected_ic": -0.01, "production_score": 0.9,
         "eligible": False},
    ])
    corr = pd.DataFrame({"s1": [1.0]}, index=["s1"])
    panel = _toy_panel(DATES, UNIVERSE)
    matrix = _toy_matrix(DATES, UNIVERSE)
    fwd_by_h = compute_forward_returns(matrix, DATES, {"3M": 3, "6M": 6})
    result = select_parent_subfactors(scored, corr, panel, fwd_by_h)
    assert result == {"selected": [], "weights": {}, "stop_reason": "no eligible candidate",
                      "decisions": []}
```

- [ ] **Step 15: Run to verify failure, then implement `select_parent_subfactors`**

Run: `pytest tests/test_regime_aware_selection.py -v -k select_parent_subfactors` → FAIL

```python
# append to research/walkforward/regime_aware_selection.py
def select_parent_subfactors(
    scored: pd.DataFrame, corr: pd.DataFrame,
    sub_panel: ScorePanel, fwd_by_h: dict[str, dict[str, pd.Series]], *,
    r2_max: float = R2_MAX, max_subs: int = MAX_SUBS,
    degradation_tol: float = DEGRADATION_TOL, min_weight: float = MIN_WEIGHT,
    single_cap: float = SINGLE_CAP,
) -> dict:
    """Select + weight one parent's subfactors from its ``scored`` evidence rows
    (regime_aware_scoring.score_table output, already filtered to one parent).

    Returns {"selected": [...], "weights": {sub: weight}, "stop_reason": str,
    "decisions": list[dict]}.
    """
    eligible = scored[scored["eligible"]].copy()
    if eligible.empty:
        return {"selected": [], "weights": {}, "stop_reason": "no eligible candidate",
               "decisions": []}
    ranked = (eligible.sort_values("production_score", ascending=False)
             .rename(columns={"expected_ic": "mean_ic"})   # reuse select_subfactors' IC gate
             .reset_index(drop=True))
    candidates, decisions, stop_reason = select_subfactors(
        ranked, corr, r2_max=r2_max, max_subs=max_subs)

    metric_map = eligible.set_index("sub_factor")
    kept, degrade_stop = _apply_degradation_check(
        candidates, metric_map,
        lambda w: _composite_ic_stats(sub_panel, w, fwd_by_h),
        cap=single_cap, min_weight=min_weight, tol=degradation_tol)
    if degrade_stop is not None:
        stop_reason = degrade_stop

    weights = parent_subfactor_weights(kept, metric_map["production_score"],
                                       cap=single_cap, min_weight=min_weight)
    return {"selected": kept, "weights": weights, "stop_reason": stop_reason,
           "decisions": decisions}
```

- [ ] **Step 16: Run tests to verify all pass**

Run: `pytest tests/test_regime_aware_selection.py -v`
Expected: PASS (10 tests)

- [ ] **Step 17: Commit**

```bash
git add research/walkforward/regime_aware_selection.py tests/test_regime_aware_selection.py
git commit -m "feat: wire up select_parent_subfactors end to end"
```

---

### Task 5: Parent Weighting & Hysteresis (`regime_aware_parents.py`)

**Files:**
- Create: `research/walkforward/regime_aware_parents.py`
- Test: `tests/test_regime_aware_parents.py`

Parent utility score, the 70/30 base/adaptive weight blend, monthly/quarterly
change caps, and the subfactor-membership hysteresis state machine. See spec
Section 3.

Key reuse insight: `regime_aware_evidence.build_monthly_cache`/`as_of()`
(Task 2) don't actually care whether the `ScorePanel` they're given holds raw
subfactors or a single constructed composite per parent -- `compose.py`'s
existing `build_parent_panel` already returns exactly that shape (one signal
per parent, self-mapped as its own "sub"). So parent-level long-run/recent/
regime evidence needs **no new evidence code** -- Task 6 just calls
`build_monthly_cache`/`as_of()` again on the constructed parent panel. This
file only adds what's genuinely new: ranking those 8 parents against each
other, blending 70% base + 30% adaptive weight, and the two kinds of change
control (weight caps, membership hysteresis).

- [ ] **Step 1: Write the failing test for `parent_utility_table`**

```python
# tests/test_regime_aware_parents.py
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.walkforward.regime_aware_parents import (
    HysteresisState, adaptive_weight, apply_change_caps, apply_quarterly_membership,
    blend_parent_weights, parent_utility_table, update_streaks,
)


def test_parent_utility_table_ranks_across_all_parents():
    evidence = pd.DataFrame([
        {"sub_factor": "A", "parent": "A", "expected_ic": 0.05, "long_run_mean_ic": 0.05,
         "long_run_std_ic": 0.02, "long_run_spread_ann": 0.08, "long_run_hit_rate": 0.7,
         "pct_positive_years": 0.8, "persistence_ir": 1.0, "spread_consistency": 0.9,
         "coverage": 0.95, "n_months": 60},
        {"sub_factor": "B", "parent": "B", "expected_ic": 0.01, "long_run_mean_ic": 0.01,
         "long_run_std_ic": 0.03, "long_run_spread_ann": 0.02, "long_run_hit_rate": 0.5,
         "pct_positive_years": 0.5, "persistence_ir": 0.2, "spread_consistency": 0.5,
         "coverage": 0.80, "n_months": 40},
    ])
    out = parent_utility_table(evidence)
    a = out[out.sub_factor == "A"].iloc[0]
    b = out[out.sub_factor == "B"].iloc[0]
    assert a["utility_score"] > b["utility_score"]
    assert a["utility_score"] == pytest.approx(1.0, abs=1e-9)   # dominates every metric
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_regime_aware_parents.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'research.walkforward.regime_aware_parents'`

- [ ] **Step 3: Implement `parent_utility_table`**

```python
# research/walkforward/regime_aware_parents.py
"""Parent utility score, 70/30 base/adaptive weight blend, monthly/quarterly
change caps, and the subfactor-membership hysteresis state machine. See
docs/superpowers/specs/2026-07-11-hierarchical-regime-aware-factor-model-design.md
Section 3.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from research.parent_selection import _pct_rank
from research.walkforward.regime_aware_scoring import PREDICTIVE_WEIGHTS, RELIABILITY_WEIGHTS
from research.walkforward.vix_overlay import _water_fill

PARENT_CAP = 0.25
MONTHLY_CAP = 0.02
QUARTERLY_CAP = 0.05
ENTRY_MONTHS = 2
EXIT_MONTHS = 3
MAX_SUBS = 3


def parent_utility_table(parent_evidence: pd.DataFrame) -> pd.DataFrame:
    """Same Predictive x Reliability construction as regime_aware_scoring.score_table,
    but ranked across ALL parents at once (no sub-grouping at this level -- every
    parent competes against every other parent). ``parent_evidence`` is the output
    of regime_aware_evidence.as_of() run on a parent-composite panel (one row per
    parent, with "sub_factor" == "parent" == the parent's own name).
    """
    df = parent_evidence.copy()
    df["ic_ir"] = (df["long_run_mean_ic"] / df["long_run_std_ic"]).replace(
        [np.inf, -np.inf], np.nan)
    for metric in PREDICTIVE_WEIGHTS:
        df[f"rank_{metric}"] = _pct_rank(df[metric])
    for metric in RELIABILITY_WEIGHTS:
        df[f"rank_{metric}"] = _pct_rank(df[metric])
    df["predictive_score"] = sum(
        w * df[f"rank_{m}"] for m, w in PREDICTIVE_WEIGHTS.items())
    df["reliability_score"] = sum(
        w * df[f"rank_{m}"] for m, w in RELIABILITY_WEIGHTS.items())
    df["utility_score"] = df["predictive_score"] * df["reliability_score"]
    return df
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_regime_aware_parents.py -v -k utility_table`
Expected: PASS (1 test)

- [ ] **Step 5: Commit**

```bash
git add research/walkforward/regime_aware_parents.py tests/test_regime_aware_parents.py
git commit -m "feat: add cross-parent utility scoring"
```

- [ ] **Step 6: Write the failing tests for `adaptive_weight` and `blend_parent_weights`**

```python
# append to tests/test_regime_aware_parents.py
def test_adaptive_weight_excludes_negative_ic_parents():
    # 4 eligible parents (p2 excluded) so a 25% cap is actually satisfiable at sum=1
    # (with only 2-3 eligible parents, 0.25 * n < 1.0 makes cap and sum=1 mutually
    # infeasible, and _water_fill will breach the cap to preserve sum=1 -- see
    # research/walkforward/vix_overlay.py::_water_fill).
    utility = pd.Series({"p1": 0.9, "p2": 0.6, "p3": 0.5, "p4": 0.4, "p5": 0.3})
    expected_ic = pd.Series({"p1": 0.02, "p2": -0.01, "p3": 0.03, "p4": 0.01, "p5": 0.02})
    w = adaptive_weight(utility, expected_ic, cap=0.25)
    assert "p2" not in w
    assert abs(sum(w.values()) - 1.0) < 1e-9
    assert max(w.values()) <= 0.25 + 1e-9


def test_blend_parent_weights_zeroes_negative_ic_and_renormalizes():
    # 4 nonzero-after-zeroing parents (p2 zeroed) so cap=0.25 stays feasible at sum=1
    # -- see the feasibility note on test_adaptive_weight_excludes_negative_ic_parents.
    base = {"p1": 0.30, "p2": 0.25, "p3": 0.20, "p4": 0.15, "p5": 0.10}
    adaptive = {"p1": 0.30, "p3": 0.30, "p4": 0.25, "p5": 0.15}    # p2: no adaptive weight
    expected_ic = {"p1": 0.02, "p2": -0.01, "p3": 0.03, "p4": 0.01, "p5": 0.015}
    out = blend_parent_weights(base, adaptive, expected_ic, base_weight_frac=0.70, cap=0.25)
    assert out.get("p2", 0.0) == pytest.approx(0.0, abs=1e-9)
    assert abs(sum(out.values()) - 1.0) < 1e-9
    assert max(out.values()) <= 0.25 + 1e-9
```

- [ ] **Step 7: Run to verify failure, then implement both functions**

Run: `pytest tests/test_regime_aware_parents.py -v -k "adaptive_weight or blend_parent_weights"` → FAIL

```python
# append to research/walkforward/regime_aware_parents.py
def adaptive_weight(utility: pd.Series, expected_ic: pd.Series, *,
                    cap: float = PARENT_CAP) -> dict[str, float]:
    """Water-fill by positive Parent Utility, restricted to parents with positive
    expected_ic -- the "30%" adaptive component of the parent-weight blend."""
    eligible = {p: max(float(utility.get(p, 0.0)), 0.0) for p in utility.index
               if expected_ic.get(p, float("nan")) > 0}
    return _water_fill(eligible, cap)


def blend_parent_weights(
    base: dict[str, float], adaptive: dict[str, float], expected_ic: dict[str, float], *,
    base_weight_frac: float = 0.70, cap: float = PARENT_CAP,
) -> dict[str, float]:
    """raw = base_weight_frac*base + (1-base_weight_frac)*adaptive; any parent with
    expected_ic <= 0 is zeroed; water-fill renormalise to ``cap``."""
    all_p = set(base) | set(adaptive)
    raw = {p: base_weight_frac * base.get(p, 0.0)
          + (1 - base_weight_frac) * adaptive.get(p, 0.0) for p in all_p}
    raw = {p: (0.0 if expected_ic.get(p, float("nan")) <= 0 else v) for p, v in raw.items()}
    return _water_fill(raw, cap)
```

- [ ] **Step 8: Run tests to verify they pass**

Run: `pytest tests/test_regime_aware_parents.py -v -k "adaptive_weight or blend_parent_weights"`
Expected: PASS (2 tests)

- [ ] **Step 9: Commit**

```bash
git add research/walkforward/regime_aware_parents.py tests/test_regime_aware_parents.py
git commit -m "feat: add 70/30 base/adaptive parent-weight blend"
```

- [ ] **Step 10: Write the failing tests for `apply_change_caps`**

```python
# append to tests/test_regime_aware_parents.py
def test_apply_change_caps_limits_monthly_move():
    target = {"p1": 0.30, "p2": 0.70}          # a big jump from prior weights
    prev_month = {"p1": 0.10, "p2": 0.90}
    quarter_start = {"p1": 0.10, "p2": 0.90}
    out = apply_change_caps(target, prev_month, quarter_start,
                            monthly_cap=0.02, quarterly_cap=0.05)
    assert out["p1"] <= 0.10 + 0.02 + 1e-9
    assert abs(sum(out.values()) - 1.0) < 1e-9


def test_apply_change_caps_limits_quarterly_move_by_third_month():
    """Three consecutive monthly calls, all pushing the same direction: by the third
    month the quarterly +/-5pp band is what's binding, not the monthly +/-2pp band."""
    quarter_start = {"p1": 0.10, "p2": 0.90}
    prev = dict(quarter_start)
    target = {"p1": 0.30, "p2": 0.70}
    for _ in range(3):
        prev = apply_change_caps(target, prev, quarter_start,
                                 monthly_cap=0.02, quarterly_cap=0.05)
    assert prev["p1"] == pytest.approx(quarter_start["p1"] + 0.05, abs=1e-9)
```

- [ ] **Step 11: Run to verify failure, then implement `apply_change_caps`**

Run: `pytest tests/test_regime_aware_parents.py -v -k apply_change_caps` → FAIL

```python
# append to research/walkforward/regime_aware_parents.py
def apply_change_caps(
    target: dict[str, float], prev_month_w: dict[str, float], quarter_start_w: dict[str, float],
    *, monthly_cap: float = MONTHLY_CAP, quarterly_cap: float = QUARTERLY_CAP,
) -> dict[str, float]:
    """Clips ``target`` toward [prev_month_w +/- monthly_cap] intersected with
    [quarter_start_w +/- quarterly_cap], then renormalises to sum to 1. The 25%
    parent cap is already enforced upstream in blend_parent_weights, so it is not
    re-applied here.
    """
    all_p = set(target) | set(prev_month_w) | set(quarter_start_w)
    clipped: dict[str, float] = {}
    for p in all_p:
        v = target.get(p, 0.0)
        pm = prev_month_w.get(p, 0.0)
        qs = quarter_start_w.get(p, 0.0)
        lo = max(pm - monthly_cap, qs - quarterly_cap, 0.0)
        hi = max(min(pm + monthly_cap, qs + quarterly_cap), lo)
        clipped[p] = float(np.clip(v, lo, hi))
    total = sum(clipped.values())
    return {p: v / total for p, v in clipped.items()} if total > 1e-9 else clipped
```

- [ ] **Step 12: Run tests to verify they pass**

Run: `pytest tests/test_regime_aware_parents.py -v -k apply_change_caps`
Expected: PASS (2 tests)

- [ ] **Step 13: Commit**

```bash
git add research/walkforward/regime_aware_parents.py tests/test_regime_aware_parents.py
git commit -m "feat: add monthly/quarterly parent-weight change caps"
```

- [ ] **Step 14: Write the failing tests for the hysteresis state machine**

```python
# append to tests/test_regime_aware_parents.py
def test_update_streaks_tracks_above_and_below():
    state = HysteresisState()
    update_streaks(state, would_select=["s1"], eligible={"s1", "s2"}, immediate_exit=set())
    assert state.above_streak["s1"] == 1
    assert state.below_streak.get("s2", 0) == 1    # eligible but not picked -> below streak
    update_streaks(state, would_select=["s1"], eligible={"s1", "s2"}, immediate_exit=set())
    assert state.above_streak["s1"] == 2


def test_apply_quarterly_membership_entry_requires_two_months():
    state = HysteresisState()
    update_streaks(state, would_select=["s1"], eligible={"s1"}, immediate_exit=set())
    apply_quarterly_membership(state, would_select=["s1"])
    assert "s1" not in state.members            # only 1 month so far -> not enough
    update_streaks(state, would_select=["s1"], eligible={"s1"}, immediate_exit=set())
    apply_quarterly_membership(state, would_select=["s1"])
    assert "s1" in state.members                 # 2 consecutive months -> entry


def test_apply_quarterly_membership_exit_requires_three_months():
    state = HysteresisState(members={"s1"})
    for _ in range(2):
        update_streaks(state, would_select=[], eligible=set(), immediate_exit=set())
    apply_quarterly_membership(state, would_select=[])
    assert "s1" in state.members                 # only 2 months below -> not enough yet
    update_streaks(state, would_select=[], eligible=set(), immediate_exit=set())
    apply_quarterly_membership(state, would_select=[])
    assert "s1" not in state.members              # 3rd consecutive month -> exit


def test_update_streaks_immediate_exit_bypasses_hysteresis():
    state = HysteresisState(members={"s1"}, above_streak={"s1": 5})
    update_streaks(state, would_select=["s1"], eligible={"s1"}, immediate_exit={"s1"})
    assert "s1" not in state.members              # removed immediately despite a strong streak


def test_apply_quarterly_membership_respects_max_subs():
    state = HysteresisState(members={"s1", "s2", "s3"})
    for _ in range(2):
        update_streaks(state, would_select=["s4"], eligible={"s1", "s2", "s3", "s4"},
                       immediate_exit=set())
    apply_quarterly_membership(state, would_select=["s4"], max_subs=3)
    assert "s4" not in state.members               # no free slot -- 3 members block entry
```

- [ ] **Step 15: Run to verify failure, then implement the hysteresis state machine**

Run: `pytest tests/test_regime_aware_parents.py -v -k "streak or membership"` → FAIL

```python
# append to research/walkforward/regime_aware_parents.py
@dataclass
class HysteresisState:
    """Per-parent subfactor-membership hysteresis, carried month-over-month.

    ``members`` is the currently active (quarter-frozen) selected-sub set.
    ``above_streak``/``below_streak`` count consecutive months each sub has been
    eligible-and-selectable / not, reset whenever the direction flips.
    """
    members: set[str] = field(default_factory=set)
    above_streak: dict[str, int] = field(default_factory=dict)
    below_streak: dict[str, int] = field(default_factory=dict)


def update_streaks(state: HysteresisState, would_select: list[str], eligible: set[str],
                   immediate_exit: set[str]) -> None:
    """Call every month. ``would_select`` is this month's one-shot greedy selection
    (regime_aware_selection.select_parent_subfactors' ``selected``, priority order);
    ``eligible`` is subs with expected_ic > 0 and not EXCLUDED; ``immediate_exit`` is
    subs with a sign-inversion/coverage-collapse flag -- removed from ``members``
    immediately, bypassing hysteresis entirely (spec Section 3).
    """
    state.members -= immediate_exit
    would_select_set = set(would_select)
    all_seen = (would_select_set | eligible | state.members
               | set(state.above_streak) | set(state.below_streak))
    for sub in all_seen:
        selectable = sub in would_select_set and sub in eligible
        if selectable:
            state.above_streak[sub] = state.above_streak.get(sub, 0) + 1
            state.below_streak[sub] = 0
        else:
            state.below_streak[sub] = state.below_streak.get(sub, 0) + 1
            state.above_streak[sub] = 0


def apply_quarterly_membership(state: HysteresisState, would_select: list[str], *,
                               max_subs: int = MAX_SUBS) -> None:
    """Call only at a calendar-quarter boundary (Jan/Apr/Jul/Oct). Exits are
    evaluated for every current member first (freeing a slot); entries are then
    considered in ``would_select`` priority order, capped at ``max_subs`` members.
    """
    for sub in list(state.members):
        if state.below_streak.get(sub, 0) >= EXIT_MONTHS:
            state.members.discard(sub)
    for sub in would_select:
        if len(state.members) >= max_subs:
            break
        if sub not in state.members and state.above_streak.get(sub, 0) >= ENTRY_MONTHS:
            state.members.add(sub)
```

- [ ] **Step 16: Run tests to verify all pass**

Run: `pytest tests/test_regime_aware_parents.py -v`
Expected: PASS (10 tests)

- [ ] **Step 17: Commit**

```bash
git add research/walkforward/regime_aware_parents.py tests/test_regime_aware_parents.py
git commit -m "feat: add subfactor-membership hysteresis state machine"
```

---

### Task 6: The Continuous Monthly Walk-Forward (`regime_aware_walkforward.py`)

**Files:**
- Create: `research/walkforward/regime_aware_walkforward.py`
- Test: `tests/test_regime_aware_walkforward.py`

Ties Tasks 1-5 together: one continuous monthly loop (variants B/C/D) that
carries hysteresis/weight state across the whole history and snapshots a
`FrozenConfig` at each of the same 19 semiannual boundaries every other
regime study in this repo uses, plus variant A (the existing baseline,
called unmodified) and the metric computation that scores every variant on
each window. See spec Section 4.

Testing approach: the small pure helpers below get full TDD unit tests; the
two orchestration functions (`run_monthly_variant`,
`run_walkforward_comparison`) get one smoke test each against a tiny
synthetic panel, proving the wiring is correct end to end -- their
*components* are already thoroughly unit-tested in Tasks 1-5. This mirrors
how this repo already tests its other walk-forward orchestrators
(`run_vix_study`/`run_overlay_study` in `vix_regime_study.py`/`vix_overlay.py`
are not unit-tested directly either -- only their pieces, plus one
DB-guarded reproduction test, which Task 8 adds for this module).

- [ ] **Step 1: Write the failing tests for `_is_quarter_boundary` and `_apply_variant_ic`**

```python
# tests/test_regime_aware_walkforward.py
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research import compute_forward_returns
from research.panel import ScorePanel
from research.walkforward.regime_aware_evidence import build_monthly_cache
from research.walkforward.regime_aware_walkforward import (
    VARIANTS, _apply_floor, _apply_variant_ic, _is_quarter_boundary,
    run_monthly_variant, run_walkforward_comparison,
)


def test_is_quarter_boundary():
    assert _is_quarter_boundary("2020-01-31")
    assert _is_quarter_boundary("2020-04-30")
    assert _is_quarter_boundary("2020-07-31")
    assert _is_quarter_boundary("2020-10-31")
    assert not _is_quarter_boundary("2020-02-29")
    assert not _is_quarter_boundary("2020-06-30")


def test_apply_variant_ic_long_run_mode():
    evidence = pd.DataFrame({"expected_ic": [0.05], "long_run_mean_ic": [0.02],
                             "recent_24m_ic": [0.09]})
    out = _apply_variant_ic(evidence, "long_run")
    assert out["expected_ic"].iloc[0] == pytest.approx(0.02)


def test_apply_variant_ic_recent_mode_falls_back_to_long_run():
    evidence = pd.DataFrame({"expected_ic": [0.05], "long_run_mean_ic": [0.02],
                             "recent_24m_ic": [np.nan]})
    out = _apply_variant_ic(evidence, "recent")
    assert out["expected_ic"].iloc[0] == pytest.approx(0.02)


def test_apply_variant_ic_full_mode_is_unchanged():
    evidence = pd.DataFrame({"expected_ic": [0.05], "long_run_mean_ic": [0.02],
                             "recent_24m_ic": [0.09]})
    out = _apply_variant_ic(evidence, "full")
    assert out["expected_ic"].iloc[0] == pytest.approx(0.05)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_regime_aware_walkforward.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'research.walkforward.regime_aware_walkforward'`

- [ ] **Step 3: Implement `_is_quarter_boundary`, `_apply_variant_ic`, and the `VariantSpec` registry**

```python
# research/walkforward/regime_aware_walkforward.py
"""The continuous monthly walk-forward: ties regime_probability, regime_aware_
evidence, regime_aware_scoring, regime_aware_selection, and regime_aware_parents
together into variants A/B/C/D(/+floor) and scores each on the same 19-window
semiannual rolling-5Y grid every other regime study in this repo uses. See
docs/superpowers/specs/2026-07-11-hierarchical-regime-aware-factor-model-design.md
Section 4.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from research import compute_forward_returns
from research.panel import ScorePanel
from research.subset_selection import correlation_matrix
from research.walkforward import analysis
from research.walkforward import portfolio as pf
from research.walkforward.compose import FrozenConfig, build_parent_panel, frozen_composite
from research.walkforward.regime_aware_evidence import as_of, build_monthly_cache
from research.walkforward.regime_aware_parents import (
    HysteresisState, adaptive_weight, apply_change_caps, apply_quarterly_membership,
    blend_parent_weights, parent_utility_table, update_streaks,
)
from research.walkforward.regime_aware_scoring import score_table
from research.walkforward.regime_aware_selection import (
    parent_subfactor_weights, select_parent_subfactors,
)
from research.walkforward.selection import select_config, slice_panel
from research.walkforward.splits import (
    DATA_START, SELECTION_HORIZON_MONTHS, semiannual_policy_splits,
)


@dataclass(frozen=True)
class VariantSpec:
    name: str
    expected_ic_mode: str          # "full" | "long_run" | "recent"
    hysteresis: bool
    weight_caps: bool
    diversification_floor: float | None = None


VARIANTS: dict[str, VariantSpec] = {
    "B": VariantSpec("B", "long_run", hysteresis=True, weight_caps=True),
    "C": VariantSpec("C", "full", hysteresis=True, weight_caps=True),
    "D": VariantSpec("D", "recent", hysteresis=False, weight_caps=False),
    "C+2%floor": VariantSpec("C+2%floor", "full", hysteresis=True, weight_caps=True,
                             diversification_floor=0.02),
    "C+3%floor": VariantSpec("C+3%floor", "full", hysteresis=True, weight_caps=True,
                             diversification_floor=0.03),
}


def _is_quarter_boundary(cutoff: str) -> bool:
    return pd.Timestamp(cutoff).month in (1, 4, 7, 10)


def _apply_variant_ic(evidence: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Overrides the "expected_ic" column per variant (spec Section 4): "full" is
    the Section 1 blend computed by as_of() (left unchanged); "long_run" (variant B)
    and "recent" (variant D) read a single evidence column directly, with "recent"
    falling back to long-run when fewer than 24 months of history exist yet."""
    df = evidence.copy()
    if mode == "long_run":
        df["expected_ic"] = df["long_run_mean_ic"]
    elif mode == "recent":
        df["expected_ic"] = df["recent_24m_ic"].where(
            df["recent_24m_ic"].notna(), df["long_run_mean_ic"])
    return df
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_regime_aware_walkforward.py -v -k "quarter_boundary or apply_variant_ic"`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add research/walkforward/regime_aware_walkforward.py tests/test_regime_aware_walkforward.py
git commit -m "feat: add variant registry and IC-mode switching"
```

- [ ] **Step 6: Write the failing tests for `_apply_floor`**

```python
# append to tests/test_regime_aware_walkforward.py
def test_apply_floor_raises_small_positive_weights():
    weights = {"p1": 0.01, "p2": 0.50, "p3": 0.49}
    out = _apply_floor(weights, 0.02)
    assert out["p1"] == pytest.approx(0.02)
    assert abs(sum(out.values()) - 1.0) < 1e-9


def test_apply_floor_no_op_when_all_above_floor():
    weights = {"p1": 0.5, "p2": 0.5}
    assert _apply_floor(weights, 0.02) == weights
```

- [ ] **Step 7: Run to verify failure, then implement `_apply_floor`**

Run: `pytest tests/test_regime_aware_walkforward.py -v -k apply_floor` → FAIL

```python
# append to research/walkforward/regime_aware_walkforward.py
def _apply_floor(weights: dict[str, float], floor: float) -> dict[str, float]:
    """Floors every parent with a positive-but-below-floor weight up to ``floor``,
    then renormalises. The "separately tested variant" diversification floor from
    spec Section 3/5 -- not part of the primary C construction."""
    below = {p: v for p, v in weights.items() if 0 < v < floor}
    if not below:
        return weights
    w = dict(weights)
    for p in below:
        w[p] = floor
    total = sum(w.values())
    return {p: v / total for p, v in w.items()}
```

- [ ] **Step 8: Run tests to verify they pass**

Run: `pytest tests/test_regime_aware_walkforward.py -v -k apply_floor`
Expected: PASS (2 tests)

- [ ] **Step 9: Commit**

```bash
git add research/walkforward/regime_aware_walkforward.py tests/test_regime_aware_walkforward.py
git commit -m "feat: add diversification-floor variant helper"
```

- [ ] **Step 10: Write the smoke test for `run_monthly_variant`**

```python
# append to tests/test_regime_aware_walkforward.py
UNIVERSE2 = [f"T{i}" for i in range(30)]
DATES2 = pd.date_range("2018-01-31", periods=30, freq="ME").strftime("%Y-%m-%d").tolist()


def _toy_multi_parent_panel(dates: list[str], universe: list[str]) -> ScorePanel:
    """Two parents, each with one positive-IC sub ("*_up", ranks ascending with
    ticker order) and one negative-IC sub ("*_down", ranks descending) -- gives each
    parent exactly one eligible candidate, so selection/hysteresis behaviour stays
    deterministic and easy to reason about."""
    scores = {}
    for k, d in enumerate(dates):
        base = np.linspace(10, 90, len(universe)) + k
        scores[d] = pd.DataFrame({
            "p1_up": base, "p1_down": base[::-1],
            "p2_up": base, "p2_down": base[::-1],
        }, index=universe)
    return ScorePanel(rebal_dates=list(dates), scores=scores,
                      parent_keys=["p1", "p2"],
                      sub_by_parent={"p1": ["p1_up", "p1_down"], "p2": ["p2_up", "p2_down"]},
                      universe=list(universe))


def _toy_matrix2(dates: list[str], universe: list[str]) -> pd.DataFrame:
    idx = pd.date_range(dates[0], periods=len(dates) + 8, freq="ME").strftime("%Y-%m-%d")
    data = {t: (1.01 + i * 0.002) ** np.arange(len(idx)) for i, t in enumerate(universe)}
    return pd.DataFrame(data, index=idx)


def test_run_monthly_variant_smoke_variant_c():
    panel = _toy_multi_parent_panel(DATES2, UNIVERSE2)
    matrix = _toy_matrix2(DATES2, UNIVERSE2)
    vix = pd.Series(20.0, index=pd.to_datetime(matrix.index))
    sub_cache = build_monthly_cache(panel, matrix, vix)
    boundary = DATES2[25]

    snapshots = run_monthly_variant(panel, matrix, vix, sub_cache, VARIANTS["C"],
                                    test_boundaries=[boundary])

    assert boundary in snapshots
    cfg = snapshots[boundary]
    assert abs(sum(cfg.parent_weights.values()) - 1.0) < 1e-6
    assert all(w == w for w in cfg.parent_weights.values())     # no NaN
    for subw in cfg.sub_weights.values():
        if subw:
            assert abs(sum(subw.values()) - 1.0) < 1e-6


def test_run_monthly_variant_smoke_variant_d_no_hysteresis():
    panel = _toy_multi_parent_panel(DATES2, UNIVERSE2)
    matrix = _toy_matrix2(DATES2, UNIVERSE2)
    vix = pd.Series(20.0, index=pd.to_datetime(matrix.index))
    sub_cache = build_monthly_cache(panel, matrix, vix)
    boundary = DATES2[25]

    snapshots = run_monthly_variant(panel, matrix, vix, sub_cache, VARIANTS["D"],
                                    test_boundaries=[boundary])

    assert boundary in snapshots
    assert abs(sum(snapshots[boundary].parent_weights.values()) - 1.0) < 1e-6
```

- [ ] **Step 11: Run to verify failure, then implement `run_monthly_variant`**

Run: `pytest tests/test_regime_aware_walkforward.py -v -k run_monthly_variant` → FAIL

```python
# append to research/walkforward/regime_aware_walkforward.py
@dataclass
class _MonthlyState:
    """Carried across the monthly loop for ONE variant."""
    hysteresis: dict[str, HysteresisState] = field(default_factory=dict)
    sub_weights: dict[str, dict[str, float]] = field(default_factory=dict)
    prev_month_parent_w: dict[str, float] = field(default_factory=dict)
    quarter_start_parent_w: dict[str, float] = field(default_factory=dict)


def run_monthly_variant(
    panel: ScorePanel, matrix: pd.DataFrame, vix: pd.Series,
    sub_cache: pd.DataFrame, variant: VariantSpec, *,
    test_boundaries: list[str],
) -> dict[str, FrozenConfig]:
    """Steps monthly from the panel's first rebalance through the last test
    boundary, carrying hysteresis/weight state, and snapshots a FrozenConfig at
    each date in ``test_boundaries``. Returns {boundary_date: FrozenConfig}.
    """
    state = _MonthlyState()
    for p in panel.parent_keys:
        state.hysteresis[p] = HysteresisState()
        state.sub_weights[p] = {}
    snapshots: dict[str, FrozenConfig] = {}

    for cutoff in panel.rebal_dates:
        if cutoff > test_boundaries[-1]:
            break
        px = matrix.loc[matrix.index <= cutoff]
        # Full expanding history -- feeds subfactor evidence/selection (spec Section 1
        # deliberately anchors long_run evidence at panel inception, not a rolling
        # window). The 70% BASE weight below is a separate, narrower window -- see
        # base_train_rebals.
        train_rebals = [d for d in panel.rebal_dates if d <= cutoff]
        # Trailing 5Y ending at cutoff, with the same forward-return-horizon safety
        # cap WalkForwardSplit.train_rebalances() applies -- "on a trailing 5-year
        # training window ending at cutoff. This is exactly variant A's construction"
        # (spec Section 3, "Base weight (the '70%')"). Without the horizon cap, a
        # training rebal too close to cutoff would have its 6M forward return
        # silently truncated by the boundary-clipped price matrix instead of dropped,
        # corrupting the IC estimate used for ic_ir_weights.
        base_train_start = max(
            (pd.Timestamp(cutoff) - pd.DateOffset(years=5)).date().isoformat(), DATA_START)
        base_cap = (pd.Timestamp(cutoff)
                   - pd.DateOffset(months=SELECTION_HORIZON_MONTHS)).date().isoformat()
        base_train_rebals = [d for d in train_rebals if base_train_start <= d <= base_cap]

        evidence = _apply_variant_ic(as_of(sub_cache, cutoff, vix), variant.expected_ic_mode)
        scored = score_table(evidence)
        fwd_by_h = compute_forward_returns(px, train_rebals, {"3M": 3, "6M": 6})
        sub_panel = slice_panel(panel, train_rebals)
        corr = correlation_matrix(sub_panel)

        for parent in panel.parent_keys:
            p_scored = scored[scored.parent == parent]
            if p_scored.empty:
                state.sub_weights[parent] = {}
                continue
            result = select_parent_subfactors(p_scored, corr, sub_panel, fwd_by_h)
            would_select = result["selected"]
            eligible = set(p_scored[p_scored.eligible]["sub_factor"])
            immediate_exit = set(p_scored[
                (p_scored.classification == "EXCLUDED") | (p_scored.coverage < 0.50)
            ]["sub_factor"])

            hstate = state.hysteresis[parent]
            if variant.hysteresis:
                update_streaks(hstate, would_select, eligible, immediate_exit)
                if _is_quarter_boundary(cutoff):
                    apply_quarterly_membership(hstate, would_select)
                active_members = list(hstate.members)
            else:
                active_members = list(would_select)     # D: no freeze, one-shot every month

            metric_map = p_scored.set_index("sub_factor")
            state.sub_weights[parent] = parent_subfactor_weights(
                active_members, metric_map["production_score"])

        # --- Parent-level evidence + weighting (reuses build_monthly_cache/as_of on
        # the constructed parent-composite panel -- see Task 5's reuse note). ---
        parent_panel = build_parent_panel(sub_panel, state.sub_weights)
        parent_cache = build_monthly_cache(parent_panel, px, vix)
        parent_evidence = _apply_variant_ic(as_of(parent_cache, cutoff, vix),
                                            variant.expected_ic_mode)
        utility = parent_utility_table(parent_evidence).set_index("sub_factor")

        base_cfg = select_config(panel, base_train_rebals, matrix, boundary=cutoff)
        adaptive_w = adaptive_weight(utility["utility_score"], utility["expected_ic"])
        target_w = blend_parent_weights(base_cfg.parent_weights, adaptive_w,
                                        dict(utility["expected_ic"]))
        if variant.diversification_floor:
            target_w = _apply_floor(target_w, variant.diversification_floor)

        if variant.weight_caps:
            realized_w = apply_change_caps(
                target_w, state.prev_month_parent_w or target_w,
                state.quarter_start_parent_w or target_w)
        else:
            realized_w = target_w

        state.prev_month_parent_w = realized_w
        if _is_quarter_boundary(cutoff):
            state.quarter_start_parent_w = realized_w

        if cutoff in test_boundaries:
            snapshots[cutoff] = FrozenConfig(
                sub_weights={p: dict(w) for p, w in state.sub_weights.items()},
                parent_weights=dict(realized_w),
                meta={"cutoff": cutoff, "variant": variant.name})
    return snapshots
```

- [ ] **Step 12: Run tests to verify they pass**

Run: `pytest tests/test_regime_aware_walkforward.py -v -k run_monthly_variant`
Expected: PASS (2 tests)

- [ ] **Step 13: Commit**

```bash
git add research/walkforward/regime_aware_walkforward.py tests/test_regime_aware_walkforward.py
git commit -m "feat: add the continuous monthly walk-forward loop"
```

- [ ] **Step 14: Write the smoke test for `run_walkforward_comparison`**

```python
# append to tests/test_regime_aware_walkforward.py
def test_run_walkforward_comparison_smoke():
    panel = _toy_multi_parent_panel(DATES2, UNIVERSE2)
    matrix = _toy_matrix2(DATES2, UNIVERSE2)
    vix = pd.Series(20.0, index=pd.to_datetime(matrix.index))
    sectors = pd.Series("Sector1", index=UNIVERSE2)

    out = run_walkforward_comparison(panel, matrix, vix, sectors,
                                     first_test_year=2019, last_end="2019-12-31",
                                     verbose=False)

    assert not out.empty
    assert set(out["variant"]) == {"A", "B", "C", "D", "C+2%floor", "C+3%floor"}
    assert {"window", "ic_6m", "cagr", "sharpe", "max_drawdown"}.issubset(out.columns)
    assert out["ic_6m"].notna().any()
```

- [ ] **Step 15: Run to verify failure, then implement `run_walkforward_comparison`**

Run: `pytest tests/test_regime_aware_walkforward.py -v -k comparison_smoke` → FAIL

```python
# append to research/walkforward/regime_aware_walkforward.py
METRIC_COLS = ["cagr", "sharpe", "sortino", "max_drawdown", "spy_excess_cagr",
              "spy_ir", "avg_turnover"]


def _score_variant(name: str, sp, cfg: FrozenConfig, panel: ScorePanel,
                   matrix: pd.DataFrame, sectors: pd.Series,
                   fwd: dict[str, dict[str, pd.Series]]) -> dict:
    test_rebals = sp.test_rebalances(panel.rebal_dates)
    scores = frozen_composite(panel, test_rebals, cfg, sectors)
    ic_df = analysis.composite_ic(scores, fwd)

    def _g(h: str, col: str) -> float:
        r = ic_df[ic_df["horizon"] == h]
        return float(r.iloc[0][col]) if not r.empty and col in r.columns else float("nan")

    qres = analysis.quantile_analysis(scores, fwd)
    q6 = qres.get("6M", {})
    spr = q6.get("spread") or {}
    book = pf.simulate(scores, matrix, sectors, top_pct=0.20, mode="equal", hold_months=1)
    row = {
        "window": sp.label.split(":", 1)[-1], "variant": name,
        "ic_3m": _g("3M", "mean_ic"), "ic_6m": _g("6M", "mean_ic"),
        "ic_ir_6m": _g("6M", "information_ratio"), "hit_rate_6m": _g("6M", "hit_rate"),
        "q5q1_ann": float(spr.get("annualized", float("nan"))),
        "monotonic_rate_6m": float(q6.get("monotonic_rate", float("nan"))),
    }
    row.update({k: book.metrics.get(k, float("nan")) for k in METRIC_COLS})
    return row


def run_walkforward_comparison(
    panel: ScorePanel, matrix: pd.DataFrame, vix: pd.Series, sectors: pd.Series, *,
    first_test_year: int = 2017, last_end: str | None = None, verbose: bool = True,
) -> pd.DataFrame:
    """Runs all 6 named variants (spec Section 4) over the same 19-window semiannual
    rolling-5Y grid every other regime study in this repo uses, and returns one row
    per (window, variant) with the full metric suite.
    """
    kwargs: dict = {"first_test_year": first_test_year}
    if last_end is not None:
        kwargs["last_end"] = last_end
    splits = semiannual_policy_splits("rolling5y", **kwargs)
    if not splits:
        return pd.DataFrame()
    test_boundaries = [sp.test_start for sp in splits]

    sub_cache = build_monthly_cache(panel, matrix, vix)

    monthly_snapshots: dict[str, dict[str, FrozenConfig]] = {}
    for name, vspec in VARIANTS.items():
        if verbose:
            print(f"  running variant {name} (monthly loop)...")
        monthly_snapshots[name] = run_monthly_variant(
            panel, matrix, vix, sub_cache, vspec, test_boundaries=test_boundaries)

    rows: list[dict] = []
    for sp in splits:
        test_rebals = sp.test_rebalances(panel.rebal_dates)
        if not test_rebals:
            continue
        fwd = compute_forward_returns(matrix, test_rebals, {"3M": 3, "6M": 6})

        cfg_a = select_config(panel, sp.train_rebalances(panel.rebal_dates), matrix,
                              boundary=sp.test_start)
        rows.append(_score_variant("A", sp, cfg_a, panel, matrix, sectors, fwd))

        for name in VARIANTS:
            cfg = monthly_snapshots[name].get(sp.test_start)
            if cfg is None:
                continue
            rows.append(_score_variant(name, sp, cfg, panel, matrix, sectors, fwd))
    return pd.DataFrame(rows)
```

- [ ] **Step 16: Run tests to verify all pass**

Run: `pytest tests/test_regime_aware_walkforward.py -v`
Expected: PASS (9 tests)

- [ ] **Step 17: Commit**

```bash
git add research/walkforward/regime_aware_walkforward.py tests/test_regime_aware_walkforward.py
git commit -m "feat: add A/B/C/D walk-forward comparison runner"
```

---

### Task 7: Entry Point & Report (`run_regime_aware_study.py`)

**Files:**
- Create: `run_regime_aware_study.py`
- Test: `tests/test_run_regime_aware_study.py`

A thin CLI script (mirrors `run_vix_overlay_study.py`/`run_factor_persistence.py`):
loads the cached panel/matrix/VIX/sectors, writes the "as of latest cutoff"
evidence/selection/parent tables (deliverables 1-5), runs the full A-D(+floor)
walk-forward comparison (Task 6), and writes `REGIME_AWARE_REPORT.md`
(deliverables 6-8). Report-writing logic lives in the script itself, matching
the precedent in `run_vix_overlay_study.py` (which also builds its own report/
plots rather than delegating to the module).

Only `_recommendation` (pure logic, no I/O) gets a unit test, following the
same "import a helper straight out of a run_*.py script into a test" pattern
`tests/test_walkforward.py` already uses (`from run_walkforward import _adapt`).
The rest of the script is an I/O-heavy CLI wrapper, consistent with how this
repo already treats every other `run_*.py` entry point (untested directly;
Task 8 adds one DB-guarded integration check of the underlying pipeline).

- [ ] **Step 1: Write the failing test for `_recommendation`**

```python
# tests/test_run_regime_aware_study.py
from __future__ import annotations

import pandas as pd
import pytest

from run_regime_aware_study import REPORT_METRIC_COLS, _recommendation


def _toy_comparison(a_sharpe, c_sharpe, a_dd, c_dd, c_turnover, d_turnover) -> pd.DataFrame:
    rows = []
    for variant, sharpe, dd, turnover in (
        ("A", a_sharpe, a_dd, 0.20), ("C", c_sharpe, c_dd, c_turnover),
        ("D", 0.0, 0.0, d_turnover),
    ):
        rows.append({
            "variant": variant, "window": "2020-H1",
            **{c: 0.01 for c in REPORT_METRIC_COLS},
            "sharpe": sharpe, "max_drawdown": dd, "avg_turnover": turnover,
        })
    return pd.DataFrame(rows)


def test_recommendation_supports_adoption_when_c_beats_a_and_d():
    comparison = _toy_comparison(a_sharpe=1.0, c_sharpe=1.5, a_dd=-0.10, c_dd=-0.08,
                                 c_turnover=0.20, d_turnover=0.60)
    lines = _recommendation(comparison)
    assert any("supports moving to a production pilot" in l for l in lines)


def test_recommendation_does_not_support_adoption_when_c_underperforms():
    comparison = _toy_comparison(a_sharpe=1.5, c_sharpe=1.0, a_dd=-0.08, c_dd=-0.15,
                                 c_turnover=0.55, d_turnover=0.60)
    lines = _recommendation(comparison)
    assert any("not yet earning its keep" in l for l in lines)


def test_recommendation_handles_empty_comparison():
    lines = _recommendation(pd.DataFrame())
    assert lines and "No walk-forward windows" in lines[0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_run_regime_aware_study.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'run_regime_aware_study'`

- [ ] **Step 3: Implement `run_regime_aware_study.py`**

```python
# run_regime_aware_study.py
"""Hierarchical, regime-aware factor model study runner.

Runs all 6 named variants (A baseline, B stable-no-VIX, C proposed model,
D dynamic-monthly ablation, C+2%floor, C+3%floor) in the same strict PIT
semiannual walk-forward (rolling-5y, 2017-2026) every other regime study in
this repo uses, and writes the full comparison + evidence tables.

Outputs (output/regime_aware/):
    monthly_subfactor_cache.csv    -- full per-(sub_factor, month) IC/spread cache
    subfactor_evidence_latest.csv  -- long-run/recent/regime evidence + scores, latest cutoff
    selection_decisions.csv        -- selected/rejected subs + reasons, latest cutoff
    parent_formulas.csv            -- intra-parent weights, latest cutoff
    parent_evidence.csv            -- parent-level evidence/utility, latest cutoff
    vix_probabilities_latest.csv   -- current smooth VIX regime probabilities
    walkforward_comparison.csv     -- one row per (window, variant), full metric suite
    REGIME_AWARE_REPORT.md

Usage:
    python run_regime_aware_study.py
    python run_regime_aware_study.py --rebuild-panel
    python run_regime_aware_study.py --first-test-year 2017 --last-end 2026-06-30
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import pandas as pd

from backtesting import data_loader as dl
from data.db import get_db
from research import compute_forward_returns
from research.subset_selection import correlation_matrix
from research.walkforward.regime_aware_evidence import as_of, build_monthly_cache
from research.walkforward.regime_aware_parents import parent_utility_table
from research.walkforward.regime_aware_scoring import score_table
from research.walkforward.regime_aware_selection import select_parent_subfactors
from research.walkforward.regime_aware_walkforward import run_walkforward_comparison
from research.walkforward.regime_probability import REGIME_ORDER, regime_probabilities
from research.walkforward.compose import build_parent_panel
from research.walkforward.splits import PANEL_START
from research.walkforward.vix_regime_study import _vix_spot, load_vix_series

from run_walkforward import PRICE_END, _load_panel

OUT_DIR = Path("output/regime_aware")
REPORT_METRIC_COLS = ["ic_6m", "ic_ir_6m", "q5q1_ann", "hit_rate_6m", "cagr", "sharpe",
                      "sortino", "max_drawdown", "spy_excess_cagr", "spy_ir", "avg_turnover"]


def _write_latest_evidence(panel, matrix, vix, sub_cache, out_dir: Path) -> None:
    """Deliverables 1-5: evidence table, selection decisions, formulas, parent
    evidence/utility, and current VIX probabilities, all as of the latest cutoff."""
    cutoff = panel.rebal_dates[-1]
    evidence = as_of(sub_cache, cutoff, vix)
    scored = score_table(evidence)
    scored.to_csv(out_dir / "subfactor_evidence_latest.csv", index=False)

    fwd_by_h = compute_forward_returns(matrix, panel.rebal_dates, {"3M": 3, "6M": 6})
    corr = correlation_matrix(panel)
    decision_rows: list[dict] = []
    formula_rows: list[dict] = []
    parent_sub_weights: dict[str, dict[str, float]] = {}
    for parent in panel.parent_keys:
        p_scored = scored[scored.parent == parent]
        if p_scored.empty:
            continue
        result = select_parent_subfactors(p_scored, corr, panel, fwd_by_h)
        for dec in result["decisions"]:
            decision_rows.append({"parent": parent, **dec})
        for sub, w in result["weights"].items():
            formula_rows.append({"parent": parent, "sub_factor": sub, "weight": w})
        parent_sub_weights[parent] = result["weights"]
    pd.DataFrame(decision_rows).to_csv(out_dir / "selection_decisions.csv", index=False)
    pd.DataFrame(formula_rows).to_csv(out_dir / "parent_formulas.csv", index=False)

    parent_panel = build_parent_panel(panel, parent_sub_weights)
    parent_cache = build_monthly_cache(parent_panel, matrix, vix)
    parent_evidence = as_of(parent_cache, cutoff, vix)
    parent_utility_table(parent_evidence).to_csv(out_dir / "parent_evidence.csv", index=False)

    vix_now = _vix_spot(vix, cutoff)
    probs = regime_probabilities(vix_now)
    pd.DataFrame([{"regime": r, "probability": probs[r]} for r in REGIME_ORDER]
                ).to_csv(out_dir / "vix_probabilities_latest.csv", index=False)


def _fmt(v, fmt: str = ".3f") -> str:
    if v is None or (isinstance(v, float) and v != v):
        return "—"
    return f"{v:{fmt}}"


def _recommendation(comparison: pd.DataFrame) -> list[str]:
    """Deliverable 8: a data-driven verdict derived from the actual walk-forward
    numbers -- never a placeholder, since the comparison table is always available
    by the time this runs."""
    if comparison.empty:
        return ["No walk-forward windows were evaluated -- check the study inputs."]
    summary = comparison.groupby("variant")[REPORT_METRIC_COLS].mean()
    if "A" not in summary.index or "C" not in summary.index:
        return ["Variant A or C did not produce results -- check the study inputs."]
    a, c = summary.loc["A"], summary.loc["C"]
    d_turnover = float(summary.loc["D", "avg_turnover"]) if "D" in summary.index else float("nan")
    c_turnover = float(c["avg_turnover"])
    sharpe_delta = float(c["sharpe"] - a["sharpe"])
    ic_delta = float(c["ic_6m"] - a["ic_6m"])
    dd_delta = float(c["max_drawdown"] - a["max_drawdown"])   # less negative = improved
    lines = [
        f"- Sharpe: C {_fmt(c['sharpe'], '.2f')} vs A {_fmt(a['sharpe'], '.2f')} "
        f"(delta {sharpe_delta:+.2f})",
        f"- IC (6M): C {_fmt(c['ic_6m'])} vs A {_fmt(a['ic_6m'])} (delta {ic_delta:+.3f})",
        f"- Max drawdown: C {_fmt(c['max_drawdown'], '.1%')} vs A "
        f"{_fmt(a['max_drawdown'], '.1%')} (delta {dd_delta:+.1%})",
        f"- Turnover: C {c_turnover:.2f} vs D (dynamic-monthly benchmark) {d_turnover:.2f} "
        "-- C should churn materially less than D if the hysteresis/shrinkage machinery "
        "is earning its complexity.",
    ]
    if sharpe_delta > 0 and dd_delta >= 0 and c_turnover < d_turnover:
        lines.append("\n**Verdict:** C improves risk-adjusted return and drawdown over A "
                     "while churning less than the D overfitting benchmark -- supports "
                     "moving to a production pilot.")
    else:
        lines.append("\n**Verdict:** C does not clearly beat A and/or does not clearly beat "
                     "D's churn -- the added complexity is not yet earning its keep; treat "
                     "as research, not a production candidate.")
    return lines


def write_report(comparison: pd.DataFrame, out_dir: Path) -> None:
    lines = [
        "# Hierarchical, Regime-Aware Factor Model — Walk-Forward Comparison",
        "",
        "Rolling-5y semiannual walk-forward, same 19-window grid as every other "
        "regime study in this repo. Variant A is the existing production "
        "construction, called unmodified; B/C/D run the new continuous monthly "
        "evidence + hysteresis + capped-weight pipeline (spec Section 4).",
        "",
        "## Full-Period Summary",
        "",
        "| Variant | n_windows | " + " | ".join(REPORT_METRIC_COLS) + " |",
        "|---|---|" + "---|" * len(REPORT_METRIC_COLS),
    ]
    if not comparison.empty:
        for variant, g in comparison.groupby("variant"):
            vals = [_fmt(g[c].mean()) for c in REPORT_METRIC_COLS]
            lines.append(f"| {variant} | {len(g)} | " + " | ".join(vals) + " |")
    lines += [
        "",
        "## Per-Window Detail",
        "",
        "See `walkforward_comparison.csv` for the full per-(window, variant) table.",
        "",
        "## Recommendation",
        "",
    ]
    lines += _recommendation(comparison)
    (out_dir / "REGIME_AWARE_REPORT.md").write_text("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rebuild-panel", action="store_true")
    parser.add_argument("--first-test-year", type=int, default=2017)
    parser.add_argument("--last-end", type=str, default=None)
    parser.add_argument("--out", type=str, default=str(OUT_DIR))
    args = parser.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== Hierarchical Regime-Aware Factor Model Study ===\n")
    print("[1/4] Loading panel + price matrix + VIX + sectors...")
    panel = _load_panel(args.rebuild_panel)
    with get_db() as db:
        matrix = dl.load_price_matrix(db, panel.universe, PANEL_START, PRICE_END)
        vix = load_vix_series(db)
        sectors = dl.global_sectors(db)
    print(f"      {len(panel.rebal_dates)} rebalances, {len(panel.universe)} names, "
          f"sectors for {sectors.notna().sum()} names.")

    print("\n[2/4] Building the monthly PIT evidence cache...")
    sub_cache = build_monthly_cache(panel, matrix, vix)
    sub_cache.to_csv(out_dir / "monthly_subfactor_cache.csv", index=False)

    print("\n[3/4] Writing latest-cutoff evidence, selection, and parent tables...")
    _write_latest_evidence(panel, matrix, vix, sub_cache, out_dir)

    print("\n[4/4] Running the A/B/C/D(+floor) walk-forward comparison...")
    comparison = run_walkforward_comparison(
        panel, matrix, vix, sectors,
        first_test_year=args.first_test_year, last_end=args.last_end)
    comparison.to_csv(out_dir / "walkforward_comparison.csv", index=False)

    write_report(comparison, out_dir)
    print(f"\nDone. Key files:\n  {out_dir / 'REGIME_AWARE_REPORT.md'}\n"
          f"  {out_dir / 'walkforward_comparison.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_run_regime_aware_study.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add run_regime_aware_study.py tests/test_run_regime_aware_study.py
git commit -m "feat: add regime-aware study entry point and report"
```

---

### Task 8: DB-Guarded Integration Smoke Test

**Files:**
- Modify: `tests/test_regime_aware_walkforward.py` (append)

Everything through Task 7 has been proven correct against small synthetic
panels. This task is the one check against the real, full-history cached
candidate panel -- mirroring the existing DB-guarded pattern in
`tests/test_walkforward.py::test_full_sample_selection_reproduces_live_v4`
(same `_CACHE` path, same `@pytest.mark.skipif` guard, same
`from run_walkforward import _adapt` import). Nothing new is implemented
here; this only proves Tasks 1-7 compose correctly on real data.

- [ ] **Step 1: Write the DB-guarded integration test**

```python
# append to tests/test_regime_aware_walkforward.py
from pathlib import Path

_CACHE = Path("cache/subfactor_expansion/cand_panel_2015-06-30_2026-06-30_monthly_v2.pkl")


@pytest.mark.skipif(not _CACHE.exists(), reason="candidate panel cache not built")
def test_run_monthly_variant_on_real_panel_produces_valid_weights():
    """DB-guarded integration check: the continuous monthly loop (variant C) runs
    end to end on the real cached candidate panel without crashing, and produces a
    well-formed FrozenConfig at the latest available cutoff.

    This intentionally steps through the ENTIRE ~130-month history once (the
    monthly loop is stateful and can't skip ahead), calling select_config every
    month for the rolling-5Y base weight -- so this test is genuinely slow
    (several minutes). That per-month select_config cost is a known follow-up
    optimisation (caching/memoizing across nearby months), flagged in the design
    spec's "Out of scope" section, not something this plan fixes.
    """
    from backtesting import data_loader as dl
    from data.db import get_db
    from research.subfactor_expansion.panel import load_cached_panel as eload
    from research.walkforward.regime_aware_walkforward import VARIANTS, run_monthly_variant
    from research.walkforward.vix_regime_study import load_vix_series
    from run_walkforward import _adapt

    panel = _adapt(eload(_CACHE))
    with get_db() as db:
        matrix = dl.load_price_matrix(db, panel.universe, "2015-06-30", "2026-07-06")
        vix = load_vix_series(db)

    sub_cache = build_monthly_cache(panel, matrix, vix)
    cutoff = panel.rebal_dates[-1]

    snapshots = run_monthly_variant(panel, matrix, vix, sub_cache, VARIANTS["C"],
                                    test_boundaries=[cutoff])

    assert cutoff in snapshots
    cfg = snapshots[cutoff]
    assert cfg.parent_weights
    assert abs(sum(cfg.parent_weights.values()) - 1.0) < 1e-6
    assert max(cfg.parent_weights.values()) <= 0.25 + 1e-6
    assert all(w == w for w in cfg.parent_weights.values())     # no NaN
    for parent, subw in cfg.sub_weights.items():
        if subw:
            assert abs(sum(subw.values()) - 1.0) < 1e-6
            assert max(subw.values()) <= 0.50 + 1e-6
```

- [ ] **Step 2: Run the test**

Run: `pytest tests/test_regime_aware_walkforward.py -v -k real_panel --timeout=1200`
Expected: PASS (or SKIPPED if `cache/subfactor_expansion/cand_panel_2015-06-30_2026-06-30_monthly_v2.pkl` hasn't been built yet -- run `python run_factor_persistence.py` first if so, which builds/caches that panel as a side effect).

- [ ] **Step 3: Commit**

```bash
git add tests/test_regime_aware_walkforward.py
git commit -m "test: add DB-guarded real-panel integration check"
```

---
