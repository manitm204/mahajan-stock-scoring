"""Strategy definitions and the entry/exit selection rule.

Each strategy is a small declarative :class:`StrategySpec`; the same
:func:`select_positions` rule drives all of them. The only behavioural knobs are
the *enter* and *exit* rank thresholds per side, which is what separates a strict
top-5 rebalance (enter 5 / exit 5) from a hysteresis rule (enter 5 / exit 10)
that keeps a name while it drifts inside the top-10 to cut churn.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class StrategySpec:
    name: str
    label: str
    description: str
    long_enabled: bool
    short_enabled: bool
    long_enter: int = 5      # enter a long when its rank is <= this
    long_exit: int = 5       # drop a held long once its rank is > this
    short_enter: int = 5     # enter a short when its rank-from-bottom is <= this
    short_exit: int = 5      # cover a held short once rank-from-bottom > this

    @property
    def primary_series(self) -> str:
        return "long_basket" if not self.short_enabled else "long_short"


STRATEGIES: dict[str, StrategySpec] = {
    "top5_strict": StrategySpec(
        name="top5_strict",
        label="Strategy A — Top 5 Strict Rebalance",
        description="Long top 5, short bottom 5, sell the moment a name leaves its band.",
        long_enabled=True, short_enabled=True,
        long_enter=5, long_exit=5, short_enter=5, short_exit=5,
    ),
    "top5_entry_top10_exit": StrategySpec(
        name="top5_entry_top10_exit",
        label="Strategy B — Top 5 Entry / Top 10 Exit",
        description="Enter on top/bottom 5, hold while inside top/bottom 10 (less churn).",
        long_enabled=True, short_enabled=True,
        long_enter=5, long_exit=10, short_enter=5, short_exit=10,
    ),
    "top10_entry_top20_exit": StrategySpec(
        name="top10_entry_top20_exit",
        label="Strategy E — Top 10 Entry / Top 20 Exit",
        description=(
            "Broader sleeve: enter on top/bottom 10, hold while inside top/bottom 20. "
            "Composite is sector-relative under the hood (sub-factor percentiles within "
            "GICS sector); entry/exit is on the resulting global composite rank."
        ),
        long_enabled=True, short_enabled=True,
        long_enter=10, long_exit=20, short_enter=10, short_exit=20,
    ),
    "long_only_top5": StrategySpec(
        name="long_only_top5",
        label="Strategy C — Long-Only Top 5",
        description="Long top 5 only, equal weight, measured against SPY.",
        long_enabled=True, short_enabled=False,
        long_enter=5, long_exit=5,
    ),
    "market_neutral": StrategySpec(
        name="market_neutral",
        label="Strategy D — Long/Short Market Neutral",
        description="Long top 5, short bottom 5, 50% long / 50% short, equal weight per side.",
        long_enabled=True, short_enabled=True,
        long_enter=5, long_exit=5, short_enter=5, short_exit=5,
    ),
}


def select_positions(
    spec: StrategySpec,
    ranked: pd.DataFrame,
    prev_long: set[str],
    prev_short: set[str],
) -> tuple[list[str], list[str]]:
    """Resolve the long and short holdings for one rebalance.

    A name is held if it newly meets the *enter* threshold, or it was already held
    and still meets the looser *exit* threshold. With enter == exit this collapses
    to "exactly the current top/bottom N". A name is never held on both sides;
    ranks are unique so the bands cannot overlap.
    """
    rank = ranked["rank"]
    rank_bottom = ranked["rank_from_bottom"]

    long_names: list[str] = []
    if spec.long_enabled:
        entered = set(rank.index[rank <= spec.long_enter])
        kept = {t for t in prev_long if t in rank.index and rank[t] <= spec.long_exit}
        long_names = _ordered(entered | kept, rank)

    short_names: list[str] = []
    if spec.short_enabled:
        entered = set(rank_bottom.index[rank_bottom <= spec.short_enter])
        kept = {t for t in prev_short if t in rank_bottom.index and rank_bottom[t] <= spec.short_exit}
        short_names = _ordered(entered | kept, rank_bottom)

    return long_names, short_names


def _ordered(names: set[str], rank: pd.Series) -> list[str]:
    """Names sorted by rank (strongest first) for stable, readable output."""
    return [t for t in rank.sort_values().index if t in names]


def equal_weights(names: list[str]) -> dict[str, float]:
    """Equal weights summing to 1.0 over ``names`` (empty -> empty)."""
    if not names:
        return {}
    w = 1.0 / len(names)
    return {t: w for t in names}


def score_weights(names: list[str], ranked: pd.DataFrame,
                  side: str) -> dict[str, float]:
    """Composite-score-proportional weights summing to 1.0 for one side.

    Uses ``composite_raw`` (the continuous, unsaturated score) rather than the
    sector-relative percentile ``composite_score`` — within the long top-10 the
    latter is essentially always 100, so percentile-proportional weighting
    would collapse to equal weight. ``composite_raw`` retains the real
    cross-sectional dispersion the IC weighter actually trained on.

    Within the held book the strength signal is the *gap* between names, so we
    subtract the within-book min before sizing: longs are sized proportionally
    to ``(raw - min(raw))`` so the weakest held long has the smallest non-zero
    weight; shorts are sized proportionally to ``(max(raw) - raw)`` (mirror).
    A small floor (10% of the dispersion range) is added so the weakest name
    is not literally zero, which would collapse the book to N-1 effective
    positions and eat too much of the breadth we explicitly bought with the
    Top10/Top20 sleeve.

    Falls back to equal weights when ``composite_raw`` is absent or all values
    are equal (no dispersion to exploit).
    """
    if not names:
        return {}
    col = "composite_raw" if "composite_raw" in ranked.columns else "composite_score"
    s = ranked[col].reindex(names).astype(float)
    if s.isna().all():
        return equal_weights(names)
    s = s.fillna(s.mean())

    if side == "long":
        raw = s - s.min()
    elif side == "short":
        raw = s.max() - s
    else:
        raise ValueError(f"side must be 'long' or 'short', got {side!r}")

    rng = float(raw.max())
    if rng <= 0:
        return equal_weights(names)
    raw = raw + 0.10 * rng  # floor so the weakest held name still gets ~10% slack
    total = float(raw.sum())
    if total <= 0:
        return equal_weights(names)
    return {t: float(raw[t] / total) for t in names}
