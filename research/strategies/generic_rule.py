"""Generic parametrized exit rule for the loop-engineering search (user
request 2026-09-02). One StrategyConfig instance is one point in the search
space that scripts/loop_round.py explores; it's a superset of strategy #13
(pure trailing stop) plus an optional hard stop-loss (measured from entry,
not the peak), take-profit, "sell if composite score craters" exit, and a
minimum holding period that suppresses every soft exit except the hard stop.

Entry-side levers (book size, entry pool width, sector cap) aren't part of
the exit rule itself -- they're passed straight through to
research.strategies.engine.simulate_managed_book as separate kwargs; see
research/loop_engineering/harness.py.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

from .engine import _months_between, _rank_pct


@dataclass
class StrategyConfig:
    k: int = 10
    trail_pct: float | None = 0.10
    hard_stop_pct: float | None = None
    take_profit_pct: float | None = None
    cap_months: float = 12
    score_floor: float | None = None
    rank_floor_pct: float | None = None   # sell if rank_pct (0=best,1=worst) exceeds this at review
    min_hold_months: float = 0.0
    entry_rank_k: int | None = None
    max_per_sector: int | None = None
    min_value_pct: float | None = None   # entry-side quality/cheapness floor (Value-parent percentile)

    def to_dict(self) -> dict:
        return asdict(self)


def build_rule(cfg: StrategyConfig):
    def rule(pos, today, price, is_review, comp_today) -> bool:
        # hard stop always fires, even inside the minimum-hold window
        if cfg.hard_stop_pct is not None and price / pos.entry_price - 1.0 <= -abs(cfg.hard_stop_pct):
            return True
        held = _months_between(pos.entry_date, today)
        if held < cfg.min_hold_months:
            return False
        if cfg.trail_pct is not None and price / pos.peak_price - 1.0 <= -abs(cfg.trail_pct):
            return True
        if cfg.take_profit_pct is not None and price / pos.entry_price - 1.0 >= cfg.take_profit_pct:
            return True
        if cfg.score_floor is not None and is_review and comp_today is not None:
            sc = comp_today.get(pos.ticker)
            if sc is not None and sc == sc and sc < cfg.score_floor:
                return True
        if cfg.rank_floor_pct is not None and is_review and comp_today is not None:
            rp = _rank_pct(comp_today, pos.ticker)
            if rp is not None and rp > cfg.rank_floor_pct:
                return True
        if held >= cfg.cap_months:
            return True
        return False
    return rule
