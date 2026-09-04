"""Institutional Flow factor (13F sponsorship).

Treats 13F data as slow-moving *confirmation* rather than a timing signal — hence
its small composite weight. Only a handful of managers are tracked upstream, so
most names have no signal and stay neutral (50); that sparsity is expected and is
reported as a degenerate factor by the crowding module.
"""
from __future__ import annotations

from .base import Factor, SubFactor
from .utils import DataContext, col


class InstitutionalFactor(Factor):
    name = "institutional"

    def compute(self, ctx: DataContext) -> list[SubFactor]:
        inst = ctx.institutional()
        fund_count = col(inst, "fund_count")
        net_share_change = col(inst, "net_share_change")
        new_positions = col(inst, "new_positions")
        increases = col(inst, "position_increases")

        # Binary flag that preserves "no data" (NaN) so absent names stay neutral
        # instead of being scored as a hard zero.
        multi_fund_open = new_positions.where(new_positions.isna(), (new_positions >= 2).astype(float))

        return [
            SubFactor("inst_fund_count", fund_count, higher_is_better=True),
            SubFactor("inst_net_share_change", net_share_change, higher_is_better=True),
            SubFactor("inst_new_positions", new_positions, higher_is_better=True),
            SubFactor("inst_multi_fund_open", multi_fund_open, higher_is_better=True),
            SubFactor("inst_high_conviction", increases, higher_is_better=True),
        ]
