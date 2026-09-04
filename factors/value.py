"""Value factor.

Measures cheapness through several independent lenses so the score never hinges
on a single ratio. Market cap and enterprise value are reconstructed in the
DataContext (Layer 1 lacks populated share counts); flow metrics come from the
latest annual statement.

All ratios are framed as yields (higher = cheaper = better) — including
EV/EBITDA, ranked as EBITDA/EV so a negative-EBITDA name sorts to the bottom
instead of sign-flipping to the top of the inverted multiple.
"""
from __future__ import annotations

import numpy as np

from .base import Factor, SubFactor
from .utils import DataContext, col


class ValueFactor(Factor):
    name = "value"

    def compute(self, ctx: DataContext) -> list[SubFactor]:
        f = ctx.fund_annual_latest()
        mcap = ctx.market_cap()
        ev = ctx.enterprise_value()

        equity = col(f, "shareholder_equity")
        book_to_price = equity / mcap.replace(0, np.nan)

        fcf = col(f, "free_cash_flow")
        fcf_yield = fcf / mcap.replace(0, np.nan)

        # Yield form (EBITDA / EV), not the inverted multiple: EV / EBITDA
        # sign-flips when EBITDA is negative and would rank a money-loser as
        # the cheapest name in its sector. Non-positive EV (a reconstruction
        # artifact for cash-heavy names and banks) is masked to neutral.
        ev_pos = ev.where(ev > 0)
        ebitda = col(f, "ebitda")
        ebitda_yield = ebitda / ev_pos

        revenue = col(f, "revenue")
        sales_to_ev = revenue / ev_pos

        # Shareholder yield: buybacks + dividends returned, as a fraction of cap.
        # Both are stored as negative cash outflows, so negate. Dividends are
        # sparsely populated upstream; treat a missing dividend as zero rather
        # than discarding the (well-populated) buyback signal.
        buybacks = col(f, "buybacks").fillna(0.0)
        dividends = col(f, "dividends_paid").fillna(0.0)
        shareholder_yield = -(buybacks + dividends) / mcap.replace(0, np.nan)
        # If both components were missing the row is genuinely unknown -> NaN.
        both_missing = col(f, "buybacks").isna() & col(f, "dividends_paid").isna()
        shareholder_yield = shareholder_yield.mask(both_missing.reindex(shareholder_yield.index, fill_value=True))

        return [
            SubFactor("val_book_to_price", book_to_price, higher_is_better=True),
            SubFactor("val_fcf_yield", fcf_yield, higher_is_better=True),
            SubFactor("val_ev_ebitda_inv", ebitda_yield, higher_is_better=True),
            SubFactor("val_shareholder_yield", shareholder_yield, higher_is_better=True),
            SubFactor("val_sales_to_ev", sales_to_ev, higher_is_better=True),
        ]
