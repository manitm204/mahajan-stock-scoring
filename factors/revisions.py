"""Estimate Revisions factor.

Captures improving analyst expectations from genuinely point-in-time FMP data
(see :mod:`data.grades`): the change in a net analyst-rating score over 30/60/90
days, consensus price-target momentum, and trailing price-target-event breadth.
Numeric consensus-estimate revisions (forward EPS / price target, from
:mod:`data.estimates`) are added once they have accrued enough live coverage —
they are forward-accruing, so gating on coverage keeps sparse early snapshots
from diluting the parent toward neutral. Where coverage is unavailable the engine
assigns a neutral 50 (per the missing-data rule), surfaced as a degenerate factor
by the crowding module rather than silently distorting the composite.
"""
from __future__ import annotations

from .base import Factor, SubFactor
from .utils import DataContext, col

# A forward-accruing numeric sub-factor is only admitted once this share of the
# universe has a non-null value, so early (mostly-empty) snapshots don't compress
# the parent by contributing a constant neutral 50 to every name.
_MIN_COVERAGE = 0.30

# Numeric estimate-revision candidates (column, higher_is_better).
_ESTIMATE_SUBS = [
    ("forward_eps_revision_30d", True),
    ("forward_eps_revision_90d", True),
    ("price_target_revision_30d", True),
    ("estimate_breadth_change", True),
]


class RevisionsFactor(Factor):
    name = "revisions"

    def compute(self, ctx: DataContext) -> list[SubFactor]:
        rev = ctx.revisions()
        subs = [
            SubFactor("rev_rating_change_30d", col(rev, "rating_change_30d"), higher_is_better=True),
            SubFactor("rev_rating_change_90d", col(rev, "rating_change_90d"), higher_is_better=True),
            SubFactor("rev_pt_momentum", col(rev, "pt_momentum"), higher_is_better=True),
            SubFactor("rev_pt_target_upside_30d", col(rev, "pt_target_upside_30d"), higher_is_better=True),
            SubFactor("rev_pt_upgrade_ratio_30d", col(rev, "pt_upgrade_ratio_30d"), higher_is_better=True),
        ]
        est = ctx.estimate_features()
        for name, higher in _ESTIMATE_SUBS:
            series = col(est, name)
            if float(series.notna().mean()) >= _MIN_COVERAGE:
                subs.append(SubFactor(f"rev_{name}", series, higher_is_better=higher))
        return subs
