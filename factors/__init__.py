"""Layer 2 — Factor Scoring Engine.

A reusable, config-driven factor framework over the Layer 1 SQLite warehouse.
Factors are sector-neutral (GICS-relative percentile ranks), robust to missing
data (missing -> neutral 50), and fully reproducible from stored data with no
external API calls.

Adding or removing a factor is a one-line change to :data:`ALL_FACTORS` below
plus its small module — the scoring engine, composite, and persistence handle
everything else generically.
"""
from __future__ import annotations

from .base import Factor, FactorResult, SubFactor, score_factor
from .growth import GrowthFactor
from .insider import InsiderFactor
from .institutional import InstitutionalFactor
from .momentum import MomentumFactor
from .parent_selection_v4 import ALL_FACTORS_V4, SELECTED_SUBS, V4_PARENT_WEIGHTS
from .quality import QualityFactor
from .revisions import RevisionsFactor
from .short_interest import ShortInterestFactor
from .value import ValueFactor

# Ordered factor registry. Each factor's `.key` must have a matching weight in
# config.yaml -> factors.default_weights (and every regime weight table).
ALL_FACTORS: list[Factor] = [
    MomentumFactor(),
    ValueFactor(),
    QualityFactor(),
    GrowthFactor(),
    RevisionsFactor(),
    ShortInterestFactor(),
    InsiderFactor(),
    InstitutionalFactor(),
]

__all__ = [
    "Factor", "FactorResult", "SubFactor", "score_factor",
    "ALL_FACTORS", "ALL_FACTORS_V4",
    "SELECTED_SUBS", "V4_PARENT_WEIGHTS",
    "MomentumFactor", "ValueFactor", "QualityFactor", "GrowthFactor",
    "RevisionsFactor", "ShortInterestFactor", "InsiderFactor", "InstitutionalFactor",
]
