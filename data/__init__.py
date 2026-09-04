"""Mahajan Hedge Fund - Layer 1 (Data Layer).

This package owns data ingestion, storage, normalization, and feature
generation. It builds a permanent historical research warehouse that all
later layers (factors, analysis, portfolio, risk, execution, reporting)
read from. It never performs scoring, ranking, or trading.
"""

__version__ = "1.0.0"
__layer__ = 1
