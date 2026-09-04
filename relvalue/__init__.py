"""Relative-value / pairs research program (2026-07-18).

Generalizes pairtrading/study.py (GGR distance pairs) to:
* new spread definitions: regression residual, log-spread, ratio-MA z-score;
* new pair types: cointegration-selected stock pairs, stock vs sector ETF,
  ETF vs ETF, dual share classes;
* direction modes: dollar-neutral, hedge-ratio, vol-neutral, long-only;
* borrow costs and invested-capital accounting.

Protocol, bars and time partitions: output/relvalue/PREREGISTRATION.md.
"""
