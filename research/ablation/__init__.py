"""Portfolio-construction ablation study (2026-07-17).

Holds the scoring model fixed (clean rolling-5y walk-forward composite, delistings
realized) and ablates only *construction*: selection breadth, bad-factor exclusion,
weighting scheme, sector constraint, VIX tilt, rebalance frequency and hysteresis —
net of transaction costs, against SPY.

Design: the expensive walk-forward scoring runs ONCE (``data.load_ablation_data``);
every config is then a cheap re-weighting of the same score panel (``engine``).
Stage A varies one family at a time around a neutral reference; Stage B crosses the
winning levels of the interacting families (breadth x weighting x sector, VIX on/off);
Stage C confirms the finalist with frequency/hysteresis/cost sensitivity.

Pre-registered decision rule (written before the first run): a winner must show
positive net alpha vs SPY over the full period AND non-negative excess return in
both eras (split 2021-12-31). Alpha t-stats are reported so a lucky cell in a
~60-config search is not mistaken for edge.
"""
from .data import AblationData, load_ablation_data
from .engine import AblationConfig, simulate_config
from .stages import run_stage, stage_a_configs, stage_b_configs, stage_c_configs

__all__ = [
    "AblationData", "load_ablation_data",
    "AblationConfig", "simulate_config",
    "run_stage", "stage_a_configs", "stage_b_configs", "stage_c_configs",
]
