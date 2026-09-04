"""Loop-engineering harness (user request 2026-09-02): score one
StrategyConfig on the same production EQEFF composite + managed-book engine
used for the original 20-strategy sweep, split into a TRAIN window (for
deciding whether an idea is an improvement) and a VAL window (to catch
overfitting before it gets promoted).

TRAIN = 2020-01 .. 2023-12 (~48 monthly reviews)
VAL   = 2024-01 .. 2026-06 (~30 monthly reviews)

Only ~78 months of data exist total, so a 20-round greedy hill-climb on
train Sharpe alone would eventually just curve-fit noise. The promotion rule
in `better_than_champion` requires the challenger to also hold up on VAL
(within PROMOTE_VAL_RATIO of the champion's val Sharpe) before it replaces
the champion -- see scripts/loop_round.py for how this gets used each round.
"""
from __future__ import annotations

from functools import lru_cache

import pandas as pd

from backtesting.data_loader import SPY, QQQ
from research.ablation.engine import alpha_tstat
from research.strategies.engine import load_data, simulate_managed_book
from research.strategies.generic_rule import StrategyConfig, build_rule
from research.walkforward.portfolio import performance_metrics
from scripts.run_strategy_sweep import bench_returns, monthly_returns

TRAIN_END = "2023-12-31"
PROMOTE_VAL_RATIO = 0.70  # challenger's val Sharpe must be >= this * champion's
MIN_TRAIN_IMPROVEMENT = 0.02  # train Sharpe must beat champion by more than this -- a
                               # smaller gap is noise, not signal, over only 46 train months

# --------------------------------------------------------------------------- #
# v2 promotion rule (user request 2026-09-02, restart after 71-round drift):
# rounds 1-71 optimized train Sharpe alone with only a floor on val, and the
# result was 11 promotions that steadily traded val away for train (1.24 -> 0.88
# val while train climbed 1.02 -> 1.41) -- each step individually looked fine,
# but the accumulated drift was overfitting. COMBINED_SCORE requires a
# candidate to improve on BOTH legs together, not just avoid collapsing one.
# --------------------------------------------------------------------------- #
MIN_COMBINED_IMPROVEMENT = 0.02   # (train+val)/2 must beat champion's by more than this
MAX_LEG_DROP = 0.10                # neither train nor val may drop by more than this,
                                    # even if the combined score improves overall


def combined_score(stats: dict) -> float:
    return (stats["train_sharpe"] + stats["val_sharpe"]) / 2.0


@lru_cache(maxsize=1)
def _data():
    return load_data()


def _window_stats(pr: pd.Series, spy: pd.Series, qqq: pd.Series) -> dict:
    if len(pr) < 3:
        return {"cagr": float("nan"), "sharpe": float("nan"), "beta": float("nan"),
               "max_dd": float("nan"), "alpha_vs_spy": float("nan"),
               "alpha_t_vs_spy": float("nan"), "n_months": len(pr)}
    m = performance_metrics(pr, hold_months=1, benchmarks={SPY: spy, QQQ: qqq})
    return {
        "cagr": m["cagr"], "sharpe": m["sharpe"], "beta": m.get("spy_beta"),
        "max_dd": m.get("max_drawdown"), "alpha_vs_spy": m.get("spy_alpha"),
        "alpha_t_vs_spy": alpha_tstat(pr, spy), "n_months": m["n_periods"],
    }


def run_candidate(cfg: StrategyConfig) -> dict:
    """Simulate `cfg` once over the full period, then slice the resulting
    monthly returns into train/val windows -- one sim per candidate, not two."""
    data = _data()
    rebal = data.rebal_dates
    spy = bench_returns(data.matrix, SPY, rebal)
    qqq = bench_returns(data.matrix, QQQ, rebal)

    rule = build_rule(cfg)
    nav = simulate_managed_book(
        data, rule, k=cfg.k, rank_entry_k=cfg.entry_rank_k,
        max_per_sector=cfg.max_per_sector, sector_map=data.sector,
        min_value_pct=cfg.min_value_pct)
    pr = monthly_returns(nav, rebal)

    def _split(s, cutoff, after):
        return s[s.index > cutoff] if after else s[s.index <= cutoff]

    full = _window_stats(pr, spy, qqq)
    train = _window_stats(_split(pr, TRAIN_END, False), _split(spy, TRAIN_END, False),
                          _split(qqq, TRAIN_END, False))
    val = _window_stats(_split(pr, TRAIN_END, True), _split(spy, TRAIN_END, True),
                        _split(qqq, TRAIN_END, True))

    out = {f"full_{k}": v for k, v in full.items()}
    out.update({f"train_{k}": v for k, v in train.items()})
    out.update({f"val_{k}": v for k, v in val.items()})
    out["config"] = cfg.to_dict()
    return out


def better_than_champion_v2(candidate: dict, champion: dict, baseline: dict | None = None) -> bool:
    """Combined-score promotion rule: a candidate must improve
    (train_sharpe + val_sharpe)/2 over the champion by more than
    MIN_COMBINED_IMPROVEMENT, AND must not drop either leg individually by
    more than MAX_LEG_DROP -- so it can't win purely by trading one metric
    for the other. `baseline` still guards against slow multi-round drift
    away from the ORIGINAL round-0 numbers on either leg."""
    ts, vs = candidate["train_sharpe"], candidate["val_sharpe"]
    if ts != ts or vs != vs:  # NaN
        return False
    cts, cvs = champion["train_sharpe"], champion["val_sharpe"]
    if combined_score(candidate) <= combined_score(champion) + MIN_COMBINED_IMPROVEMENT:
        return False
    if ts < cts - MAX_LEG_DROP or vs < cvs - MAX_LEG_DROP:
        return False
    if baseline is not None:
        if vs < PROMOTE_VAL_RATIO * baseline["val_sharpe"]:
            return False
        if ts < PROMOTE_VAL_RATIO * baseline["train_sharpe"]:
            return False
    return True


MIN_DUAL_IMPROVEMENT = 0.02  # both legs must clear the champion by more than this


def better_than_champion_v3(candidate: dict, champion: dict, baseline: dict | None = None) -> bool:
    """v3 promotion rule (user request 2026-09-03): a candidate is only
    promoted if it beats the champion on BOTH train_sharpe AND val_sharpe
    individually (each by more than MIN_DUAL_IMPROVEMENT) -- strict
    dominance, not a combined average and not just "doesn't collapse."
    `baseline` is accepted for interface parity with v1/v2 but unused: since
    every promotion under this rule strictly increases val, the champion's
    val can never fall below the round-0 baseline by construction, so no
    separate drift guard is needed."""
    ts, vs = candidate["train_sharpe"], candidate["val_sharpe"]
    if ts != ts or vs != vs:  # NaN
        return False
    if ts <= champion["train_sharpe"] + MIN_DUAL_IMPROVEMENT:
        return False
    if vs <= champion["val_sharpe"] + MIN_DUAL_IMPROVEMENT:
        return False
    return True


def better_than_champion(candidate: dict, champion: dict, baseline: dict | None = None) -> bool:
    """`baseline` (round-0's stats, if supplied) guards against SLOW DRIFT: many
    small round-over-round promotions can each individually pass the local
    val-ratio check against the immediately preceding champion while the val
    Sharpe erodes steadily over many rounds. Requiring val to also hold up
    against the ORIGINAL baseline (not just the last champion) catches that."""
    ts = candidate["train_sharpe"]
    if ts != ts:  # NaN
        return False
    if ts <= champion["train_sharpe"] + MIN_TRAIN_IMPROVEMENT:
        return False
    vs, cvs = candidate["val_sharpe"], champion["val_sharpe"]
    if vs != vs:
        return False
    if vs < PROMOTE_VAL_RATIO * cvs:
        return False
    if baseline is not None and vs < PROMOTE_VAL_RATIO * baseline["val_sharpe"]:
        return False
    return True
