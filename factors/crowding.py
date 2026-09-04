"""Factor crowding detection.

Builds a synthetic long/short return series for each factor from its *current*
top- and bottom-quintile baskets' historical daily returns, then inspects rolling
relationships to surface five informational warnings:

* ``factor_crowding_warning`` — a pair of factors whose 60-day return correlation
  exceeds the threshold (their bets are converging).
* ``high_factor_correlation_warning`` — acute co-movement in the 20-day window.
* ``momentum_reversal_warning`` — momentum's synthetic series has turned over
  (recent loss after a longer-run gain).
* ``sector_concentration_warning`` — the LONG book is concentrated in one sector.
* ``degenerate_factor_warning`` — a factor scores almost everything neutral and
  is not differentiating the universe.

These warnings are strictly informational. They never alter scores — a deliberate
separation so risk context can be read without contaminating the ranking.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Sequence

import numpy as np
import pandas as pd

from data.config import Config

from .base import FactorResult
from .utils import DataContext


@dataclass
class CrowdingReport:
    factor_returns: pd.DataFrame = field(default_factory=pd.DataFrame)
    correlations: dict[int, pd.DataFrame] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    degenerate_factors: list[str] = field(default_factory=list)


def detect_crowding(
    ctx: DataContext,
    results: Sequence[FactorResult],
    composite: pd.DataFrame,
    cfg: Config,
) -> CrowdingReport:
    cc = cfg.get("factors", "crowding", default={}) or {}
    quintile = float(cc.get("quintile", 0.20))
    windows = [int(w) for w in cc.get("return_windows", [20, 60, 120])]
    corr_threshold = float(cc.get("high_correlation_threshold", 0.70))
    degen_frac = float(cc.get("degenerate_neutral_fraction", 0.80))
    sector_threshold = float(cc.get("sector_concentration_threshold", 0.40))
    reversal_lb = int(cc.get("momentum_reversal_lookback", 20))

    report = CrowdingReport()

    # 1. Degenerate factors -------------------------------------------------
    for res in results:
        if res.neutral_fraction() >= degen_frac:
            report.degenerate_factors.append(res.key)
            report.warnings.append(
                f"degenerate_factor_warning: '{res.key}' scores "
                f"{res.neutral_fraction():.0%} of names neutral (insufficient data to differentiate)"
            )

    # 2. Synthetic factor returns from current quintile baskets -------------
    max_window = max(windows) if windows else 120
    returns = ctx.daily_returns(lookback_days=int(max_window * 1.7) + 40)
    factor_returns = _synthetic_factor_returns(results, returns, quintile)
    report.factor_returns = factor_returns

    if not factor_returns.empty and factor_returns.shape[1] >= 2:
        # 3. Rolling correlations per window --------------------------------
        for w in windows:
            tail = factor_returns.tail(w)
            if len(tail) >= max(5, w // 2):
                report.correlations[w] = tail.corr()

        # factor_crowding_warning from the 60d (structural) window.
        structural = report.correlations.get(60)
        if structural is None and report.correlations:
            structural = report.correlations[max(report.correlations)]
        if structural is not None:
            for a, b in combinations(structural.columns, 2):
                c = structural.loc[a, b]
                if pd.notna(c) and abs(c) > corr_threshold:
                    report.warnings.append(
                        f"factor_crowding_warning: '{a}' and '{b}' 60d return corr = {c:+.2f}"
                    )

        # high_factor_correlation_warning from the acute 20d window.
        acute = report.correlations.get(20)
        if acute is not None:
            off = acute.where(~np.eye(len(acute), dtype=bool))
            mx = off.abs().max().max()
            if pd.notna(mx) and mx > max(corr_threshold + 0.10, 0.80):
                report.warnings.append(
                    f"high_factor_correlation_warning: peak 20d factor corr = {mx:.2f} "
                    "(factors moving together acutely)"
                )

        # 4. Momentum reversal ---------------------------------------------
        if "momentum" in factor_returns.columns:
            mom = factor_returns["momentum"].dropna()
            recent = mom.tail(reversal_lb).sum()
            longrun = mom.tail(max_window).sum()
            if recent < 0 < longrun:
                report.warnings.append(
                    f"momentum_reversal_warning: momentum factor {reversal_lb}d return "
                    f"{recent:+.2%} reversing a {max_window}d gain of {longrun:+.2%}"
                )

    # 5. Sector concentration in the LONG book ------------------------------
    longs = composite[composite["long_short_flag"] == "LONG"]
    if len(longs) > 0:
        shares = longs["sector"].value_counts(normalize=True)
        top_sector, top_share = shares.index[0], float(shares.iloc[0])
        if top_share > sector_threshold:
            report.warnings.append(
                f"sector_concentration_warning: {top_share:.0%} of LONG candidates "
                f"are in {top_sector}"
            )

    return report


def _synthetic_factor_returns(
    results: Sequence[FactorResult],
    returns: pd.DataFrame,
    quintile: float,
) -> pd.DataFrame:
    """Daily top-minus-bottom quintile return series for each factor."""
    if returns.empty:
        return pd.DataFrame()
    series: dict[str, pd.Series] = {}
    for res in results:
        parent = res.parent.dropna()
        if parent.empty:
            continue
        hi_cut = parent.quantile(1 - quintile)
        lo_cut = parent.quantile(quintile)
        top = [t for t in parent.index[parent >= hi_cut] if t in returns.columns]
        bot = [t for t in parent.index[parent <= lo_cut] if t in returns.columns]
        if len(top) < 2 or len(bot) < 2:
            continue
        series[res.key] = returns[top].mean(axis=1) - returns[bot].mean(axis=1)
    if not series:
        return pd.DataFrame()
    return pd.DataFrame(series).dropna(how="all")
