"""Trailing risk metrics (63-trading-day realized vol, beta-to-SPY, drawdown
from trailing high) per ticker as of each rebal date -- computed once from
the same price matrix the harness already loads, cached to disk since
several risk-filter candidates in the batch below need it repeatedly."""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
CACHE = REPO / "output" / "loop_research" / "risk_metrics_cache.pkl"
WINDOW = 63


def build_risk_metrics(bundle) -> dict:
    if CACHE.exists():
        with CACHE.open("rb") as fh:
            return pickle.load(fh)
    from backtesting.data_loader import SPY

    data = bundle["data"]
    matrix = data.matrix
    rets = matrix.pct_change()
    spy_ret = rets[SPY]

    out = {}
    for d in data.rebal_dates:
        if d not in matrix.index:
            continue
        loc = matrix.index.get_loc(d)
        if loc < WINDOW:
            continue
        window = matrix.index[loc - WINDOW:loc + 1]
        r = rets.loc[window]
        spy_r = spy_ret.loc[window]
        vol = r.std() * np.sqrt(252)
        cov = r.apply(lambda col: col.cov(spy_r))
        var_spy = spy_r.var()
        beta = cov / var_spy if var_spy and var_spy > 0 else cov * np.nan
        px = matrix.loc[window]
        trailing_high = px.max()
        dd = px.loc[d] / trailing_high - 1.0
        out[d] = {"vol": vol, "beta": beta, "dd": dd}

    CACHE.parent.mkdir(parents=True, exist_ok=True)
    with CACHE.open("wb") as fh:
        pickle.dump(out, fh)
    return out
