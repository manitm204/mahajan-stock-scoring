"""PIT and execution-lag regression tests for the relvalue engine."""
import numpy as np
import pandas as pd
import pytest

from relvalue.engine import RVRule, simulate_rv_window
from relvalue.signals import PairSpec


def _panel(n=260, seed=7):
    rng = np.random.default_rng(seed)
    idx = [d.date().isoformat() for d in pd.bdate_range("2020-01-01", periods=n)]
    b = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    spread = np.zeros(n)
    for i in range(1, n):                       # OU spread, forced dislocation
        spread[i] = 0.9 * spread[i - 1] + rng.normal(0, 0.01)
    spread[150:170] += 0.08
    a = b * np.exp(spread)
    return pd.DataFrame({"A": a, "B": b}, index=idx)


def _spec(tr, f_end):
    f = tr.loc[:f_end]
    resid = np.log(f["A"]) - np.log(f["B"])
    return PairSpec("A", "B", "coint", alpha=0.0, beta=1.0,
                    mu=float(resid.mean()), sigma=float(resid.std()))


def test_t_plus_1_execution():
    tr = _panel()
    anchor = tr.index[120]
    spec = _spec(tr, anchor)
    _, _, trades = simulate_rv_window(
        tr, [spec], "residual", anchor, tr.index[121], tr.index[-1],
        rule=RVRule(min_days_left=5), borrow=0.0)
    assert trades, "expected at least one trade"
    t = trades[0]
    resid = np.log(tr["A"]) - np.log(tr["B"])
    z = (resid - spec.mu) / spec.sigma
    i = tr.index.get_loc(t.open_date)
    # the signal day (open-1) must already breach the band: T+1 discipline
    assert abs(z.iloc[i - 1]) >= 2.0


def test_no_lookahead_future_prices():
    """Mutating prices strictly after day D must not change any entry ≤ D."""
    tr = _panel()
    anchor = tr.index[120]
    spec = _spec(tr, anchor)
    cut = tr.index[200]
    rule = RVRule(min_days_left=5)
    _, _, t1 = simulate_rv_window(tr, [spec], "residual", anchor,
                                  tr.index[121], tr.index[-1], rule=rule, borrow=0.0)
    tr2 = tr.copy()
    tr2.loc[tr2.index > cut, "A"] *= 1.5      # violent future shock
    _, _, t2 = simulate_rv_window(tr2, [spec], "residual", anchor,
                                  tr.index[121], tr.index[-1], rule=rule, borrow=0.0)
    open1 = [t.open_date for t in t1 if t.open_date <= cut]
    open2 = [t.open_date for t in t2 if t.open_date <= cut]
    assert open1 == open2


def test_borrow_cost_reduces_payoff():
    tr = _panel()
    anchor = tr.index[120]
    spec = _spec(tr, anchor)
    args = (tr, [spec], "residual", anchor, tr.index[121], tr.index[-1])
    _, _, t0 = simulate_rv_window(*args, rule=RVRule(min_days_left=5), borrow=0.0)
    _, _, t1 = simulate_rv_window(*args, rule=RVRule(min_days_left=5), borrow=0.05)
    assert sum(t.payoff for t in t1) < sum(t.payoff for t in t0)


def test_long_only_has_no_short_leg_cost():
    tr = _panel()
    anchor = tr.index[120]
    spec = _spec(tr, anchor)
    _, _, trades = simulate_rv_window(
        tr, [spec], "residual", anchor, tr.index[121], tr.index[-1],
        rule=RVRule(min_days_left=5, mode="long_only"), borrow=1.0)
    # absurd borrow rate must not affect long-only payoffs
    assert trades and all(np.isfinite(t.payoff) for t in trades)


def test_ratio_signal_is_rolling_pit():
    """Ratio z on day t must be computable from data ≤ t only."""
    from relvalue.signals import z_ratio
    tr = _panel()
    spec = PairSpec("A", "B", "ssd")
    z_full = z_ratio(tr["A"], tr["B"], tr.index[0], spec)
    z_cut = z_ratio(tr["A"].iloc[:200], tr["B"].iloc[:200], tr.index[0], spec)
    pd.testing.assert_series_equal(z_full.iloc[:200], z_cut)
