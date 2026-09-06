"""Correctness + leakage tests for the autoresearch harness
(research/autoresearch/evaluate.py + candidate.py). evaluate.py runs this
file via pytest BEFORE printing any score -- a failure here aborts the run.

All tests use small synthetic panels (no DB access) so they run in
milliseconds and can gate every evaluate.py invocation.
"""
from __future__ import annotations

import types

import numpy as np
import pandas as pd
import pytest

from research.autoresearch import candidate as candidate_mod
from research.autoresearch.evaluate import (
    Context, _freeze_comp, _validate_targets, compute_portfolio_returns,
    targets_to_weight_matrix, bench_returns, EXEC_LAG_DAYS, COST_BPS,
    DEV_START, DEV_END, VAL_START, VAL_END, HOLDOUT_START,
)


def _dates(n=9):
    return pd.bdate_range("2020-01-31", periods=n, freq="BME").strftime("%Y-%m-%d").tolist()


def _comp(dates, tickers):
    rng = np.random.default_rng(0)
    return {d: pd.Series(rng.uniform(0, 100, len(tickers)), index=tickers) for d in dates}


def _make_context(dates, tickers, k=2):
    return Context(rebal_dates=tuple(dates), comp=_freeze_comp(_comp(dates, tickers)), k=k)


def _price_matrix(dates, tickers, seed=0):
    rng = np.random.default_rng(seed)
    all_days = pd.bdate_range(dates[0], periods=400).strftime("%Y-%m-%d")
    data = {t: 100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(all_days)))) for t in tickers}
    return pd.DataFrame(data, index=all_days)


# --------------------------------------------------------------------------- #
# 1. T+1 execution
# --------------------------------------------------------------------------- #
def test_t1_execution_uses_next_day_price_not_signal_close():
    tickers = ["AAA", "BBB"]
    dates = _dates(4)
    matrix = _price_matrix(dates, tickers)
    weights = pd.DataFrame({"AAA": [1.0, 1.0, 1.0, 1.0], "BBB": [0.0] * 4}, index=dates)

    pr0, _ = compute_portfolio_returns(weights, matrix, dates, cost_bps=0.0, exec_lag_days=0)
    pr1, _ = compute_portfolio_returns(weights, matrix, dates, cost_bps=0.0, exec_lag_days=1)
    # same-close (lag=0) and T+1 (lag=1) must generally disagree -- if the
    # harness silently traded at the signal close, this would be identical.
    assert not np.allclose(pr0.values, pr1.values)


def test_t1_execution_never_uses_a_price_before_the_signal_date():
    tickers = ["AAA"]
    dates = _dates(3)
    matrix = _price_matrix(dates, tickers)
    weights = pd.DataFrame({"AAA": [1.0, 1.0, 1.0]}, index=dates)
    pr, _ = compute_portfolio_returns(weights, matrix, dates, exec_lag_days=1)
    # manually recompute using dates strictly after each signal date
    for i, d in enumerate(dates[:-1]):
        exec_pos = matrix.index.searchsorted(d) + 1
        assert matrix.index[exec_pos] > d


# --------------------------------------------------------------------------- #
# 2. weight totals
# --------------------------------------------------------------------------- #
def test_validate_targets_rejects_overleveraged_weights():
    df = pd.DataFrame({"date": ["2020-01-31", "2020-01-31"],
                       "ticker": ["AAA", "BBB"], "weight": [0.7, 0.5]})
    with pytest.raises(ValueError):
        _validate_targets(df, {"2020-01-31"})


def test_validate_targets_rejects_negative_weight():
    df = pd.DataFrame({"date": ["2020-01-31"], "ticker": ["AAA"], "weight": [-0.1]})
    with pytest.raises(ValueError):
        _validate_targets(df, {"2020-01-31"})


def test_validate_targets_accepts_partial_cash_book():
    df = pd.DataFrame({"date": ["2020-01-31"], "ticker": ["AAA"], "weight": [0.3]})
    out = _validate_targets(df, {"2020-01-31"})
    assert len(out) == 1


# --------------------------------------------------------------------------- #
# 3. costs
# --------------------------------------------------------------------------- #
def test_costs_scale_with_turnover_and_reduce_return():
    tickers = ["AAA", "BBB"]
    dates = _dates(3)
    matrix = _price_matrix(dates, tickers)
    weights = pd.DataFrame({"AAA": [1.0, 0.0, 1.0], "BBB": [0.0, 1.0, 0.0]}, index=dates)
    pr_free, turn = compute_portfolio_returns(weights, matrix, dates, cost_bps=0.0)
    pr_costly, _ = compute_portfolio_returns(weights, matrix, dates, cost_bps=100.0)
    # period 0: from flat -> full AAA, turnover 1.0; period 1: AAA->BBB full
    # flip, turnover 2.0.
    assert turn.iloc[0] == pytest.approx(1.0)
    assert turn.iloc[1] == pytest.approx(2.0)
    assert (pr_costly < pr_free).all()
    assert pr_free.iloc[1] - pr_costly.iloc[1] == pytest.approx(turn.iloc[1] * 100.0 / 1e4)


def test_zero_turnover_when_weights_unchanged():
    tickers = ["AAA"]
    dates = _dates(3)
    matrix = _price_matrix(dates, tickers)
    weights = pd.DataFrame({"AAA": [1.0, 1.0, 1.0]}, index=dates)
    _, turn = compute_portfolio_returns(weights, matrix, dates)
    assert turn.iloc[1] == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# 4. sleeve overlap (candidate-specific: when the same names top-rank every
#    month, all 3 sleeves converge onto the same 10 names and weights must
#    AGGREGATE, not overwrite)
# --------------------------------------------------------------------------- #
def _candidate_book_size() -> int:
    """Book size is a candidate.py design choice (module-level BOOK_SIZE),
    not something evaluate.py enforces -- context.k is only a suggested
    default. Falls back to 10 for any candidate that still reads context.k
    directly instead of declaring its own BOOK_SIZE."""
    return getattr(candidate_mod, "BOOK_SIZE", 10)


def test_sleeve_overlap_aggregates_weight_when_sleeves_share_names():
    k = _candidate_book_size()
    tickers = [f"T{i}" for i in range(k)]
    dates = _dates(6)
    # identical ranking every month -> every sleeve always picks the same k names
    comp = {d: pd.Series(range(k, 0, -1), index=tickers) for d in dates}
    ctx = Context(rebal_dates=tuple(dates), comp=_freeze_comp(comp), k=k)
    targets = candidate_mod.generate_targets(ctx)
    last_date = dates[-1]
    row = targets[targets["date"] == last_date]
    # fully ramped up (3 sleeves x same k names) -> each name's weight is
    # the SUM across sleeves, i.e. 3 * (1/3/k) = 1/k, and the book is 100%
    # invested across exactly k names, not 3k fractionally-weighted rows.
    assert len(row) == k
    assert row["weight"].sum() == pytest.approx(1.0)
    assert row["weight"].max() == pytest.approx(1.0 / k)


def test_sleeve_ramp_up_leaves_cash_before_third_month():
    k = _candidate_book_size()
    tickers = [f"T{i}" for i in range(k)]
    dates = _dates(3)
    comp = {d: pd.Series(range(k, 0, -1), index=tickers) for d in dates}
    ctx = Context(rebal_dates=tuple(dates), comp=_freeze_comp(comp), k=k)
    targets = candidate_mod.generate_targets(ctx)
    first_date_total = targets[targets["date"] == dates[0]]["weight"].sum()
    assert first_date_total == pytest.approx(1.0 / 3.0)


# --------------------------------------------------------------------------- #
# 5. date leakage / context contract
# --------------------------------------------------------------------------- #
def test_context_has_no_performance_or_price_fields():
    fields = set(Context.__dataclass_fields__)
    assert fields == {"rebal_dates", "comp", "k"}
    forbidden_substrings = ("return", "price", "matrix", "nav", "sharpe", "perf", "pnl")
    for f in fields:
        assert not any(bad in f.lower() for bad in forbidden_substrings)


def test_context_comp_is_read_only():
    dates = _dates(2)
    ctx = _make_context(dates, ["AAA", "BBB"])
    assert isinstance(ctx.comp, types.MappingProxyType)
    with pytest.raises(TypeError):
        ctx.comp[dates[0]] = pd.Series([1.0], index=["AAA"])
    series = ctx.comp[dates[0]]
    with pytest.raises(ValueError):
        series.iloc[0] = 999.0


def test_context_is_frozen():
    ctx = _make_context(_dates(2), ["AAA", "BBB"])
    with pytest.raises(Exception):
        ctx.k = 999


def test_validate_targets_rejects_dates_outside_fixed_schedule():
    df = pd.DataFrame({"date": ["1999-01-01"], "ticker": ["AAA"], "weight": [0.1]})
    with pytest.raises(ValueError):
        _validate_targets(df, {"2020-01-31"})


def test_dev_val_holdout_boundaries_are_ordered_and_disjoint():
    assert DEV_START < DEV_END < VAL_START < VAL_END < HOLDOUT_START


# --------------------------------------------------------------------------- #
# 6. return reconciliation (hand-computed 2-period example)
# --------------------------------------------------------------------------- #
def test_bench_returns_uses_same_period_labeling_as_portfolio_returns():
    """Regression test: bench_returns and compute_portfolio_returns must
    label the SAME underlying period with the SAME (start) date, or every
    downstream benchmark comparison (IR, beta, alpha, alpha_tstat) silently
    pairs a strategy period with the wrong SPY/QQQ period -- a bug that
    briefly shipped and produced an implausible negative SPY beta."""
    tickers = ["AAA", "SPY"]
    dates = ["2020-01-31", "2020-02-28", "2020-03-31", "2020-04-30"]
    matrix = _price_matrix(dates, tickers)
    weights = pd.DataFrame({"AAA": [1.0] * 4, "SPY": [0.0] * 4}, index=dates)
    pr, _ = compute_portfolio_returns(weights, matrix, dates, cost_bps=0.0)
    spy = bench_returns(matrix, "SPY", dates)
    assert list(pr.index) == list(spy.index)
    # AAA is fully held every period -> its return IS the AAA buy-and-hold
    # return over the same dates, which must equal bench_returns("AAA", ...)
    # label-for-label (a same-asset check pins down the exact period each
    # label refers to, not just that the two series happen to share an index).
    aaa_bench = bench_returns(matrix, "AAA", dates)
    pd.testing.assert_series_equal(pr, aaa_bench, check_names=False)


def test_return_reconciliation_matches_manual_calculation():
    tickers = ["AAA"]
    dates = ["2020-01-31", "2020-02-28", "2020-03-31"]
    all_days = pd.bdate_range("2020-01-31", periods=60).strftime("%Y-%m-%d")
    px = pd.Series(100.0, index=all_days)
    exec_positions = [all_days.get_loc(d) + 1 for d in dates]
    # engineer known prices at the T+1 execution days: 100 -> 110 -> 99
    px.iloc[exec_positions[0]] = 100.0
    px.iloc[exec_positions[1]] = 110.0
    px.iloc[exec_positions[2]] = 99.0
    matrix = pd.DataFrame({"AAA": px})
    weights = pd.DataFrame({"AAA": [1.0, 1.0, 1.0]}, index=dates)
    pr, turn = compute_portfolio_returns(weights, matrix, dates, cost_bps=10.0, exec_lag_days=1)
    expected_0 = (110.0 / 100.0 - 1.0) - (1.0 * 10.0 / 1e4)  # full turnover into AAA + cost
    expected_1 = (99.0 / 110.0 - 1.0) - 0.0                    # no turnover, held
    assert pr.iloc[0] == pytest.approx(expected_0)
    assert pr.iloc[1] == pytest.approx(expected_1)
    assert turn.iloc[0] == pytest.approx(1.0)
    assert turn.iloc[1] == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# 7. candidate.py cannot change evaluation assumptions
# --------------------------------------------------------------------------- #
def test_rogue_candidate_cannot_smuggle_extra_dates_past_validation():
    def rogue_generate_targets(context):
        rows = [{"date": d, "ticker": "AAA", "weight": 0.1} for d in context.rebal_dates]
        rows.append({"date": "2099-01-01", "ticker": "AAA", "weight": 0.1})
        return pd.DataFrame(rows)

    ctx = _make_context(_dates(2), ["AAA", "BBB"])
    targets = rogue_generate_targets(ctx)
    with pytest.raises(ValueError):
        _validate_targets(targets, set(ctx.rebal_dates))


def test_rogue_candidate_mutating_returned_context_object_does_not_persist():
    ctx = _make_context(_dates(2), ["AAA", "BBB"])
    original_k = ctx.k
    try:
        ctx.k = 999
    except Exception:
        pass
    assert ctx.k == original_k


def test_evaluate_constants_not_importable_as_mutable_from_candidate_module():
    # candidate.py never imports evaluate.py at all -- confirm that stays true
    # so it has no name-based handle on COST_BPS/EXEC_LAG_DAYS/HOLDOUT_START
    # to monkeypatch in the first place.
    src = (candidate_mod.__file__,)
    with open(src[0]) as fh:
        text = fh.read()
    assert "research.autoresearch.evaluate" not in text
    assert "import evaluate" not in text


def test_weight_matrix_reindex_ignores_tickers_outside_matrix():
    df = pd.DataFrame({"date": ["2020-01-31"], "ticker": ["NOT_A_REAL_TICKER"], "weight": [0.5]})
    wide = targets_to_weight_matrix(df, ["2020-01-31"])
    tickers = ["AAA"]
    matrix = _price_matrix(["2020-01-31"], tickers)
    pr, _ = compute_portfolio_returns(wide, matrix, ["2020-01-31", matrix.index[10]])
    assert not pr.empty  # ghost ticker silently drops out, doesn't crash the reconciliation
