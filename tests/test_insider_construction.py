"""Rebuilt insider factor — construction for dispersion.

Pins the behaviour that makes the factor populate ~90% of the universe instead of
sitting near neutral: a continuous buy/sell ratio and signed flow defined for any
name with open-market activity, plus a single combined high-conviction flag.
"""
from __future__ import annotations

import pandas as pd
import pytest

from factors.insider import _aggregate


class _Ctx:
    """Minimal stand-in exposing only insider_window()."""
    def __init__(self, df: pd.DataFrame):
        self._df = df

    def insider_window(self) -> pd.DataFrame:
        return self._df


class _FakeDb:
    """Records query params so tests can pin the PIT bounds of the window."""
    def __init__(self, as_of: str):
        self._as_of = as_of
        self.captured: list[tuple[str, tuple]] = []

    def max_value(self, table, column, where=None, params=()):
        return self._as_of

    def query_df(self, sql, params=()):
        self.captured.append((sql, params))
        return pd.DataFrame(columns=["ticker", "insider_name", "insider_title",
                                     "transaction_code", "shares", "price",
                                     "value", "transaction_date"])


def test_insider_window_is_point_in_time():
    """Regression: a historical as_of must bound the window at as_of, not today.

    The old code (`... if self.as_of > "9000-01-01" else today`) sent every
    real backtest date down the `today` branch, leaking future transactions
    into historical scoring.
    """
    from factors.utils import DataContext

    ctx = DataContext(db=_FakeDb("2020-06-30"), as_of="2020-06-30",
                      universe=["AAPL"])
    ctx.insider_window()
    (_, params), = ctx.db.captured
    upper, start = params
    assert upper == "2020-06-30"
    assert start == (pd.Timestamp("2020-06-30")
                     - pd.Timedelta(days=ctx.insider_window_days)).date().isoformat()


def _txn(ticker, code, value, name="John Insider", title=""):
    return {"ticker": ticker, "insider_name": name, "insider_title": title,
            "transaction_code": code, "shares": 100, "price": value / 100,
            "value": value, "transaction_date": "2026-01-15"}


def _agg(rows):
    return _aggregate(_Ctx(pd.DataFrame(rows)))


def test_buy_sell_ratio_extremes_and_coverage():
    agg = _agg([
        _txn("BUYONLY", "P", 50_000),
        _txn("SELLONLY", "S", 80_000),
        _txn("MIXED", "P", 30_000), _txn("MIXED", "S", 10_000),
        _txn("AWARDS", "A", 100_000),   # option award only → no P/S
    ])
    r = agg["buy_sell_ratio"]
    assert r["BUYONLY"] == 1.0
    assert r["SELLONLY"] == 0.0
    assert r["MIXED"] == pytest.approx(30_000 / 40_000)
    # Award-only name has no open-market activity → not covered by the ratio.
    assert "AWARDS" not in r.index or pd.isna(r.get("AWARDS"))


def test_signed_net_flow_weights_sales_less():
    agg = _agg([_txn("X", "P", 100_000), _txn("X", "S", 100_000)])
    # +100k buy, -0.5*100k sell = +50k
    assert agg["net_flow"]["X"] == pytest.approx(50_000.0)


def test_high_conviction_flag_combines_patterns():
    agg = _agg([
        _txn("CEO", "P", 20_000, title="Chief Executive Officer"),
        _txn("BIG", "P", 2_000_000),                       # > $1M
        _txn("CLUSTER", "P", 5_000, name="A"),
        _txn("CLUSTER", "P", 5_000, name="B"),
        _txn("CLUSTER", "P", 5_000, name="C"),             # 3 distinct buyers
        _txn("SMALLBUY", "P", 1_000),                      # activity, no conviction
        _txn("SELLER", "S", 40_000),                       # activity, no buys
    ])
    hc = agg["high_conviction"]
    assert hc["CEO"] == 1.0
    assert hc["BIG"] == 1.0
    assert hc["CLUSTER"] == 1.0
    assert hc["SMALLBUY"] == 0.0     # active but not a conviction buy
    assert hc["SELLER"] == 0.0       # active (sells) → scored low, not missing


def test_absent_names_are_missing_not_zero():
    agg = _agg([_txn("ACTIVE", "S", 10_000)])
    # A name that never appears is simply absent from every series (→ neutral 50
    # once sector-percentiled), never a hard 0.
    assert "GHOST" not in agg["high_conviction"].index
    assert "GHOST" not in agg["net_flow"].index


def test_empty_window_returns_empty_series():
    agg = _aggregate(_Ctx(pd.DataFrame()))
    for k in ("net_flow", "buy_sell_ratio", "high_conviction"):
        assert agg[k].empty
