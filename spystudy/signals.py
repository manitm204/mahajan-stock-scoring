"""Signal construction. All signals use only data up to and including date t.

Signals are computed on price-only ``close`` (the chart a trader sees);
forward returns live in ``data.forward_return`` and use total return.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from spystudy.data import SECTORS

RSI_PERIOD = 14
TREND_DAYS = 84  # ~4 months
MA_DAYS = 200
AR_WINDOW = 500  # Kritzman's absorption estimation window
AR_SHORT = 15  # days in the "recent" absorption average
AR_LONG = 252  # days in the "baseline" absorption average
MIN_SECTORS = 9  # XLRE lists 2015-10, XLC 2018-06; below this we don't score


def _wilder_average(x: pd.Series, period: int) -> pd.Series:
    """Wilder smoothing: an SMA seed over the first ``period`` values, then
    the recursion avg[t] = (avg[t-1]*(period-1) + x[t]) / period.

    A plain ``ewm(adjust=False)`` seeds on the *first* value instead, which
    leaves the first few months of RSI wrong — and this study's sample starts
    only 13 bars before its first observation.
    """
    seeded = x.copy()
    seeded.iloc[:period] = np.nan
    seeded.iloc[period] = x.iloc[1:period + 1].mean()
    # ewm starts at the first non-NaN, i.e. exactly the seed, then recurses
    return seeded.ewm(alpha=1 / period, adjust=False).mean()


def rsi(close: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    """Wilder's RSI on daily closes."""
    delta = close.diff()
    avg_gain = _wilder_average(delta.clip(lower=0.0), period)
    avg_loss = _wilder_average((-delta).clip(lower=0.0), period)
    rs = avg_gain / avg_loss
    out = 100 - 100 / (1 + rs)
    return out.where(avg_loss > 0, 100.0).where(avg_gain.notna())


def trend(close: pd.Series, days: int = TREND_DAYS) -> pd.Series:
    """Trailing return over ``days`` trading days."""
    return close / close.shift(days) - 1.0


def ma_distance(close: pd.Series, days: int = MA_DAYS) -> pd.Series:
    """Price premium over its own simple moving average, as a fraction."""
    return close / close.rolling(days, min_periods=days).mean() - 1.0


def _sector_returns(close: pd.DataFrame) -> pd.DataFrame:
    # fill_method=None: a gap must stay NaN so the complete-case window drops
    # that sector, rather than being padded into a fake 0% return
    return close[SECTORS].pct_change(fill_method=None)


def sector_correlation(close: pd.DataFrame, window: int) -> pd.Series:
    """Mean off-diagonal pairwise correlation of daily sector-ETF returns.

    Sectors absent for the whole window (XLC pre-2018, XLRE pre-2015) are
    dropped from that window's matrix rather than filled, so the statistic is
    always a genuine complete-case correlation over the sectors that existed.
    """
    rets = _sector_returns(close)
    out = pd.Series(np.nan, index=close.index, dtype=float)
    for i in range(window, len(rets) + 1):
        block = rets.iloc[i - window:i].dropna(axis=1, how="any")
        if block.shape[1] < MIN_SECTORS:
            continue
        c = np.corrcoef(block.to_numpy(), rowvar=False)
        n = c.shape[0]
        out.iloc[i - 1] = (c.sum() - n) / (n * (n - 1))
    return out


def _corr_block(rets: pd.DataFrame, i: int, window: int):
    """Complete-case correlation matrix of the window ending at position i-1."""
    block = rets.iloc[i - window:i].dropna(axis=1, how="any")
    if block.shape[1] < MIN_SECTORS:
        return None, None
    return np.corrcoef(block.to_numpy(), rowvar=False), block.columns


def sector_correlation_by_sector(close: pd.DataFrame, window: int,
                                 dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Each sector's mean correlation to the *other* sectors, on ``dates``."""
    rets = _sector_returns(close)
    pos = rets.index.get_indexer(dates)
    rows = {}
    for d, i in zip(dates, pos):
        if i < window - 1:
            continue
        c, cols = _corr_block(rets, i + 1, window)
        if c is None:
            continue
        n = c.shape[0]
        rows[d] = pd.Series((c.sum(axis=0) - 1) / (n - 1), index=cols)
    return pd.DataFrame(rows).T


def sector_correlation_leave_one_out(close: pd.DataFrame, window: int,
                                     dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Mean pairwise correlation recomputed with each sector removed.

    Answers "is one sector dragging the aggregate down?" — a large positive
    value means excluding that sector lifts the market-wide correlation, i.e.
    that sector is the one decoupling.
    """
    rets = _sector_returns(close)
    pos = rets.index.get_indexer(dates)
    rows = {}
    for d, i in zip(dates, pos):
        if i < window - 1:
            continue
        c, cols = _corr_block(rets, i + 1, window)
        if c is None:
            continue
        n = c.shape[0]
        full = (c.sum() - n) / (n * (n - 1))
        vals = {}
        for j, name in enumerate(cols):
            keep = np.delete(np.arange(n), j)
            sub = c[np.ix_(keep, keep)]
            m = sub.shape[0]
            vals[name] = (sub.sum() - m) / (m * (m - 1)) - full
        rows[d] = pd.Series(vals)
    return pd.DataFrame(rows).T


def absorption_ratio(close: pd.DataFrame, window: int = AR_WINDOW) -> pd.Series:
    """Share of sector-ETF return variance captured by the top eigenvectors.

    Kritzman et al. (2010): n_pc = round(n_assets / 5), which is 2 for both the
    9- and 11-sector eras, so the series stays comparable as sectors list.
    High AR = variance concentrated in few factors = fragile market.
    """
    rets = _sector_returns(close)
    out = pd.Series(np.nan, index=close.index, dtype=float)
    for i in range(window, len(rets) + 1):
        block = rets.iloc[i - window:i].dropna(axis=1, how="any")
        n = block.shape[1]
        if n < MIN_SECTORS:
            continue
        eig = np.linalg.eigvalsh(np.cov(block.to_numpy(), rowvar=False))[::-1]
        k = max(1, int(round(n / 5)))
        out.iloc[i - 1] = eig[:k].sum() / eig.sum()
    return out


def absorption_shift(ar: pd.Series, short: int = AR_SHORT, long: int = AR_LONG) -> pd.Series:
    """Kritzman's standardized absorption shift.

    (AR_15day_avg - AR_1year_avg) / stdev(AR over the year). Negative = the
    market is de-concentrating relative to its own recent baseline.
    """
    return ((ar.rolling(short).mean() - ar.rolling(long).mean())
            / ar.rolling(long).std())


def absorption_raw_change(ar: pd.Series, short: int = AR_SHORT) -> pd.Series:
    """Plain change in the absorption ratio over ``short`` days."""
    return ar - ar.shift(short)


RV_DAYS = 21
IVR_DAYS = 252
BETA_DAYS = 60


def beta_to(fund: pd.Series, bench: pd.Series, days: int = BETA_DAYS) -> pd.Series:
    """Rolling OLS beta of ``fund`` on ``bench`` from daily simple returns.

    cov(f, b) / var(b) over a trailing window — a risk-model beta, not a
    correlation, so it carries the amplification factor the user cares about
    (IWM at beta 1.25 moves 1.25x SPY, not merely with it).
    """
    f = fund.pct_change(fill_method=None)
    b = bench.pct_change(fill_method=None)
    cov = f.rolling(days, min_periods=days).cov(b)
    var = b.rolling(days, min_periods=days).var()
    return cov / var


def realized_vol(close: pd.Series, days: int = RV_DAYS) -> pd.Series:
    """Annualised trailing realised volatility of daily log returns."""
    lr = np.log(close / close.shift(1))
    return lr.rolling(days, min_periods=days).std() * np.sqrt(252)


def variance_risk_premium(vix: pd.Series, close: pd.Series,
                          days: int = RV_DAYS) -> pd.Series:
    """VIX minus trailing realised vol, both as decimals.

    Positive = options were priced above what the market actually delivered.
    Realised vol is *trailing*, never the forward window being predicted.
    """
    return vix / 100.0 - realized_vol(close, days)


def iv_rank(vix: pd.Series, days: int = IVR_DAYS) -> pd.Series:
    """Where VIX sits in its own trailing 1-year high-low range, 0-100."""
    lo = vix.rolling(days, min_periods=days).min()
    hi = vix.rolling(days, min_periods=days).max()
    return (vix - lo) / (hi - lo).replace(0.0, np.nan) * 100.0


def build(close: pd.DataFrame, vix: pd.Series | None = None) -> pd.DataFrame:
    """All signal series on the daily grid."""
    spy = close["SPY"]
    ar = absorption_ratio(close)
    iv = {}
    if vix is not None:
        v = vix.reindex(close.index)
        iv = {"vix": v, "realized_vol": realized_vol(spy),
              "vrp": variance_risk_premium(v, spy), "ivr": iv_rank(v)}
    return pd.DataFrame({**iv,
        "rsi14": rsi(spy),
        "sector_corr_21d": sector_correlation(close, 21),
        "sector_corr_60d": sector_correlation(close, 60),
        "trend_4m": trend(spy),
        "ma200_dist": ma_distance(spy),
        "absorption": ar,
        "absorption_shift": absorption_shift(ar),
        "absorption_chg": absorption_raw_change(ar),
    })
