"""Multi-horizon forward returns realized after each rebalance date.

For every rebalance date ``d_k`` and horizon H in {1, 3, 6, 12} months we realize
the return each name earned from ``d_k`` to the trading date nearest ``d_k + H
months``. Horizons longer than one month *overlap* (a 12-month window started this
month still has 11 months in common with next month's), which is standard for IC
studies but inflates apparent significance — the information-ratio denominators
downstream are therefore read as indicative, not as independent-sample t-stats.

Strictly forward-looking: a window's end date is always > its start date, and a
window whose end would fall past the last available price is simply dropped (no
look-ahead, just missing). All prices come from the same adjusted-close matrix the
backtester uses.

Delisting realization: a name whose series *ends* inside a window (delisting,
acquisition, bankruptcy) realizes its last print — the proceeds sit in cash for the
remainder of the window — instead of being dropped from the cross-section. Dropping
was a survivorship leak: it silently excluded exactly the terminal losses (FRC, SIVB,
…) a loser-avoiding signal should be judged against.
"""
from __future__ import annotations

import pandas as pd

HORIZON_MONTHS: dict[str, int] = {"1M": 1, "3M": 3, "6M": 6, "12M": 12}
_MAX_GAP_DAYS = 25   # tolerance when snapping d_k + H months to a real trading date


def _nearest_trading_date(
    index_ts: pd.DatetimeIndex, target: pd.Timestamp, max_gap_days: int = _MAX_GAP_DAYS
) -> pd.Timestamp | None:
    """Trading date in ``index_ts`` closest to ``target`` within the tolerance."""
    if len(index_ts) == 0:
        return None
    pos = index_ts.searchsorted(target)
    candidates = []
    if pos < len(index_ts):
        candidates.append(index_ts[pos])
    if pos > 0:
        candidates.append(index_ts[pos - 1])
    best = min(candidates, key=lambda d: abs((d - target).days), default=None)
    if best is None or abs((best - target).days) > max_gap_days:
        return None
    return best


def realize_delistings(matrix: pd.DataFrame) -> pd.DataFrame:
    """Forward-fill each name's price series within the matrix's date range.

    After a name's last print its price is carried forward, so any return window
    spanning the delisting realizes ``last_print / entry − 1`` and then holds flat
    (cash). Uses only past prices — no look-ahead — and never extends the matrix
    index, so windows past the end of data are still dropped. Names not yet listed
    stay NaN (ffill has nothing to fill before the first print).
    """
    return matrix.ffill()


def compute_forward_returns(
    matrix: pd.DataFrame,
    rebal_dates: list[str],
    horizons: dict[str, int] | None = None,
) -> dict[str, dict[str, pd.Series]]:
    """Forward returns per horizon.

    Returns ``{horizon_label: {start_date: per-ticker forward-return Series}}``.
    The start dates are a subset of ``rebal_dates`` — those with a valid window
    end inside the price matrix.
    """
    horizons = horizons or HORIZON_MONTHS
    matrix = realize_delistings(matrix)
    index_ts = pd.DatetimeIndex(pd.to_datetime(matrix.index))
    # Map normalized timestamp -> original index label so we can .loc the matrix.
    label_by_ts = {pd.Timestamp(ts): lbl for ts, lbl in zip(index_ts, matrix.index)}

    out: dict[str, dict[str, pd.Series]] = {h: {} for h in horizons}
    for d in rebal_dates:
        if d not in matrix.index:
            continue
        start_ts = pd.Timestamp(d)
        start_px = matrix.loc[d]
        for label, months in horizons.items():
            target = start_ts + pd.DateOffset(months=months)
            end_ts = _nearest_trading_date(index_ts, target)
            if end_ts is None or end_ts <= start_ts:
                continue
            end_px = matrix.loc[label_by_ts[end_ts]]
            fwd = (end_px / start_px) - 1.0
            out[label][d] = fwd.dropna()
    return out
