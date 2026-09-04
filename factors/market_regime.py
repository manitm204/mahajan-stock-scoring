"""Market regime detection for the Factor Weight Engine.

The engine learns how predictive each factor has been *separately by regime*, so
it first needs a small, stable, point-in-time label for "what kind of market is
this." Two observable, look-ahead-free inputs drive the label:

* the **VIX close** — a direct read on implied volatility / fear; and
* **SPY relative to its 200-day moving average** — the classic trend filter for
  risk-on vs risk-off.

They are combined into three coarse states (rather than a finer grid) so each
regime accumulates enough monthly observations for the per-regime IC estimates
to mean something:

* ``risk_on``  — calm *and* trending up   (VIX low  AND SPY above its 200dma)
* ``risk_off`` — fearful *or* trending down (VIX high OR  SPY below its 200dma)
* ``neutral``  — everything in between.

Both inputs are optional: with no VIX (or too little SPY history for a 200dma)
the classifier degrades gracefully toward ``neutral`` instead of guessing.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from data.db import Database

RISK_ON = "risk_on"
NEUTRAL = "neutral"
RISK_OFF = "risk_off"
REGIMES = (RISK_ON, NEUTRAL, RISK_OFF)

LOW = "low"
MID = "normal"
HIGH = "high"

_SPY = "SPY"
_MA_WINDOW = 200


@dataclass(frozen=True)
class RegimeState:
    """Point-in-time market regime label plus the inputs that produced it."""

    label: str                          # risk_on / neutral / risk_off
    vix: float | None
    vix_bucket: str | None              # low / normal / high
    spy_to_200dma: float | None         # SPY/MA200 - 1 (None if <200 obs)
    spy_above_200dma: bool | None
    notes: list[str] = field(default_factory=list)

    @property
    def trend_up(self) -> bool | None:
        return self.spy_above_200dma


def _vix_bucket(vix: float | None, low_max: float, high_min: float) -> str | None:
    if vix is None:
        return None
    if vix < low_max:
        return LOW
    if vix > high_min:
        return HIGH
    return MID


def classify_regime(
    vix: float | None,
    spy_to_200dma: float | None,
    *,
    low_vix: float = 15.0,
    high_vix: float = 25.0,
    trend_buffer: float = 0.0,
) -> RegimeState:
    """Bucket the market into ``risk_on`` / ``neutral`` / ``risk_off``.

    ``trend_buffer`` is a dead-band around the 200dma (e.g. 0.01 = 1%) so SPY
    hovering on the line does not flip the trend flag day to day. The regime is
    deliberately asymmetric: it takes *both* calm and uptrend to be risk-on, but
    *either* fear or downtrend to be risk-off — losing money is the thing the
    weight tilt most needs to react to.
    """
    notes: list[str] = []
    bucket = _vix_bucket(vix, low_vix, high_vix)
    if vix is None:
        notes.append("VIX unavailable on/before as-of; regime ignores volatility.")

    above: bool | None
    if spy_to_200dma is None:
        above = None
        notes.append("SPY 200dma unavailable (insufficient history); regime ignores trend.")
    elif spy_to_200dma > trend_buffer:
        above = True
    elif spy_to_200dma < -trend_buffer:
        above = False
    else:
        above = None  # inside the dead-band -> treat trend as neutral

    calm = bucket == LOW
    fear = bucket == HIGH
    uptrend = above is True
    downtrend = above is False

    if fear or downtrend:
        label = RISK_OFF
    elif calm and uptrend:
        label = RISK_ON
    else:
        label = NEUTRAL

    return RegimeState(
        label=label,
        vix=vix,
        vix_bucket=bucket,
        spy_to_200dma=spy_to_200dma,
        spy_above_200dma=above,
        notes=notes,
    )


def spy_trend_to_200dma(db: Database, as_of: str) -> float | None:
    """SPY close / 200-day moving average − 1, using only prices ``<= as_of``.

    Returns ``None`` when fewer than 200 SPY closes are available on/before the
    date (the moving average would be ill-defined), which the classifier reads
    as "trend unknown."
    """
    df = db.query_df(
        "SELECT date, adj_close, close FROM daily_prices "
        "WHERE ticker = ? AND date <= ? ORDER BY date DESC LIMIT ?",
        (_SPY, as_of, _MA_WINDOW),
    )
    if df.empty or len(df) < _MA_WINDOW:
        return None
    px = df["adj_close"].fillna(df["close"]).astype(float)
    last = float(px.iloc[0])           # most recent close (DESC order)
    ma = float(px.mean())
    if ma <= 0:
        return None
    return last / ma - 1.0


def regime_from_db(
    db: Database,
    as_of: str,
    vix: float | None,
    *,
    low_vix: float = 15.0,
    high_vix: float = 25.0,
    trend_buffer: float = 0.0,
) -> RegimeState:
    """Convenience: classify the regime at ``as_of`` from the warehouse.

    ``vix`` is passed in (callers usually already hold a :class:`DataContext`
    whose ``vix()`` is point-in-time) so this only has to fetch the SPY trend.
    """
    spy = spy_trend_to_200dma(db, as_of)
    return classify_regime(
        vix, spy, low_vix=low_vix, high_vix=high_vix, trend_buffer=trend_buffer
    )
