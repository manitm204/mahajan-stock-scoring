"""Live 'what would you buy' snapshot for the two loop-engineering research
strategies shown on the Strategy Lab page (user request 2026-09-06).

Unlike the equity-curve/metrics comparison (scripts/generate_strategy_lab_
comparison.py, run against the frozen 2020-2026 research backtest panel),
this module reads the LIVE production ``composite_scores`` / ``daily_prices``
tables so "today" is the real current date.

Two views per strategy:

* :func:`form_today` -- the basket you'd buy if you started the portfolio
  right now, using the strategy's own entry rule on the latest live
  composite scores. No performance to show yet.
* :func:`month_ago_scenarios` -- the basket you'd have bought ~1 month ago,
  what actually happened to it since (v3: mechanically re-applies its real
  10%-trailing-stop / worst-quartile-rank exit rules day by day against live
  prices, refilling any emptied slots at today's review; book4: buy-and-hold
  on sleeve 0, plus sleeve 1 coming online today -- see below), and the
  realized CAGR/beta/alpha vs SPY over that window.

book4/evict3's true mechanic (research/autoresearch/candidate.py) is 3
STAGGERED sleeves, not 3 sleeves formed all at once: sleeve j only forms
``j * STEP`` months after inception (STEP=1 here), each holding 4 names, so
a fresh deploy only buys sleeve 0's 4 names on day one -- sleeve 1 comes
online a month later, sleeve 2 two months later. A basket a sleeve hasn't
reached yet holds no names and its 1/12-per-slot allocation sits in cash.
This is why "buy today" is only 4 names, and "started a month ago" shows 4
(sleeve 0, held since inception) + 4 more (sleeve 1, just formed today) --
not the full 12 right away.

Both baskets use plain equal weighting (1/12 per book4 slot, 1/10 per v3
slot), matching each strategy's own position-sizing convention in
research/strategies/engine.py and research/autoresearch/candidate.py.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import streamlit as st

from backtesting.data_loader import SPY, load_price_matrix
from data.db import get_db

DEFAULT_TTL = 300

V3_K = 10
V3_TRAIL_PCT = 0.10
V3_MIN_HOLD_DAYS = 7          # ~min_hold_months=0.25 (1 week) in the research config
V3_RANK_FLOOR_PCT = 0.75      # worst-quartile exit at review
V3_COST_BPS = 10.0

BOOK4_SIZE = 4
BOOK4_SLEEVES = 3
BOOK4_STEP_MONTHS = 1          # max(HOLD_MONTHS // SLEEVE_COUNT, 1) = max(4//3, 1)
BOOK4_SLOT_WEIGHT = 1.0 / (BOOK4_SLEEVES * BOOK4_SIZE)   # 1/12, same as candidate.py


# ---------------------------------------------------------------------------
# Live data access
# ---------------------------------------------------------------------------
@st.cache_data(ttl=DEFAULT_TTL, show_spinner=False)
def _score_dates() -> list[str]:
    db = get_db()
    df = db.query_df("SELECT DISTINCT as_of_date FROM composite_scores ORDER BY as_of_date")
    return df["as_of_date"].tolist()


@st.cache_data(ttl=DEFAULT_TTL, show_spinner=False)
def _scores(date: str) -> pd.Series:
    db = get_db()
    df = db.query_df(
        "SELECT ticker, composite_score FROM composite_scores WHERE as_of_date = ?", (date,))
    return df.set_index("ticker")["composite_score"].dropna()


@st.cache_data(ttl=DEFAULT_TTL, show_spinner=False)
def _prices(tickers: tuple[str, ...], start: str, end: str) -> pd.DataFrame:
    return load_price_matrix(get_db(), list(tickers), start, end)


def _nearest_prior_date(dates: list[str], target: pd.Timestamp) -> str:
    """The available score date closest to ``target``, restricted to dates
    strictly before the latest one (so "a month ago" never collides with
    "today")."""
    candidates = dates[:-1] if len(dates) > 1 else dates
    ts = pd.to_datetime(candidates)
    idx = int((ts - target).map(lambda d: abs(d.days)).values.argmin())
    return candidates[idx]


def _top_k(scores: pd.Series, k: int, exclude: set[str] = frozenset()) -> list[str]:
    s = scores[~scores.index.isin(exclude)] if exclude else scores
    s = s.sort_index()                      # deterministic secondary key: ticker asc
    s = s.sort_values(ascending=False, kind="stable")
    return list(s.index[:k])


def _rank_pct(scores: pd.Series, ticker: str) -> float | None:
    s = scores.dropna().sort_values(ascending=False, kind="stable")
    if ticker not in s.index:
        return None
    return float(s.index.get_loc(ticker)) / max(len(s) - 1, 1)


# ---------------------------------------------------------------------------
# "Buy today" baskets
# ---------------------------------------------------------------------------
@dataclass
class Basket:
    as_of: str
    weights: dict[str, float]      # ticker -> target weight of total NAV
    note: str = ""                  # e.g. cash-drag / staggered-onboarding caveat


def form_v3_basket(date: str, scores: pd.Series) -> Basket:
    names = _top_k(scores, V3_K)
    w = 1.0 / len(names) if names else 0.0
    return Basket(as_of=date, weights={t: w for t in names})


def form_book4_basket(date: str, scores: pd.Series) -> Basket:
    """Only sleeve 0 forms on a fresh deploy -- sleeves 1 and 2 are staggered
    1 and 2 months later respectively (see module docstring), so a cold
    start only buys 4 names, with 2/3 of target NAV sitting in cash until
    sleeves 1/2 come online."""
    names = _top_k(scores, BOOK4_SIZE)
    weights = {t: BOOK4_SLOT_WEIGHT for t in names}
    cash_pct = 1.0 - sum(weights.values())
    note = (f"Only sleeve 1 of 3 has started -- sleeves 2 and 3 come online "
           f"in ~{BOOK4_STEP_MONTHS} and ~{2 * BOOK4_STEP_MONTHS} month(s), each "
           f"adding 4 more names. {cash_pct:.0%} of target NAV is uncommitted "
           f"cash until then.")
    return Basket(as_of=date, weights=weights, note=note)


@st.cache_data(ttl=DEFAULT_TTL, show_spinner=False)
def form_today() -> dict[str, Basket]:
    dates = _score_dates()
    if not dates:
        return {}
    d = dates[-1]
    scores = _scores(d)
    return {
        "v3_loopeng_cap9_minhold1wk_quartile": form_v3_basket(d, scores),
        "21_loopeng_book4_hold4_evict3": form_book4_basket(d, scores),
    }


# ---------------------------------------------------------------------------
# "Started a month ago" scenario
# ---------------------------------------------------------------------------
@dataclass
class MonthAgoResult:
    formed_on: str
    as_of_today: str
    initial_weights: dict[str, float]
    events: list[dict]            # {ticker, date, reason}
    current_weights: dict[str, float]
    equity: pd.Series             # daily NAV, start = 1.0
    spy_equity: pd.Series
    cagr: float
    beta: float
    alpha: float
    note: str = ""


def _perf_stats(pr: pd.Series, bench: pd.Series) -> dict:
    """Daily-return CAPM stats (ppy=252) -- same beta/alpha definitions as
    research/walkforward/portfolio.py::benchmark_stats, just annualized off
    daily rather than monthly periods since this window is only ~1 month."""
    df = pd.concat([pr.rename("p"), bench.rename("b")], axis=1).dropna()
    if len(df) < 2:
        return {"cagr": float("nan"), "beta": float("nan"), "alpha": float("nan")}
    p, b = df["p"], df["b"]
    var_b = float(b.var(ddof=1))
    beta = float(((p - p.mean()) * (b - b.mean())).sum() / ((b - b.mean()) ** 2).sum()) \
        if var_b > 0 else float("nan")
    alpha = float((p.mean() - beta * b.mean()) * 252) if beta == beta else float("nan")
    n = len(p)
    total_return = float((1.0 + p).prod())
    cagr = total_return ** (252.0 / n) - 1.0 if n > 0 else float("nan")
    return {"cagr": cagr, "beta": beta, "alpha": alpha}


def _v3_month_ago_scenario(formed_on: str, today: str, scores0: pd.Series,
                           scores_today: pd.Series) -> MonthAgoResult:
    names0 = _top_k(scores0, V3_K)
    w0 = 1.0 / len(names0) if names0 else 0.0
    universe = sorted(set(names0) | set(scores_today.index))
    prices = _prices(tuple(universe), formed_on, today)
    trading_days = [d for d in prices.index if formed_on <= d <= today]

    entry_px = {t: prices.at[formed_on, t] for t in names0 if formed_on in prices.index
               and t in prices.columns and pd.notna(prices.at[formed_on, t])}
    names0 = [t for t in names0 if t in entry_px]
    peak = dict(entry_px)
    held = set(names0)
    exit_value = {}   # ticker -> (1 - cost) locked value multiple, frozen from exit day
    events: list[dict] = []
    min_hold_cutoff = pd.to_datetime(formed_on) + pd.Timedelta(days=V3_MIN_HOLD_DAYS)

    nav_path = {}
    alloc = w0
    for day in trading_days:
        for t in list(held):
            px = prices.at[day, t] if t in prices.columns else np.nan
            if pd.isna(px):
                continue
            peak[t] = max(peak[t], px)
            if day == formed_on or pd.to_datetime(day) < min_hold_cutoff:
                continue
            if px / peak[t] - 1.0 <= -V3_TRAIL_PCT:
                mult = (px / entry_px[t]) * (1.0 - V3_COST_BPS / 1e4)
                exit_value[t] = mult
                held.discard(t)
                events.append({"ticker": t, "date": day,
                              "reason": f"-10% trailing stop (peak ${peak[t]:.2f} -> ${px:.2f})"})
        total = 0.0
        for t in names0:
            if t in held:
                px = prices.at[day, t] if t in prices.columns else np.nan
                mult = (px / entry_px[t]) if pd.notna(px) else 1.0
            else:
                mult = exit_value.get(t, 1.0)
            total += alloc * mult
        nav_path[day] = total

    # today's review: worst-quartile rank exit, then refill any open slots
    for t in list(held):
        rp = _rank_pct(scores_today, t)
        if rp is not None and rp > V3_RANK_FLOOR_PCT:
            px = prices.at[today, t] if today in prices.index and t in prices.columns else np.nan
            mult = (px / entry_px[t]) * (1.0 - V3_COST_BPS / 1e4) if pd.notna(px) else 1.0
            exit_value[t] = mult
            held.discard(t)
            events.append({"ticker": t, "date": today,
                          "reason": f"worst-quartile rank exit (rank_pct={rp:.2f})"})
    open_slots = V3_K - len(held)
    current_names = list(held)
    if open_slots > 0:
        fill = _top_k(scores_today, open_slots, exclude=held)
        for t in fill:
            events.append({"ticker": t, "date": today, "reason": "refilled open slot at today's review"})
        current_names += fill

    nav = pd.Series(nav_path).sort_index()
    if not nav.empty:
        nav = nav / nav.iloc[0]
    spy = prices[SPY].reindex(nav.index).ffill() if SPY in prices.columns else pd.Series(dtype=float)
    spy_eq = (spy / spy.iloc[0]) if not spy.empty else spy
    pr = nav.pct_change().dropna()
    spy_r = spy.pct_change().dropna()
    stats = _perf_stats(pr, spy_r)

    cw = 1.0 / len(current_names) if current_names else 0.0
    return MonthAgoResult(
        formed_on=formed_on, as_of_today=today,
        initial_weights={t: w0 for t in names0}, events=events,
        current_weights={t: cw for t in current_names},
        equity=nav, spy_equity=spy_eq, **stats)


def _book4_month_ago_scenario(formed_on: str, today: str, scores0: pd.Series,
                              scores_today: pd.Series) -> MonthAgoResult:
    """Sleeve 0 formed a month ago (buy-and-hold since -- HOLD_MONTHS=4 means
    no eviction review is due yet); sleeve 1 forms today (STEP=1 month after
    inception), adding 4 more names using TODAY's scores. Sleeve 2 doesn't
    start for another month, so 1/3 of target NAV still sits in cash."""
    sleeve0 = _top_k(scores0, BOOK4_SIZE)
    sleeve1 = _top_k(scores_today, BOOK4_SIZE)   # independent pick, may overlap sleeve0

    universe = tuple(sorted(set(sleeve0) | set(sleeve1)))
    prices = _prices(universe, formed_on, today)
    sleeve0 = [t for t in sleeve0 if t in prices.columns and formed_on in prices.index
              and pd.notna(prices.at[formed_on, t])]

    entry_px = prices.loc[formed_on, sleeve0]
    rel = prices[sleeve0].divide(entry_px, axis=1)
    invested_nav = (rel * BOOK4_SLOT_WEIGHT).sum(axis=1)
    cash = 1.0 - BOOK4_SLOT_WEIGHT * len(sleeve0)
    nav = (invested_nav + cash).reindex([d for d in prices.index if formed_on <= d <= today])
    if not nav.empty:
        nav = nav / nav.iloc[0]

    spy = prices[SPY].reindex(nav.index).ffill() if SPY in prices.columns else pd.Series(dtype=float)
    spy_eq = (spy / spy.iloc[0]) if not spy.empty else spy
    pr = nav.pct_change().dropna()
    spy_r = spy.pct_change().dropna()
    stats = _perf_stats(pr, spy_r)

    current = Counter(sleeve0) + Counter(sleeve1)
    current_weights = {t: n * BOOK4_SLOT_WEIGHT for t, n in current.items()}
    events = [{"ticker": t, "date": today, "reason": "sleeve 2 of 3 forms today (STEP=1 month)"}
             for t in sleeve1]
    cash_pct = 1.0 - sum(current_weights.values())

    return MonthAgoResult(
        formed_on=formed_on, as_of_today=today,
        initial_weights={t: BOOK4_SLOT_WEIGHT for t in sleeve0}, events=events,
        current_weights=current_weights, equity=nav, spy_equity=spy_eq,
        note=f"Sleeve 3 of 3 still hasn't started ({cash_pct:.0%} of target NAV in cash).",
        **stats)


@st.cache_data(ttl=DEFAULT_TTL, show_spinner=False)
def month_ago_scenarios() -> dict[str, MonthAgoResult]:
    dates = _score_dates()
    if len(dates) < 2:
        return {}
    today = dates[-1]
    formed_on = _nearest_prior_date(dates, pd.to_datetime(today) - pd.Timedelta(days=30))
    scores0, scores_today = _scores(formed_on), _scores(today)
    return {
        "v3_loopeng_cap9_minhold1wk_quartile":
            _v3_month_ago_scenario(formed_on, today, scores0, scores_today),
        "21_loopeng_book4_hold4_evict3":
            _book4_month_ago_scenario(formed_on, today, scores0, scores_today),
    }
