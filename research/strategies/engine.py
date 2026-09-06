"""Position-level backtest engine for the 20 top-10 holding/exit strategies
(user request 2026-09-02): given the SAME production EQEFF composite score
each month, the only thing that varies across strategies is how long a name
is held and what triggers an exit.

Composite scores are the cached fully-PIT EQEFF derivation from
scripts/full_pit_backtest_10configs.py (78 monthly dates, 2020-01->2026-06,
sub-factor selection + parent weights re-derived every 6 months from a
trailing 5y window that has already fully realized -- see that script's
docstring for the walk-forward protocol). Nothing here re-derives weights;
this module only decides WHEN to buy/sell the names the production formula
already ranked.

Two families of strategy, one engine each:

  simulate_calendar_sleeves -- fixed-horizon holds, optionally split into
    staggered sleeves (a new sleeve started every `step` months, each held
    for `hold_months` untouched). Strategies 1-8, 20.

  simulate_managed_book -- a single book of K names where each POSITION has
    its own exit rule (price move, trailing stop, score deterioration, rank
    hysteresis, or a value-cheapness PT-gap proxy -- see rules.py), checked
    daily for price-based rules and monthly (review-date only) for rank/score
    rules. New entries only happen on monthly review dates, since composite
    scores are only computed monthly. Strategies 9-19.

Position sizing: each new position is allocated 1/K of TOTAL portfolio NAV at
the moment the slot is filled (not re-normalized daily), so unfilled slots
between reviews sit in zero-yield cash -- a deliberately conservative choice
(realistic: you don't get a new PIT-ranked replacement until the next monthly
review). 10bps per side on every entry and exit, matching the repo's
cost convention (research/ablation/engine.py).

Known limitation carried over from every score-ranked backtest in this repo:
the composite is a PER-SECTOR percentile, so many names tie at 100.0 in a
given month (11+ ties observed on some dates) -- "top 10" ties are broken
deterministically (score desc, ticker asc), not economically. See prior
memory note on tie-determinism in small-effect studies.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
COMP_PKL = REPO / "output" / "crowding" / "weight_config_study" / "comp_10configs.pkl"
PANEL_PKL = REPO / "cache" / "subfactor_expansion" / "cand_panel_2015-06-30_2026-06-30_monthly_v2.pkl"
METHOD = "EQEFF"
COST_BPS = 10.0
K_DEFAULT = 10


# --------------------------------------------------------------------------- #
# Data bundle
# --------------------------------------------------------------------------- #
@dataclass
class StratData:
    comp: dict            # {date: Series(ticker -> 0-100 composite score)}
    value_pct: dict        # {date: Series(ticker -> 0-100 value-parent score)}
    matrix: pd.DataFrame   # daily px (delistings realized, ffilled), incl SPY/QQQ
    rebal_dates: list       # the 78 monthly review dates, sorted
    trading_days: list      # matrix.index restricted to [rebal_dates[0], rebal_dates[-1]]
    sector: pd.Series = None   # ticker -> GICS sector, for max_per_sector entry caps


def load_data() -> StratData:
    from backtesting import data_loader as dl
    from data.db import get_db
    from research.forward_returns import realize_delistings
    from research.subfactor_expansion.panel import load_cached_panel
    from factors.parent_selection_v4 import SELECTED_SUBS
    from scripts.crowding_diagnostics import parent_score

    with COMP_PKL.open("rb") as fh:
        d = pickle.load(fh)
    comp = d["comp"][METHOD]
    universe = d["universe"]
    rebal_dates = sorted(comp)

    db = get_db()
    matrix = realize_delistings(
        dl.load_price_matrix(db, universe, "2019-11-01", "2026-07-31"))
    trading_days = [d for d in matrix.index if rebal_dates[0] <= d <= rebal_dates[-1]]
    sector = dl.global_sectors(db)

    sub_panel = load_cached_panel(PANEL_PKL)
    value_pct = {d: parent_score(sub_panel.scores[d], SELECTED_SUBS["value"])
                 for d in rebal_dates if d in sub_panel.scores}

    return StratData(comp=comp, value_pct=value_pct, matrix=matrix,
                     rebal_dates=rebal_dates, trading_days=trading_days, sector=sector)


def _top_k(scores: pd.Series, k: int, exclude: set[str] | None = None) -> list[str]:
    s = scores.dropna()
    if exclude:
        s = s[~s.index.isin(exclude)]
    s = s.sort_index()             # deterministic secondary key: ticker asc
    s = s.sort_values(ascending=False, kind="stable")
    return list(s.index[:k])


def _rank_pct(scores: pd.Series, ticker: str) -> float | None:
    """0 = best-ranked name that date, 1 = worst. None if not scored that date."""
    s = scores.dropna().sort_values(ascending=False, kind="stable")
    if ticker not in s.index:
        return None
    return float(s.index.get_loc(ticker)) / max(len(s) - 1, 1)


def _months_between(d0: str, d1: str) -> float:
    t0, t1 = pd.Timestamp(d0), pd.Timestamp(d1)
    return (t1.year - t0.year) * 12 + (t1.month - t0.month) + (t1.day - t0.day) / 30.44


# --------------------------------------------------------------------------- #
# Book bookkeeping shared by both engines
# --------------------------------------------------------------------------- #
@dataclass
class Position:
    ticker: str
    entry_date: str
    entry_price: float
    value: float               # current dollar value
    peak_price: float
    entry_score: float = np.nan
    entry_value_pct: float = np.nan
    target_exit_date: str | None = None   # for calendar holds


@dataclass
class Book:
    nav: float = 1.0
    cash: float = 1.0
    positions: dict = field(default_factory=dict)   # ticker -> Position

    def total(self) -> float:
        return self.cash + sum(p.value for p in self.positions.values())


def _mark(book: Book, matrix: pd.DataFrame, prev_day: str, day: str) -> None:
    for p in book.positions.values():
        px_prev, px = matrix.at[prev_day, p.ticker], matrix.at[day, p.ticker]
        if pd.notna(px_prev) and pd.notna(px) and px_prev > 0:
            p.value *= px / px_prev
        if pd.notna(px):
            p.peak_price = max(p.peak_price, px)


def _close(book: Book, ticker: str, cost_bps: float) -> None:
    p = book.positions.pop(ticker)
    book.cash += p.value * (1.0 - cost_bps / 1e4)


def _open(book: Book, ticker: str, day: str, price: float, k_slots: int,
         cost_bps: float, score: float = np.nan, value_pct: float = np.nan,
         target_exit_date: str | None = None) -> None:
    alloc = book.total() / k_slots
    alloc = min(alloc, book.cash)
    if alloc <= 0 or pd.isna(price):
        return
    book.cash -= alloc
    book.positions[ticker] = Position(
        ticker=ticker, entry_date=day, entry_price=price,
        value=alloc * (1.0 - cost_bps / 1e4), peak_price=price,
        entry_score=score, entry_value_pct=value_pct, target_exit_date=target_exit_date)


# --------------------------------------------------------------------------- #
# Engine 1: fixed-horizon calendar holds, optionally staggered into sleeves
# --------------------------------------------------------------------------- #
def _exec_date(trading_days: list, trading_days_pos: dict, day: str, lag: int) -> str:
    """The trading day ``lag`` sessions after ``day`` (clamped at the series
    end). ``lag=0`` reproduces the old same-close behavior.

    Also snaps ``day`` forward to the next real trading day first: a few
    rebal dates are calendar month-ends that fall on a market holiday
    (e.g. 2021-05-31 is Memorial Day, 2024-03-29 is Good Friday) and are
    therefore never a key in ``trading_days_pos`` -- previously that silently
    dropped the sleeve's reform for that cycle instead of executing on the
    next open session."""
    if day not in trading_days_pos:
        import bisect
        i0 = bisect.bisect_left(trading_days, day)
        day = trading_days[min(i0, len(trading_days) - 1)]
    i = min(trading_days_pos[day] + lag, len(trading_days) - 1)
    return trading_days[i]


def simulate_calendar_sleeves(data: StratData, k: int = K_DEFAULT,
                              hold_months: int | None = 3, sleeve_count: int = 1,
                              cost_bps: float = COST_BPS,
                              exec_lag_days: int = 0) -> pd.Series:
    """hold_months=None => buy once, hold forever (strategy #20, sleeve_count
    forced to 1). Otherwise sleeve j (0..sleeve_count-1) forms at rebal-date
    index j*step, then reforms every hold_months, where step =
    hold_months // sleeve_count each sleeve gets 1/sleeve_count of NAV.

    ``exec_lag_days`` fixes the same-close bug: with the default of 0 a
    sleeve trades at the SAME close used to rank/select that day's names
    (unrealistic -- you cannot transact at a price you only observe once the
    close prints). Set ``exec_lag_days=1`` for T+1 execution: the signal is
    read off ``data.comp[rebal_date]`` (still the PIT score as of the rebal
    date) but the trade -- both the close of the outgoing sleeve and the
    open of the incoming one, so there is never a mixed-basis rebalance --
    executes at the price ``exec_lag_days`` trading sessions later."""
    rebal = data.rebal_dates
    matrix = data.matrix
    nav = pd.Series(1.0, index=data.trading_days)
    if hold_months is None:
        sleeve_count = 1
    step = 1 if hold_months is None else max(hold_months // sleeve_count, 1)
    trading_days_pos = {d: i for i, d in enumerate(data.trading_days)}

    # formation schedule per sleeve: exec (traded) date -> signal (ranked) date
    sleeve_forms = []
    exec_to_signal: dict = {}
    for j in range(sleeve_count):
        idx = list(range(j * step, len(rebal), hold_months)) if hold_months else [0]
        signal_dates = [rebal[i] for i in idx]
        exec_dates = [_exec_date(data.trading_days, trading_days_pos, d, exec_lag_days)
                     for d in signal_dates]
        exec_to_signal.update(dict(zip(exec_dates, signal_dates)))
        sleeve_forms.append(exec_dates)

    books = [Book(nav=1.0 / sleeve_count, cash=1.0 / sleeve_count) for _ in range(sleeve_count)]
    day0 = data.trading_days[0]
    total_nav = {}

    def _form(day):
        for j, b in enumerate(books):
            if day in sleeve_forms[j]:
                signal_day = exec_to_signal[day]
                for t in list(b.positions):
                    _close(b, t, cost_bps)
                names = _top_k(data.comp[signal_day], k)
                for t in names:
                    px = matrix.at[day, t] if t in matrix.columns else np.nan
                    _open(b, t, day, px, k, cost_bps)

    _form(day0)
    total_nav[day0] = sum(b.total() for b in books)
    prev_day = day0
    for day in data.trading_days[1:]:
        for b in books:
            _mark(b, matrix, prev_day, day)
        _form(day)
        total_nav[day] = sum(b.total() for b in books)
        prev_day = day

    return pd.Series(total_nav).sort_index()


# --------------------------------------------------------------------------- #
# Engine 2: single managed book, per-position exit rules
# --------------------------------------------------------------------------- #
def simulate_managed_book(data: StratData, exit_rule, k: int = K_DEFAULT,
                          rank_entry_k: int | None = None,
                          cost_bps: float = COST_BPS,
                          max_per_sector: int | None = None,
                          sector_map: pd.Series | None = None,
                          min_value_pct: float | None = None) -> pd.Series:
    """`exit_rule(pos, today, price, is_review, comp_today) -> bool`.
    New entries only happen on review (monthly) dates, taken from the top
    `rank_entry_k or k` names that date, excluding names already held.

    `max_per_sector` (with `sector_map`, e.g. `data.sector`) optionally caps
    how many held positions may share one GICS sector -- candidates that
    would breach the cap are skipped in favor of the next-best name, so the
    effective entry pool widens automatically when the cap binds.

    `min_value_pct` optionally requires a name's Value-parent percentile
    (data.value_pct) to clear a floor at entry -- a quality/cheapness gate on
    top of the pure composite rank, skipping candidates that are expensive
    on Value even if they rank well overall."""
    rebal = set(data.rebal_dates)
    matrix = data.matrix
    entry_k = rank_entry_k or k
    book = Book()

    def _review(day):
        is_review = day in rebal
        comp_today = data.comp.get(day) if is_review else None

        for t in list(book.positions):
            p = book.positions[t]
            px = matrix.at[day, t] if t in matrix.columns else np.nan
            if pd.isna(px):
                continue
            if exit_rule(p, day, px, is_review, comp_today):
                _close(book, t, cost_bps)

        if is_review:
            open_slots = k - len(book.positions)
            if open_slots > 0:
                widen = max_per_sector is not None or min_value_pct is not None
                pool_size = entry_k if not widen else max(entry_k, k * 4)
                candidates = _top_k(comp_today, pool_size, exclude=set(book.positions))
                vpct = data.value_pct.get(day)
                sector_counts: dict = {}
                if max_per_sector is not None and sector_map is not None:
                    for t in book.positions:
                        sec = sector_map.get(t, "Unknown")
                        sector_counts[sec] = sector_counts.get(sec, 0) + 1
                filled = 0
                for t in candidates:
                    if filled >= open_slots:
                        break
                    if max_per_sector is not None and sector_map is not None:
                        sec = sector_map.get(t, "Unknown")
                        if sector_counts.get(sec, 0) >= max_per_sector:
                            continue
                    if min_value_pct is not None and vpct is not None:
                        v = vpct.get(t, np.nan)
                        if pd.isna(v) or v < min_value_pct:
                            continue
                    px = matrix.at[day, t] if t in matrix.columns else np.nan
                    sc = float(comp_today.get(t, np.nan))
                    vp = float(vpct.get(t, np.nan)) if vpct is not None else np.nan
                    before = len(book.positions)
                    _open(book, t, day, px, k, cost_bps, score=sc, value_pct=vp)
                    if len(book.positions) > before:
                        filled += 1
                        if max_per_sector is not None and sector_map is not None:
                            sector_counts[sec] = sector_counts.get(sec, 0) + 1

    day0 = data.trading_days[0]
    nav = {}
    _review(day0)
    nav[day0] = book.total()
    prev_day = day0
    for day in data.trading_days[1:]:
        _mark(book, matrix, prev_day, day)
        _review(day)
        nav[day] = book.total()
        prev_day = day
    return pd.Series(nav).sort_index()
