"""Pairs-trading study on the PIT S&P 500 universe.

Method follows Gatev, Goetzmann & Rouwenhorst (2006) "Pairs Trading: Performance
of a Relative-Value Arbitrage Rule":

* 12-month formation window immediately before each semiannual trading window
  (the same 19 windows 2017-H1 .. 2026-H1 as ``vixtilt.windows``);
* candidate pairs restricted to the same (normalised) GICS sector — the
  literature finds within-industry pairs are the robust subset;
* pairs ranked by the sum of squared deviations (SSD) between normalised
  cumulative-return paths over the formation window; top N traded;
* trading rule: open when |spread| >= 2 x formation-sigma (long the leg below,
  short the leg above), with GGR's one-day wait (signal at close t, execute at
  close t+1); close on spread sign-flip, window end, or data end (delisting);
* explicit costs of ``COST_PER_SIDE`` per leg per side (open + close).

PIT hygiene: pairs are formed only on prices <= formation end; the universe is
``members_as_of(test_start)`` (no future entrants); composite-score gates use
the frozen per-window vixtilt caches and only ever read a rebalance dated
<= the trading day (asserted).

Score-gated variants use the production-faithful composite from the vixtilt
window caches: the gate compares the composite of the leg being bought vs the
leg being shorted at the latest available rebalance on/before the entry day.
"""
from __future__ import annotations

import pickle
import sqlite3
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd

from vixtilt import backtest as bt
from vixtilt.baseline import CACHE_DIR, CACHE_VERSION
from vixtilt.windows import Window, semiannual_windows

COST_PER_SIDE = 0.0010
ENTRY_Z = 2.0
FORMATION_MONTHS = 12
MIN_COVERAGE = 0.95          # of formation trading days
DB_PATH = Path("cache/mahajan.db")

# FMP mixes GICS with its own taxonomy; collapse synonyms so "same sector"
# candidate pools are real.
SECTOR_SYNONYMS = {
    "Healthcare": "Health Care",
    "Financial Services": "Financials",
    "Technology": "Information Technology",
    "Consumer Cyclical": "Consumer Discretionary",
    "Consumer Defensive": "Consumer Staples",
    "Basic Materials": "Materials",
}


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_prices(db_path: Path = DB_PATH, start: str = "2015-06-01") -> pd.DataFrame:
    con = sqlite3.connect(db_path)
    df = pd.read_sql(
        "SELECT date, ticker, adj_close FROM daily_prices WHERE date >= ?",
        con, params=(start,))
    con.close()
    return df.pivot(index="date", columns="ticker", values="adj_close").sort_index()


def load_sectors(db_path: Path = DB_PATH) -> pd.Series:
    con = sqlite3.connect(db_path)
    df = pd.read_sql("SELECT ticker, gics_sector FROM universe", con)
    con.close()
    s = df.set_index("ticker")["gics_sector"].fillna("Unknown")
    return s.replace(SECTOR_SYNONYMS)


def members_as_of(as_of: str, db_path: Path = DB_PATH) -> list[str]:
    con = sqlite3.connect(db_path)
    rows = con.execute(
        "SELECT DISTINCT ticker FROM universe_history "
        "WHERE start_date <= ? AND (end_date IS NULL OR end_date > ?)",
        (as_of, as_of)).fetchall()
    con.close()
    return sorted(r[0] for r in rows)


def load_composites(windows: list[Window], sectors: pd.Series) -> dict[str, pd.Series]:
    """{rebal_date: composite Series} from the frozen vixtilt window caches.

    Each composite is computed as-of its rebalance date, so it is valid for any
    trading day >= that date (carry-forward across window boundaries is fine —
    windows are contiguous)."""
    out: dict[str, pd.Series] = {}
    for w in windows:
        path = CACHE_DIR / f"window_{w.label}_v{CACHE_VERSION}.pkl"
        if not path.exists():
            continue
        wb = pickle.loads(path.read_bytes())
        for d in wb.test_rebals:
            frame = wb.parent_test.get(d)
            if frame is None or frame.empty:
                continue
            out[d] = bt.composite_from_parents(frame, wb.parent_weights, sectors)
    return out


def score_asof(composites: dict[str, pd.Series], day: str) -> pd.Series | None:
    dates = [d for d in composites if d <= day]
    if not dates:
        return None
    d = max(dates)
    assert d <= day, "score look-ahead"
    return composites[d]


# --------------------------------------------------------------------------- #
# Formation
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Pair:
    a: str            # legs, sorted alphabetically
    b: str
    ssd: float
    sigma: float      # formation-period spread std (normalised-price units)
    sector: str
    corr: float = float("nan")   # formation daily-return correlation


def _half_life(spread: pd.Series) -> float:
    """AR(1) mean-reversion half-life of the spread, in trading days."""
    s0, s1 = spread.shift(1).iloc[1:], spread.iloc[1:]
    var = s0.var()
    if not var or np.isnan(var):
        return float("inf")
    b = ((s1 - s0) * (s0 - s0.mean())).mean() / var
    if b >= 0 or b <= -1:
        return float("inf")
    return float(-np.log(2) / np.log(1 + b))


def _crossings(spread: pd.Series) -> int:
    sign = np.sign(spread.to_numpy())
    return int((np.diff(sign[sign != 0]) != 0).sum())


def form_pairs(px: pd.DataFrame, members: list[str], sectors: pd.Series,
               formation_start: str, formation_end: str,
               n_candidates: int = 100, quality: bool = False,
               min_crossings: int = 10,
               half_life_range: tuple[float, float] = (5.0, 40.0),
               min_sigma: float = 0.0, min_corr: float = 0.0) -> list[Pair]:
    """Top-``n_candidates`` same-sector pairs by SSD (ascending, one appearance
    per ticker); caller trades the first ``n_pairs`` (or a filtered subset).

    ``quality=True`` additionally requires the formation spread to be a proven
    mean-reverter: >= ``min_crossings`` zero-crossings and an AR(1) half-life
    inside ``half_life_range`` trading days, with sigma >= ``min_sigma`` so the
    entry band clears trading costs (Do & Faff's minimum-profitability point —
    without it dual-class twins like GOOG/GOOGL top the ranking with bands too
    tight to trade)."""
    fpx = px.loc[(px.index >= formation_start) & (px.index <= formation_end)]
    cols = [t for t in members if t in fpx.columns]
    # the pivot keeps holiday rows where some off-exchange ticker printed;
    # drop rows with no member data so iloc[0]/iloc[-1] are real trading days
    fpx = fpx[cols].dropna(how="all")
    if fpx.empty:
        return []
    ok = fpx.columns[(fpx.notna().mean() >= MIN_COVERAGE)
                     & fpx.iloc[0].notna() & fpx.iloc[-1].notna()]
    fpx = fpx[ok].ffill()
    norm = fpx / fpx.iloc[0]

    cands: list[Pair] = []
    sec_map = sectors.reindex(ok).fillna("Unknown")
    for sec, group in sec_map.groupby(sec_map):
        names = sorted(group.index)
        if len(names) < 2:
            continue
        block = norm[names].to_numpy()
        rets = fpx[names].pct_change(fill_method=None).iloc[1:].fillna(0.0)
        C = np.corrcoef(rets.to_numpy(), rowvar=False)
        for i in range(len(names)):
            diff = block[:, i + 1:] - block[:, [i]]
            ssd = np.nansum(diff ** 2, axis=0)
            sig = np.nanstd(diff, axis=0)
            for j, (s, sd) in enumerate(zip(ssd, sig), start=i + 1):
                cands.append(Pair(names[i], names[j], float(s), float(sd), sec,
                                  float(C[i, j])))
    cands.sort(key=lambda p: (p.ssd, p.a, p.b))
    # one appearance per ticker: caps single-name concentration in a 20-pair book
    used: set[str] = set()
    picked = []
    checked = 0
    for p in cands:
        if p.a in used or p.b in used or p.sigma <= 0:
            continue
        if min_corr and not p.corr >= min_corr:
            continue
        if quality:
            if p.sigma < min_sigma:
                continue
            if checked >= 5000:     # quality checks are O(days) each; cap them
                break
            checked += 1
            spread = norm[p.a] - norm[p.b]
            if _crossings(spread) < min_crossings:
                continue
            hl = _half_life(spread)
            if not (half_life_range[0] <= hl <= half_life_range[1]):
                continue
        picked.append(p)
        used.update((p.a, p.b))
        if len(picked) >= n_candidates:
            break
    return picked


# --------------------------------------------------------------------------- #
# Simulation
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TradeRule:
    """Entry/exit discipline. Defaults reproduce classic GGR."""
    entry_z: float = ENTRY_Z
    confirm: bool = False         # require the spread to have ticked back
                                  # toward the mean (|s_t| < |s_{t-1}|) so we
                                  # never enter a still-widening dislocation
    max_entry_z: float | None = None   # never enter beyond this (broken pair)
    stop_z: float | None = None        # abandon an open trade at this depth
    block_after_stop: bool = False     # stopped pair is dead for the window
    max_days: int | None = None        # time stop for unconverged trades
    min_days_left: int | None = None   # no fresh entries without this much
                                       # window runway to converge in
    exit_z: float = 0.0                # take profit at this depth instead of
                                       # waiting for the full zero-cross
    add_z: float | None = None         # pile on (double the position) if an
                                       # open trade deepens to this level


STRICT = TradeRule(entry_z=2.5, confirm=True, max_entry_z=4.0, stop_z=4.0,
                   block_after_stop=True, max_days=45)

# forensics-derived (2017-21 discovery, both rules replicate in 2022-26):
# entries deeper than 2.75σ lose in both eras; entries with <40 days of window
# left can't converge in time (converged trades win 97%, window-end 28%)
ADJUSTED = TradeRule(max_entry_z=2.75, min_days_left=40)

# round-4 forensics: converged trades touch 0.5σ a week before the zero-cross
# and 27% of window-end losers touch it too — banking there is the most
# era-consistent exit (paired with the composite tilt gate at entry)
TUNED = TradeRule(max_entry_z=2.75, min_days_left=40, exit_z=0.5)


@dataclass
class Trade:
    pair: Pair
    long: str
    short: str
    open_date: str
    close_date: str = ""
    payoff: float = 0.0       # net, per $1 per leg of the INITIAL unit
    days: int = 0
    reason: str = ""          # converged | window_end | data_end | stop
    long_score: float = float("nan")
    short_score: float = float("nan")
    units: float = 1.0        # entry size (entry_filter) x2 after a pile-on
    added: bool = False       # pile-on already executed


def simulate_window(px: pd.DataFrame, pairs: list[Pair], anchor: str,
                    test_start: str, test_end: str,
                    gate=None, rule: TradeRule = TradeRule(),
                    cost: float = COST_PER_SIDE, entry_filter=None,
                    ) -> tuple[pd.DataFrame, list[Trade]]:
    """Daily pair payoffs over one trading window.

    ``anchor`` is the last formation day: prices re-normalised there so the
    trading-period spread is measured in the same units as formation sigma.
    Returns (frame indexed by day with per-pair payoff columns, trades)."""
    tickers = sorted({p.a for p in pairs} | {p.b for p in pairs})
    span = (px.loc[(px.index >= anchor) & (px.index <= test_end), tickers]
            .dropna(how="all").ffill())
    if span.empty or len(span) < 3:
        return pd.DataFrame(), []
    norm = span / span.iloc[0]
    rets = span.pct_change().fillna(0.0)
    days = [d for d in span.index if d >= test_start]

    payoff = pd.DataFrame(0.0, index=days, columns=range(len(pairs)))
    trades: list[Trade] = []

    for k, p in enumerate(pairs):
        spread = norm[p.a] - norm[p.b]
        prev_abs = spread.abs().shift(1)   # previous *span* day (incl. anchor)
        valid_a, valid_b = px[p.a].notna(), px[p.b].notna()
        open_trade: Trade | None = None
        entry_sign = 0
        pending: int = 0            # +1/-1 = open signal sign, 0 = none
        pending_close = False
        pending_add = False
        blocked = False             # after a stop-out: no re-entry until the
                                    # spread returns inside the entry band
        for idx, d in enumerate(days):
            # execute yesterday's signal at today's close (one-day wait)
            if open_trade is None and pending != 0:
                if rule.max_entry_z is not None \
                        and abs(spread[d]) > rule.max_entry_z * p.sigma:
                    pending = 0     # kept widening through the wait day
                    continue
                if gate is not None and not getattr(gate, "allows",
                                                    lambda _d: True)(d):
                    pending = 0     # regime filter says stand down
                    continue
                # pair-aware hook: 0/False blocks, a float sets the unit size
                mult = 1.0 if entry_filter is None else float(entry_filter(p, d))
                if mult <= 0:
                    pending = 0
                    continue
                long, short = (p.a, p.b) if pending < 0 else (p.b, p.a)
                ls = ss = float("nan")
                if gate is not None:
                    sc = gate(d)
                    if sc is not None:
                        ls, ss = sc.get(long, float("nan")), sc.get(short, float("nan"))
                        if not _gate_pass(gate.rule, ls, ss, sc):
                            pending = 0
                            continue
                open_trade = Trade(p, long, short, d, long_score=ls,
                                   short_score=ss, units=mult)
                open_trade.payoff -= 2 * cost * mult
                payoff.at[d, k] -= 2 * cost * mult
                entry_sign = pending
                pending = 0
                continue        # entered at close; P&L starts next day
            if open_trade is not None:
                if pending_add:
                    add_cost = 2 * cost * open_trade.units  # doubling
                    open_trade.units *= 2
                    open_trade.added = True
                    open_trade.payoff -= add_cost
                    payoff.at[d, k] -= add_cost
                    pending_add = False
                u = open_trade.units
                day_pnl = (rets.at[d, open_trade.long]
                           - rets.at[d, open_trade.short]) * u
                open_trade.payoff += day_pnl
                open_trade.days += 1
                payoff.at[d, k] += day_pnl
                dead = not (valid_a.get(d, False) and valid_b.get(d, False))
                last = idx == len(days) - 1
                if pending_close or dead or last:
                    open_trade.close_date = d
                    blocked = open_trade.reason == "stop"
                    dead_pair = blocked and rule.block_after_stop
                    if not open_trade.reason:
                        open_trade.reason = ("converged" if pending_close
                                             else "data_end" if dead else "window_end")
                    open_trade.payoff -= 2 * cost * u
                    payoff.at[d, k] -= 2 * cost * u
                    trades.append(open_trade)
                    open_trade = None
                    pending_close = False
                    if dead_pair:
                        break
                    continue
                if np.sign(spread[d]) != entry_sign \
                        or abs(spread[d]) <= rule.exit_z * p.sigma:
                    pending_close = True
                elif rule.stop_z is not None and abs(spread[d]) >= rule.stop_z * p.sigma:
                    pending_close = True
                    open_trade.reason = "stop"
                elif rule.max_days is not None and open_trade.days >= rule.max_days:
                    pending_close = True
                    open_trade.reason = "time"
                elif rule.add_z is not None and not open_trade.added \
                        and abs(spread[d]) >= rule.add_z * p.sigma:
                    pending_add = True
                continue
            # flat: watch for entry signal (never re-signal on the last day)
            if blocked:
                if abs(spread[d]) < rule.entry_z * p.sigma:
                    blocked = False
                continue
            z = abs(spread[d])
            depth_ok = z >= rule.entry_z * p.sigma
            not_broken = rule.max_entry_z is None or z <= rule.max_entry_z * p.sigma
            pa = prev_abs.get(d, float("nan"))
            confirmed = not rule.confirm or (not np.isnan(pa) and z < pa)
            runway = (rule.min_days_left is None
                      or len(days) - 1 - idx >= rule.min_days_left)
            if idx < len(days) - 1 and depth_ok and not_broken and confirmed \
                    and runway and valid_a.get(d, False) and valid_b.get(d, False):
                pending = int(np.sign(spread[d]))

    return payoff, trades


def _gate_pass(rule: str, long_score: float, short_score: float,
               sc: pd.Series) -> bool:
    """Missing scores never block a trade (noted in the report)."""
    if np.isnan(long_score) or np.isnan(short_score):
        return True
    if rule == "tilt":
        return long_score >= short_score
    if rule == "strict":
        return long_score - short_score >= 10.0
    if rule == "veto":
        lo, hi = sc.quantile(0.20), sc.quantile(0.80)
        return long_score > lo and short_score < hi
    if rule == "levels":            # long a liked name, short a disliked one
        return long_score >= 50.0 and short_score <= 25.0
    if rule == "long50":
        return long_score >= 50.0
    raise ValueError(rule)


class ScoreGate:
    def __init__(self, composites: dict[str, pd.Series], rule: str):
        self.composites, self.rule = composites, rule

    def __call__(self, day: str) -> pd.Series | None:
        return score_asof(self.composites, day)


class RegimeGate(ScoreGate):
    """Tilt gate that also stands down on new entries after a hot streak.

    Forensics (round 6): entries made after the book gained > ``thresh`` over
    the trailing ``lookback`` days are the losing cohort in both eras — the
    mean-reversion regime exhausts. ``daily`` is the strategy's own realized
    daily return series (only past values are consulted; PIT by shift)."""

    def __init__(self, composites: dict[str, pd.Series], daily: pd.Series,
                 lookback: int = 42, thresh: float = 0.01):
        super().__init__(composites, "tilt")
        trail = daily.sort_index().rolling(lookback).sum().shift(1)
        self.allow_s = ~(trail > thresh)

    def allows(self, day: str) -> bool:
        a = self.allow_s.loc[:day]
        return bool(a.iloc[-1]) if len(a) else True


# --------------------------------------------------------------------------- #
# Study driver
# --------------------------------------------------------------------------- #
@dataclass
class VariantResult:
    name: str
    desc: str
    daily: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    trades: list[Trade] = field(default_factory=list)
    pairs_per_window: dict[str, int] = field(default_factory=dict)


def committed_capital_returns(payoff: pd.DataFrame, n_pairs: int) -> pd.Series:
    return payoff.sum(axis=1) / n_pairs


def run_grid(n_pairs: int = 20, db_path: Path = DB_PATH) -> dict[str, VariantResult]:
    """Classic GGR vs strict entry discipline on the PIT S&P 500 universe.

    Strict = quality-filtered pairs (>=10 crossings, half-life 5-40d, sigma
    floor vs costs) traded under ``STRICT`` (2.5σ entry with reversal
    confirmation, 4σ broken-pair guard with window blacklist, 45d time stop).
    Round-1 findings (coint filter and naive stop both hurt; score gates mild
    positive) motivated the cells."""
    windows = semiannual_windows()
    px, sectors = load_prices(db_path), load_sectors(db_path)
    composites = load_composites(windows, sectors)
    # entry depth must be worth >= 3 round-trips of cost (4 legs*sides)
    min_sigma = 3 * 4 * COST_PER_SIDE / STRICT.entry_z

    cells = {
        "classic": (TradeRule(), None, False,
                    "classic GGR (round-1 baseline)"),
        "adjusted": (ADJUSTED, None, False,
                     "classic + forensics rules (≤2.75σ entry, ≥40d runway)"),
        "adjusted_veto": (ADJUSTED, ScoreGate(composites, "veto"), False,
                          "adjusted + composite quintile veto"),
        "tuned": (TUNED, ScoreGate(composites, "tilt"), False,
                  "adjusted + 0.5σ take-profit + composite tilt gate"),
        "tuned_pileon": (replace(TUNED, add_z=3.0), ScoreGate(composites, "tilt"), False,
                         "tuned + double the position if it deepens to 3σ"),
    }
    variants = {k: VariantResult(k, desc) for k, (*_, desc) in cells.items()}

    for w in windows:
        f_end = (pd.Timestamp(w.test_start) - pd.Timedelta(days=1)).date().isoformat()
        f_start = (pd.Timestamp(w.test_start)
                   - pd.DateOffset(months=FORMATION_MONTHS)).date().isoformat()
        assert f_end < w.test_start, "formation overlaps trading"
        members = members_as_of(w.test_start, db_path)
        if not members:
            raise RuntimeError("universe_history empty")
        formed = {
            quality: form_pairs(px, members, sectors, f_start, f_end,
                                n_candidates=n_pairs, quality=quality,
                                min_sigma=min_sigma if quality else 0.0)
            for quality in (False, True)}
        anchor = (px.loc[(px.index >= f_start) & (px.index <= f_end)]
                  .dropna(how="all").index[-1])

        for name, (rule, gate, quality, _) in cells.items():
            prs = formed[quality]
            v = variants[name]
            v.pairs_per_window[w.label] = len(prs)
            if not prs:
                continue
            payoff, trades = simulate_window(px, prs, anchor, w.test_start,
                                             w.test_end, gate=gate, rule=rule)
            if payoff.empty:
                continue
            v.daily = pd.concat([v.daily, committed_capital_returns(payoff, n_pairs)])
            v.trades.extend(trades)
        print(f"  {w.label}: " + ", ".join(
            f"{name}={variants[name].pairs_per_window.get(w.label, 0)}p"
            for name in cells))

    # second pass: hot-streak stand-down cells (need tuned's own daily P&L)
    hot_cells = {
        "tuned_hot": (TUNED, "tuned + stand down after +1%/42d hot streak"),
        "pileon_hot": (replace(TUNED, add_z=3.0),
                       "tuned_pileon + hot-streak stand-down"),
    }
    ref = variants["tuned"].daily.sort_index()
    for name, (rule, desc) in hot_cells.items():
        v = variants[name] = VariantResult(name, desc)
        gate = RegimeGate(composites, ref)
        for w in windows:
            f_end = (pd.Timestamp(w.test_start) - pd.Timedelta(days=1)).date().isoformat()
            f_start = (pd.Timestamp(w.test_start)
                       - pd.DateOffset(months=FORMATION_MONTHS)).date().isoformat()
            prs = form_pairs(px, members_as_of(w.test_start, db_path), sectors,
                             f_start, f_end, n_candidates=n_pairs)
            if not prs:
                continue
            v.pairs_per_window[w.label] = len(prs)
            anchor = (px.loc[(px.index >= f_start) & (px.index <= f_end)]
                      .dropna(how="all").index[-1])
            payoff, trades = simulate_window(px, prs, anchor, w.test_start,
                                             w.test_end, gate=gate, rule=rule)
            if payoff.empty:
                continue
            v.daily = pd.concat([v.daily, committed_capital_returns(payoff, n_pairs)])
            v.trades.extend(trades)
    return variants


# --------------------------------------------------------------------------- #
# Stats
# --------------------------------------------------------------------------- #
def perf_stats(daily: pd.Series) -> dict:
    if daily.empty:
        return {}
    mu, sd = daily.mean(), daily.std()
    ann_ret = mu * 252
    ann_vol = sd * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else float("nan")
    eq = (1 + daily).cumprod()
    dd = (eq / eq.cummax() - 1).min()
    years = len(daily) / 252
    return {"ann_ret": ann_ret, "ann_vol": ann_vol, "sharpe": sharpe,
            "max_dd": dd, "t_stat": sharpe * np.sqrt(years), "years": years,
            "cum": eq.iloc[-1] - 1}


def trade_stats(trades: list[Trade]) -> dict:
    if not trades:
        return {"n_trades": 0}
    pay = np.array([t.payoff for t in trades])
    days = np.array([t.days for t in trades])
    reasons = pd.Series([t.reason for t in trades]).value_counts()
    return {"n_trades": len(trades), "hit_rate": float((pay > 0).mean()),
            "avg_payoff": float(pay.mean()), "med_payoff": float(np.median(pay)),
            "avg_days": float(days.mean()),
            "pct_converged": float(reasons.get("converged", 0) / len(trades))}
