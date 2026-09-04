"""Generalized pair simulator (z-based), extending pairtrading.study semantics.

Same discipline as the validated GGR engine: signal at close t → execute at
close t+1; explicit per-side costs on each leg at open and close; forced exit
at window end or on missing data (delisting); optional score gate and entry
filter. New here: arbitrary z-series (any spread definition), leg-weight
modes (dollar / hedge-ratio / vol-neutral / long-only / short-only), a daily
short-borrow debit, and an open-position mask for invested-capital accounting.

Payoff convention: per 1.0 "unit"; a unit is $1 gross per leg in dollar mode
(gross $2), weight-scaled in other modes with gross kept at $2 ($1 long-only).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .signals import BUILDERS, PairSpec

COST_PER_SIDE = 0.0010
BORROW_ANNUAL = 0.0025          # general-collateral baseline; stress 0.01


@dataclass(frozen=True)
class RVRule:
    entry_z: float = 2.0
    exit_z: float = 0.5           # take-profit depth (0 = wait for zero-cross)
    max_entry_z: float | None = 2.75
    stop_z: float | None = None
    max_days: int | None = None
    min_days_left: int | None = 40
    confirm: bool = False         # require |z| shrinking vs prior day at signal
    persist: int = 1              # signal must hold this many consecutive days
    mode: str = "dollar"          # dollar | hedge | vol | long_only | short_only
    staged: tuple[float, ...] = ()  # e.g. (1.5, 2.0, 2.5): add 1/n at each level
    move_pct: float | None = None  # same-day dislocation trigger: only signal
                                   # if the would-be long fell ≥ this today OR
                                   # the would-be short rose ≥ this today
    half_exit_z: float | None = None  # scale out half the position here
    # for mode="score_legs": trade the long leg only if its composite ≥
    # leg_long_min, the short leg only if its composite ≤ leg_short_max;
    # both qualify → classic pair; neither → skip (missing scores → classic)
    leg_long_min: float = 70.0
    leg_short_max: float = 30.0


@dataclass
class RVTrade:
    spec: PairSpec
    long: str
    short: str
    open_date: str
    close_date: str = ""
    payoff: float = 0.0
    days: int = 0
    reason: str = ""
    entry_z: float = float("nan")
    units: float = 1.0
    long_score: float = float("nan")
    short_score: float = float("nan")


def _leg_weights(spec: PairSpec, rule: RVRule, vola: float, volb: float,
                 a_is_long: bool, long_score: float = float("nan"),
                 short_score: float = float("nan")) -> tuple[float, float]:
    """(w_long, w_short) dollar weights per unit; gross 2.0 for two-leg modes.
    Returns (0, 0) to skip the trade entirely."""
    if rule.mode == "long_only":
        return 1.0, 0.0
    if rule.mode == "short_only":
        return 0.0, 1.0
    if rule.mode == "score_legs":
        if np.isnan(long_score) and np.isnan(short_score):
            return 1.0, 1.0                      # missing scores → classic
        wl = 1.0 if long_score >= rule.leg_long_min else 0.0
        ws = 1.0 if short_score <= rule.leg_short_max else 0.0
        return wl, ws
    if rule.mode == "vol_leg":                   # trade only the mover
        vol_long, vol_short = (vola, volb) if a_is_long else (volb, vola)
        if not (vol_long > 0 and vol_short > 0):
            return 1.0, 1.0
        return (1.0, 0.0) if vol_long >= vol_short else (0.0, 1.0)
    if rule.mode == "hedge":
        b = min(max(abs(spec.beta), 0.25), 4.0)      # sanity clamp
        wa, wb = 2.0 / (1.0 + b), 2.0 * b / (1.0 + b)
        return (wa, wb) if a_is_long else (wb, wa)
    if rule.mode == "vol":
        if not (vola > 0 and volb > 0):
            return 1.0, 1.0
        wa, wb = 2.0 * volb / (vola + volb), 2.0 * vola / (vola + volb)
        return (wa, wb) if a_is_long else (wb, wa)
    return 1.0, 1.0                                   # dollar-neutral


def simulate_rv_window(tr: pd.DataFrame, specs: list[PairSpec], signal: str,
                       anchor: str, test_start: str, test_end: str,
                       rule: RVRule = RVRule(), gate=None,
                       cost: float = COST_PER_SIDE,
                       borrow: float = BORROW_ANNUAL,
                       signal_kwargs: dict | None = None,
                       ) -> tuple[pd.DataFrame, pd.DataFrame, list[RVTrade]]:
    """One trading window. Returns (payoff, open_units, trades); payoff and
    open_units are day × pair-index frames in per-unit terms."""
    build = BUILDERS[signal]
    tickers = sorted({s.a for s in specs} | {s.b for s in specs})
    # include pre-anchor history so rolling signals are warm from day one
    lead = (pd.Timestamp(anchor) - pd.DateOffset(months=6)).date().isoformat()
    span = (tr.loc[(tr.index >= lead) & (tr.index <= test_end), tickers]
            .dropna(how="all").ffill())
    if span.empty or anchor not in span.index:
        return pd.DataFrame(), pd.DataFrame(), []
    rets = span.pct_change().fillna(0.0)
    vol60 = rets.rolling(60).std() * np.sqrt(252)
    days = [d for d in span.index if test_start <= d <= test_end]
    if len(days) < 3:
        return pd.DataFrame(), pd.DataFrame(), []

    payoff = pd.DataFrame(0.0, index=days, columns=range(len(specs)))
    open_units = pd.DataFrame(0.0, index=days, columns=range(len(specs)))
    trades: list[RVTrade] = []
    raw = {t: tr[t] for t in tickers}      # un-ffilled: delisting detection

    for k, spec in enumerate(specs):
        z = build(span[spec.a], span[spec.b], anchor, spec,
                  **(signal_kwargs or {}))
        open_trade: RVTrade | None = None
        wl = ws = 0.0
        entry_sign = 0
        pending = 0
        run = 0                      # consecutive days signal has held
        pending_close = False
        pending_half = False
        scaled = False               # half-exit already taken
        stage_next = 0               # staged-entry level already reached
        for idx, d in enumerate(days):
            zd = z.get(d, np.nan)
            valid = (not pd.isna(raw[spec.a].get(d, np.nan))
                     and not pd.isna(raw[spec.b].get(d, np.nan)))
            # ---- execute yesterday's entry signal at today's close ----
            if open_trade is None and pending != 0:
                bad = (np.isnan(zd)
                       or (rule.max_entry_z is not None
                           and abs(zd) > rule.max_entry_z)
                       or abs(zd) < rule.entry_z * 0.5)   # evaporated overnight
                if bad or (gate is not None and not getattr(
                        gate, "allows", lambda _d: True)(d)):
                    pending = 0
                    continue
                a_long = pending < 0        # z<0 → A cheap → long A
                long, short = (spec.a, spec.b) if a_long else (spec.b, spec.a)
                ls = ss = float("nan")
                if gate is not None:
                    sc = gate(d)
                    if sc is not None:
                        ls = sc.get(long, float("nan"))
                        ss = sc.get(short, float("nan"))
                        if not gate.passes(ls, ss, sc):
                            pending = 0
                            continue
                wl, ws = _leg_weights(spec, rule, vol60[spec.a].get(d, np.nan),
                                      vol60[spec.b].get(d, np.nan), a_long,
                                      long_score=ls, short_score=ss)
                if wl + ws <= 0:                 # no leg qualifies
                    pending = 0
                    continue
                units = (1.0 / len(rule.staged)) if rule.staged else 1.0
                open_trade = RVTrade(spec, long, short, d, entry_z=zd,
                                     units=units, long_score=ls, short_score=ss)
                stage_next = 1
                fee = (wl + ws) * cost * units
                open_trade.payoff -= fee
                payoff.at[d, k] -= fee
                entry_sign = pending
                pending = 0
                continue
            # ---- manage open trade ----
            if open_trade is not None:
                if pending_half:                 # execute half-exit at close
                    half = open_trade.units / 2.0
                    fee = (wl + ws) * cost * half
                    open_trade.units -= half
                    open_trade.payoff -= fee
                    payoff.at[d, k] -= fee
                    pending_half = False
                    scaled = True
                u = open_trade.units
                pnl = (wl * rets.at[d, open_trade.long]
                       - ws * rets.at[d, open_trade.short]
                       - ws * borrow / 252.0) * u
                open_trade.payoff += pnl
                open_trade.days += 1
                payoff.at[d, k] += pnl
                open_units.at[d, k] = u * (wl + ws) / 2.0
                last = idx == len(days) - 1
                if pending_close or not valid or last:
                    open_trade.close_date = d
                    if not open_trade.reason:
                        open_trade.reason = ("converged" if pending_close
                                             else "window_end" if last else "data_end")
                    fee = (wl + ws) * cost * u
                    open_trade.payoff -= fee
                    payoff.at[d, k] -= fee
                    trades.append(open_trade)
                    open_trade = None
                    pending_close = False
                    pending_half = False
                    scaled = False
                    continue
                if np.isnan(zd):
                    continue
                if np.sign(zd) != entry_sign or abs(zd) <= rule.exit_z:
                    pending_close = True
                elif (rule.half_exit_z is not None and not scaled
                      and abs(zd) <= rule.half_exit_z):
                    pending_half = True
                elif rule.stop_z is not None and abs(zd) >= rule.stop_z:
                    pending_close = True
                    open_trade.reason = "stop"
                elif rule.max_days is not None and open_trade.days >= rule.max_days:
                    pending_close = True
                    open_trade.reason = "time"
                elif (rule.staged and stage_next < len(rule.staged)
                      and abs(zd) >= rule.staged[stage_next]):
                    add = 1.0 / len(rule.staged)
                    fee = (wl + ws) * cost * add
                    open_trade.units += add
                    open_trade.payoff -= fee
                    payoff.at[d, k] -= fee
                    stage_next += 1
                continue
            # ---- flat: watch for entry ----
            if np.isnan(zd) or not valid or idx >= len(days) - 1:
                run = 0
                continue
            first_level = rule.staged[0] if rule.staged else rule.entry_z
            depth = abs(zd) >= first_level
            not_broken = rule.max_entry_z is None or abs(zd) <= rule.max_entry_z
            zprev = z.get(days[idx - 1], np.nan) if idx else np.nan
            confirmed = (not rule.confirm
                         or (not np.isnan(zprev) and abs(zd) < abs(zprev)))
            runway = (rule.min_days_left is None
                      or len(days) - 1 - idx >= rule.min_days_left)
            run = run + 1 if depth else 0
            moved = True
            if rule.move_pct is not None:
                # z>0 → A rich → would short A, long B (and vice versa)
                lng, sht = (spec.b, spec.a) if zd > 0 else (spec.a, spec.b)
                moved = (rets.at[d, lng] <= -rule.move_pct
                         or rets.at[d, sht] >= rule.move_pct)
            if depth and not_broken and confirmed and runway and moved \
                    and run >= rule.persist:
                pending = int(np.sign(zd))
    return payoff, open_units, trades
