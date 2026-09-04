"""Diagnostic: how often does #13 (trailing stop 10%/12M cap) actually trade?
Answers the user's question (2026-09-02) about whether the trailing-stop rule
accidentally starves entries. Instruments simulate_managed_book's exit/entry
events directly rather than just returning the NAV curve.

Usage: python scripts/diag_trailstop.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from research.strategies.engine import load_data, Book, _mark, _close, _open, _top_k, _months_between
from research.strategies.rules import _trailing_stop_rule

data = load_data()
rebal = set(data.rebal_dates)
matrix = data.matrix
rule = _trailing_stop_rule(0.10, 12)
k = 10

book = Book()
events = []
open_slot_months = []


def review(day):
    is_review = day in rebal
    comp_today = data.comp.get(day) if is_review else None
    for t in list(book.positions):
        p = book.positions[t]
        px = matrix.at[day, t] if t in matrix.columns else float("nan")
        if pd.isna(px):
            continue
        if rule(p, day, px, is_review, comp_today):
            reason = "trailing_stop" if px / p.peak_price - 1.0 <= -0.10 else "time_cap"
            hold_m = _months_between(p.entry_date, day)
            events.append({"ticker": t, "entry": p.entry_date, "exit": day,
                          "reason": reason, "hold_months": round(hold_m, 1),
                          "ret_pct": round(px / p.entry_price - 1.0, 4)})
            _close(book, t, 10.0)
    if is_review:
        open_slots = k - len(book.positions)
        open_slot_months.append({"date": day, "held": len(book.positions), "open": open_slots})
        if open_slots > 0:
            candidates = _top_k(comp_today, k, exclude=set(book.positions))
            for t in candidates[:open_slots]:
                px = matrix.at[day, t] if t in matrix.columns else float("nan")
                _open(book, t, day, px, k, 10.0)


day0 = data.trading_days[0]
review(day0)
prev = day0
for day in data.trading_days[1:]:
    _mark(book, matrix, prev, day)
    review(day)
    prev = day

ev = pd.DataFrame(events)
slots = pd.DataFrame(open_slot_months)

print(f"backtest span: {data.trading_days[0]} -> {data.trading_days[-1]}")
print(f"total exits: {len(ev)}  (trailing_stop: {(ev.reason=='trailing_stop').sum()}, "
      f"time_cap: {(ev.reason=='time_cap').sum()})")
print(f"total entries: {slots['open'].sum() - slots.iloc[0]['open'] + slots.iloc[0]['held']}"
      f"  (sum of slots filled across all {len(slots)} monthly reviews)")
print(f"\npositions held at each review (min/mean/max): "
      f"{slots.held.min()}/{slots.held.mean():.1f}/{slots.held.max()} out of target {k}")
print(f"months with >0 empty slots at review time: {(slots.open>0).sum()} / {len(slots)}")
print(f"\nholding period (months) of closed positions: "
      f"min {ev.hold_months.min():.1f}, mean {ev.hold_months.mean():.1f}, max {ev.hold_months.max():.1f}")
print(f"\nreturn at exit, by reason:")
print(ev.groupby("reason").ret_pct.describe()[["count", "mean", "min", "max"]])
print(f"\nfirst 10 exit events:")
print(ev.head(10).to_string(index=False))
print(f"\nmonths where book ran with empty slots (first 10):")
print(slots[slots.open > 0].head(10).to_string(index=False))
