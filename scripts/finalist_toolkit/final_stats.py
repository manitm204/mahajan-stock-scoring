"""After-tax risk/return of the locked config vs SPY/QQQ, full period + both eras.

Config: top25% / cap5 / cap_match / VIX tilt, 13-month hold, 4 staggered sleeves.
After-tax NAV = mark-to-liquidation each month (pre-tax NAV minus the cap-gains tax
owed if liquidated that month, by lot age: ST<12mo @32%, LT @15%). Smooth, so Sharpe/
Sortino/maxDD are meaningful; terminal = walk-away value. SPY/QQQ: single lot, expense
ratio as annual drag, marked to liquidation at the long-term rate.
"""
from __future__ import annotations
import sys, warnings
from dataclasses import dataclass
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/manit/Desktop/fun_projects/mahajan_hedge_fund")

import numpy as np
import pandas as pd

from run_walkforward import PANEL_START, PRICE_END, _load_panel, get_db
from research.ablation import load_ablation_data
from research.ablation.engine import (AblationConfig, apply_exclusion, select_book,
                                      base_weights, sector_overlay)
from research.forward_returns import realize_delistings
from research.walkforward.portfolio import _sortino
from backtesting.data_loader import SPY

ST, LT = 0.32, 0.15
COST = 0.0010
START = 100_000.0
LT_M = 12
HOLD, SLEEVES = 13, 4
EXPENSE = {"SPY": 0.000945, "QQQ": 0.0020}
ERA = "2021-12-31"
CFG = AblationConfig(name="new", top_pct=0.25, exclusion="none", weighting="cap5",
                     sector="cap_match", vix_tilt=True)


@dataclass
class Lot:
    name: str
    value: float
    basis: float
    open_i: int


def weights_for(data, cfg):
    scores = data.tilt_scores() if cfg.vix_tilt else data.run.pooled_scores
    out, prev = {}, []
    for d in data.rebal_dates:
        if d not in scores:
            continue
        sc = apply_exclusion(scores[d].dropna(), d, data.parent_ranks, cfg.exclusion)
        names = select_book(sc, cfg.top_pct, cfg.exit_pct, prev)
        if not names:
            continue
        w = base_weights(cfg, names, sc, d, data)
        w = sector_overlay(w, cfg, scores[d].dropna().index.tolist(), d, data)
        out[d] = w / w.sum()
        prev = names
    return out


def sleeve_nav(dates, wts, matrix, phase, start):
    """One sleeve, defer-tax. Returns monthly (pretax NAV, after-tax NAV) at dates[1:]."""
    lots: list[Lot] = []
    cash = start
    realized_tax = 0.0
    ptn, atn = [], []

    def holdings():
        s: dict[str, float] = {}
        for l in lots:
            s[l.name] = s.get(l.name, 0.0) + l.value
        return pd.Series(s, dtype=float)

    def sell(name, amount, i):
        nonlocal realized_tax
        mine = sorted([l for l in lots if l.name == name],
                      key=lambda l: l.basis / l.value if l.value > 0 else 0, reverse=True)
        left, proceeds = amount, 0.0
        for l in mine:
            take = min(l.value, left)
            if take <= 0:
                continue
            frac = take / l.value
            gain = take - l.basis * frac
            rate = ST if i - l.open_i < LT_M else LT
            realized_tax += rate * max(gain, 0.0)
            l.basis *= (1 - frac); l.value -= take
            proceeds += take; left -= take
            if left <= 1e-9:
                break
        lots[:] = [l for l in lots if l.value > 1e-9]
        return proceeds * (1 - COST)

    def buy(name, amount, i):
        if amount <= 0:
            return
        net = amount * (1 - COST)
        lots.append(Lot(name, net, net, i))

    for i, d in enumerate(dates[:-1]):
        nxt = dates[i + 1]
        w_t = wts[d]
        h = holdings()
        total = float(h.sum()) + cash
        if i % HOLD == phase % HOLD or i == 0:
            target = w_t * total
            for name in h.index.difference(target.index):
                cash += sell(name, h[name], i)
            for name in target.index:
                diff = float(target[name]) - float(h.get(name, 0.0))
                if diff < -1e-6 * total:
                    cash += sell(name, -diff, i)
            h = holdings()
            buys = {n: float(target[n]) - float(h.get(n, 0.0)) for n in target.index}
            buys = {n: v for n, v in buys.items() if v > 1e-6 * total}
            sc = min(1.0, cash / sum(buys.values())) if buys else 1.0
            for n, v in buys.items():
                buy(n, v * sc, i)
            cash -= sum(v * sc for v in buys.values())

        ret = matrix.loc[nxt] / matrix.loc[d] - 1.0
        for l in lots:
            r = ret.get(l.name)
            if pd.isna(r):
                rate = ST if i - l.open_i < LT_M else LT
                realized_tax += rate * max(l.value - l.basis, 0.0)
                cash += l.value; l.value = 0.0
            else:
                l.value *= (1 + float(r))
        lots[:] = [l for l in lots if l.value > 1e-9]

        pretax = float(holdings().sum()) + cash
        unreal = sum((ST if i - l.open_i < LT_M else LT) * max(l.value - l.basis, 0.0)
                     for l in lots)
        ptn.append(pretax); atn.append(pretax - realized_tax - unreal)
    return np.array(ptn), np.array(atn)


def config_nav(dates, wts, matrix):
    off = HOLD // SLEEVES
    per = START / SLEEVES
    pt = np.zeros(len(dates) - 1); at = np.zeros(len(dates) - 1)
    for k in range(SLEEVES):
        p, a = sleeve_nav(dates, wts, matrix, k * off, per)
        pt += p; at += a
    return np.concatenate([[START], at])          # after-tax NAV at dates[0:]


def bench_nav(matrix, dates, tkr):
    p = matrix[tkr].reindex(dates).values.astype(float)
    er = (1 - EXPENSE[tkr]) ** (1 / 12)
    pretax = START * (p / p[0]) * er ** np.arange(len(p))
    return pretax - LT * np.maximum(pretax - START, 0.0)


def stats(nav):
    nav = np.asarray(nav, float)
    yrs = (len(nav) - 1) / 12.0
    r = np.diff(nav) / nav[:-1]
    cagr = (nav[-1] / nav[0]) ** (1 / yrs) - 1
    sharpe = r.mean() / r.std(ddof=1) * np.sqrt(12)
    sortino = _sortino(pd.Series(r), 12.0)
    dd = float((nav / np.maximum.accumulate(nav) - 1).min())
    return dict(cagr=cagr, sharpe=sharpe, sortino=sortino, mdd=dd,
                calmar=cagr / abs(dd) if dd else float("nan"))


def main():
    panel = _load_panel(False)
    with get_db() as db:
        data = load_ablation_data(panel, db, PANEL_START, PRICE_END, splits="rolling5y")
    matrix = realize_delistings(data.matrix)
    wts = weights_for(data, CFG)
    dates = [d for d in data.rebal_dates if d in wts and d in matrix.index]

    navs = {"New portfolio (25%/cap5)": config_nav(dates, wts, matrix),
            "SPY": bench_nav(matrix, dates, SPY),
            "QQQ": bench_nav(matrix, dates, "QQQ")}
    ds = np.array(dates)

    def window(nav, lo, hi):
        m = (ds >= lo) & (ds <= hi)
        j = np.where(m)[0]
        s = max(j[0] - 1, 0)
        return nav[s:j[-1] + 1]

    spans = [("FULL 2017-2026", dates[0], dates[-1]),
             ("ERA 1 2017-2021", dates[0], ERA),
             ("ERA 2 2022-2026", "2022-01-01", dates[-1])]
    print(f"\nAfter-tax (mark-to-liquidation, ST {ST:.0%}/LT {LT:.0%}), "
          f"{HOLD}mo hold x{SLEEVES} sleeves, $ {START:,.0f} start.\n")
    for label, lo, hi in spans:
        print(f"=== {label} ===")
        print(f"{'':26}{'CAGR':>8}{'Sharpe':>8}{'Sortino':>9}{'Calmar':>8}{'maxDD':>9}")
        for name, nav in navs.items():
            m = stats(window(nav, lo, hi))
            print(f"{name:26}{m['cagr']*100:7.1f}%{m['sharpe']:8.2f}{m['sortino']:9.2f}"
                  f"{m['calmar']:8.2f}{m['mdd']*100:8.1f}%")
        print()
    fv = navs["New portfolio (25%/cap5)"][-1]
    print(f"New portfolio after-tax walk-away: ${fv:,.0f}  "
          f"(SPY ${navs['SPY'][-1]:,.0f}, QQQ ${navs['QQQ'][-1]:,.0f})")


if __name__ == "__main__":
    main()
