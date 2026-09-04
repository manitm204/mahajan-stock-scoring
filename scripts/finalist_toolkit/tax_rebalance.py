"""Rebalance-frequency x capital-gains-tax study on the finalist book.

Finalist construction (top10% / cap5 / cap_match / VIX tilt) is held fixed; we vary
how often we reform to target (monthly/quarterly/semiannual/annual), drifting in
between. Lot-level HIFO accounting, ST/LT split at 12 months, losses net within year
and carry forward, year-end tax paid from the account, final full liquidation taxes
the remaining unrealized gains by lot age. 10 bps/side trading cost.

Benchmarks: SPY / QQQ buy-and-hold, expense ratio applied as an annual drag, single
lot -> all long-term at terminal liquidation.

Rates: ST 32% / LT 15% (federal). Approximations (stated): adj_close = total return, so
dividends are deferred rather than taxed annually (favorable to the buy-and-hold
benchmarks); wash sales ignored; year-end tax deducted as cash without forcing sales.
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
from backtesting.data_loader import SPY

ST_RATE, LT_RATE = 0.32, 0.15
COST = 0.0010
START = 100_000.0
LT_MONTHS = 12
EXPENSE = {"SPY": 0.000945, "QQQ": 0.0020}
HOLD_M = 13           # 13-month reform -> every gain clears the >1yr long-term line
SLEEVES = 4
TOPS = [0.10, 0.25, 0.50, 0.75, 1.00]
WEIGHTINGS = ["cap5", "cap", "ew", "ewcap", "rank_lin", "inv_vol"]
SECTORS = ["cap_match", "none"]
FINALIST = (0.10, "cap5", "cap_match")   # current pick, flagged in output


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


def simulate(dates, weights, matrix, every, st_rate, lt_rate, phase=0, start=START):
    lots: list[Lot] = []
    cash = start
    realized = {"st": 0.0, "lt": 0.0}
    carry = 0.0
    tax_st = tax_lt = traded = 0.0
    gain_st = gain_lt = 0.0
    values = []

    def holdings():
        s: dict[str, float] = {}
        for l in lots:
            s[l.name] = s.get(l.name, 0.0) + l.value
        return pd.Series(s, dtype=float)

    def sell(name, amount, i):
        nonlocal traded
        mine = sorted([l for l in lots if l.name == name],
                      key=lambda l: l.basis / l.value if l.value > 0 else 0, reverse=True)
        left, proceeds = amount, 0.0
        for l in mine:
            take = min(l.value, left)
            if take <= 0:
                continue
            frac = take / l.value
            gain = take - l.basis * frac
            realized["st" if i - l.open_i < LT_MONTHS else "lt"] += gain
            l.basis *= (1 - frac); l.value -= take
            proceeds += take; left -= take
            if left <= 1e-9:
                break
        lots[:] = [l for l in lots if l.value > 1e-9]
        traded += proceeds
        return proceeds * (1 - COST)

    def buy(name, amount, i):
        nonlocal traded
        if amount <= 0:
            return
        traded += amount
        net = amount * (1 - COST)
        lots.append(Lot(name, net, net, i))

    for i, d in enumerate(dates[:-1]):
        nxt = dates[i + 1]
        w_t = weights[d]
        h = holdings()
        total = float(h.sum()) + cash
        if i % every == phase % every or i == 0:
            target = w_t * total
            for name in h.index.difference(target.index):
                cash += sell(name, h[name], i)
            for name in target.index:
                cur = float(h.get(name, 0.0))
                diff = float(target[name]) - cur
                if diff < -1e-6 * total:
                    cash += sell(name, -diff, i)
            h = holdings()
            buys = {n: float(target[n]) - float(h.get(n, 0.0)) for n in target.index}
            buys = {n: v for n, v in buys.items() if v > 1e-6 * total}
            s = min(1.0, cash / sum(buys.values())) if buys else 1.0
            for n, v in buys.items():
                buy(n, v * s, i)
            cash -= sum(v * s for v in buys.values())

        ret = matrix.loc[nxt] / matrix.loc[d] - 1.0
        for l in lots:
            r = ret.get(l.name)
            if pd.isna(r):
                realized["st" if i - l.open_i < LT_MONTHS else "lt"] += l.value - l.basis
                cash += l.value; l.value = 0.0
            else:
                l.value *= (1 + float(r))
        lots[:] = [l for l in lots if l.value > 1e-9]

        if pd.Timestamp(nxt).year != pd.Timestamp(d).year or i == len(dates) - 2:
            net = realized["st"] + realized["lt"] + carry
            if net <= 0:
                carry = net
            else:
                st_g = max(realized["st"] + carry, 0.0)
                lt_g = max(net - st_g, 0.0)
                tax_st += st_rate * st_g; tax_lt += lt_rate * lt_g
                gain_st += st_g; gain_lt += lt_g
                carry = 0.0
                cash -= st_rate * st_g + lt_rate * lt_g
            realized = {"st": 0.0, "lt": 0.0}
        values.append(float(holdings().sum()) + cash)

    pre_liq = values[-1]
    n_last = len(dates) - 2
    lst = llt = 0.0
    for l in lots:
        g = max(l.value - l.basis, 0.0)
        if n_last - l.open_i < LT_MONTHS:
            lst += st_rate * g; gain_st += g
        else:
            llt += lt_rate * g; gain_lt += g
    tax_st += lst; tax_lt += llt
    after = pre_liq - lst - llt
    return {"pre_liq": pre_liq, "after_tax": after, "tax_st": tax_st, "tax_lt": tax_lt,
            "gain_st": gain_st, "gain_lt": gain_lt, "traded": traded,
            "values": np.array(values)}


def run_config(dates, wts, matrix, every, S):
    """S staggered sleeves at frequency `every`; aggregate pre-tax and after-tax."""
    off = every // S
    per = START / S
    yrs = (len(dates) - 1) / 12.0
    pre_liq = after = tax_st = tax_lt = gain_st = gain_lt = traded = 0.0
    val_at = np.zeros(len(dates) - 1)
    for k in range(S):
        ph = k * off
        p0 = simulate(dates, wts, matrix, every, 0.0, 0.0, phase=ph, start=per)
        pt = simulate(dates, wts, matrix, every, ST_RATE, LT_RATE, phase=ph, start=per)
        pre_liq += p0["pre_liq"]
        after += pt["after_tax"]
        tax_st += pt["tax_st"]; tax_lt += pt["tax_lt"]
        gain_st += pt["gain_st"]; gain_lt += pt["gain_lt"]
        traded += pt["traded"]; val_at += pt["values"]
    return {"pre_liq": pre_liq, "after_tax": after, "tax_st": tax_st, "tax_lt": tax_lt,
            "gain_st": gain_st, "gain_lt": gain_lt,
            "pre_cagr": (pre_liq / START) ** (1 / yrs) - 1,
            "after_cagr": (after / START) ** (1 / yrs) - 1,
            "turn": traded / 2 / np.mean(val_at) / yrs}


def phase_range(dates, wts, matrix, every):
    """Single-sleeve after-tax CAGR across every possible start phase (phase luck)."""
    yrs = (len(dates) - 1) / 12.0
    cs = []
    for ph in range(every):
        r = simulate(dates, wts, matrix, every, ST_RATE, LT_RATE, phase=ph)
        cs.append((r["after_tax"] / START) ** (1 / yrs) - 1)
    return min(cs), float(np.mean(cs)), max(cs)


def bench(matrix, dates, tkr):
    yrs = (len(dates) - 1) / 12.0
    gross = float(matrix.loc[dates[-1], tkr] / matrix.loc[dates[0], tkr])
    net_mult = gross * (1 - EXPENSE[tkr]) ** yrs      # expense ratio as annual drag
    final = START * net_mult
    tax = LT_RATE * max(final - START, 0.0)           # single lot -> all long-term
    after = final - tax
    return {"pre_liq": final, "after_tax": after, "tax_st": 0.0, "tax_lt": tax,
            "gain_st": 0.0, "gain_lt": max(final - START, 0.0),
            "pre_cagr": net_mult ** (1 / yrs) - 1,
            "after_cagr": (after / START) ** (1 / yrs) - 1, "turn": 0.0, "sharpe": np.nan}


def main():
    panel = _load_panel(False)
    with get_db() as db:
        data = load_ablation_data(panel, db, PANEL_START, PRICE_END, splits="rolling5y")
    matrix = realize_delistings(data.matrix)
    dref = [d for d in data.rebal_dates if d in matrix.index]
    span = f"{dref[0]} -> {dref[-1]}"

    grid = [(t, w, s) for t in TOPS for w in WEIGHTINGS for s in SECTORS]
    print(f"\nTaxable, ${START:,.0f}, {span}. Hold {HOLD_M}mo, {SLEEVES} sleeves, "
          f"ST {ST_RATE:.0%}/LT {LT_RATE:.0%}, 10bps/side, VIX tilt on, no exclusion.")
    print(f"Evaluating {len(grid)} configs (phase-averaged after-tax) ...\n", flush=True)

    rows = []
    for j, (t, w, s) in enumerate(grid):
        cfg = AblationConfig(name="g", top_pct=t, exclusion="none", weighting=w,
                             sector=s, vix_tilt=True)
        wts = weights_for(data, cfg)
        dates = [d for d in data.rebal_dates if d in wts and d in matrix.index]
        r = run_config(dates, wts, matrix, HOLD_M, SLEEVES)
        r.update(top=t, wt=w, sec=s,
                 fin=(t, w, s) == FINALIST)
        rows.append(r)
        if (j + 1) % 12 == 0:
            print(f"  ... {j+1}/{len(grid)} done", flush=True)

    spy, qqq = bench(matrix, dref, SPY), bench(matrix, dref, "QQQ")

    rows.sort(key=lambda r: -r["after_cagr"])
    h = (f"{'#':>3} {'top%':>5} {'weighting':>9} {'sector':>9}"
         f"{'PreTaxCAGR':>11}{'AftTaxCAGR':>11}{'WalkAway$':>11}{'Tax$':>9}"
         f"{'%LTgain':>8}{'Turn/yr':>8}")
    print("\n" + h); print("-" * len(h))
    for i, r in enumerate(rows, 1):
        lt_share = r["gain_lt"] / max(r["gain_st"] + r["gain_lt"], 1e-9)
        star = " *FIN" if r["fin"] else ""
        print(f"{i:>3} {r['top']*100:>4.0f}% {r['wt']:>9} {r['sec']:>9}"
              f"{r['pre_cagr']*100:10.1f}%{r['after_cagr']*100:10.1f}%"
              f"{r['after_tax']:11,.0f}{r['tax_st']+r['tax_lt']:9,.0f}"
              f"{lt_share*100:7.0f}%{r['turn']*100:7.0f}%{star}")
    print("-" * len(h))
    for nm, b in (("SPY", spy), ("QQQ", qqq)):
        print(f"    {nm:>5}  buy&hold          {b['pre_cagr']*100:10.1f}%"
              f"{b['after_cagr']*100:10.1f}%{b['after_tax']:11,.0f}"
              f"{b['tax_lt']:9,.0f}{'100':>7}%{'0':>7}%")

    import os
    recs = []
    for i, r in enumerate(rows, 1):
        g = r["gain_st"] + r["gain_lt"]
        recs.append({"rank": i, "top_pct": r["top"], "weighting": r["wt"],
                     "sector": r["sec"], "pre_tax_cagr": round(r["pre_cagr"], 4),
                     "after_tax_cagr": round(r["after_cagr"], 4),
                     "walkaway_after_tax": round(r["after_tax"]),
                     "final_pre_tax": round(r["pre_liq"]),
                     "total_tax": round(r["tax_st"] + r["tax_lt"]),
                     "tax_st": round(r["tax_st"]), "tax_lt": round(r["tax_lt"]),
                     "gain_st": round(r["gain_st"]), "gain_lt": round(r["gain_lt"]),
                     "pct_lt_gain": round(r["gain_lt"] / max(g, 1e-9), 3),
                     "turnover_yr": round(r["turn"], 3), "is_finalist": r["fin"]})
    for nm, b in (("SPY", spy), ("QQQ", qqq)):
        recs.append({"rank": "", "top_pct": "", "weighting": f"{nm} buy&hold",
                     "sector": "", "pre_tax_cagr": round(b["pre_cagr"], 4),
                     "after_tax_cagr": round(b["after_cagr"], 4),
                     "walkaway_after_tax": round(b["after_tax"]),
                     "final_pre_tax": round(b["pre_liq"]),
                     "total_tax": round(b["tax_lt"]), "tax_st": 0,
                     "tax_lt": round(b["tax_lt"]), "gain_st": 0,
                     "gain_lt": round(b["gain_lt"]), "pct_lt_gain": 1.0,
                     "turnover_yr": 0.0, "is_finalist": False})
    csv_path = "/home/manit/Desktop/fun_projects/mahajan_hedge_fund/output/ablation/tax_ablation_13mo_4sleeve.csv"
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    pd.DataFrame(recs).to_csv(csv_path, index=False)
    print(f"\nwrote {csv_path}")

    fin = next(r for r in rows if r["fin"])
    print(f"Finalist (top10/cap5/cap_match) ranks "
          f"#{rows.index(fin)+1}/{len(rows)} by after-tax CAGR ({fin['after_cagr']*100:.1f}%).")
    print("[note] WalkAway$ = after liquidating all and paying every gain's tax by lot "
          f"age; ${START:,.0f} start. %LTgain = share of realized gains taxed at 15%.")


if __name__ == "__main__":
    main()
