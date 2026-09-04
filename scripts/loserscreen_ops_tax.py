"""Operational-friction study: rebalance frequency x taxes x book size.

Question (user, 2026-07-15): monthly full reform of ~267 names is scary —
transaction costs and short-term capital-gains tax. How much performance do
lower-friction policies give up, and does condensing to ~100 names help?

Lot-level simulation on the vpos6_20 weight stream (and a condensed 100-name
variant that matches the full book's sector weights, filled with the
largest-cap survivors per sector — cap chosen for liquidity, NOT composite
rank, which has no ordering value). Policies:
  monthly / quarterly / semiannual / annual full reform (drift in between);
  changes_only = monthly trade of entries/exits only, full reweight yearly.
Tax model: HIFO lot selection, ST/LT split at 365 days (ST 24%, LT 15%,
parameters below), losses net within year and carry forward, tax paid from
the portfolio at each year-end. Final values shown pre-liquidation and
after full liquidation (remaining gains taxed by lot age). 10 bps/side on
traded notional. Approximations (stated): dividend taxes ignored (adj_close
is total-return for both model and SPY), wash sales ignored, year-end tax
deducted as cash without forcing sales.

Usage: python scripts/loserscreen_ops_tax.py   (reuses cache/vixtilt/, ~2min)
Outputs: output/loserscreen_final/ops_tax.png / ops_tax_summary.csv
"""
from __future__ import annotations

import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from backtesting import data_loader as dl
from data.db import get_db
from loserscreen import study as st
from loserscreen.mcap import market_caps
from run_loserscreen_study import PANEL_START, PRICE_END, _load_panel

ST_RATE, LT_RATE = 0.24, 0.15
COST_PER_SIDE = 0.0010
START_CAPITAL = 100_000
LT_MONTHS = 12
POLICIES = {"monthly": (1, False), "quarterly": (3, False),
            "semiannual": (6, False), "annual": (12, False),
            "changes_only": (12, True)}   # (reform_every, monthly entry/exit)
BOOK = st._veto_spec("vpos6_20", "ratified working spec",
                     st._VETO2_SUBSETS["vpos6_20"], 0.20)
OUT = Path("output/loserscreen_final")


@dataclass
class Lot:
    name: str
    value: float
    basis: float
    open_i: int


def condense(w: pd.Series, sectors: pd.Series, mcap: pd.Series,
             n: int = 100) -> pd.Series:
    """Sector-stratified top-cap subset; weights = renormalised full-book mix."""
    sec = sectors.reindex(w.index).fillna("Unknown")
    counts = (w.groupby(sec).sum() * n).round().astype(int)
    while counts.sum() != n:                       # fix rounding drift
        counts[counts.idxmax()] += int(np.sign(n - counts.sum()))
    keep: list[str] = []
    mc = mcap.reindex(w.index).fillna(0.0)
    for s, k in counts.items():
        names = sec.index[sec == s]
        keep += list(mc[names].sort_index()
                     .sort_values(ascending=False, kind="mergesort").index[:k])
    sub = w.reindex(keep)
    return sub / sub.sum()


def simulate(dates: list[str], weights: dict[str, pd.Series],
             matrix: pd.DataFrame, reform_every: int, changes_monthly: bool,
             st_rate: float, lt_rate: float, phase: int = 0,
             start: float = START_CAPITAL,
             emergency: dict[str, set] | None = None) -> dict:
    """`emergency`: date -> set of still-acceptable names; at non-reform
    months any held name NOT in the set is sold (proceeds held as cash)."""
    lots: list[Lot] = []
    cash = float(start)
    emerg_sells = 0
    realized = {"st": 0.0, "lt": 0.0}
    carry = 0.0
    tax_paid = traded_total = 0.0
    orders: list[int] = []
    values: list[float] = []

    def holdings() -> pd.Series:
        s: dict[str, float] = {}
        for l in lots:
            s[l.name] = s.get(l.name, 0.0) + l.value
        return pd.Series(s, dtype=float)

    def sell(name: str, amount: float, i: int) -> float:
        """HIFO sell `amount` of name; returns proceeds after costs."""
        nonlocal traded_total
        mine = sorted([l for l in lots if l.name == name],
                      key=lambda l: l.basis / l.value if l.value > 0 else 0,
                      reverse=True)
        left, proceeds = amount, 0.0
        for l in mine:
            take = min(l.value, left)
            if take <= 0:
                continue
            frac = take / l.value
            gain = take - l.basis * frac
            realized["st" if i - l.open_i < LT_MONTHS else "lt"] += gain
            l.basis *= (1 - frac)
            l.value -= take
            proceeds += take
            left -= take
            if left <= 1e-9:
                break
        lots[:] = [l for l in lots if l.value > 1e-9]
        traded_total += proceeds
        return proceeds * (1 - COST_PER_SIDE)

    def buy(name: str, amount: float, i: int) -> None:
        nonlocal traded_total
        if amount <= 0:
            return
        traded_total += amount
        lots.append(Lot(name, amount * (1 - COST_PER_SIDE),
                        amount * (1 - COST_PER_SIDE), i))

    for i, d in enumerate(dates[:-1]):
        nxt = dates[i + 1]
        w_t = weights[d]
        h = holdings()
        total = float(h.sum()) + cash
        reform = (i % reform_every == phase % reform_every) or i == 0
        n_orders = 0

        if reform:
            target = w_t * total
            for name in h.index.difference(target.index):
                cash += sell(name, h[name], i); n_orders += 1
            for name in target.index:
                cur = float(h.get(name, 0.0))
                diff = float(target[name]) - cur
                if diff < -1e-6 * total:
                    cash += sell(name, -diff, i); n_orders += 1
            h = holdings()
            buys = {n: float(target[n]) - float(h.get(n, 0.0))
                    for n in target.index}
            buys = {n: v for n, v in buys.items() if v > 1e-6 * total}
            scale = min(1.0, cash / sum(buys.values())) if buys else 1.0
            for n, v in buys.items():
                buy(n, v * scale, i); n_orders += 1
            cash -= sum(v * scale for v in buys.values())
        elif emergency is not None:
            allowed = emergency.get(d, set())
            for name in h.index:
                if name not in allowed:
                    cash += sell(name, h[name], i)
                    n_orders += 1
                    emerg_sells += 1
        elif changes_monthly:
            for name in h.index.difference(w_t.index):      # exits
                cash += sell(name, h[name], i); n_orders += 1
            entries = [n for n in w_t.index if n not in h.index]
            want = {n: float(w_t[n]) * total for n in entries}
            scale = min(1.0, cash / sum(want.values())) if want else 1.0
            for n, v in want.items():
                buy(n, v * scale, i); n_orders += 1
            cash -= sum(v * scale for v in want.values())
        orders.append(n_orders)

        ret = matrix.loc[nxt] / matrix.loc[d] - 1.0
        for l in lots:
            r = ret.get(l.name)
            if pd.isna(r):                       # delisted/no price: cash out
                gain = l.value - l.basis
                realized["st" if i - l.open_i < LT_MONTHS else "lt"] += gain
                cash += l.value
                l.value = 0.0
            else:
                l.value *= (1 + float(r))
        lots[:] = [l for l in lots if l.value > 1e-9]

        if pd.Timestamp(nxt).year != pd.Timestamp(d).year or i == len(dates) - 2:
            net = realized["st"] + realized["lt"] + carry
            if net <= 0:
                carry = net
                tax = 0.0
            else:
                st_g = max(realized["st"] + carry, 0.0)
                lt_g = max(net - st_g, 0.0)
                tax = st_rate * st_g + lt_rate * lt_g
                carry = 0.0
            cash -= tax
            tax_paid += tax
            realized = {"st": 0.0, "lt": 0.0}
        values.append(float(holdings().sum()) + cash)

    pre_liq = values[-1]
    liq_tax = 0.0
    n_last = len(dates) - 2
    for l in lots:
        rate = st_rate if n_last - l.open_i < LT_MONTHS else lt_rate
        liq_tax += rate * max(l.value - l.basis, 0.0)
    rets = pd.Series(values).pct_change().dropna()
    return {"final_pre_liq": pre_liq, "final_liquidated": pre_liq - liq_tax,
            "tax_paid": tax_paid + liq_tax,
            "ann_turnover_1way": traded_total / 2
            / np.mean(values) / (len(values) / 12),
            "orders_per_month": float(np.mean(orders)),
            "emergency_sells": emerg_sells,
            "values": values,
            "after_tax_sharpe": float(rets.mean() / rets.std(ddof=1)
                                      * np.sqrt(12))}


def main() -> int:
    panel = _load_panel()
    with get_db() as db:
        matrix = dl.load_price_matrix(db, panel.universe, PANEL_START, PRICE_END)
        sectors = dl.global_sectors(db)
        mcaps = market_caps(db, list(panel.rebal_dates), panel.universe, matrix)
    res = st.run_study(panel, matrix, sectors, books=[BOOK], mcaps=mcaps,
                       verbose=False)
    w_full = res.weights[BOOK.name]
    dates = [d for d in sorted(w_full) if d in matrix.index and len(w_full[d])]
    w_100 = {d: condense(w_full[d], sectors, mcaps[d]) for d in dates}

    rows = []
    for label, wdict in (("267 names", w_full), ("100 names", w_100)):
        for pol, (every, chg) in POLICIES.items():
            for taxed in (False, True):
                r = simulate(dates, wdict, matrix, every, chg,
                             ST_RATE if taxed else 0.0,
                             LT_RATE if taxed else 0.0)
                r.pop("values", None)
                rows.append({"book": label, "policy": pol,
                             "taxed": taxed, **r})
    df = pd.DataFrame(rows)

    # SPY buy-and-hold benchmark (single lot, LT tax at liquidation only)
    spy_ret = float((matrix.loc[dates[-1], "SPY"]
                     / matrix.loc[dates[0], "SPY"]) - 1)
    spy_final = START_CAPITAL * (1 + spy_ret)
    spy_after = spy_final - LT_RATE * (spy_final - START_CAPITAL)

    OUT.mkdir(parents=True, exist_ok=True)
    df.round(4).to_csv(OUT / "ops_tax_summary.csv", index=False)

    taxed = df[df.taxed].pivot(index="policy", columns="book",
                               values="final_liquidated") \
        .reindex(POLICIES.keys())
    pre = df[~df.taxed].pivot(index="policy", columns="book",
                              values="final_pre_liq").reindex(POLICIES.keys())

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    fig.suptitle(f"Friction study on vpos6_20 — ${START_CAPITAL:,} taxable "
                 f"account, ST {ST_RATE:.0%}/LT {LT_RATE:.0%}, HIFO, "
                 "10bps/side — final value AFTER full liquidation", fontsize=11)
    ax = axes[0]
    x = np.arange(len(taxed.index))
    for i, (b, col) in enumerate((("267 names", "tab:orange"),
                                  ("100 names", "tab:cyan"))):
        ax.bar(x + (i - 0.5) * 0.36, taxed[b] / 1000, width=0.34,
               label=f"{b} (after tax)", color=col)
    ax.axhline(spy_after / 1000, color="tab:gray", ls="--", lw=1.5,
               label=f"SPY buy&hold after tax (${spy_after / 1000:,.0f}k)")
    ax.set_xticks(x, taxed.index, rotation=20)
    ax.set_ylabel("$ thousands")
    ax.set_title("After-tax, after-liquidation final value")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    ax = axes[1]
    tp = df[df.taxed].pivot(index="policy", columns="book", values="tax_paid") \
        .reindex(POLICIES.keys())
    for i, (b, col) in enumerate((("267 names", "tab:orange"),
                                  ("100 names", "tab:cyan"))):
        ax.bar(x + (i - 0.5) * 0.36, tp[b] / 1000, width=0.34, label=b,
               color=col)
    ax.set_xticks(x, tp.index, rotation=20)
    ax.set_ylabel("$ thousands")
    ax.set_title("Total tax paid (incl. final liquidation)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(OUT / "ops_tax.png", dpi=130)

    pd.set_option("display.float_format", "{:,.2f}".format)
    print("=== PRE-TAX final value (costs only) ===")
    print(pre.to_string())
    print("\n=== AFTER-TAX final value (after full liquidation) ===")
    print(taxed.to_string())
    print(f"\nSPY buy&hold: pre-tax ${spy_final:,.0f} | "
          f"after-tax ${spy_after:,.0f}")
    print("\n=== Detail (taxed runs) ===")
    print(df[df.taxed][["book", "policy", "final_liquidated", "tax_paid",
                        "ann_turnover_1way", "orders_per_month"]]
          .to_string(index=False))
    print(f"\nwrote {OUT}/ops_tax.png, ops_tax_summary.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
