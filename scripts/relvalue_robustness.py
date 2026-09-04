"""Robustness battery for relvalue finalists.

Consumes saved eval artifacts (output/relvalue/eval/<config>/) — no new
simulations, so it cannot expand the multiple-testing ledger. Reports:
fragility (drop best pair / year / month), regime splits (VIX, bear/bull),
benchmark blends, and cross-book correlations.

Usage: python scripts/relvalue_robustness.py f5dt_hot f1_tilt ...
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from relvalue.portfolio import blend, perf_stats, yearly_table  # noqa: E402
from relvalue.prices import build_tr_panel  # noqa: E402

OUT = Path("output/relvalue")


def load_daily(cfg: str, basis: str = "invested") -> pd.Series:
    p = OUT / "eval" / cfg / f"daily_{basis}.csv"
    return pd.read_csv(p, index_col=0)["ret"]


def load_trades(cfg: str) -> pd.DataFrame:
    return pd.read_csv(OUT / "eval" / cfg / "trades.csv")


def fragility(cfg: str) -> dict:
    d = load_daily(cfg)
    tr = load_trades(cfg)
    base = perf_stats(d)
    out = {"sharpe": base["sharpe"], "cagr": base["cagr"]}
    # drop best pair: remove its trades' P&L on their close dates (approx:
    # payoff booked at close date; conservative for Sharpe, exact for CAGR)
    tr["pair"] = tr.a + "/" + tr.b
    best_pair = tr.groupby("pair").payoff.sum().idxmax()
    drop = tr[tr.pair == best_pair]
    d2 = d.copy()
    n_slots = 8
    for _, t in drop.iterrows():
        if t["close"] in d2.index:
            d2.loc[t["close"]] -= t.payoff / n_slots
    out["drop_best_pair"] = perf_stats(d2)["sharpe"]
    out["best_pair"] = best_pair
    out["best_pair_share"] = float(drop.payoff.sum() / tr.payoff.sum()) \
        if tr.payoff.sum() != 0 else np.nan
    # drop best year / month
    yr = yearly_table(d)
    best_year = yr.idxmax()
    d3 = d[d.index.str[:4] != best_year]
    out["drop_best_year"] = perf_stats(d3)["sharpe"]
    out["best_year"] = best_year
    mo = (1 + d).groupby(d.index.str[:7]).prod() - 1
    best_mo = mo.idxmax()
    d4 = d[d.index.str[:7] != best_mo]
    out["drop_best_month"] = perf_stats(d4)["sharpe"]
    return out


def regime_splits(cfg: str, tr_panel: pd.DataFrame) -> dict:
    d = load_daily(cfg)
    spy = tr_panel["SPY"].pct_change(fill_method=None).reindex(d.index)
    out = {}
    for name, (a, b) in {"2018Q4": ("2018-10-01", "2018-12-31"),
                         "covid": ("2020-02-15", "2020-06-30"),
                         "bear2022": ("2022-01-01", "2022-12-31"),
                         "melt2023_25": ("2023-01-01", "2025-06-30")}.items():
        seg = d[(d.index >= a) & (d.index <= b)]
        out[name] = perf_stats(seg).get("ann_ret", np.nan)
    vix = _vix().reindex(d.index).ffill()
    hi, lo = d[vix > 25], d[vix <= 25]
    out["vix_hi_ann"] = perf_stats(hi).get("ann_ret", np.nan)
    out["vix_lo_ann"] = perf_stats(lo).get("ann_ret", np.nan)
    return out


def _vix() -> pd.Series:
    import sqlite3
    con = sqlite3.connect("cache/mahajan.db")
    v = pd.read_sql("SELECT date, close FROM daily_prices WHERE ticker='VIX'",
                    con, index_col="date")["close"]
    con.close()
    return v.sort_index()


def blends(cfg: str, tr_panel: pd.DataFrame) -> pd.DataFrame:
    d = load_daily(cfg)
    rows = []
    for bench in ("SPY", "QQQ"):
        b = tr_panel[bench].pct_change(fill_method=None).dropna()
        rows.append({"mix": f"{bench} 100%", **perf_stats(b[b.index.isin(d.index)])})
        for w in (0.2, 0.3, 0.5):
            m = blend(d, b, w)
            rows.append({"mix": f"{int((1-w)*100)}/{int(w*100)} {bench}+{cfg}",
                         **perf_stats(m)})
    cols = ["mix", "cagr", "ann_vol", "sharpe", "sortino", "max_dd", "calmar"]
    return pd.DataFrame(rows)[cols]


def main(cfgs: list[str]) -> None:
    tr_panel = build_tr_panel()
    lines = ["# Relvalue robustness battery", ""]
    dailies = {}
    for cfg in cfgs:
        dailies[cfg] = load_daily(cfg)
        lines += [f"## {cfg}", "", "### Fragility",
                  pd.Series(fragility(cfg)).to_string(), "",
                  "### Regime splits (ann. return)",
                  pd.Series(regime_splits(cfg, tr_panel)).round(3).to_string(),
                  "", "### Yearly", (yearly_table(dailies[cfg]) * 100).round(1).to_string(),
                  "", "### Blends", blends(cfg, tr_panel).round(3).to_string(index=False), ""]
    if len(cfgs) > 1:
        cor = pd.DataFrame(dailies).corr().round(2)
        lines += ["## Cross-book correlation", cor.to_string(), ""]
        combo = pd.DataFrame(dailies).mean(axis=1)
        lines += ["## Equal-weight combo of the above books",
                  pd.Series(perf_stats(combo, {"SPY": tr_panel["SPY"].pct_change(fill_method=None)})).round(3).to_string(), ""]
    text = "\n".join(str(x) for x in lines)
    (OUT / "ROBUSTNESS.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main(sys.argv[1:])
