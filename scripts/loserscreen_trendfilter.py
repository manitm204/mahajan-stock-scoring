"""Trend filter on the ratified book (vpos6_20): de-risk under the 200d SMA.

PRE-REGISTERED 2026-07-14 (before first run; parameters fixed):
  Signal, evaluated at each monthly formation date d (PIT): SPY adjusted
  close (last close <= d) vs the 200-trading-day simple moving average
  ending at that close.
  PRIMARY rule: exposure = 1.0 when SPY >= SMA200, else 0.5. The de-risked
  half earns fed funds (annual table / 12). Exposure changes pay real
  costs: 10 bps/side on |delta exposure| of book notional.
  Bar vs the unfiltered book (ALL three required to adopt):
    (a) higher full-period net Sharpe;
    (b) shallower max drawdown;
    (c) among the semiannual windows where the filter was active (exposure
        < 1 in >= 1 month), wins >= losses on net total return.
  A-priori expectation: (b) should pass (that is what trend filters buy);
  (a) is the real test — whipsaw (2020-style V recovery) is the known
  failure mode. If only (b) passes, the filter is a drawdown tool, not a
  Sharpe tool, and is NOT adopted.
  Exploratory (no bar): binary Faber variant (exposure 0.0 below the SMA).

Usage: python scripts/loserscreen_trendfilter.py   (reuses cache/vixtilt/, ~40s)
Outputs: output/loserscreen_final/trendfilter.png / trendfilter_summary.csv /
         trendfilter_windows.csv
"""
from __future__ import annotations

import sys
import warnings
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
from scripts.loserscreen_leverage import FFR

SMA_D = 200
E_BELOW = 0.5           # primary de-risked exposure
COST_PER_SIDE = 0.0010  # on |delta exposure|
START_CAPITAL = 10_000
BOOK = st._veto_spec("vpos6_20", "ratified working spec",
                     st._VETO2_SUBSETS["vpos6_20"], 0.20)
OUT = Path("output/loserscreen_final")


def main() -> int:
    panel = _load_panel()
    with get_db() as db:
        matrix = dl.load_price_matrix(db, panel.universe, PANEL_START, PRICE_END)
        sectors = dl.global_sectors(db)
        mcaps = market_caps(db, list(panel.rebal_dates), panel.universe, matrix)
        spy = db.query_df("SELECT date, adj_close FROM daily_prices "
                          "WHERE ticker='SPY' ORDER BY date")
    spy = spy.set_index("date")["adj_close"].astype(float)
    sma = spy.rolling(SMA_D).mean()

    res = st.run_study(panel, matrix, sectors, books=[BOOK], mcaps=mcaps,
                       verbose=False)
    pf = res.portfolios[BOOK.name]
    rets = pd.DataFrame({"model": pf["net"], "SPY": pf["spy"]}).dropna()

    def pit(series: pd.Series) -> pd.Series:
        return series.reindex(series.index.union(rets.index)).ffill() \
            .reindex(rets.index)

    above = (pit(spy) >= pit(sma)).astype(float)
    ffr = pd.Series([FFR[pd.Timestamp(d).year] for d in rets.index],
                    index=rets.index)

    expos: dict[str, pd.Series] = {}
    for name, e_below in (("filtered 0.5", E_BELOW), ("faber 0/1", 0.0)):
        e = above + (1 - above) * e_below
        expos[name] = e
        switch_cost = COST_PER_SIDE * e.diff().abs().fillna(0.0)
        rets[name] = (e * rets["model"] + (1 - e) * ffr / 12 - switch_cost)

    wod = res.window_of_date
    win_rows = []
    for w in sorted(set(wod.values())):
        d = [x for x in rets.index if wod.get(x) == w]
        if not d:
            continue
        active = bool((expos["filtered 0.5"].loc[d] < 1).any())
        rf = float((1 + rets.loc[d, "filtered 0.5"]).prod() - 1)
        rm = float((1 + rets.loc[d, "model"]).prod() - 1)
        win_rows.append({"window": w, "filter_active": active,
                         "filtered": rf, "model": rm, "win": rf > rm})
    wins = pd.DataFrame(win_rows)

    rets.index = pd.to_datetime(rets.index)
    eq = (1 + rets).cumprod() * START_CAPITAL
    years = len(rets) / 12
    summary = pd.DataFrame({c: {
        "final_value": float(eq[c].iloc[-1]),
        "cagr": float((eq[c].iloc[-1] / START_CAPITAL) ** (1 / years) - 1),
        "sharpe": float(rets[c].mean() / rets[c].std(ddof=1) * np.sqrt(12)),
        "ann_vol": float(rets[c].std(ddof=1) * np.sqrt(12)),
        "max_drawdown": float((eq[c] / eq[c].cummax() - 1).min()),
    } for c in rets.columns}).round(4)

    act = wins[wins["filter_active"]]
    n_w, n_l = int(act["win"].sum()), int((~act["win"]).sum())
    sh_f, sh_m = summary.loc["sharpe", "filtered 0.5"], \
        summary.loc["sharpe", "model"]
    dd_f, dd_m = summary.loc["max_drawdown", "filtered 0.5"], \
        summary.loc["max_drawdown", "model"]
    checks = [
        ("(a) net Sharpe", sh_f > sh_m, f"{sh_f:.3f} vs {sh_m:.3f}"),
        ("(b) max drawdown shallower", dd_f > dd_m, f"{dd_f:.1%} vs {dd_m:.1%}"),
        ("(c) active-window wins >= losses", n_w >= n_l,
         f"{n_w}W/{n_l}L of {len(act)} active windows"),
    ]
    passed = all(ok for _, ok, _ in checks)

    OUT.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT / "trendfilter_summary.csv")
    wins.to_csv(OUT / "trendfilter_windows.csv", index=False)

    colors = {"model": "tab:blue", "filtered 0.5": "tab:purple",
              "faber 0/1": "tab:red", "SPY": "tab:gray"}
    fig, axes = plt.subplots(2, 2, figsize=(16, 9))
    fig.suptitle(f"SPY 200d-SMA trend filter on vpos6_20 (below: e={E_BELOW}, "
                 f"cash at fed funds, 10bps/side on switches) — "
                 f"VERDICT: {'PASS' if passed else 'FAIL'}", fontsize=12)

    ax = axes[0, 0]
    for c in ("model", "filtered 0.5", "faber 0/1", "SPY"):
        ax.plot(eq.index, eq[c], label=f"{c}  (${eq[c].iloc[-1]:,.0f})",
                lw=2 if c == "filtered 0.5" else 1.2,
                ls="--" if c == "SPY" else "-", color=colors[c])
    ax.set_yscale("log")
    ax.set_title("Growth of $10,000")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    e = expos["filtered 0.5"].copy()
    e.index = rets.index
    ax.step(e.index, e.values, where="post", color="tab:purple", lw=1.4,
            label=f"exposure (mean {float(e.mean()):.2f})")
    ax.set_ylim(-0.05, 1.1)
    ax.set_title(f"Exposure path — de-risked in {int((e < 1).sum())} of "
                 f"{len(e)} months")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    for c in ("model", "filtered 0.5", "faber 0/1"):
        v = eq[c]
        ax.plot(v.index, v / v.cummax() - 1, label=c, lw=1.2, color=colors[c])
    ax.set_title("Drawdown")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    yr = rets.groupby(rets.index.year)
    sh = yr.apply(lambda g: g.mean() / g.std(ddof=1) * np.sqrt(12))
    x = np.arange(len(sh.index))
    for i, c in enumerate(("model", "filtered 0.5", "faber 0/1")):
        ax.bar(x + (i - 1) * 0.28, sh[c], width=0.26, label=c, color=colors[c])
    ax.set_xticks(x, sh.index, rotation=45)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_title("Yearly Sharpe")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(OUT / "trendfilter.png", dpi=130)

    print(f"VERDICT: {'PASS' if passed else 'FAIL'}")
    for name, ok, detail in checks:
        print(f"  {name}: {detail} -> {'PASS' if ok else 'FAIL'}")
    print()
    print(summary.to_string())
    print(f"\nwrote {OUT}/trendfilter.png, trendfilter_summary.csv, "
          f"trendfilter_windows.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
