"""Vol-targeted leverage on the chosen book (mix_screen25) vs static 1.25x.

PRE-REGISTERED 2026-07-14 (before first run; parameters fixed):
  exposure_t = clip(0.20 / trailing_63d_SPY_realized_vol, 0.75, 1.50)
  evaluated at each monthly formation date (PIT: vol window ends at formation).
  Financing: (e-1) at fed funds + 1% when e > 1; (1-e) earns fed funds when
  e < 1. Monthly reset.
  Bar vs static 1.25x margin (ALL required):
    (a) higher full-period net Sharpe;
    (b) shallower max drawdown;
    (c) >= 10 of 19 semiannual-window wins on net total return.

Usage: python scripts/loserscreen_voltarget.py    (reuses cache/vixtilt/, ~30s)
Outputs: output/loserscreen_final/voltarget.png / voltarget_summary.csv
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
from scripts.loserscreen_leverage import FFR, MARGIN_SPREAD

TARGET_VOL, LOOKBACK_D, E_MIN, E_MAX = 0.20, 63, 0.75, 1.50
STATIC_LEV = 1.25
START_CAPITAL = 10_000
BOOK = st.BookSpec("mix_screen25", "chosen construction", weighting="mix")
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
    spy_vol = spy.pct_change().rolling(LOOKBACK_D).std() * np.sqrt(252)

    res = st.run_study(panel, matrix, sectors, books=[BOOK], mcaps=mcaps,
                       verbose=False)
    pf = res.portfolios[BOOK.name]

    rets = pd.DataFrame({"model": pf["net"], "SPY": pf["spy"]}).dropna()
    # PIT trailing vol: last available value on/before each formation date
    vol_at = spy_vol.reindex(spy_vol.index.union(rets.index)).ffill() \
        .reindex(rets.index)
    expo = (TARGET_VOL / vol_at).clip(E_MIN, E_MAX)
    ffr = pd.Series([FFR[pd.Timestamp(d).year] for d in rets.index],
                    index=rets.index)
    borrow, park = (expo - 1).clip(lower=0), (1 - expo).clip(lower=0)
    rets["voltarget"] = (expo * rets["model"]
                         - borrow * (ffr + MARGIN_SPREAD) / 12
                         + park * ffr / 12)
    rets["static 1.25x"] = (STATIC_LEV * rets["model"]
                            - (STATIC_LEV - 1) * (ffr + MARGIN_SPREAD) / 12)
    rets.index = pd.to_datetime(rets.index)
    expo.index = rets.index

    eq = (1 + rets).cumprod() * START_CAPITAL
    years = len(rets) / 12
    summary = pd.DataFrame({c: {
        "final_value": float(eq[c].iloc[-1]),
        "cagr": float((eq[c].iloc[-1] / START_CAPITAL) ** (1 / years) - 1),
        "sharpe": float(rets[c].mean() / rets[c].std(ddof=1) * np.sqrt(12)),
        "ann_vol": float(rets[c].std(ddof=1) * np.sqrt(12)),
        "max_drawdown": float((eq[c] / eq[c].cummax() - 1).min()),
    } for c in rets.columns}).round(4)

    # per semiannual window: voltarget vs static
    wod = res.window_of_date
    win_rows = []
    for w in sorted({v for v in wod.values()}):
        d = [x for x in rets.index if wod.get(x.strftime("%Y-%m-%d")) == w]
        if not d:
            continue
        rv = float((1 + rets.loc[d, "voltarget"]).prod() - 1)
        rs = float((1 + rets.loc[d, "static 1.25x"]).prod() - 1)
        win_rows.append({"window": w, "voltarget": rv, "static": rs,
                         "win": rv > rs})
    wins = pd.DataFrame(win_rows)

    sh_v, sh_s = summary.loc["sharpe", "voltarget"], \
        summary.loc["sharpe", "static 1.25x"]
    dd_v, dd_s = summary.loc["max_drawdown", "voltarget"], \
        summary.loc["max_drawdown", "static 1.25x"]
    n_win = int(wins["win"].sum())
    checks = [
        ("(a) net Sharpe", sh_v > sh_s, f"{sh_v:.3f} vs {sh_s:.3f}"),
        ("(b) max drawdown shallower", dd_v > dd_s, f"{dd_v:.1%} vs {dd_s:.1%}"),
        ("(c) window wins >= 10/19", n_win >= 10, f"{n_win}/{len(wins)}"),
    ]
    passed = all(ok for _, ok, _ in checks)

    OUT.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT / "voltarget_summary.csv")
    wins.to_csv(OUT / "voltarget_windows.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(16, 9))
    fig.suptitle(f"Vol-targeted exposure (target {TARGET_VOL:.0%}, 63d SPY vol, "
                 f"clip {E_MIN}-{E_MAX}) vs static {STATIC_LEV}x — "
                 f"VERDICT: {'PASS' if passed else 'FAIL'}", fontsize=12)

    ax = axes[0, 0]
    for c, col in (("model", "tab:blue"), ("voltarget", "tab:purple"),
                   ("static 1.25x", "tab:red"), ("SPY", "tab:gray")):
        ax.plot(eq.index, eq[c], label=f"{c}  (${eq[c].iloc[-1]:,.0f})",
                lw=2 if c == "voltarget" else 1.2,
                ls="--" if c == "SPY" else "-", color=col)
    ax.set_yscale("log")
    ax.set_title("Growth of $10,000")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.step(expo.index, expo.values, where="post", color="tab:purple", lw=1.5)
    ax.axhline(STATIC_LEV, color="tab:red", ls="--", lw=1,
               label=f"static {STATIC_LEV}x")
    ax.axhline(1.0, color="k", lw=0.5)
    ax.set_title(f"Exposure path (mean {float(expo.mean()):.2f}x)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    for c, col in (("voltarget", "tab:purple"), ("static 1.25x", "tab:red"),
                   ("model", "tab:blue")):
        e = eq[c]
        ax.plot(e.index, e / e.cummax() - 1, label=c, lw=1.2, color=col)
    ax.set_title("Drawdown")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    yr = rets.groupby(rets.index.year)
    sh = yr.apply(lambda g: g.mean() / g.std(ddof=1) * np.sqrt(12))
    x = np.arange(len(sh.index))
    for i, (c, col) in enumerate((("voltarget", "tab:purple"),
                                  ("static 1.25x", "tab:red"))):
        ax.bar(x + (i - 0.5) * 0.36, sh[c], width=0.34, label=c, color=col)
    ax.set_xticks(x, sh.index, rotation=45)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_title("Yearly Sharpe")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(OUT / "voltarget.png", dpi=130)

    print(f"VERDICT: {'PASS' if passed else 'FAIL'}")
    for name, ok, detail in checks:
        print(f"  {name}: {detail} -> {'PASS' if ok else 'FAIL'}")
    print(f"\nmean exposure {float(expo.mean()):.2f}x | "
          f"min {float(expo.min()):.2f} | max {float(expo.max()):.2f}")
    print()
    print(summary.to_string())
    print(f"\nwrote {OUT}/voltarget.png, voltarget_summary.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
