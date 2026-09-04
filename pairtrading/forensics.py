"""Trade forensics for the classic pairs baseline: what do winning entries look
like, what do losing entries look like, and can any entry-time feature tell
them apart?

Re-simulates the classic cell (identical machinery to ``run_grid``), captures
entry-time features for every trade, compares winner/loser distributions
(Mann-Whitney), and draws the comparison charts. All features are computable
at the entry close — nothing post-entry leaks in.

Run:  python -m pairtrading.forensics
"""
from __future__ import annotations

import pickle
import sqlite3
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from vixtilt.baseline import CACHE_DIR, CACHE_VERSION
from vixtilt.windows import semiannual_windows

from .study import (ADJUSTED, COST_PER_SIDE, DB_PATH, FORMATION_MONTHS,
                    TradeRule, _crossings, _half_life, form_pairs, load_composites,
                    load_prices, load_sectors, members_as_of, simulate_window)

PARENTS = ["momentum", "value", "quality", "growth", "revisions",
           "institutional", "insider", "short"]

OUT = Path("output/pairtrading/forensics")
PATH_PRE, PATH_POST = 20, 60      # |z| path window around entry for the chart


def _macro(db_path: Path = DB_PATH) -> tuple[pd.Series, pd.Series]:
    con = sqlite3.connect(db_path)
    df = pd.read_sql("SELECT ticker, date, adj_close FROM daily_prices "
                     "WHERE ticker IN ('VIX','SPY')", con)
    con.close()
    piv = df.pivot(index="date", columns="ticker", values="adj_close").sort_index()
    return piv["VIX"].ffill(), piv["SPY"].ffill()


def _parent_frames() -> dict[str, pd.DataFrame]:
    """{rebal_date: ticker x parent frame} from the frozen vixtilt caches
    (each frame computed as-of its rebalance date — PIT for any later day)."""
    frames: dict[str, pd.DataFrame] = {}
    for w in semiannual_windows():
        path = CACHE_DIR / f"window_{w.label}_v{CACHE_VERSION}.pkl"
        if path.exists():
            frames.update(pickle.loads(path.read_bytes()).parent_test)
    return frames


def _asof(frames: dict, day: str):
    dates = [d for d in frames if d <= day]
    return frames[max(dates)] if dates else None


def collect(n_pairs: int = 20, db_path: Path = DB_PATH,
            rule: TradeRule = TradeRule()) -> pd.DataFrame:
    """One row per trade with entry-time features and the |z| path."""
    px, sectors = load_prices(db_path), load_sectors(db_path)
    vix, spy = _macro(db_path)
    pframes = _parent_frames()
    comps = load_composites(semiannual_windows(), sectors)
    rows = []
    for w in semiannual_windows():
        f_end = (pd.Timestamp(w.test_start) - pd.Timedelta(days=1)).date().isoformat()
        f_start = (pd.Timestamp(w.test_start)
                   - pd.DateOffset(months=FORMATION_MONTHS)).date().isoformat()
        members = members_as_of(w.test_start, db_path)
        pairs = form_pairs(px, members, sectors, f_start, f_end,
                           n_candidates=n_pairs)
        if not pairs:
            continue
        fpx = px.loc[(px.index >= f_start) & (px.index <= f_end)].dropna(how="all")
        anchor = fpx.index[-1]
        _, trades = simulate_window(px, pairs, anchor, w.test_start, w.test_end,
                                    rule=rule)

        tickers = sorted({p.a for p in pairs} | {p.b for p in pairs})
        span = (px.loc[(px.index >= anchor) & (px.index <= w.test_end), tickers]
                .dropna(how="all").ffill())
        norm = span / span.iloc[0]
        # formation stats per pair (crossings / half-life on formation spread)
        fnorm = fpx[[t for t in tickers if t in fpx.columns]].ffill()
        fnorm = fnorm / fnorm.iloc[0]
        fstats = {}
        for p in pairs:
            fs = fnorm[p.a] - fnorm[p.b]
            fstats[(p.a, p.b)] = (_crossings(fs), _half_life(fs))

        for t in trades:
            p = t.pair
            s = (norm[p.a] - norm[p.b]) / p.sigma      # spread in σ units
            si = list(s.index)
            i = si.index(t.open_date)
            sig_i = i - 1                              # signal day (one-day wait)
            z_entry, z_signal = abs(s.iloc[i]), abs(s.iloc[sig_i])
            z5 = abs(s.iloc[sig_i - 5]) if sig_i >= 5 else np.nan
            look = px[[p.a, p.b]].loc[:t.open_date].ffill().iloc[-21:]
            r20 = look.iloc[-1] / look.iloc[0] - 1.0
            days_total = len([d for d in si if d >= w.test_start])
            day_in = len([d for d in si if w.test_start <= d <= t.open_date])
            path = [(s.iloc[j] if 0 <= j < len(si) else np.nan)
                    for j in range(i - PATH_PRE, i + PATH_POST + 1)]
            cr, hl = fstats[(p.a, p.b)]
            # post-entry excursions to window end (exit-rule research)
            post = s.iloc[i:].abs()
            below = {th: post[post <= th] for th in (1.0, 0.5)}
            excur = {
                "mfe_z": post.min(), "mae_z": post.max(),
                "t_below_1.0z": (si.index(below[1.0].index[0]) - i
                                 if len(below[1.0]) else np.nan),
                "t_below_0.5z": (si.index(below[0.5].index[0]) - i
                                 if len(below[0.5]) else np.nan),
            }
            # PIT scores at entry (latest rebalance <= open date)
            scores: dict[str, float] = {}
            pf, cs = _asof(pframes, t.open_date), _asof(comps, t.open_date)
            if cs is not None:
                scores["comp_long"] = cs.get(t.long, np.nan)
                scores["comp_short"] = cs.get(t.short, np.nan)
                scores["comp_diff"] = scores["comp_long"] - scores["comp_short"]
            if pf is not None:
                for par in PARENTS:
                    if par not in pf.columns:
                        continue
                    lv = pf[par].get(t.long, np.nan)
                    sv = pf[par].get(t.short, np.nan)
                    scores[f"{par}_long"] = lv
                    scores[f"{par}_short"] = sv
                    scores[f"{par}_diff"] = lv - sv
            rows.append({
                "window": w.label, "pair": f"{p.a}/{p.b}", "sector": p.sector,
                "long": t.long, "short": t.short, "open": t.open_date,
                "close": t.close_date, "days": t.days, "payoff": t.payoff,
                "reason": t.reason, "win": t.payoff > 0,
                "sigma": p.sigma, "ssd": p.ssd, "crossings": cr, "half_life": hl,
                "z_entry": z_entry, "z_signal": z_signal,
                "widened_at_entry": z_entry > z_signal,
                "widen_speed_5d": (z_signal - z5) / 5 if not np.isnan(z5) else np.nan,
                "ret20_long": r20[t.long], "ret20_short": r20[t.short],
                "vix": vix.loc[:t.open_date].iloc[-1], "spy_ret20": (
                    spy.loc[:t.open_date].iloc[-1] / spy.loc[:t.open_date].iloc[-21] - 1.0),
                "day_in_window": day_in, "days_left": days_total - day_in,
                "path": path, **excur, **scores,
            })
    return pd.DataFrame(rows)


FEATURES = ["z_entry", "z_signal", "widen_speed_5d", "sigma", "half_life",
            "crossings", "ret20_long", "ret20_short", "vix", "spy_ret20",
            "day_in_window", "days_left"]


def feature_cols(df: pd.DataFrame) -> list[str]:
    scores = [c for c in df.columns
              if c.startswith("comp_") or any(c.startswith(f"{p}_") for p in PARENTS)]
    return FEATURES + scores


def compare(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for f in feature_cols(df):
        wv = df.loc[df.win, f].dropna()
        lv = df.loc[~df.win, f].dropna()
        pval = mannwhitneyu(wv, lv).pvalue if len(wv) and len(lv) else np.nan
        out.append({"feature": f, "win_med": wv.median(), "loss_med": lv.median(),
                    "win_mean": wv.mean(), "loss_mean": lv.mean(), "mw_p": pval})
    return pd.DataFrame(out).set_index("feature").sort_values("mw_p")


def charts(df: pd.DataFrame) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    x = np.arange(-PATH_PRE, PATH_POST + 1)

    # 1 — average |z| path around entry, winners vs losers
    fig, ax = plt.subplots(figsize=(10, 6))
    for label, sub, c in [("winners", df[df.win], "tab:green"),
                          ("losers", df[~df.win], "tab:red")]:
        paths = np.abs(np.array(sub["path"].tolist(), dtype=float))
        mean = np.nanmean(paths, axis=0)
        lo, hi = (np.nanpercentile(paths, q, axis=0) for q in (25, 75))
        ax.plot(x, mean, color=c, label=f"{label} (n={len(sub)})")
        ax.fill_between(x, lo, hi, color=c, alpha=0.15)
    ax.axvline(0, color="k", lw=0.8, ls="--")
    ax.axhline(2.0, color="grey", lw=0.8, ls=":")
    ax.set(xlabel="trading days from entry", ylabel="|spread| / formation σ",
           title="Spread path around entry — winners vs losers (classic, all trades)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "spread_paths.png", dpi=120); plt.close(fig)

    # 2 — feature distributions
    feats = ["z_entry", "widen_speed_5d", "half_life", "sigma",
             "ret20_long", "vix", "days_left", "crossings"]
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for ax, f in zip(axes.ravel(), feats):
        data = [df.loc[df.win, f].dropna(), df.loc[~df.win, f].dropna()]
        ax.boxplot(data, tick_labels=["win", "loss"], showfliers=False)
        ax.set_title(f); ax.grid(alpha=0.3)
    fig.suptitle("Entry-time features by outcome")
    fig.tight_layout(); fig.savefig(OUT / "feature_dists.png", dpi=120); plt.close(fig)

    # 3 — win rate / avg payoff by entry-condition buckets
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    specs = [("z_entry", [2.0, 2.25, 2.5, 2.75, 3.0, 3.5, 10], "entry depth (σ)"),
             ("widen_speed_5d", [-2, -0.1, -0.02, 0.02, 0.1, 2], "widening speed pre-entry (σ/day)"),
             ("days_left", [0, 40, 70, 100, 130], "days left in window")]
    for ax, (f, bins, xlab) in zip(axes, specs):
        b = pd.cut(df[f], bins)
        g = df.groupby(b, observed=True).agg(win=("win", "mean"),
                                             pay=("payoff", "mean"), n=("win", "size"))
        ax2 = ax.twinx()
        ax.bar(range(len(g)), g["win"], color="tab:blue", alpha=0.6)
        ax2.plot(range(len(g)), g["pay"], color="tab:red", marker="o")
        ax2.axhline(0, color="tab:red", lw=0.6, ls=":")
        ax.set_xticks(range(len(g)))
        ax.set_xticklabels([f"{iv}\nn={n}" for iv, n in zip(g.index.astype(str), g["n"])],
                           fontsize=7)
        ax.set(xlabel=xlab, ylabel="win rate"); ax2.set_ylabel("avg payoff", color="tab:red")
        ax.grid(alpha=0.3)
    fig.suptitle("Entry conditions vs outcome (bars = win rate, line = avg payoff)")
    fig.tight_layout(); fig.savefig(OUT / "entry_conditions.png", dpi=120); plt.close(fig)


def main() -> None:
    df = collect(rule=ADJUSTED)
    OUT.mkdir(parents=True, exist_ok=True)
    df.drop(columns="path").to_csv(OUT / "trade_features.csv", index=False)
    print(f"{len(df)} trades (ADJUSTED rule), win rate {df.win.mean():.3f}")
    for era, sub in df.groupby(df["open"] >= "2022"):
        label = "2022-26" if era else "2017-21"
        print(f"\n=== winners vs losers, {label} (n={len(sub)}) ===")
        with pd.option_context("display.float_format", "{:.4f}".format):
            c = compare(sub)
            print(c[c.mw_p < 0.25].to_string())
    print("\n=== by exit reason ===")
    print(df.groupby("reason").agg(n=("payoff", "size"), avg=("payoff", "mean"),
                                   win=("win", "mean")).to_string())
    print("\n=== by sector ===")
    print(df.groupby("sector").agg(n=("payoff", "size"), avg=("payoff", "mean"),
                                   win=("win", "mean")).sort_values("n").to_string())
    charts(df)
    print(f"\ncharts -> {OUT}")


if __name__ == "__main__":
    main()
