"""Clean total-return price panel for relative-value research.

Why this exists (audit 2026-07-18): ``daily_prices.adj_close`` mixes adjustment
conventions BY SOURCE ERA within the same ticker — FMP rows (bulk history to
2022-06) hold split-adjusted price-only values, yfinance rows (2022-06-21 →
2026-06) are dividend+split adjusted, polygon rows (2026-06-15 →) split-only.
Result: 395/660 tickers show an artificial adj_close break at 2022-06-21 (size
≈ cumulative 2022-26 dividend yield, up to −13%). Any level/spread statistic
crossing mid-2022 is contaminated.

``close`` IS convention-consistent (retroactively split-adjusted, price-only)
across all three sources for every ticker except CRWD/DD/HON (splice
discontinuity >25%, differing spinoff/adjustment handling). So we rebuild:

    total_return_price = close, backward-adjusted by cash dividends
                         (historical_dividends, ex-date convention)

Blacklist (excluded from ALL pair research, never repaired/fabricated):
  CRWD, DD, HON — close-series discontinuity at the 2022-06-21 source splice;
  CCE — fabricated +2280% zero-volume bar (2019-02);
  ACE — single stale bar years after its 2015 delisting;
  AIV, BKR, KDP — spinoff/merger special distributions inside the eval window
  whose dividend rows do NOT reconcile with the ex-date close move (AIV
  2020-11 −16% move vs 25% distribution; BKR 2017-07 −8% vs 43%; KDP 2018-07
  impossible 5.2x ratio) — their TR series cannot be trusted around the event.

Special distributions (yield 25-60%) are applied only when the ex-date close
move confirms them (within 10pp), e.g. GEN 2020-02 and LDOS 2016-08; otherwise
skipped and logged (a skipped-but-unreflected row means the close series was
already retro-adjusted by the vendor, so skipping is correct, e.g. DHR 2016).

Known residual limitation: a handful of delisted ex-payers have no FMP
dividend history, so their TR series is price-only (documented by the
yfinance cross-check in ``validate_tr_panel``).
"""
from __future__ import annotations

import pickle
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

DB_PATH = Path("cache/mahajan.db")
CACHE = Path("cache/relvalue_tr_panel.pkl")

BLACKLIST = {"CRWD", "DD", "HON", "CCE", "ACE", "AIV", "BKR", "KDP"}

# regular cash dividends are never >25% of the prior close; 25-60% rows are
# special distributions applied only with ex-date price confirmation
MAX_DIV_YIELD = 0.25
MAX_SPECIAL_YIELD = 0.60
SPECIAL_TOL = 0.10


def load_close_panel(db_path: Path = DB_PATH, start: str = "2014-06-01") -> pd.DataFrame:
    con = sqlite3.connect(db_path)
    df = pd.read_sql(
        "SELECT date, ticker, close FROM daily_prices WHERE date >= ?",
        con, params=(start,))
    con.close()
    px = df.pivot(index="date", columns="ticker", values="close").sort_index()
    return px.drop(columns=[t for t in BLACKLIST if t in px.columns])


def load_dividends(db_path: Path = DB_PATH) -> pd.DataFrame:
    con = sqlite3.connect(db_path)
    df = pd.read_sql(
        "SELECT ticker, ex_date, COALESCE(adj_dividend, dividend) AS div "
        "FROM historical_dividends WHERE COALESCE(adj_dividend, dividend) > 0",
        con)
    con.close()
    return df


def _adjust_ticker(s: pd.Series, exdiv: pd.DataFrame) -> tuple[pd.Series, list]:
    """Backward dividend adjustment of one close series.

    On ex-date d the factor f = 1 - D / close(prev trading day) multiplies every
    price strictly before d. Returns (adjusted series, skipped-outlier rows)."""
    vals = s.dropna()
    if vals.empty or exdiv.empty:
        return s, []
    idx = vals.index
    fac = np.ones(len(idx))
    skipped = []
    pos = idx.searchsorted(exdiv["ex_date"].to_numpy())
    for (_, row), i in zip(exdiv.iterrows(), pos):
        if i <= 0 or i >= len(idx):
            continue
        prev = vals.iloc[i - 1]
        y = row["div"] / prev
        if not 0 < y <= MAX_DIV_YIELD:
            # special distribution: apply only if the ex-date close move
            # confirms it (else the vendor already adjusted the close series)
            move = vals.iloc[i] / prev - 1
            if not (y <= MAX_SPECIAL_YIELD and abs(move + y) <= SPECIAL_TOL):
                skipped.append((s.name, row["ex_date"], row["div"], prev))
                continue
        fac[i] *= 1.0 - y
    # A[t] = prod of factors at positions > t  → applies to prices before ex-dates
    a = np.append(np.cumprod(fac[::-1])[::-1][1:], 1.0)
    out = s.copy()
    out.loc[idx] = vals.to_numpy() * a
    return out, skipped


def build_tr_panel(db_path: Path = DB_PATH, cache: Path | None = CACHE,
                   rebuild: bool = False) -> pd.DataFrame:
    """Date × ticker total-return price panel (dividends reinvested)."""
    if cache and cache.exists() and not rebuild:
        return pickle.loads(cache.read_bytes())
    close = load_close_panel(db_path)
    divs = load_dividends(db_path)
    divs = divs[divs["ticker"].isin(close.columns)]
    out = {}
    skipped_all = []
    by_ticker = dict(iter(divs.groupby("ticker")))
    for t in close.columns:
        d = by_ticker.get(t)
        if d is None:
            out[t] = close[t]
            continue
        adj, skipped = _adjust_ticker(close[t], d.sort_values("ex_date"))
        out[t] = adj
        skipped_all.extend(skipped)
    panel = pd.DataFrame(out).sort_index()
    if skipped_all:
        print(f"[prices] skipped {len(skipped_all)} outlier dividend rows "
              f"(>25% of prior close), e.g. {skipped_all[:3]}")
    if cache:
        cache.write_bytes(pickle.dumps(panel))
    return panel


def validate_tr_panel(panel: pd.DataFrame, db_path: Path = DB_PATH) -> pd.DataFrame:
    """Cross-check our dividend factors against yfinance's embedded ones.

    On 2022-06-21 the yfinance rows' adj_close/close ratio equals the cumulative
    dividend factor from then until each ticker's last yfinance row. Our TR
    reconstruction implies the same factor over the same span; mismatch >2%
    flags missing/wrong dividend history (typically delisted ex-payers)."""
    con = sqlite3.connect(db_path)
    yf = pd.read_sql(
        "SELECT ticker, date, close, adj_close FROM daily_prices "
        "WHERE source='yfinance'", con)
    con.close()
    first = yf[yf["date"] == "2022-06-21"].set_index("ticker")
    last = yf.sort_values("date").groupby("ticker").tail(1).set_index("ticker")
    rows = []
    for t in first.index.intersection(last.index):
        if t not in panel.columns:
            continue
        d0, d1 = "2022-06-21", last.at[t, "date"]
        if d0 not in panel.index or d1 not in panel.index:
            continue
        p0, p1 = panel.at[d0, t], panel.at[d1, t]
        c0, c1 = first.at[t, "close"], last.at[t, "close"]
        if any(pd.isna(v) or v <= 0 for v in (p0, p1, c0, c1)):
            continue
        ours = (p1 / p0) / (c1 / c0)          # our implied div factor over span
        theirs = (last.at[t, "adj_close"] / c1) / (first.at[t, "adj_close"] / c0)
        rows.append({"ticker": t, "ours": ours, "theirs": theirs,
                     "mismatch": ours / theirs - 1})
    rep = pd.DataFrame(rows).set_index("ticker")
    return rep.reindex(rep["mismatch"].abs().sort_values(ascending=False).index)
