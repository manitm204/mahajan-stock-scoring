"""Deep history (2004→) for the SPY signal study.

``cache/mahajan.db`` only reaches back to 2014-06, so this module pulls a
longer close-price panel straight from FMP's stable EOD endpoint plus VIX from
FRED, and caches it. It deliberately does NOT write to mahajan.db: that is the
production database and this is research-only history.

The fetch is validated against the DB over the 2014-2026 overlap before use —
if FMP's adjustment convention disagreed with the DB's, every level statistic
crossing 2014 would be silently wrong.
"""
from __future__ import annotations

import os
import pickle
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

from spystudy.data import TICKERS

# the funds under test and the volatility index that matches each one's index
FUNDS = {"SPY": "VIXCLS", "QQQ": "VXNCLS", "IWM": "RVXCLS"}
ALL_TICKERS = TICKERS + ["QQQ", "IWM"]

load_dotenv()

CACHE = Path("cache/spystudy_deep.pkl")
CACHE_MULTI = Path("cache/spystudy_deep_multi.pkl")
START = "2004-01-01"  # a year of warmup before the 2005 study start
FMP = "https://financialmodelingprep.com/stable/historical-price-eod/full"
FRED = "https://api.stlouisfed.org/fred/series/observations"
# the endpoint returns at most 5000 rows, so pull in windows and stitch
WINDOWS = [("2004-01-01", "2010-12-31"), ("2011-01-01", "2017-12-31"),
           ("2018-01-01", "2026-12-31")]


def _fmp_series(ticker: str, key: str) -> pd.Series:
    out = []
    for a, b in WINDOWS:
        r = requests.get(FMP, params={"symbol": ticker, "from": a, "to": b,
                                      "apikey": key}, timeout=90)
        r.raise_for_status()
        rows = r.json()
        if rows:
            out.append(pd.DataFrame(rows)[["date", "close"]])
        time.sleep(0.25)
    if not out:
        return pd.Series(dtype=float, name=ticker)
    df = pd.concat(out).drop_duplicates("date").set_index("date")["close"]
    df.index = pd.to_datetime(df.index)
    return df.sort_index().rename(ticker)


def fetch_vol_index(key: str, series_id: str = "VIXCLS") -> pd.Series:
    r = requests.get(FRED, params={"series_id": series_id, "api_key": key,
                                   "file_type": "json",
                                   "observation_start": START}, timeout=90)
    r.raise_for_status()
    obs = pd.DataFrame(r.json()["observations"])
    obs = obs[obs["value"] != "."]
    s = pd.Series(obs["value"].astype(float).values,
                  index=pd.to_datetime(obs["date"]), name=series_id)
    return s.sort_index()


def fetch_vix(key: str) -> pd.Series:
    return fetch_vol_index(key, "VIXCLS").rename("VIX")


def build_multi(rebuild: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Close panel for the sectors + SPY/QQQ/IWM, and the three vol indices.

    Each fund is paired with the volatility index written on its own underlying
    (VIX/SPX, VXN/NDX, RVX/RUT) rather than reusing VIX for all three.
    """
    if CACHE_MULTI.exists() and not rebuild:
        return pickle.loads(CACHE_MULTI.read_bytes())
    fkey, rkey = os.getenv("FMP_API_KEY"), os.getenv("FRED_API_KEY")
    if not fkey or not rkey:
        raise RuntimeError("FMP_API_KEY and FRED_API_KEY must be set")
    cols = {}
    for t in ALL_TICKERS:
        cols[t] = _fmp_series(t, fkey)
        print(f"  {t}: {cols[t].notna().sum()} bars")
    close = pd.DataFrame(cols)
    close = close.loc[close["SPY"].notna()]
    vols = pd.DataFrame({
        f: fetch_vol_index(rkey, sid).reindex(close.index).ffill(limit=3)
        for f, sid in FUNDS.items()})
    print(f"  vol indices: {', '.join(f'{f}={s}' for f, s in FUNDS.items())}")
    CACHE_MULTI.write_bytes(pickle.dumps((close, vols)))
    return close, vols


def build(rebuild: bool = False) -> tuple[pd.DataFrame, pd.Series]:
    """Return (close panel 2004→, VIX series). Cached."""
    if CACHE.exists() and not rebuild:
        return pickle.loads(CACHE.read_bytes())
    fkey, rkey = os.getenv("FMP_API_KEY"), os.getenv("FRED_API_KEY")
    if not fkey or not rkey:
        raise RuntimeError("FMP_API_KEY and FRED_API_KEY must be set")
    cols = {}
    for t in TICKERS:
        cols[t] = _fmp_series(t, fkey)
        print(f"  {t}: {cols[t].notna().sum()} bars "
              f"{cols[t].index.min().date()} → {cols[t].index.max().date()}")
    close = pd.DataFrame(cols)
    close = close.loc[close["SPY"].notna()]
    vix = fetch_vix(rkey).reindex(close.index).ffill(limit=3)
    CACHE.write_bytes(pickle.dumps((close, vix)))
    return close, vix


def validate(close: pd.DataFrame, db_close: pd.DataFrame) -> pd.DataFrame:
    """Compare the fetched panel to the DB over their overlap.

    A convention mismatch (different split/adjustment handling) would show up
    as a large median ratio deviation on some ticker.
    """
    idx = close.index.intersection(db_close.index)
    rows = []
    for t in close.columns:
        a, b = close.loc[idx, t], db_close.loc[idx, t]
        ok = a.notna() & b.notna()
        if ok.sum() == 0:
            rows.append({"ticker": t, "n": 0, "median_ratio": float("nan"),
                         "max_abs_dev": float("nan")})
            continue
        ratio = (a[ok] / b[ok])
        rows.append({"ticker": t, "n": int(ok.sum()),
                     "median_ratio": float(ratio.median()),
                     "max_abs_dev": float((ratio - 1).abs().max())})
    return pd.DataFrame(rows)
