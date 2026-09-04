"""Shared infrastructure for the analyst deep-dive research program (2026-08-02).

RESEARCH-ONLY. Nothing here is imported by production code. Reuses:
  * the cached Aug-1 candidate panel (42 monthly PIT rebalances, all 91
    candidates' raw + sector-percentile scores) so every panel-level number is
    directly comparable to the official subfactor-expansion battery;
  * research/forward_returns.py on the same adj_close matrix the battery used;
  * factors.utils.sector_percentile for scoring parity.

PIT conventions mirrored from the battery:
  * analyst_grades rows admitted only once the month is complete +1 month
    (strict reporting-lag rule);
  * analyst_price_target_events admitted at published_date;
  * all firm-level baselines are expanding-window (data strictly < cutoff)
    with effective-sample-size shrinkage toward the cross-firm mean.
"""
from __future__ import annotations

import pickle
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
DB_PATH = REPO / "cache" / "mahajan.db"
PANEL_PKL = REPO / "cache" / "subfactor_expansion" / "cand_panel_2023-01-01_2026-06-30_monthly_v2.pkl"
OUT_DIR = REPO / "output" / "analyst_deep_dive"
CHART_DIR = OUT_DIR / "charts"

# A-priori economic bounds on 12M implied upside used to exclude the known
# stock-split unit-mismatch artifacts in analyst_price_target_events (found
# 2026-08-02: ~1,200 rows where price_target is pre-split units but
# price_when_posted is a retroactively split-adjusted snapshot). Real 12M sell
# side targets essentially never imply >+300% or <-90% at issue. These bounds
# were chosen before looking at any outcome data.
UPSIDE_MAX = 3.0
UPSIDE_MIN = -0.9

SECTOR_NORMALIZE = {
    "Healthcare": "Health Care",
    "Consumer Cyclical": "Consumer Discretionary",
    "Consumer Defensive": "Consumer Staples",
    "Financial Services": "Financials",
    "Basic Materials": "Materials",
    "Technology": "Information Technology",
}
SECTOR_ETF = {
    "Information Technology": "XLK", "Financials": "XLF", "Health Care": "XLV",
    "Consumer Discretionary": "XLY", "Consumer Staples": "XLP", "Energy": "XLE",
    "Industrials": "XLI", "Materials": "XLB", "Utilities": "XLU",
    "Real Estate": "XLRE", "Communication Services": "XLC",
}

HORIZON_MONTHS = {"1M": 1, "3M": 3, "6M": 6, "12M": 12}


def connect() -> sqlite3.Connection:
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con


def load_panel():
    with open(PANEL_PKL, "rb") as f:
        return pickle.load(f)


def load_price_matrix(start: str = "2021-01-01", end: str = "2026-07-29",
                      extra_tickers: list[str] | None = None) -> pd.DataFrame:
    """adj_close matrix, same source/column as the official battery."""
    con = connect()
    df = pd.read_sql_query(
        "SELECT ticker, date, adj_close FROM daily_prices WHERE date >= ? AND date <= ?",
        con, params=(start, end))
    con.close()
    wide = df.pivot_table(index="date", columns="ticker", values="adj_close").sort_index()
    return wide


def sector_map(normalized: bool = False) -> pd.Series:
    con = connect()
    df = pd.read_sql_query("SELECT ticker, gics_sector FROM universe", con)
    con.close()
    s = df.set_index("ticker")["gics_sector"]
    if normalized:
        s = s.map(lambda x: SECTOR_NORMALIZE.get(x, x))
    return s


def load_events(clean: bool = True) -> pd.DataFrame:
    """Price-target events with implied upside; optionally artifact-filtered.

    Duplicate same (ticker, firm, day) rows are collapsed to the last quote so
    one firm's repeated same-day prints can't get extra weight.
    """
    con = connect()
    # COALESCE(adj_price_target, ...) — split-adjusted target basis, matching
    # the 2026-08-02 production fix in data/grades.py.
    df = pd.read_sql_query(
        "SELECT ticker, published_date, analyst_company, "
        "COALESCE(adj_price_target, price_target) AS price_target, price_when_posted "
        "FROM analyst_price_target_events "
        "WHERE COALESCE(adj_price_target, price_target) IS NOT NULL "
        "AND price_when_posted > 0", con)
    con.close()
    df["date"] = df["published_date"].str[:10]
    df["upside"] = df["price_target"] / df["price_when_posted"] - 1.0
    df["log_upside"] = np.log(df["price_target"] / df["price_when_posted"])
    df = df.sort_values("published_date")
    df = df.drop_duplicates(["ticker", "analyst_company", "date"], keep="last")
    if clean:
        df = df[(df["upside"] <= UPSIDE_MAX) & (df["upside"] >= UPSIDE_MIN)]
    return df.reset_index(drop=True)


def load_grades_pit() -> pd.DataFrame:
    """analyst_grades with the visibility date each monthly row becomes PIT-admissible.

    Mirrors the battery's strict rule: a month-dated row is visible only from
    (month start + 2 months), i.e. `date(date,'+1 month') <= cutoff` where the
    row date is the first of the month it covers.
    """
    con = connect()
    df = pd.read_sql_query(
        "SELECT ticker, date, strong_buy, buy, hold, sell, strong_sell, total "
        "FROM analyst_grades", con)
    con.close()
    d = pd.to_datetime(df["date"])
    df["visible_from"] = (d + pd.DateOffset(months=1)).dt.strftime("%Y-%m-%d")
    return df


# ---------------------------------------------------------------------------
# Forward-return / horizon helpers
# ---------------------------------------------------------------------------

def nearest_idx(index_ts: pd.DatetimeIndex, target: pd.Timestamp,
                max_gap_days: int = 15):
    pos = index_ts.searchsorted(target)
    cands = []
    if pos < len(index_ts):
        cands.append(index_ts[pos])
    if pos > 0:
        cands.append(index_ts[pos - 1])
    best = min(cands, key=lambda dd: abs((dd - target).days), default=None)
    if best is None or abs((best - target).days) > max_gap_days:
        return None
    return best


def panel_forward_returns(matrix: pd.DataFrame, rebal_dates: list[str]):
    """Delegates to the battery's own implementation for exact parity."""
    import sys
    sys.path.insert(0, str(REPO))
    from research.forward_returns import compute_forward_returns
    return compute_forward_returns(matrix, rebal_dates)


# ---------------------------------------------------------------------------
# IC / stats utilities
# ---------------------------------------------------------------------------

def spearman_ic(scores: pd.Series, fwd: pd.Series, min_names: int = 20):
    df = pd.DataFrame({"s": scores, "f": fwd}).dropna()
    if len(df) < min_names or df["s"].nunique() < 2:
        return None
    ic = df["s"].corr(df["f"], method="spearman")
    return None if pd.isna(ic) else float(ic)


def ic_series(score_by_date: dict[str, pd.Series],
              fwd_by_date: dict[str, pd.Series]) -> pd.Series:
    out = {}
    for d, s in score_by_date.items():
        if d not in fwd_by_date:
            continue
        ic = spearman_ic(s, fwd_by_date[d])
        if ic is not None:
            out[d] = ic
    return pd.Series(out).sort_index()


def summarize_ic(ics: pd.Series) -> dict:
    if len(ics) == 0:
        return {"mean_ic": np.nan, "ic_ir": np.nan, "hit_rate": np.nan, "n_periods": 0}
    return {
        "mean_ic": float(ics.mean()),
        "ic_ir": float(ics.mean() / ics.std(ddof=1)) if ics.std(ddof=1) > 0 else np.nan,
        "hit_rate": float((ics > 0).mean()),
        "n_periods": int(len(ics)),
    }


def bootstrap_mean_ci(series: pd.Series, n_boot: int = 2000, seed: int = 42,
                      alpha: float = 0.10) -> tuple[float, float, float]:
    """Bootstrap CI for the mean by resampling periods (months) with replacement."""
    rng = np.random.default_rng(seed)
    vals = series.dropna().values
    if len(vals) == 0:
        return (np.nan, np.nan, np.nan)
    means = np.array([rng.choice(vals, size=len(vals), replace=True).mean()
                      for _ in range(n_boot)])
    return (float(vals.mean()), float(np.quantile(means, alpha / 2)),
            float(np.quantile(means, 1 - alpha / 2)))


def month_cluster_bootstrap_slope(df: pd.DataFrame, xcol: str, ycol: str,
                                  n_boot: int = 1000, seed: int = 42):
    """OLS slope with month-cluster bootstrap CI (resample event-months).

    Handles the overlapping-forward-window dependence: all events in the same
    calendar month are kept or dropped together.
    """
    rng = np.random.default_rng(seed)
    d = df[[xcol, ycol]].dropna().copy()
    d["month"] = df.loc[d.index, "date"].str[:7]
    months = d["month"].unique()
    x, y = d[xcol].values, d[ycol].values
    beta = np.polyfit(x, y, 1)[0]
    alpha_hat = y.mean() - beta * x.mean()
    slopes = []
    grouped = {m: g for m, g in d.groupby("month")}
    for _ in range(n_boot):
        pick = rng.choice(months, size=len(months), replace=True)
        sample = pd.concat([grouped[m] for m in pick], ignore_index=True)
        if sample[xcol].nunique() < 2:
            continue
        slopes.append(np.polyfit(sample[xcol].values, sample[ycol].values, 1)[0])
    slopes = np.array(slopes)
    se = float(slopes.std(ddof=1))
    return {
        "alpha": float(alpha_hat), "beta": float(beta), "boot_se": se,
        "t_boot": float(beta / se) if se > 0 else np.nan,
        "ci_lo": float(np.quantile(slopes, 0.05)),
        "ci_hi": float(np.quantile(slopes, 0.95)),
        "n": int(len(d)), "n_months": int(len(months)),
    }


def wins_series(s: pd.Series, lo: float = 0.05, hi: float = 0.95) -> pd.Series:
    """Plain (non-sector) winsorization for event-level work."""
    if s.dropna().empty:
        return s
    ql, qh = s.quantile(lo), s.quantile(hi)
    return s.clip(ql, qh)


def year_of(d: str) -> str:
    return d[:4]


def semester_of(d: str) -> str:
    y, m = d[:4], int(d[5:7])
    return f"{y}H{1 if m <= 6 else 2}"
