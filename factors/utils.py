"""Layer 2 data access and the universal scoring primitives.

Two responsibilities live here so the factor modules stay small and pure:

1. :class:`DataContext` — a *point-in-time*, read-only view over the Layer 1
   SQLite warehouse. Every loader returns a pandas object indexed by ticker and
   respects an ``as_of`` cutoff so the same code produces today's ranking or a
   historical one (backtesting) with no look-ahead. Loaders are cached per
   context instance.

2. :func:`sector_percentile` — the one ranking rule every sub-factor flows
   through: GICS-sector-relative percentile ranks on a 0–100 scale where missing
   values are neutral (50) and never penalized.

Market capitalization is reconstructed from ``net_income / eps_diluted`` because
Layer 1's ``shares_outstanding`` (and therefore its price-based valuation
features) is unpopulated for most of the universe. This implied-share count
matches reported diluted share counts within a couple percent and is available
for ~99% of names.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Iterable

import numpy as np
import pandas as pd

from data.db import Database, get_db

# Sectors where Altman Z-Score is not meaningful (different balance-sheet shape).
ALTMAN_EXCLUDED_SECTORS = {"Financials", "Real Estate"}


# ---------------------------------------------------------------------------
# Universal ranking rule
# ---------------------------------------------------------------------------
def sector_percentile(
    values: pd.Series,
    sectors: pd.Series,
    higher_is_better: bool = True,
    min_obs: int = 5,
) -> pd.Series:
    """Convert raw metric values to 0–100 GICS-sector-relative percentile ranks.

    Ranking is done *within* each sector so a Technology name only competes
    against Technology peers. The result is aligned to the full ``sectors``
    index (the scoring universe). Two neutrality rules apply:

    * A ticker with a missing value scores 50 (never penalized).
    * A sector with fewer than ``min_obs`` valid observations scores 50 for the
      whole group — too few peers to rank meaningfully.

    ``higher_is_better=False`` flips the direction so smaller raw values (e.g.
    EV/EBITDA, debt/equity, short interest) earn the high scores.
    """
    frame = pd.DataFrame({"v": values})
    frame = frame.reindex(sectors.index)
    frame["sec"] = sectors.values
    out = pd.Series(50.0, index=sectors.index, dtype="float64")
    for _, grp in frame.groupby("sec", sort=False):
        valid = grp["v"].dropna()
        if len(valid) < min_obs:
            continue
        ranks = valid.rank(pct=True, ascending=higher_is_better)
        out.loc[valid.index] = (ranks * 100.0).round(4)
    return out


# ---------------------------------------------------------------------------
# Point-in-time data context
# ---------------------------------------------------------------------------
class DataContext:
    """Read-only, ``as_of``-aware accessor over the Layer 1 warehouse.

    All loaders return pandas objects indexed by ticker, restricted to the
    scoring universe and to data observable on/before ``as_of``.
    """

    def __init__(
        self,
        db: Database | None = None,
        as_of: str | None = None,
        universe: Iterable[str] | None = None,
        insider_window_days: int = 90,
        reporting_lag: bool = False,
        lag_quarterly_days: int = 45,
        lag_annual_days: int = 90,
        lag_short_interest_days: int = 14,
    ) -> None:
        self.db = db or get_db()
        self._owns_db = db is None
        self.universe: list[str] = sorted(universe) if universe else self.db.universe_tickers()
        # `cutoff` is the upper bound for every "<= ?" filter. With no --date we
        # use a far-future sentinel so each table returns its own freshest row
        # (snapshot tables can lead the price calendar by a day or two).
        self.cutoff: str = as_of or "9999-12-31"
        # `ref_date` is a concrete trading date (latest price on/before cutoff),
        # used for window math (insider/returns) and as the reported as-of date.
        ref = self.db.max_value("daily_prices", "date", "date <= ?", (self.cutoff,))
        self.ref_date: str = ref or (as_of or "9999-12-31")
        self.as_of: str = self.ref_date
        self.insider_window_days = insider_window_days
        # Point-in-time reporting lag. Fundamentals are filed weeks after the
        # fiscal period closes, so for historical (backtest) scoring we only
        # admit a statement once its conservative availability date has passed.
        # `cutoff` already bounds the *fiscal* date; the lagged cutoffs below
        # bound it further so a Q1 statement is invisible until ~45 days later.
        # Disabled by default (latest/live scoring keeps using `cutoff` as-is).
        self.reporting_lag = reporting_lag
        self.lag_quarterly_days = lag_quarterly_days
        self.lag_annual_days = lag_annual_days
        self.lag_short_interest_days = lag_short_interest_days
        self.pit_notes: list[str] = []
        self._cache: dict[str, object] = {}

        # Record the structural point-in-time assumptions so a backtest can
        # surface them (the schema carries no fundamentals filing dates, so the
        # lag below is an explicit, conservative stand-in — see `_fund_cutoff`).
        if self.reporting_lag and self.cutoff <= "9000-01-01":
            self._note(
                f"Fundamentals carry no filing date in Layer 1; admitted with a "
                f"conservative reporting lag ({self.lag_quarterly_days}d after a "
                f"quarter end, {self.lag_annual_days}d after a fiscal year end).")
            self._note(
                "13F institutional signals admitted by filing_date (not report_date) "
                "to avoid look-ahead on filings published weeks after quarter end.")
            self._note(
                f"Short interest admitted {self.lag_short_interest_days}d after the "
                f"FINRA settlement date (dissemination lag).")

    def _note(self, msg: str) -> None:
        """Record a point-in-time assumption / missing-data warning (deduped)."""
        if msg not in self.pit_notes:
            self.pit_notes.append(msg)

    def close(self) -> None:
        if self._owns_db:
            self.db.close()

    def __enter__(self) -> "DataContext":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- point-in-time helpers ---------------------------------------------
    def _fund_cutoff(self, period_type: str) -> str:
        """Fiscal-date cutoff for fundamentals, optionally lagged for availability.

        Without ``reporting_lag`` (live scoring) the raw ``cutoff`` is used so the
        freshest filed statement is visible. With it (backtesting) the cutoff is
        pulled back by the reporting lag so a statement is only admitted once it
        would realistically have been filed: ``fiscal_date <= cutoff - lag``.
        The sentinel far-future cutoff is left untouched.
        """
        if not self.reporting_lag or self.cutoff > "9000-01-01":
            return self.cutoff
        lag = self.lag_annual_days if period_type == "annual" else self.lag_quarterly_days
        return (pd.Timestamp(self.cutoff) - pd.Timedelta(days=lag)).date().isoformat()

    # -- low-level helper ---------------------------------------------------
    def _latest_per_ticker(self, table: str, date_col: str, columns: str = "*") -> pd.DataFrame:
        """Latest row per ticker on/before ``as_of``, indexed by ticker."""
        sql = f"""
            WITH latest AS (
                SELECT ticker, MAX({date_col}) AS _d
                FROM {table}
                WHERE {date_col} <= ?
                GROUP BY ticker
            )
            SELECT t.{columns if columns == '*' else columns}
            FROM {table} t
            JOIN latest l ON t.ticker = l.ticker AND t.{date_col} = l._d
        """
        df = self.db.query_df(sql, (self.cutoff,))
        if df.empty:
            return df
        df = df.drop_duplicates(subset="ticker", keep="last").set_index("ticker")
        return df.reindex(self.universe)

    # -- reference ----------------------------------------------------------
    def sectors(self) -> pd.Series:
        if "sectors" not in self._cache:
            df = self.db.query_df("SELECT ticker, gics_sector FROM universe")
            s = df.set_index("ticker")["gics_sector"].reindex(self.universe)
            self._cache["sectors"] = s.fillna("Unknown")
        return self._cache["sectors"]  # type: ignore[return-value]

    # -- prices & price features -------------------------------------------
    def prices(self) -> pd.Series:
        if "prices" not in self._cache:
            df = self._latest_per_ticker("daily_prices", "date", "ticker, adj_close, close")
            px = df["adj_close"].fillna(df["close"]) if not df.empty else pd.Series(dtype=float)
            self._cache["prices"] = px.reindex(self.universe)
        return self._cache["prices"]  # type: ignore[return-value]

    def price_features(self) -> pd.DataFrame:
        if "price_features" not in self._cache:
            self._cache["price_features"] = self._latest_per_ticker("price_features", "date")
        return self._cache["price_features"]  # type: ignore[return-value]

    # -- fundamentals -------------------------------------------------------
    def fund_annual(self) -> pd.DataFrame:
        """Full annual raw-fundamentals history (<= as_of), ticker x fiscal_date."""
        if "fund_annual" not in self._cache:
            df = self.db.query_df(
                "SELECT * FROM fundamentals WHERE period_type='annual' AND fiscal_date <= ? "
                "ORDER BY ticker, fiscal_date",
                (self._fund_cutoff("annual"),),
            )
            self._cache["fund_annual"] = df[df["ticker"].isin(self.universe)]
        return self._cache["fund_annual"]  # type: ignore[return-value]

    def fund_annual_latest(self) -> pd.DataFrame:
        if "fund_annual_latest" not in self._cache:
            df = self.fund_annual()
            latest = df.sort_values("fiscal_date").drop_duplicates("ticker", keep="last")
            self._cache["fund_annual_latest"] = latest.set_index("ticker").reindex(self.universe)
        return self._cache["fund_annual_latest"]  # type: ignore[return-value]

    def fund_annual_prior(self) -> pd.DataFrame:
        """Second-to-last annual statement per ticker (for YoY/Piotroski deltas)."""
        if "fund_annual_prior" not in self._cache:
            df = self.fund_annual().sort_values("fiscal_date")
            # nth(-2) keeps the original row index, so set the ticker index back.
            prior = df.groupby("ticker").nth(-2).set_index("ticker")
            self._cache["fund_annual_prior"] = prior.reindex(self.universe)
        return self._cache["fund_annual_prior"]  # type: ignore[return-value]

    def fund_features_latest(self) -> pd.DataFrame:
        """Latest quarterly fundamental_features per ticker (TTM-based ratios)."""
        if "ff_latest" not in self._cache:
            sql = """
                WITH latest AS (
                    SELECT ticker, MAX(fiscal_date) AS _d
                    FROM fundamental_features
                    WHERE period_type='quarterly' AND fiscal_date <= ?
                    GROUP BY ticker
                )
                SELECT ff.* FROM fundamental_features ff
                JOIN latest l ON ff.ticker=l.ticker AND ff.fiscal_date=l._d
                WHERE ff.period_type='quarterly'
            """
            df = self.db.query_df(sql, (self._fund_cutoff("quarterly"),))
            if not df.empty:
                df = df.drop_duplicates("ticker", keep="last").set_index("ticker")
            self._cache["ff_latest"] = df.reindex(self.universe)
        return self._cache["ff_latest"]  # type: ignore[return-value]

    def fund_features_annual(self) -> pd.DataFrame:
        """Annual fundamental_features history (for stability/trend/acceleration)."""
        if "ff_annual" not in self._cache:
            df = self.db.query_df(
                "SELECT * FROM fundamental_features WHERE period_type='annual' AND fiscal_date <= ? "
                "ORDER BY ticker, fiscal_date",
                (self._fund_cutoff("annual"),),
            )
            self._cache["ff_annual"] = df[df["ticker"].isin(self.universe)]
        return self._cache["ff_annual"]  # type: ignore[return-value]

    # -- derived market cap / EV -------------------------------------------
    def implied_shares(self) -> pd.Series:
        """Diluted share count backed out of net_income / eps_diluted.

        Layer 1 stores shares_outstanding for only a handful of names, so we
        reconstruct it. Guards reject tiny/zero EPS (explosive estimates) and
        non-physical share counts; rejected names become NaN -> downstream value
        sub-factors score a neutral 50.
        """
        if "implied_shares" not in self._cache:
            f = self.fund_annual_latest()
            ni, eps = f["net_income"], f["eps_diluted"]
            shares = ni / eps.where(eps.abs() >= 0.01)
            shares = shares.where((shares > 1e6) & (shares < 1e11))
            # Prefer a reported value when present.
            reported = f["shares_outstanding"].where(f["shares_outstanding"] > 1e6)
            self._cache["implied_shares"] = reported.fillna(shares)
        return self._cache["implied_shares"]  # type: ignore[return-value]

    def market_cap(self) -> pd.Series:
        if "market_cap" not in self._cache:
            self._cache["market_cap"] = self.prices() * self.implied_shares()
        return self._cache["market_cap"]  # type: ignore[return-value]

    def enterprise_value(self) -> pd.Series:
        if "ev" not in self._cache:
            nd = self.fund_annual_latest()["net_debt"]
            self._cache["ev"] = self.market_cap() + nd
        return self._cache["ev"]  # type: ignore[return-value]

    # -- estimates / short / insider / institutional -----------------------
    def estimates(self) -> pd.DataFrame:
        if "estimates" not in self._cache:
            df = self._latest_per_ticker("analyst_estimates", "snapshot_date")
            if df.empty:
                self._note("analyst_estimates: no snapshot on/before the as-of date; "
                           "estimate-based value sub-factors scored neutral (50).")
            self._cache["estimates"] = df
        return self._cache["estimates"]  # type: ignore[return-value]

    def revisions(self) -> pd.DataFrame:
        if "revisions" not in self._cache:
            df = self._latest_per_ticker("analyst_revision_features", "snapshot_date")
            if df.empty:
                self._note("analyst_revision_features: no snapshot on/before the as-of "
                           "date; estimate-revision factor scored neutral (50).")
            self._cache["revisions"] = df
        return self._cache["revisions"]  # type: ignore[return-value]

    def estimate_features(self) -> pd.DataFrame:
        """Numeric consensus-estimate revision features (FMP), latest per ticker.

        Forward-accruing (EPS/revenue/price-target revisions need snapshot history
        to accumulate), so early on most columns are sparse; the revisions factor
        gates each on live coverage before using it.
        """
        if "estimate_features" not in self._cache:
            self._cache["estimate_features"] = self._latest_per_ticker(
                "analyst_estimate_features", "snapshot_date")
        return self._cache["estimate_features"]  # type: ignore[return-value]

    def short_interest(self) -> pd.DataFrame:
        if "short" not in self._cache:
            if self.reporting_lag and self.cutoff <= "9000-01-01":
                df = self._short_interest_pit()
            else:
                df = self._latest_per_ticker("short_interest", "date")
            if df.empty:
                self._note("short_interest: no snapshot on/before the as-of date; "
                           "short-interest factor scored neutral (50).")
            self._cache["short"] = df
        return self._cache["short"]  # type: ignore[return-value]

    def _short_interest_pit(self) -> pd.DataFrame:
        """Latest short-interest row per ticker, admitted by publication availability.

        ``date`` is the FINRA settlement date; FINRA does not disseminate the
        data until roughly two weeks later, so admitting a row on its settlement
        date would be look-ahead. Availability is the conservative
        ``date + lag_short_interest_days``.
        """
        avail_modifier = f"+{self.lag_short_interest_days} day"
        sql = """
            WITH usable AS (
                SELECT s.*, date(s.date, ?) AS avail_date
                FROM short_interest s
            ),
            latest AS (
                SELECT ticker, MAX(date) AS _d
                FROM usable WHERE avail_date <= ? GROUP BY ticker
            )
            SELECT u.* FROM usable u
            JOIN latest l ON u.ticker = l.ticker AND u.date = l._d
        """
        df = self.db.query_df(sql, (avail_modifier, self.cutoff))
        if df.empty:
            return pd.DataFrame(index=self.universe)
        df = df.drop_duplicates(subset="ticker", keep="last").set_index("ticker")
        return df.reindex(self.universe)

    def institutional(self) -> pd.DataFrame:
        if "institutional" not in self._cache:
            if self.reporting_lag and self.cutoff <= "9000-01-01":
                df = self._institutional_pit()
            else:
                df = self._latest_per_ticker("institutional_signals", "report_date")
            if df.empty:
                self._note("institutional_signals (13F): none filed (by filing_date) "
                           "on/before the as-of date; institutional factor scored neutral (50).")
            self._cache["institutional"] = df
        return self._cache["institutional"]  # type: ignore[return-value]

    def _institutional_pit(self) -> pd.DataFrame:
        """Latest 13F signal per ticker, admitted by filing availability.

        A 13F covering quarter-end R is not public until it is filed weeks later
        (the SEC deadline is 45 days after quarter end), so scoring on R itself
        would be look-ahead. The signals now come from FMP's whole-market
        ownership summary, which carries no per-filer filing date, so availability
        is the conservative regulatory deadline ``report_date + lag_quarterly_days``.
        """
        avail_modifier = f"+{self.lag_quarterly_days} day"
        sql = """
            WITH usable AS (
                SELECT s.*, date(s.report_date, ?) AS avail_date
                FROM institutional_signals s
            ),
            latest AS (
                SELECT ticker, MAX(report_date) AS _d
                FROM usable WHERE avail_date <= ? GROUP BY ticker
            )
            SELECT u.* FROM usable u
            JOIN latest l ON u.ticker = l.ticker AND u.report_date = l._d
        """
        df = self.db.query_df(sql, (avail_modifier, self.cutoff))
        if df.empty:
            return pd.DataFrame(index=self.universe)
        df = df.drop_duplicates(subset="ticker", keep="last").set_index("ticker")
        return df.reindex(self.universe)

    def insider_window(self) -> pd.DataFrame:
        """Insider transactions in the trailing window ending ``as_of``.

        Raw transaction rows (not aggregated) so factor logic can apply the
        per-code treatment (P positive, S mildly negative, A/M/F neutral).
        """
        if "insider" not in self._cache:
            # Window ends at as_of (PIT: historical scoring must not see later
            # transactions), additionally capped at "today" so a malformed future
            # transaction_date (the raw table spans 1988..2035) can never leak in
            # even if as_of is the far-future sentinel; floor at 1990 to drop
            # ancient junk.
            today = pd.Timestamp.today().date().isoformat()
            upper = min(self.as_of, today)
            start = (pd.Timestamp(upper) - pd.Timedelta(days=self.insider_window_days)).date().isoformat()
            start = max(start, "1990-01-01")
            df = self.db.query_df(
                "SELECT ticker, insider_name, insider_title, transaction_code, shares, "
                "price, value, transaction_date FROM insider_transactions "
                "WHERE transaction_date <= ? AND transaction_date >= ?",
                (upper, start),
            )
            self._cache["insider"] = df[df["ticker"].isin(self.universe)]
        return self._cache["insider"]  # type: ignore[return-value]

    def beneficial_ownership_window(self) -> pd.DataFrame:
        """13D/13G filings gated by ``filing_date`` (the real disclosure date —
        filed within 10 days of crossing 5%, so no additional PIT lag is
        needed on top of it, unlike short interest or 13F).

        ``filing_type`` ('13D'/'13G'/None) is parsed from the filing URL and
        only populated for filings since ~2019 (see
        :func:`data.beneficial_ownership._filing_type`); callers isolating
        real activist stakes (13D) from routine passive crossings (13G) must
        filter on it and accept the resulting survivorship to recent years.
        """
        if "beneficial_ownership" not in self._cache:
            df = self.db.query_df(
                "SELECT ticker, filing_date, reporting_person, reporting_person_type, "
                "filing_type, amount_beneficially_owned, percent_of_class "
                "FROM beneficial_ownership WHERE filing_date <= ?", (self.as_of,))
            self._cache["beneficial_ownership"] = (
                df[df["ticker"].isin(self.universe)] if not df.empty else df)
        return self._cache["beneficial_ownership"]  # type: ignore[return-value]

    def congressional_trades_window(self, lookback_days: int = 180) -> pd.DataFrame:
        """Senate/House trades in the trailing window ending ``as_of``, gated by
        ``disclosure_date`` (the STOCK Act allows up to 45d between a trade and
        its disclosure, so gating on ``transaction_date`` would be look-ahead)."""
        if "congressional_trades" not in self._cache:
            start = (pd.Timestamp(self.as_of) - pd.Timedelta(days=lookback_days)).date().isoformat()
            df = self.db.query_df(
                "SELECT ticker, chamber, member_id, transaction_type, transaction_date, "
                "disclosure_date, amount_range FROM congressional_trades "
                "WHERE disclosure_date <= ? AND disclosure_date >= ?",
                (self.as_of, start))
            self._cache["congressional_trades"] = (
                df[df["ticker"].isin(self.universe)] if not df.empty else df)
        return self._cache["congressional_trades"]  # type: ignore[return-value]

    # -- regime / crowding inputs ------------------------------------------
    def vix(self) -> float | None:
        row = self.db.query_one(
            "SELECT close FROM daily_prices WHERE ticker='VIX' AND date <= ? "
            "ORDER BY date DESC LIMIT 1",
            (self.cutoff,),
        )
        return float(row["close"]) if row and row["close"] is not None else None

    def price_matrix(self, lookback_days: int = 420) -> pd.DataFrame:
        """Adjusted-close matrix (date x ticker) over the trailing window.

        Cached and shared by the momentum factor (horizon returns) and any other
        consumer needing point-in-time price history.
        """
        if "price_matrix" not in self._cache:
            start = (pd.Timestamp(self.as_of) - pd.Timedelta(days=lookback_days)).date().isoformat()
            df = self.db.query_df(
                "SELECT ticker, date, adj_close FROM daily_prices "
                "WHERE date <= ? AND date >= ? ORDER BY date",
                (self.as_of, start),
            )
            if df.empty:
                self._cache["price_matrix"] = pd.DataFrame()
            else:
                wide = df.pivot_table(index="date", columns="ticker", values="adj_close")
                self._cache["price_matrix"] = wide.sort_index()
        return self._cache["price_matrix"]  # type: ignore[return-value]

    def horizon_return(self, trading_days: int, offset: int = 0) -> pd.Series:
        """Return over ``trading_days`` ending ``offset`` bars before ``as_of``.

        ``offset`` lets momentum skip the most recent month (the 12-1 construct)
        or measure a prior window (acceleration). NaN where history is too short.
        """
        px = self.price_matrix()
        if px.empty or len(px) <= trading_days + offset:
            return pd.Series(index=self.universe, dtype=float)
        end = px.iloc[-1 - offset]
        start = px.iloc[-1 - offset - trading_days]
        ret = (end / start) - 1.0
        return ret.reindex(self.universe)

    def daily_returns(self, lookback_days: int = 200) -> pd.DataFrame:
        """Daily simple returns (date x ticker) over the trailing window.

        Used by the crowding module to build synthetic factor return series from
        current quintile baskets.
        """
        start = (pd.Timestamp(self.as_of) - pd.Timedelta(days=lookback_days)).date().isoformat()
        df = self.db.query_df(
            "SELECT ticker, date, adj_close FROM daily_prices "
            "WHERE date <= ? AND date >= ? ORDER BY date",
            (self.as_of, start),
        )
        if df.empty:
            return pd.DataFrame()
        wide = df.pivot_table(index="date", columns="ticker", values="adj_close")
        wide = wide.reindex(columns=[t for t in self.universe if t in wide.columns])
        return wide.pct_change(fill_method=None).iloc[1:]


def col(df: pd.DataFrame, name: str) -> pd.Series:
    """A column from a (possibly empty/missing) frame as a numeric Series."""
    if df is None or df.empty or name not in df.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(df[name], errors="coerce")


def yoy_change(series: pd.Series) -> float:
    """Latest-minus-prior change of a per-period series (NaN if <2 points)."""
    vals = series.dropna().values
    if len(vals) < 2:
        return np.nan
    return float(vals[-1] - vals[-2])
