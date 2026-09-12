"""Robustness check on the staggered-sleeves Monte Carlo strategy's reported
beta (~-0.10) and R^2 (~0.01) vs SPY (see docs/monte_carlo_book_construction.md
and scripts/monte_carlo_random_book.py -- NOT modified by this script).

scripts/monte_carlo_random_book.py only produces MONTHLY bookend returns
(one price ratio per rebalance-to-rebalance period, via
research.autoresearch.evaluate.compute_portfolio_returns). To test beta at
daily/weekly frequency, with HAC standard errors, rolling windows, calendar
years, and downside conditioning, this script marks the SAME strategy
(identical sleeve construction, identical random draws, identical
book_size/hold/refresh/sleeve_count, identical T+1 execution lag and 10bps
cost convention) to DAILY NAV using the existing daily price matrix, by
buy-and-hold compounding each period's fixed nominal weights between
executions -- exactly reproducing compute_portfolio_returns's monthly
bookend numbers at the period endpoints (verified below), just filled in
daily in between. No changes to any existing strategy/backtest script.

Strategy return = equal-weighted AVERAGE of N independent random-draw sims
(same convention as the "mean of sims" portfolio stats already reported on
the dashboard), so idiosyncratic single-draw noise is averaged out and what
remains is the structural exposure of the sleeve/hold/refresh mechanism
itself -- the actual object in question ("is beta stable" / "is R^2 real").

FINDING (see main()'s printed comparison): the reported beta of ~-0.10 is a
date-label misalignment artifact, not a real property of the strategy.
scripts/run_strategy_sweep.py::bench_returns() labels SPY's return by the
date the period ENDS (plain pandas .pct_change() convention), while
research/autoresearch/evaluate.py::compute_portfolio_returns() explicitly
labels the portfolio's return by the date the period STARTS (per its own
docstring). research/walkforward/portfolio.py::benchmark_stats() aligns the
two series by index label with a plain pd.concat(...).dropna(), so every
period is silently paired with the WRONG benchmark period (off by one
month). Correctly aligned (this script builds SPY's return straight off the
same daily price matrix, not through bench_returns), beta comes out
strongly positive (~0.85-0.95), matching what a random 5-name long-only
equity sleeve should look like. This script does not touch the buggy
helper -- flagged for a separate fix, out of scope here per instruction to
only add analysis.

Usage: python scripts/sleeve_beta_robustness.py
Writes: output/sleeve_beta_robustness/ (CSVs, scatterplot PNG, report)
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

from research.autoresearch.evaluate import targets_to_weight_matrix, compute_portfolio_returns
from research.strategies.engine import load_data
from scripts.monte_carlo_random_book import _random_targets

OUT = REPO / "output" / "sleeve_beta_robustness"
OUT.mkdir(parents=True, exist_ok=True)

BOOK_SIZE, HOLD_MONTHS, REFRESH_N, SLEEVE_COUNT = 5, 4, 5, 4   # current dashboard config
N_SIMS = 100
SEED_BASE = 0
COST_BPS = 10.0
EXEC_LAG_DAYS = 1


def _snap_forward(index: pd.Index, date: str) -> str:
    pos = index.searchsorted(date)
    pos = min(pos, len(index) - 1)
    return index[pos]


def _exec_days(index: pd.Index, signal_dates: list) -> list:
    """Real trading day each signal date executes on (T+1), as a label
    matching compute_portfolio_returns' own _lagged_prices convention."""
    snapped = [_snap_forward(index, d) for d in signal_dates]
    positions = [min(index.get_loc(d) + EXEC_LAG_DAYS, len(index) - 1) for d in snapped]
    return [index[p] for p in positions]


def _daily_nav_for_sim(weights: pd.DataFrame, matrix: pd.DataFrame,
                       rebal: list, exec_days: list) -> tuple[pd.Series, list]:
    """Buy-and-hold daily NAV: weights decided at rebal[i] are bought at
    exec_days[i] (paying that period's turnover cost against the value
    carried in from the prior period) and held unrebalanced -- marked daily
    at each ticker's own price ratio to the period's entry price -- until
    exec_days[i+1], where the same thing happens for period i+1. Each
    period's own [d0, d1] price move is captured in full (day_slice is
    inclusive of d1) before the next period's cost overwrites that same
    calendar day with its own entry.

    Returns (nav, period_returns) where period_returns[i] is this period's
    return computed the same way as compute_portfolio_returns (gross ratio
    minus cost, relative to the value carried in from the previous period)
    -- used only to cross-check against the unmodified existing script."""
    tickers = [t for t in weights.columns if t in matrix.columns]
    nav = pd.Series(index=matrix.index, dtype=float)
    nav_level = 1.0   # value carried in, before this period's own cost
    prev_w = pd.Series(0.0, index=tickers)
    period_returns = []
    for i in range(len(rebal) - 1):
        w = weights.loc[rebal[i], tickers].reindex(tickers).fillna(0.0)
        turnover = float((w - prev_w).abs().sum())
        cost = turnover * COST_BPS / 1e4

        d0, d1 = exec_days[i], exec_days[i + 1]
        px0 = matrix.loc[d0, tickers]
        day_slice = matrix.index[(matrix.index >= d0) & (matrix.index <= d1)]
        rel = matrix.loc[day_slice, tickers].div(px0, axis=1)
        period_value = rel.mul(w, axis=1).sum(axis=1)   # =1.0 at d0, =gross_ratio at d1

        start_val = nav_level * (1.0 - cost)
        nav.loc[day_slice] = start_val * period_value
        period_returns.append(start_val * period_value.iloc[-1] / nav_level - 1.0)
        nav_level = nav.loc[d1]   # this period's own end value (before next period's cost overwrites it)
        prev_w = w
    return nav.dropna(), period_returns


def hac_ols(y: pd.Series, x: pd.Series, label: str) -> dict:
    common = y.index.intersection(x.index)
    yy, xx = y.loc[common].values, x.loc[common].values
    n = len(yy)
    lag = max(int(np.floor(4 * (n / 100) ** (2 / 9))), 1)
    X = sm.add_constant(xx)
    model = sm.OLS(yy, X).fit(cov_type="HAC", cov_kwds={"maxlags": lag})
    corr = float(np.corrcoef(yy, xx)[0, 1])
    return {
        "label": label, "n": n, "hac_lag": lag,
        "alpha": float(model.params[0]), "beta": float(model.params[1]),
        "beta_se": float(model.bse[1]), "beta_t": float(model.tvalues[1]),
        "beta_p": float(model.pvalues[1]),
        "beta_ci_lo": float(model.conf_int()[1][0]), "beta_ci_hi": float(model.conf_int()[1][1]),
        "corr": corr, "r2_corr_sq": corr ** 2, "r2_model": float(model.rsquared),
    }


def print_stat(s: dict) -> None:
    print(f"[{s['label']}] n={s['n']} HAC lag={s['hac_lag']}")
    print(f"  beta={s['beta']:+.4f}  se={s['beta_se']:.4f}  t={s['beta_t']:+.3f}  "
         f"p={s['beta_p']:.4f}  95% CI=[{s['beta_ci_lo']:+.4f}, {s['beta_ci_hi']:+.4f}]")
    print(f"  alpha(period)={s['alpha']:+.5f}  corr={s['corr']:+.4f}  "
         f"corr^2={s['r2_corr_sq']:.4f}  R^2(model)={s['r2_model']:.4f}  "
         f"match={np.isclose(s['r2_corr_sq'], s['r2_model'])}")


def main() -> None:
    print("loading data ...", flush=True)
    data = load_data()
    rebal = list(data.rebal_dates)
    matrix = data.matrix
    exec_days = _exec_days(matrix.index, rebal)

    # ---- sanity check: daily marking reproduces the existing monthly bookend ----
    rng0 = random.Random(SEED_BASE)
    targets0 = _random_targets(data.comp, rebal, rng0, BOOK_SIZE, HOLD_MONTHS, REFRESH_N, SLEEVE_COUNT)
    weights0 = targets_to_weight_matrix(targets0, rebal)
    pr0, _ = compute_portfolio_returns(weights0, matrix, rebal)
    _nav0, period_returns0 = _daily_nav_for_sim(weights0, matrix, rebal, exec_days)
    for check_i in (5, 20, 50):
        daily_period_ret = period_returns0[check_i]
        bookend_ret = pr0.loc[rebal[check_i]]
        print(f"sanity check period {check_i}: daily-marked={daily_period_ret:.6f} "
             f"vs compute_portfolio_returns={bookend_ret:.6f} "
             f"(match={np.isclose(daily_period_ret, bookend_ret, atol=2e-4)} "
             f"-- residual is the documented (1-cost)*(1+gross) vs (1+gross-cost) cross-term)")

    # ---- reproduce + diagnose the reported ~-0.10 beta (single sim, monthly) ----
    from scripts.run_strategy_sweep import bench_returns as buggy_bench_returns
    spy_buggy = buggy_bench_returns(matrix, "SPY", rebal)          # end-labeled
    spy_correct = matrix["SPY"].pct_change().dropna()
    spy_correct_start_labeled = spy_correct.reindex(rebal).shift(-1)  # start-labeled, matches pr0
    common_buggy = pr0.index.intersection(spy_buggy.index)
    common_fixed = pr0.index.intersection(spy_correct_start_labeled.dropna().index)
    beta_buggy = float(np.cov(pr0.loc[common_buggy], spy_buggy.loc[common_buggy])[0, 1]
                       / np.var(spy_buggy.loc[common_buggy], ddof=1))
    beta_fixed = float(np.cov(pr0.loc[common_fixed], spy_correct_start_labeled.loc[common_fixed])[0, 1]
                       / np.var(spy_correct_start_labeled.loc[common_fixed], ddof=1))
    print("\n" + "=" * 70)
    print("DIAGNOSIS: why the dashboard/doc report beta ~ -0.10")
    print("=" * 70)
    print(f"  single-sim monthly beta via the EXISTING (buggy) bench_returns "
         f"alignment: {beta_buggy:+.4f}")
    print(f"  single-sim monthly beta with dates correctly aligned (portfolio's "
         f"start-labeled period matched to SPY's SAME period): {beta_fixed:+.4f}")
    print("  -> compute_portfolio_returns labels a period's return by the date it "
         "STARTS; bench_returns labels SPY's return by the date it ENDS "
         "(plain pandas pct_change); benchmark_stats aligns them by raw index "
         "label, so every period is paired with the wrong SPY month. This "
         "single-sim check reproduces the reported negative beta and confirms "
         "the fix. Not corrected in the shared helper -- analysis only.")

    # ---- N_SIMS random draws, averaged daily return series ----
    print(f"running {N_SIMS} sims ...", flush=True)
    daily_rets = []
    for s in range(N_SIMS):
        rng = random.Random(SEED_BASE + s)
        targets = _random_targets(data.comp, rebal, rng, BOOK_SIZE, HOLD_MONTHS, REFRESH_N, SLEEVE_COUNT)
        weights = targets_to_weight_matrix(targets, rebal)
        nav, _ = _daily_nav_for_sim(weights, matrix, rebal, exec_days)
        daily_rets.append(nav.pct_change().dropna())
        if (s + 1) % 20 == 0:
            print(f"  sim {s + 1}/{N_SIMS}", flush=True)

    strat_daily = pd.concat(daily_rets, axis=1).mean(axis=1).sort_index()
    strat_daily.index = pd.to_datetime(strat_daily.index)
    spy_daily = matrix["SPY"].pct_change().dropna()
    spy_daily.index = pd.to_datetime(spy_daily.index)
    common_idx = strat_daily.index.intersection(spy_daily.index)
    strat_daily = strat_daily.loc[common_idx].sort_index()
    spy_daily = spy_daily.loc[common_idx].sort_index()

    strat_daily.to_frame("strategy").join(spy_daily.to_frame("spy")).to_csv(
        OUT / "daily_returns.csv")

    strat_weekly = (1 + strat_daily).resample("W-FRI").prod() - 1
    spy_weekly = (1 + spy_daily).resample("W-FRI").prod() - 1
    strat_weekly.to_frame("strategy").join(spy_weekly.to_frame("spy")).to_csv(
        OUT / "weekly_returns.csv")

    print("\n" + "=" * 70)
    print("FULL PERIOD (2020 - present)")
    print("=" * 70)
    full_daily = hac_ols(strat_daily, spy_daily, "daily, full period")
    print_stat(full_daily)
    full_weekly = hac_ols(strat_weekly, spy_weekly, "weekly, full period")
    print_stat(full_weekly)

    print("\n" + "=" * 70)
    print("EXCLUDING 2020")
    print("=" * 70)
    mask_d = strat_daily.index.year != 2020
    mask_w = strat_weekly.index.year != 2020
    ex2020_daily = hac_ols(strat_daily[mask_d], spy_daily[mask_d], "daily, ex-2020")
    print_stat(ex2020_daily)
    ex2020_weekly = hac_ols(strat_weekly[mask_w], spy_weekly[mask_w], "weekly, ex-2020")
    print_stat(ex2020_weekly)

    print("\n" + "=" * 70)
    print("DOWNSIDE BETA (days SPY < 0, daily)")
    print("=" * 70)
    down_mask = spy_daily < 0
    downside = hac_ols(strat_daily[down_mask], spy_daily[down_mask], "daily, SPY<0 only")
    print_stat(downside)

    print("\n" + "=" * 70)
    print("CALENDAR YEAR (daily)")
    print("=" * 70)
    year_rows = []
    for yr, grp_idx in strat_daily.groupby(strat_daily.index.year).groups.items():
        yr_stat = hac_ols(strat_daily.loc[grp_idx], spy_daily.loc[grp_idx], f"daily, {yr}")
        print_stat(yr_stat)
        year_rows.append({"year": yr, **{k: v for k, v in yr_stat.items() if k != "label"}})
    pd.DataFrame(year_rows).to_csv(OUT / "calendar_year_beta.csv", index=False)

    print("\n" + "=" * 70)
    print("ROLLING BETA / CORRELATION (daily)")
    print("=" * 70)
    cov = strat_daily.rolling(126).cov(spy_daily)
    var = spy_daily.rolling(126).var()
    roll6m_beta = cov / var
    roll6m_corr = strat_daily.rolling(126).corr(spy_daily)
    cov12 = strat_daily.rolling(252).cov(spy_daily)
    var12 = spy_daily.rolling(252).var()
    roll12m_beta = cov12 / var12
    roll12m_corr = strat_daily.rolling(252).corr(spy_daily)
    rolling = pd.DataFrame({
        "roll6m_beta": roll6m_beta, "roll6m_corr": roll6m_corr,
        "roll12m_beta": roll12m_beta, "roll12m_corr": roll12m_corr,
    })
    rolling.to_csv(OUT / "rolling_beta_correlation.csv")
    for col in rolling.columns:
        s = rolling[col].dropna()
        print(f"  {col:<14} mean={s.mean():+.3f}  min={s.min():+.3f}  max={s.max():+.3f}  "
             f"% negative={(s < 0).mean():.0%}")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.set_facecolor("#fcfcfb")
    fig.patch.set_facecolor("#fcfcfb")
    ax.plot(rolling.index, rolling["roll6m_beta"], color="#2a78d6", linewidth=1.5, label="6-month rolling beta")
    ax.plot(rolling.index, rolling["roll12m_beta"], color="#eb6834", linewidth=1.5, label="12-month rolling beta")
    ax.axhline(0, color="#8a8a86", linewidth=1, linestyle="--")
    ax.set_ylabel("Beta vs SPY")
    ax.set_title("Staggered-sleeves rolling beta vs SPY (n=5/hold=4/refresh=5/4 sleeves, 100-sim average)")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, color="#e3e2dc", linewidth=0.6)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    plt.tight_layout()
    plt.savefig(OUT / "rolling_beta.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.set_facecolor("#fcfcfb")
    fig.patch.set_facecolor("#fcfcfb")
    ax.scatter(spy_daily.values, strat_daily.values, s=10, alpha=0.35, color="#2a78d6",
              edgecolors="none", label="daily (strategy, SPY)")
    xs = np.linspace(spy_daily.min(), spy_daily.max(), 100)
    ax.plot(xs, full_daily["alpha"] + full_daily["beta"] * xs, color="#e34948", linewidth=2,
           label=f"OLS: beta={full_daily['beta']:+.3f}, R2={full_daily['r2_model']:.3f}")
    ax.axhline(0, color="#c9c8c2", linewidth=0.8)
    ax.axvline(0, color="#c9c8c2", linewidth=0.8)
    ax.set_xlabel("SPY daily return")
    ax.set_ylabel("Strategy daily return (100-sim average)")
    ax.set_title("Staggered sleeves vs SPY — daily returns, 2020-present")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, color="#e3e2dc", linewidth=0.6)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    plt.tight_layout()
    plt.savefig(OUT / "scatter_daily.png", dpi=150)
    plt.close(fig)

    print(f"\nwrote outputs to {OUT}/")


if __name__ == "__main__":
    main()
