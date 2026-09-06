"""Fixed autoresearch evaluation harness (Karpathy-style propose/evaluate loop)
for the top-10 EQEFF holding-period strategy family (started from
05_sleeves_3M_monthly -- see research/strategies/ for the original 20-strategy
sweep this replaces).

FIXED HERE -- candidate.py cannot see or change any of this:
  - loading cached PIT composite scores + the candidate/price panel
  - development / validation date boundaries
  - the locked holdout window (never scored into research_score)
  - benchmark (SPY/QQQ) returns
  - T+1 execution (the same-close bug fixed 2026-09-05 in
    research/strategies/engine.py's simulate_calendar_sleeves is fixed here
    too, via the same reusable _lagged_prices helper)
  - transaction costs (10 bps/side)
  - portfolio-return calculation (independent of whatever candidate.py did
    internally -- only the returned target-weight table is trusted)
  - metric definitions (sharpe / cagr / max_dd / alpha_tstat / turnover / IR)
  - the research-score formula
  - correctness + leakage tests (run before every score is produced; a
    failing test aborts the run with no score printed)

ONLY research/autoresearch/candidate.py's generate_targets(context) may vary.

Contract enforced on candidate.py's output (see _validate_targets):
  columns {"date", "ticker", "weight"}; dates drawn only from the fixed
  rebal-date schedule handed to it via Context; weights >= 0 and summing to
  <= 1.0 per date (long-only, cash for unfilled slots). Nothing else about
  candidate.py's internals is trusted -- evaluate.py recomputes trades,
  costs and returns itself from that table alone.

Usage: python -m research.autoresearch.evaluate
"""
from __future__ import annotations

import hashlib
import importlib
import json
import subprocess
import sys
import types
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from backtesting.data_loader import SPY, QQQ                       # noqa: E402
from research.ablation.engine import alpha_tstat, _lagged_prices   # noqa: E402
from research.walkforward.portfolio import performance_metrics     # noqa: E402
from research.strategies.engine import load_data, StratData        # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS_TSV = HERE / "results.tsv"
CANDIDATE_PATH = HERE / "candidate.py"
TEST_FILE = REPO / "tests" / "test_autoresearch_evaluate.py"

# ---- fixed assumptions (candidate.py has no way to read or change these) -- #
COST_BPS = 10.0
EXEC_LAG_DAYS = 1            # T+1 execution
K_DEFAULT = 10

DEV_START, DEV_END = "2020-01-01", "2022-12-31"
VAL_START, VAL_END = "2023-01-01", "2024-12-31"
HOLDOUT_START = "2025-01-01"  # locked: computed for diagnostics, never gates research_score

RESULT_COLUMNS = [
    "timestamp", "candidate_sha256", "research_score", "dev_ir", "val_ir",
    "sharpe", "cagr", "beta", "alpha", "max_dd", "alpha_tstat", "turnover",
    "holdout_sharpe_diagnostic", "holdout_cagr_diagnostic", "n_dev", "n_val",
    "n_holdout", "notes",
]


# --------------------------------------------------------------------------- #
# The only thing candidate.py ever receives
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Context:
    """No prices. No returns. No performance metrics of any kind, for any
    period -- dev, val, or holdout alike. That is the entire leakage
    safeguard: there is no field here a candidate could read to learn how
    well anything performed, so there is nothing to overfit to except the
    market-data inputs (composite rank scores) a real trading rule would
    use anyway. `comp` is already point-in-time correct (see
    research/strategies/engine.py's load_data docstring: sub-factor
    selection + parent weights are re-derived every 6 months from an
    already-fully-realized trailing window) -- so exposing the full
    2020-01->2026-06 date range here is a market-data input, not a leak.

    Both fields are made read-only (MappingProxyType / non-writeable arrays)
    so a candidate cannot mutate shared state across calls."""
    rebal_dates: tuple
    comp: types.MappingProxyType
    k: int = K_DEFAULT


def _freeze_comp(comp: dict) -> types.MappingProxyType:
    frozen = {}
    for d, s in comp.items():
        s = s.copy()
        s.values.setflags(write=False)
        frozen[d] = s
    return types.MappingProxyType(frozen)


def build_context(data: StratData) -> Context:
    return Context(rebal_dates=tuple(sorted(data.comp)),
                   comp=_freeze_comp(data.comp), k=K_DEFAULT)


# --------------------------------------------------------------------------- #
# Loading candidate.py fresh (never a stale cached import)
# --------------------------------------------------------------------------- #
def load_candidate():
    if "research.autoresearch.candidate" in sys.modules:
        mod = importlib.reload(sys.modules["research.autoresearch.candidate"])
    else:
        mod = importlib.import_module("research.autoresearch.candidate")
    if not hasattr(mod, "generate_targets"):
        raise AttributeError("candidate.py must define generate_targets(context)")
    return mod


def candidate_sha256() -> str:
    return hashlib.sha256(CANDIDATE_PATH.read_bytes()).hexdigest()[:12]


# --------------------------------------------------------------------------- #
# Independent trade / return reconstruction -- candidate.py has no input here
# --------------------------------------------------------------------------- #
def _validate_targets(targets: pd.DataFrame, allowed_dates: set) -> pd.DataFrame:
    if not isinstance(targets, pd.DataFrame):
        raise TypeError("generate_targets must return a pandas DataFrame")
    required = {"date", "ticker", "weight"}
    missing = required - set(targets.columns)
    if missing:
        raise ValueError(f"generate_targets missing columns: {missing}")
    bad_dates = set(targets["date"]) - allowed_dates
    if bad_dates:
        raise ValueError("generate_targets used dates outside the fixed schedule "
                         f"handed to it via Context.rebal_dates: {sorted(bad_dates)[:5]}")
    if (targets["weight"] < -1e-9).any():
        raise ValueError("generate_targets returned a negative weight (long-only)")
    totals = targets.groupby("date")["weight"].sum()
    if (totals > 1.0 + 1e-6).any():
        raise ValueError("generate_targets returned weights summing above 1.0 on "
                         f"date(s) {totals[totals > 1.0 + 1e-6].index.tolist()[:5]} "
                         "(no leverage allowed)")
    return targets


def _snap_forward(index: pd.Index, date: str) -> str:
    """Nearest real trading day on/after `date` (some monthly rebal dates
    are calendar month-ends that land on a market holiday)."""
    pos = index.searchsorted(date)
    pos = min(pos, len(index) - 1)
    return index[pos]


def targets_to_weight_matrix(targets: pd.DataFrame, dates: list) -> pd.DataFrame:
    wide = targets.pivot_table(index="date", columns="ticker", values="weight",
                               aggfunc="sum", fill_value=0.0)
    return wide.reindex(dates, fill_value=0.0)


def compute_portfolio_returns(weights: pd.DataFrame, matrix: pd.DataFrame,
                              signal_dates: list, cost_bps: float = COST_BPS,
                              exec_lag_days: int = EXEC_LAG_DAYS
                              ) -> tuple[pd.Series, pd.Series]:
    """T+1 execution: the weight vector decided as of `signal_dates[i]` is
    only tradeable `exec_lag_days` real trading sessions later; the position
    then drifts with price (unrebalanced) until the next execution date's
    trade. Returns (net_period_return, turnover), both indexed by the
    signal date at which the period *starts* (so dev/val/holdout slicing
    uses the same date semantics as Context.rebal_dates)."""
    tickers = [t for t in weights.columns if t in matrix.columns]
    exec_dates = [_snap_forward(matrix.index, d) for d in signal_dates]
    exec_px = _lagged_prices(matrix[tickers], exec_dates, exec_lag_days)
    rets, turns = {}, {}
    prev_w = pd.Series(0.0, index=tickers)
    for i in range(len(signal_dates) - 1):
        d = signal_dates[i]
        w = weights.loc[d, tickers].reindex(tickers).fillna(0.0)
        turnover = float((w - prev_w).abs().sum())
        px0, px1 = exec_px.iloc[i], exec_px.iloc[i + 1]
        rel = (px1 / px0 - 1.0).fillna(0.0)
        gross = float((w * rel).sum())
        cost = turnover * cost_bps / 1e4
        rets[d] = gross - cost
        turns[d] = turnover
        prev_w = w
    return pd.Series(rets), pd.Series(turns)


def bench_returns(matrix: pd.DataFrame, ticker: str, signal_dates: list,
                  exec_lag_days: int = EXEC_LAG_DAYS) -> pd.Series:
    """Buy-and-hold benchmark return per period, labeled by the period's
    START date -- must match compute_portfolio_returns's convention
    (rets[signal_dates[i]] = return over [i, i+1]) exactly, or every
    downstream benchmark comparison (IR, beta, alpha, alpha_tstat) silently
    pairs each strategy period with the WRONG SPY/QQQ period."""
    exec_dates = [_snap_forward(matrix.index, d) for d in signal_dates]
    px = _lagged_prices(matrix[[ticker]], exec_dates, exec_lag_days)[ticker]
    r = px.pct_change(fill_method=None).dropna()
    r.index = signal_dates[:-1]
    return r


def _slice(s: pd.Series, start: str, end: str | None = None) -> pd.Series:
    mask = pd.Series(s.index) >= start
    if end is not None:
        mask &= pd.Series(s.index) <= end
    return s[mask.values]


def _ir(returns: pd.Series, bench: pd.Series) -> float:
    if returns.empty:
        return float("nan")
    m = performance_metrics(returns, hold_months=1, benchmarks={"SPY": bench})
    return m.get("spy_ir", float("nan"))


# --------------------------------------------------------------------------- #
# Correctness / leakage tests -- must pass before a score is ever printed
# --------------------------------------------------------------------------- #
def run_correctness_tests() -> bool:
    proc = subprocess.run([sys.executable, "-m", "pytest", str(TEST_FILE), "-q"],
                          cwd=REPO, capture_output=True, text=True)
    print(proc.stdout[-4000:])
    if proc.returncode != 0:
        print(proc.stderr[-4000:], file=sys.stderr)
    return proc.returncode == 0


# --------------------------------------------------------------------------- #
# Research score -- worst of dev/val IR, so a candidate can't win by being
# great in one era and mediocre in the other (discourages regime-specific
# overfitting to whichever of 2020-22 / 2023-24 it happens to fit better).
# --------------------------------------------------------------------------- #
def research_score(dev_ir: float, val_ir: float) -> float:
    if dev_ir != dev_ir or val_ir != val_ir:
        return float("nan")
    return min(dev_ir, val_ir)


def append_result(row: dict) -> None:
    is_new = not RESULTS_TSV.exists()
    with RESULTS_TSV.open("a") as fh:
        if is_new:
            fh.write("\t".join(RESULT_COLUMNS) + "\n")
        fh.write("\t".join(str(row.get(c, "")) for c in RESULT_COLUMNS) + "\n")


def main() -> int:
    print("[1/4] running correctness + leakage tests ...", flush=True)
    if not run_correctness_tests():
        print("CORRECTNESS TESTS FAILED -- aborting, no score produced.", file=sys.stderr)
        return 1

    print("[2/4] loading fixed PIT data (comp scores + price panel) ...", flush=True)
    data = load_data()
    context = build_context(data)
    allowed_dates = set(context.rebal_dates)

    print("[3/4] calling candidate.generate_targets(context) ...", flush=True)
    candidate = load_candidate()
    targets = candidate.generate_targets(context)
    targets = _validate_targets(targets, allowed_dates)

    print("[4/4] independently reconstructing trades/returns + scoring ...", flush=True)
    dates = sorted(allowed_dates)
    weights = targets_to_weight_matrix(targets, dates)
    pr, turnover = compute_portfolio_returns(weights, data.matrix, dates)
    spy = bench_returns(data.matrix, SPY, dates)
    qqq = bench_returns(data.matrix, QQQ, dates)

    dev_r, val_r, hold_r = (_slice(pr, DEV_START, DEV_END), _slice(pr, VAL_START, VAL_END),
                            _slice(pr, HOLDOUT_START))
    dev_spy, val_spy, hold_spy = (_slice(spy, DEV_START, DEV_END), _slice(spy, VAL_START, VAL_END),
                                  _slice(spy, HOLDOUT_START))

    dev_ir, val_ir = _ir(dev_r, dev_spy), _ir(val_r, val_spy)
    dev_val_r = pd.concat([dev_r, val_r]).sort_index()
    dev_val_spy = pd.concat([dev_spy, val_spy]).sort_index()
    dev_val_qqq = pd.concat([_slice(qqq, DEV_START, DEV_END), _slice(qqq, VAL_START, VAL_END)]).sort_index()
    m = performance_metrics(dev_val_r, hold_months=1, benchmarks={"SPY": dev_val_spy, "QQQ": dev_val_qqq})
    at = alpha_tstat(dev_val_r, dev_val_spy)
    turn_dev_val = float(pd.concat([_slice(turnover, DEV_START, DEV_END),
                                    _slice(turnover, VAL_START, VAL_END)]).mean())

    hold_m = performance_metrics(hold_r, hold_months=1, benchmarks={"SPY": hold_spy}) \
        if not hold_r.empty else {"sharpe": float("nan"), "cagr": float("nan")}

    score = research_score(dev_ir, val_ir)
    summary = {
        "research_score": score,
        "dev_ir": dev_ir,
        "val_ir": val_ir,
        "sharpe": m["sharpe"],
        "cagr": m["cagr"],
        "beta": m.get("spy_beta"),
        "alpha": m.get("spy_alpha"),
        "max_dd": m["max_drawdown"],
        "alpha_tstat": at,
        "turnover": turn_dev_val,
        "n_dev": int(len(dev_r)),
        "n_val": int(len(val_r)),
        "_diagnostic_only_not_gated": {
            "holdout_sharpe": hold_m.get("sharpe"),
            "holdout_cagr": hold_m.get("cagr"),
            "n_holdout": int(len(hold_r)),
            "note": "2025+ locked holdout; reported for human eyes only, "
                    "must never be used to accept/reject/rank candidates",
        },
    }
    print(json.dumps(summary, default=lambda x: None if x != x else x))

    append_result({
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "candidate_sha256": candidate_sha256(),
        "research_score": score, "dev_ir": dev_ir, "val_ir": val_ir,
        "sharpe": m["sharpe"], "cagr": m["cagr"], "beta": m.get("spy_beta"),
        "alpha": m.get("spy_alpha"), "max_dd": m["max_drawdown"],
        "alpha_tstat": at, "turnover": turn_dev_val,
        "holdout_sharpe_diagnostic": hold_m.get("sharpe"),
        "holdout_cagr_diagnostic": hold_m.get("cagr"),
        "n_dev": len(dev_r), "n_val": len(val_r), "n_holdout": len(hold_r),
        "notes": "",
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
