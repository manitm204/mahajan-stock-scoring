"""Consolidated baselines leaderboard (Phase B of the 2026-07-18 gap-filling program).

One table, common period (2017 -> mid-2026), comparing on equal footing:
  - SPY / QQQ buy-and-hold
  - PIT equal-weight S&P universe (top_pct=1.0, ew)
  - PIT cap-weight S&P universe (top_pct=1.0, cap)  [SPY replication check]
  - the locked finalist (top25% / cap5 / cap_match / VIX tilt), monthly engine
  - finalist sensitivities: T+1 delayed execution, 25 bps/side costs

Pre-tax leg: research/ablation engine, net of 10 bps/side unless varied.
After-tax leg: 13-month hold, 4 staggered sleeves, lot-level HIFO, ST 32%/LT 15%
(scripts/finalist_toolkit/tax_rebalance.py machinery).

Outputs: output/leaderboard/baselines.csv + BASELINES.md
"""
from __future__ import annotations

import importlib.util
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd

REPO = Path("/home/manit/Desktop/fun_projects/mahajan_hedge_fund")
sys.path.insert(0, str(REPO))

from run_walkforward import PANEL_START, PRICE_END, _load_panel, get_db  # noqa: E402
from research.ablation import load_ablation_data  # noqa: E402
from research.ablation.engine import (AblationConfig, benchmark_row,  # noqa: E402
                                      simulate_config)

_spec = importlib.util.spec_from_file_location(
    "tax_rebalance", REPO / "scripts/finalist_toolkit/tax_rebalance.py")
tr = importlib.util.module_from_spec(_spec)
sys.modules["tax_rebalance"] = tr
_spec.loader.exec_module(tr)

OUT = REPO / "output/leaderboard"

PRETAX_CONFIGS = [
    AblationConfig("EW-universe(PIT)", top_pct=1.0, weighting="ew"),
    AblationConfig("CAP-universe(PIT)", top_pct=1.0, weighting="cap"),
    AblationConfig("finalist(25/cap5/capmatch/vix)", top_pct=0.25, weighting="cap5",
                   sector="cap_match", vix_tilt=True),
]
FINALIST = PRETAX_CONFIGS[-1]
SENSITIVITIES = [
    replace(FINALIST, name="finalist T+1", exec_lag_days=1),
    replace(FINALIST, name="finalist 25bps", cost_bps=25.0),
    replace(FINALIST, name="finalist 0bps", cost_bps=0.0),
]

AFTERTAX_CONFIGS = [
    ("EW-universe(PIT)", AblationConfig("ew100", top_pct=1.0, weighting="ew",
                                        vix_tilt=True)),
    ("CAP-universe(PIT)", AblationConfig("cap100", top_pct=1.0, weighting="cap",
                                         vix_tilt=True)),
    ("finalist(25/cap5/capmatch/vix)",
     AblationConfig("fin", top_pct=0.25, weighting="cap5", sector="cap_match",
                    vix_tilt=True)),
]


def main() -> None:
    panel = _load_panel(False)
    with get_db() as db:
        data = load_ablation_data(panel, db, PANEL_START, PRICE_END,
                                  splits="rolling5y")
    OUT.mkdir(parents=True, exist_ok=True)

    rows = [benchmark_row(data, "SPY"), benchmark_row(data, "QQQ")]
    for cfg in PRETAX_CONFIGS + SENSITIVITIES:
        print(f"[pretax] {cfg.name} ...", flush=True)
        rows.append(simulate_config(data, cfg))
    pre = pd.DataFrame(rows)
    pre.to_csv(OUT / "baselines_pretax.csv", index=False)

    print("[aftertax] 13mo hold / 4 sleeves / HIFO / ST32-LT15 ...", flush=True)
    matrix = data.matrix
    at_rows = []
    for label, cfg in AFTERTAX_CONFIGS:
        wts = tr.weights_for(data, cfg)
        dates = [d for d in data.rebal_dates if d in wts and d in matrix.index]
        r = tr.run_config(dates, wts, matrix, tr.HOLD_M, tr.SLEEVES)
        at_rows.append({"config": label, "pre_tax_cagr": r["pre_cagr"],
                        "after_tax_cagr": r["after_cagr"],
                        "walkaway_after_tax": r["after_tax"],
                        "total_tax": r["tax_st"] + r["tax_lt"],
                        "pct_lt_gain": r["gain_lt"]
                        / max(r["gain_st"] + r["gain_lt"], 1e-9),
                        "turnover_yr": r["turn"]})
        print(f"  {label}: after-tax CAGR {r['after_cagr']:.1%}", flush=True)
    dref = [d for d in data.rebal_dates if d in matrix.index]
    for tkr in ("SPY", "QQQ"):
        b = tr.bench(matrix, dref, tkr)
        at_rows.append({"config": f"{tkr} buy&hold", "pre_tax_cagr": b["pre_cagr"],
                        "after_tax_cagr": b["after_cagr"],
                        "walkaway_after_tax": b["after_tax"],
                        "total_tax": b["tax_lt"], "pct_lt_gain": 1.0,
                        "turnover_yr": 0.0})
    at = pd.DataFrame(at_rows)
    at.to_csv(OUT / "baselines_aftertax.csv", index=False)

    cols = ["config", "gross_cagr", "net_cagr", "sharpe", "sortino", "calmar",
            "max_dd", "beta", "alpha", "alpha_t", "ex_spy", "ex_spy_e1", "ex_spy_e2",
            "beta_qqq", "alpha_qqq", "ex_qqq", "alpha_search", "alpha_holdout",
            "turnover", "avg_names", "eff_n"]
    with (OUT / "BASELINES.md").open("w") as fh:
        fh.write("# Consolidated baselines (2026-07-18)\n\n"
                 f"Common period {data.rebal_dates[0]} -> {data.rebal_dates[-1]}, "
                 "monthly grid. Pre-tax = ablation engine net of stated costs "
                 "(10 bps/side unless varied). adj_close total-return proxy; "
                 "dividends compound untaxed (favours all rows equally, slightly "
                 "favours benchmarks after tax). 2024+ columns are the DISCOUNTED "
                 "holdout (consulted in prior selection — not a pristine gate).\n\n"
                 "## Pre-tax (monthly rebalance)\n\n")
        fh.write(pre[[c for c in cols if c in pre.columns]]
                 .to_markdown(index=False, floatfmt=".3f"))
        fh.write("\n\n## After-tax (13-month hold, 4 sleeves, HIFO, ST 32%/LT 15%)\n\n")
        fh.write(at.to_markdown(index=False, floatfmt=".3f"))
        fh.write("\n")
    print(f"\nwrote {OUT}/baselines_pretax.csv, baselines_aftertax.csv, BASELINES.md",
          flush=True)


if __name__ == "__main__":
    main()
