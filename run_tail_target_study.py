"""Rank ICs of the regime factors against short-call-spread risk targets.

Reuses the multi-fund panels (SPY / QQQ / IWM, 2005-2026, weekly grid) and the
overlap-aware block-bootstrap Spearman from the fund signal study, but swaps
the target from the mean forward return to three quantities that matter when
you are short upside:

  breach      1 if the fund's max close over the next 21 trading days exceeds
              close_t * (1 + 1.2 * vol_t/100 * sqrt(21/252)) — i.e. price
              pushed through ~1.2 implied monthly standard deviations at any
              point, where vol_t is the fund's own implied vol index
              (VIX / VXN / RVX)
  tail_size   max(close_{t+1..t+21}) / close_t - 1, the continuous version
  vol_norm    fwd 21d total return / (vol_t/100 * sqrt(21/252)) — the move in
              implied-standard-deviation units, so a factor that only predicts
              returns *because* it tracks vol collapses here

Breach and tail use price-only closes (options settle on the price index);
vol_norm reuses the study's dividend-adjusted forward return so its IC is
comparable to ic_mean_return. The 1.2 multiplier is fixed, not tuned; 1.0 and
1.5 are reported once as robustness checks, not searched.

    python run_tail_target_study.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

from spystudy import stats as st
from spystudy.data import FWD_DAYS
from spystudy.multi import build_panels

OUT = Path("output/tail_target_study")
PRIOR = Path("output/fund_signal_study")
MULT = 1.2
ROBUST = [1.0, 1.5]
SQRT_M = np.sqrt(FWD_DAYS / 252)

FACTORS = [("rsi14", "RSI(14)"), ("sector_corr_60d", "Sector corr 60d"),
           ("trend_4m", "4-month trend"), ("ma200_dist", "Price vs MA200"),
           ("absorption_shift", "Absorption shift"), ("vix", "Own vol index"),
           ("vrp", "VRP (IV − realised)"), ("ivr", "IV Rank"),
           ("beta_60", "Beta to SPY (60d)")]


def add_targets(w: pd.DataFrame, px: pd.Series) -> pd.DataFrame:
    """Attach breach / tail_size / vol_norm to a fund's weekly panel."""
    # rolling(21).max() at t+21 covers closes t+1..t+21; shift(-21) puts it at t
    fwd_max = px.rolling(FWD_DAYS).max().shift(-FWD_DAYS)
    tail = (fwd_max / px - 1.0).reindex(w.index)
    sigma_m = w["vix"] / 100.0 * SQRT_M
    out = w.copy()
    out["tail_size"] = tail
    out["vol_norm"] = w["fwd_abs"] / sigma_m
    for m in [MULT] + ROBUST:
        out[f"breach_{m:.1f}"] = (tail > m * sigma_m).where(
            tail.notna() & sigma_m.notna()).astype(float)
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    panels, close = build_panels()
    robust_rows = []
    for fund, w in panels.items():
        w = add_targets(w, close[fund])
        prior = pd.read_csv(PRIOR / f"{fund}_abs_levels.csv").set_index("signal")
        base = w[f"breach_{MULT:.1f}"].mean()
        print(f"\n=== {fund}: breach base rate {base * 100:.1f}%  "
              f"(1.0x {w['breach_1.0'].mean() * 100:.1f}%, "
              f"1.5x {w['breach_1.5'].mean() * 100:.1f}%)  "
              f"median tail {w['tail_size'].median() * 100:+.2f}%")
        rows = []
        for col, label in FACTORS:
            br = st.rank_ic(w[col], w[f"breach_{MULT:.1f}"])
            tl = st.rank_ic(w[col], w["tail_size"])
            vn = st.rank_ic(w[col], w["vol_norm"])
            rows.append({"factor": col,
                         "ic_mean_return": prior.loc[col, "rho"],
                         "ic_breach": br["rho"], "ic_tail_size": tl["rho"],
                         "ic_vol_norm_return": vn["rho"], "n_obs": br["n"]})
            robust_rows.append({"fund": fund, "factor": col,
                                "ic_breach_1.0": st.rank_ic(w[col], w["breach_1.0"])["rho"],
                                "ic_breach_1.2": br["rho"],
                                "ic_breach_1.5": st.rank_ic(w[col], w["breach_1.5"])["rho"]})
            star = lambda r: "*" if r["p"] < 0.05 else " "
            print(f"  {label:22s} breach {br['rho']:+.3f}{star(br)} "
                  f"tail {tl['rho']:+.3f}{star(tl)} "
                  f"volnorm {vn['rho']:+.3f}{star(vn)} (p={vn['p']:.3f})")
        pd.DataFrame(rows).to_csv(OUT / f"{fund}_tail_targets.csv", index=False)
    pd.DataFrame(robust_rows).to_csv(OUT / "breach_robustness.csv", index=False)
    print(f"\nWrote {OUT}/")


if __name__ == "__main__":
    main()
