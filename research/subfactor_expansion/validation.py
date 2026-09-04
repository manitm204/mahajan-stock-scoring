"""Validation harness for the expanded candidate library.

Runs every check the proposal enumerated, in one pass over a
:class:`CandidatePanel` + a forward-return set:

* **coverage** — mean non-null share of the raw series across every rebalance
* **active_periods** — how many rebalance dates the candidate produced any data
* **distribution** — mean/median/std of the *raw* series and the *scored* series
* **neutral_50_share** — fraction of the scored frame that lands exactly on 50
  (the missing-data neutral); a high value means the sector-percentile step
  couldn't rank the name
* **ic_by_horizon** — Spearman rank IC vs forward returns at 1M / 3M / 6M / 12M
* **quintile_spread + monotonicity** — Q5-Q1 spread and rank correlation
* **hit_rate + year_stability** — hit rate of positive IC and calendar-year
  stability (fraction of years whose mean IC keeps the full-sample sign)
* **sign_sanity** — flags candidates whose sign flag disagrees with the raw
  correlation to forward returns (i.e. we labelled it ``higher_is_better=True``
  but the raw negatively correlates with 6M returns most of the time)

Deliverable: one long-form CSV per horizon and a wide "summary" CSV per
candidate, dropped in ``output/subfactor_expansion/``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from research.forward_returns import HORIZON_MONTHS, compute_forward_returns
from research.ic import ic_timeseries, summarize_ic
from research.quintiles import aggregate_quintiles

from .panel import CandidatePanel


# ---------------------------------------------------------------------------
# Coverage / distribution / active-periods
# ---------------------------------------------------------------------------
def coverage_report(panel: CandidatePanel) -> pd.DataFrame:
    """Per-candidate raw coverage, active periods, distribution stats."""
    rows: list[dict] = []
    for name in panel.all_candidates:
        raw_series: list[pd.Series] = []
        score_series: list[pd.Series] = []
        active_periods = 0
        for d in panel.rebal_dates:
            raw = panel.raws.get(d, pd.DataFrame()).get(name, None)
            score = panel.scores.get(d, pd.DataFrame()).get(name, None)
            if raw is not None:
                if raw.notna().any():
                    active_periods += 1
                raw_series.append(raw)
            if score is not None:
                score_series.append(score)
        raw_all = pd.concat(raw_series) if raw_series else pd.Series(dtype=float)
        score_all = pd.concat(score_series) if score_series else pd.Series(dtype=float)
        rows.append({
            "candidate": name,
            "parent": panel.parent_of(name),
            "coverage": float(raw_all.notna().mean()) if not raw_all.empty else 0.0,
            "active_periods": active_periods,
            "raw_mean": float(raw_all.mean()) if not raw_all.empty else float("nan"),
            "raw_median": float(raw_all.median()) if not raw_all.empty else float("nan"),
            "raw_std": float(raw_all.std()) if not raw_all.empty else float("nan"),
            "raw_p05": float(raw_all.quantile(0.05)) if raw_all.notna().any() else float("nan"),
            "raw_p95": float(raw_all.quantile(0.95)) if raw_all.notna().any() else float("nan"),
            "score_std": float(score_all.std()) if not score_all.empty else float("nan"),
            "neutral_50_share": float(((score_all.round(4) == 50.0).mean())) if not score_all.empty else float("nan"),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# IC (with hit rate + year-by-year stability) + quintiles
# ---------------------------------------------------------------------------
def ic_by_horizon(
    panel: CandidatePanel,
    fwd_returns: dict[str, dict[str, pd.Series]],
) -> pd.DataFrame:
    """Long-form IC by (candidate, horizon)."""
    rows: list[pd.DataFrame] = []
    for horizon, fwd_map in fwd_returns.items():
        ic_ts = ic_timeseries(panel, fwd_map, panel.all_candidates, horizon=horizon)
        summary = summarize_ic(ic_ts, panel.all_candidates)
        summary["horizon"] = horizon
        # Add year-by-year stability separately (production summarize_ic already
        # gives rolling-window stability; we add calendar-year stability).
        year_stab = _year_stability(ic_ts)
        summary = summary.merge(year_stab, on="signal", how="left")
        rows.append(summary)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    out.rename(columns={"signal": "candidate"}, inplace=True)
    out["parent"] = out["candidate"].map(panel.parent_of)
    return out


def _year_stability(ic_ts: pd.DataFrame) -> pd.DataFrame:
    """Fraction of calendar-year mean-ICs whose sign matches the full-sample sign."""
    if ic_ts.empty:
        return pd.DataFrame(columns=["signal", "year_stability", "years_observed"])
    ic_ts = ic_ts.copy()
    ic_ts["year"] = pd.to_datetime(ic_ts["date"]).dt.year
    rows: list[dict] = []
    for sig, grp in ic_ts.groupby("signal"):
        overall = grp["ic"].mean()
        if not np.isfinite(overall):
            rows.append({"signal": sig, "year_stability": float("nan"),
                         "years_observed": 0})
            continue
        yearly = grp.groupby("year")["ic"].mean()
        matches = int(((np.sign(yearly) == np.sign(overall)) & (yearly != 0)).sum())
        rows.append({
            "signal": sig,
            "year_stability": float(matches / len(yearly)) if len(yearly) else float("nan"),
            "years_observed": int(len(yearly)),
        })
    return pd.DataFrame(rows)


def quintile_report(
    panel: CandidatePanel,
    fwd_returns: dict[str, dict[str, pd.Series]],
) -> pd.DataFrame:
    """Q5-Q1 spread + monotonicity per candidate per horizon."""
    frames: list[pd.DataFrame] = []
    for horizon, fwd_map in fwd_returns.items():
        agg = aggregate_quintiles(panel, fwd_map, panel.all_candidates)
        if agg.empty:
            continue
        agg["horizon"] = horizon
        frames.append(agg)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out.rename(columns={"signal": "candidate"}, inplace=True)
    out["parent"] = out["candidate"].map(panel.parent_of)
    return out


# ---------------------------------------------------------------------------
# Sign sanity — does the declared direction match the observed correlation?
# ---------------------------------------------------------------------------
def sign_sanity(
    panel: CandidatePanel,
    fwd_returns: dict[str, dict[str, pd.Series]],
    horizon: str = "6M",
) -> pd.DataFrame:
    """For each candidate flag whether the raw ↔ forward-return correlation
    contradicts the declared ``higher_is_better`` direction.

    The check is on the *raw* series (not the scored one) so a wrong direction
    surfaces as a negative raw correlation, independent of the sector-percentile
    inversion. Signals with too few periods (< 6) get a NaN verdict.
    """
    rows: list[dict] = []
    fwd_map = fwd_returns.get(horizon, {})
    for name in panel.all_candidates:
        declared = panel.higher_by_candidate.get(name, True)
        per_period: list[float] = []
        for d, fwd in fwd_map.items():
            raw = panel.raws.get(d, pd.DataFrame()).get(name, None)
            if raw is None:
                continue
            df = pd.DataFrame({"r": raw, "f": fwd}).dropna()
            if len(df) < 20 or df["r"].nunique() < 2:
                continue
            per_period.append(float(df["r"].corr(df["f"], method="spearman")))
        if len(per_period) < 6:
            rows.append({
                "candidate": name, "parent": panel.parent_of(name),
                "declared_higher_is_better": declared, "raw_corr_mean": float("nan"),
                "raw_corr_hit_rate": float("nan"), "sign_agrees": None,
                "n_periods": len(per_period),
            })
            continue
        arr = np.asarray(per_period)
        mean_corr = float(arr.mean())
        expected_sign = 1.0 if declared else -1.0
        # For "higher is better" the raw ↔ forward-return correlation should be
        # positive on average. If declared ``lower_better`` (higher_is_better
        # False), we expect negative raw correlation.
        agrees = bool(np.sign(mean_corr) == expected_sign) if mean_corr != 0 else None
        rows.append({
            "candidate": name, "parent": panel.parent_of(name),
            "declared_higher_is_better": declared,
            "raw_corr_mean": mean_corr,
            "raw_corr_hit_rate": float((np.sign(arr) == expected_sign).mean()),
            "sign_agrees": agrees,
            "n_periods": len(per_period),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Combined summary
# ---------------------------------------------------------------------------
def _mean_ic_3m6m(ic: pd.DataFrame) -> pd.DataFrame:
    """Per-candidate **3M/6M mean IC** — the primary signal-strength metric shared
    with the parent selector — plus the two component ICs, from the long-form frame.

    Uses the mean of the 3M and 6M horizon mean-ICs: 1M is turnover-noisy and 12M
    overlaps heavily, so both are excluded from the headline strength read (they
    remain in ``ic_by_horizon.csv`` for reference)."""
    sub = ic[ic["horizon"].isin(["3M", "6M"])]
    if sub.empty:
        return pd.DataFrame(columns=["candidate", "mean_ic_3m6m", "ic_3M", "ic_6M"])
    piv = sub.pivot_table(index="candidate", columns="horizon", values="mean_ic")
    out = pd.DataFrame({"candidate": piv.index})
    out["ic_3M"] = piv["3M"].to_numpy() if "3M" in piv.columns else np.nan
    out["ic_6M"] = piv["6M"].to_numpy() if "6M" in piv.columns else np.nan
    out["mean_ic_3m6m"] = piv.reindex(columns=["3M", "6M"]).mean(axis=1, skipna=True).to_numpy()
    return out.reset_index(drop=True)


def combined_summary(
    coverage: pd.DataFrame,
    ic: pd.DataFrame,
    quintiles: pd.DataFrame,
    sign: pd.DataFrame,
    ic_horizon: str = "3M",
) -> pd.DataFrame:
    """Per-candidate wide summary at one focus horizon.

    The report uses this for the top/bottom-by-IC and low-coverage tables. The focus
    horizon supplies ``mean_ic``/``information_ratio``/spread etc.; on top of that we
    always attach ``mean_ic_3m6m`` (+ the ``ic_3M``/``ic_6M`` components) as the primary
    3M/6M signal-strength read used by the charts and the selector, independent of the
    focus horizon. Other horizons stay in the long-form ``ic`` / ``quintiles`` frames
    on disk; this is a compact merge for the ranking view.
    """
    ic_focus = ic[ic["horizon"] == ic_horizon].drop(columns=["horizon"])
    q_focus = quintiles[quintiles["horizon"] == ic_horizon].drop(columns=["horizon", "parent"], errors="ignore")
    merged = (coverage.merge(ic_focus, on=["candidate", "parent"], how="left")
              .merge(q_focus, on="candidate", how="left")
              .merge(sign[["candidate", "raw_corr_mean", "raw_corr_hit_rate",
                           "sign_agrees"]], on="candidate", how="left")
              .merge(_mean_ic_3m6m(ic), on="candidate", how="left"))
    return merged


# ---------------------------------------------------------------------------
# End-to-end runner
# ---------------------------------------------------------------------------
def run_validation(
    panel: CandidatePanel,
    price_matrix: pd.DataFrame,
    out_dir: Path,
    ic_horizon: str = "3M",
    horizons: dict[str, int] | None = None,
) -> dict[str, pd.DataFrame]:
    """Produce every validation frame and write CSVs. Returns the frames dict."""
    out_dir.mkdir(parents=True, exist_ok=True)
    horizons = horizons or HORIZON_MONTHS

    fwd = compute_forward_returns(price_matrix, panel.rebal_dates, horizons=horizons)

    coverage = coverage_report(panel)
    coverage.to_csv(out_dir / "coverage.csv", index=False)

    ic = ic_by_horizon(panel, fwd)
    ic.to_csv(out_dir / "ic_by_horizon.csv", index=False)

    quintiles = quintile_report(panel, fwd)
    quintiles.to_csv(out_dir / "quintiles_by_horizon.csv", index=False)

    sign = sign_sanity(panel, fwd, horizon=ic_horizon)
    sign.to_csv(out_dir / "sign_sanity.csv", index=False)

    summary = combined_summary(coverage, ic, quintiles, sign, ic_horizon=ic_horizon)
    summary.to_csv(out_dir / f"summary_{ic_horizon}.csv", index=False)

    return {"coverage": coverage, "ic": ic, "quintiles": quintiles,
            "sign": sign, "summary": summary, "fwd": fwd}  # type: ignore[dict-item]
