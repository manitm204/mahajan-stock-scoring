from __future__ import annotations

import pandas as pd
import pytest

from research.walkforward.regime_probability import (
    REGIME_ORDER, SHRINKAGE_K, shrink_regime_ic,
)
from run_regime_aware_study import (
    REPORT_METRIC_COLS, _recommendation, _shrunk_regime_summary,
    write_churn_chart, write_report,
)


def _toy_comparison(a_sharpe, c_sharpe, a_dd, c_dd, c_turnover, d_turnover) -> pd.DataFrame:
    rows = []
    for variant, sharpe, dd, turnover in (
        ("A", a_sharpe, a_dd, 0.20), ("C", c_sharpe, c_dd, c_turnover),
        ("D", 0.0, 0.0, d_turnover),
    ):
        rows.append({
            "variant": variant, "window": "2020-H1",
            **{c: 0.01 for c in REPORT_METRIC_COLS},
            "sharpe": sharpe, "max_drawdown": dd, "avg_turnover": turnover,
        })
    return pd.DataFrame(rows)


def test_recommendation_supports_adoption_when_c_beats_a_and_d():
    comparison = _toy_comparison(a_sharpe=1.0, c_sharpe=1.5, a_dd=-0.10, c_dd=-0.08,
                                 c_turnover=0.20, d_turnover=0.60)
    lines = _recommendation(comparison)
    assert any("supports moving to a production pilot" in l for l in lines)


def test_recommendation_does_not_support_adoption_when_c_underperforms():
    comparison = _toy_comparison(a_sharpe=1.5, c_sharpe=1.0, a_dd=-0.08, c_dd=-0.15,
                                 c_turnover=0.55, d_turnover=0.60)
    lines = _recommendation(comparison)
    assert any("not yet earning its keep" in l for l in lines)


def test_recommendation_handles_empty_comparison():
    lines = _recommendation(pd.DataFrame())
    assert lines and "No walk-forward windows" in lines[0]


def _regime_key(regime: str) -> str:
    return regime.split(" ")[0].lower()


def _toy_evidence() -> pd.DataFrame:
    """3 subfactor rows, deliberately including NaN/zero-n_eff cases (sub3 has no
    usable regime observation for low/high, and a NaN long-run anchor for medium)
    so the averaging in _shrunk_regime_summary has to skip NaNs correctly rather
    than letting one subfactor's missing data poison a whole regime's average."""
    return pd.DataFrame([
        {"sub_factor": "sub1", "long_run_mean_ic": 0.02,
         "regime_ic_low": 0.05, "regime_n_eff_low": 10,
         "regime_ic_medium": 0.01, "regime_n_eff_medium": 5,
         "regime_ic_high": -0.01, "regime_n_eff_high": 2},
        {"sub_factor": "sub2", "long_run_mean_ic": 0.00,
         "regime_ic_low": 0.03, "regime_n_eff_low": 20,
         "regime_ic_medium": float("nan"), "regime_n_eff_medium": 0,
         "regime_ic_high": 0.02, "regime_n_eff_high": 8},
        {"sub_factor": "sub3", "long_run_mean_ic": float("nan"),
         "regime_ic_low": float("nan"), "regime_n_eff_low": 0,
         "regime_ic_medium": 0.04, "regime_n_eff_medium": 15,
         "regime_ic_high": 0.0, "regime_n_eff_high": 0},
    ])


def test_shrunk_regime_summary_matches_hand_computed_shrinkage():
    evidence = _toy_evidence()
    summary = _shrunk_regime_summary(evidence).set_index("regime")

    for regime in REGIME_ORDER:
        key = _regime_key(regime)
        shrunk_per_sub = [
            shrink_regime_ic(row[f"regime_ic_{key}"], row["long_run_mean_ic"],
                             row[f"regime_n_eff_{key}"], k=SHRINKAGE_K)
            for _, row in evidence.iterrows()
        ]
        expected_ic = float(pd.Series(shrunk_per_sub).mean(skipna=True))
        expected_n_eff = float(evidence[f"regime_n_eff_{key}"].mean(skipna=True))
        assert summary.loc[regime, "avg_shrunk_ic"] == pytest.approx(expected_ic)
        assert summary.loc[regime, "avg_n_eff"] == pytest.approx(expected_n_eff)

    # sub3 contributes NaN to every regime (no usable low/high observation, and a
    # NaN long-run anchor poisons its medium shrinkage) -- confirm that didn't wipe
    # out the other two subfactors' contributions.
    assert summary.loc[REGIME_ORDER[0], "avg_shrunk_ic"] == pytest.approx(
        (shrink_regime_ic(0.05, 0.02, 10) + shrink_regime_ic(0.03, 0.00, 20)) / 2)


def test_shrunk_regime_summary_handles_empty_evidence():
    summary = _shrunk_regime_summary(pd.DataFrame())
    assert list(summary["regime"]) == list(REGIME_ORDER)
    assert summary["avg_shrunk_ic"].isna().all()
    assert summary["avg_n_eff"].isna().all()


def _toy_comparison_multi_window() -> pd.DataFrame:
    rows = []
    for window in ("2019-H2", "2020-H1"):
        for variant in ("A", "C"):
            rows.append({
                "window": window, "variant": variant,
                **{c: 0.01 for c in REPORT_METRIC_COLS},
            })
    return pd.DataFrame(rows)


def test_write_report_includes_per_window_table(tmp_path):
    comparison = _toy_comparison_multi_window()
    write_report(comparison, tmp_path)
    text = (tmp_path / "REGIME_AWARE_REPORT.md").read_text()

    assert "## Per-Window Detail" in text
    # every (window, variant) pair should appear on its own table row
    for window in ("2019-H2", "2020-H1"):
        for variant in ("A", "C"):
            assert f"| {window} | {variant} |" in text


def test_write_report_handles_empty_comparison(tmp_path):
    write_report(pd.DataFrame(), tmp_path)
    text = (tmp_path / "REGIME_AWARE_REPORT.md").read_text()
    assert "## Per-Window Detail" in text
    assert "No walk-forward windows" in text


def test_write_report_includes_vix_probabilities_table(tmp_path):
    vix_probs = pd.DataFrame([
        {"regime": "Low (<15)", "probability": 0.7, "avg_shrunk_ic": 0.02, "avg_n_eff": 40.0},
        {"regime": "Medium (15-25)", "probability": 0.3, "avg_shrunk_ic": 0.01, "avg_n_eff": 20.0},
        {"regime": "High (>25)", "probability": 0.0, "avg_shrunk_ic": float("nan"),
         "avg_n_eff": float("nan")},
    ])
    write_report(_toy_comparison_multi_window(), tmp_path, vix_probs)
    text = (tmp_path / "REGIME_AWARE_REPORT.md").read_text()

    assert "## VIX Regime Probabilities" in text
    assert "| Low (<15) | 70.0% | 0.020 | 40.000 |" in text
    assert "| High (>25) | 0.0% | — | — |" in text


def test_write_report_omits_vix_section_when_not_given(tmp_path):
    write_report(_toy_comparison_multi_window(), tmp_path)
    text = (tmp_path / "REGIME_AWARE_REPORT.md").read_text()
    assert "## VIX Regime Probabilities" not in text


def test_write_churn_chart_skips_on_empty_state_log(tmp_path):
    write_churn_chart(pd.DataFrame(), tmp_path)
    assert not (tmp_path / "churn_chart.png").exists()


def test_write_churn_chart_writes_png_for_variant_c(tmp_path):
    cutoffs = ["2020-01-31", "2020-02-29", "2020-03-31"]
    rows = []
    for c in cutoffs:
        rows.append({"cutoff": c, "variant": "C", "parent": "p1", "weight": 0.6,
                     "active_subs": "subA"})
        rows.append({"cutoff": c, "variant": "C", "parent": "p2", "weight": 0.4,
                     "active_subs": "subX"})
        rows.append({"cutoff": c, "variant": "B", "parent": "p1", "weight": 0.5,
                     "active_subs": "subA"})
    state_log = pd.DataFrame(rows)

    write_churn_chart(state_log, tmp_path)

    out = tmp_path / "churn_chart.png"
    assert out.exists()
    assert out.stat().st_size > 0
