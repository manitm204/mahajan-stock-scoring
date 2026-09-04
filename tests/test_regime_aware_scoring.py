from __future__ import annotations

import pandas as pd
import pytest

from research.walkforward.regime_aware_scoring import classify_regime_subfactors, score_table


def test_classify_core_regime_dependent_excluded_watchlist():
    evidence = pd.DataFrame([
        # CORE: strong IC, mostly positive years, low regime swing.
        {"sub_factor": "core1", "long_run_mean_ic": 0.02, "pct_positive_years": 0.80,
         "regime_dependence": 0.01, "n_years": 5},
        # EXCLUDED: negative IC, mostly negative years, enough history to trust it.
        {"sub_factor": "bad1", "long_run_mean_ic": -0.02, "pct_positive_years": 0.20,
         "regime_dependence": 0.02, "n_years": 5},
        # REGIME_DEPENDENT: non-negative IC but a big regime swing.
        {"sub_factor": "regime1", "long_run_mean_ic": 0.005, "pct_positive_years": 0.50,
         "regime_dependence": 0.08, "n_years": 5},
        # WATCHLIST: negative IC but too little history to call it EXCLUDED.
        {"sub_factor": "thin1", "long_run_mean_ic": -0.02, "pct_positive_years": 0.20,
         "regime_dependence": 0.01, "n_years": 1},
    ])
    flags = classify_regime_subfactors(evidence)
    assert flags.tolist() == ["CORE", "EXCLUDED", "REGIME_DEPENDENT", "WATCHLIST"]


def test_score_table_ranks_within_parent_only():
    evidence = pd.DataFrame([
        # parent A: a1 dominates every metric, a2 is worse on every metric.
        {"sub_factor": "a1", "parent": "A", "expected_ic": 0.05, "long_run_mean_ic": 0.05,
         "long_run_std_ic": 0.02, "long_run_spread_ann": 0.08, "long_run_hit_rate": 0.7,
         "pct_positive_years": 0.8, "persistence_ir": 1.0, "spread_consistency": 0.9,
         "coverage": 0.95, "n_months": 60, "n_years": 5, "regime_dependence": 0.01},
        {"sub_factor": "a2", "parent": "A", "expected_ic": 0.01, "long_run_mean_ic": 0.01,
         "long_run_std_ic": 0.03, "long_run_spread_ann": 0.02, "long_run_hit_rate": 0.5,
         "pct_positive_years": 0.5, "persistence_ir": 0.2, "spread_consistency": 0.5,
         "coverage": 0.80, "n_months": 40, "n_years": 4, "regime_dependence": 0.02},
        # parent B: single, absolutely weaker sub -- should still rank 1.0 within its
        # own bucket, proving ranking is per-parent, not global.
        {"sub_factor": "b1", "parent": "B", "expected_ic": 0.001, "long_run_mean_ic": 0.001,
         "long_run_std_ic": 0.05, "long_run_spread_ann": 0.001, "long_run_hit_rate": 0.51,
         "pct_positive_years": 0.51, "persistence_ir": 0.1, "spread_consistency": 0.51,
         "coverage": 0.60, "n_months": 36, "n_years": 3, "regime_dependence": 0.01},
    ])
    out = score_table(evidence)
    a1 = out[out.sub_factor == "a1"].iloc[0]
    a2 = out[out.sub_factor == "a2"].iloc[0]
    b1 = out[out.sub_factor == "b1"].iloc[0]

    assert a1["production_score"] > a2["production_score"]
    assert b1["predictive_score"] == pytest.approx(1.0, abs=1e-9)
    assert b1["reliability_score"] == pytest.approx(1.0, abs=1e-9)
    assert a1["eligible"] and a2["eligible"] and b1["eligible"]
    assert a1["ic_ir"] == pytest.approx(0.05 / 0.02, abs=1e-9)


def test_score_table_ineligible_when_expected_ic_not_positive():
    evidence = pd.DataFrame([
        {"sub_factor": "neg1", "parent": "A", "expected_ic": -0.01, "long_run_mean_ic": -0.01,
         "long_run_std_ic": 0.02, "long_run_spread_ann": -0.01, "long_run_hit_rate": 0.4,
         "pct_positive_years": 0.3, "persistence_ir": -0.5, "spread_consistency": 0.3,
         "coverage": 0.9, "n_months": 50, "n_years": 4, "regime_dependence": 0.01},
    ])
    out = score_table(evidence)
    assert not out.iloc[0]["eligible"]


def test_score_table_empty_evidence_returns_all_expected_columns():
    # as_of() can legitimately return an empty frame (e.g. before panel inception);
    # score_table() must still carry every column the populated path produces, so a
    # caller doing out["production_score"] on either path never KeyErrors.
    evidence = pd.DataFrame(columns=[
        "sub_factor", "parent", "expected_ic", "long_run_mean_ic", "long_run_std_ic",
        "long_run_spread_ann", "long_run_hit_rate", "pct_positive_years",
        "persistence_ir", "spread_consistency", "coverage", "n_months", "n_years",
        "regime_dependence",
    ])
    out = score_table(evidence)
    assert out.empty
    assert "production_score" in out.columns
    assert "eligible" in out.columns
    assert "rank_expected_ic" in out.columns
