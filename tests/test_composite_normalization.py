"""Score-normalization behaviour in build_composite.

Verifies that `factors.score_normalization=zscore` equalizes each parent
factor's dispersion before the weighted blend, so `weight` is the true influence
lever and a wide-dispersion factor no longer out-votes a compressed one at equal
weight. See output/factor_distribution/REPORT.md for the motivating diagnosis.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from data.config import Config
from factors.base import FactorResult
from factors.composite import NEUTRAL, _normalize_parents, build_composite


def _cfg(mode: str, target_std: float = 20.0) -> Config:
    return Config({
        "factors": {
            "score_normalization": mode,
            "normalization_target_std": target_std,
            "classification": {"long_percentile": 80, "short_percentile": 20},
        }
    })


def _result(key: str, parent: pd.Series) -> FactorResult:
    return FactorResult(name=key, key=key, parent=parent,
                        sub_scores=pd.DataFrame(index=parent.index))


def _universe(n: int = 60):
    idx = [f"T{i:03d}" for i in range(n)]
    # One sector so sector_percentile ranks the whole universe together.
    sectors = pd.Series(["Tech"] * n, index=idx)
    return idx, sectors


def test_normalize_equalizes_std_about_neutral():
    idx, _ = _universe()
    wide = pd.Series(np.linspace(1, 99, len(idx)), index=idx)      # std ~28
    narrow = pd.Series(np.linspace(45, 55, len(idx)), index=idx)   # std ~3
    parents = pd.DataFrame({"momentum": wide, "insider": narrow})

    norm = _normalize_parents(parents, "zscore", target_std=20.0)
    # Every factor is rescaled to the shared target std, regardless of its
    # original spread — that is the whole point.
    assert np.isclose(norm["momentum"].std(), 20.0, atol=1e-6)
    assert np.isclose(norm["insider"].std(), 20.0, atol=1e-6)
    # The transform is linear about the fixed neutral 50, so a name sitting at
    # exactly 50 stays at 50 and ordering is preserved.
    mid = pd.Series(50.0, index=idx)
    only_mid = _normalize_parents(pd.DataFrame({"f": mid}), "zscore", 20.0)["f"]
    assert (only_mid == 50.0).all()
    assert norm["momentum"].is_monotonic_increasing  # same order as wide


def test_normalize_none_is_identity():
    idx, _ = _universe()
    parents = pd.DataFrame({"momentum": pd.Series(np.linspace(1, 99, len(idx)), index=idx)})
    out = _normalize_parents(parents, "none", target_std=20.0)
    pd.testing.assert_frame_equal(out, parents)


def test_dead_factor_left_at_neutral():
    idx, _ = _universe()
    dead = pd.Series(NEUTRAL, index=idx)  # zero dispersion → all missing/neutral
    parents = pd.DataFrame({"revisions": dead})
    out = _normalize_parents(parents, "zscore", target_std=20.0)
    assert (out["revisions"] == NEUTRAL).all()


def test_normalization_collapses_influence_onto_weight():
    """Equal weights + very different dispersions on *independent* factors:
    normalization makes the two factors move the ranking equally; the raw blend
    lets the wide-dispersion one dominate."""
    idx, sectors = _universe()
    rng = np.random.default_rng(0)
    # Two independent orderings so each factor can pull the blend differently.
    wide = pd.Series(rng.permutation(np.linspace(1, 99, len(idx))), index=idx)    # std ~28
    narrow = pd.Series(rng.permutation(np.linspace(40, 60, len(idx))), index=idx)  # std ~6
    results = [_result("momentum", wide), _result("insider", narrow)]
    weights = {"momentum": 0.5, "insider": 0.5}

    raw = build_composite(results, weights, sectors, _cfg("none"), min_obs=5)
    norm = build_composite(results, weights, sectors, _cfg("zscore"), min_obs=5)

    def corr(frame, col):
        return float(frame["composite_raw"].corr(frame[col]))

    # Raw blend: composite tracks the wide factor much more than the narrow one.
    raw_gap = corr(raw, "momentum_score") - corr(raw, "insider_score")
    assert raw_gap > 0.4
    # Normalized: equal weight → equal pull, so the composite correlates about
    # the same with each factor (gap collapses toward 0).
    norm_gap = abs(corr(norm, "momentum_score") - corr(norm, "insider_score"))
    assert norm_gap < 0.1
    assert norm_gap < raw_gap


def test_reported_parent_scores_are_unchanged_by_normalization():
    """The `*_score` columns keep the raw parent percentiles; only the blend is
    normalized."""
    idx, sectors = _universe()
    wide = pd.Series(np.linspace(1, 99, len(idx)), index=idx)
    results = [_result("momentum", wide)]
    weights = {"momentum": 1.0}
    raw = build_composite(results, weights, sectors, _cfg("none"), min_obs=5)
    norm = build_composite(results, weights, sectors, _cfg("zscore"), min_obs=5)
    pd.testing.assert_series_equal(raw["momentum_score"], norm["momentum_score"])
