"""Subfactor Expansion — research library for bringing each parent bucket to
8–10 well-designed candidate subfactors.

Isolated from production ``factors/``; does not touch the daily scoring
pipeline. Consumers:

* :mod:`research.subfactor_expansion.library` — the ``Candidate`` computations
  themselves, organized by parent.
* :mod:`research.subfactor_expansion.panel` — a candidate score panel that
  runs each candidate through the same GICS-sector percentile rule as
  production, so ICs are directly comparable.
* :mod:`research.subfactor_expansion.validation` — coverage/IC/quintile/
  monotonicity/hit-rate/year-stability/sign-sanity checks.
* :mod:`research.subfactor_expansion.report` — writes the REPORT.md deliverable.
"""
from .library import Candidate, CANDIDATE_BUILDERS, iter_parents

__all__ = ["Candidate", "CANDIDATE_BUILDERS", "iter_parents"]
