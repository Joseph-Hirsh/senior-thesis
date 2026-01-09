"""
Hypothesis-specific analysis runners.

H1: Ideology → Specialization (country-year)
H2: Alliance type → Division of labor (dyad-year)
H3: Ideological similarity → Division of labor (dyad-year)
"""
from __future__ import annotations

from senior_thesis.config import Paths
from senior_thesis.descriptives import h1_descriptives, h2_descriptives
from senior_thesis.regressions import (
    run_h1_regressions,
    run_h2_regressions,
    run_h3_regressions,
)

__all__ = [
    "run_h1",
    "run_h2",
    "run_h3",
]


def run_h1(paths: Paths) -> None:
    """
    Run H1 analyses: Ideology → Specialization.

    - Descriptive statistics and figures
    - Primary specification (continuous RILE, 5-year lag)
    - Placebo test (future ideology should not predict)
    - Event study around ideology transitions
    """
    print("\n" + "=" * 60)
    print(" H1: IDEOLOGY → SPECIALIZATION")
    print("=" * 60)

    paths.h1_dir.mkdir(parents=True, exist_ok=True)

    # Descriptives
    h1_descriptives(paths)

    # Regressions
    run_h1_regressions(paths)

    print(f"\n  Outputs: {paths.h1_dir}/")


def run_h2(paths: Paths) -> None:
    """
    Run H2 analyses: Alliance type → Division of labor.

    Uses Gannon UNION sample (1980-2010, ATOP OR DCAD aligned).

    - Descriptive statistics
    - Ordinal specification (vertical_integration 0-3)
    - Categorical specification (hierarchical + voice_driven dummies)
    - H2B test (hierarchical > voice-driven)
    """
    print("\n" + "=" * 60)
    print(" H2: ALLIANCE TYPE → DIVISION OF LABOR")
    print("=" * 60)

    paths.h2_dir.mkdir(parents=True, exist_ok=True)

    # Descriptives
    h2_descriptives(paths)

    # Regressions
    run_h2_regressions(paths)

    print(f"\n  Outputs: {paths.h2_dir}/")


def run_h3(paths: Paths) -> None:
    """
    Run H3 analyses: Ideological Similarity → Division of Labor.

    Uses Gannon UNION sample (1980-2010).

    - Minimal specification (ideology distance + FE only)
    - Full specification (+ controls)
    """
    print("\n" + "=" * 60)
    print(" H3: IDEOLOGICAL SIMILARITY → DIVISION OF LABOR")
    print("=" * 60)

    paths.h3_dir.mkdir(parents=True, exist_ok=True)

    # Regressions
    run_h3_regressions(paths)

    print(f"\n  Outputs: {paths.h3_dir}/")
