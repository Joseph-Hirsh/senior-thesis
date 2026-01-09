"""
Hypothesis-specific analysis runners.

Orchestrates descriptives and regressions for each hypothesis:
- H1: Ideology -> Specialization (country-year)
- H2: Alliance type -> Division of labor (dyad-year)
"""
from __future__ import annotations

from senior_thesis.config import Paths
from senior_thesis.descriptives import h1_descriptives, h2_descriptives
from senior_thesis.regressions import (
    model_h1,
    model_h1_event_study,
    model_h1_did,
    model_h2,
    model_h2_event_study,
)

__all__ = [
    "run_h1",
    "run_h2",
]


def run_h1(paths: Paths) -> None:
    """
    Run all H1 analyses: Ideology -> Specialization.

    Includes:
    - Descriptive statistics and visualizations
    - Lagged regression models (1-10 year lags)
    - Event study: ideology transitions
    """
    print("\n" + "=" * 60)
    print(" H1: IDEOLOGY -> SPECIALIZATION")
    print("=" * 60)

    paths.h1_dir.mkdir(parents=True, exist_ok=True)

    h1_descriptives(paths)
    model_h1(paths)
    model_h1_event_study(paths)
    model_h1_did(paths)

    print(f"\n  Outputs saved to: {paths.h1_dir}/")


def run_h2(paths: Paths) -> None:
    """
    Run all H2 analyses: Alliance type -> Division of labor.

    Division of labor is a DYAD-LEVEL concept (complementarity between partners).
    This is distinct from specialization (country-level portfolio concentration).

    Includes:
    - Descriptive statistics and visualizations
    - H2: Institution type -> division of labor (dyad-level)
    - Event study: alliance entry -> division of labor
    """
    print("\n" + "=" * 60)
    print(" H2: ALLIANCE TYPE -> DIVISION OF LABOR")
    print("=" * 60)

    paths.h2_dir.mkdir(parents=True, exist_ok=True)

    h2_descriptives(paths)
    model_h2(paths)
    model_h2_event_study(paths)

    print(f"\n  Outputs saved to: {paths.h2_dir}/")
