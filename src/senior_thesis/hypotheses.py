"""
Hypothesis-specific analysis runners.

Orchestrates descriptives and regressions for each hypothesis:
- H1: Ideology -> Specialization (country-year)
- H2/H2A/H2B: Alliance institutions -> Partner specialization (dyad-year)
"""
from __future__ import annotations

from senior_thesis.config import Paths
from senior_thesis.descriptives import h1_descriptives, h2_descriptives
from senior_thesis.regressions import (
    model_h1,
    model_h1_lagged,
    model_h2,
    model_h2ab,
    model_robustness,
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
    - Main regression model
    - Lagged robustness models
    """
    print("\n" + "=" * 60)
    print(" H1: IDEOLOGY -> SPECIALIZATION")
    print("=" * 60)

    paths.h1_dir.mkdir(parents=True, exist_ok=True)

    h1_descriptives(paths)
    model_h1(paths)
    model_h1_lagged(paths)

    print(f"\n  Outputs saved to: {paths.h1_dir}/")


def run_h2(paths: Paths) -> None:
    """
    Run all H2/H2A/H2B analyses: Alliance institutions -> Specialization.

    Includes:
    - Descriptive statistics and visualizations
    - H2: Alliance depth model
    - H2A/H2B: Institution type model
    - Robustness: Joint model with depth_within_type
    """
    print("\n" + "=" * 60)
    print(" H2/H2A/H2B: ALLIANCE INSTITUTIONS -> SPECIALIZATION")
    print("=" * 60)

    paths.h2_dir.mkdir(parents=True, exist_ok=True)

    h2_descriptives(paths)
    model_h2(paths)
    model_h2ab(paths)
    model_robustness(paths)

    print(f"\n  Outputs saved to: {paths.h2_dir}/")
