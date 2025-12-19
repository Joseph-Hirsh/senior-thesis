"""Descriptive statistics and data coverage reporting."""
from __future__ import annotations

import pandas as pd

from senior_thesis.config import Paths, get_controls
from senior_thesis.data_prep import prep_master


def print_coverage(df: pd.DataFrame, columns: list[str], label: str) -> None:
    """Print non-missing data counts for specified columns."""
    print(f"\n--- Coverage check: {label} ---")
    for col in columns:
        non_missing = df[col].notna().sum()
        print(f"{col:>22}: {non_missing:>6} / {len(df)} ({non_missing/len(df):.1%})")


def run_descriptive_analysis() -> None:
    """
    Run descriptive analysis and coverage checks on the master dataset.

    Prints coverage statistics for key variables including:
    - Dependent variable (specialization)
    - Independent variables (ideology)
    - Control variables (GDP, CINC, war, alliances)
    """
    paths = Paths()
    df = prep_master(paths)

    # Define variable groups for coverage checks
    all_controls = get_controls()

    # Sample of lagged ideology variables
    lag_vars = [f"right_of_center_lag{i}" for i in range(1, 6)]

    # Check coverage for key variables
    print("\n" + "="*80)
    print("DESCRIPTIVE ANALYSIS: Data Coverage")
    print("="*80)

    print_coverage(
        df,
        ["spec_y", "rile", "right_of_center"] + lag_vars + all_controls,
        "Key variables"
    )

    print("\n" + "="*80)
    print(f"Total observations: {len(df)}")
    print(f"Countries: {df['country_code_cow'].nunique()}")
    print(f"Year range: {df['year'].min()}-{df['year'].max()}")
    print("="*80)


def main() -> None:
    """Run descriptive analysis from command line."""
    run_descriptive_analysis()


if __name__ == "__main__":
    main()
