"""Utility functions for data validation and auditing."""
from __future__ import annotations

import hashlib
import pandas as pd


def sha256_file(path: str) -> str:
    """Compute SHA-256 hash of a file for reproducibility tracking."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def print_schema(name: str, df: pd.DataFrame) -> None:
    """Print DataFrame schema information."""
    print("\n" + "=" * 80)
    print(f"SCHEMA: {name}")
    print("=" * 80)
    df.info(verbose=True, show_counts=True)
    print()


def assert_unique_key(df: pd.DataFrame, keys: list[str], name: str) -> None:
    """
    Verify that the specified columns form a unique key in the DataFrame.
    Raises an error with examples if duplicates are found.
    """
    duplicate_count = df.duplicated(keys).sum()
    if duplicate_count:
        duplicate_examples = (
            df[df.duplicated(keys, keep=False)]
            .groupby(keys)
            .size()
            .sort_values(ascending=False)
            .head(10)
        )
        raise ValueError(
            f"{name}: key not unique on {keys}. "
            f"Duplicate rows={duplicate_count}. Examples:\n{duplicate_examples}"
        )
    print(f"✓ {name}: key unique on {keys}")


def merge_audit(merged: pd.DataFrame, indicator_col: str, label: str) -> None:
    """Print summary statistics for a pandas merge operation."""
    print(f"\n--- Merge audit: {label} ---")
    merge_counts = merged[indicator_col].value_counts(dropna=False)
    total = len(merged)

    for merge_type in ["both", "left_only", "right_only"]:
        count = int(merge_counts.get(merge_type, 0))
        print(f"  {merge_type:10s}: {count:>8,} ({count/total:>6.1%})")
    print()


def coverage_audit(df: pd.DataFrame, country_col: str, year_col: str, label: str) -> None:
    """Print coverage statistics for a country-year panel dataset."""
    print(f"\n--- Coverage: {label} ---")
    print(f"Rows: {len(df):,}")
    print(f"Countries: {df[country_col].nunique():,}")
    print(f"Years: {int(df[year_col].min())}–{int(df[year_col].max())}")
    print()


def within_country_variation(df: pd.DataFrame, country_col: str, variable: str) -> None:
    """
    Analyze within-country variation in a binary or categorical variable.
    Useful for identifying variables suitable for fixed effects estimation.
    """
    unique_values_per_country = df.dropna(subset=[variable]).groupby(country_col)[variable].nunique()

    countries_with_variation = (unique_values_per_country >= 2).sum()
    countries_without_variation = (unique_values_per_country == 1).sum()

    print(f"\n--- Within-country variation: {variable} ---")
    print(f"Countries with variation: {countries_with_variation:,}")
    print(f"Countries without variation: {countries_without_variation:,}")
    print()


def main() -> None:
    """Run data auditing utilities from command line."""
    # Import here to avoid circular dependencies
    from senior_thesis.build_master import load_spec, load_manifesto, load_rr
    from senior_thesis.config import Paths

    paths = Paths()

    print("\n" + "="*80)
    print("DATA AUDIT REPORT")
    print("="*80)

    # Audit raw data files
    spec = load_spec(paths)
    manifesto = load_manifesto(paths)
    rr = load_rr(paths)

    print_schema("Specialization Data", spec)
    print_schema("Manifesto Data", manifesto)
    print_schema("R&R Alliance Data", rr)


if __name__ == "__main__":
    main()
