from __future__ import annotations

import hashlib
import pandas as pd

def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def print_schema(name: str, df: pd.DataFrame) -> None:
    print("\n" + "=" * 80)
    print(f"SCHEMA: {name}")
    print("=" * 80)
    df.info(verbose=True, show_counts=True)
    print()

def assert_unique_key(df: pd.DataFrame, keys: list[str], name: str) -> None:
    dup = df.duplicated(keys).sum()
    if dup:
        examples = df[df.duplicated(keys, keep=False)].groupby(keys).size().sort_values(ascending=False).head(10)
        raise ValueError(f"{name}: key not unique on {keys}. Duplicate rows={dup}. Examples:\n{examples}")
    print(f"✓ {name}: key unique on {keys}")

def merge_audit(merged: pd.DataFrame, indicator_col: str, label: str) -> None:
    print(f"\n--- Merge audit: {label} ---")
    vc = merged[indicator_col].value_counts(dropna=False)
    total = len(merged)
    for k in ["both", "left_only", "right_only"]:
        n = int(vc.get(k, 0))
        print(f"  {k:10s}: {n:>8,} ({n/total:>6.1%})")
    print()

def coverage_audit(df: pd.DataFrame, country: str, year: str, label: str) -> None:
    print(f"\n--- Coverage: {label} ---")
    print(f"Rows: {len(df):,}")
    print(f"Countries: {df[country].nunique():,}")
    print(f"Years: {int(df[year].min())}–{int(df[year].max())}")
    print()

def within_country_variation(df: pd.DataFrame, country: str, var: str) -> None:
    # counts of countries that ever take both 0 and 1
    tmp = df.dropna(subset=[var]).groupby(country)[var].nunique()
    both = (tmp >= 2).sum()
    one = (tmp == 1).sum()
    print(f"\n--- Within-country variation: {var} ---")
    print(f"Countries with both 0 and 1: {both:,}")
    print(f"Countries with only one value (always 0 or always 1): {one:,}")
    print()
