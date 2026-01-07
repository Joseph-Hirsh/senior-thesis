"""
Utility functions for data validation and reproducibility.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd


def file_hash(path: Path) -> str:
    """Compute SHA-256 hash of a file for reproducibility tracking."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def assert_unique_key(df: pd.DataFrame, keys: list[str], name: str) -> None:
    """Verify that columns form a unique key, raising an error if duplicates exist."""
    dups = df.duplicated(keys).sum()
    if dups:
        examples = df[df.duplicated(keys, keep=False)].groupby(keys).size().head(5)
        raise ValueError(f"{name}: {dups} duplicate rows on {keys}.\n{examples}")
    print(f"  [{name}] unique on {keys}")


def merge_report(df: pd.DataFrame, indicator: str, label: str) -> None:
    """Print merge statistics."""
    counts = df[indicator].value_counts()
    both = counts.get("both", 0)
    left = counts.get("left_only", 0)
    right = counts.get("right_only", 0)
    print(f"  [{label}] both={both:,} | left_only={left:,} | right_only={right:,}")


def coverage_report(df: pd.DataFrame, id_col: str, year_col: str, label: str) -> None:
    """Print coverage statistics for a panel dataset."""
    n_rows = len(df)
    n_ids = df[id_col].nunique()
    year_min = int(df[year_col].min())
    year_max = int(df[year_col].max())
    print(f"  [{label}] {n_rows:,} rows | {n_ids} units | {year_min}-{year_max}")
