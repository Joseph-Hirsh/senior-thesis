"""Data preparation utilities for analysis."""
from __future__ import annotations

import pandas as pd

from senior_thesis.config import Paths


def prep_master(paths: Paths) -> pd.DataFrame:
    """
    Load master dataset for analysis.

    The master dataset is pre-computed by build_master.py and includes:
    - Dependent variable (spec_y)
    - Independent variables (rile, right_of_center, with 1-20 year lags)
    - Control variables (lngdp, cinc, war5_lag, ln_milex_allies, cinc_allies_ratio)
    - Time trend variables (t, t2, t3)

    This function simply loads the CSV and ensures clean data types.
    All variable transformations are computed once in build_master.py.
    """
    df = pd.read_csv(paths.master_out, low_memory=False)

    # Ensure clean data types for regression
    df["country_code_cow"] = pd.to_numeric(df["country_code_cow"], errors="coerce")
    df["year"] = pd.to_numeric(df["year"], errors="coerce")

    # Drop rows with missing country or year
    df = df.dropna(subset=["country_code_cow", "year"])

    df["country_code_cow"] = df["country_code_cow"].astype(int)
    df["year"] = df["year"].astype(int)

    return df
