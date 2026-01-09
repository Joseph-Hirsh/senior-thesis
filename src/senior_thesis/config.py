"""
Configuration for the military specialization and alliance institutions study.

Defines file paths, variable lists, and model specifications for testing:
- H1: Ideology -> Specialization (country-year)
- H2/H2A/H2B: Alliance institutions -> Division of labor (dyad-year)

Statistical notes:
- All models use two-way fixed effects and clustered standard errors
- H1 tests lags 1-10 with Bonferroni correction for multiple testing
- H2 uses dyad FE (following Gannon 2023) for within-dyad identification
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import pandas as pd

__all__ = [
    "Paths",
    "COUNTRY_CONTROLS",
    "DYAD_CONTROLS",
    "INST_LABELS",
    "YEAR_START",
    "YEAR_END",
    "RILE_RIGHT_THRESHOLD",
    "RILE_LEFT_THRESHOLD",
    "FIGURE_DPI",
    "FORMULAS",
    "VARIABLE_MAP",
    "get_available_controls",
    "load_dataset",
]

# =============================================================================
# Project paths
# =============================================================================

# Package root directory (senior-thesis/)
_ROOT = Path(__file__).parent.parent.parent


@dataclass(frozen=True)
class Paths:
    """File paths for input assets and output files."""

    # Input assets (absolute paths from project root)
    spec_rds: Path = field(default_factory=lambda: _ROOT / "assets" / "datasets" / "03_DF-full.rds")
    manifesto_csv: Path = field(default_factory=lambda: _ROOT / "assets" / "datasets" / "MPDataset_MPDS2025a.csv")
    rr_dta: Path = field(default_factory=lambda: _ROOT / "assets" / "datasets" / "partiestoanallianceR&R.dta")
    crosswalk_csv: Path = field(default_factory=lambda: _ROOT / "assets" / "datasets" / "manifesto_to_cow_crosswalk.csv")
    atop_csv: Path = field(default_factory=lambda: _ROOT / "assets" / "datasets" / "atop5_1m.csv")
    div_labor_csv: Path = field(default_factory=lambda: _ROOT / "assets" / "datasets" / "division_of_labor.csv")
    contiguity_csv: Path = field(default_factory=lambda: _ROOT / "assets" / "datasets" / "contdird.csv")
    rdmc_rds: Path = field(default_factory=lambda: _ROOT / "assets" / "rDMC_wide_v1.rds")

    # Output datasets
    country_year_csv: Path = field(default_factory=lambda: _ROOT / "results" / "master_country_year.csv")
    dyad_year_csv: Path = field(default_factory=lambda: _ROOT / "results" / "master_dyad_year.csv")

    # Hypothesis-specific output directories
    h1_dir: Path = field(default_factory=lambda: _ROOT / "results" / "h1")
    h2_dir: Path = field(default_factory=lambda: _ROOT / "results" / "h2")

    def validate(self) -> list[str]:
        """
        Validate that all required input files exist.

        Returns:
            List of missing file paths (empty if all exist)
        """
        input_paths = [
            self.spec_rds,
            self.manifesto_csv,
            self.rr_dta,
            self.crosswalk_csv,
            self.atop_csv,
            self.div_labor_csv,
            self.contiguity_csv,
        ]
        return [str(p) for p in input_paths if not p.exists()]


# =============================================================================
# Analysis parameters
# =============================================================================

# Time bounds (Gannon specialization data coverage)
YEAR_START: int = 1970
YEAR_END: int = 2014

# RILE ideology thresholds for binary classification
# RILE >= RIGHT_THRESHOLD -> right_of_center = 1
# RILE <= LEFT_THRESHOLD -> right_of_center = 0
# Between thresholds -> right_of_center = NaN (excluded from binary analysis)
RILE_RIGHT_THRESHOLD: float = 10.0
RILE_LEFT_THRESHOLD: float = -10.0

# Output settings
FIGURE_DPI: int = 200


# =============================================================================
# Control variables
# =============================================================================

# Country-level controls for H1 (from Gannon)
# - lngdp: Log GDP (economic capacity)
# - cinc: CINC score (military capabilities)
# - war5_lag: Interstate war in past 5 years (threat environment)
# - in_hierarchical, in_voice, in_uninst: Alliance type membership
COUNTRY_CONTROLS: list[str] = [
    "lngdp",
    "cinc",
    "war5_lag",
    "in_hierarchical",
    "in_voice",
    "in_uninst",
]

# Dyad-level controls for H2
# - contiguous: Binary land contiguity
# - gdp_ratio: Ratio of log GDPs (larger/smaller)
# - cinc_ratio: Ratio of CINC scores (larger/smaller)
DYAD_CONTROLS: list[str] = [
    "contiguous",
    "gdp_ratio",
    "cinc_ratio",
]


# =============================================================================
# Institution type coding (Leeds & Anac 2005)
# =============================================================================

# Institution type labels: integer code -> human-readable label
INST_LABELS: dict[int, str] = {
    1: "Uninstitutionalized",
    2: "Voice-Driven",
    3: "Hierarchical",
}


# =============================================================================
# Regression formulas
# =============================================================================

# Use {controls} placeholder for control variables, {lag_var} for lag variable
FORMULAS: dict[str, str] = {
    "h1_lagged": "spec_y ~ {lag_var} + {controls} + C(country_code_cow) + C(year)",
    "h2_type": "div_labor ~ hierarchical + voice_driven + {controls} + C(dyad_id) + C(decade)",
}


# =============================================================================
# Variable mapping
# =============================================================================

# Data dictionary: source column name -> analysis variable name
VARIABLE_MAP: dict[str, str] = {
    # Specialization
    "spec_stand": "spec_y",
    "spec_intscale": "spec_y",
    # Economic controls
    "lngdp_WDI_full": "lngdp",
    "gdp_WDI_full": "lngdp",  # requires log transform
    # Military controls
    "cinc_MC": "cinc",
    "interstatewar_5yrlag_binary": "war5_lag",
}


# =============================================================================
# Helper functions
# =============================================================================


def get_available_controls(df: pd.DataFrame, controls: list[str]) -> list[str]:
    """
    Return only controls that exist in the dataframe.

    Args:
        df: DataFrame to check
        controls: List of control variable names

    Returns:
        Filtered list containing only columns present in df
    """
    return [c for c in controls if c in df.columns]


@lru_cache(maxsize=4)
def load_dataset(path: Path) -> pd.DataFrame:
    """
    Load a CSV dataset with caching.

    Args:
        path: Path to CSV file

    Returns:
        Loaded DataFrame (cached on subsequent calls)
    """
    return pd.read_csv(path)
