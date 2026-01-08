"""
Configuration for the military specialization and alliance institutions study.

Defines file paths, variable lists, and model specifications for testing:
- H1: Ideology -> Specialization (country-year)
- H2/H2A/H2B: Alliance institutions -> Partner specialization (dyad-year)
"""
from __future__ import annotations

import logging
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
    "setup_logging",
    "get_available_controls",
    "load_dataset",
]

# Package root directory (senior-thesis/)
_ROOT = Path(__file__).parent.parent.parent

# Analysis time bounds (Gannon specialization data coverage)
YEAR_START: int = 1970
YEAR_END: int = 2014

# RILE ideology thresholds for binary classification
RILE_RIGHT_THRESHOLD: float = 10.0
RILE_LEFT_THRESHOLD: float = -10.0

# Output settings
FIGURE_DPI: int = 200


@dataclass(frozen=True)
class Paths:
    """File paths for input data and output files."""

    # Input data (absolute paths from project root)
    spec_rds: Path = field(default_factory=lambda: _ROOT / "data" / "03_DF-full.rds")
    manifesto_csv: Path = field(default_factory=lambda: _ROOT / "data" / "MPDataset_MPDS2025a.csv")
    rr_dta: Path = field(default_factory=lambda: _ROOT / "data" / "partiestoanallianceR&R.dta")
    crosswalk_csv: Path = field(default_factory=lambda: _ROOT / "data" / "manifesto_to_cow_crosswalk.csv")
    atop_csv: Path = field(default_factory=lambda: _ROOT / "data" / "atop5_1m.csv")
    depth_csv: Path = field(default_factory=lambda: _ROOT / "data" / "AllianceDataScoreJCR_RR.csv")

    # Output datasets
    country_year_csv: Path = field(default_factory=lambda: _ROOT / "results" / "master_country_year.csv")
    dyad_year_csv: Path = field(default_factory=lambda: _ROOT / "results" / "master_dyad_year.csv")

    # Hypothesis-specific output directories
    h1_dir: Path = field(default_factory=lambda: _ROOT / "results" / "h1")
    h2_dir: Path = field(default_factory=lambda: _ROOT / "results" / "h2")

    def validate(self) -> list[str]:
        """
        Validate that all input files exist.

        Returns:
            List of missing file paths (empty if all exist)
        """
        input_paths = [
            self.spec_rds,
            self.manifesto_csv,
            self.rr_dta,
            self.crosswalk_csv,
            self.atop_csv,
            self.depth_csv,
        ]
        return [str(p) for p in input_paths if not p.exists()]


# Control variables from Gannon (country-level)
COUNTRY_CONTROLS: list[str] = ["lngdp", "cinc", "war5_lag"]

# Control variables from Rapport & Rathbun (dyad-level)
DYAD_CONTROLS: list[str] = [
    "coldwar",
    "tot_rivals",
    "totmids2",
    "s_un_glo",
    "undist",
    "jntdem",
    "priorviol",
    "symm",
    "lncprtio",
    "priorviol_x_symm",
]

# Institution type labels (R&R coding: 1=uninst, 2=voice, 3=hierarchical)
INST_LABELS: dict[int, str] = {
    1: "Uninstitutionalized",
    2: "Voice-Driven",
    3: "Hierarchical",
}

# Regression formulas (use {controls} placeholder for control variables)
FORMULAS: dict[str, str] = {
    "h1_main": "spec_y ~ right_of_center + {controls} + C(country_code_cow) + C(year)",
    "h1_lagged": "spec_y ~ {lag_var} + {controls} + C(country_code_cow) + C(year)",
    "h2_depth": "spec_dyad_mean ~ Depth_score + {controls} + C(year)",
    "h2ab_type": "spec_dyad_mean ~ hierarchical + voice_driven + {controls} + C(year)",
    "robustness": "spec_dyad_mean ~ hierarchical + voice_driven + depth_within_type + {controls} + C(year)",
}

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
    # Depth score (rename for formula compatibility)
    "Depth.score": "Depth_score",
}


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """
    Configure logging for the thesis pipeline.

    Args:
        level: Logging level (default: INFO)

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger("senior_thesis")
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("  [%(levelname)s] %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


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
