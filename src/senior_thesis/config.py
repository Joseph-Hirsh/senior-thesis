"""Configuration for data paths and analysis specifications."""
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class Paths:
    """File paths for input data and output files."""
    # Input data files
    spec_rds: str = "data/03_DF-full.rds"
    manifesto_csv: str = "data/MPDataset_MPDS2025a.csv"
    rr_dta: str = "data/partiestoanallianceR&R.dta"
    manifesto_crosswalk_csv: str = "data/manifesto_to_cow_crosswalk.csv"

    # Output files
    ideology_out: str = "results/ideology_panel_country_year.csv"
    rr_out: str = "results/rr_aggregates_country_year.csv"
    master_out: str = "results/master_country_year.csv"


# Control variables for regression analysis
REGRESSION_CONTROLS = {
    "base": ["lngdp", "cinc", "war5_lag"],
    "alliance": ["ln_milex_allies", "cinc_allies_ratio"],
}


def get_controls() -> list[str]:
    """Return standard control variables for regression analysis."""
    return REGRESSION_CONTROLS["base"] + REGRESSION_CONTROLS["alliance"]
