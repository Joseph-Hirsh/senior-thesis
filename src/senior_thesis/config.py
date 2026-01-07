"""
Configuration for the military specialization and alliance institutions study.

Defines file paths, variable lists, and model specifications for testing:
- H1: Ideology -> Specialization (country-year)
- H2/H2A/H2B: Alliance institutions -> Partner specialization (dyad-year)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Paths:
    """File paths for input data and output files."""

    # Input data
    spec_rds: Path = Path("data/03_DF-full.rds")
    manifesto_csv: Path = Path("data/MPDataset_MPDS2025a.csv")
    rr_dta: Path = Path("data/partiestoanallianceR&R.dta")
    crosswalk_csv: Path = Path("data/manifesto_to_cow_crosswalk.csv")
    atop_csv: Path = Path("data/atop5_1m.csv")
    depth_csv: Path = Path("data/AllianceDataScoreJCR_RR.csv")

    # Output datasets
    country_year_csv: Path = Path("results/master_country_year.csv")
    dyad_year_csv: Path = Path("results/master_dyad_year.csv")

    # Output tables and figures
    figures_dir: Path = Path("results/figures")
    tables_dir: Path = Path("results/tables")


# Control variables from Gannon (country-level)
COUNTRY_CONTROLS = ["lngdp", "cinc", "war5_lag"]

# Control variables from Rapport & Rathbun (dyad-level)
DYAD_CONTROLS = [
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

# Institution type labels
INST_LABELS = {1: "Uninstitutionalized", 2: "Voice-Driven", 3: "Hierarchical"}
