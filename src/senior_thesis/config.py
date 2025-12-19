from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class Paths:
    spec_rds: str = "data/03_DF-full.rds"
    manifesto_csv: str = "data/MPDataset_MPDS2025a.csv"
    rr_dta: str = "data/partiestoanallianceR&R.dta"
    manifesto_crosswalk_csv: str = "data/manifesto_to_cow_crosswalk.csv"

    ideology_out: str = "data/ideology_panel_country_year.csv"
    rr_out: str = "data/rr_aggregates_country_year.csv"
    master_out: str = "data/master_country_year.csv"

@dataclass(frozen=True)
class AnalysisSpec:
    dv: str = "spec_stand"
    iv: str = "right_of_center"
    baseline_controls: tuple[str, ...] = ("lngdppc_WDI_full", "lnpop_WDI_full", "polity2_P4")
