"""
Build master dataset by merging specialization, ideology, and alliance data.

This script:
1. Loads raw data from multiple sources (specialization spine, Manifesto, R&R)
2. Constructs a country-year ideology panel from party manifestos
3. Aggregates dyadic alliance data to country-year level
4. Merges everything into a master country-year dataset for analysis
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pyreadr

from senior_thesis.config import Paths
from senior_thesis.audit import (
    sha256_file,
    print_schema,
    assert_unique_key,
    merge_audit,
    coverage_audit,
)


# Column names required in the Manifesto dataset
REQUIRED_MANIFESTO_COLS = {"country", "edate", "date", "absseat", "totseats", "pervote", "rile", "partyname"}


def _safe_log(series: pd.Series) -> pd.Series:
    """Compute natural log, setting non-positive values to NaN."""
    numeric_series = pd.to_numeric(series, errors="coerce")
    positive_only = numeric_series.where(numeric_series > 0)
    return np.log(positive_only)


def _require_columns(df: pd.DataFrame, cols: set[str], name: str) -> None:
    """Verify that DataFrame contains all required columns, raising an error if any are missing."""
    missing = sorted([c for c in cols if c not in df.columns])
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _normalize_crosswalk(crosswalk: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize the Manifesto-to-COW country code crosswalk.

    Accepts either 'country_code_cow' or 'cow_ccode' as the COW code column name.
    Returns a clean crosswalk with just [manifesto_country_code, country_code_cow].
    """
    if "manifesto_country_code" not in crosswalk.columns:
        raise ValueError("Crosswalk must include column 'manifesto_country_code'.")

    # Handle alternate column name for COW code
    if "country_code_cow" not in crosswalk.columns:
        if "cow_ccode" in crosswalk.columns:
            crosswalk = crosswalk.rename(columns={"cow_ccode": "country_code_cow"})
        else:
            raise ValueError(
                "Crosswalk must include either 'country_code_cow' or 'cow_ccode' as the COW code column."
            )

    # Keep only the two columns we need
    crosswalk = crosswalk[["manifesto_country_code", "country_code_cow"]].copy()

    # Convert to numeric and drop invalid rows
    crosswalk["manifesto_country_code"] = pd.to_numeric(crosswalk["manifesto_country_code"], errors="coerce")
    crosswalk["country_code_cow"] = pd.to_numeric(crosswalk["country_code_cow"], errors="coerce")
    crosswalk = crosswalk.dropna().copy()
    crosswalk["manifesto_country_code"] = crosswalk["manifesto_country_code"].astype(int)
    crosswalk["country_code_cow"] = crosswalk["country_code_cow"].astype(int)

    # Verify one-to-one mapping
    if crosswalk.duplicated(["manifesto_country_code"]).any():
        duplicates = (
            crosswalk[crosswalk.duplicated(["manifesto_country_code"], keep=False)]
            .sort_values("manifesto_country_code")
        )
        raise ValueError(
            "Crosswalk is not unique on manifesto_country_code. Fix the CSV.\n"
            f"Examples:\n{duplicates.head(20).to_string(index=False)}"
        )

    return crosswalk


def _parse_manifesto_election_year(manifesto: pd.DataFrame) -> pd.Series:
    """
    Extract election year from Manifesto dataset.

    Primary method: Parse 'edate' column (Manifesto uses DD/MM/YYYY format).
    Fallback: Extract year from 'date' column (YYYYMM integer format).
    """
    # Parse date strings
    election_dates = pd.to_datetime(manifesto["edate"], errors="coerce", dayfirst=True)
    election_year = election_dates.dt.year

    # Use fallback for missing values
    missing_year = election_year.isna() & manifesto["date"].notna()
    if missing_year.any():
        date_as_number = pd.to_numeric(manifesto.loc[missing_year, "date"], errors="coerce")
        election_year.loc[missing_year] = date_as_number // 100

    return pd.to_numeric(election_year, errors="coerce")

def load_spec(paths: Paths) -> pd.DataFrame:
    """Load specialization data (the country-year spine for our panel)."""
    print("\nRaw file hash (spec RDS):", sha256_file(paths.spec_rds))
    result = pyreadr.read_r(paths.spec_rds)
    df = list(result.values())[0].copy()

    # Ensure clean country-year keys
    df["country_code_cow"] = pd.to_numeric(df["country_code_cow"], errors="coerce")
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["country_code_cow", "year"]).copy()
    df["country_code_cow"] = df["country_code_cow"].astype(int)
    df["year"] = df["year"].astype(int)

    assert_unique_key(df, ["country_code_cow", "year"], "Specialization spine")
    return df


def load_manifesto(paths: Paths) -> pd.DataFrame:
    """Load Manifesto Project party platform data."""
    print("\nRaw file hash (manifesto CSV):", sha256_file(paths.manifesto_csv))
    return pd.read_csv(paths.manifesto_csv, low_memory=False)


def load_rr(paths: Paths) -> pd.DataFrame:
    """Load Robustness & Reliability (R&R) alliance data."""
    print("\nRaw file hash (R&R DTA):", sha256_file(paths.rr_dta))
    return pd.read_stata(paths.rr_dta)

def build_ideology_panel(spec: pd.DataFrame, manifesto: pd.DataFrame, paths: Paths) -> pd.DataFrame:
    """
    Build country-year ideology panel from party manifesto data.

    Strategy:
    1. Link Manifesto country codes to COW codes via crosswalk
    2. Extract election years from manifesto data
    3. Identify governing party (largest by seat share, with vote share fallback)
    4. Expand to full country-year panel and forward-fill ideology scores
    5. Create binary right-of-center indicator (RILE >= 10)
    """
    print("\n--- Building ideology panel ---")
    _require_columns(manifesto, {"country", "edate", "date", "absseat", "totseats", "pervote", "rile"}, "Manifesto")

    # Link Manifesto countries to COW country codes
    print("\nRaw file hash (manifesto crosswalk CSV):", sha256_file(paths.manifesto_crosswalk_csv))
    crosswalk = _normalize_crosswalk(pd.read_csv(paths.manifesto_crosswalk_csv))

    manifesto = manifesto.merge(
        crosswalk,
        left_on="country",
        right_on="manifesto_country_code",
        how="left",
        indicator="_merge_indicator",
    )
    merge_audit(manifesto, "_merge_indicator", "Manifesto → COW crosswalk")
    manifesto = manifesto.drop(columns=["_merge_indicator"])

    if "country_code_cow" not in manifesto.columns:
        raise ValueError("Crosswalk merge failed to create 'country_code_cow' column.")

    # Extract election years and convert relevant columns to numeric
    manifesto["election_year"] = _parse_manifesto_election_year(manifesto)
    for col in ["absseat", "totseats", "pervote", "rile", "country_code_cow", "election_year"]:
        manifesto[col] = pd.to_numeric(manifesto[col], errors="coerce")

    if manifesto["election_year"].notna().any():
        year_range = (int(manifesto["election_year"].min()), int(manifesto["election_year"].max()))
        print(f"Election year range: {year_range[0]}–{year_range[1]}")

    # Measure party strength (seat share preferred, vote share as fallback)
    manifesto["party_strength"] = manifesto["absseat"] / manifesto["totseats"]
    missing_seats = manifesto["party_strength"].isna()
    manifesto.loc[missing_seats, "party_strength"] = manifesto.loc[missing_seats, "pervote"] / 100.0

    # Select governing party (strongest party per country-election)
    usable = manifesto.dropna(subset=["country_code_cow", "election_year", "party_strength"]).copy()
    usable["country_code_cow"] = usable["country_code_cow"].astype(int)
    usable["election_year"] = usable["election_year"].astype(int)
    usable = usable.sort_values(["country_code_cow", "election_year", "party_strength"], ascending=[True, True, False])
    governing = usable.groupby(["country_code_cow", "election_year"], as_index=False).first()

    # Keep only relevant columns
    columns_to_keep = ["country_code_cow", "election_year", "rile", "party_strength"]
    if "partyname" in governing.columns:
        columns_to_keep.append("partyname")
    governing = governing[columns_to_keep].copy()

    # Expand to full country-year panel and forward-fill ideology
    spine = spec[["country_code_cow", "year"]].drop_duplicates().sort_values(["country_code_cow", "year"])
    governing = governing.rename(columns={"election_year": "year"})

    panel = spine.merge(governing, on=["country_code_cow", "year"], how="left")
    panel["rile"] = pd.to_numeric(panel["rile"], errors="coerce")

    # Forward-fill ideology within each country (assumes persistence between elections)
    panel["rile"] = panel.groupby("country_code_cow", sort=False)["rile"].ffill()
    if "partyname" in panel.columns:
        panel["partyname"] = panel.groupby("country_code_cow", sort=False)["partyname"].ffill()
    if "party_strength" in panel.columns:
        panel["party_strength"] = panel.groupby("country_code_cow", sort=False)["party_strength"].ffill()

    # Create binary right-of-center indicator
    panel["right_of_center"] = np.where(
        panel["rile"] >= 10, 1,
        np.where(panel["rile"] <= -10, 0, np.nan)
    )

    # Create lagged ideology variables (force structure responds with delay)
    print("Creating lagged ideology variables (1-20 years)...")
    for lag in range(1, 21):
        panel[f"rile_lag{lag}"] = panel.groupby("country_code_cow", sort=False)["rile"].shift(lag)
        panel[f"right_of_center_lag{lag}"] = panel.groupby("country_code_cow", sort=False)["right_of_center"].shift(lag)

    assert_unique_key(panel, ["country_code_cow", "year"], "Ideology panel")
    coverage_audit(panel, "country_code_cow", "year", "Ideology panel")

    panel.to_csv(paths.ideology_out, index=False)
    print("Saved ideology panel:", paths.ideology_out)
    return panel


def build_rr_aggregates(rr: pd.DataFrame, paths: Paths) -> pd.DataFrame:
    """
    Aggregate dyadic alliance data to country-year level.

    For each country-year, we:
    - Average alliance characteristics across all dyads involving that country
    - Count the number of alliance dyads
    """
    print("\n--- Building R&R aggregates ---")
    _require_columns(rr, {"state_a", "state_b", "yrent"}, "R&R")

    # Clean and prepare dyadic data
    rr = rr.copy()
    rr["state_a"] = pd.to_numeric(rr["state_a"], errors="coerce")
    rr["state_b"] = pd.to_numeric(rr["state_b"], errors="coerce")
    rr["year"] = pd.to_numeric(rr["yrent"], errors="coerce")
    rr = rr.dropna(subset=["state_a", "state_b", "year"]).copy()
    rr["state_a"] = rr["state_a"].astype(int)
    rr["state_b"] = rr["state_b"].astype(int)
    rr["year"] = rr["year"].astype(int)

    # Handle duplicate dyad-years by averaging
    dyad_key = ["state_a", "state_b", "year"]
    if rr.duplicated(dyad_key).any():
        print("R&R: Averaging duplicate dyad-years (numeric columns only)")
        rr = rr.groupby(dyad_key, as_index=False).mean(numeric_only=True)

    # Identify feature columns (exclude IDs)
    id_columns = {"atopid", "state_a", "state_b", "year", "yrent", "dyadid"}
    feature_columns = [col for col in rr.columns if col not in id_columns]

    for col in feature_columns:
        rr[col] = pd.to_numeric(rr[col], errors="coerce")

    # Convert dyadic data to monadic (each country appears in both state_a and state_b roles)
    state_a_data = rr[["year", "state_a"] + feature_columns].rename(columns={"state_a": "country_code_cow"})
    state_b_data = rr[["year", "state_b"] + feature_columns].rename(columns={"state_b": "country_code_cow"})
    monadic = pd.concat([state_a_data, state_b_data], ignore_index=True)
    monadic["dyad_count"] = 1

    # Aggregate to country-year level
    grouped = monadic.groupby(["country_code_cow", "year"], as_index=False)
    aggregated = grouped[feature_columns].mean()
    aggregated = aggregated.merge(grouped["dyad_count"].sum(), on=["country_code_cow", "year"], how="left")

    # Add 'rr_' prefix to alliance variables for clarity
    rename_map = {col: f"rr_{col}" for col in aggregated.columns if col not in ["country_code_cow", "year", "dyad_count"]}
    aggregated = aggregated.rename(columns=rename_map)

    assert_unique_key(aggregated, ["country_code_cow", "year"], "R&R aggregates")
    coverage_audit(aggregated, "country_code_cow", "year", "R&R aggregates")

    aggregated.to_csv(paths.rr_out, index=False)
    print("Saved R&R aggregates:", paths.rr_out)
    return aggregated

def build_master() -> None:
    """
    Build the master country-year dataset for analysis.

    Steps:
    1. Load all raw data sources
    2. Build ideology panel from party manifestos
    3. Aggregate alliance data to country-year level
    4. Merge everything together
    5. Filter to observations with ideology data
    """
    paths = Paths()

    # Load raw data
    spec = load_spec(paths)
    manifesto = load_manifesto(paths)
    rr = load_rr(paths)

    print_schema("Specialization (raw spine)", spec)
    print_schema("Manifesto (raw)", manifesto)
    print_schema("R&R (raw)", rr)

    # Build intermediate datasets
    ideology = build_ideology_panel(spec, manifesto, paths)
    alliance_aggregates = build_rr_aggregates(rr, paths)

    # Merge specialization data with ideology
    master = spec.merge(ideology, on=["country_code_cow", "year"], how="left", indicator="_merge_ideology")
    merge_audit(master, "_merge_ideology", "Specialization + Ideology")
    master = master.drop(columns=["_merge_ideology"])

    # Filter to observations with ideology data (required for analysis)
    rows_before = len(master)
    master = master.dropna(subset=["rile"]).copy()
    rows_after = len(master)
    print(f"\nFiltered to observations with ideology: {rows_after} of {rows_before} rows ({rows_after/rows_before:.1%})")

    # Add alliance data
    master = master.merge(alliance_aggregates, on=["country_code_cow", "year"], how="left", indicator="_merge_alliance")
    merge_audit(master, "_merge_alliance", "Master + Alliance Data")
    master = master.drop(columns=["_merge_alliance"])

    # Create control variables for analysis
    print("\nCreating control variables...")

    # Alliance variables: standardize naming and create log-transformed versions
    # R&R variables have 'rr_' prefix from build_rr_aggregates
    if "rr_milex" in master.columns:
        master["milex_allies"] = master["rr_milex"]
        master["ln_milex_allies"] = _safe_log(master["milex_allies"])
    elif "milex_allies" in master.columns:
        master["ln_milex_allies"] = _safe_log(master["milex_allies"])

    if "rr_cinc_ratio" in master.columns:
        master["cinc_allies_ratio"] = master["rr_cinc_ratio"]
    # cinc_allies_ratio might already exist, ensure numeric
    if "cinc_allies_ratio" in master.columns:
        master["cinc_allies_ratio"] = pd.to_numeric(master["cinc_allies_ratio"], errors="coerce")

    # War variable: create 5-year lag if not already present
    if "interstatewar_5yrlag_binary" in master.columns:
        war_col = master["interstatewar_5yrlag_binary"]
        master["war5_lag"] = pd.to_numeric(
            war_col.astype(str) if war_col.dtype in ["category", object] else war_col,
            errors="coerce"
        )
    elif "interstatewar_binary" in master.columns and "war5_lag" not in master.columns:
        master["war5_lag"] = master.groupby("country_code_cow")["interstatewar_binary"].shift(5)
        master["war5_lag"] = pd.to_numeric(master["war5_lag"], errors="coerce")

    # GDP: create log version if not already present
    if "lngdp_WDI_full" in master.columns:
        master["lngdp"] = pd.to_numeric(master["lngdp_WDI_full"], errors="coerce")
    elif "gdp_WDI_full" in master.columns and "lngdp" not in master.columns:
        master["lngdp"] = _safe_log(master["gdp_WDI_full"])

    # CINC: standardize variable name
    if "cinc_MC" in master.columns:
        master["cinc"] = pd.to_numeric(master["cinc_MC"], errors="coerce")
    elif "cinc" in master.columns:
        master["cinc"] = pd.to_numeric(master["cinc"], errors="coerce")

    # Time trend variables (for polynomial time controls)
    master["year_num"] = pd.to_numeric(master["year"], errors="coerce")
    master["t"] = master["year_num"] - master["year_num"].min()
    master["t2"] = master["t"] ** 2
    master["t3"] = master["t"] ** 3

    # Select specialization measure (prefer standardized version)
    specialization_candidates = ["spec_stand", "spec_intscale", "spec_raw"]
    available = [col for col in specialization_candidates if col in master.columns]
    if available:
        master["spec_y"] = pd.to_numeric(master[available[0]], errors="coerce")
        print(f"Created spec_y from {available[0]}")

    assert_unique_key(master, ["country_code_cow", "year"], "Master dataset")
    coverage_audit(master, "country_code_cow", "year", "Master dataset")

    master.to_csv(paths.master_out, index=False)
    print("Saved master dataset:", paths.master_out)


def main() -> int:
    build_master()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
