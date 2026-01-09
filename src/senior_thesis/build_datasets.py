"""
Dataset construction for military specialization and alliance institutions study.

Creates analysis datasets:
1. master_country_year.csv - For testing H1 (ideology -> specialization)
2. master_dyad_year.csv - Full dyad-year panel (1970-2014)
3. master_dyad_year_gannon_1980_2010.csv - Aligned sample for H2/H3 (1980-2010, DCA observed)

CRITICAL DATA NOTES:
- DCAD (Defense Cooperation Agreements) covers only 1980-2010
  - Years outside this window: any_dca_link = NA (unknown, not absent)
  - Years inside this window: any_dca_link = 0 or 1
- Dyad-year collapse uses invariance checks (not unsafe "first")
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pyreadr
import statsmodels.api as sm

from senior_thesis.config import (
    Paths,
    YEAR_START,
    YEAR_END,
    RILE_RIGHT_THRESHOLD,
    RILE_LEFT_THRESHOLD,
)
from senior_thesis.utils import (
    file_hash,
    assert_unique_key,
    merge_report,
    coverage_report,
)

# DCAD coverage window - outside this range, DCA status is UNKNOWN (not absent)
DCAD_YEAR_START: int = 1980
DCAD_YEAR_END: int = 2010

# Provenance control: forbidden column patterns from external alliance-type coding
# These patterns match columns that would indicate RR or other non-ATOP alliance type sources
import re
FORBIDDEN_COL_PATTERNS = [
    r'(?i)\brr\b',            # RR dataset references
    r'(?i)\brapport\b',       # Rapport & Rathbun
    r'(?i)\brathbun\b',
    r'(?i)\balliance_type\b', # Pre-coded alliance types
    r'(?i)\balliance_depth\b',
    r'(?i)\bgovernance\b',    # Governance coding from external sources
    r'(?i)\bdepth\b',         # Alliance depth from external sources
]
# ALLOWED exceptions: our own derived variables
ALLOWED_INST_COLS = {
    'inst', 'inst_max', 'hierarchical', 'voice_driven', 'uninstitutionalized',
    'vertical_integration', 'inst_dyad_year', 'inst_alliance', 'n_shared_alliances',
    'atopid_max_inst', 'atopid_first'
}

__all__ = [
    "build_country_year",
    "build_dyad_year",
    "build_dyad_year_gannon_union",
    "build_dyad_year_h3",
    "build_all",
]


def _assert_no_forbidden_columns(df: pd.DataFrame, dataset_name: str) -> None:
    """
    Provenance assertion: ensure no forbidden alliance-type columns from external sources.

    Alliance-type classification (hierarchical / voice-driven / uninstitutionalized)
    must be derived ONLY from ATOP treaty provisions. This function raises ValueError
    if any columns from RR (Rapport-Rathbun) or other external alliance-type coding
    are detected.

    Args:
        df: DataFrame to check
        dataset_name: Name for error messages

    Raises:
        ValueError: If forbidden columns are detected
    """
    forbidden_found = []
    for col in df.columns:
        # Skip allowed columns (our own derivations)
        if col in ALLOWED_INST_COLS:
            continue
        # Check against forbidden patterns
        for pattern in FORBIDDEN_COL_PATTERNS:
            if re.search(pattern, col):
                forbidden_found.append(col)
                break

    if forbidden_found:
        raise ValueError(
            f"PROVENANCE VIOLATION in {dataset_name}:\n"
            f"Forbidden alliance-type columns detected: {forbidden_found}\n"
            f"Alliance-type must be derived ONLY from ATOP provisions.\n"
            f"Remove these columns or rename if they are legitimately ATOP-derived."
        )


def _assert_inst_from_atop_only(df: pd.DataFrame, dataset_name: str) -> None:
    """
    Verify that inst-related variables exist and are properly typed.

    This doesn't verify derivation logic (that's in _build_institutionalization),
    but confirms that the expected variables are present and reasonable.

    Args:
        df: DataFrame to check
        dataset_name: Name for error messages

    Raises:
        ValueError: If inst variables are missing or invalid
    """
    # Check that inst-related columns use integer codes 0-3
    inst_cols = [c for c in df.columns if c in {'inst', 'inst_max', 'vertical_integration'}]
    for col in inst_cols:
        if col in df.columns:
            valid_values = {0, 1, 2, 3}
            unique_values = set(df[col].dropna().unique())
            invalid = unique_values - valid_values
            if invalid:
                raise ValueError(
                    f"PROVENANCE VIOLATION in {dataset_name}:\n"
                    f"Column '{col}' has unexpected values: {invalid}\n"
                    f"Expected only {valid_values} (0=DCA-only, 1=uninst, 2=voice, 3=hier)"
                )

    # Check that hierarchical/voice_driven are binary if present
    for col in ['hierarchical', 'voice_driven']:
        if col in df.columns:
            unique_values = set(df[col].dropna().unique())
            if not unique_values.issubset({0, 1, True, False}):
                raise ValueError(
                    f"PROVENANCE VIOLATION in {dataset_name}:\n"
                    f"Column '{col}' has non-binary values: {unique_values}"
                )


def _safe_log(x: pd.Series) -> pd.Series:
    """Natural log, setting non-positive values to NaN."""
    return np.log(x.where(x > 0))


def _load_specialization(paths: Paths) -> pd.DataFrame:
    """Load Gannon's military specialization assets (country-year spine)."""
    print(f"  Loading specialization assets [{file_hash(paths.spec_rds)}]")
    df = list(pyreadr.read_r(paths.spec_rds).values())[0].copy()

    # Clean keys
    df["country_code_cow"] = pd.to_numeric(df["country_code_cow"], errors="coerce")
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["country_code_cow", "year"])
    df["country_code_cow"] = df["country_code_cow"].astype(int)
    df["year"] = df["year"].astype(int)

    return df


def _load_manifesto(paths: Paths) -> pd.DataFrame:
    """Load Manifesto Project party ideology assets."""
    print(f"  Loading Manifesto assets [{file_hash(paths.manifesto_csv)}]")
    return pd.read_csv(paths.manifesto_csv, low_memory=False)


def _load_crosswalk(paths: Paths) -> pd.DataFrame:
    """Load Manifesto-to-COW country code crosswalk."""
    print(f"  Loading crosswalk [{file_hash(paths.crosswalk_csv)}]")
    cw = pd.read_csv(paths.crosswalk_csv)

    # Normalize column names
    if "cow_ccode" in cw.columns:
        cw = cw.rename(columns={"cow_ccode": "country_code_cow"})

    cw = cw[["manifesto_country_code", "country_code_cow"]].dropna()
    cw["manifesto_country_code"] = cw["manifesto_country_code"].astype(int)
    cw["country_code_cow"] = cw["country_code_cow"].astype(int)

    return cw


def _load_atop(paths: Paths) -> pd.DataFrame:
    """Load ATOP alliance membership assets."""
    print(f"  Loading ATOP membership [{file_hash(paths.atop_csv)}]")
    return pd.read_csv(paths.atop_csv, low_memory=False)


def _load_contiguity(paths: Paths) -> pd.DataFrame:
    """
    Load COW Direct Contiguity dataset.

    Creates binary contiguity indicator (1 = land contiguous, conttype=1).
    Normalizes state ordering (state_a < state_b) to match dyad panel.
    """
    print(f"  Loading contiguity [{file_hash(paths.contiguity_csv)}]")
    df = pd.read_csv(paths.contiguity_csv, low_memory=False)

    # Rename columns
    df = df.rename(columns={"state1no": "s1", "state2no": "s2"})

    # Create binary: 1 if land contiguous (conttype=1), 0 otherwise
    df["contiguous"] = (df["conttype"] == 1).astype(int)

    # Normalize ordering (state_a < state_b)
    df["state_a"] = df[["s1", "s2"]].min(axis=1)
    df["state_b"] = df[["s1", "s2"]].max(axis=1)

    # Keep unique dyad-year rows (take max contiguity if multiple entries)
    df = df.groupby(["state_a", "state_b", "year"], as_index=False)["contiguous"].max()

    return df[["state_a", "state_b", "year", "contiguous"]]


def _load_nmc(paths: Paths) -> pd.DataFrame:
    """
    Load COW National Material Capabilities v6.0.

    Creates country-year panel with military expenditure (milex).
    Following Gannon (2023), we use milex for ratio controls rather than CINC,
    as CINC includes demographic factors less relevant to military burden-sharing.

    Key variables:
    - ccode: COW country code
    - year: Year
    - milex: Military expenditure (thousands of current-year USD)
    """
    print(f"  Loading NMC 6.0 [{file_hash(paths.nmc_csv)}]")
    # NMC 6.0 has some non-UTF-8 characters in country names, use latin-1 encoding
    df = pd.read_csv(paths.nmc_csv, low_memory=False, encoding="latin-1")

    # Select and clean key variables
    df = df[["ccode", "year", "milex"]].copy()
    df["ccode"] = pd.to_numeric(df["ccode"], errors="coerce")
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["milex"] = pd.to_numeric(df["milex"], errors="coerce")

    # Drop rows with missing identifiers
    df = df.dropna(subset=["ccode", "year"])
    df["ccode"] = df["ccode"].astype(int)
    df["year"] = df["year"].astype(int)

    # Note: milex can be missing (-9 or NaN in original data)
    # Keep NaN values; they'll propagate through ratio calculations
    df.loc[df["milex"] < 0, "milex"] = np.nan

    n_nonmissing = df["milex"].notna().sum()
    print(f"    Country-years with milex: {n_nonmissing:,} ({n_nonmissing/len(df):.1%})")

    return df


def _load_dcad(paths: Paths) -> pd.DataFrame:
    """
    Load Defense Cooperation Agreement Dataset (DCAD) v1.0 dyadic data.

    Creates dyad-year panel indicating DCA ties between country pairs.
    DCAs are bilateral defense agreements outside of ATOP alliance treaties,
    capturing informal security cooperation.

    Key variables:
    - ccode1, ccode2: COW country codes
    - year: Year
    - dcaAnyV2: Binary indicator for any DCA (most inclusive measure)

    Normalizes state ordering (state_a < state_b) to match alliance panel.
    """
    print(f"  Loading DCAD [{file_hash(paths.dcad_csv)}]")
    df = pd.read_csv(paths.dcad_csv, low_memory=False)

    # Select key variables - use dcaAnyV2 as most inclusive DCA measure
    df = df[["ccode1", "ccode2", "year", "dcaAnyV2"]].copy()
    df = df.rename(columns={"dcaAnyV2": "has_dca"})

    # Clean types
    for col in ["ccode1", "ccode2", "year"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["ccode1", "ccode2", "year"])
    df["ccode1"] = df["ccode1"].astype(int)
    df["ccode2"] = df["ccode2"].astype(int)
    df["year"] = df["year"].astype(int)
    df["has_dca"] = df["has_dca"].fillna(0).astype(int)

    # Normalize ordering (state_a < state_b)
    df["state_a"] = df[["ccode1", "ccode2"]].min(axis=1)
    df["state_b"] = df[["ccode1", "ccode2"]].max(axis=1)

    # Keep unique dyad-year rows (take max if duplicates)
    df = df.groupby(["state_a", "state_b", "year"], as_index=False)["has_dca"].max()

    n_dca = df["has_dca"].sum()
    print(f"    Dyad-years with DCA: {n_dca:,} ({n_dca/len(df):.1%})")
    print(f"    Year range: {df['year'].min()}-{df['year'].max()}")

    return df[["state_a", "state_b", "year", "has_dca"]]


def _build_ideology_panel(
    spec: pd.DataFrame,
    manifesto: pd.DataFrame,
    crosswalk: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build country-year ideology panel from party manifestos.

    Strategy:
    1. Link Manifesto countries to COW codes
    2. Identify governing party (largest seat share) per country-election
    3. Expand to country-year panel, forward-filling between elections
    4. Create right_of_center binary (RILE >= threshold -> 1, RILE <= -threshold -> 0)
    """
    # Link to COW codes
    mf = manifesto.merge(
        crosswalk, left_on="country", right_on="manifesto_country_code", how="inner"
    )

    # Parse election year from edate (primary) or date column (fallback)
    year_from_edate = pd.to_datetime(
        mf["edate"], errors="coerce", dayfirst=True
    ).dt.year
    year_from_date = (pd.to_numeric(mf["date"], errors="coerce") // 100).astype("Int64")
    mf["election_year"] = year_from_edate.fillna(year_from_date)

    # Compute party strength (seat share, vote share fallback)
    mf["strength"] = mf["absseat"] / mf["totseats"]
    no_seats = mf["strength"].isna()
    mf.loc[no_seats, "strength"] = mf.loc[no_seats, "pervote"] / 100

    # Create nationalism score (per601 - per602)
    if "per601" in mf.columns and "per602" in mf.columns:
        mf["nationalism"] = mf["per601"] - mf["per602"]
    else:
        mf["nationalism"] = np.nan

    # Create military attitude score (per104 - per105)
    if "per104" in mf.columns and "per105" in mf.columns:
        mf["military_positive"] = mf["per104"] - mf["per105"]
    else:
        mf["military_positive"] = np.nan

    # Select governing party (strongest per country-election)
    mf = mf.dropna(subset=["country_code_cow", "election_year", "strength"])
    mf["country_code_cow"] = mf["country_code_cow"].astype(int)
    mf["election_year"] = mf["election_year"].astype(int)
    mf = mf.sort_values(
        ["country_code_cow", "election_year", "strength"], ascending=[True, True, False]
    )
    gov = mf.groupby(["country_code_cow", "election_year"], as_index=False).first()
    gov = gov[["country_code_cow", "election_year", "rile", "nationalism", "military_positive"]].rename(
        columns={"election_year": "year"}
    )

    # Expand to country-year spine
    spine = spec[["country_code_cow", "year"]].drop_duplicates()
    panel = spine.merge(gov, on=["country_code_cow", "year"], how="left")

    # Forward-fill ideology within country
    panel = panel.sort_values(["country_code_cow", "year"])
    panel["rile"] = panel.groupby("country_code_cow")["rile"].ffill()
    panel["nationalism"] = panel.groupby("country_code_cow")["nationalism"].ffill()
    panel["military_positive"] = panel.groupby("country_code_cow")["military_positive"].ffill()

    # Create binary indicator using config thresholds
    panel["right_of_center"] = np.where(
        panel["rile"] >= RILE_RIGHT_THRESHOLD,
        1,
        np.where(panel["rile"] <= RILE_LEFT_THRESHOLD, 0, np.nan),
    )

    return panel


def _merge_by_partner(
    df: pd.DataFrame,
    data: pd.DataFrame,
    var: str,
    key_col: str = "country_code_cow",
) -> pd.DataFrame:
    """
    Merge assets for both dyad partners (state_a and state_b).

    Args:
        df: Target dataframe with state_a, state_b, and year columns
        data: Source dataframe with key_col, year, and var columns
        var: Variable name to merge (will become var_a and var_b)
        key_col: Key column name in source assets

    Returns:
        DataFrame with var_a and var_b columns added
    """
    for suffix in ["a", "b"]:
        df = df.merge(
            data.rename(columns={key_col: f"state_{suffix}", var: f"{var}_{suffix}"}),
            on=[f"state_{suffix}", "year"],
            how="left",
        )
    return df


def _expand_dyad_years(dyads: pd.DataFrame) -> pd.DataFrame:
    """
    Expand ATOP dyads to dyad-year panel using vectorized operations.

    Args:
        dyads: ATOP dyad data with dyad_start and dyad_end columns

    Returns:
        Expanded dyad-year panel with atopid, state_a, state_b, year
    """
    # Calculate number of years for each dyad
    dyads = dyads.copy()
    dyads["n_years"] = (dyads["dyad_end"] - dyads["dyad_start"] + 1).clip(lower=0)

    # Repeat rows by number of years
    expanded = dyads.loc[dyads.index.repeat(dyads["n_years"])].copy()

    # Generate year sequence within each original row
    expanded["year"] = expanded.groupby(level=0).cumcount() + expanded["dyad_start"]
    expanded["year"] = expanded["year"].astype(int)

    # Reset index and select key columns
    expanded = expanded.reset_index(drop=True)

    return expanded[["atopid", "state_a", "state_b", "year"]]


def build_country_year(paths: Paths) -> pd.DataFrame:
    """
    Build country-year dataset for H1 (ideology -> specialization).

    Output variables:
    - country_code_cow, year (IDs)
    - spec_y (DV: standardized specialization)
    - rile, right_of_center (IV: ideology)
    - lngdp, cinc, war5_lag (controls)
    """
    print("\n  Building country-year dataset...")

    # Load raw assets
    spec = _load_specialization(paths)
    manifesto = _load_manifesto(paths)
    crosswalk = _load_crosswalk(paths)

    # Build ideology panel
    ideology = _build_ideology_panel(spec, manifesto, crosswalk)

    # Merge with specialization assets
    master = spec.merge(ideology, on=["country_code_cow", "year"], how="left", indicator="_m")
    merge_report(master, "_m", "spec + ideology")
    master = master.drop(columns=["_m"])

    # Filter to observations with ideology assets
    n_before = len(master)
    master = master.dropna(subset=["rile"])
    print(f"  Filtered to rows with RILE: {len(master):,} of {n_before:,}")

    # Create/rename key variables
    if "spec_stand" in master.columns:
        master["spec_y"] = master["spec_stand"]
    elif "spec_intscale" in master.columns:
        master["spec_y"] = master["spec_intscale"]

    # GDP variables: preserve BOTH log and level forms
    # Level form is needed for Gannon-style bounded parity ratios
    if "lngdp_WDI_full" in master.columns:
        master["lngdp"] = master["lngdp_WDI_full"]
        # Back-transform to get level (GDP is in billions typically)
        master["gdp_level"] = np.exp(master["lngdp_WDI_full"])
    elif "gdp_WDI_full" in master.columns:
        master["gdp_level"] = master["gdp_WDI_full"]  # Keep raw level
        master["lngdp"] = _safe_log(master["gdp_WDI_full"])

    if "cinc_MC" in master.columns:
        master["cinc"] = master["cinc_MC"]

    if "interstatewar_5yrlag_binary" in master.columns:
        master["war5_lag"] = pd.to_numeric(
            master["interstatewar_5yrlag_binary"], errors="coerce"
        )

    # Create lagged and lead ideology variables for master regression and placebo test
    # Sort by country and year to ensure correct lag/lead computation
    master = master.sort_values(["country_code_cow", "year"])

    # rile_lag5: ideology from 5 years ago (main IV for master regression)
    # 5-year lag accounts for defense procurement cycles
    master["rile_lag5"] = master.groupby("country_code_cow")["rile"].shift(5)

    # rile_lead1: ideology from NEXT year (for placebo test)
    # If our model is correct, future ideology should NOT predict current specialization
    master["rile_lead1"] = master.groupby("country_code_cow")["rile"].shift(-1)

    # Report coverage of lagged variables
    n_lag1 = master["rile_lag5"].notna().sum()
    n_lead1 = master["rile_lead1"].notna().sum()
    print(f"  Created ideology lags/leads:")
    print(f"    rile_lag5:  {n_lag1:,} ({n_lag1/len(master):.1%})")
    print(f"    rile_lead1: {n_lead1:,} ({n_lead1/len(master):.1%})")

    # Verify and save
    assert_unique_key(master, ["country_code_cow", "year"], "country-year")
    coverage_report(master, "country_code_cow", "year", "country-year")

    # Save preliminary version (will add in_alliance after dyad assets is built)
    master.to_csv(paths.country_year_csv, index=False)
    print(f"  Saved: {paths.country_year_csv}")

    return master


def _add_alliance_membership(paths: Paths) -> None:
    """
    Add alliance membership indicators to country-year assets.

    Creates:
    - in_alliance: 1 if country is in any alliance
    - in_hierarchical: 1 if country is in any hierarchical alliance (inst=3)
    - in_voice: 1 if country is in any voice-driven alliance (inst=2)
    - in_uninst: 1 if country is in any uninstitutionalized alliance (inst=1)

    Must be called after dyad-year assets is built.
    """
    # Load datasets
    cy = pd.read_csv(paths.country_year_csv)
    dy = pd.read_csv(paths.dyad_year_csv)

    # Helper to get country-years in alliances of a given type
    def get_alliance_indicator(dy_subset, col_name):
        a = dy_subset[["state_a", "year"]].drop_duplicates()
        a.columns = ["country_code_cow", "year"]
        b = dy_subset[["state_b", "year"]].drop_duplicates()
        b.columns = ["country_code_cow", "year"]
        result = pd.concat([a, b]).drop_duplicates()
        result[col_name] = 1
        return result

    # All alliances
    in_alliance = get_alliance_indicator(dy, "in_alliance")

    # By type
    in_hierarchical = get_alliance_indicator(dy[dy["inst"] == 3], "in_hierarchical")
    in_voice = get_alliance_indicator(dy[dy["inst"] == 2], "in_voice")
    in_uninst = get_alliance_indicator(dy[dy["inst"] == 1], "in_uninst")

    # Merge all indicators
    cy = cy.merge(in_alliance, on=["country_code_cow", "year"], how="left")
    cy = cy.merge(in_hierarchical, on=["country_code_cow", "year"], how="left")
    cy = cy.merge(in_voice, on=["country_code_cow", "year"], how="left")
    cy = cy.merge(in_uninst, on=["country_code_cow", "year"], how="left")

    # Fill NaN with 0
    for col in ["in_alliance", "in_hierarchical", "in_voice", "in_uninst"]:
        cy[col] = cy[col].fillna(0).astype(int)

    # Report
    n_alliance = cy["in_alliance"].sum()
    n_hier = cy["in_hierarchical"].sum()
    n_voice = cy["in_voice"].sum()
    n_uninst = cy["in_uninst"].sum()
    print(f"  Alliance membership added:")
    print(f"    in_alliance:     {n_alliance:,} ({n_alliance/len(cy):.1%})")
    print(f"    in_hierarchical: {n_hier:,} ({n_hier/len(cy):.1%})")
    print(f"    in_voice:        {n_voice:,} ({n_voice/len(cy):.1%})")
    print(f"    in_uninst:       {n_uninst:,} ({n_uninst/len(cy):.1%})")

    cy.to_csv(paths.country_year_csv, index=False)


def _build_institutionalization(atop: pd.DataFrame) -> pd.DataFrame:
    """
    Build alliance institutionalization using Leeds & Anac (2005) treaty-provision criteria.

    This is a 3-category NOMINAL outcome (not ordinal) reflecting distinct governance modes.
    Hierarchy dominates voice; voice dominates absence.

    Variables used (ATOP codebook):
        INTCOM, MILCON, BASE, SUBORD, ORGAN1, ORGPURP1, ORGAN2, ORGPURP2, MILAID, CONTRIB

    Categories (mutually exclusive, applied in order):

        1. "hierarchical" - Authority, command, or structural control provisions:
           - INTCOM == 1 (integrated command in peacetime and wartime)
           - MILCON == 3 (common defense policy: doctrine, training, procurement, joint planning)
           - BASE > 0 (joint or unilateral troop placement / basing)
           - SUBORD in {1,2} (explicit subordination of forces during conflict)

        2. "voice_driven" - Coordination/consultation mechanisms (if NOT hierarchical):
           - MILCON == 2 (peacetime military consultation)
           - mil_org_present: (ORGAN1 in {1,2,3} & ORGPURP1==1) OR (ORGAN2 in {1,2,3} & ORGPURP2==1)
           - MILAID in {3,4} (training and/or technology transfer)
           - CONTRIB == 1 (specified troop/supply/funding contributions)

        3. "uninstitutionalized" - None of the above provisions

    Missing value handling:
        Missing values are treated as "provision absent" (equivalent to 0) following ATOP
        coding conventions. This is explicitly documented behavior, not silent coercion.
        Alliances with missing values on ALL institutional variables are flagged.
    """
    # =========================================================================
    # DEFENSIVE ASSERTIONS: Ensure inst is calculated from raw ATOP provisions
    # =========================================================================

    # Required ATOP provision columns (must all be present)
    REQUIRED_PROVISION_COLS = [
        "intcom", "milcon", "base", "subord",
        "organ1", "orgpurp1", "organ2", "orgpurp2",
        "milaid", "contrib"
    ]

    # Forbidden columns that should NOT be used
    FORBIDDEN_COLS = ["conwtin", "medarb", "armred", "acqterr", "divgains", "demwith"]

    # Convert column names to lowercase for comparison
    atop_cols_lower = [c.lower() for c in atop.columns]

    # Assertion 1: Verify all required provision columns exist
    missing_cols = [c for c in REQUIRED_PROVISION_COLS if c not in atop_cols_lower]
    if missing_cols:
        raise ValueError(
            f"ATOP data missing required provision columns: {missing_cols}. "
            f"inst MUST be calculated from raw ATOP treaty provisions."
        )

    # Assertion 2: Reject any pre-coded inst column in input
    if "inst" in atop_cols_lower:
        raise ValueError(
            "ATOP data contains pre-coded 'inst' column. "
            "inst MUST be calculated from raw treaty provisions, not imported. "
            "Remove the inst column from the input data."
        )

    # Assertion 3: Warn about forbidden columns (don't error, just warn)
    present_forbidden = [c for c in FORBIDDEN_COLS if c in atop_cols_lower]
    if present_forbidden:
        print(f"    Warning: Forbidden columns present but NOT used: {present_forbidden}")

    # =========================================================================
    # CALCULATION: Build inst from raw ATOP provisions
    # =========================================================================

    # Select relevant variables
    # IMPORTANT: ATOP codes some variables at the member level (e.g., base rights).
    # We aggregate by alliance, taking MAX of indicators since ANY member having
    # a provision means the alliance has that provision.
    inst_vars = [
        "atopid", "intcom", "milcon", "base", "subord",
        "organ1", "orgpurp1", "organ2", "orgpurp2", "milaid", "contrib"
    ]

    # Aggregate by alliance: take MAX of each institutional variable
    # This ensures that if ANY member has base=3, the alliance gets base=3
    inst = atop[inst_vars].groupby("atopid").max().reset_index()

    # Track which alliances have missing data before filling
    provision_cols = inst_vars[1:]  # exclude atopid
    n_missing_per_alliance = inst[provision_cols].isna().sum(axis=1)
    all_missing = n_missing_per_alliance == len(provision_cols)
    if all_missing.sum() > 0:
        print(f"    Warning: {all_missing.sum()} alliances have ALL provisions missing")

    # Fill missing values with 0 (ATOP convention: missing = provision absent)
    for col in provision_cols:
        inst[col] = inst[col].fillna(0)

    # Helper: military organization present for coordination purposes
    # ORGPURP == 1 means "military coordination" purpose
    mil_org_present = (
        ((inst["organ1"].isin([1, 2, 3])) & (inst["orgpurp1"] == 1)) |
        ((inst["organ2"].isin([1, 2, 3])) & (inst["orgpurp2"] == 1))
    )

    # Category 1: Hierarchical (authority-based governance)
    # Includes ANY provision creating authority, command, or structural control
    is_hierarchical = (
        (inst["intcom"] == 1) |           # Integrated command
        (inst["milcon"] == 3) |           # Common defense policy
        (inst["base"] > 0) |              # Any basing arrangement
        (inst["subord"].isin([1, 2]))     # Subordination of forces
    )

    # Category 2: Voice-driven (coordination mechanisms, if NOT hierarchical)
    is_voice_driven = (
        (inst["milcon"] == 2) |           # Peacetime consultation
        mil_org_present |                  # Military coordinating organization
        (inst["milaid"].isin([3, 4])) |   # Training/technology transfer
        (inst["contrib"] == 1)             # Contribution requirements
    )

    # Assign categories (hierarchical > voice_driven > uninstitutionalized)
    # Using integer codes: 1=uninst, 2=voice, 3=hierarchical
    inst["inst"] = 1  # default: uninstitutionalized
    inst.loc[is_voice_driven, "inst"] = 2  # voice_driven
    inst.loc[is_hierarchical, "inst"] = 3  # hierarchical (overrides voice)

    # =========================================================================
    # OUTPUT ASSERTION: Verify calculated values are valid
    # =========================================================================
    assert inst["inst"].isin([1, 2, 3]).all(), (
        "inst contains invalid values. Expected only {1, 2, 3}. "
        f"Found: {inst['inst'].unique()}"
    )

    return inst[["atopid", "inst"]]


def _check_invariance(
    panel: pd.DataFrame,
    cols: list[str],
    group_cols: list[str],
) -> None:
    """
    Check that specified columns are invariant within groups.

    Raises ValueError with details if any column varies within a group.
    """
    violations = []
    for col in cols:
        if col not in panel.columns:
            continue
        # Check nunique within each group (treating NA as a distinct value)
        nunique = panel.groupby(group_cols)[col].nunique(dropna=False)
        violating_groups = nunique[nunique > 1]
        if len(violating_groups) > 0:
            # Get example
            example_group = violating_groups.index[0]
            example_rows = panel[
                (panel[group_cols[0]] == example_group[0]) &
                (panel[group_cols[1]] == example_group[1]) &
                (panel[group_cols[2]] == example_group[2])
            ][[col] + group_cols + ["atopid"]].head(5)
            violations.append({
                "column": col,
                "n_violations": len(violating_groups),
                "example_group": example_group,
                "example_values": example_rows[col].unique().tolist()
            })

    if violations:
        msg = "INVARIANCE CHECK FAILED - columns that should be constant within dyad-year vary:\n"
        for v in violations:
            msg += f"\n  Column: {v['column']}\n"
            msg += f"    Violations: {v['n_violations']} dyad-years\n"
            msg += f"    Example group: {v['example_group']}\n"
            msg += f"    Distinct values: {v['example_values']}\n"
        raise ValueError(msg)


def build_dyad_year(paths: Paths) -> pd.DataFrame:
    """
    Build dyad-year dataset for H2 (alliance type -> division of labor).

    Uses Leeds & Anac (2005) institutionalization coding from ATOP treaty provisions.
    This is a 3-category NOMINAL classification (not ordinal):

    - Hierarchical (inst=3): INTCOM=1 OR MILCON=3 OR BASE>0 OR SUBORD in {1,2}
    - Voice-driven (inst=2): MILCON=2 OR military org present OR MILAID in {3,4} OR CONTRIB=1
    - Uninstitutionalized (inst=1): None of above

    CRITICAL FIXES:
    1. DCAD coverage window: any_dca_link = NA outside 1980-2010 (unknown, not absent)
    2. Safe collapse: invariance checks before using "first" aggregation
    3. Creates aligned 1980-2010 sample for H2/H3 Gannon+DCA analyses

    Output variables:
    - atopid, state_a, state_b, year (IDs)
    - div_labor (DV: pairwise portfolio complementarity, 0=identical, 1=fully dissimilar)
    - hierarchical, voice_driven (IVs, nominal dummies)
    - any_atop_link, any_dca_link, any_alignment, dca_only (alignment indicators)
    """
    print("\n  Building dyad-year dataset...")
    print("  Using Leeds & Anac (2005) institutionalization coding from ATOP")
    print(f"  DCAD coverage window: {DCAD_YEAR_START}-{DCAD_YEAR_END} (outside = NA)")

    # Load raw assets
    atop = _load_atop(paths)

    # Load country-year master for specialization and ideology
    master_cy = pd.read_csv(paths.country_year_csv)

    # Clean ATOP assets
    atop = atop.copy()
    for col in ["atopid", "member", "yrent", "yrexit"]:
        atop[col] = pd.to_numeric(atop[col], errors="coerce")
    atop = atop.dropna(subset=["atopid", "member", "yrent"])
    atop["atopid"] = atop["atopid"].astype(int)
    atop["member"] = atop["member"].astype(int)
    atop["yrent"] = atop["yrent"].astype(int)
    atop["yrexit"] = atop["yrexit"].fillna(YEAR_END).replace(0, YEAR_END).astype(int)

    # Filter to meaningful security cooperation: offense or defense pacts only
    # Excludes: neutrality pacts, non-aggression pacts, consultation-only agreements
    n_before = atop["atopid"].nunique()
    atop["offense"] = pd.to_numeric(atop["offense"], errors="coerce").fillna(0)
    atop["defense"] = pd.to_numeric(atop["defense"], errors="coerce").fillna(0)
    atop = atop[(atop["defense"] == 1) | (atop["offense"] == 1)]
    n_after = atop["atopid"].nunique()
    print(f"  Filtered to offense/defense pacts: {n_after} of {n_before} alliances")

    # Build institutionalization from ATOP variables
    inst_df = _build_institutionalization(atop)
    print(
        f"  ATOP alliances: {len(inst_df)} "
        f"(uninst={int((inst_df['inst']==1).sum())}, "
        f"voice={int((inst_df['inst']==2).sum())}, "
        f"hier={int((inst_df['inst']==3).sum())})"
    )

    # Build all dyads from ATOP members
    # First consolidate member entry/exit (some have multiple phases)
    members = (
        atop.groupby(["atopid", "member"])
        .agg({"yrent": "min", "yrexit": "max"})
        .reset_index()
    )

    dyads = members.merge(members, on="atopid", suffixes=("_a", "_b"))
    dyads = dyads[dyads["member_a"] < dyads["member_b"]]  # avoid duplicates/self-pairs
    dyads = dyads.rename(columns={"member_a": "state_a", "member_b": "state_b"})

    # Compute dyad entry/exit (overlap period)
    dyads["dyad_start"] = dyads[["yrent_a", "yrent_b"]].max(axis=1).clip(lower=YEAR_START)
    dyads["dyad_end"] = dyads[["yrexit_a", "yrexit_b"]].min(axis=1).clip(upper=YEAR_END)
    dyads = dyads[dyads["dyad_end"] >= dyads["dyad_start"]]  # valid time ranges only

    # Remove any remaining duplicates
    dyads = dyads.drop_duplicates(subset=["atopid", "state_a", "state_b"])

    print(f"  Total dyads: {len(dyads):,}")

    # Expand to dyad-year panel
    panel = _expand_dyad_years(dyads)
    print(f"  Expanded to {len(panel):,} dyad-years")

    # Merge institutionalization
    panel = panel.merge(inst_df, on="atopid", how="left")

    # Merge division of labor (pairwise portfolio complementarity)
    # Data already transformed: high values = high dissimilarity = high division of labor
    div_labor_raw = pd.read_csv(paths.div_labor_csv)

    # Normalize ordering to match panel (state_a < state_b)
    div_labor_raw["s_min"] = div_labor_raw[["state_a", "state_b"]].min(axis=1)
    div_labor_raw["s_max"] = div_labor_raw[["state_a", "state_b"]].max(axis=1)
    div_labor_raw["state_a"] = div_labor_raw["s_min"]
    div_labor_raw["state_b"] = div_labor_raw["s_max"]
    div_labor_raw = div_labor_raw.drop(columns=["s_min", "s_max"])

    panel = panel.merge(
        div_labor_raw[["state_a", "state_b", "year", "div_labor"]],
        on=["state_a", "state_b", "year"],
        how="left"
    )

    # Merge ideology for both states using helper
    rile_data = master_cy[["country_code_cow", "year", "rile"]].copy()
    panel = _merge_by_partner(panel, rile_data, "rile")
    panel["rile_dyad_mean"] = (panel["rile_a"] + panel["rile_b"]) / 2

    # Merge partner-level controls (GDP, CINC) for both states
    if "lngdp" in master_cy.columns:
        lngdp_data = master_cy[["country_code_cow", "year", "lngdp"]].copy()
        panel = _merge_by_partner(panel, lngdp_data, "lngdp")

    # Also merge GDP levels for Gannon-style bounded parity ratio
    if "gdp_level" in master_cy.columns:
        gdp_level_data = master_cy[["country_code_cow", "year", "gdp_level"]].copy()
        panel = _merge_by_partner(panel, gdp_level_data, "gdp_level")

    if "cinc" in master_cy.columns:
        cinc_data = master_cy[["country_code_cow", "year", "cinc"]].copy()
        panel = _merge_by_partner(panel, cinc_data, "cinc")

    # Merge contiguity
    contiguity = _load_contiguity(paths)
    panel = panel.merge(
        contiguity,
        on=["state_a", "state_b", "year"],
        how="left"
    )
    panel["contiguous"] = panel["contiguous"].fillna(0).astype(int)

    # =========================================================================
    # Add DCA ties (Defense Cooperation Agreements)
    # CRITICAL FIX: DCAD only covers 1980-2010!
    # Outside this window, DCA status is UNKNOWN (NA), not absent (0)
    # =========================================================================
    dcad = _load_dcad(paths)
    panel = panel.merge(
        dcad,
        on=["state_a", "state_b", "year"],
        how="left"
    )

    # Within DCAD window (1980-2010): fill missing with 0 (observed absence)
    # Outside DCAD window: keep as NA (unknown status)
    in_dcad_window = (panel["year"] >= DCAD_YEAR_START) & (panel["year"] <= DCAD_YEAR_END)
    panel.loc[in_dcad_window & panel["has_dca"].isna(), "has_dca"] = 0
    # Convert to int only for observed values, keep NA for unknown
    panel["has_dca"] = panel["has_dca"].astype("Int64")  # nullable integer

    print(f"  DCAD coverage:")
    print(f"    In-window (1980-2010): {in_dcad_window.sum():,} dyad-years")
    print(f"    Out-of-window:         {(~in_dcad_window).sum():,} dyad-years (DCA status = NA)")

    # =========================================================================
    # Add military expenditure from NMC 6.0 for both partners
    # =========================================================================
    nmc = _load_nmc(paths)

    # Merge for state_a
    panel = panel.merge(
        nmc[["ccode", "year", "milex"]].rename(columns={"ccode": "state_a", "milex": "milex_a"}),
        on=["state_a", "year"],
        how="left"
    )

    # Merge for state_b
    panel = panel.merge(
        nmc[["ccode", "year", "milex"]].rename(columns={"ccode": "state_b", "milex": "milex_b"}),
        on=["state_b", "year"],
        how="left"
    )

    # =========================================================================
    # Create Gannon-style BOUNDED ratio variables
    # =========================================================================
    # Following Gannon (2023): Use min/max ratio, bounded in [0,1]
    # This creates symmetric, interpretable ratios where 1 = equal partners
    #
    # CRITICAL: Gannon computes parity ratios on LEVELS, not logs.
    # gdp_ratio = min(gdp_a, gdp_b) / max(gdp_a, gdp_b)
    # This is the PRIMARY ratio for regressions.

    # GDP LEVEL ratio (bounded): PRIMARY ratio following Gannon
    if "gdp_level_a" in panel.columns and "gdp_level_b" in panel.columns:
        gdp_min = panel[["gdp_level_a", "gdp_level_b"]].min(axis=1)
        gdp_max = panel[["gdp_level_a", "gdp_level_b"]].max(axis=1)
        # Only defined for strictly positive levels
        valid_mask = (gdp_min > 0) & (gdp_max > 0)
        panel["gdp_ratio"] = np.nan
        panel.loc[valid_mask, "gdp_ratio"] = gdp_min[valid_mask] / gdp_max[valid_mask]
        # Assertion: ratios must be in (0, 1]
        gdp_valid = panel["gdp_ratio"].dropna()
        assert ((gdp_valid > 0) & (gdp_valid <= 1)).all(), "gdp_ratio must be in (0, 1]"
        print(f"    gdp_ratio (on levels): {panel['gdp_ratio'].notna().sum():,} valid values")

    # Log GDP ratio (bounded): LEGACY, kept for backward compatibility
    if "lngdp_a" in panel.columns and "lngdp_b" in panel.columns:
        lngdp_min = panel[["lngdp_a", "lngdp_b"]].min(axis=1)
        lngdp_max = panel[["lngdp_a", "lngdp_b"]].max(axis=1)
        panel["lngdp_ratio"] = lngdp_min / lngdp_max.replace(0, np.nan)

    # Military expenditure ratio (bounded): min(milex_a, milex_b) / max(milex_a, milex_b)
    if "milex_a" in panel.columns and "milex_b" in panel.columns:
        milex_min = panel[["milex_a", "milex_b"]].min(axis=1)
        milex_max = panel[["milex_a", "milex_b"]].max(axis=1)
        # Only defined for strictly positive milex
        valid_mask = (milex_min > 0) & (milex_max > 0)
        panel["milex_ratio"] = np.nan
        panel.loc[valid_mask, "milex_ratio"] = milex_min[valid_mask] / milex_max[valid_mask]
        # Assertion: ratios must be in (0, 1]
        milex_valid = panel["milex_ratio"].dropna()
        assert ((milex_valid > 0) & (milex_valid <= 1)).all(), "milex_ratio must be in (0, 1]"

    # Create decade variable for FE
    panel["decade"] = (panel["year"] // 10) * 10

    # =========================================================================
    # SAFE COLLAPSE TO DYAD-YEAR LEVEL (Critical Fix #2)
    # =========================================================================
    # Following Gannon (2023): When dyads share multiple alliances in a year,
    # code dyad-year institutionalization as the MAXIMUM among shared alliances.
    #
    # INVARIANCE CHECKS: Before collapsing, verify that dyad-year columns
    # (div_labor, contiguous, etc.) are identical across alliance rows.
    # If they vary, something is wrong with the merge logic.
    # =========================================================================

    # Audit: Count dyad-years with multiple alliances BEFORE collapsing
    rows_before = len(panel)
    dyad_year_counts = panel.groupby(["state_a", "state_b", "year"]).size()
    multi_alliance_dyad_years = (dyad_year_counts > 1).sum()
    total_dyad_years = len(dyad_year_counts)
    pct_multi = 100 * multi_alliance_dyad_years / total_dyad_years if total_dyad_years > 0 else 0

    print(f"\n  Collapsing to dyad-year level (Gannon 2023 approach)...")
    print(f"    Rows before collapse:           {rows_before:,} (dyad-alliance-years)")
    print(f"    Unique dyad-years:              {total_dyad_years:,}")
    print(f"    Dyad-years with >1 alliance:    {multi_alliance_dyad_years:,} ({pct_multi:.1f}%)")

    # -------------------------------------------------------------------------
    # INVARIANCE CHECK: These columns MUST be identical within each dyad-year
    # -------------------------------------------------------------------------
    # True dyad-year variables (not alliance-specific):
    dyad_year_invariant_cols = [
        "div_labor",       # DV: pairwise complementarity
        "contiguous",      # Dyad geography
        "gdp_ratio",       # PRIMARY: Bounded parity ratio on GDP LEVELS (Gannon style)
        "lngdp_ratio",     # LEGACY: Bounded ratio from log GDP
        "milex_ratio",     # Bounded ratio from country-year data
        "rile_a", "rile_b", "rile_dyad_mean",  # Country-year ideology
        "lngdp_a", "lngdp_b", "cinc_a", "cinc_b",  # Country-year controls
        "gdp_level_a", "gdp_level_b",  # GDP levels for ratio computation
        "milex_a", "milex_b",  # Country-year milex
        "has_dca",         # DCA status (dyad-year level)
        "decade",          # Year-derived
    ]

    # Only check columns that exist
    check_cols = [c for c in dyad_year_invariant_cols if c in panel.columns]

    print(f"    Checking invariance of {len(check_cols)} dyad-year columns...")
    _check_invariance(panel, check_cols, ["state_a", "state_b", "year"])
    print(f"    ✓ Invariance check passed")

    # -------------------------------------------------------------------------
    # COLLAPSE with safe aggregation
    # -------------------------------------------------------------------------
    # Sort by inst descending so "first" picks the most institutionalized alliance's atopid
    panel = panel.sort_values(["state_a", "state_b", "year", "inst"], ascending=[True, True, True, False])

    # Aggregation rules:
    # - inst: MAX (most institutionalized alliance wins)
    # - n_shared_alliances: COUNT (number of distinct alliances in this dyad-year)
    # - atopid_max_inst: FIRST after sort (atopid of most institutionalized alliance)
    # - Invariant cols: FIRST (verified identical above)

    agg_dict = {
        "inst": "max",                    # KEY: Most institutionalized alliance wins
        "atopid": ["first", "nunique"],   # First = atopid of max inst; nunique = count alliances
    }

    # Invariant columns: safe to take "first" after invariance check
    for col in check_cols:
        agg_dict[col] = "first"

    # Collapse to dyad-year
    collapsed = panel.groupby(["state_a", "state_b", "year"], as_index=False).agg(agg_dict)

    # Flatten multi-level columns from agg
    new_columns = []
    for col in collapsed.columns:
        if isinstance(col, tuple):
            # Handle multi-level columns from agg with list like ["first", "nunique"]
            name = "_".join(str(c) for c in col if c).rstrip("_")
            new_columns.append(name)
        else:
            new_columns.append(col)
    collapsed.columns = new_columns

    # Rename specific columns
    rename_map = {
        "inst_max": "inst",
        "atopid_first": "atopid_max_inst",
        "atopid_nunique": "n_shared_alliances",
    }
    # Also handle invariant columns that got "_first" suffix
    for col in check_cols:
        suffixed = f"{col}_first"
        if suffixed in collapsed.columns:
            rename_map[suffixed] = col

    collapsed = collapsed.rename(columns=rename_map)

    panel = collapsed

    rows_after = len(panel)
    print(f"    Rows after collapse:            {rows_after:,} (dyad-years)")

    # -------------------------------------------------------------------------
    # Post-collapse assertions
    # -------------------------------------------------------------------------
    # 1. No duplicates on (state_a, state_b, year) AND (dyad_id, year)
    panel["dyad_id"] = panel["state_a"].astype(str) + "_" + panel["state_b"].astype(str)
    assert_unique_key(panel, ["state_a", "state_b", "year"], "dyad-year (state_a, state_b, year)")
    assert_unique_key(panel, ["dyad_id", "year"], "dyad-year (dyad_id, year)")

    # 2. n_shared_alliances is integer >= 0
    assert (panel["n_shared_alliances"] >= 0).all(), "n_shared_alliances must be >= 0"
    assert panel["n_shared_alliances"].notna().all(), "n_shared_alliances should not have NA"

    print(f"    ✓ Post-collapse assertions passed")

    # Recreate binary dummies from collapsed inst
    panel["hierarchical"] = (panel["inst"] == 3).astype(int)
    panel["voice_driven"] = (panel["inst"] == 2).astype(int)

    # =========================================================================
    # Create alignment indicators (DCAD-aware)
    # =========================================================================
    # any_atop_link: 1 if dyad shares an ATOP alliance (always 1 in ATOP-based panel)
    panel["any_atop_link"] = 1

    # any_dca_link: 1 if dyad has a DCA, NA if outside DCAD window
    # has_dca is already correctly coded (NA outside window, 0/1 inside)
    panel["any_dca_link"] = panel["has_dca"]

    # For in-window observations: compute alignment and dca_only
    # any_alignment: 1 if dyad has either ATOP or DCA link
    # In this ATOP-based panel, any_atop_link = 1 always, so any_alignment = 1 always
    # But we'll compute it correctly for documentation
    panel["any_alignment"] = 1  # Always 1 in ATOP panel

    # dca_only: 1 if DCA but no ATOP (impossible in ATOP panel, but for completeness)
    panel["dca_only"] = 0

    # Distribution of dyad-years by institutionalization
    inst_counts = panel["inst"].value_counts().sort_index()
    print(f"    Distribution by inst_max:")
    for inst_val, count in inst_counts.items():
        label = {1: "Uninst", 2: "Voice", 3: "Hier"}.get(int(inst_val), f"Type {inst_val}")
        print(f"      {label} (inst={int(inst_val)}): {count:,} ({100*count/rows_after:.1f}%)")

    # Coverage summary
    coverage_report(panel, "dyad_id", "year", "dyad-year")

    div_labor_n = panel["div_labor"].notna().sum()
    contiguous_n = panel["contiguous"].sum()
    gdp_ratio_n = panel["gdp_ratio"].notna().sum() if "gdp_ratio" in panel.columns else 0
    lngdp_ratio_n = panel["lngdp_ratio"].notna().sum() if "lngdp_ratio" in panel.columns else 0
    milex_ratio_n = panel["milex_ratio"].notna().sum() if "milex_ratio" in panel.columns else 0
    dca_observed = panel["has_dca"].notna().sum() if "has_dca" in panel.columns else 0
    dca_positive = (panel["has_dca"] == 1).sum() if "has_dca" in panel.columns else 0
    print(f"  Coverage:")
    print(f"    div_labor:    {div_labor_n:,} ({div_labor_n/len(panel):.1%})")
    print(f"    contiguous:   {contiguous_n:,} dyad-years with land border ({contiguous_n/len(panel):.1%})")
    print(f"    gdp_ratio:    {gdp_ratio_n:,} ({gdp_ratio_n/len(panel):.1%}) [PRIMARY: on levels]")
    print(f"    lngdp_ratio:  {lngdp_ratio_n:,} ({lngdp_ratio_n/len(panel):.1%}) [legacy: on logs]")
    print(f"    milex_ratio:  {milex_ratio_n:,} ({milex_ratio_n/len(panel):.1%})")
    print(f"    has_dca:      {dca_observed:,} observed ({dca_observed/len(panel):.1%}), {dca_positive:,} positive")

    # PROVENANCE CHECK: Ensure no forbidden columns from external alliance-type sources
    _assert_no_forbidden_columns(panel, "master_dyad_year")
    _assert_inst_from_atop_only(panel, "master_dyad_year")
    print("  ✓ Provenance check passed: No RR-based alliance-type columns")

    # Save full panel
    panel.to_csv(paths.dyad_year_csv, index=False)
    print(f"  Saved: {paths.dyad_year_csv}")

    # =========================================================================
    # CREATE ALIGNED SAMPLE (1980-2010) FOR H2/H3 GANNON+DCA ANALYSES
    # =========================================================================
    print(f"\n  Creating aligned sample for 1980-2010 (DCAD observed window)...")

    # Filter to DCAD window
    aligned = panel[
        (panel["year"] >= DCAD_YEAR_START) &
        (panel["year"] <= DCAD_YEAR_END)
    ].copy()

    # In this ATOP-based panel, all rows have any_atop_link = 1
    # So any_alignment = 1 for all rows
    # Verify this
    assert (aligned["any_alignment"] == 1).all(), "All rows in ATOP panel should have any_alignment=1"

    # Add dca_only indicator (correct in-window)
    aligned["dca_only"] = ((aligned["any_dca_link"] == 1) & (aligned["any_atop_link"] == 0)).astype(int)
    # Note: In ATOP-based panel, dca_only will always be 0 since any_atop_link = 1

    print(f"    Aligned sample: {len(aligned):,} dyad-years")
    print(f"    Year range: {aligned['year'].min()}-{aligned['year'].max()}")
    print(f"    Unique dyads: {aligned['dyad_id'].nunique()}")

    # Verify no any_alignment = 0 rows
    n_unaligned = (aligned["any_alignment"] == 0).sum()
    assert n_unaligned == 0, f"Found {n_unaligned} rows with any_alignment=0 in aligned sample!"
    print(f"    ✓ Verified: 0 rows with any_alignment=0")

    # PROVENANCE CHECK
    _assert_no_forbidden_columns(aligned, "master_dyad_year_gannon_1980_2010")
    _assert_inst_from_atop_only(aligned, "master_dyad_year_gannon_1980_2010")

    # Save aligned sample
    aligned.to_csv(paths.dyad_year_gannon_csv, index=False)
    print(f"  Saved: {paths.dyad_year_gannon_csv}")

    return panel


def build_dyad_year_gannon_union(paths: Paths) -> pd.DataFrame:
    """
    Build Gannon-style UNION dyad-year dataset (1980-2010).

    This follows Gannon's approach EXACTLY:
    - Sample: Dyad-years where partners share ATOP offense/defense pact OR DCAD agreement
    - Institutionalization: For ATOP alliances, use Leeds & Anac coding; take MAX if multiple
    - DCA-only dyad-years: vertical_integration = 0 (no ATOP treaty provisions to code)
    - Window: 1980-2010 only (DCAD coverage period)

    This is the PRIMARY dataset for H2/H3 analyses replicating Gannon's approach.

    Output: master_dyad_year_gannon_union_1980_2010.csv
    """
    print("\n  Building Gannon UNION dyad-year dataset (1980-2010)...")
    print("  " + "=" * 60)
    print("  GANNON REPLICATION: UNION of ATOP and DCAD aligned dyad-years")
    print("  " + "=" * 60)

    # =========================================================================
    # STEP 1: Load raw data sources
    # =========================================================================
    atop = _load_atop(paths)
    dcad = _load_dcad(paths)
    master_cy = pd.read_csv(paths.country_year_csv)

    # Clean ATOP
    atop = atop.copy()
    for col in ["atopid", "member", "yrent", "yrexit"]:
        atop[col] = pd.to_numeric(atop[col], errors="coerce")
    atop = atop.dropna(subset=["atopid", "member", "yrent"])
    atop["atopid"] = atop["atopid"].astype(int)
    atop["member"] = atop["member"].astype(int)
    atop["yrent"] = atop["yrent"].astype(int)
    atop["yrexit"] = atop["yrexit"].fillna(YEAR_END).replace(0, YEAR_END).astype(int)

    # Filter to meaningful security cooperation: offense or defense pacts only
    # Excludes: neutrality pacts, non-aggression pacts, consultation-only agreements
    n_before = atop["atopid"].nunique()
    atop["offense"] = pd.to_numeric(atop["offense"], errors="coerce").fillna(0)
    atop["defense"] = pd.to_numeric(atop["defense"], errors="coerce").fillna(0)
    atop = atop[(atop["defense"] == 1) | (atop["offense"] == 1)]
    n_after = atop["atopid"].nunique()
    print(f"  Filtered to offense/defense pacts: {n_after} of {n_before} alliances")

    # Build institutionalization for ATOP alliances
    inst_df = _build_institutionalization(atop)
    print(f"  ATOP alliances with inst: {len(inst_df)}")

    # =========================================================================
    # STEP 2: Build ATOP dyad-years (1980-2010)
    # =========================================================================
    print("\n  Building ATOP dyad-years...")

    # Consolidate member entry/exit
    members = (
        atop.groupby(["atopid", "member"])
        .agg({"yrent": "min", "yrexit": "max"})
        .reset_index()
    )

    # Create dyads from all pairs within each alliance
    dyads_atop = members.merge(members, on="atopid", suffixes=("_a", "_b"))
    dyads_atop = dyads_atop[dyads_atop["member_a"] < dyads_atop["member_b"]]
    dyads_atop = dyads_atop.rename(columns={"member_a": "state_a", "member_b": "state_b"})

    # Compute overlap period
    dyads_atop["dyad_start"] = dyads_atop[["yrent_a", "yrent_b"]].max(axis=1)
    dyads_atop["dyad_end"] = dyads_atop[["yrexit_a", "yrexit_b"]].min(axis=1)

    # Restrict to 1980-2010
    dyads_atop["dyad_start"] = dyads_atop["dyad_start"].clip(lower=DCAD_YEAR_START)
    dyads_atop["dyad_end"] = dyads_atop["dyad_end"].clip(upper=DCAD_YEAR_END)
    dyads_atop = dyads_atop[dyads_atop["dyad_end"] >= dyads_atop["dyad_start"]]

    # Expand to dyad-alliance-years
    atop_rows = []
    for _, row in dyads_atop.iterrows():
        for y in range(int(row["dyad_start"]), int(row["dyad_end"]) + 1):
            atop_rows.append({
                "state_a": row["state_a"],
                "state_b": row["state_b"],
                "year": y,
                "atopid": row["atopid"],
            })
    atop_panel = pd.DataFrame(atop_rows)

    # Merge institutionalization
    atop_panel = atop_panel.merge(inst_df, on="atopid", how="left")

    print(f"    ATOP dyad-alliance-years (1980-2010): {len(atop_panel):,}")

    # Collapse to dyad-year level: take max(inst), count alliances
    atop_collapsed = atop_panel.groupby(["state_a", "state_b", "year"]).agg(
        inst_atop_max=("inst", "max"),
        n_shared_alliances=("atopid", "nunique"),
        atopid_list=("atopid", lambda x: sorted(x.unique())),
    ).reset_index()

    # Get atopid with max inst (tie-breaker: smallest atopid)
    def get_atopid_max_inst(group):
        max_inst = group["inst"].max()
        candidates = group[group["inst"] == max_inst]["atopid"]
        return candidates.min()

    atopid_max = atop_panel.groupby(["state_a", "state_b", "year"]).apply(
        get_atopid_max_inst, include_groups=False
    ).reset_index(name="atopid_max_inst")

    atop_collapsed = atop_collapsed.merge(
        atopid_max, on=["state_a", "state_b", "year"], how="left"
    )
    atop_collapsed["any_atop_link"] = 1

    print(f"    ATOP dyad-years (collapsed): {len(atop_collapsed):,}")

    # =========================================================================
    # STEP 3: Build DCAD dyad-years (1980-2010)
    # =========================================================================
    print("\n  Building DCAD dyad-years...")

    # DCAD is already at dyad-year level with has_dca indicator
    dcad_panel = dcad[
        (dcad["year"] >= DCAD_YEAR_START) &
        (dcad["year"] <= DCAD_YEAR_END) &
        (dcad["has_dca"] == 1)
    ].copy()
    dcad_panel["any_dca_link"] = 1

    print(f"    DCAD dyad-years with DCA=1: {len(dcad_panel):,}")

    # =========================================================================
    # STEP 4: UNION the two sets
    # =========================================================================
    print("\n  Creating UNION of ATOP and DCAD aligned dyad-years...")

    # Create unique dyad-year keys from both sources
    atop_keys = atop_collapsed[["state_a", "state_b", "year"]].drop_duplicates()
    dcad_keys = dcad_panel[["state_a", "state_b", "year"]].drop_duplicates()

    # Union
    union_keys = pd.concat([atop_keys, dcad_keys]).drop_duplicates()
    union_keys["dyad_id"] = union_keys["state_a"].astype(str) + "_" + union_keys["state_b"].astype(str)

    print(f"    ATOP-only dyad-years:  {len(atop_keys):,}")
    print(f"    DCAD-only dyad-years:  {len(dcad_keys):,}")
    print(f"    UNION dyad-years:      {len(union_keys):,}")

    # =========================================================================
    # STEP 5: Merge ATOP and DCAD info onto union scaffold
    # =========================================================================
    panel = union_keys.copy()

    # Merge ATOP info
    panel = panel.merge(
        atop_collapsed[["state_a", "state_b", "year", "inst_atop_max", "n_shared_alliances",
                        "atopid_max_inst", "any_atop_link"]],
        on=["state_a", "state_b", "year"],
        how="left"
    )

    # Merge DCAD info
    panel = panel.merge(
        dcad_panel[["state_a", "state_b", "year", "any_dca_link"]],
        on=["state_a", "state_b", "year"],
        how="left"
    )

    # Fill NAs
    panel["any_atop_link"] = panel["any_atop_link"].fillna(0).astype(int)
    panel["any_dca_link"] = panel["any_dca_link"].fillna(0).astype(int)
    panel["n_shared_alliances"] = panel["n_shared_alliances"].fillna(0).astype(int)
    panel["inst_atop_max"] = panel["inst_atop_max"].fillna(0).astype(int)

    # =========================================================================
    # STEP 6: Create alignment indicators and vertical integration
    # =========================================================================
    panel["any_alignment"] = ((panel["any_atop_link"] == 1) | (panel["any_dca_link"] == 1)).astype(int)
    panel["dca_only"] = ((panel["any_dca_link"] == 1) & (panel["any_atop_link"] == 0)).astype(int)

    # vertical_integration = inst_atop_max
    # For DCA-only dyad-years, this is 0 (no ATOP treaty provisions to code)
    panel["vertical_integration"] = panel["inst_atop_max"]

    # Create dummies for each category (reference: DCA-only = 0)
    # IMPORTANT: Do NOT pool uninstitutionalized ATOP with DCA-only!
    # These are conceptually different: formal alliance vs. informal agreement
    panel["hierarchical"] = (panel["vertical_integration"] == 3).astype(int)
    panel["voice_driven"] = (panel["vertical_integration"] == 2).astype(int)
    panel["uninstitutionalized"] = (panel["vertical_integration"] == 1).astype(int)

    # Create decade for FE
    panel["decade"] = (panel["year"] // 10) * 10

    print(f"\n  Alignment breakdown:")
    print(f"    ATOP-only:  {((panel['any_atop_link']==1) & (panel['any_dca_link']==0)).sum():,}")
    print(f"    DCAD-only:  {panel['dca_only'].sum():,}")
    print(f"    Both:       {((panel['any_atop_link']==1) & (panel['any_dca_link']==1)).sum():,}")

    # =========================================================================
    # STEP 7: Merge division of labor (DV)
    # =========================================================================
    print("\n  Merging division of labor...")
    div_labor_raw = pd.read_csv(paths.div_labor_csv)

    # Normalize ordering
    div_labor_raw["s_min"] = div_labor_raw[["state_a", "state_b"]].min(axis=1)
    div_labor_raw["s_max"] = div_labor_raw[["state_a", "state_b"]].max(axis=1)
    div_labor_raw["state_a"] = div_labor_raw["s_min"]
    div_labor_raw["state_b"] = div_labor_raw["s_max"]
    div_labor_raw = div_labor_raw.drop(columns=["s_min", "s_max"])

    panel = panel.merge(
        div_labor_raw[["state_a", "state_b", "year", "div_labor"]],
        on=["state_a", "state_b", "year"],
        how="left"
    )

    # =========================================================================
    # STEP 8: Merge contiguity
    # =========================================================================
    contiguity = _load_contiguity(paths)
    panel = panel.merge(contiguity, on=["state_a", "state_b", "year"], how="left")
    panel["contiguous"] = panel["contiguous"].fillna(0).astype(int)

    # =========================================================================
    # STEP 9: Merge partner-level controls from country-year
    # =========================================================================
    # GDP (both log and level forms)
    if "lngdp" in master_cy.columns:
        lngdp_data = master_cy[["country_code_cow", "year", "lngdp"]].copy()
        panel = _merge_by_partner(panel, lngdp_data, "lngdp")

    # GDP levels for Gannon-style bounded parity ratio
    if "gdp_level" in master_cy.columns:
        gdp_level_data = master_cy[["country_code_cow", "year", "gdp_level"]].copy()
        panel = _merge_by_partner(panel, gdp_level_data, "gdp_level")

    # CINC
    if "cinc" in master_cy.columns:
        cinc_data = master_cy[["country_code_cow", "year", "cinc"]].copy()
        panel = _merge_by_partner(panel, cinc_data, "cinc")

    # Milex from NMC
    nmc = _load_nmc(paths)
    panel = panel.merge(
        nmc[["ccode", "year", "milex"]].rename(columns={"ccode": "state_a", "milex": "milex_a"}),
        on=["state_a", "year"], how="left"
    )
    panel = panel.merge(
        nmc[["ccode", "year", "milex"]].rename(columns={"ccode": "state_b", "milex": "milex_b"}),
        on=["state_b", "year"], how="left"
    )

    # =========================================================================
    # STEP 10: Create Gannon-style bounded ratios
    # =========================================================================
    # CRITICAL: Gannon computes parity ratios on LEVELS, not logs.
    # gdp_ratio = min(gdp_a, gdp_b) / max(gdp_a, gdp_b)
    # This is the PRIMARY ratio for regressions.

    # GDP LEVEL ratio (bounded): PRIMARY ratio following Gannon
    if "gdp_level_a" in panel.columns and "gdp_level_b" in panel.columns:
        gdp_min = panel[["gdp_level_a", "gdp_level_b"]].min(axis=1)
        gdp_max = panel[["gdp_level_a", "gdp_level_b"]].max(axis=1)
        valid_mask = (gdp_min > 0) & (gdp_max > 0)
        panel["gdp_ratio"] = np.nan
        panel.loc[valid_mask, "gdp_ratio"] = gdp_min[valid_mask] / gdp_max[valid_mask]
        # Assertion: ratios must be in (0, 1]
        gdp_valid = panel["gdp_ratio"].dropna()
        if len(gdp_valid) > 0:
            assert ((gdp_valid > 0) & (gdp_valid <= 1)).all(), "gdp_ratio must be in (0, 1]"

    # lngdp_ratio = min/max (bounded in [0,1]) - LEGACY
    if "lngdp_a" in panel.columns and "lngdp_b" in panel.columns:
        lngdp_min = panel[["lngdp_a", "lngdp_b"]].min(axis=1)
        lngdp_max = panel[["lngdp_a", "lngdp_b"]].max(axis=1)
        panel["lngdp_ratio"] = lngdp_min / lngdp_max.replace(0, np.nan)

    # milex_ratio = min/max (bounded in [0,1])
    if "milex_a" in panel.columns and "milex_b" in panel.columns:
        milex_min = panel[["milex_a", "milex_b"]].min(axis=1)
        milex_max = panel[["milex_a", "milex_b"]].max(axis=1)
        valid_mask = (milex_min > 0) & (milex_max > 0)
        panel["milex_ratio"] = np.nan
        panel.loc[valid_mask, "milex_ratio"] = milex_min[valid_mask] / milex_max[valid_mask]
        # Assertion: ratios must be in (0, 1]
        milex_valid = panel["milex_ratio"].dropna()
        if len(milex_valid) > 0:
            assert ((milex_valid > 0) & (milex_valid <= 1)).all(), "milex_ratio must be in (0, 1]"

    # =========================================================================
    # STEP 11: Merge ideology for H3
    # =========================================================================
    rile_data = master_cy[["country_code_cow", "year", "rile"]].copy()
    panel = _merge_by_partner(panel, rile_data, "rile")
    panel["rile_dyad_mean"] = (panel["rile_a"] + panel["rile_b"]) / 2

    # Compute ideological distance
    panel["ideo_dist"] = (panel["rile_a"] - panel["rile_b"]).abs()

    # Create lagged ideology distance (within dyad)
    panel = panel.sort_values(["dyad_id", "year"])
    panel["ideo_dist_lag5"] = panel.groupby("dyad_id")["ideo_dist"].shift(5)

    # =========================================================================
    # STEP 12: Final assertions and audit
    # =========================================================================
    print("\n  Final dataset verification...")

    # Verify no duplicates
    assert_unique_key(panel, ["state_a", "state_b", "year"], "dyad-year")
    assert_unique_key(panel, ["dyad_id", "year"], "dyad-year by dyad_id")

    # Verify all rows are aligned
    assert (panel["any_alignment"] == 1).all(), "All rows must have any_alignment=1"
    print(f"    ✓ No duplicates on (dyad_id, year)")
    print(f"    ✓ All rows have any_alignment=1")

    # =========================================================================
    # AUDIT SUMMARY
    # =========================================================================
    print("\n  " + "=" * 60)
    print("  GANNON UNION SAMPLE AUDIT")
    print("  " + "=" * 60)

    print(f"\n  Sample composition:")
    print(f"    Total dyad-years:     {len(panel):,}")
    print(f"    Unique dyads:         {panel['dyad_id'].nunique():,}")
    print(f"    Year range:           {panel['year'].min()}-{panel['year'].max()}")

    print(f"\n  Alignment breakdown:")
    atop_only = ((panel["any_atop_link"]==1) & (panel["any_dca_link"]==0)).sum()
    dca_only = panel["dca_only"].sum()
    both = ((panel["any_atop_link"]==1) & (panel["any_dca_link"]==1)).sum()
    print(f"    ATOP-only:  {atop_only:,} ({100*atop_only/len(panel):.1f}%)")
    print(f"    DCA-only:   {dca_only:,} ({100*dca_only/len(panel):.1f}%)")
    print(f"    Both:       {both:,} ({100*both/len(panel):.1f}%)")

    print(f"\n  Vertical integration distribution:")
    vi_dist = panel["vertical_integration"].value_counts().sort_index()
    for vi_val, count in vi_dist.items():
        label = {0: "None/DCA-only", 1: "Uninst", 2: "Voice", 3: "Hier"}.get(int(vi_val), f"Type {vi_val}")
        print(f"    {label} (vi={int(vi_val)}): {count:,} ({100*count/len(panel):.1f}%)")

    print(f"\n  Variable coverage:")
    div_labor_n = panel["div_labor"].notna().sum()
    gdp_ratio_n = panel["gdp_ratio"].notna().sum() if "gdp_ratio" in panel.columns else 0
    lngdp_ratio_n = panel["lngdp_ratio"].notna().sum() if "lngdp_ratio" in panel.columns else 0
    milex_ratio_n = panel["milex_ratio"].notna().sum() if "milex_ratio" in panel.columns else 0
    ideo_dist_n = panel["ideo_dist_lag5"].notna().sum()
    print(f"    div_labor:      {div_labor_n:,} ({100*div_labor_n/len(panel):.1f}%)")
    print(f"    gdp_ratio:      {gdp_ratio_n:,} ({100*gdp_ratio_n/len(panel):.1f}%) [PRIMARY: on levels]")
    print(f"    lngdp_ratio:    {lngdp_ratio_n:,} ({100*lngdp_ratio_n/len(panel):.1f}%) [legacy: on logs]")
    print(f"    milex_ratio:    {milex_ratio_n:,} ({100*milex_ratio_n/len(panel):.1f}%)")
    print(f"    ideo_dist_lag5: {ideo_dist_n:,} ({100*ideo_dist_n/len(panel):.1f}%)")

    # PROVENANCE CHECK: Ensure no forbidden columns from external alliance-type sources
    _assert_no_forbidden_columns(panel, "master_dyad_year_gannon_union_1980_2010")
    _assert_inst_from_atop_only(panel, "master_dyad_year_gannon_union_1980_2010")
    print("  ✓ Provenance check passed: No RR-based alliance-type columns")

    # Save
    panel.to_csv(paths.dyad_year_gannon_union_csv, index=False)
    print(f"\n  Saved: {paths.dyad_year_gannon_union_csv}")

    return panel


def build_dyad_year_h3(paths: Paths) -> pd.DataFrame:
    """
    Build enriched dyad-year dataset for H3 (ideological symmetry -> division of labor).

    This function creates a SEPARATE dataset from master_dyad_year.csv, enriched with
    partner ideology variables for testing whether ideologically similar dyads exhibit
    greater division of labor.

    NOTE: This does NOT modify master_dyad_year.csv used by H2.

    Key variables created:
    - rile_a, rile_b: Partner ideology scores
    - ideo_dist: Absolute difference |rile_a - rile_b|
    - ideo_dist_lag5: Lagged ideological distance (within dyad)
    - bucket_a, bucket_b: Categorical buckets (RoC/LoC/Mod/NR)
    - same_bucket_10: 1 if partners in same non-NR bucket
    - same_bucket_10_lag5: Lagged symmetry indicator
    """
    print("\n  Building H3 dyad-year dataset (ideological symmetry)...")

    # Load base dyad-year data (do NOT modify this file)
    dy = pd.read_csv(paths.dyad_year_csv)
    print(f"    Base dyad-year dataset: {len(dy):,} rows")

    # Check if ideology columns already exist from H2 build
    if "rile_a" in dy.columns and "rile_b" in dy.columns:
        print("    Using existing rile_a/rile_b from base dataset")
    else:
        # Merge ideology from country-year if not present
        cy = pd.read_csv(paths.country_year_csv)
        print(f"    Country-year dataset: {len(cy):,} rows")

        ideo = cy[["country_code_cow", "year", "rile"]].copy()
        ideo = ideo.rename(columns={"country_code_cow": "state"})

        dy = dy.merge(
            ideo.rename(columns={"state": "state_a", "rile": "rile_a"}),
            on=["state_a", "year"],
            how="left",
        )
        dy = dy.merge(
            ideo.rename(columns={"state": "state_b", "rile": "rile_b"}),
            on=["state_b", "year"],
            how="left",
        )

    # Report ideology coverage
    n_both_ideo = dy[["rile_a", "rile_b"]].notna().all(axis=1).sum()
    print(f"    Dyad-years with both partners' ideology: {n_both_ideo:,} ({n_both_ideo/len(dy):.1%})")

    # ==========================================================================
    # A) Continuous similarity: ideo_dist = |rile_a - rile_b|
    # ==========================================================================
    dy["ideo_dist"] = (dy["rile_a"] - dy["rile_b"]).abs()

    # Create dyad_id if not present
    if "dyad_id" not in dy.columns:
        dy["dyad_id"] = dy["state_a"].astype(str) + "_" + dy["state_b"].astype(str)

    # Sort for correct lag computation
    dy = dy.sort_values(["dyad_id", "year"])

    # Create lagged ideological distance
    dy["ideo_dist_lag5"] = dy.groupby("dyad_id")["ideo_dist"].shift(5)

    # ==========================================================================
    # B) Categorical symmetry (±10 thresholds)
    # ==========================================================================
    def _categorize_ideology(rile: float, right_thresh: float = 10.0, left_thresh: float = -10.0) -> str:
        """Categorize RILE into buckets: RoC, LoC, Mod, or NR."""
        if pd.isna(rile):
            return "NR"
        if rile >= right_thresh:
            return "RoC"
        if rile <= left_thresh:
            return "LoC"
        return "Mod"

    dy["bucket_a"] = dy["rile_a"].apply(_categorize_ideology)
    dy["bucket_b"] = dy["rile_b"].apply(_categorize_ideology)

    # same_bucket_10: 1 if both partners in same bucket AND neither is NR
    dy["same_bucket_10"] = (
        (dy["bucket_a"] == dy["bucket_b"]) &
        (dy["bucket_a"] != "NR")
    ).astype(int)

    # any_NR: 1 if either partner has missing ideology
    dy["any_NR"] = (
        (dy["bucket_a"] == "NR") | (dy["bucket_b"] == "NR")
    ).astype(int)

    # Lagged categorical symmetry
    dy["same_bucket_10_lag5"] = dy.groupby("dyad_id")["same_bucket_10"].shift(5)

    # ==========================================================================
    # Optional: specific bucket pair dummies for exploratory analysis
    # ==========================================================================
    dy["both_RoC"] = ((dy["bucket_a"] == "RoC") & (dy["bucket_b"] == "RoC")).astype(int)
    dy["both_LoC"] = ((dy["bucket_a"] == "LoC") & (dy["bucket_b"] == "LoC")).astype(int)
    dy["both_Mod"] = ((dy["bucket_a"] == "Mod") & (dy["bucket_b"] == "Mod")).astype(int)

    # Lagged versions
    for col in ["both_RoC", "both_LoC", "both_Mod"]:
        dy[f"{col}_lag5"] = dy.groupby("dyad_id")[col].shift(5)

    # ==========================================================================
    # Verify unique key (dyad-year level, consistent with H2)
    # ==========================================================================
    key_cols = ["state_a", "state_b", "year"]
    if dy.duplicated(key_cols).sum() > 0:
        print(f"    WARNING: {dy.duplicated(key_cols).sum()} duplicate rows on key!")
    else:
        print(f"    ✓ Unique key verified: {key_cols}")

    # ==========================================================================
    # Coverage summary
    # ==========================================================================
    print(f"  H3 variable coverage:")
    ideology_vars = ["rile_a", "rile_b", "ideo_dist", "ideo_dist_lag5", "same_bucket_10", "same_bucket_10_lag5"]
    gannon_vars = ["lngdp_ratio", "milex_ratio", "has_dca"]
    for col in ideology_vars + gannon_vars:
        if col in dy.columns:
            n = dy[col].notna().sum()
            print(f"    {col}: {n:,} ({n/len(dy):.1%})")

    # Bucket distribution
    bucket_dist = dy.groupby(["bucket_a", "bucket_b"]).size().reset_index(name="count")
    print(f"  Bucket pair distribution (top 5):")
    for _, row in bucket_dist.nlargest(5, "count").iterrows():
        print(f"    {row['bucket_a']}-{row['bucket_b']}: {row['count']:,}")

    # ==========================================================================
    # Provenance check and Save
    # ==========================================================================
    _assert_no_forbidden_columns(dy, "master_dyad_year_h3")
    _assert_inst_from_atop_only(dy, "master_dyad_year_h3")

    dy.to_csv(paths.dyad_year_h3_csv, index=False)
    print(f"  Saved H3 dataset: {paths.dyad_year_h3_csv}")

    return dy


def build_all() -> None:
    """Build all datasets."""
    paths = Paths()

    # Validate input files
    missing = paths.validate()
    if missing:
        print(f"Error: Missing input files: {missing}")
        return

    # Ensure output directories exist
    paths.country_year_csv.parent.mkdir(parents=True, exist_ok=True)
    paths.dyad_year_csv.parent.mkdir(parents=True, exist_ok=True)
    paths.dyad_year_h3_csv.parent.mkdir(parents=True, exist_ok=True)
    paths.audit_dir.mkdir(parents=True, exist_ok=True)

    build_country_year(paths)
    build_dyad_year(paths)

    # Build Gannon UNION dataset (1980-2010, ATOP OR DCAD aligned)
    # This is the PRIMARY dataset for H2/H3 Gannon replication
    build_dyad_year_gannon_union(paths)

    # Build H3 dataset (requires country-year for ideology, dyad-year as base)
    build_dyad_year_h3(paths)

    # Add alliance membership to country-year (requires dyad assets)
    _add_alliance_membership(paths)

    # Generate comprehensive audit
    generate_audit(paths)

    print("\n  All datasets built successfully.")


def generate_audit(paths: Paths) -> None:
    """
    Generate comprehensive audit outputs for sample construction.

    Creates detailed reports on:
    1. Sample flow and attrition at each stage
    2. Variable coverage and missingness patterns
    3. Gannon-style alignment indicators distribution
    4. DCAD coverage window diagnostics (CRITICAL)
    5. Control variable distributions by sample
    6. Dyad-year collapse diagnostics
    """
    print("\n  Generating audit outputs...")
    paths.audit_dir.mkdir(parents=True, exist_ok=True)

    # Load datasets
    cy = pd.read_csv(paths.country_year_csv)
    dy = pd.read_csv(paths.dyad_year_csv)
    dy_h3 = pd.read_csv(paths.dyad_year_h3_csv)

    # Also load the 1980-2010 aligned sample if it exists
    if paths.dyad_year_gannon_csv.exists():
        dy_aligned = pd.read_csv(paths.dyad_year_gannon_csv)
    else:
        dy_aligned = None

    audit_rows = []

    # =========================================================================
    # H2 AUDIT: Dyad-Year Sample
    # =========================================================================
    print("    H2 sample audit...")

    # Stage 1: Full dyad-year panel
    audit_rows.append({
        "Hypothesis": "H2",
        "Stage": "1. Full dyad-year panel",
        "N_rows": len(dy),
        "N_units": dy["dyad_id"].nunique(),
        "Pct_retained": 100.0
    })

    # Stage 2: Require div_labor (DV)
    n_with_dv = dy["div_labor"].notna().sum()
    audit_rows.append({
        "Hypothesis": "H2",
        "Stage": "2. Require div_labor (DV)",
        "N_rows": n_with_dv,
        "N_units": dy[dy["div_labor"].notna()]["dyad_id"].nunique(),
        "Pct_retained": 100 * n_with_dv / len(dy)
    })

    # Stage 3: Require inst variables
    has_inst = dy["hierarchical"].notna() & dy["voice_driven"].notna()
    audit_rows.append({
        "Hypothesis": "H2",
        "Stage": "3. Require inst variables",
        "N_rows": has_inst.sum(),
        "N_units": dy[has_inst]["dyad_id"].nunique(),
        "Pct_retained": 100 * has_inst.sum() / len(dy)
    })

    # Stage 4: Require Gannon controls (lngdp_ratio, milex_ratio, contiguous)
    gannon_controls = ["lngdp_ratio", "milex_ratio", "contiguous"]
    has_gannon = dy[gannon_controls].notna().all(axis=1)
    audit_rows.append({
        "Hypothesis": "H2",
        "Stage": "4. Require Gannon controls",
        "N_rows": has_gannon.sum(),
        "N_units": dy[has_gannon]["dyad_id"].nunique(),
        "Pct_retained": 100 * has_gannon.sum() / len(dy)
    })

    # Stage 5: Full estimation sample
    full_sample = dy["div_labor"].notna() & has_inst & has_gannon
    audit_rows.append({
        "Hypothesis": "H2",
        "Stage": "5. Full estimation sample",
        "N_rows": full_sample.sum(),
        "N_units": dy[full_sample]["dyad_id"].nunique(),
        "Pct_retained": 100 * full_sample.sum() / len(dy)
    })

    # =========================================================================
    # H3 AUDIT: Ideological Symmetry
    # =========================================================================
    print("    H3 sample audit...")

    audit_rows.append({
        "Hypothesis": "H3",
        "Stage": "1. Full H3 panel",
        "N_rows": len(dy_h3),
        "N_units": dy_h3["dyad_id"].nunique(),
        "Pct_retained": 100.0
    })

    # Require ideo_dist_lag5
    has_ideo = dy_h3["ideo_dist_lag5"].notna()
    audit_rows.append({
        "Hypothesis": "H3",
        "Stage": "2. Require ideo_dist_lag5",
        "N_rows": has_ideo.sum(),
        "N_units": dy_h3[has_ideo]["dyad_id"].nunique(),
        "Pct_retained": 100 * has_ideo.sum() / len(dy_h3)
    })

    # Require Gannon controls
    h3_controls = ["lngdp_ratio", "milex_ratio", "contiguous"]
    available_h3_controls = [c for c in h3_controls if c in dy_h3.columns]
    if available_h3_controls:
        has_h3_controls = dy_h3[available_h3_controls].notna().all(axis=1)
        audit_rows.append({
            "Hypothesis": "H3",
            "Stage": "3. Require Gannon controls",
            "N_rows": has_h3_controls.sum(),
            "N_units": dy_h3[has_h3_controls]["dyad_id"].nunique(),
            "Pct_retained": 100 * has_h3_controls.sum() / len(dy_h3)
        })

    # Full H3 estimation sample
    h3_est = dy_h3["div_labor"].notna() & has_ideo
    if available_h3_controls:
        h3_est = h3_est & has_h3_controls
    audit_rows.append({
        "Hypothesis": "H3",
        "Stage": "4. Full estimation sample",
        "N_rows": h3_est.sum(),
        "N_units": dy_h3[h3_est]["dyad_id"].nunique(),
        "Pct_retained": 100 * h3_est.sum() / len(dy_h3)
    })

    # Save audit
    audit_df = pd.DataFrame(audit_rows)
    audit_df.to_csv(paths.audit_dir / "sample_attrition.csv", index=False)

    # =========================================================================
    # ALIGNMENT INDICATORS DISTRIBUTION
    # =========================================================================
    print("    Alignment indicators distribution...")

    align_dist = []
    if "has_dca" in dy.columns:
        align_dist.append({
            "Variable": "has_dca",
            "N_ones": int(dy["has_dca"].sum()),
            "N_zeros": int((dy["has_dca"] == 0).sum()),
            "Pct_ones": 100 * dy["has_dca"].mean()
        })
    if "any_atop_link" in dy.columns:
        align_dist.append({
            "Variable": "any_atop_link",
            "N_ones": int(dy["any_atop_link"].sum()),
            "N_zeros": int((dy["any_atop_link"] == 0).sum()),
            "Pct_ones": 100 * dy["any_atop_link"].mean()
        })
    if "any_alignment" in dy.columns:
        align_dist.append({
            "Variable": "any_alignment",
            "N_ones": int(dy["any_alignment"].sum()),
            "N_zeros": int((dy["any_alignment"] == 0).sum()),
            "Pct_ones": 100 * dy["any_alignment"].mean()
        })

    if align_dist:
        pd.DataFrame(align_dist).to_csv(paths.audit_dir / "alignment_distribution.csv", index=False)

    # =========================================================================
    # VARIABLE COVERAGE SUMMARY
    # =========================================================================
    print("    Variable coverage summary...")

    coverage = []
    key_vars = ["div_labor", "hierarchical", "voice_driven", "lngdp_ratio", "milex_ratio",
                "contiguous", "has_dca", "rile_dyad_mean"]
    for var in key_vars:
        if var in dy.columns:
            n_nonmissing = dy[var].notna().sum()
            coverage.append({
                "Variable": var,
                "N_nonmissing": n_nonmissing,
                "N_missing": len(dy) - n_nonmissing,
                "Pct_nonmissing": 100 * n_nonmissing / len(dy)
            })

    if coverage:
        pd.DataFrame(coverage).to_csv(paths.audit_dir / "variable_coverage.csv", index=False)

    # =========================================================================
    # INSTITUTION TYPE DISTRIBUTION
    # =========================================================================
    print("    Institution type distribution...")

    if "inst" in dy.columns:
        inst_dist = dy["inst"].value_counts().sort_index()
        inst_df = pd.DataFrame({
            "inst_code": inst_dist.index,
            "label": inst_dist.index.map({1: "Uninstitutionalized", 2: "Voice-driven", 3: "Hierarchical"}),
            "N": inst_dist.values,
            "Pct": 100 * inst_dist.values / len(dy)
        })
        inst_df.to_csv(paths.audit_dir / "inst_distribution.csv", index=False)

    # =========================================================================
    # GANNON CONTROLS SUMMARY STATISTICS
    # =========================================================================
    print("    Gannon controls summary statistics...")

    control_vars = ["lngdp_ratio", "milex_ratio", "contiguous"]
    summary_rows = []
    for var in control_vars:
        if var in dy.columns:
            col = dy[var]
            summary_rows.append({
                "Variable": var,
                "Mean": col.mean(),
                "Std": col.std(),
                "Min": col.min(),
                "P25": col.quantile(0.25),
                "Median": col.median(),
                "P75": col.quantile(0.75),
                "Max": col.max(),
                "N_nonmissing": col.notna().sum()
            })

    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(paths.audit_dir / "gannon_controls_summary.csv", index=False)

    # =========================================================================
    # DCAD COVERAGE WINDOW DIAGNOSTICS (CRITICAL)
    # =========================================================================
    print("    DCAD coverage window diagnostics...")

    dcad_audit = []

    # In-window vs out-of-window counts
    if "year" in dy.columns:
        in_window = (dy["year"] >= DCAD_YEAR_START) & (dy["year"] <= DCAD_YEAR_END)
        dcad_audit.append({
            "Metric": "Dyad-years in DCAD window (1980-2010)",
            "Value": int(in_window.sum()),
            "Pct_of_total": 100 * in_window.mean()
        })
        dcad_audit.append({
            "Metric": "Dyad-years outside DCAD window",
            "Value": int((~in_window).sum()),
            "Pct_of_total": 100 * (~in_window).mean()
        })

    # DCA status: observed vs unknown
    if "has_dca" in dy.columns:
        dca_observed = dy["has_dca"].notna().sum()
        dca_unknown = dy["has_dca"].isna().sum()
        dca_positive = (dy["has_dca"] == 1).sum()
        dcad_audit.append({
            "Metric": "DCA status observed (1980-2010)",
            "Value": int(dca_observed),
            "Pct_of_total": 100 * dca_observed / len(dy)
        })
        dcad_audit.append({
            "Metric": "DCA status unknown (outside 1980-2010)",
            "Value": int(dca_unknown),
            "Pct_of_total": 100 * dca_unknown / len(dy)
        })
        dcad_audit.append({
            "Metric": "Dyad-years with DCA=1 (in-window)",
            "Value": int(dca_positive),
            "Pct_of_total": 100 * dca_positive / len(dy) if len(dy) > 0 else 0
        })

    # 1980-2010 aligned sample diagnostics
    if dy_aligned is not None:
        dcad_audit.append({
            "Metric": "Aligned sample size (1980-2010)",
            "Value": len(dy_aligned),
            "Pct_of_total": 100 * len(dy_aligned) / len(dy) if len(dy) > 0 else 0
        })

        # Verify no unaligned rows
        n_unaligned = (dy_aligned.get("any_alignment", 1) == 0).sum()
        dcad_audit.append({
            "Metric": "Rows with any_alignment=0 in aligned sample",
            "Value": int(n_unaligned),
            "Pct_of_total": 0.0  # Should always be 0
        })

        # Share of aligned sample that is DCA-only
        if "dca_only" in dy_aligned.columns:
            n_dca_only = (dy_aligned["dca_only"] == 1).sum()
            dcad_audit.append({
                "Metric": "DCA-only dyad-years in aligned sample",
                "Value": int(n_dca_only),
                "Pct_of_total": 100 * n_dca_only / len(dy_aligned) if len(dy_aligned) > 0 else 0
            })

    if dcad_audit:
        pd.DataFrame(dcad_audit).to_csv(paths.audit_dir / "dcad_coverage_audit.csv", index=False)

    # =========================================================================
    # DYAD-YEAR COLLAPSE DIAGNOSTICS
    # =========================================================================
    print("    Dyad-year collapse diagnostics...")

    collapse_audit = []

    if "n_shared_alliances" in dy.columns:
        # Distribution of shared alliances
        n_alliances = dy["n_shared_alliances"]
        collapse_audit.append({
            "Metric": "Mean shared alliances per dyad-year",
            "Value": n_alliances.mean()
        })
        collapse_audit.append({
            "Metric": "Max shared alliances in any dyad-year",
            "Value": int(n_alliances.max())
        })
        collapse_audit.append({
            "Metric": "Dyad-years with >1 shared alliance",
            "Value": int((n_alliances > 1).sum())
        })
        collapse_audit.append({
            "Metric": "Pct dyad-years with >1 shared alliance",
            "Value": 100 * (n_alliances > 1).mean()
        })

    if collapse_audit:
        pd.DataFrame(collapse_audit).to_csv(paths.audit_dir / "collapse_diagnostics.csv", index=False)

    print(f"    Audit files saved to: {paths.audit_dir}/")
    print(f"      - sample_attrition.csv")
    print(f"      - alignment_distribution.csv")
    print(f"      - variable_coverage.csv")
    print(f"      - inst_distribution.csv")
    print(f"      - dcad_coverage_audit.csv")
    print(f"      - collapse_diagnostics.csv")
    print(f"      - gannon_controls_summary.csv")


if __name__ == "__main__":
    build_all()
