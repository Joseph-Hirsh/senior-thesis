"""
Dataset construction for military specialization and alliance institutions study.

Creates two analysis datasets:
1. master_country_year.csv - For testing H1 (ideology -> specialization)
2. master_dyad_year.csv - For testing H2/H2A/H2B (alliance institutions -> specialization)
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

__all__ = [
    "build_country_year",
    "build_dyad_year",
    "build_all",
]


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


def _load_rr(paths: Paths) -> pd.DataFrame:
    """Load Rapport & Rathbun alliance dyad assets."""
    print(f"  Loading R&R assets [{file_hash(paths.rr_dta)}]")
    return pd.read_stata(paths.rr_dta)


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


def _expand_dyad_years(rr: pd.DataFrame) -> pd.DataFrame:
    """
    Expand dyads to dyad-year panel using vectorized operations.

    Args:
        rr: R&R dyad assets with dyad_start and dyad_end columns

    Returns:
        Expanded dyad-year panel
    """
    # Calculate number of years for each dyad
    rr = rr.copy()
    rr["n_years"] = (rr["dyad_end"] - rr["dyad_start"] + 1).clip(lower=0)

    # Repeat rows by number of years
    expanded = rr.loc[rr.index.repeat(rr["n_years"])].copy()

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

    if "lngdp_WDI_full" in master.columns:
        master["lngdp"] = master["lngdp_WDI_full"]
    elif "gdp_WDI_full" in master.columns:
        master["lngdp"] = _safe_log(master["gdp_WDI_full"])

    if "cinc_MC" in master.columns:
        master["cinc"] = master["cinc_MC"]

    if "interstatewar_5yrlag_binary" in master.columns:
        master["war5_lag"] = pd.to_numeric(
            master["interstatewar_5yrlag_binary"], errors="coerce"
        )

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
        coding conventions. Alliances with missing values on ALL institutional variables
        are flagged in the output.
    """
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

    return inst[["atopid", "inst"]]


def build_dyad_year(paths: Paths) -> pd.DataFrame:
    """
    Build dyad-year dataset for H2 (alliance type -> division of labor).

    Uses Leeds & Anac (2005) institutionalization coding from ATOP treaty provisions.
    This is a 3-category NOMINAL classification (not ordinal):

    - Hierarchical (inst=3): INTCOM=1 OR MILCON=3 OR BASE>0 OR SUBORD in {1,2}
    - Voice-driven (inst=2): MILCON=2 OR military org present OR MILAID in {3,4} OR CONTRIB=1
    - Uninstitutionalized (inst=1): None of above

    Output variables:
    - atopid, state_a, state_b, year (IDs)
    - div_labor (DV: pairwise portfolio complementarity, 0=identical, 1=fully dissimilar)
    - hierarchical, voice_driven (IVs, nominal dummies)
    - rile_dyad_mean
    """
    print("\n  Building dyad-year dataset...")
    print("  Using Leeds & Anac (2005) institutionalization coding from ATOP")

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

    # Create ratio variables (larger / smaller for consistent interpretation)
    # GDP ratio: ratio of log GDPs
    if "lngdp_a" in panel.columns and "lngdp_b" in panel.columns:
        lngdp_max = panel[["lngdp_a", "lngdp_b"]].max(axis=1)
        lngdp_min = panel[["lngdp_a", "lngdp_b"]].min(axis=1)
        panel["gdp_ratio"] = lngdp_max / lngdp_min.replace(0, np.nan)

    # CINC ratio: ratio of military capabilities
    if "cinc_a" in panel.columns and "cinc_b" in panel.columns:
        cinc_max = panel[["cinc_a", "cinc_b"]].max(axis=1)
        cinc_min = panel[["cinc_a", "cinc_b"]].min(axis=1)
        panel["cinc_ratio"] = cinc_max / cinc_min.replace(0, np.nan)

    # Create decade variable for FE
    panel["decade"] = (panel["year"] // 10) * 10

    # Create binary dummies for institution type
    panel["hierarchical"] = (panel["inst"] == 3).astype(int)
    panel["voice_driven"] = (panel["inst"] == 2).astype(int)

    # Verify and save
    assert_unique_key(panel, ["atopid", "state_a", "state_b", "year"], "dyad-year")
    coverage_report(panel, "atopid", "year", "dyad-year")

    # Coverage summary
    div_labor_n = panel["div_labor"].notna().sum()
    contiguous_n = panel["contiguous"].sum()
    gdp_ratio_n = panel["gdp_ratio"].notna().sum() if "gdp_ratio" in panel.columns else 0
    cinc_ratio_n = panel["cinc_ratio"].notna().sum() if "cinc_ratio" in panel.columns else 0
    print(f"  Coverage:")
    print(f"    div_labor={div_labor_n:,} ({div_labor_n/len(panel):.1%})")
    print(f"    contiguous={contiguous_n:,} dyad-years ({contiguous_n/len(panel):.1%})")
    print(f"    gdp_ratio={gdp_ratio_n:,} ({gdp_ratio_n/len(panel):.1%})")
    print(f"    cinc_ratio={cinc_ratio_n:,} ({cinc_ratio_n/len(panel):.1%})")

    panel.to_csv(paths.dyad_year_csv, index=False)
    print(f"  Saved: {paths.dyad_year_csv}")

    return panel


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

    build_country_year(paths)
    build_dyad_year(paths)

    # Add alliance membership to country-year (requires dyad assets)
    _add_alliance_membership(paths)

    print("\n  All datasets built successfully.")


if __name__ == "__main__":
    build_all()
