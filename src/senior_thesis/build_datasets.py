"""
Dataset construction for military specialization and alliance institutions study.

Creates two analysis datasets:
1. master_country_year.csv - For testing H1 (ideology -> specialization)
2. master_dyad_year.csv - For testing H2/H2A/H2B (alliance institutions -> specialization)
"""
from __future__ import annotations

import logging

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
    setup_logging,
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

logger = logging.getLogger("senior_thesis")


def _safe_log(x: pd.Series) -> pd.Series:
    """Natural log, setting non-positive values to NaN."""
    return np.log(x.where(x > 0))


def _load_specialization(paths: Paths) -> pd.DataFrame:
    """Load Gannon's military specialization data (country-year spine)."""
    logger.info(f"Loading specialization data [{file_hash(paths.spec_rds)}]")
    df = list(pyreadr.read_r(paths.spec_rds).values())[0].copy()

    # Clean keys
    df["country_code_cow"] = pd.to_numeric(df["country_code_cow"], errors="coerce")
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["country_code_cow", "year"])
    df["country_code_cow"] = df["country_code_cow"].astype(int)
    df["year"] = df["year"].astype(int)

    return df


def _load_manifesto(paths: Paths) -> pd.DataFrame:
    """Load Manifesto Project party ideology data."""
    logger.info(f"Loading Manifesto data [{file_hash(paths.manifesto_csv)}]")
    return pd.read_csv(paths.manifesto_csv, low_memory=False)


def _load_crosswalk(paths: Paths) -> pd.DataFrame:
    """Load Manifesto-to-COW country code crosswalk."""
    logger.info(f"Loading crosswalk [{file_hash(paths.crosswalk_csv)}]")
    cw = pd.read_csv(paths.crosswalk_csv)

    # Normalize column names
    if "cow_ccode" in cw.columns:
        cw = cw.rename(columns={"cow_ccode": "country_code_cow"})

    cw = cw[["manifesto_country_code", "country_code_cow"]].dropna()
    cw["manifesto_country_code"] = cw["manifesto_country_code"].astype(int)
    cw["country_code_cow"] = cw["country_code_cow"].astype(int)

    return cw


def _load_rr(paths: Paths) -> pd.DataFrame:
    """Load Rapport & Rathbun alliance dyad data."""
    logger.info(f"Loading R&R data [{file_hash(paths.rr_dta)}]")
    return pd.read_stata(paths.rr_dta)


def _load_atop(paths: Paths) -> pd.DataFrame:
    """Load ATOP alliance membership data."""
    logger.info(f"Loading ATOP membership [{file_hash(paths.atop_csv)}]")
    return pd.read_csv(paths.atop_csv, low_memory=False)


def _load_depth(paths: Paths) -> pd.DataFrame:
    """Load Benson & Clinton alliance depth scores."""
    logger.info(f"Loading alliance depth [{file_hash(paths.depth_csv)}]")
    return pd.read_csv(paths.depth_csv, low_memory=False)


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

    # Select governing party (strongest per country-election)
    mf = mf.dropna(subset=["country_code_cow", "election_year", "strength"])
    mf["country_code_cow"] = mf["country_code_cow"].astype(int)
    mf["election_year"] = mf["election_year"].astype(int)
    mf = mf.sort_values(
        ["country_code_cow", "election_year", "strength"], ascending=[True, True, False]
    )
    gov = mf.groupby(["country_code_cow", "election_year"], as_index=False).first()
    gov = gov[["country_code_cow", "election_year", "rile"]].rename(
        columns={"election_year": "year"}
    )

    # Expand to country-year spine
    spine = spec[["country_code_cow", "year"]].drop_duplicates()
    panel = spine.merge(gov, on=["country_code_cow", "year"], how="left")

    # Forward-fill ideology within country
    panel = panel.sort_values(["country_code_cow", "year"])
    panel["rile"] = panel.groupby("country_code_cow")["rile"].ffill()

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
    Merge data for both dyad partners (state_a and state_b).

    Args:
        df: Target dataframe with state_a, state_b, and year columns
        data: Source dataframe with key_col, year, and var columns
        var: Variable name to merge (will become var_a and var_b)
        key_col: Key column name in source data

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
        rr: R&R dyad data with dyad_start and dyad_end columns

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
    logger.info("Building country-year dataset...")

    # Load raw data
    spec = _load_specialization(paths)
    manifesto = _load_manifesto(paths)
    crosswalk = _load_crosswalk(paths)

    # Build ideology panel
    ideology = _build_ideology_panel(spec, manifesto, crosswalk)

    # Merge with specialization data
    master = spec.merge(ideology, on=["country_code_cow", "year"], how="left", indicator="_m")
    merge_report(master, "_m", "spec + ideology")
    master = master.drop(columns=["_m"])

    # Filter to observations with ideology data
    n_before = len(master)
    master = master.dropna(subset=["rile"])
    logger.info(f"Filtered to rows with RILE: {len(master):,} of {n_before:,}")

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

    master.to_csv(paths.country_year_csv, index=False)
    logger.info(f"Saved: {paths.country_year_csv}")

    return master


def build_dyad_year(paths: Paths) -> pd.DataFrame:
    """
    Build dyad-year dataset for H2/H2A/H2B (alliance institutions -> specialization).

    Output variables:
    - atopid, state_a, state_b, year (IDs)
    - spec_dyad_mean (DV: mean partner specialization)
    - Depth.score (IV for H2)
    - hierarchical, voice_driven (IVs for H2A/H2B)
    - rile_dyad_mean, R&R controls
    """
    logger.info("Building dyad-year dataset...")

    # Load raw data
    rr = _load_rr(paths)
    atop = _load_atop(paths)
    depth = _load_depth(paths)

    # Load country-year master for specialization and ideology
    master_cy = pd.read_csv(paths.country_year_csv)

    # Clean R&R dyads
    rr = rr.copy()
    for col in ["atopid", "state_a", "state_b", "yrent"]:
        rr[col] = pd.to_numeric(rr[col], errors="coerce")
    rr = rr.dropna(subset=["atopid", "state_a", "state_b", "yrent"])
    rr["atopid"] = rr["atopid"].astype(int)
    rr["state_a"] = rr["state_a"].astype(int)
    rr["state_b"] = rr["state_b"].astype(int)
    rr["yrent"] = rr["yrent"].astype(int)

    logger.info(
        f"R&R dyads: {len(rr)} (inst=1: {(rr['inst']==1).sum()}, "
        f"inst=2: {(rr['inst']==2).sum()}, inst=3: {(rr['inst']==3).sum()})"
    )

    # Get exit years from ATOP
    atop["atopid"] = pd.to_numeric(atop["atopid"], errors="coerce")
    atop["member"] = pd.to_numeric(atop["member"], errors="coerce")
    atop["yrexit"] = pd.to_numeric(atop["yrexit"], errors="coerce")
    atop = atop.dropna(subset=["atopid", "member"])
    atop["atopid"] = atop["atopid"].astype(int)
    atop["member"] = atop["member"].astype(int)

    exits = atop.groupby(["atopid", "member"])["yrexit"].max().reset_index()
    exits.columns = ["atopid", "member", "exit_year"]

    # Merge exit years for both states
    rr = rr.merge(
        exits.rename(columns={"member": "state_a", "exit_year": "exit_a"}),
        on=["atopid", "state_a"],
        how="left",
    )
    rr = rr.merge(
        exits.rename(columns={"member": "state_b", "exit_year": "exit_b"}),
        on=["atopid", "state_b"],
        how="left",
    )

    # Handle missing/zero exit years (0 = still active)
    rr["exit_a"] = rr["exit_a"].fillna(YEAR_END).replace(0, YEAR_END).astype(int)
    rr["exit_b"] = rr["exit_b"].fillna(YEAR_END).replace(0, YEAR_END).astype(int)
    rr["dyad_end"] = rr[["exit_a", "exit_b"]].min(axis=1).clip(upper=YEAR_END)
    rr["dyad_start"] = rr["yrent"].clip(lower=YEAR_START)

    # Expand to dyad-year panel (vectorized)
    panel = _expand_dyad_years(rr)
    logger.info(f"Expanded to {len(panel):,} dyad-years")

    # Merge R&R controls
    rr_cols = [
        "atopid",
        "state_a",
        "state_b",
        "inst",
        "jntdem",
        "priorviol",
        "s_un_glo",
        "tot_rivals",
        "totmids2",
        "undist",
        "coldwar",
        "symm",
        "lncprtio",
    ]
    rr_cols = [c for c in rr_cols if c in rr.columns]
    panel = panel.merge(rr[rr_cols], on=["atopid", "state_a", "state_b"], how="left")

    # Merge specialization for both states using helper
    spec_data = master_cy[["country_code_cow", "year", "spec_y"]].copy()
    panel = _merge_by_partner(panel, spec_data, "spec_y")
    panel["spec_dyad_mean"] = (panel["spec_y_a"] + panel["spec_y_b"]) / 2

    # Merge ideology for both states using helper
    rile_data = master_cy[["country_code_cow", "year", "rile"]].copy()
    panel = _merge_by_partner(panel, rile_data, "rile")
    panel["rile_dyad_mean"] = (panel["rile_a"] + panel["rile_b"]) / 2

    # Merge depth scores
    depth_data = depth[["atopid", "Depth.score"]].copy()
    depth_data["atopid"] = pd.to_numeric(depth_data["atopid"], errors="coerce").astype(int)
    panel = panel.merge(depth_data, on="atopid", how="left")

    # Create binary dummies
    panel["hierarchical"] = (panel["inst"] == 3).astype(int)
    panel["voice_driven"] = (panel["inst"] == 2).astype(int)
    panel["priorviol_x_symm"] = panel["priorviol"] * panel["symm"]

    # Create depth_within_type (residualized depth)
    mask = panel["Depth.score"].notna() & panel["inst"].notna()
    if mask.sum() > 0:
        X = sm.add_constant(
            panel.loc[mask, ["hierarchical", "voice_driven"]].values.astype(float)
        )
        y = panel.loc[mask, "Depth.score"].values.astype(float)
        resid = sm.OLS(y, X).fit().resid
        panel.loc[mask, "depth_within_type"] = resid

    # Verify and save
    assert_unique_key(panel, ["atopid", "state_a", "state_b", "year"], "dyad-year")
    coverage_report(panel, "atopid", "year", "dyad-year")

    # Coverage summary
    spec_n = panel["spec_dyad_mean"].notna().sum()
    depth_n = panel["Depth.score"].notna().sum()
    logger.info(
        f"Coverage: spec_dyad_mean={spec_n:,} ({spec_n/len(panel):.1%}), "
        f"Depth.score={depth_n:,} ({depth_n/len(panel):.1%})"
    )

    panel.to_csv(paths.dyad_year_csv, index=False)
    logger.info(f"Saved: {paths.dyad_year_csv}")

    return panel


def build_all() -> None:
    """Build all datasets."""
    setup_logging()
    paths = Paths()

    # Validate input files
    missing = paths.validate()
    if missing:
        logger.error(f"Missing input files: {missing}")
        return

    # Ensure output directories exist
    paths.country_year_csv.parent.mkdir(parents=True, exist_ok=True)
    paths.dyad_year_csv.parent.mkdir(parents=True, exist_ok=True)

    build_country_year(paths)
    build_dyad_year(paths)

    logger.info("All datasets built successfully.")


if __name__ == "__main__":
    build_all()
