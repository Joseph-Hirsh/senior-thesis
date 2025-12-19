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


# =============================================================================
# Helpers (small, explicit, reproducible)
# =============================================================================

REQUIRED_MANIFESTO_COLS = {"country", "edate", "date", "absseat", "totseats", "pervote", "rile", "partyname"}


def _require_columns(df: pd.DataFrame, cols: set[str], name: str) -> None:
    missing = sorted([c for c in cols if c not in df.columns])
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _normalize_crosswalk(xw: pd.DataFrame) -> pd.DataFrame:
    """
    Accept either:
      - manifesto_country_code + country_code_cow
      - manifesto_country_code + cow_ccode
    and output exactly: manifesto_country_code, country_code_cow
    """
    if "manifesto_country_code" not in xw.columns:
        raise ValueError("Crosswalk must include column 'manifesto_country_code'.")

    if "country_code_cow" not in xw.columns:
        if "cow_ccode" in xw.columns:
            xw = xw.rename(columns={"cow_ccode": "country_code_cow"})
        else:
            raise ValueError(
                "Crosswalk must include either 'country_code_cow' or 'cow_ccode' as the COW code column."
            )

    xw = xw[["manifesto_country_code", "country_code_cow"]].copy()

    xw["manifesto_country_code"] = pd.to_numeric(xw["manifesto_country_code"], errors="coerce")
    xw["country_code_cow"] = pd.to_numeric(xw["country_code_cow"], errors="coerce")

    xw = xw.dropna(subset=["manifesto_country_code", "country_code_cow"]).copy()
    xw["manifesto_country_code"] = xw["manifesto_country_code"].astype(int)
    xw["country_code_cow"] = xw["country_code_cow"].astype(int)

    # Crosswalk should be one-to-one on manifesto_country_code
    if xw.duplicated(["manifesto_country_code"]).any():
        dup = xw[xw.duplicated(["manifesto_country_code"], keep=False)].sort_values("manifesto_country_code")
        raise ValueError(
            "Crosswalk is not unique on manifesto_country_code. Fix the CSV.\n"
            f"Examples:\n{dup.head(20).to_string(index=False)}"
        )

    return xw


def _parse_manifesto_election_year(manifesto: pd.DataFrame) -> pd.Series:
    """
    Transparent election-year parsing:
    - primary: parse edate using dayfirst=True (Manifesto often uses DD/MM/YYYY)
    - fallback: if still missing, use YYYYMM integer 'date' (year = date // 100)
    """
    # primary parse
    edate_parsed = pd.to_datetime(manifesto["edate"], errors="coerce", dayfirst=True)
    election_year = edate_parsed.dt.year

    # fallback: YYYYMM stored in date
    miss = election_year.isna() & manifesto["date"].notna()
    if miss.any():
        as_num = pd.to_numeric(manifesto.loc[miss, "date"], errors="coerce")
        election_year.loc[miss] = (as_num // 100)

    election_year = pd.to_numeric(election_year, errors="coerce")
    return election_year


# =============================================================================
# Loaders (hash + minimal coercion, no silent changes)
# =============================================================================

def load_spec(paths: Paths) -> pd.DataFrame:
    print("\nRaw file hash (spec RDS):", sha256_file(paths.spec_rds))
    r = pyreadr.read_r(paths.spec_rds)
    df = list(r.values())[0].copy()

    # Minimal, explicit key coercion
    df["country_code_cow"] = pd.to_numeric(df["country_code_cow"], errors="coerce")
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["country_code_cow", "year"]).copy()
    df["country_code_cow"] = df["country_code_cow"].astype(int)
    df["year"] = df["year"].astype(int)

    assert_unique_key(df, ["country_code_cow", "year"], "Specialization spine (spec)")
    return df


def load_manifesto(paths: Paths) -> pd.DataFrame:
    print("\nRaw file hash (manifesto CSV):", sha256_file(paths.manifesto_csv))
    df = pd.read_csv(paths.manifesto_csv, low_memory=False)
    return df


def load_rr(paths: Paths) -> pd.DataFrame:
    print("\nRaw file hash (R&R DTA):", sha256_file(paths.rr_dta))
    df = pd.read_stata(paths.rr_dta)
    return df


# =============================================================================
# Stage: Ideology panel (country-year)
# =============================================================================

def build_ideology_panel(spec: pd.DataFrame, manifesto: pd.DataFrame, paths: Paths) -> pd.DataFrame:
    """
    Builds a country-year ideology panel on the specialization spine.

    Steps:
    1) Merge manifesto → COW crosswalk
    2) Parse election year
    3) Define party strength: seat share, vote fallback
    4) Select "governing" proxy: largest party by strength within country-election
    5) Carry-forward RILE within country across years on the spec spine
    6) Construct RoC indicator (>=10 right, <=-10 left, else missing)
    """
    print("\n--- Building ideology panel ---")

    # Ensure manifesto has what we rely on (explicit, transparent)
    needed = {"country", "edate", "date", "absseat", "totseats", "pervote", "rile"}
    _require_columns(manifesto, needed, "Manifesto")

    # 1) attach crosswalk
    print("\nRaw file hash (manifesto crosswalk CSV):", sha256_file(paths.manifesto_crosswalk_csv))
    xw_raw = pd.read_csv(paths.manifesto_crosswalk_csv)
    xw = _normalize_crosswalk(xw_raw)

    manifesto = manifesto.merge(
        xw,
        left_on="country",
        right_on="manifesto_country_code",
        how="left",
        indicator="_xwalk_merge",
    )
    merge_audit(manifesto, "_xwalk_merge", "Manifesto → COW crosswalk")
    manifesto = manifesto.drop(columns=["_xwalk_merge"])

    # Fail fast if crosswalk didn’t produce what we need
    if "country_code_cow" not in manifesto.columns:
        raise ValueError("Crosswalk merge did not create 'country_code_cow' in Manifesto.")

    # 2) parse election year (edate primary, date fallback)
    manifesto["election_year"] = _parse_manifesto_election_year(manifesto)

    # 3) numeric coercions for relevant columns
    for c in ["absseat", "totseats", "pervote", "rile", "country_code_cow", "election_year"]:
        manifesto[c] = pd.to_numeric(manifesto[c], errors="coerce")

    # sanity: election year range
    if manifesto["election_year"].notna().any():
        yr_min = int(manifesto["election_year"].dropna().min())
        yr_max = int(manifesto["election_year"].dropna().max())
        print(f"Election year range (manifesto): {yr_min}–{yr_max}")

    # 4) governing proxy: largest party by seat_share, vote fallback
    manifesto["seat_share"] = manifesto["absseat"] / manifesto["totseats"]
    manifesto["party_strength"] = manifesto["seat_share"]
    fallback = manifesto["party_strength"].isna()
    manifesto.loc[fallback, "party_strength"] = manifesto.loc[fallback, "pervote"] / 100.0

    # Keep only usable records for gov-party selection
    m = manifesto.dropna(subset=["country_code_cow", "election_year", "party_strength"]).copy()
    m["country_code_cow"] = m["country_code_cow"].astype(int)
    m["election_year"] = m["election_year"].astype(int)

    # Sort and take first = max strength per country-election
    m = m.sort_values(["country_code_cow", "election_year", "party_strength"], ascending=[True, True, False])
    gov = m.groupby(["country_code_cow", "election_year"], as_index=False).first()

    # Keep only what we use downstream (transparent)
    keep_cols = ["country_code_cow", "election_year", "rile", "party_strength"]
    if "partyname" in gov.columns:
        keep_cols.append("partyname")
    gov = gov[keep_cols].copy()

    # 5) expand to spec spine and carry-forward
    spine = (
        spec[["country_code_cow", "year"]]
        .drop_duplicates()
        .sort_values(["country_code_cow", "year"])
        .copy()
    )
    gov = gov.rename(columns={"election_year": "year"}).sort_values(["country_code_cow", "year"])

    # Merge elections onto spine and ffill within each country
    panel = spine.merge(gov, on=["country_code_cow", "year"], how="left")
    panel["rile"] = pd.to_numeric(panel["rile"], errors="coerce")

    # Carry-forward within country
    panel["rile"] = panel.groupby("country_code_cow", sort=False)["rile"].ffill()
    if "partyname" in panel.columns:
        panel["partyname"] = panel.groupby("country_code_cow", sort=False)["partyname"].ffill()
    if "party_strength" in panel.columns:
        panel["party_strength"] = panel.groupby("country_code_cow", sort=False)["party_strength"].ffill()

    # 6) RoC/LoC indicator
    panel["right_of_center"] = np.where(
        panel["rile"] >= 10, 1,
        np.where(panel["rile"] <= -10, 0, np.nan),
    )

    assert_unique_key(panel, ["country_code_cow", "year"], "Ideology panel")
    coverage_audit(panel, "country_code_cow", "year", "Ideology panel")

    panel.to_csv(paths.ideology_out, index=False)
    print("Saved ideology panel:", paths.ideology_out)
    return panel


# =============================================================================
# Stage: R&R dyad-year → country-year aggregates
# =============================================================================

def build_rr_aggregates(rr: pd.DataFrame, paths: Paths) -> pd.DataFrame:
    """
    Converts dyad-year to monadic country-year aggregates.

    Notes:
    - This is a *robustness* dataset if it is small.
    - We average dyad features across a country's dyads in a given year,
      and also compute dyad_count.
    """
    print("\n--- Building R&R aggregates ---")

    _require_columns(rr, {"state_a", "state_b", "yrent"}, "R&R")

    rr = rr.copy()
    rr["state_a"] = pd.to_numeric(rr["state_a"], errors="coerce")
    rr["state_b"] = pd.to_numeric(rr["state_b"], errors="coerce")
    rr["year"] = pd.to_numeric(rr["yrent"], errors="coerce")
    rr = rr.dropna(subset=["state_a", "state_b", "year"]).copy()
    rr["state_a"] = rr["state_a"].astype(int)
    rr["state_b"] = rr["state_b"].astype(int)
    rr["year"] = rr["year"].astype(int)

    # dedupe dyad-year transparently
    key = ["state_a", "state_b", "year"]
    if rr.duplicated(key).any():
        print("R&R: duplicate dyad-years detected → averaging duplicates (numeric columns only).")
        rr = rr.groupby(key, as_index=False).mean(numeric_only=True)

    id_cols = {"atopid", "state_a", "state_b", "year", "yrent", "dyadid"}
    feat = [c for c in rr.columns if c not in id_cols]

    # numeric coercion for features
    for c in feat:
        rr[c] = pd.to_numeric(rr[c], errors="coerce")

    # stack to monadic
    a = rr[["year", "state_a"] + feat].rename(columns={"state_a": "country_code_cow"})
    b = rr[["year", "state_b"] + feat].rename(columns={"state_b": "country_code_cow"})
    stacked = pd.concat([a, b], ignore_index=True)
    stacked["dyad_count"] = 1

    grouped = stacked.groupby(["country_code_cow", "year"], as_index=False)
    out = grouped[feat].mean()
    out = out.merge(grouped["dyad_count"].sum(), on=["country_code_cow", "year"], how="left")

    # prefix all rr variables
    rename = {c: f"rr_{c}" for c in out.columns if c not in ["country_code_cow", "year", "dyad_count"]}
    out = out.rename(columns=rename)

    assert_unique_key(out, ["country_code_cow", "year"], "R&R aggregates")
    coverage_audit(out, "country_code_cow", "year", "R&R aggregates")

    out.to_csv(paths.rr_out, index=False)
    print("Saved R&R aggregates:", paths.rr_out)
    return out


# =============================================================================
# Build master
# =============================================================================

def build_master() -> None:
    paths = Paths()

    spec = load_spec(paths)
    manifesto = load_manifesto(paths)
    rr = load_rr(paths)

    print_schema("Specialization (raw spine)", spec)
    print_schema("Manifesto (raw)", manifesto)
    print_schema("R&R (raw)", rr)

    ideology = build_ideology_panel(spec, manifesto, paths)
    rr_aggs = build_rr_aggregates(rr, paths)

    # Merge 1: spec + ideology
    master = spec.merge(ideology, on=["country_code_cow", "year"], how="left", indicator="_m1")
    merge_audit(master, "_m1", "Spec + Ideology")
    master = master.drop(columns=["_m1"])

    # FILTER: Only keep observations where we have a RILE score
    pre_filter_n = len(master)
    master = master.dropna(subset=["rile"]).copy()
    post_filter_n = len(master)
    print(f"\nRILE filter: kept {post_filter_n} of {pre_filter_n} rows ({post_filter_n/pre_filter_n:.1%})")

    # Merge 2: + rr
    master = master.merge(rr_aggs, on=["country_code_cow", "year"], how="left", indicator="_m2")
    merge_audit(master, "_m2", "Master + R&R")
    master = master.drop(columns=["_m2"])

    assert_unique_key(master, ["country_code_cow", "year"], "Master dataset")
    coverage_audit(master, "country_code_cow", "year", "Master dataset")

    master.to_csv(paths.master_out, index=False)
    print("Saved master dataset:", paths.master_out)


def main() -> int:
    build_master()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
