from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from senior_thesis.config import Paths
from senior_thesis.audit import sha256_file


def _safe_log(x: pd.Series) -> pd.Series:
    x = pd.to_numeric(x, errors="coerce")
    x = x.where(x > 0)
    return np.log(x)


def _prep_master(paths: Paths) -> pd.DataFrame:
    print("\nRaw file hash (master CSV):", sha256_file(paths.master_out))
    df = pd.read_csv(paths.master_out, low_memory=False)

    # --- core vars ---
    # Specialization outcome: prefer spec_stand if present, else spec_intscale, else spec_raw
    y_candidates = [c for c in ["spec_stand", "spec_intscale", "spec_raw"] if c in df.columns]
    if not y_candidates:
        raise ValueError("No specialization variable found (expected one of spec_stand/spec_intscale/spec_raw).")
    df["spec_y"] = pd.to_numeric(df[y_candidates[0]], errors="coerce")

    df["country_code_cow"] = pd.to_numeric(df["country_code_cow"], errors="coerce").astype("Int64")
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")

    # ideology
    if "rile" not in df.columns:
        raise ValueError("Column 'rile' not found in master dataset (expected from ideology panel).")
    df["rile"] = pd.to_numeric(df["rile"], errors="coerce")

    # RoC/LoC coding: RoC=1 if rile>=10; LoC=0 if rile<=-10; else missing (centrist/unclear)
    df["right_of_center"] = np.where(df["rile"] >= 10, 1, np.where(df["rile"] <= -10, 0, np.nan))

    # CRITICAL: Create lagged ideology variables (force structure responds with delay)
    # Sort by country-year to ensure proper lagging within countries
    df = df.sort_values(["country_code_cow", "year"])

    # Create 1-20 year lags of ideology (grouped by country to avoid cross-country contamination)
    for lag in range(1, 21):
        df[f"rile_lag{lag}"] = df.groupby("country_code_cow", sort=False)["rile"].shift(lag)
        df[f"right_of_center_lag{lag}"] = df.groupby("country_code_cow", sort=False)["right_of_center"].shift(lag)

    # --- controls in the style of your example table ---
    # Allies' Mil Spend (Log)
    if "milex_allies" in df.columns:
        df["ln_milex_allies"] = _safe_log(df["milex_allies"])
    else:
        df["ln_milex_allies"] = np.nan

    # Allies' CINC Ratio
    if "cinc_allies_ratio" in df.columns:
        df["cinc_allies_ratio"] = pd.to_numeric(df["cinc_allies_ratio"], errors="coerce")
    else:
        df["cinc_allies_ratio"] = np.nan

    # NOTE: Democracy binary excluded - all cases in sample are democracies (no variation)

    # Interstate War (5 yr Lag)
    if "interstatewar_5yrlag_binary" in df.columns:
        w = df["interstatewar_5yrlag_binary"]
        if w.dtype.name == "category" or w.dtype == object:
            df["war5_lag"] = pd.to_numeric(w.astype(str), errors="coerce")
        else:
            df["war5_lag"] = pd.to_numeric(w, errors="coerce")
    elif "interstatewar_binary" in df.columns:
        df["war5_lag"] = pd.to_numeric(df["interstatewar_binary"], errors="coerce")
    else:
        df["war5_lag"] = np.nan

    # GDP (Log): use lngdp_WDI_full if present, else log(gdp_WDI_full)
    if "lngdp_WDI_full" in df.columns:
        df["lngdp"] = pd.to_numeric(df["lngdp_WDI_full"], errors="coerce")
    elif "gdp_WDI_full" in df.columns:
        df["lngdp"] = _safe_log(df["gdp_WDI_full"])
    else:
        df["lngdp"] = np.nan

    # CINC: use cinc_MC if present
    if "cinc_MC" in df.columns:
        df["cinc"] = pd.to_numeric(df["cinc_MC"], errors="coerce")
    else:
        df["cinc"] = np.nan

    # Time trends (for "Cubic Poly" specs)
    df["year_num"] = pd.to_numeric(df["year"], errors="coerce")
    df["t"] = df["year_num"] - df["year_num"].min()
    df["t2"] = df["t"] ** 2
    df["t3"] = df["t"] ** 3

    # cast FE ids to int for cleaner formulas
    df = df.dropna(subset=["country_code_cow", "year"])
    df["country_code_cow"] = df["country_code_cow"].astype(int)
    df["year"] = df["year"].astype(int)

    return df


def _print_coverage(df: pd.DataFrame, cols: list[str], label: str) -> None:
    print(f"\n--- Coverage check: {label} ---")
    for c in cols:
        nn = df[c].notna().sum()
        print(f"{c:>22}: {nn:>6} / {len(df)} ({nn/len(df):.1%})")


def _run_ols(df: pd.DataFrame, formula: str, label: str) -> None:
    # cluster by country (typical in country-year panels)
    model = smf.ols(formula, data=df)
    res = model.fit(cov_type="cluster", cov_kwds={"groups": df["country_code_cow"]})
    print(f"\n=== {label} ===")
    print(f"Sample size: {int(res.nobs)}")
    print(res.summary().tables[1])
    # You can also print res.summary() if you want the full block.


def run_analysis() -> None:
    paths = Paths()
    df = _prep_master(paths)

    # Controls: base + both alliance variables
    base_controls = [
        "lngdp",
        "cinc",
        "war5_lag",
    ]

    alliance_controls = ["ln_milex_allies", "cinc_allies_ratio"]
    all_controls = base_controls + alliance_controls

    # Coverage diagnostics
    lag_vars = [f"right_of_center_lag{i}" for i in range(1, 6)]
    _print_coverage(df, ["spec_y", "right_of_center"] + lag_vars + all_controls,
                    "Key variables (full master)")

    print("\n" + "="*80)
    print("IMPORTANT: Using LAGGED ideology to allow for policy implementation delays")
    print("Force structure changes take 2-5 years. Current-year ideology is misspecified.")
    print("="*80)

    # Helper to build formulas
    def fml(ideo: str) -> str:
        rhs = [ideo] + all_controls + ["C(country_code_cow)", "C(year)"]
        return "spec_y ~ " + " + ".join(rhs)

    # -------------------------
    # LAG SENSITIVITY ANALYSIS: Lags 1-5
    # -------------------------
    print("\n" + "="*80)
    print("LAG SENSITIVITY ANALYSIS: Testing 1, 2, 3, 4, and 5 year lags")
    print("Specification: Year FE + GDP (Log) + CINC + War (5yr lag) + Allies' Mil Spend (Log) + Allies' CINC Ratio")
    print("="*80)

    for lag in range(1, 21):
        df_lag = df.dropna(subset=["spec_y", f"right_of_center_lag{lag}"] + all_controls + ["country_code_cow", "year"])
        _run_ols(
            df_lag,
            fml(f"right_of_center_lag{lag}"),
            f"Binary RoC (t-{lag}) | Year FE + Full Controls | N={len(df_lag)}"
        )


def main() -> None:
    run_analysis()


if __name__ == "__main__":
    main()
