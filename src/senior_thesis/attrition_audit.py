"""
Attrition and Missingness Audit for Military Specialization Study.

This module provides rigorous documentation of sample attrition at each stage
of the analysis pipeline, enabling transparent reporting of:
- Which observations enter each regression and why
- How missingness affects sample composition
- Whether estimation samples differ systematically from full samples

The audit framework is designed to be extensible: as data coverage improves,
the audit automatically reflects those improvements.

Author: [Your Name]
Date: 2024
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from senior_thesis.config import Paths, COUNTRY_CONTROLS, DYAD_CONTROLS

__all__ = [
    "StageResult",
    "apply_stage",
    "audit_h1",
    "audit_h2",
    "run_full_audit",
]


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class StageResult:
    """
    Container for attrition audit results at a single pipeline stage.

    Captures comprehensive information about sample composition and attrition
    to enable transparent reporting of data flow through the analysis.
    """
    stage_name: str
    n_rows: int
    n_units: int  # countries or dyads depending on context
    year_min: int
    year_max: int
    pct_remaining: float  # relative to previous stage
    required_cols: list[str] = field(default_factory=list)
    missing_counts: dict[str, int] = field(default_factory=dict)
    drop_reasons: list[tuple[str, int]] = field(default_factory=list)  # (reason, count)

    def to_dict(self) -> dict:
        """Convert to dictionary for DataFrame construction."""
        return {
            "stage": self.stage_name,
            "n_rows": self.n_rows,
            "n_units": self.n_units,
            "year_min": self.year_min,
            "year_max": self.year_max,
            "pct_remaining": round(self.pct_remaining * 100, 2),
            "required_cols": ", ".join(self.required_cols) if self.required_cols else "",
            "top_drop_reasons": "; ".join([f"{r}: {c}" for r, c in self.drop_reasons[:5]]),
        }


@dataclass
class BalanceResult:
    """Container for balance comparison between samples."""
    variable: str
    mean_full: float
    mean_estimation: float
    std_full: float
    std_estimation: float
    n_full: int
    n_estimation: int
    smd: float  # standardized mean difference

    def to_dict(self) -> dict:
        return {
            "variable": self.variable,
            "mean_full": round(self.mean_full, 4),
            "mean_estimation": round(self.mean_estimation, 4),
            "std_full": round(self.std_full, 4),
            "std_estimation": round(self.std_estimation, 4),
            "n_full": self.n_full,
            "n_estimation": self.n_estimation,
            "smd": round(self.smd, 4),
        }


# =============================================================================
# Core Audit Functions
# =============================================================================


def apply_stage(
    df: pd.DataFrame,
    name: str,
    prev_n: int,
    required_cols: Optional[list[str]] = None,
    filter_fn: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
    unit_cols: list[str] = None,
    year_col: str = "year",
    key_cols: list[str] = None,
    prev_df: Optional[pd.DataFrame] = None,
) -> tuple[pd.DataFrame, StageResult]:
    """
    Apply a pipeline stage and compute attrition statistics.

    Args:
        df: Input DataFrame
        name: Human-readable stage name
        prev_n: Row count from previous stage (for % remaining calculation)
        required_cols: Columns that must be non-missing (will filter to complete cases)
        filter_fn: Optional custom filter function
        unit_cols: Columns defining unique units (e.g., ["country_code_cow"])
        year_col: Column containing year
        key_cols: Columns forming unique row identifier for drop reason analysis
        prev_df: Previous stage DataFrame for drop reason analysis

    Returns:
        Tuple of (filtered DataFrame, StageResult)
    """
    df = df.copy()

    # Track missing counts before filtering
    missing_counts = {}
    if required_cols:
        for col in required_cols:
            if col in df.columns:
                missing_counts[col] = int(df[col].isna().sum())
            else:
                missing_counts[col] = len(df)  # Column doesn't exist

    # Apply required columns filter
    if required_cols:
        cols_present = [c for c in required_cols if c in df.columns]
        if cols_present:
            df = df.dropna(subset=cols_present)

    # Apply custom filter
    if filter_fn is not None:
        df = filter_fn(df)

    # Compute statistics
    n_rows = len(df)
    pct_remaining = n_rows / prev_n if prev_n > 0 else 1.0

    # Compute unique units
    if unit_cols:
        n_units = df[unit_cols].drop_duplicates().shape[0]
    else:
        n_units = n_rows

    # Year range
    if year_col in df.columns and len(df) > 0:
        year_min = int(df[year_col].min())
        year_max = int(df[year_col].max())
    else:
        year_min = year_max = 0

    # Compute drop reasons (comparing to previous stage)
    drop_reasons = []
    if prev_df is not None and key_cols and required_cols:
        # Identify dropped rows
        prev_keys = set(prev_df[key_cols].apply(tuple, axis=1))
        curr_keys = set(df[key_cols].apply(tuple, axis=1))
        dropped_keys = prev_keys - curr_keys

        if dropped_keys:
            # Get dropped rows
            dropped_mask = prev_df[key_cols].apply(tuple, axis=1).isin(dropped_keys)
            dropped_df = prev_df[dropped_mask]

            # Count missing values per column among dropped rows
            reasons = {}
            for col in required_cols:
                if col in dropped_df.columns:
                    n_missing = int(dropped_df[col].isna().sum())
                    if n_missing > 0:
                        reasons[f"missing {col}"] = n_missing

            # Sort by frequency
            drop_reasons = sorted(reasons.items(), key=lambda x: -x[1])

    result = StageResult(
        stage_name=name,
        n_rows=n_rows,
        n_units=n_units,
        year_min=year_min,
        year_max=year_max,
        pct_remaining=pct_remaining,
        required_cols=required_cols or [],
        missing_counts=missing_counts,
        drop_reasons=drop_reasons,
    )

    return df, result


def compute_smd(mean1: float, mean2: float, std1: float, std2: float, n1: int, n2: int) -> float:
    """
    Compute standardized mean difference (SMD) between two groups.

    Uses pooled standard deviation for continuous variables.
    For binary variables, this is equivalent to Cohen's d.
    """
    if n1 <= 1 or n2 <= 1:
        return np.nan

    # Pooled standard deviation
    pooled_var = ((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2)
    pooled_std = np.sqrt(pooled_var)

    if pooled_std == 0 or np.isnan(pooled_std):
        return np.nan

    return (mean1 - mean2) / pooled_std


def compute_balance(
    full_df: pd.DataFrame,
    estimation_df: pd.DataFrame,
    variables: list[str],
) -> list[BalanceResult]:
    """
    Compute balance statistics comparing full sample to estimation sample.

    Args:
        full_df: Full sample DataFrame
        estimation_df: Estimation sample DataFrame
        variables: Variables to compare

    Returns:
        List of BalanceResult objects
    """
    results = []

    for var in variables:
        if var not in full_df.columns:
            continue

        # Full sample statistics
        full_data = full_df[var].dropna()
        est_data = estimation_df[var].dropna() if var in estimation_df.columns else pd.Series(dtype=float)

        if len(full_data) == 0:
            continue

        mean_full = float(full_data.mean())
        std_full = float(full_data.std())
        n_full = len(full_data)

        if len(est_data) > 0:
            mean_est = float(est_data.mean())
            std_est = float(est_data.std())
            n_est = len(est_data)
        else:
            mean_est = std_est = np.nan
            n_est = 0

        smd = compute_smd(mean_est, mean_full, std_est, std_full, n_est, n_full)

        results.append(BalanceResult(
            variable=var,
            mean_full=mean_full,
            mean_estimation=mean_est,
            std_full=std_full,
            std_estimation=std_est,
            n_full=n_full,
            n_estimation=n_est,
            smd=smd,
        ))

    return results


def run_missingness_model(
    df: pd.DataFrame,
    in_sample_indicator: str,
    covariates: list[str],
    output_path: Path,
    label: str,
) -> None:
    """
    Run missingness regression to detect systematic patterns.

    Estimates a linear probability model where DV is an indicator for being
    in the estimation sample and IVs are observed covariates.
    """
    # Prepare data
    model_df = df.copy()

    # Filter to rows with at least some covariates observed
    available_covs = [c for c in covariates if c in model_df.columns]
    if not available_covs:
        with open(output_path, "w") as f:
            f.write(f"# Missingness Model: {label}\n\n")
            f.write("No covariates available for missingness analysis.\n")
        return

    # Create decade variable for FE if year is present
    if "year" in model_df.columns:
        model_df["decade"] = (model_df["year"] // 10) * 10
        use_decade_fe = True
    else:
        use_decade_fe = False

    # Build formula
    cov_terms = " + ".join(available_covs)
    if use_decade_fe:
        formula = f"{in_sample_indicator} ~ {cov_terms} + C(decade)"
    else:
        formula = f"{in_sample_indicator} ~ {cov_terms}"

    # Estimate model
    try:
        analysis_df = model_df.dropna(subset=available_covs + [in_sample_indicator])
        if len(analysis_df) < 50:
            with open(output_path, "w") as f:
                f.write(f"# Missingness Model: {label}\n\n")
                f.write(f"Insufficient observations for missingness model: {len(analysis_df)}\n")
            return

        model = smf.ols(formula, data=analysis_df).fit()

        # Write results
        with open(output_path, "w") as f:
            f.write(f"# Missingness Model: {label}\n")
            f.write("=" * 70 + "\n\n")
            f.write("## Interpretation\n\n")
            f.write("This model tests whether inclusion in the estimation sample is\n")
            f.write("systematically related to observed covariates. Significant coefficients\n")
            f.write("indicate potential selection bias that should be acknowledged.\n\n")
            f.write(f"DV: {in_sample_indicator} (1 = in estimation sample, 0 = excluded)\n\n")
            f.write("## Results\n\n")
            f.write(f"N = {int(model.nobs):,}\n")
            f.write(f"R-squared = {model.rsquared:.4f}\n\n")
            f.write("### Coefficients (covariates only, excluding decade FE):\n\n")
            f.write(f"{'Variable':<25} {'Coef':>12} {'SE':>12} {'p-value':>12}\n")
            f.write("-" * 65 + "\n")

            for var in available_covs:
                if var in model.params:
                    coef = model.params[var]
                    se = model.bse[var]
                    pval = model.pvalues[var]
                    sig = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.10 else ""
                    f.write(f"{var:<25} {coef:>12.4f} {se:>12.4f} {pval:>12.4f} {sig}\n")

            f.write("\n## Summary\n\n")
            sig_covs = [v for v in available_covs if v in model.pvalues and model.pvalues[v] < 0.10]
            if sig_covs:
                f.write(f"Variables significantly predicting sample inclusion: {', '.join(sig_covs)}\n")
                f.write("This suggests potential selection on observables.\n")
            else:
                f.write("No covariates significantly predict sample inclusion at p < 0.10.\n")
                f.write("Sample attrition appears largely unsystematic with respect to observables.\n")

    except Exception as e:
        with open(output_path, "w") as f:
            f.write(f"# Missingness Model: {label}\n\n")
            f.write(f"Model estimation failed: {str(e)}\n")


# =============================================================================
# H1 Audit Pipeline (Country-Year)
# =============================================================================


def audit_h1(paths: Paths) -> dict:
    """
    Run comprehensive attrition audit for H1 (ideology -> specialization).

    Documents sample flow from full country-year data through to each
    lag-specific estimation sample.

    Returns:
        Dictionary containing:
        - stages: List of StageResult objects
        - balance_by_lag: Dict mapping lag -> list of BalanceResult
        - summary_df: DataFrame summarizing attrition
    """
    print("\n  [AUDIT] H1: Country-Year Attrition Analysis")
    print("  " + "─" * 60)

    # Load full dataset
    df = pd.read_csv(paths.country_year_csv)
    key_cols = ["country_code_cow", "year"]
    unit_cols = ["country_code_cow"]

    stages = []
    stage_dfs = {}

    # Stage 1: Load full data
    stage_df, result = apply_stage(
        df=df,
        name="1. Load master_country_year.csv",
        prev_n=len(df),
        unit_cols=unit_cols,
    )
    stages.append(result)
    stage_dfs["full"] = stage_df.copy()
    print(f"    Stage 1: {result.n_rows:,} rows, {result.n_units} countries")

    # Stage 2: Analysis year range (from actual regression usage)
    prev_df = stage_df.copy()
    stage_df, result = apply_stage(
        df=stage_df,
        name="2. Analysis year range (1970-2014)",
        prev_n=len(prev_df),
        filter_fn=lambda d: d[(d["year"] >= 1970) & (d["year"] <= 2014)],
        unit_cols=unit_cols,
        key_cols=key_cols,
        prev_df=prev_df,
    )
    stages.append(result)
    print(f"    Stage 2: {result.n_rows:,} rows ({result.pct_remaining:.1%} remaining)")

    # Stage 3: Require DV (spec_y)
    prev_df = stage_df.copy()
    stage_df, result = apply_stage(
        df=stage_df,
        name="3. Require spec_y (DV)",
        prev_n=len(prev_df),
        required_cols=["spec_y"],
        unit_cols=unit_cols,
        key_cols=key_cols,
        prev_df=prev_df,
    )
    stages.append(result)
    print(f"    Stage 3: {result.n_rows:,} rows ({result.pct_remaining:.1%} remaining)")

    # Stage 4: Require ideology (right_of_center)
    prev_df = stage_df.copy()
    stage_df, result = apply_stage(
        df=stage_df,
        name="4. Require right_of_center (IV base)",
        prev_n=len(prev_df),
        required_cols=["right_of_center"],
        unit_cols=unit_cols,
        key_cols=key_cols,
        prev_df=prev_df,
    )
    stages.append(result)
    stage_dfs["ideology"] = stage_df.copy()
    print(f"    Stage 4: {result.n_rows:,} rows ({result.pct_remaining:.1%} remaining)")

    # Stage 5: Require country controls
    controls_required = [c for c in COUNTRY_CONTROLS if c in stage_df.columns]
    prev_df = stage_df.copy()
    stage_df, result = apply_stage(
        df=stage_df,
        name=f"5. Require controls ({', '.join(controls_required)})",
        prev_n=len(prev_df),
        required_cols=controls_required,
        unit_cols=unit_cols,
        key_cols=key_cols,
        prev_df=prev_df,
    )
    stages.append(result)
    stage_dfs["controls"] = stage_df.copy()
    print(f"    Stage 5: {result.n_rows:,} rows ({result.pct_remaining:.1%} remaining)")

    # Stage 6: FE feasibility
    prev_df = stage_df.copy()
    stage_df, result = apply_stage(
        df=stage_df,
        name="6. Require FE keys (country_code_cow, year)",
        prev_n=len(prev_df),
        required_cols=["country_code_cow", "year"],
        unit_cols=unit_cols,
        key_cols=key_cols,
        prev_df=prev_df,
    )
    stages.append(result)
    stage_dfs["fe_ready"] = stage_df.copy()
    print(f"    Stage 6: {result.n_rows:,} rows ({result.pct_remaining:.1%} remaining)")

    # Stages 7+: Each lag specification
    lag_results = {}
    balance_by_lag = {}

    # Create lagged variables
    base_df = stage_dfs["fe_ready"].copy()
    base_df = base_df.sort_values(["country_code_cow", "year"])
    for lag in range(1, 11):
        base_df[f"roc_lag{lag}"] = base_df.groupby("country_code_cow")["right_of_center"].shift(lag)

    for lag in range(1, 11):
        lag_var = f"roc_lag{lag}"
        prev_df = base_df.copy()
        lag_df, result = apply_stage(
            df=base_df,
            name=f"7.{lag}. Require {lag_var}",
            prev_n=len(prev_df),
            required_cols=[lag_var],
            unit_cols=unit_cols,
            key_cols=key_cols,
            prev_df=prev_df,
        )
        lag_results[lag] = result
        stage_dfs[f"lag{lag}"] = lag_df.copy()

        # Compute balance for this lag
        balance_vars = ["spec_y", "rile", "right_of_center", "lngdp", "cinc", "war5_lag"]
        if "in_alliance" in stage_dfs["full"].columns:
            balance_vars.append("in_alliance")
        balance_by_lag[lag] = compute_balance(stage_dfs["full"], lag_df, balance_vars)

    print(f"    Lag-specific samples: {lag_results[1].n_rows:,} (lag 1) to {lag_results[10].n_rows:,} (lag 10)")

    # Event study sample audit
    print("\n    Event study sample:")
    es_df = stage_dfs["fe_ready"].copy()
    es_df = es_df.sort_values(["country_code_cow", "year"])

    # Create transition indicators (using RILE = 0 threshold as in regressions.py)
    es_df["right_zero"] = (es_df["rile"] >= 0).astype(float)
    es_df["right_zero"] = es_df["right_zero"].where(es_df["rile"].notna())
    es_df["roc_prev"] = es_df.groupby("country_code_cow")["right_zero"].shift(1)
    es_df["transition"] = (es_df["right_zero"] != es_df["roc_prev"]) & es_df["roc_prev"].notna()

    n_transitions = int(es_df["transition"].sum())
    n_transition_countries = es_df[es_df["transition"]]["country_code_cow"].nunique()
    print(f"      Total transitions: {n_transitions} across {n_transition_countries} countries")

    # Store event study df for balance
    stage_dfs["event_study"] = es_df[es_df["rile"].notna()].copy()
    balance_by_lag["event_study"] = compute_balance(
        stage_dfs["full"],
        stage_dfs["event_study"],
        ["spec_y", "rile", "right_of_center", "lngdp", "cinc", "war5_lag"]
    )

    return {
        "stages": stages,
        "lag_results": lag_results,
        "balance_by_lag": balance_by_lag,
        "stage_dfs": stage_dfs,
    }


# =============================================================================
# H2 Audit Pipeline (Dyad-Year)
# =============================================================================


def audit_h2(paths: Paths) -> dict:
    """
    Run comprehensive attrition audit for H2 (alliance type -> division of labor).

    Documents sample flow from full dyad-year data through to estimation samples,
    with separate branches for minimal vs full control specifications.
    """
    print("\n  [AUDIT] H2: Dyad-Year Attrition Analysis")
    print("  " + "─" * 60)

    # Load full dataset
    df = pd.read_csv(paths.dyad_year_csv)

    # Create dyad_id if not present
    if "dyad_id" not in df.columns:
        df["dyad_id"] = df["state_a"].astype(str) + "_" + df["state_b"].astype(str)

    # After dyad-year collapse, the key is (state_a, state_b, year), not (atopid, state_a, state_b, year)
    key_cols = ["state_a", "state_b", "year"]
    unit_cols = ["dyad_id"]

    stages = []
    stage_dfs = {}

    # Stage 1: Load full data
    stage_df, result = apply_stage(
        df=df,
        name="1. Load master_dyad_year.csv",
        prev_n=len(df),
        unit_cols=unit_cols,
    )
    stages.append(result)
    stage_dfs["full"] = stage_df.copy()
    print(f"    Stage 1: {result.n_rows:,} rows, {result.n_units:,} dyads")

    # Stage 2: Require DV (div_labor)
    prev_df = stage_df.copy()
    stage_df, result = apply_stage(
        df=stage_df,
        name="2. Require div_labor (DV)",
        prev_n=len(prev_df),
        required_cols=["div_labor"],
        unit_cols=unit_cols,
        key_cols=key_cols,
        prev_df=prev_df,
    )
    stages.append(result)
    stage_dfs["dv"] = stage_df.copy()
    print(f"    Stage 2: {result.n_rows:,} rows ({result.pct_remaining:.1%} remaining)")

    # Stage 3: Require IVs (hierarchical, voice_driven, inst)
    prev_df = stage_df.copy()
    stage_df, result = apply_stage(
        df=stage_df,
        name="3. Require IVs (hierarchical, voice_driven, inst)",
        prev_n=len(prev_df),
        required_cols=["hierarchical", "voice_driven", "inst"],
        unit_cols=unit_cols,
        key_cols=key_cols,
        prev_df=prev_df,
    )
    stages.append(result)
    stage_dfs["ivs"] = stage_df.copy()
    print(f"    Stage 3: {result.n_rows:,} rows ({result.pct_remaining:.1%} remaining)")

    # Stage 4: Require FE keys (dyad_id, decade)
    # Create decade if needed
    stage_df["decade"] = (stage_df["year"] // 10) * 10
    prev_df = stage_df.copy()
    stage_df, result = apply_stage(
        df=stage_df,
        name="4. Require FE keys (dyad_id, decade)",
        prev_n=len(prev_df),
        required_cols=["dyad_id", "decade"],
        unit_cols=unit_cols,
        key_cols=key_cols,
        prev_df=prev_df,
    )
    stages.append(result)
    stage_dfs["fe_ready"] = stage_df.copy()
    print(f"    Stage 4: {result.n_rows:,} rows ({result.pct_remaining:.1%} remaining)")

    # Stage 5a: Minimal controls (contiguous only)
    prev_df = stage_df.copy()
    stage_df_minimal, result_minimal = apply_stage(
        df=stage_df,
        name="5a. Minimal controls (contiguous)",
        prev_n=len(prev_df),
        required_cols=["contiguous"],
        unit_cols=unit_cols,
        key_cols=key_cols,
        prev_df=prev_df,
    )
    stage_dfs["minimal_controls"] = stage_df_minimal.copy()
    print(f"    Stage 5a (minimal): {result_minimal.n_rows:,} rows ({result_minimal.pct_remaining:.1%} remaining)")

    # Stage 5b: Full controls (contiguous + gdp_ratio + cinc_ratio)
    full_control_cols = ["contiguous"]
    if "gdp_ratio" in stage_df.columns:
        full_control_cols.append("gdp_ratio")
    if "cinc_ratio" in stage_df.columns:
        full_control_cols.append("cinc_ratio")

    stage_df_full, result_full = apply_stage(
        df=stage_df,
        name=f"5b. Full controls ({', '.join(full_control_cols)})",
        prev_n=len(prev_df),
        required_cols=full_control_cols,
        unit_cols=unit_cols,
        key_cols=key_cols,
        prev_df=prev_df,
    )
    stage_dfs["full_controls"] = stage_df_full.copy()
    print(f"    Stage 5b (full): {result_full.n_rows:,} rows ({result_full.pct_remaining:.1%} remaining)")

    # Compute control-specific attrition
    branches = {
        "minimal": result_minimal,
        "full": result_full,
    }

    # Detailed breakdown of control missingness
    print("\n    Control variable coverage (from Stage 4):")
    for col in ["contiguous", "gdp_ratio", "cinc_ratio"]:
        if col in stage_dfs["fe_ready"].columns:
            n_present = stage_dfs["fe_ready"][col].notna().sum()
            pct = n_present / len(stage_dfs["fe_ready"])
            print(f"      {col}: {n_present:,} ({pct:.1%})")

    # Balance comparison
    balance_vars = ["div_labor", "hierarchical", "voice_driven", "contiguous"]
    if "gdp_ratio" in df.columns:
        balance_vars.append("gdp_ratio")
    if "cinc_ratio" in df.columns:
        balance_vars.append("cinc_ratio")

    balance_minimal = compute_balance(stage_dfs["full"], stage_dfs["minimal_controls"], balance_vars)
    balance_full = compute_balance(stage_dfs["full"], stage_dfs["full_controls"], balance_vars)

    # Event study sample audit
    print("\n    Event study sample:")

    # Load full div_labor data for event study context
    div_labor_full = pd.read_csv(paths.div_labor_csv)

    # Normalize ordering
    div_labor_full["s_min"] = div_labor_full[["state_a", "state_b"]].min(axis=1)
    div_labor_full["s_max"] = div_labor_full[["state_a", "state_b"]].max(axis=1)
    div_labor_full["state_a"] = div_labor_full["s_min"]
    div_labor_full["state_b"] = div_labor_full["s_max"]
    div_labor_full = div_labor_full.drop(columns=["s_min", "s_max"])

    # Get first alliance entry for each dyad
    dyad_df = stage_dfs["full"].copy()
    first_alliance = dyad_df.groupby(["state_a", "state_b"])["year"].min().reset_index()
    first_alliance.columns = ["state_a", "state_b", "alliance_entry_year"]

    n_entering_dyads = len(first_alliance)
    print(f"      Dyads entering alliances: {n_entering_dyads:,}")

    # Merge and create event window
    es_panel = div_labor_full.merge(first_alliance, on=["state_a", "state_b"], how="inner")
    es_panel["event_time"] = es_panel["year"] - es_panel["alliance_entry_year"]
    es_panel_window = es_panel[es_panel["event_time"].between(-5, 5)].copy()

    n_es_rows = len(es_panel_window)
    n_es_dyads = es_panel_window.groupby(["state_a", "state_b"]).ngroups
    print(f"      Event study sample (±5 window): {n_es_rows:,} rows, {n_es_dyads:,} dyads")

    stage_dfs["event_study"] = es_panel_window
    balance_event = compute_balance(stage_dfs["full"], es_panel_window, balance_vars)

    return {
        "stages": stages,
        "branches": branches,
        "balance_minimal": balance_minimal,
        "balance_full": balance_full,
        "balance_event": balance_event,
        "stage_dfs": stage_dfs,
    }


# =============================================================================
# Output Generation
# =============================================================================


def _box(text: str, width: int = 70) -> str:
    """Create a boxed header."""
    lines = [
        "╔" + "═" * (width - 2) + "╗",
        "║" + text.center(width - 2) + "║",
        "╚" + "═" * (width - 2) + "╝",
    ]
    return "\n".join(lines)


def save_h1_audit(audit_result: dict, paths: Paths) -> None:
    """Save H1 audit results to files."""
    audit_dir = paths.h1_dir.parent / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)

    # Overview CSV
    overview_data = [s.to_dict() for s in audit_result["stages"]]
    pd.DataFrame(overview_data).to_csv(audit_dir / "h1_attrition_overview.csv", index=False)

    # By-lag CSV
    lag_data = []
    for lag, result in audit_result["lag_results"].items():
        row = result.to_dict()
        row["lag"] = lag
        lag_data.append(row)
    pd.DataFrame(lag_data).to_csv(audit_dir / "h1_attrition_by_lag.csv", index=False)

    # Balance by lag CSV
    balance_rows = []
    for lag, balance_list in audit_result["balance_by_lag"].items():
        for b in balance_list:
            row = b.to_dict()
            row["lag"] = lag
            balance_rows.append(row)
    pd.DataFrame(balance_rows).to_csv(audit_dir / "h1_balance_by_lag.csv", index=False)

    # Missingness model
    full_df = audit_result["stage_dfs"]["full"].copy()

    # Create in-sample indicator for lag 3 (representative)
    if "lag3" in audit_result["stage_dfs"]:
        est_keys = set(
            audit_result["stage_dfs"]["lag3"][["country_code_cow", "year"]].apply(tuple, axis=1)
        )
        full_df["in_sample"] = full_df[["country_code_cow", "year"]].apply(tuple, axis=1).isin(est_keys).astype(int)

        covariates = ["spec_y", "rile", "lngdp", "cinc"]
        run_missingness_model(
            full_df,
            "in_sample",
            covariates,
            audit_dir / "h1_missingness_model.txt",
            "H1 (lag 3 specification)",
        )

    print(f"    Saved H1 audit to: {audit_dir}/")


def save_h2_audit(audit_result: dict, paths: Paths) -> None:
    """Save H2 audit results to files."""
    audit_dir = paths.h2_dir.parent / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)

    # Overview CSV
    overview_data = [s.to_dict() for s in audit_result["stages"]]
    pd.DataFrame(overview_data).to_csv(audit_dir / "h2_attrition_overview.csv", index=False)

    # Branches CSV
    branches_data = []
    for branch_name, result in audit_result["branches"].items():
        row = result.to_dict()
        row["branch"] = branch_name
        branches_data.append(row)
    pd.DataFrame(branches_data).to_csv(audit_dir / "h2_attrition_branches.csv", index=False)

    # Balance CSV (combine all)
    balance_rows = []
    for sample_name, balance_list in [
        ("minimal_controls", audit_result["balance_minimal"]),
        ("full_controls", audit_result["balance_full"]),
        ("event_study", audit_result["balance_event"]),
    ]:
        for b in balance_list:
            row = b.to_dict()
            row["sample"] = sample_name
            balance_rows.append(row)
    pd.DataFrame(balance_rows).to_csv(audit_dir / "h2_balance.csv", index=False)

    # Missingness model
    full_df = audit_result["stage_dfs"]["full"].copy()

    if "full_controls" in audit_result["stage_dfs"]:
        est_df = audit_result["stage_dfs"]["full_controls"]
        est_keys = set(
            est_df[["state_a", "state_b", "year"]].apply(tuple, axis=1)
        )
        full_df["in_sample"] = full_df[["state_a", "state_b", "year"]].apply(tuple, axis=1).isin(est_keys).astype(int)

        covariates = ["div_labor", "hierarchical", "voice_driven", "contiguous"]
        run_missingness_model(
            full_df,
            "in_sample",
            covariates,
            audit_dir / "h2_missingness_model.txt",
            "H2 (full controls specification)",
        )

    print(f"    Saved H2 audit to: {audit_dir}/")


def generate_audit_summary(h1_result: dict, h2_result: dict, paths: Paths) -> None:
    """Generate human-readable audit summary."""
    audit_dir = paths.h1_dir.parent / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)

    with open(audit_dir / "audit_summary.md", "w") as f:
        f.write("# Attrition and Missingness Audit Summary\n\n")
        f.write("This document summarizes sample attrition and potential selection issues\n")
        f.write("for the Military Specialization & Alliance Institutions Study.\n\n")

        # H1 Summary
        f.write("## H1: Ideology → Specialization (Country-Year)\n\n")
        f.write("### Sample Flow\n\n")
        f.write("| Stage | N Rows | N Countries | % Remaining |\n")
        f.write("|-------|--------|-------------|-------------|\n")

        for stage in h1_result["stages"]:
            f.write(f"| {stage.stage_name} | {stage.n_rows:,} | {stage.n_units} | {stage.pct_remaining*100:.1f}% |\n")

        f.write("\n### Lag-Specific Samples\n\n")
        f.write("| Lag | N Rows | N Countries | Year Range |\n")
        f.write("|-----|--------|-------------|------------|\n")

        for lag in [1, 3, 5, 10]:
            if lag in h1_result["lag_results"]:
                r = h1_result["lag_results"][lag]
                f.write(f"| {lag} | {r.n_rows:,} | {r.n_units} | {r.year_min}-{r.year_max} |\n")

        f.write("\n### Balance Check (Lag 3 vs Full Sample)\n\n")
        if 3 in h1_result["balance_by_lag"]:
            f.write("| Variable | Mean (Full) | Mean (Est) | SMD |\n")
            f.write("|----------|-------------|------------|-----|\n")
            for b in h1_result["balance_by_lag"][3]:
                smd_str = f"{b.smd:.3f}" if not np.isnan(b.smd) else "N/A"
                f.write(f"| {b.variable} | {b.mean_full:.3f} | {b.mean_estimation:.3f} | {smd_str} |\n")

        f.write("\n**Interpretation**: SMD > 0.1 suggests meaningful difference; SMD > 0.25 is concerning.\n\n")

        # H2 Summary
        f.write("## H2: Alliance Type → Division of Labor (Dyad-Year)\n\n")
        f.write("### Sample Flow\n\n")
        f.write("| Stage | N Rows | N Dyads | % Remaining |\n")
        f.write("|-------|--------|---------|-------------|\n")

        for stage in h2_result["stages"]:
            f.write(f"| {stage.stage_name} | {stage.n_rows:,} | {stage.n_units:,} | {stage.pct_remaining*100:.1f}% |\n")

        f.write("\n### Control Specification Comparison\n\n")
        f.write("| Specification | N Rows | N Dyads | % of FE-Ready |\n")
        f.write("|---------------|--------|---------|---------------|\n")

        fe_ready_n = h2_result["stages"][-1].n_rows  # Last main stage
        for branch_name, result in h2_result["branches"].items():
            pct = result.n_rows / fe_ready_n * 100 if fe_ready_n > 0 else 0
            f.write(f"| {branch_name} | {result.n_rows:,} | {result.n_units:,} | {pct:.1f}% |\n")

        f.write("\n### Balance Check (Full Controls vs Full Sample)\n\n")
        f.write("| Variable | Mean (Full) | Mean (Est) | SMD |\n")
        f.write("|----------|-------------|------------|-----|\n")
        for b in h2_result["balance_full"]:
            smd_str = f"{b.smd:.3f}" if not np.isnan(b.smd) else "N/A"
            f.write(f"| {b.variable} | {b.mean_full:.3f} | {b.mean_estimation:.3f} | {smd_str} |\n")

        f.write("\n## Key Findings\n\n")

        # H1 key finding
        h1_full_n = h1_result["stages"][0].n_rows
        h1_est_n = h1_result["lag_results"][3].n_rows if 3 in h1_result["lag_results"] else 0
        h1_pct_retained = h1_est_n / h1_full_n * 100 if h1_full_n > 0 else 0

        f.write(f"1. **H1 Sample Retention**: {h1_pct_retained:.1f}% of country-years make it to estimation (lag 3)\n")
        f.write("   - Primary drivers of attrition: ideology data coverage, control variable availability\n\n")

        # H2 key finding
        h2_full_n = h2_result["stages"][0].n_rows
        h2_est_n = h2_result["branches"]["full"].n_rows
        h2_pct_retained = h2_est_n / h2_full_n * 100 if h2_full_n > 0 else 0

        h2_minimal_n = h2_result["branches"]["minimal"].n_rows
        gdp_cinc_cost = h2_minimal_n - h2_est_n
        gdp_cinc_pct = gdp_cinc_cost / h2_minimal_n * 100 if h2_minimal_n > 0 else 0

        f.write(f"2. **H2 Sample Retention**: {h2_pct_retained:.1f}% of dyad-years make it to estimation (full controls)\n")
        f.write(f"   - GDP/CINC ratio requirements drop {gdp_cinc_cost:,} rows ({gdp_cinc_pct:.1f}%)\n")
        f.write("   - Minimal controls specification retains more data if GDP/CINC are not critical\n\n")

        f.write("## Files Generated\n\n")
        f.write("- `h1_attrition_overview.csv`: Stage-by-stage H1 attrition\n")
        f.write("- `h1_attrition_by_lag.csv`: Lag-specific sample sizes\n")
        f.write("- `h1_balance_by_lag.csv`: Balance statistics by lag\n")
        f.write("- `h1_missingness_model.txt`: Missingness regression results\n")
        f.write("- `h2_attrition_overview.csv`: Stage-by-stage H2 attrition\n")
        f.write("- `h2_attrition_branches.csv`: Minimal vs full control samples\n")
        f.write("- `h2_balance.csv`: Balance statistics for H2 samples\n")
        f.write("- `h2_missingness_model.txt`: Missingness regression results\n")

    print(f"    Saved summary to: {audit_dir / 'audit_summary.md'}")


# =============================================================================
# Main Entry Point
# =============================================================================


def run_full_audit(paths: Paths) -> None:
    """
    Run complete attrition and missingness audit for both H1 and H2.

    Produces comprehensive documentation of sample flow, balance checks,
    and missingness analysis for transparent reporting.
    """
    print()
    print(_box("ATTRITION & MISSINGNESS AUDIT"))

    print("\n  Running comprehensive sample audit...")
    print("  This documents exactly which observations enter each regression and why.\n")

    # Run H1 audit
    h1_result = audit_h1(paths)
    save_h1_audit(h1_result, paths)

    # Run H2 audit
    h2_result = audit_h2(paths)
    save_h2_audit(h2_result, paths)

    # Generate summary
    generate_audit_summary(h1_result, h2_result, paths)

    print()
    print(_box("AUDIT COMPLETE"))
    print()


if __name__ == "__main__":
    paths = Paths()
    run_full_audit(paths)
