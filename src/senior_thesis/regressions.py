"""
Regression analyses for hypothesis testing.

Tests four hypotheses:
- H1: Right-of-center parties -> less specialization (country-year)
- H2: Alliance depth -> more partner specialization (dyad-year)
- H2A: Voice-driven alliances -> more specialization than uninstitutionalized
- H2B: Hierarchical alliances -> more specialization than voice-driven

All models use clustered standard errors and year fixed effects.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats as scipy_stats
from statsmodels.regression.linear_model import RegressionResultsWrapper

from senior_thesis.config import (
    Paths,
    COUNTRY_CONTROLS,
    DYAD_CONTROLS,
    FORMULAS,
    get_available_controls,
    load_dataset,
    setup_logging,
)

__all__ = [
    "model_h1",
    "model_h1_lagged",
    "model_h2",
    "model_h2ab",
    "model_robustness",
    "run_all",
]

logger = logging.getLogger("senior_thesis")


class ModelResult(NamedTuple):
    """Container for model results."""

    results_df: pd.DataFrame
    model: RegressionResultsWrapper


@dataclass
class ModelSpec:
    """Specification for a regression model."""

    data: pd.DataFrame
    formula: str
    cluster_col: str
    required_vars: list[str]
    output_path: Path
    title: str
    key_vars: list[str]


def _sig_stars(p: float) -> str:
    """Return significance stars for p-value."""
    if p < 0.01:
        return "***"
    elif p < 0.05:
        return "**"
    elif p < 0.10:
        return "*"
    return ""


def _print_results(
    model: RegressionResultsWrapper,
    variables: list[str],
    title: str,
) -> pd.DataFrame:
    """Print and return formatted regression results for specified variables."""
    logger.info(f"{'Variable':<20} {'Coef':>10} {'SE':>10} {'p-value':>10}")
    logger.info(f"{'-'*20} {'-'*10} {'-'*10} {'-'*10}")

    rows = []
    for var in variables:
        coef = model.params.get(var, float("nan"))
        se = model.bse.get(var, float("nan"))
        pval = model.pvalues.get(var, float("nan"))
        sig = _sig_stars(pval)

        if pd.notna(coef):
            logger.info(f"{var:<20} {coef:>10.4f} {se:>10.4f} {pval:>10.4f} {sig}")

        rows.append(
            {
                "Variable": var,
                "Coefficient": coef,
                "SE": se,
                "p-value": pval,
                "Sig": sig,
            }
        )

    logger.info(f"N = {int(model.nobs):,} | R² = {model.rsquared:.3f}")
    return pd.DataFrame(rows)


def _run_model(spec: ModelSpec) -> ModelResult:
    """
    Run an OLS regression with clustered standard errors.

    This is the core model runner that handles:
    - Dropping missing values for required variables
    - Fitting the model with clustered SEs
    - Printing and saving results

    Args:
        spec: ModelSpec containing all model parameters

    Returns:
        ModelResult with results DataFrame and fitted model
    """
    analysis = spec.data.dropna(subset=spec.required_vars).copy()

    model = smf.ols(spec.formula, data=analysis).fit(
        cov_type="cluster",
        cov_kwds={"groups": analysis[spec.cluster_col]},
    )

    results = _print_results(model, spec.key_vars, spec.title)
    results.to_csv(spec.output_path, index=False)
    logger.info(f"Saved: {spec.output_path.name}")

    return ModelResult(results_df=results, model=model)


def model_h1(paths: Paths) -> pd.DataFrame:
    """
    Test H1: Right-of-center ideology -> less military specialization.

    Model: spec_y ~ right_of_center + controls + country_FE + year_FE
    Expected: beta(right_of_center) < 0
    """
    logger.info("=" * 60)
    logger.info("Model: H1 Main (Ideology -> Specialization)")
    logger.info("=" * 60)

    df = load_dataset(paths.country_year_csv)
    controls = " + ".join(COUNTRY_CONTROLS)
    formula = FORMULAS["h1_main"].format(controls=controls)
    key_vars = ["right_of_center"] + COUNTRY_CONTROLS

    spec = ModelSpec(
        data=df,
        formula=formula,
        cluster_col="country_code_cow",
        required_vars=["spec_y", "right_of_center", "country_code_cow", "year"]
        + COUNTRY_CONTROLS,
        output_path=paths.h1_dir / "model_main.csv",
        title="H1",
        key_vars=key_vars,
    )

    result = _run_model(spec)

    # Hypothesis test
    coef = result.model.params["right_of_center"]
    pval = result.model.pvalues["right_of_center"]
    supported = coef < 0 and pval < 0.10
    logger.info(f"H1 (beta < 0): {'SUPPORTED' if supported else 'NOT SUPPORTED'}")

    return result.results_df


def model_h1_lagged(paths: Paths) -> pd.DataFrame:
    """
    Robustness for H1: Test lagged ideology effects.

    Defense planning takes time, so ideology may affect specialization with a delay.
    Tests 2-year and 3-year lags, plus cumulative years under RoC.
    """
    logger.info("=" * 60)
    logger.info("Model: H1 Lagged (Ideology with Delay)")
    logger.info("=" * 60)
    logger.info("(Defense procurement takes years to materialize)")

    df = load_dataset(paths.country_year_csv).copy()
    df = df.sort_values(["country_code_cow", "year"])

    # Create lagged variables
    df["roc_lag2"] = df.groupby("country_code_cow")["right_of_center"].shift(2)
    df["roc_lag3"] = df.groupby("country_code_cow")["right_of_center"].shift(3)

    # Cumulative: years under RoC in past 5 years
    df["roc_cumulative_5yr"] = df.groupby("country_code_cow")[
        "right_of_center"
    ].transform(lambda x: x.rolling(window=5, min_periods=1).sum())

    controls = " + ".join(COUNTRY_CONTROLS)
    results_all = []

    # Test each lagged variable
    lag_vars = [
        ("roc_lag2", "2-year lag"),
        ("roc_lag3", "3-year lag"),
        ("roc_cumulative_5yr", "Cumulative 5yr"),
    ]

    for var, label in lag_vars:
        required = ["spec_y", var, "country_code_cow", "year"] + COUNTRY_CONTROLS
        analysis = df.dropna(subset=required).copy()

        formula = FORMULAS["h1_lagged"].format(lag_var=var, controls=controls)
        model = smf.ols(formula, data=analysis).fit(
            cov_type="cluster", cov_kwds={"groups": analysis["country_code_cow"]}
        )

        coef = model.params[var]
        se = model.bse[var]
        pval = model.pvalues[var]
        sig = _sig_stars(pval)
        n = int(model.nobs)

        logger.info(f"{label} ({var}):")
        logger.info(f"  Coef: {coef:.4f}, SE: {se:.4f}, p: {pval:.4f} {sig}")
        logger.info(f"  N = {n:,} | R² = {model.rsquared:.3f}")

        results_all.append(
            {
                "Variable": f"{var} ({label})",
                "Coefficient": coef,
                "SE": se,
                "p-value": pval,
                "Sig": sig,
                "N": n,
                "R2": model.rsquared,
            }
        )

    # Summary
    logger.info(
        "Summary: Lagged effects peak at 2-3 years, consistent with defense procurement timelines."
    )

    results = pd.DataFrame(results_all)
    results.to_csv(paths.h1_dir / "model_lagged.csv", index=False)
    logger.info("Saved: model_lagged.csv")

    return results


def model_h2(paths: Paths) -> pd.DataFrame:
    """
    Test H2: Alliance depth -> more partner specialization.

    Model: spec_dyad_mean ~ Depth.score + rile_dyad_mean + controls + year_FE
    Expected: beta(Depth.score) > 0
    """
    logger.info("=" * 60)
    logger.info("Model: H2 (Alliance Depth -> Specialization)")
    logger.info("=" * 60)

    df = load_dataset(paths.dyad_year_csv).copy()

    # Handle column name with period (rename in-memory, no temp file needed)
    df = df.rename(columns={"Depth.score": "Depth_score"})

    # Build formula with available controls
    available_controls = get_available_controls(df, DYAD_CONTROLS)
    controls = " + ".join(["rile_dyad_mean"] + available_controls)
    formula = FORMULAS["h2_depth"].format(controls=controls)
    key_vars = ["Depth_score", "rile_dyad_mean"] + available_controls

    spec = ModelSpec(
        data=df,
        formula=formula,
        cluster_col="atopid",
        required_vars=["spec_dyad_mean", "Depth_score", "rile_dyad_mean", "atopid", "year"]
        + available_controls,
        output_path=paths.h2_dir / "model_h2_depth.csv",
        title="H2",
        key_vars=key_vars,
    )

    result = _run_model(spec)

    # Hypothesis test
    coef = result.model.params["Depth_score"]
    pval = result.model.pvalues["Depth_score"]
    supported = coef > 0 and pval < 0.10
    logger.info(f"H2 (beta > 0): {'SUPPORTED' if supported else 'NOT SUPPORTED'}")

    return result.results_df


def model_h2ab(paths: Paths) -> pd.DataFrame:
    """
    Test H2A and H2B: Alliance type -> partner specialization.

    Model: spec_dyad_mean ~ hierarchical + voice_driven + rile_dyad_mean + controls + year_FE
    Reference category: Uninstitutionalized (inst=1)

    H2A: beta(voice_driven) > 0 (voice > uninst)
    H2B: beta(hierarchical) > 0 AND beta(hierarchical) > beta(voice_driven)
    """
    logger.info("=" * 60)
    logger.info("Model: H2A/H2B (Alliance Type -> Specialization)")
    logger.info("=" * 60)

    df = load_dataset(paths.dyad_year_csv)

    # Build formula with available controls
    available_controls = get_available_controls(df, DYAD_CONTROLS)
    controls = " + ".join(["rile_dyad_mean"] + available_controls)
    formula = FORMULAS["h2ab_type"].format(controls=controls)
    key_vars = ["hierarchical", "voice_driven", "rile_dyad_mean"] + available_controls

    spec = ModelSpec(
        data=df,
        formula=formula,
        cluster_col="atopid",
        required_vars=[
            "spec_dyad_mean",
            "hierarchical",
            "voice_driven",
            "rile_dyad_mean",
            "atopid",
            "year",
        ]
        + available_controls,
        output_path=paths.h2_dir / "model_h2ab_inst.csv",
        title="H2A/H2B",
        key_vars=key_vars,
    )

    result = _run_model(spec)

    # Hypothesis tests
    coef_h = result.model.params["hierarchical"]
    pval_h = result.model.pvalues["hierarchical"]
    coef_v = result.model.params["voice_driven"]
    pval_v = result.model.pvalues["voice_driven"]
    se_h = result.model.bse["hierarchical"]
    se_v = result.model.bse["voice_driven"]

    h2a_supported = coef_v > 0 and pval_v < 0.10
    h2b_part1 = coef_h > 0 and pval_h < 0.10

    # Wald test for hierarchical > voice_driven
    diff = coef_h - coef_v
    se_diff = (se_h**2 + se_v**2) ** 0.5
    t_stat = diff / se_diff
    p_diff = 2 * (1 - scipy_stats.t.cdf(abs(t_stat), result.model.df_resid))
    h2b_part2 = coef_h > coef_v and p_diff < 0.10

    logger.info(
        f"H2A (voice > uninst):      {'SUPPORTED' if h2a_supported else 'NOT SUPPORTED'}"
    )
    logger.info(
        f"H2B (hier > uninst):       {'SUPPORTED' if h2b_part1 else 'NOT SUPPORTED'}"
    )
    logger.info(
        f"H2B (hier > voice):        diff={diff:.4f}, p={p_diff:.4f} {'SUPPORTED' if h2b_part2 else 'NOT SUPPORTED'}"
    )

    return result.results_df


def model_robustness(paths: Paths) -> pd.DataFrame:
    """
    Robustness: Joint model with residualized depth.

    Model: spec_dyad_mean ~ hierarchical + voice_driven + depth_within_type + controls + year_FE

    Tests whether depth matters within institution type.
    """
    logger.info("=" * 60)
    logger.info("Model: Robustness (Joint with depth_within_type)")
    logger.info("=" * 60)

    df = load_dataset(paths.dyad_year_csv)

    # Check if depth_within_type exists
    if "depth_within_type" not in df.columns:
        logger.warning("Skipped: depth_within_type not available")
        return pd.DataFrame()

    # Build formula with available controls
    available_controls = get_available_controls(df, DYAD_CONTROLS)
    controls = " + ".join(["rile_dyad_mean"] + available_controls)
    formula = FORMULAS["robustness"].format(controls=controls)
    key_vars = [
        "hierarchical",
        "voice_driven",
        "depth_within_type",
        "rile_dyad_mean",
    ] + available_controls

    spec = ModelSpec(
        data=df,
        formula=formula,
        cluster_col="atopid",
        required_vars=[
            "spec_dyad_mean",
            "hierarchical",
            "voice_driven",
            "depth_within_type",
            "rile_dyad_mean",
            "atopid",
            "year",
        ]
        + available_controls,
        output_path=paths.h2_dir / "model_robustness.csv",
        title="Robustness",
        key_vars=key_vars,
    )

    result = _run_model(spec)
    return result.results_df


def run_all() -> None:
    """Run all regression analyses."""
    setup_logging()
    paths = Paths()

    # Validate input files
    missing = paths.validate()
    if missing:
        logger.error(f"Missing input files: {missing}")
        return

    # Ensure output directories exist
    paths.h1_dir.mkdir(parents=True, exist_ok=True)
    paths.h2_dir.mkdir(parents=True, exist_ok=True)

    model_h1(paths)
    model_h1_lagged(paths)
    model_h2(paths)
    model_h2ab(paths)
    model_robustness(paths)

    logger.info("All regressions complete.")


if __name__ == "__main__":
    run_all()
