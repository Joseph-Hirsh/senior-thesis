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

import pandas as pd
import statsmodels.formula.api as smf

from senior_thesis.config import Paths, COUNTRY_CONTROLS, DYAD_CONTROLS


def _sig_stars(p: float) -> str:
    """Return significance stars for p-value."""
    if p < 0.01:
        return "***"
    elif p < 0.05:
        return "**"
    elif p < 0.10:
        return "*"
    return ""


def _print_results(model, variables: list[str], title: str) -> pd.DataFrame:
    """Print and return formatted regression results for all variables."""
    print(f"\n  {'Variable':<20} {'Coef':>10} {'SE':>10} {'p-value':>10}")
    print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*10}")

    rows = []
    for var in variables:
        coef = model.params.get(var, float("nan"))
        se = model.bse.get(var, float("nan"))
        pval = model.pvalues.get(var, float("nan"))
        sig = _sig_stars(pval)

        if pd.notna(coef):
            print(f"  {var:<20} {coef:>10.4f} {se:>10.4f} {pval:>10.4f} {sig}")

        rows.append({
            "Variable": var,
            "Coefficient": coef,
            "SE": se,
            "p-value": pval,
            "Sig": sig,
        })

    print(f"\n  N = {int(model.nobs):,} | R² = {model.rsquared:.3f}")
    return pd.DataFrame(rows)


def model_h1(paths: Paths) -> pd.DataFrame:
    """
    Test H1: Right-of-center ideology -> less military specialization.

    Model: spec_y ~ right_of_center + controls + country_FE + year_FE
    Expected: beta(right_of_center) < 0
    """
    print("\n" + "="*60)
    print(" Model 1: H1 (Ideology -> Specialization)")
    print("="*60)

    df = pd.read_csv(paths.country_year_csv)

    # Required variables
    required = ["spec_y", "right_of_center", "country_code_cow", "year"] + COUNTRY_CONTROLS
    analysis = df.dropna(subset=required).copy()

    # Build formula
    controls = " + ".join(COUNTRY_CONTROLS)
    formula = f"spec_y ~ right_of_center + {controls} + C(country_code_cow) + C(year)"

    # Fit with clustered SEs
    model = smf.ols(formula, data=analysis).fit(
        cov_type="cluster", cov_kwds={"groups": analysis["country_code_cow"]}
    )

    # Print all results
    all_vars = ["right_of_center"] + COUNTRY_CONTROLS
    results = _print_results(model, all_vars, "H1")

    # Hypothesis test
    coef = model.params["right_of_center"]
    pval = model.pvalues["right_of_center"]
    supported = coef < 0 and pval < 0.10
    print(f"\n  H1 (β < 0): {'SUPPORTED' if supported else 'NOT SUPPORTED'}")

    results.to_csv(paths.tables_dir / "model1_h1.csv", index=False)
    print(f"  Saved: model1_h1.csv")

    return results


def model_h2(paths: Paths) -> pd.DataFrame:
    """
    Test H2: Alliance depth -> more partner specialization.

    Model: spec_dyad_mean ~ Depth.score + rile_dyad_mean + controls + year_FE
    Expected: beta(Depth.score) > 0
    """
    print("\n" + "="*60)
    print(" Model 2: H2 (Alliance Depth -> Specialization)")
    print("="*60)

    df = pd.read_csv(paths.dyad_year_csv)

    # Handle column name with period
    df = df.rename(columns={"Depth.score": "Depth_score"})

    # Required variables (subset of DYAD_CONTROLS that exist)
    available_controls = [c for c in DYAD_CONTROLS if c in df.columns]
    required = ["spec_dyad_mean", "Depth_score", "rile_dyad_mean", "atopid", "year"] + available_controls
    analysis = df.dropna(subset=required).copy()

    # Build formula
    controls = " + ".join(["rile_dyad_mean"] + available_controls)
    formula = f"spec_dyad_mean ~ Depth_score + {controls} + C(year)"

    # Fit with clustered SEs
    model = smf.ols(formula, data=analysis).fit(
        cov_type="cluster", cov_kwds={"groups": analysis["atopid"]}
    )

    # Print all results
    all_vars = ["Depth_score", "rile_dyad_mean"] + available_controls
    results = _print_results(model, all_vars, "H2")

    # Hypothesis test
    coef = model.params["Depth_score"]
    pval = model.pvalues["Depth_score"]
    supported = coef > 0 and pval < 0.10
    print(f"\n  H2 (β > 0): {'SUPPORTED' if supported else 'NOT SUPPORTED'}")

    results.to_csv(paths.tables_dir / "model2_h2.csv", index=False)
    print(f"  Saved: model2_h2.csv")

    return results


def model_h2ab(paths: Paths) -> pd.DataFrame:
    """
    Test H2A and H2B: Alliance type -> partner specialization.

    Model: spec_dyad_mean ~ hierarchical + voice_driven + rile_dyad_mean + controls + year_FE
    Reference category: Uninstitutionalized (inst=1)

    H2A: beta(voice_driven) > 0 (voice > uninst)
    H2B: beta(hierarchical) > 0 AND beta(hierarchical) > beta(voice_driven)
    """
    print("\n" + "="*60)
    print(" Model 3: H2A/H2B (Alliance Type -> Specialization)")
    print("="*60)

    df = pd.read_csv(paths.dyad_year_csv)

    # Required variables
    available_controls = [c for c in DYAD_CONTROLS if c in df.columns]
    required = ["spec_dyad_mean", "hierarchical", "voice_driven", "rile_dyad_mean", "atopid", "year"] + available_controls
    analysis = df.dropna(subset=required).copy()

    # Build formula
    controls = " + ".join(["rile_dyad_mean"] + available_controls)
    formula = f"spec_dyad_mean ~ hierarchical + voice_driven + {controls} + C(year)"

    # Fit with clustered SEs
    model = smf.ols(formula, data=analysis).fit(
        cov_type="cluster", cov_kwds={"groups": analysis["atopid"]}
    )

    # Print all results
    all_vars = ["hierarchical", "voice_driven", "rile_dyad_mean"] + available_controls
    results = _print_results(model, all_vars, "H2A/H2B")

    # Hypothesis tests
    coef_h = model.params["hierarchical"]
    pval_h = model.pvalues["hierarchical"]
    coef_v = model.params["voice_driven"]
    pval_v = model.pvalues["voice_driven"]
    se_h = model.bse["hierarchical"]
    se_v = model.bse["voice_driven"]

    h2a_supported = coef_v > 0 and pval_v < 0.10
    h2b_part1 = coef_h > 0 and pval_h < 0.10

    # Wald test for hierarchical > voice_driven
    from scipy import stats as scipy_stats
    diff = coef_h - coef_v
    se_diff = (se_h**2 + se_v**2)**0.5
    t_stat = diff / se_diff
    p_diff = 2 * (1 - scipy_stats.t.cdf(abs(t_stat), model.df_resid))
    h2b_part2 = coef_h > coef_v and p_diff < 0.10

    print(f"\n  H2A (voice > uninst):      {'SUPPORTED' if h2a_supported else 'NOT SUPPORTED'}")
    print(f"  H2B (hier > uninst):       {'SUPPORTED' if h2b_part1 else 'NOT SUPPORTED'}")
    print(f"  H2B (hier > voice):        diff={diff:.4f}, p={p_diff:.4f} {'SUPPORTED' if h2b_part2 else 'NOT SUPPORTED'}")

    results.to_csv(paths.tables_dir / "model3_h2ab.csv", index=False)
    print(f"  Saved: model3_h2ab.csv")

    return results


def model_robustness(paths: Paths) -> pd.DataFrame:
    """
    Robustness: Joint model with residualized depth.

    Model: spec_dyad_mean ~ hierarchical + voice_driven + depth_within_type + controls + year_FE

    Tests whether depth matters within institution type.
    """
    print("\n" + "="*60)
    print(" Model 4: Robustness (Joint with depth_within_type)")
    print("="*60)

    df = pd.read_csv(paths.dyad_year_csv)

    # Check if depth_within_type exists
    if "depth_within_type" not in df.columns:
        print("  Skipped: depth_within_type not available")
        return pd.DataFrame()

    # Required variables
    available_controls = [c for c in DYAD_CONTROLS if c in df.columns]
    required = ["spec_dyad_mean", "hierarchical", "voice_driven", "depth_within_type",
                "rile_dyad_mean", "atopid", "year"] + available_controls
    analysis = df.dropna(subset=required).copy()

    # Build formula
    controls = " + ".join(["rile_dyad_mean"] + available_controls)
    formula = f"spec_dyad_mean ~ hierarchical + voice_driven + depth_within_type + {controls} + C(year)"

    # Fit with clustered SEs
    model = smf.ols(formula, data=analysis).fit(
        cov_type="cluster", cov_kwds={"groups": analysis["atopid"]}
    )

    # Print all results
    all_vars = ["hierarchical", "voice_driven", "depth_within_type", "rile_dyad_mean"] + available_controls
    results = _print_results(model, all_vars, "Robustness")

    results.to_csv(paths.tables_dir / "model4_robustness.csv", index=False)
    print(f"  Saved: model4_robustness.csv")

    return results


def create_summary_table(paths: Paths) -> pd.DataFrame:
    """Create publication-ready summary table of all models."""
    print("\n" + "="*60)
    print(" Summary Table")
    print("="*60)

    # Load all model results
    files = [
        ("Model 1 (H1)", "model1_h1.csv"),
        ("Model 2 (H2)", "model2_h2.csv"),
        ("Model 3 (H2A/H2B)", "model3_h2ab.csv"),
        ("Model 4 (Robustness)", "model4_robustness.csv"),
    ]

    summaries = []
    for name, filename in files:
        filepath = paths.tables_dir / filename
        if filepath.exists():
            df = pd.read_csv(filepath)
            df["Model"] = name
            summaries.append(df)

    if summaries:
        summary = pd.concat(summaries, ignore_index=True)
        summary = summary[["Model", "Variable", "Coefficient", "SE", "p-value", "Sig"]]
        summary.to_csv(paths.tables_dir / "all_models_summary.csv", index=False)
        print(f"  Saved: all_models_summary.csv")
        return summary

    return pd.DataFrame()


def run_all() -> None:
    """Run all regression analyses."""
    paths = Paths()

    # Ensure output directory exists
    paths.tables_dir.mkdir(parents=True, exist_ok=True)

    model_h1(paths)
    model_h2(paths)
    model_h2ab(paths)
    model_robustness(paths)
    create_summary_table(paths)

    print("\n[Done] All regressions complete.")


if __name__ == "__main__":
    run_all()
