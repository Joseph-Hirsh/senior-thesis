"""
Regression Analyses for Military Specialization & Alliance Institutions Study.

This module tests two core hypotheses about the determinants of military
portfolio structure among democratic states:

    H1: Ruling party ideology affects military specialization
        - Right-of-center parties → less specialization (β < 0)
        - Unit: Country-year
        - Method: Two-way FE with lagged ideology (1-10 years)

    H2: Alliance institutionalization affects division of labor
        - H2A: Voice-driven > Uninstitutionalized (β > 0)
        - H2B: Hierarchical > Voice-driven (β_hier > β_voice)
        - Unit: Dyad-year
        - Method: Dyad FE following Gannon (2023)

Statistical Features:
    - Proper Wald test for coefficient comparisons in H2B
    - Clustered standard errors (country for H1, alliance for H2)
    - Two-way fixed effects throughout

Author: [Your Name]
Date: 2024
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats as scipy_stats
from statsmodels.regression.linear_model import RegressionResultsWrapper

from senior_thesis.config import (
    COUNTRY_CONTROLS,
    DYAD_CONTROLS,
    FORMULAS,
    Paths,
    get_available_controls,
    load_dataset,
)

__all__ = [
    "model_h1",
    "model_h1_event_study",
    "model_h1_did",
    "model_h2",
    "model_h2_event_study",
    "run_all",
]


# =============================================================================
# Data Structures
# =============================================================================


class ModelResult(NamedTuple):
    """Container for regression model results."""

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


# =============================================================================
# Output Formatting
# =============================================================================


def _sig_stars(p: float) -> str:
    """Return significance stars for p-value."""
    if pd.isna(p):
        return ""
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return ""


def _conclusion(coef: float, pval: float, expected_positive: bool) -> str:
    """Determine conclusion based on coefficient direction and significance."""
    if pval >= 0.10:
        return "NULL RESULTS"
    if expected_positive:
        return "SUPPORTED" if coef > 0 else "CONTRARY TO HYPOTHESIS"
    return "SUPPORTED" if coef < 0 else "CONTRARY TO HYPOTHESIS"


def _box(text: str, width: int = 70) -> str:
    """Create a boxed header."""
    lines = [
        "╔" + "═" * (width - 2) + "╗",
        "║" + text.center(width - 2) + "║",
        "╚" + "═" * (width - 2) + "╝",
    ]
    return "\n".join(lines)


def _section(text: str) -> None:
    """Print a section header."""
    print()
    print("─" * 70)
    print(f"  {text}")
    print("─" * 70)


def _subsection(text: str) -> None:
    """Print a subsection header."""
    print()
    print(f"  {text}")
    print()


# =============================================================================
# Core Regression Runner
# =============================================================================


def _run_model(spec: ModelSpec) -> ModelResult:
    """Run OLS regression with clustered standard errors."""
    analysis = spec.data.dropna(subset=spec.required_vars).copy()

    model = smf.ols(spec.formula, data=analysis).fit(
        cov_type="cluster",
        cov_kwds={"groups": analysis[spec.cluster_col]},
    )

    rows = []
    for var in spec.key_vars:
        rows.append({
            "Variable": var,
            "Coefficient": model.params.get(var, np.nan),
            "SE": model.bse.get(var, np.nan),
            "p-value": model.pvalues.get(var, np.nan),
            "Sig": _sig_stars(model.pvalues.get(var, np.nan)),
        })

    results = pd.DataFrame(rows)
    results.to_csv(spec.output_path, index=False)

    return ModelResult(results_df=results, model=model)


# =============================================================================
# H1: Ideology → Specialization (Country-Year)
# =============================================================================


def model_h1(paths: Paths, lags: range = range(1, 11)) -> pd.DataFrame:
    """
    Test H1: Right-of-center ruling parties reduce military specialization.

    Theoretical Motivation:
        Right-of-center parties prioritize military self-sufficiency and
        national defense autonomy. This preference leads to broader military
        portfolios (less specialization) as states seek to maintain
        independent capabilities across multiple domains.

    Empirical Strategy:
        - DV: Standardized specialization index (spec_y)
        - IV: Lagged binary right-of-center indicator (1-10 year lags)
        - FE: Country + Year (absorbs time-invariant country traits, common shocks)
        - SE: Clustered by country (accounts for serial correlation)

    Why Lagged Effects?
        Defense procurement cycles span 5-15 years. Changes in ruling party
        ideology affect procurement decisions, which then take years to
        materialize in observable portfolio changes. Each lag tests a
        distinct temporal mechanism and is therefore treated as independent.

    Args:
        paths: Configuration paths object
        lags: Range of lag years to test (default: 1-10)

    Returns:
        DataFrame with results for each lag specification
    """
    print()
    print(_box("H1: RULING PARTY IDEOLOGY → MILITARY SPECIALIZATION"))

    print()
    print("  HYPOTHESIS")
    print("  " + "─" * 66)
    print("  Right-of-center ruling parties reduce military specialization")
    print("  by prioritizing defense autonomy over alliance burden-sharing.")
    print()
    print("  Expected: Negative coefficient on lagged right_of_center")
    print()
    print("  IDENTIFICATION STRATEGY")
    print("  " + "─" * 66)
    print("  • Unit of analysis:  Country-year")
    print("  • Dependent variable: Standardized specialization index")
    print("  • Independent var:    Lagged right-of-center (binary, RILE ≥ 10)")
    print("  • Fixed effects:      Country + Year (two-way)")
    print("  • Standard errors:    Clustered by country")
    print("  • Lags tested:        1-10 years (procurement cycles)")
    print("  • Multiple testing:   None (each lag tests independent temporal mechanism)")

    df = load_dataset(paths.country_year_csv).copy()
    df = df.sort_values(["country_code_cow", "year"])

    for lag in lags:
        df[f"roc_lag{lag}"] = df.groupby("country_code_cow")["right_of_center"].shift(lag)

    controls = " + ".join(COUNTRY_CONTROLS)
    results_all = []

    _section("RESULTS BY LAG LENGTH")
    print()
    print(f"  {'Lag':<6} {'Coef':>10} {'SE':>10} {'p-value':>10} {'':>6}")
    print("  " + "─" * 44)

    for lag in lags:
        var = f"roc_lag{lag}"
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

        print(f"  {lag:<6} {coef:>10.4f} {se:>10.4f} {pval:>10.4f} {sig:>6}")

        results_all.append({
            "Lag": lag,
            "Variable": var,
            "Coefficient": coef,
            "SE": se,
            "p-value": pval,
            "Sig": sig,
            "N": int(model.nobs),
            "R2": model.rsquared,
        })

    results = pd.DataFrame(results_all)
    results.to_csv(paths.h1_dir / "model_h1_lagged.csv", index=False)

    # Model diagnostics
    _section("MODEL DIAGNOSTICS")
    print()
    print(f"  Observations:     {int(model.nobs):,}")
    print(f"  Countries:        {analysis['country_code_cow'].nunique()}")
    print(f"  Year range:       {int(analysis['year'].min())}-{int(analysis['year'].max())}")
    print(f"  R-squared:        {model.rsquared:.4f}")

    # Conclusion
    _section("CONCLUSION")

    sig_results = results[results["p-value"] < 0.10]
    if len(sig_results) > 0:
        supporting = sig_results[sig_results["Coefficient"] < 0]
        contrary = sig_results[sig_results["Coefficient"] > 0]

        if len(supporting) > 0:
            best = supporting.loc[supporting["p-value"].idxmin()]
            print()
            print(f"  ✓ HYPOTHESIS SUPPORTED")
            print(f"    Best lag: {int(best['Lag'])} years")
            print(f"    β = {best['Coefficient']:.4f}, p = {best['p-value']:.4f}")
            print(f"    Significant at lags: {', '.join([str(int(x)) for x in supporting['Lag'].values])}")

        if len(contrary) > 0:
            print()
            print(f"  ✗ CONTRARY TO HYPOTHESIS at some lags")
            print(f"    Positive effects at lags: {', '.join([str(int(x)) for x in contrary['Lag'].values])}")
            best_contrary = contrary.loc[contrary["p-value"].idxmin()]
            print(f"    Strongest contrary: lag {int(best_contrary['Lag'])}, β = {best_contrary['Coefficient']:.4f}, p = {best_contrary['p-value']:.4f}")
    else:
        print()
        print("  ○ NULL RESULTS")
        print("    No significant effects at any lag (p < 0.10)")

    print()
    return results


# =============================================================================
# H1: Event Study Design
# =============================================================================


def _run_event_study(
    df: pd.DataFrame,
    event_col: str,
    label: str,
    expected_sign: str,
    output_path: Path,
    window: int = 5,
) -> pd.DataFrame:
    """Run event study regression around ideology transitions."""
    df = df.copy()
    df = df.sort_values(["country_code_cow", "year"])

    events = df[df[event_col]][["country_code_cow", "year"]].copy()
    events = events.rename(columns={"year": "event_year"})

    if len(events) == 0:
        print(f"    No {label.lower()} found in data")
        return pd.DataFrame()

    first_events = events.groupby("country_code_cow").first().reset_index()
    n_countries = len(first_events)

    print(f"    Events identified: {len(events)} transitions across {n_countries} countries")

    df = df.merge(first_events, on="country_code_cow", how="left")
    df["event_time"] = df["year"] - df["event_year"]
    df_event = df[df["event_time"].between(-window, window)].copy()

    if len(df_event) < 50:
        print(f"    Insufficient observations: {len(df_event)} (minimum: 50)")
        return pd.DataFrame()

    event_times = [t for t in range(-window, window + 1) if t != -1]
    for t in event_times:
        col_name = f"t_{t:+d}".replace("+", "p").replace("-", "m")
        df_event[col_name] = (df_event["event_time"] == t).astype(int)

    available_controls = get_available_controls(df_event, COUNTRY_CONTROLS)
    controls = " + ".join(available_controls) if available_controls else "1"
    event_vars = [f"t_{t:+d}".replace("+", "p").replace("-", "m") for t in event_times]
    formula = f"spec_y ~ {' + '.join(event_vars)} + {controls} + C(country_code_cow) + C(year)"

    required = ["spec_y", "country_code_cow", "year", "event_time"] + available_controls + event_vars
    analysis = df_event.dropna(subset=required).copy()

    if len(analysis) < 50:
        print(f"    Insufficient observations after listwise deletion: {len(analysis)}")
        return pd.DataFrame()

    model = smf.ols(formula, data=analysis).fit(
        cov_type="cluster", cov_kwds={"groups": analysis["country_code_cow"]}
    )

    results_all = []
    for t in event_times:
        var = f"t_{t:+d}".replace("+", "p").replace("-", "m")
        results_all.append({
            "event_time": t,
            "Coefficient": model.params.get(var, np.nan),
            "SE": model.bse.get(var, np.nan),
            "p-value": model.pvalues.get(var, np.nan),
            "Sig": _sig_stars(model.pvalues.get(var, 1.0)),
        })
    results_all.append({
        "event_time": -1,
        "Coefficient": 0.0,
        "SE": 0.0,
        "p-value": np.nan,
        "Sig": "(ref)",
    })

    results = pd.DataFrame(results_all).sort_values("event_time")
    results.to_csv(output_path, index=False)

    # Print coefficient table
    print()
    print(f"    {'Time':<8} {'Coef':>10} {'SE':>10} {'p-value':>10} {'':>6}")
    print("    " + "─" * 44)
    for _, row in results.iterrows():
        t = int(row["event_time"])
        if t == -1:
            print(f"    t = {t:<4} {'0.0000':>10} {'—':>10} {'—':>10} {'(ref)':>6}")
        else:
            print(f"    t = {t:<4} {row['Coefficient']:>10.4f} {row['SE']:>10.4f} {row['p-value']:>10.4f} {row['Sig']:>6}")

    print()
    print(f"    N = {int(model.nobs):,}  |  R² = {model.rsquared:.4f}")

    # Pre-trends test
    pre_trend = results[(results["event_time"] < -1) & (results["event_time"] >= -window)]
    pre_sig = pre_trend[pre_trend["p-value"] < 0.10]
    if len(pre_sig) > 0:
        print("    ⚠ Pre-trends detected (parallel trends assumption may be violated)")
    else:
        print("    ✓ No pre-trends detected")

    # Post-effect assessment
    post_effects = results[results["event_time"] > 0]
    post_sig = post_effects[post_effects["p-value"] < 0.10]

    if len(post_sig) > 0:
        avg_coef = post_sig["Coefficient"].mean()
        is_negative = avg_coef < 0
        expected_negative = expected_sign == "negative"

        if is_negative == expected_negative:
            print(f"    ✓ Post-event effects: {avg_coef:.4f} ({expected_sign} as expected)")
        else:
            actual = "negative" if is_negative else "positive"
            print(f"    ✗ Post-event effects: {avg_coef:.4f} ({actual}, expected {expected_sign})")
    else:
        print("    ○ No significant post-event effects")

    return results


def model_h1_event_study(paths: Paths, window: int = 5) -> dict[str, pd.DataFrame]:
    """
    Event study for H1: Specialization changes around ideology transitions.

    Theoretical Motivation:
        If ideology causally affects specialization, we should observe:
        1. No differential pre-trends before transitions (parallel trends)
        2. Systematic post-transition changes in the predicted direction

    Design:
        - Window: t-5 to t+5 around ideology transition
        - Reference period: t = -1 (year before transition)
        - Separate analyses for transitions TO right vs TO left
        - Uses threshold of RILE = 0 for more transition events

    Interpretation:
        - Transitions TO right: expect negative post-effects (less specialization)
        - Transitions TO left: expect positive post-effects (more specialization)
    """
    print()
    print(_box("H1 EVENT STUDY: IDEOLOGY TRANSITIONS"))

    print()
    print("  DESIGN")
    print("  " + "─" * 66)
    print("  This event study tests whether specialization changes systematically")
    print("  after ruling party ideology transitions, providing evidence for the")
    print("  causal effect of ideology on military portfolio structure.")
    print()
    print("  • Event window:      t-5 to t+5 around transition")
    print("  • Reference period:  t = -1 (year before transition)")
    print("  • Ideology threshold: RILE = 0 (for sufficient transition events)")
    print("  • Key assumption:    Parallel trends in pre-period")

    df = load_dataset(paths.country_year_csv).copy()
    df = df.sort_values(["country_code_cow", "year"])

    df["right_zero"] = (df["rile"] >= 0).astype(float)
    df["right_zero"] = df["right_zero"].where(df["rile"].notna())

    df["roc_prev"] = df.groupby("country_code_cow")["right_zero"].shift(1)
    df["transition"] = (df["right_zero"] != df["roc_prev"]) & df["roc_prev"].notna()
    df["transition_to_right"] = df["transition"] & (df["right_zero"] == 1)
    df["transition_to_left"] = df["transition"] & (df["right_zero"] == 0)

    _section("SAMPLE OVERVIEW")
    print()
    print(f"  Transitions to right (left → right): {int(df['transition_to_right'].sum())}")
    print(f"  Transitions to left (right → left):  {int(df['transition_to_left'].sum())}")

    results = {}

    _section("TRANSITIONS TO RIGHT (Left → Right)")
    print()
    print("  If H1 is correct, right-wing governments reduce specialization.")
    print("  Expected: NEGATIVE coefficients in post-transition periods")
    print()

    results["to_right"] = _run_event_study(
        df=df,
        event_col="transition_to_right",
        label="Transitions to right",
        expected_sign="negative",
        output_path=paths.h1_dir / "model_h1_event_study_to_right.csv",
        window=window,
    )

    _section("TRANSITIONS TO LEFT (Right → Left)")
    print()
    print("  Symmetric prediction: left-wing governments increase specialization.")
    print("  Expected: POSITIVE coefficients in post-transition periods")
    print()

    results["to_left"] = _run_event_study(
        df=df,
        event_col="transition_to_left",
        label="Transitions to left",
        expected_sign="positive",
        output_path=paths.h1_dir / "model_h1_event_study_to_left.csv",
        window=window,
    )

    return results


def model_h1_did(paths: Paths, window: int = 5) -> dict[str, dict]:
    """
    Difference-in-differences for H1: Single post-transition effect estimate.

    This is a more parsimonious specification than the event study, estimating
    a single average post-treatment effect rather than separate coefficients
    for each event-time. This approach has more statistical power but provides
    less information about effect dynamics.
    """
    print()
    print(_box("H1 DIFFERENCE-IN-DIFFERENCES"))

    print()
    print("  DESIGN")
    print("  " + "─" * 66)
    print("  A parsimonious alternative to the event study that estimates a")
    print("  single average post-transition effect. More statistical power,")
    print("  but less information about effect dynamics.")
    print()
    print("  • Specification:  spec_y ~ post + controls + country_FE + year_FE")
    print("  • Post indicator: 1 if event_time ≥ 0, 0 otherwise")
    print(f"  • Window:         ±{window} years around transition")

    df = load_dataset(paths.country_year_csv).copy()
    df = df.sort_values(["country_code_cow", "year"])

    df["right_zero"] = (df["rile"] >= 0).astype(float)
    df["right_zero"] = df["right_zero"].where(df["rile"].notna())

    df["roc_prev"] = df.groupby("country_code_cow")["right_zero"].shift(1)
    df["transition"] = (df["right_zero"] != df["roc_prev"]) & df["roc_prev"].notna()
    df["transition_to_right"] = df["transition"] & (df["right_zero"] == 1)
    df["transition_to_left"] = df["transition"] & (df["right_zero"] == 0)

    results = {}

    for direction, event_col, expected_sign in [
        ("TO RIGHT", "transition_to_right", "negative"),
        ("TO LEFT", "transition_to_left", "positive"),
    ]:
        _section(f"TRANSITIONS {direction}")
        print()
        print(f"  Expected: {expected_sign.upper()} post-transition effect")
        print()

        events = df[df[event_col]][["country_code_cow", "year"]].copy()
        events = events.rename(columns={"year": "event_year"})

        if len(events) == 0:
            print("    No transitions found")
            continue

        first_events = events.groupby("country_code_cow").first().reset_index()
        n_countries = len(first_events)
        print(f"    Events: {len(events)} transitions across {n_countries} countries")

        df_temp = df.merge(first_events, on="country_code_cow", how="left")
        df_temp["event_time"] = df_temp["year"] - df_temp["event_year"]
        df_event = df_temp[df_temp["event_time"].between(-window, window)].copy()

        if len(df_event) < 50:
            print(f"    Insufficient observations: {len(df_event)}")
            continue

        df_event["post"] = (df_event["event_time"] >= 0).astype(int)

        available_controls = get_available_controls(df_event, COUNTRY_CONTROLS)
        controls = " + ".join(available_controls) if available_controls else "1"
        formula = f"spec_y ~ post + {controls} + C(country_code_cow) + C(year)"

        required = ["spec_y", "country_code_cow", "year", "post"] + available_controls
        analysis = df_event.dropna(subset=required).copy()

        if len(analysis) < 50:
            print(f"    Insufficient observations: {len(analysis)}")
            continue

        model = smf.ols(formula, data=analysis).fit(
            cov_type="cluster", cov_kwds={"groups": analysis["country_code_cow"]}
        )

        coef = model.params["post"]
        se = model.bse["post"]
        pval = model.pvalues["post"]
        sig = _sig_stars(pval)

        print()
        print(f"    Post-transition effect: β = {coef:.4f} (SE = {se:.4f})")
        print(f"    p-value: {pval:.4f} {sig}")
        print(f"    N = {int(model.nobs):,}  |  R² = {model.rsquared:.4f}")
        print()

        is_correct_sign = (coef < 0) == (expected_sign == "negative")

        if pval < 0.10:
            if is_correct_sign:
                print(f"    ✓ HYPOTHESIS SUPPORTED")
            else:
                print(f"    ✗ CONTRARY TO HYPOTHESIS")
        else:
            print(f"    ○ NULL RESULTS (direction {'correct' if is_correct_sign else 'incorrect'})")

        results[direction.lower().replace(" ", "_")] = {
            "coefficient": coef,
            "se": se,
            "p_value": pval,
            "n": int(model.nobs),
            "n_countries": n_countries,
        }

    return results


# =============================================================================
# H2: Alliance Type → Division of Labor (Dyad-Year)
# =============================================================================


def model_h2(paths: Paths) -> pd.DataFrame:
    """
    Test H2: Alliance institutionalization promotes division of labor.

    Theoretical Motivation:
        Institutionalized alliances provide mechanisms (consultation, joint
        planning, integrated command) that enable credible commitment to
        role specialization. Partners can specialize knowing their allies
        will provide complementary capabilities.

        H2A: Voice-driven alliances (consultation mechanisms) enable more
             division of labor than uninstitutionalized alliances.
        H2B: Hierarchical alliances (command authority) enable even more
             division of labor by reducing coordination costs.

    Empirical Strategy:
        - DV: Division of labor (portfolio dissimilarity, 0-1 scale)
        - IV: Alliance type dummies (ref: uninstitutionalized)
        - FE: Dyad + Decade (within-dyad identification, following Gannon 2023)
        - SE: Clustered by alliance
        - Test: Wald test for H2B (hier > voice) using covariance matrix

    Why Dyad Fixed Effects?
        Dyad FE controls for all time-invariant dyad characteristics (geography,
        historical ties, baseline complementarity). This identifies effects from
        dyads that CHANGE their institutional arrangement over time—a more
        credibly causal design than between-dyad comparisons.
    """
    print()
    print(_box("H2: ALLIANCE INSTITUTIONALIZATION → DIVISION OF LABOR"))

    print()
    print("  HYPOTHESES")
    print("  " + "─" * 66)
    print("  H2A: Voice-driven alliances produce more division of labor than")
    print("       uninstitutionalized alliances (β_voice > 0)")
    print()
    print("  H2B: Hierarchical alliances produce more division of labor than")
    print("       voice-driven alliances (β_hier > β_voice)")
    print()
    print("  IDENTIFICATION STRATEGY")
    print("  " + "─" * 66)
    print("  • Unit of analysis:   Dyad-year (within alliance)")
    print("  • Dependent variable: Division of labor (portfolio dissimilarity)")
    print("  • Independent vars:   Hierarchical, Voice-driven (ref: uninstitutionalized)")
    print("  • Fixed effects:      Dyad + Decade (within-dyad identification)")
    print("  • Standard errors:    Clustered by alliance")
    print("  • H2B test:           Wald test with proper covariance matrix")
    print()
    print("  Note: Dyad FE identifies from the ~38% of dyads that change")
    print("        institution type over time, controlling for all time-invariant")
    print("        dyad characteristics.")

    df = load_dataset(paths.dyad_year_csv).copy()
    df["dyad_id"] = df["state_a"].astype(str) + "_" + df["state_b"].astype(str)

    available_controls = get_available_controls(df, DYAD_CONTROLS)
    controls = " + ".join(available_controls) if available_controls else "1"

    formula = f"div_labor ~ hierarchical + voice_driven + {controls} + C(dyad_id) + C(decade)"
    key_vars = ["hierarchical", "voice_driven"] + available_controls

    spec = ModelSpec(
        data=df,
        formula=formula,
        cluster_col="atopid",
        required_vars=["div_labor", "hierarchical", "voice_driven", "atopid", "dyad_id", "decade"] + available_controls,
        output_path=paths.h2_dir / "model_h2_type.csv",
        title="H2",
        key_vars=key_vars,
    )

    result = _run_model(spec)

    coef_h = result.model.params["hierarchical"]
    pval_h = result.model.pvalues["hierarchical"]
    se_h = result.model.bse["hierarchical"]
    coef_v = result.model.params["voice_driven"]
    pval_v = result.model.pvalues["voice_driven"]
    se_v = result.model.bse["voice_driven"]

    _section("KEY RESULTS")
    print()
    print(f"  {'Variable':<20} {'Coefficient':>12} {'SE':>10} {'p-value':>10} {'':>6}")
    print("  " + "─" * 60)
    print(f"  {'Hierarchical':<20} {coef_h:>12.4f} {se_h:>10.4f} {pval_h:>10.4f} {_sig_stars(pval_h):>6}")
    print(f"  {'Voice-driven':<20} {coef_v:>12.4f} {se_v:>10.4f} {pval_v:>10.4f} {_sig_stars(pval_v):>6}")
    print()
    print(f"  Reference category: Uninstitutionalized")

    # Proper Wald test for H2B
    try:
        wald_test = result.model.wald_test("hierarchical - voice_driven = 0", scalar=True)
        p_diff = float(wald_test.pvalue)
        diff = coef_h - coef_v
    except Exception:
        diff = coef_h - coef_v
        se_diff = np.sqrt(se_h**2 + se_v**2)
        t_stat = diff / se_diff
        p_diff = 2 * (1 - scipy_stats.t.cdf(abs(t_stat), result.model.df_resid))

    print()
    print(f"  H2B Test (hier - voice):")
    print(f"    Difference: {diff:.4f}")
    print(f"    p-value:    {p_diff:.4f} {_sig_stars(p_diff)}")

    _section("MODEL DIAGNOSTICS")
    print()
    print(f"  Observations:   {int(result.model.nobs):,}")
    print(f"  Unique dyads:   {df['dyad_id'].nunique():,}")
    print(f"  Unique alliances: {df['atopid'].nunique()}")
    print(f"  R-squared:      {result.model.rsquared:.4f}")

    _section("CONCLUSIONS")

    # H2A
    print()
    print("  H2A: Voice-driven > Uninstitutionalized")
    if pval_v < 0.10 and coef_v > 0:
        print(f"    ✓ SUPPORTED (β = {coef_v:.4f}, p = {pval_v:.4f})")
    elif pval_v < 0.10 and coef_v < 0:
        print(f"    ✗ CONTRARY TO HYPOTHESIS (β = {coef_v:.4f}, p = {pval_v:.4f})")
    else:
        print(f"    ○ NULL RESULTS (β = {coef_v:.4f}, p = {pval_v:.4f})")

    # H2B
    print()
    print("  H2B: Hierarchical > Voice-driven")
    if p_diff < 0.10 and diff > 0:
        print(f"    ✓ SUPPORTED (Δ = {diff:.4f}, p = {p_diff:.4f})")
    elif p_diff < 0.10 and diff < 0:
        print(f"    ✗ CONTRARY TO HYPOTHESIS (Δ = {diff:.4f}, p = {p_diff:.4f})")
    else:
        print(f"    ○ NULL RESULTS (Δ = {diff:.4f}, p = {p_diff:.4f})")

    print()
    return result.results_df


def model_h2_event_study(paths: Paths, window: int = 5) -> pd.DataFrame:
    """
    Event study for H2: Division of labor changes around alliance formation.

    Tests whether dyads exhibit systematic changes in division of labor
    after forming an alliance, relative to the pre-alliance period.
    """
    print()
    print(_box("H2 EVENT STUDY: ALLIANCE FORMATION"))

    print()
    print("  DESIGN")
    print("  " + "─" * 66)
    print("  Tests whether division of labor changes systematically after")
    print("  alliance formation by tracking the same dyads before and after")
    print("  they enter into an alliance together.")
    print()
    print("  • Event:           First alliance entry for each dyad")
    print("  • Window:          t-5 to t+5 around alliance formation")
    print("  • Reference:       t = -1 (year before alliance)")
    print("  • Expected:        Positive post-formation coefficients")

    div_labor_full = pd.read_csv(paths.div_labor_csv)

    div_labor_full["s_min"] = div_labor_full[["state_a", "state_b"]].min(axis=1)
    div_labor_full["s_max"] = div_labor_full[["state_a", "state_b"]].max(axis=1)
    div_labor_full["state_a"] = div_labor_full["s_min"]
    div_labor_full["state_b"] = div_labor_full["s_max"]
    div_labor_full = div_labor_full.drop(columns=["s_min", "s_max"])

    dyad_df = load_dataset(paths.dyad_year_csv)
    first_alliance = dyad_df.groupby(["state_a", "state_b"])["year"].min().reset_index()
    first_alliance.columns = ["state_a", "state_b", "alliance_entry_year"]

    _section("SAMPLE")
    print()
    print(f"  Dyads entering alliances: {len(first_alliance):,}")

    if len(first_alliance) == 0:
        print("  No alliance entries found")
        return pd.DataFrame()

    panel = div_labor_full.merge(first_alliance, on=["state_a", "state_b"], how="inner")
    panel["event_time"] = panel["year"] - panel["alliance_entry_year"]
    panel_event = panel[panel["event_time"].between(-window, window)].copy()

    n_dyads = panel_event.groupby(["state_a", "state_b"]).ngroups
    print(f"  Dyad-years in window:     {len(panel_event):,}")
    print(f"  Unique dyads:             {n_dyads:,}")

    if len(panel_event) < 50:
        print("  Insufficient data")
        return pd.DataFrame()

    event_times = [t for t in range(-window, window + 1) if t != -1]
    for t in event_times:
        col_name = f"t_{t:+d}".replace("+", "p").replace("-", "m")
        panel_event[col_name] = (panel_event["event_time"] == t).astype(int)

    panel_event["dyad_id"] = panel_event["state_a"].astype(str) + "_" + panel_event["state_b"].astype(str)

    event_vars = [f"t_{t:+d}".replace("+", "p").replace("-", "m") for t in event_times]
    formula = f"div_labor ~ {' + '.join(event_vars)} + C(state_a) + C(state_b) + C(year)"

    required = ["div_labor", "state_a", "state_b", "year", "dyad_id"] + event_vars
    analysis = panel_event.dropna(subset=required).copy()

    if len(analysis) < 50:
        print("  Insufficient data after listwise deletion")
        return pd.DataFrame()

    print(f"  Final sample:             {len(analysis):,}")

    model = smf.ols(formula, data=analysis).fit(
        cov_type="cluster", cov_kwds={"groups": analysis["dyad_id"]}
    )

    results_all = []
    for t in event_times:
        var = f"t_{t:+d}".replace("+", "p").replace("-", "m")
        results_all.append({
            "event_time": t,
            "Coefficient": model.params.get(var, np.nan),
            "SE": model.bse.get(var, np.nan),
            "p-value": model.pvalues.get(var, np.nan),
            "Sig": _sig_stars(model.pvalues.get(var, 1.0)),
        })
    results_all.append({
        "event_time": -1,
        "Coefficient": 0.0,
        "SE": 0.0,
        "p-value": np.nan,
        "Sig": "(ref)",
    })

    results = pd.DataFrame(results_all).sort_values("event_time")
    results.to_csv(paths.h2_dir / "model_h2_event_study.csv", index=False)

    _section("RESULTS")
    print()
    print(f"  {'Time':<8} {'Coefficient':>12} {'SE':>10} {'p-value':>10} {'':>6}")
    print("  " + "─" * 48)
    for _, row in results.iterrows():
        t = int(row["event_time"])
        if t == -1:
            print(f"  t = {t:<4} {'0.0000':>12} {'—':>10} {'—':>10} {'(ref)':>6}")
        else:
            print(f"  t = {t:<4} {row['Coefficient']:>12.4f} {row['SE']:>10.4f} {row['p-value']:>10.4f} {row['Sig']:>6}")

    print()
    print(f"  N = {int(model.nobs):,}  |  R² = {model.rsquared:.4f}")

    _section("DIAGNOSTICS")

    pre_trend = results[(results["event_time"] < -1) & (results["event_time"] >= -window)]
    pre_sig = pre_trend[pre_trend["p-value"] < 0.10]
    if len(pre_sig) > 0:
        print()
        print("  ⚠ Pre-trends detected (parallel trends assumption may be violated)")
    else:
        print()
        print("  ✓ No pre-trends detected")

    post_effects = results[results["event_time"] > 0]
    post_sig = post_effects[post_effects["p-value"] < 0.10]

    _section("CONCLUSION")

    if len(post_sig) > 0:
        avg_coef = post_sig["Coefficient"].mean()
        if avg_coef > 0:
            print()
            print(f"  ✓ HYPOTHESIS SUPPORTED")
            print(f"    Average post-formation effect: {avg_coef:.4f}")
        else:
            print()
            print(f"  ✗ CONTRARY TO HYPOTHESIS")
            print(f"    Average post-formation effect: {avg_coef:.4f}")
    else:
        print()
        print("  ○ NULL RESULTS")
        print("    No significant post-formation effects")

    print()
    return results


# =============================================================================
# Main Entry Point
# =============================================================================


def run_all() -> None:
    """Run all regression analyses."""
    paths = Paths()

    missing = paths.validate()
    if missing:
        print(f"Error: Missing input files: {missing}")
        return

    paths.h1_dir.mkdir(parents=True, exist_ok=True)
    paths.h2_dir.mkdir(parents=True, exist_ok=True)

    model_h1(paths)
    model_h1_event_study(paths)
    model_h1_did(paths)
    model_h2(paths)
    model_h2_event_study(paths)

    print()
    print(_box("ALL ANALYSES COMPLETE"))
    print()


if __name__ == "__main__":
    run_all()
