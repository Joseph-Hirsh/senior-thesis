"""
Regression models for testing hypotheses.

H1: Ruling party ideology → Military specialization (country-year)
H2: Alliance institutionalization → Division of labor (dyad-year)
H3: Ideological similarity → Division of labor (dyad-year)

All models use:
- Two-way fixed effects (unit + time)
- Clustered standard errors
- ATOP-only alliance type coding (Leeds & Anac 2005)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats as scipy_stats

from senior_thesis.config import Paths, DYAD_CONTROLS, get_available_controls

__all__ = [
    "run_h1_regressions",
    "run_h2_regressions",
    "run_h3_regressions",
]

# Store results for summary table
_ALL_RESULTS = {}


# =============================================================================
# Output Formatting
# =============================================================================

def _box(title: str) -> str:
    """Create a boxed section header."""
    width = 70
    return (
        f"\n{'═' * width}\n"
        f"  {title}\n"
        f"{'═' * width}"
    )


def _subhead(title: str) -> str:
    """Create a subsection header."""
    return f"\n  ▶ {title}\n  {'─' * 60}"


def _sig(p: float) -> str:
    """Return significance stars."""
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    elif p < 0.10:
        return "†"
    return ""


def _sig_word(p: float) -> str:
    """Return significance description."""
    if p < 0.01:
        return "significant at p<0.01"
    elif p < 0.05:
        return "significant at p<0.05"
    elif p < 0.10:
        return "marginally significant (p<0.10)"
    return "not significant"


def _format_coef(coef: float, se: float, p: float) -> str:
    """Format coefficient with SE and significance."""
    return f"β = {coef:.4f} (SE = {se:.4f}) {_sig(p)}"


# =============================================================================
# H1: IDEOLOGY → SPECIALIZATION
# =============================================================================

def run_h1_regressions(paths: Paths) -> dict:
    """
    Run H1 regressions: Ideology → Specialization.
    """
    print(_box("H1: IDEOLOGY → SPECIALIZATION"))
    print("""
  THEORY: Right-of-center ruling parties, with preferences for lower
  government spending, should produce less military specialization
  (more generalist forces) compared to left-of-center parties.

  PREDICTION: β < 0 (higher RILE → less specialization)
""")

    results = {}
    df = pd.read_csv(paths.country_year_csv)

    # -------------------------------------------------------------------------
    # PRIMARY SPECIFICATION
    # -------------------------------------------------------------------------
    print(_subhead("TEST 1: Primary Specification"))
    print("""
  What this tests: Does ruling party ideology predict military specialization?

  Design:
    • Outcome (DV):    spec_y = standardized specialization index
    • Predictor (IV):  rile_lag5 = ruling party ideology (5-year lag)
                       Scale: -100 (far left) to +100 (far right)
    • Fixed Effects:   Country + Year (absorbs time-invariant country
                       traits and common shocks)
    • Standard Errors: Clustered by country
""")

    controls = ["lngdp", "cinc", "war5_lag"]
    available = get_available_controls(df, controls)
    required = ["spec_y", "rile_lag5", "country_code_cow", "year"] + available
    analysis = df.dropna(subset=required).copy()

    control_str = " + ".join(available) if available else "1"
    formula = f"spec_y ~ rile_lag5 + {control_str} + C(country_code_cow) + C(year)"

    model = smf.ols(formula, data=analysis).fit(
        cov_type="cluster",
        cov_kwds={"groups": analysis["country_code_cow"]},
    )

    coef = model.params["rile_lag5"]
    se = model.bse["rile_lag5"]
    pval = model.pvalues["rile_lag5"]

    print(f"  Sample size:     N = {int(model.nobs):,} country-years")
    print(f"  Countries:       {analysis['country_code_cow'].nunique()}")
    print(f"  Year range:      {int(analysis['year'].min())}–{int(analysis['year'].max())}")
    print()
    print(f"  RESULT:          {_format_coef(coef, se, pval)}")
    print(f"  Interpretation:  A 10-point rightward shift in RILE is associated with")
    print(f"                   a {abs(coef * 10):.3f} SD change in specialization")
    print(f"  Significance:    {_sig_word(pval)}")

    results["primary"] = {
        "coefficient": coef, "se": se, "p_value": pval,
        "n": int(model.nobs), "r2": model.rsquared,
    }

    # Store for summary
    _ALL_RESULTS["H1"] = {
        "coef": coef, "se": se, "p": pval, "n": int(model.nobs),
        "supported": pval < 0.05 and coef < 0,
        "direction": "negative" if coef < 0 else "positive",
    }

    pd.DataFrame([results["primary"]]).to_csv(
        paths.h1_dir / "model_h1_primary.csv", index=False
    )

    # -------------------------------------------------------------------------
    # PLACEBO TEST
    # -------------------------------------------------------------------------
    print(_subhead("TEST 2: Placebo Test (Reverse Causality Check)"))
    print("""
  What this tests: Does FUTURE ideology predict CURRENT specialization?

  Logic: If we find that next year's ideology predicts this year's
  specialization, that would suggest reverse causality or confounding.
  Future events cannot cause past outcomes, so this should be null.

  Design:
    • Predictor (IV):  rile_lead1 = ideology 1 year IN THE FUTURE
    • Expected:        No significant effect (β ≈ 0)
""")

    if "rile_lead1" not in df.columns:
        df = df.sort_values(["country_code_cow", "year"])
        df["rile_lead1"] = df.groupby("country_code_cow")["rile"].shift(-1)

    required_placebo = ["spec_y", "rile_lead1", "country_code_cow", "year"] + available
    analysis_placebo = df.dropna(subset=required_placebo).copy()

    formula_placebo = f"spec_y ~ rile_lead1 + {control_str} + C(country_code_cow) + C(year)"
    model_placebo = smf.ols(formula_placebo, data=analysis_placebo).fit(
        cov_type="cluster",
        cov_kwds={"groups": analysis_placebo["country_code_cow"]},
    )

    coef_p = model_placebo.params["rile_lead1"]
    se_p = model_placebo.bse["rile_lead1"]
    pval_p = model_placebo.pvalues["rile_lead1"]

    print(f"  Sample size:     N = {int(model_placebo.nobs):,}")
    print()
    print(f"  RESULT:          {_format_coef(coef_p, se_p, pval_p)}")

    passed = pval_p >= 0.10
    if passed:
        print(f"  Verdict:         ✓ PASSED — Future ideology does not predict")
        print(f"                   current specialization (rules out reverse causality)")
    else:
        print(f"  Verdict:         ⚠ CONCERN — Possible confounding detected")

    results["placebo"] = {
        "coefficient": coef_p, "se": se_p, "p_value": pval_p, "passed": passed,
    }

    # -------------------------------------------------------------------------
    # EVENT STUDY
    # -------------------------------------------------------------------------
    print(_subhead("TEST 3: Event Study (Ideology Transitions)"))
    print("""
  What this tests: How does specialization evolve around ideology shifts?

  Logic: If ideology causally affects specialization, we should see:
    • No pre-trends (parallel trends before transition)
    • Effects emerge AFTER the transition, not before

  Design:
    • Event:      First time a country's ruling party crosses RILE = 0
    • Window:     5 years before to 5 years after transition
    • Reference:  t = -1 (year before transition)
""")

    event_results = _run_event_study(df, paths, available, control_str)
    results["event_study"] = event_results

    return results


def _run_event_study(
    df: pd.DataFrame,
    paths: Paths,
    available: list[str],
    control_str: str,
) -> dict:
    """Run ideology transition event study."""

    df = df.sort_values(["country_code_cow", "year"])
    df["ideology_sign"] = np.sign(df["rile"])
    df["sign_change"] = df.groupby("country_code_cow")["ideology_sign"].diff().abs() > 0

    transitions = df[df["sign_change"]].copy()
    transitions = transitions.groupby("country_code_cow").first().reset_index()
    transitions = transitions.rename(columns={"year": "event_year"})
    transitions["to_right"] = transitions["ideology_sign"] > 0

    results = {}

    for direction, label in [(True, "to_right"), (False, "to_left")]:
        trans_subset = transitions[transitions["to_right"] == direction]
        direction_desc = "Left → Right" if direction else "Right → Left"

        if len(trans_subset) < 10:
            print(f"  {direction_desc}: Only {len(trans_subset)} transitions (need ≥10)")
            continue

        event_df = df.merge(
            trans_subset[["country_code_cow", "event_year"]],
            on="country_code_cow", how="inner"
        )
        event_df["event_time"] = event_df["year"] - event_df["event_year"]
        event_df = event_df[(event_df["event_time"] >= -5) & (event_df["event_time"] <= 5)]

        def _varname(t: int) -> str:
            return f"tm{abs(t)}" if t < 0 else f"tp{t}"

        for t in range(-5, 6):
            if t != -1:
                event_df[_varname(t)] = (event_df["event_time"] == t).astype(int)

        time_dummies = " + ".join([_varname(t) for t in range(-5, 6) if t != -1])
        formula = f"spec_y ~ {time_dummies} + {control_str} + C(country_code_cow) + C(year)"

        required = ["spec_y", "country_code_cow", "year"] + available
        analysis = event_df.dropna(subset=required)

        try:
            model = smf.ols(formula, data=analysis).fit(
                cov_type="cluster",
                cov_kwds={"groups": analysis["country_code_cow"]},
            )

            print(f"\n  {direction_desc} transitions: {len(trans_subset)} countries")
            print(f"  {'Time':>8}  {'Coefficient':>12}  {'Std.Err.':>10}  {'Sig.':>5}")
            print(f"  {'─' * 40}")

            coefs = []
            post_coefs = []  # For summary: average post-transition effect
            for t in range(-5, 6):
                if t == -1:
                    print(f"  {'t = -1':>8}  {'(reference)':>12}")
                    coefs.append({"time": t, "coef": 0, "se": 0, "p": np.nan})
                else:
                    c = model.params[_varname(t)]
                    s = model.bse[_varname(t)]
                    p = model.pvalues[_varname(t)]
                    print(f"  {f't = {t:>2}':>8}  {c:>12.4f}  {s:>10.4f}  {_sig(p):>5}")
                    coefs.append({"time": t, "coef": c, "se": s, "p": p})
                    if t >= 0:  # Post-transition periods
                        post_coefs.append(c)

            results[label] = pd.DataFrame(coefs)
            results[label].to_csv(paths.h1_dir / f"event_study_{label}.csv", index=False)

            # Store average post-transition effect for summary table
            # For Left→Right: expect negative (less specialization)
            # For Right→Left: expect positive (more specialization)
            avg_post = np.mean(post_coefs) if post_coefs else 0

            # Joint test of post-transition coefficients
            post_vars = [_varname(t) for t in range(0, 6)]
            try:
                joint_test = model.f_test(" = ".join([f"{v} = 0" for v in post_vars[:1]]) if len(post_vars) == 1
                                          else " = ".join(post_vars) + " = 0")
                joint_p = float(joint_test.pvalue)
            except:
                # Fallback: use p-value of t=0 coefficient
                joint_p = model.pvalues[_varname(0)]

            key = f"ES_{label}"
            if direction:  # to_right: expect avg_post < 0
                _ALL_RESULTS[key] = {
                    "coef": avg_post, "p": joint_p,
                    "supported": joint_p < 0.05 and avg_post < 0,
                    "n": len(trans_subset),
                }
            else:  # to_left: expect avg_post > 0
                _ALL_RESULTS[key] = {
                    "coef": avg_post, "p": joint_p,
                    "supported": joint_p < 0.05 and avg_post > 0,
                    "n": len(trans_subset),
                }

        except Exception as e:
            print(f"  {direction_desc}: Model estimation failed ({e})")

    return results


# =============================================================================
# H2: ALLIANCE INSTITUTIONALIZATION → DIVISION OF LABOR
# =============================================================================

def run_h2_regressions(paths: Paths) -> dict:
    """
    Run H2 regressions: Alliance type → Division of labor.
    """
    print(_box("H2: ALLIANCE TYPE → DIVISION OF LABOR"))
    print("""
  THEORY: More institutionalized alliances (hierarchical > voice-driven >
  uninstitutionalized) facilitate greater division of labor between partners.

  PREDICTIONS:
    H2A: Voice-driven alliances > Uninstitutionalized (β_voice > 0)
    H2B: Hierarchical alliances > Voice-driven (β_hier > β_voice)
""")

    results = {}
    df = pd.read_csv(paths.dyad_year_gannon_union_csv)

    # Sample description
    print(_subhead("SAMPLE: Gannon UNION (1980-2010)"))
    print(f"""
  Definition:    Dyad-years where partners share an ATOP pact OR DCAD agreement

  Total observations:     {len(df):,} dyad-years
  Unique dyads:           {df['dyad_id'].nunique():,} alliance partnerships
  Year range:             1980–2010
""")

    n_atop_only = ((df["any_atop_link"] == 1) & (df["any_dca_link"] != 1)).sum()
    n_dca_only = ((df["any_atop_link"] != 1) & (df["any_dca_link"] == 1)).sum()
    n_both = ((df["any_atop_link"] == 1) & (df["any_dca_link"] == 1)).sum()

    print(f"  Alignment breakdown:")
    print(f"    ATOP-only:    {n_atop_only:>8,} ({100*n_atop_only/len(df):.1f}%)")
    print(f"    DCAD-only:    {n_dca_only:>8,} ({100*n_dca_only/len(df):.1f}%)")
    print(f"    Both:         {n_both:>8,} ({100*n_both/len(df):.1f}%)")

    # -------------------------------------------------------------------------
    # ORDINAL SPECIFICATION
    # -------------------------------------------------------------------------
    print(_subhead("TEST 1: Ordinal Specification"))
    print("""
  What this tests: Does alliance depth predict division of labor?

  Design:
    • Outcome (DV):    div_labor = portfolio dissimilarity (0 = identical, 1 = fully different)
    • Predictor (IV):  vertical_integration = ordinal scale
                       0 = DCA-only (no formal treaty)
                       1 = Uninstitutionalized ATOP alliance
                       2 = Voice-driven ATOP alliance
                       3 = Hierarchical ATOP alliance
    • Fixed Effects:   Dyad + Decade
    • Standard Errors: Clustered by dyad
""")

    controls = get_available_controls(df, DYAD_CONTROLS)
    control_str = " + ".join(controls) if controls else "1"

    required = ["div_labor", "vertical_integration", "dyad_id", "decade"] + controls
    analysis = df.dropna(subset=required).copy()

    # Distribution
    vi_dist = analysis["vertical_integration"].value_counts().sort_index()
    labels = {0: "DCA-only (vi=0)", 1: "Uninstitutionalized (vi=1)",
              2: "Voice-driven (vi=2)", 3: "Hierarchical (vi=3)"}

    print("  Sample distribution by alliance type:")
    for vi, count in vi_dist.items():
        print(f"    {labels.get(int(vi), vi):<30} {count:>8,} ({100*count/len(analysis):>5.1f}%)")

    formula = f"div_labor ~ vertical_integration + {control_str} + C(dyad_id) + C(decade)"

    model = smf.ols(formula, data=analysis).fit(
        cov_type="cluster",
        cov_kwds={"groups": analysis["dyad_id"]},
    )

    coef = model.params["vertical_integration"]
    se = model.bse["vertical_integration"]
    pval = model.pvalues["vertical_integration"]

    print()
    print(f"  Sample size:     N = {int(model.nobs):,} dyad-years")
    print()
    print(f"  RESULT:          {_format_coef(coef, se, pval)}")
    print(f"  Interpretation:  Moving up one level of institutionalization is")
    print(f"                   associated with a {abs(coef):.4f} change in div_labor")
    print(f"  Significance:    {_sig_word(pval)}")

    results["ordinal"] = {
        "coefficient": coef, "se": se, "p_value": pval,
        "n": int(model.nobs), "r2": model.rsquared,
    }

    _ALL_RESULTS["H2_ordinal"] = {
        "coef": coef, "se": se, "p": pval, "n": int(model.nobs),
        "supported": pval < 0.05 and coef > 0,
        "direction": "positive" if coef > 0 else "negative",
    }

    # -------------------------------------------------------------------------
    # CATEGORICAL SPECIFICATION
    # -------------------------------------------------------------------------
    print(_subhead("TEST 2: Categorical Specification (H2A & H2B)"))
    print("""
  What this tests: Separate effects of each alliance type

  Design:
    • Predictors:  hierarchical, voice_driven, uninstitutionalized (dummies)
    • Reference:   DCA-only (informal defense cooperation, no formal treaty)
    • NOTE: Uninst ATOP and DCA-only are NOT pooled — they are distinct categories

  Expected:
    H2A: β_voice > β_uninst (voice-driven > uninstitutionalized)
    H2B: β_hier > β_voice (hierarchical > voice-driven)
""")

    analysis["hierarchical"] = (analysis["vertical_integration"] == 3).astype(int)
    analysis["voice_driven"] = (analysis["vertical_integration"] == 2).astype(int)
    analysis["uninstitutionalized"] = (analysis["vertical_integration"] == 1).astype(int)

    formula_cat = f"div_labor ~ hierarchical + voice_driven + uninstitutionalized + {control_str} + C(dyad_id) + C(decade)"

    try:
        model_cat = smf.ols(formula_cat, data=analysis).fit(
            cov_type="cluster",
            cov_kwds={"groups": analysis["dyad_id"]},
        )

        hier_coef = model_cat.params["hierarchical"]
        hier_se = model_cat.bse["hierarchical"]
        hier_p = model_cat.pvalues["hierarchical"]

        voice_coef = model_cat.params["voice_driven"]
        voice_se = model_cat.bse["voice_driven"]
        voice_p = model_cat.pvalues["voice_driven"]

        uninst_coef = model_cat.params["uninstitutionalized"]
        uninst_se = model_cat.bse["uninstitutionalized"]
        uninst_p = model_cat.pvalues["uninstitutionalized"]

        print(f"  Sample size:     N = {int(model_cat.nobs):,}")
        print()
        print(f"  RESULTS (vs. DCA-only reference):")
        print(f"    Uninstitutionalized: {_format_coef(uninst_coef, uninst_se, uninst_p)}")
        print(f"    Voice-driven:        {_format_coef(voice_coef, voice_se, voice_p)}")
        print(f"    Hierarchical:        {_format_coef(hier_coef, hier_se, hier_p)}")
        print()

        # H2A test: voice > uninst
        try:
            wald_h2a = model_cat.wald_test("voice_driven - uninstitutionalized = 0", scalar=True)
            p_h2a = float(wald_h2a.pvalue)
        except Exception:
            diff_h2a = voice_coef - uninst_coef
            se_diff_h2a = np.sqrt(voice_se**2 + uninst_se**2)
            t_stat_h2a = diff_h2a / se_diff_h2a
            p_h2a = 2 * (1 - scipy_stats.t.cdf(abs(t_stat_h2a), model_cat.df_resid))

        diff_h2a = voice_coef - uninst_coef
        print(f"  H2A TEST (voice-driven > uninstitutionalized):")
        print(f"    Difference:    Δβ = {diff_h2a:.4f} (voice - uninst)")
        print(f"    p-value:       p = {p_h2a:.4f} {_sig(p_h2a)}")
        print(f"    Result:        {_sig_word(p_h2a)}, {'supported' if p_h2a < 0.05 and diff_h2a > 0 else 'not supported'}")

        # H2B test: hier > voice
        try:
            wald_h2b = model_cat.wald_test("hierarchical - voice_driven = 0", scalar=True)
            p_h2b = float(wald_h2b.pvalue)
        except Exception:
            diff_h2b = hier_coef - voice_coef
            se_diff_h2b = np.sqrt(hier_se**2 + voice_se**2)
            t_stat_h2b = diff_h2b / se_diff_h2b
            p_h2b = 2 * (1 - scipy_stats.t.cdf(abs(t_stat_h2b), model_cat.df_resid))

        diff_h2b = hier_coef - voice_coef
        print()
        print(f"  H2B TEST (hierarchical > voice-driven):")
        print(f"    Difference:    Δβ = {diff_h2b:.4f} (hier - voice)")
        print(f"    p-value:       p = {p_h2b:.4f} {_sig(p_h2b)}")
        print(f"    Result:        {_sig_word(p_h2b)}, {'supported' if p_h2b < 0.05 and diff_h2b > 0 else 'not supported'}")

        results["categorical"] = {
            "hierarchical": hier_coef, "voice_driven": voice_coef, "uninstitutionalized": uninst_coef,
            "h2a_diff": diff_h2a, "h2a_p": p_h2a,
            "h2b_diff": diff_h2b, "h2b_p": p_h2b, "n": int(model_cat.nobs),
        }

        _ALL_RESULTS["H2A"] = {
            "coef": diff_h2a, "se": se_diff_h2a if 'se_diff_h2a' in dir() else np.nan, "p": p_h2a,
            "n": int(model_cat.nobs),
            "supported": p_h2a < 0.05 and diff_h2a > 0,
        }
        _ALL_RESULTS["H2B"] = {
            "coef": diff_h2b, "se": se_diff_h2b if 'se_diff_h2b' in dir() else np.nan, "p": p_h2b,
            "n": int(model_cat.nobs),
            "supported": p_h2b < 0.05 and diff_h2b > 0,
        }

    except np.linalg.LinAlgError:
        print("  ⚠ Model estimation failed due to multicollinearity")
        print("    This often occurs when dyad fixed effects absorb too much variation.")
        print("    The ordinal specification above provides the primary test.")
        results["categorical"] = None

    # Save
    results_df = pd.DataFrame([{
        "specification": "ordinal",
        "variable": "vertical_integration",
        **results["ordinal"],
    }])
    results_df.to_csv(paths.h2_dir / "model_h2_primary.csv", index=False)

    return results


# =============================================================================
# H3: IDEOLOGICAL SIMILARITY → DIVISION OF LABOR
# =============================================================================

def run_h3_regressions(paths: Paths) -> dict:
    """
    Run H3 regressions: Ideological similarity → Division of labor.
    """
    print(_box("H3: IDEOLOGICAL SIMILARITY → DIVISION OF LABOR"))
    print("""
  THEORY: Alliance partners with similar ideologies should find it easier
  to coordinate and specialize, leading to greater division of labor.

  PREDICTION: β < 0 (greater ideological distance → less division of labor)
              OR equivalently: similarity → more division of labor
""")

    results = {}
    df = pd.read_csv(paths.dyad_year_gannon_union_csv)

    print(_subhead("SAMPLE: Gannon UNION (1980-2010)"))
    print(f"""
  Total observations:     {len(df):,} dyad-years
  With ideology data:     {df['ideo_dist_lag5'].notna().sum():,} ({100*df['ideo_dist_lag5'].notna().mean():.1f}%)
""")

    # -------------------------------------------------------------------------
    # MINIMAL SPECIFICATION
    # -------------------------------------------------------------------------
    print(_subhead("TEST 1: Minimal Specification"))
    print("""
  What this tests: Does ideological distance predict division of labor?

  Design:
    • Outcome (DV):    div_labor = portfolio dissimilarity
    • Predictor (IV):  ideo_dist_lag5 = |RILE_a - RILE_b| (5-year lag)
                       Scale: 0 (identical ideology) to 200 (maximum distance)
    • Fixed Effects:   Dyad + Year
    • Standard Errors: Clustered by dyad
    • Controls:        None (minimal specification)
""")

    required = ["div_labor", "ideo_dist_lag5", "dyad_id", "year"]
    analysis = df.dropna(subset=required).copy()

    formula = "div_labor ~ ideo_dist_lag5 + C(dyad_id) + C(year)"

    model = smf.ols(formula, data=analysis).fit(
        cov_type="cluster",
        cov_kwds={"groups": analysis["dyad_id"]},
    )

    coef = model.params["ideo_dist_lag5"]
    se = model.bse["ideo_dist_lag5"]
    pval = model.pvalues["ideo_dist_lag5"]

    print(f"  Sample size:     N = {int(model.nobs):,} dyad-years")
    print(f"  Unique dyads:    {analysis['dyad_id'].nunique():,}")
    print()
    print(f"  RESULT:          {_format_coef(coef, se, pval)}")
    print(f"  Interpretation:  A 10-point increase in ideological distance is")
    print(f"                   associated with a {abs(coef * 10):.4f} change in div_labor")
    print(f"  Significance:    {_sig_word(pval)}")

    results["minimal"] = {
        "coefficient": coef, "se": se, "p_value": pval,
        "n": int(model.nobs), "r2": model.rsquared,
    }

    # -------------------------------------------------------------------------
    # FULL SPECIFICATION
    # -------------------------------------------------------------------------
    print(_subhead("TEST 2: Full Specification (with controls)"))

    controls = get_available_controls(df, DYAD_CONTROLS)
    print(f"""
  Same as above, but adding controls for confounders:
    • Controls: {', '.join(controls)}
""")

    required_full = required + controls
    analysis_full = df.dropna(subset=required_full).copy()

    control_str = " + ".join(controls)
    formula_full = f"div_labor ~ ideo_dist_lag5 + {control_str} + C(dyad_id) + C(year)"

    model_full = smf.ols(formula_full, data=analysis_full).fit(
        cov_type="cluster",
        cov_kwds={"groups": analysis_full["dyad_id"]},
    )

    coef_f = model_full.params["ideo_dist_lag5"]
    se_f = model_full.bse["ideo_dist_lag5"]
    pval_f = model_full.pvalues["ideo_dist_lag5"]

    print(f"  Sample size:     N = {int(model_full.nobs):,}")
    print()
    print(f"  RESULT:          {_format_coef(coef_f, se_f, pval_f)}")
    print(f"  Significance:    {_sig_word(pval_f)}")

    results["full"] = {
        "coefficient": coef_f, "se": se_f, "p_value": pval_f,
        "n": int(model_full.nobs), "r2": model_full.rsquared,
    }

    _ALL_RESULTS["H3"] = {
        "coef": coef_f, "se": se_f, "p": pval_f, "n": int(model_full.nobs),
        "supported": pval_f < 0.05,
        "direction": "negative" if coef_f < 0 else "positive",
    }

    # Save
    results_df = pd.DataFrame([
        {"specification": "minimal", **results["minimal"]},
        {"specification": "full", **results["full"]},
    ])
    results_df.to_csv(paths.h3_dir / "model_h3_primary.csv", index=False)

    return results


# =============================================================================
# SUMMARY TABLE
# =============================================================================

def print_summary_table() -> None:
    """Print a grand summary table of all hypothesis tests."""
    if not _ALL_RESULTS:
        return

    # Column widths (must sum to w when including separating spaces)
    col_test = 12
    col_question = 38
    col_coef = 9
    col_p = 7
    col_finding = 13
    # Total: 12 + 1 + 38 + 1 + 9 + 1 + 7 + 1 + 13 = 83, plus 2 for "| " and " |" = 87
    w = col_test + 1 + col_question + 1 + col_coef + 1 + col_p + 1 + col_finding

    print("\n")
    print("+" + "-" * (w + 2) + "+")
    print("|" + "HYPOTHESIS TEST SUMMARY".center(w + 2) + "|")
    print("+" + "-" * (w + 2) + "+")
    print("| {:<{}} {:<{}} {:>{}} {:>{}} {:>{}} |".format(
        "Test", col_test,
        "Question", col_question,
        "Coef", col_coef,
        "p-value", col_p,
        "Finding", col_finding))
    print("+" + "-" * (w + 2) + "+")

    # Define all tests with their predictions
    tests = [
        # (key, question)
        ("H1", "Right ideology -> less spec?"),
        ("ES_to_right", "L->R: spec decrease?"),
        ("ES_to_left", "R->L: spec increase?"),
        ("H2_ordinal", "Alliance depth -> div labor?"),
        ("H2A", "Voice > uninstitutionalized?"),
        ("H2B", "Hierarchical > voice?"),
        ("H3", "Ideo distance -> less div labor?"),
    ]

    for key, question in tests:
        if key in _ALL_RESULTS:
            r = _ALL_RESULTS[key]
            coef = r['coef']
            p = r['p']
            coef_str = f"{coef:+.4f}"
            p_str = f"{p:.3f}"

            # Determine finding based on significance and direction
            if p < 0.05:
                if r.get('supported', False):
                    finding = "SUPPORTED"
                else:
                    finding = "CONTRARY"
            elif p < 0.10:
                finding = "Marginal"
            else:
                finding = "No effect"

            print("| {:<{}} {:<{}} {:>{}} {:>{}} {:>{}} |".format(
                key, col_test,
                question[:col_question], col_question,
                coef_str, col_coef,
                p_str, col_p,
                finding, col_finding))
        else:
            # Test not available (e.g., due to multicollinearity)
            print("| {:<{}} {:<{}} {:>{}} {:>{}} {:>{}} |".format(
                key, col_test,
                question[:col_question], col_question,
                "-", col_coef,
                "-", col_p,
                "(not estim.)", col_finding))

    print("+" + "-" * (w + 2) + "+")
    legend = [
        "SUPPORTED = significant (p<0.05) in predicted direction",
        "CONTRARY = significant in opposite direction",
        "Marginal = p<0.10, No effect = not significant",
    ]
    for line in legend:
        print("| {:<{}} |".format(line, w))
    print("+" + "-" * (w + 2) + "+")
    print()


def get_all_results() -> dict:
    """Return all stored results for external use."""
    return _ALL_RESULTS.copy()
