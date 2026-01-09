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

    GANNON REPLICATION with EXPLICIT DEVIATIONS:
    - D1: Alliance governance is 3-category NOMINAL (not ordinal)
    - D2: SUBORD is classified as HIERARCHICAL
    - D3: Main contrasts are voice > uninst and hier > uninst (not vs DCA-only)
    - D4: MAIN uses ATOP-only sample; UNION is robustness only
    """
    print(_box("H2: ALLIANCE TYPE → DIVISION OF LABOR"))
    print("""
  ═══════════════════════════════════════════════════════════════════════════
  GANNON REPLICATION + EXPLICIT DEVIATIONS
  ═══════════════════════════════════════════════════════════════════════════

  THEORY: More institutionalized alliances facilitate greater division of
  labor between partners through coordination mechanisms.

  PREDICTIONS (per D3 - deviation from Gannon):
    H2A: Voice-driven > Uninstitutionalized (β_voice > 0 vs uninst reference)
    H2B: Hierarchical > Uninstitutionalized (β_hier > 0 vs uninst reference)

  SAMPLE (per D4 - deviation from Gannon):
    MAIN TESTS use ATOP-only offense/defense dyad-years (1980-2010)
    UNION sample (ATOP + DCAD) is robustness only
""")

    results = {}

    # =========================================================================
    # MAIN SPECIFICATION: ATOP-ONLY SAMPLE
    # =========================================================================
    print(_subhead("MAIN: ATOP-Only Sample (1980-2010)"))

    # Load ATOP-only dataset (NOT union!)
    df_main = pd.read_csv(paths.dyad_year_gannon_csv)

    print(f"""
  Definition:    Dyad-years where partners share ATOP offense/defense pact
                 (excludes DCA-only dyad-years for main test per D4)

  Total observations:     {len(df_main):,} dyad-years
  Unique dyads:           {df_main['dyad_id'].nunique():,} alliance partnerships
  Year range:             {int(df_main['year'].min())}–{int(df_main['year'].max())}
""")

    # Recode inst for ATOP-only: 1/2/3 → 0/1/2 (uninst=0, voice=1, hier=2)
    # This makes uninst the reference category
    df_main["inst_012"] = df_main["inst"] - 1  # 1→0, 2→1, 3→2

    # Sample distribution
    inst_dist = df_main["inst_012"].value_counts().sort_index()
    labels_012 = {0: "Uninstitutionalized (inst_012=0)",
                  1: "Voice-driven (inst_012=1)",
                  2: "Hierarchical (inst_012=2)"}

    print("  Sample distribution by alliance type:")
    for inst_val, count in inst_dist.items():
        print(f"    {labels_012.get(int(inst_val), f'Type {inst_val}'):<35} {count:>8,} ({100*count/len(df_main):>5.1f}%)")

    # Get available controls (using gdp_ratio as PRIMARY per Gannon)
    controls = get_available_controls(df_main, DYAD_CONTROLS)
    control_str = " + ".join(controls) if controls else "1"
    print(f"\n  Controls: {', '.join(controls) if controls else 'none'}")

    # -------------------------------------------------------------------------
    # TEST 1: CATEGORICAL SPECIFICATION (PRIMARY - per D1/D3)
    # -------------------------------------------------------------------------
    print(_subhead("TEST 1: Categorical Specification (PRIMARY)"))
    print("""
  Per D1 (deviation): Alliance governance is 3-category NOMINAL, not ordinal.
  Per D3 (deviation): Reference category is UNINSTITUTIONALIZED (not DCA-only).

  Design:
    • Outcome (DV):    div_labor = portfolio dissimilarity (0 = identical, 1 = fully different)
    • Predictors:      voice (inst_012=1), hier (inst_012=2) as dummies
    • Reference:       Uninstitutionalized ATOP alliance (inst_012=0)
    • Fixed Effects:   Dyad + Decade
    • Standard Errors: Clustered by dyad

  Tests:
    H2A: β_voice > 0 (voice-driven > uninstitutionalized)
    H2B: β_hier > 0 (hierarchical > uninstitutionalized)
""")

    # Create dummies with uninst as reference
    df_main["voice"] = (df_main["inst_012"] == 1).astype(int)
    df_main["hier"] = (df_main["inst_012"] == 2).astype(int)

    required = ["div_labor", "inst_012", "dyad_id", "decade"] + controls
    analysis = df_main.dropna(subset=required).copy()

    formula_cat = f"div_labor ~ voice + hier + {control_str} + C(dyad_id) + C(decade)"

    try:
        model_cat = smf.ols(formula_cat, data=analysis).fit(
            cov_type="cluster",
            cov_kwds={"groups": analysis["dyad_id"]},
        )

        voice_coef = model_cat.params["voice"]
        voice_se = model_cat.bse["voice"]
        voice_p = model_cat.pvalues["voice"]

        hier_coef = model_cat.params["hier"]
        hier_se = model_cat.bse["hier"]
        hier_p = model_cat.pvalues["hier"]

        print(f"  Sample size:     N = {int(model_cat.nobs):,} dyad-years")
        print(f"  Unique dyads:    {analysis['dyad_id'].nunique():,}")
        print()
        print(f"  RESULTS (vs. Uninstitutionalized reference):")
        print(f"    Voice-driven:   {_format_coef(voice_coef, voice_se, voice_p)}")
        print(f"    Hierarchical:   {_format_coef(hier_coef, hier_se, hier_p)}")
        print()

        # H2A test: voice > uninst (direct coefficient test)
        print(f"  H2A TEST (voice-driven > uninstitutionalized):")
        print(f"    Coefficient:   β = {voice_coef:.4f} (SE = {voice_se:.4f})")
        print(f"    p-value:       p = {voice_p:.4f} {_sig(voice_p)}")
        supported_h2a = voice_p < 0.05 and voice_coef > 0
        print(f"    Result:        {_sig_word(voice_p)}, {'SUPPORTED' if supported_h2a else 'not supported'}")

        # H2B test: hier > uninst (direct coefficient test)
        print()
        print(f"  H2B TEST (hierarchical > uninstitutionalized):")
        print(f"    Coefficient:   β = {hier_coef:.4f} (SE = {hier_se:.4f})")
        print(f"    p-value:       p = {hier_p:.4f} {_sig(hier_p)}")
        supported_h2b = hier_p < 0.05 and hier_coef > 0
        print(f"    Result:        {_sig_word(hier_p)}, {'SUPPORTED' if supported_h2b else 'not supported'}")

        # Additional: hier vs voice comparison (using proper covariance-based SE)
        print()
        print(f"  SUPPLEMENTARY: Hierarchical vs Voice-driven:")
        try:
            wald_hier_voice = model_cat.wald_test("hier - voice = 0", scalar=True)
            p_hier_voice = float(wald_hier_voice.pvalue)
        except Exception:
            # Fallback: compute from covariance matrix
            cov_matrix = model_cat.cov_params()
            var_diff = (cov_matrix.loc["hier", "hier"] + cov_matrix.loc["voice", "voice"]
                       - 2 * cov_matrix.loc["hier", "voice"])
            se_diff = np.sqrt(var_diff) if var_diff > 0 else np.nan
            diff_hier_voice = hier_coef - voice_coef
            if not np.isnan(se_diff):
                t_stat = diff_hier_voice / se_diff
                p_hier_voice = 2 * (1 - scipy_stats.t.cdf(abs(t_stat), model_cat.df_resid))
            else:
                p_hier_voice = np.nan

        diff_hier_voice = hier_coef - voice_coef
        print(f"    Difference:    Δβ = {diff_hier_voice:.4f} (hier - voice)")
        print(f"    p-value:       p = {p_hier_voice:.4f} {_sig(p_hier_voice)}")

        results["categorical_main"] = {
            "voice": voice_coef, "voice_se": voice_se, "voice_p": voice_p,
            "hier": hier_coef, "hier_se": hier_se, "hier_p": hier_p,
            "hier_vs_voice_diff": diff_hier_voice, "hier_vs_voice_p": p_hier_voice,
            "n": int(model_cat.nobs), "n_dyads": analysis['dyad_id'].nunique(),
            "r2": model_cat.rsquared,
        }

        _ALL_RESULTS["H2A"] = {
            "coef": voice_coef, "se": voice_se, "p": voice_p,
            "n": int(model_cat.nobs),
            "supported": supported_h2a,
        }
        _ALL_RESULTS["H2B"] = {
            "coef": hier_coef, "se": hier_se, "p": hier_p,
            "n": int(model_cat.nobs),
            "supported": supported_h2b,
        }

    except np.linalg.LinAlgError as e:
        print(f"  ⚠ Model estimation failed: {e}")
        print("    This often occurs when dyad fixed effects absorb too much variation.")
        results["categorical_main"] = None

    # -------------------------------------------------------------------------
    # TEST 2: ORDINAL SPECIFICATION (SUPPLEMENTARY - per D1)
    # -------------------------------------------------------------------------
    print(_subhead("TEST 2: Ordinal Specification (SUPPLEMENTARY)"))
    print("""
  Per D1 (deviation): This ordinal test is SUPPLEMENTARY only.
  The categorical test above is the primary theory test.

  Design:
    • Predictor (IV):  inst_012 = ordinal scale (0=uninst, 1=voice, 2=hier)
    • Interpretation:  Effect of moving up one institutionalization level
""")

    formula_ord = f"div_labor ~ inst_012 + {control_str} + C(dyad_id) + C(decade)"

    try:
        model_ord = smf.ols(formula_ord, data=analysis).fit(
            cov_type="cluster",
            cov_kwds={"groups": analysis["dyad_id"]},
        )

        coef = model_ord.params["inst_012"]
        se = model_ord.bse["inst_012"]
        pval = model_ord.pvalues["inst_012"]

        print(f"  Sample size:     N = {int(model_ord.nobs):,}")
        print()
        print(f"  RESULT:          {_format_coef(coef, se, pval)}")
        print(f"  Interpretation:  Moving up one institutionalization level is")
        print(f"                   associated with a {abs(coef):.4f} change in div_labor")
        print(f"  Significance:    {_sig_word(pval)}")

        results["ordinal_main"] = {
            "coefficient": coef, "se": se, "p_value": pval,
            "n": int(model_ord.nobs), "r2": model_ord.rsquared,
        }

        _ALL_RESULTS["H2_ordinal"] = {
            "coef": coef, "se": se, "p": pval, "n": int(model_ord.nobs),
            "supported": pval < 0.05 and coef > 0,
            "direction": "positive" if coef > 0 else "negative",
        }

    except np.linalg.LinAlgError as e:
        print(f"  ⚠ Model estimation failed: {e}")
        results["ordinal_main"] = None

    # =========================================================================
    # ROBUSTNESS: UNION SAMPLE (per D4)
    # =========================================================================
    print(_subhead("ROBUSTNESS: UNION Sample (ATOP + DCAD)"))
    print("""
  Per D4 (deviation): UNION sample is for ROBUSTNESS only.
  DCA-only dyad-years are included but kept as a DISTINCT category (not pooled).
""")

    # Load UNION dataset
    df_union = pd.read_csv(paths.dyad_year_gannon_union_csv)

    n_atop_only = ((df_union["any_atop_link"] == 1) & (df_union["any_dca_link"] != 1)).sum()
    n_dca_only = ((df_union["any_atop_link"] != 1) & (df_union["any_dca_link"] == 1)).sum()
    n_both = ((df_union["any_atop_link"] == 1) & (df_union["any_dca_link"] == 1)).sum()

    print(f"  UNION sample: {len(df_union):,} dyad-years")
    print(f"    ATOP-only:    {n_atop_only:>8,} ({100*n_atop_only/len(df_union):.1f}%)")
    print(f"    DCA-only:     {n_dca_only:>8,} ({100*n_dca_only/len(df_union):.1f}%)")
    print(f"    Both:         {n_both:>8,} ({100*n_both/len(df_union):.1f}%)")

    # For UNION: use full 4-category specification
    # Create dummies: uninst, voice, hier (ref = DCA-only = vi==0)
    df_union["uninst"] = (df_union["vertical_integration"] == 1).astype(int)
    df_union["voice"] = (df_union["vertical_integration"] == 2).astype(int)
    df_union["hier"] = (df_union["vertical_integration"] == 3).astype(int)

    controls_union = get_available_controls(df_union, DYAD_CONTROLS)
    control_str_union = " + ".join(controls_union) if controls_union else "1"

    required_union = ["div_labor", "vertical_integration", "dyad_id", "decade"] + controls_union
    analysis_union = df_union.dropna(subset=required_union).copy()

    formula_union = f"div_labor ~ uninst + voice + hier + {control_str_union} + C(dyad_id) + C(decade)"

    try:
        model_union = smf.ols(formula_union, data=analysis_union).fit(
            cov_type="cluster",
            cov_kwds={"groups": analysis_union["dyad_id"]},
        )

        print(f"\n  UNION categorical results (vs. DCA-only reference):")
        print(f"  Sample size:     N = {int(model_union.nobs):,}")
        for var in ["uninst", "voice", "hier"]:
            c = model_union.params[var]
            s = model_union.bse[var]
            p = model_union.pvalues[var]
            print(f"    {var:<12}: {_format_coef(c, s, p)}")

        results["categorical_union"] = {
            var: {"coef": model_union.params[var], "se": model_union.bse[var], "p": model_union.pvalues[var]}
            for var in ["uninst", "voice", "hier"]
        }
        results["categorical_union"]["n"] = int(model_union.nobs)

    except np.linalg.LinAlgError as e:
        print(f"  ⚠ UNION model estimation failed: {e}")
        results["categorical_union"] = None

    # =========================================================================
    # SAVE RESULTS
    # =========================================================================
    # Save main results
    if results.get("categorical_main"):
        main_df = pd.DataFrame([{
            "specification": "categorical_main_ATOP_only",
            "reference": "uninstitutionalized",
            **{f"voice_{k}": v for k, v in [("coef", results["categorical_main"]["voice"]),
                                             ("se", results["categorical_main"]["voice_se"]),
                                             ("p", results["categorical_main"]["voice_p"])]},
            **{f"hier_{k}": v for k, v in [("coef", results["categorical_main"]["hier"]),
                                            ("se", results["categorical_main"]["hier_se"]),
                                            ("p", results["categorical_main"]["hier_p"])]},
            "n": results["categorical_main"]["n"],
            "n_dyads": results["categorical_main"]["n_dyads"],
            "r2": results["categorical_main"]["r2"],
        }])
        main_df.to_csv(paths.h2_dir / "model_h2_primary.csv", index=False)

    # Save sample IDs for reproducibility
    if analysis is not None and len(analysis) > 0:
        sample_ids = analysis[["dyad_id", "year"]].drop_duplicates()
        sample_ids.to_csv(paths.h2_dir / "h2_main_sample_ids.csv", index=False)
        print(f"\n  Saved sample IDs: {paths.h2_dir / 'h2_main_sample_ids.csv'}")

    # =========================================================================
    # DIAGNOSTIC: PLACEBO TEST (Future inst predicting current div_labor)
    # =========================================================================
    print(_subhead("DIAGNOSTIC: Placebo Test (Falsification)"))
    print("""
  What this tests: Does FUTURE institutionalization predict CURRENT div_labor?
  If yes, this would suggest confounding or reverse causality.
  Expected: No significant effect.
""")

    try:
        # Create lead institutionalization (1 year ahead)
        df_main_sorted = df_main.sort_values(["dyad_id", "year"])
        df_main_sorted["inst_012_lead1"] = df_main_sorted.groupby("dyad_id")["inst_012"].shift(-1)

        placebo_required = ["div_labor", "inst_012_lead1", "dyad_id", "decade"] + controls
        placebo_analysis = df_main_sorted.dropna(subset=placebo_required).copy()

        if len(placebo_analysis) > 100:
            formula_placebo = f"div_labor ~ inst_012_lead1 + {control_str} + C(dyad_id) + C(decade)"
            model_placebo = smf.ols(formula_placebo, data=placebo_analysis).fit(
                cov_type="cluster",
                cov_kwds={"groups": placebo_analysis["dyad_id"]},
            )

            coef_p = model_placebo.params["inst_012_lead1"]
            se_p = model_placebo.bse["inst_012_lead1"]
            pval_p = model_placebo.pvalues["inst_012_lead1"]

            print(f"  Sample size:     N = {int(model_placebo.nobs):,}")
            print(f"  RESULT:          {_format_coef(coef_p, se_p, pval_p)}")

            if pval_p >= 0.10:
                print(f"  Verdict:         ✓ PASSED — Future inst does not predict current div_labor")
            else:
                print(f"  Verdict:         ⚠ CONCERN — Possible confounding/reverse causality")

            results["placebo"] = {"coefficient": coef_p, "se": se_p, "p_value": pval_p}
        else:
            print(f"  ⚠ Insufficient observations for placebo test ({len(placebo_analysis)})")
    except Exception as e:
        print(f"  ⚠ Placebo test failed: {e}")

    # =========================================================================
    # HETEROGENEITY: Capability Asymmetry
    # =========================================================================
    print(_subhead("HETEROGENEITY: By Capability Asymmetry"))
    print("""
  Does institutionalization matter MORE when partners are capability-unequal?
  Split sample by milex_ratio (median split): High parity vs Low parity.
""")

    try:
        if "milex_ratio" in analysis.columns:
            median_milex = analysis["milex_ratio"].median()
            analysis["high_parity"] = (analysis["milex_ratio"] >= median_milex).astype(int)

            for parity_val, parity_label in [(1, "High Parity (similar capabilities)"),
                                              (0, "Low Parity (asymmetric capabilities)")]:
                sub = analysis[analysis["high_parity"] == parity_val]
                if len(sub) > 100:
                    model_het = smf.ols(formula_cat, data=sub).fit(
                        cov_type="cluster",
                        cov_kwds={"groups": sub["dyad_id"]},
                    )
                    voice_c = model_het.params.get("voice", np.nan)
                    hier_c = model_het.params.get("hier", np.nan)
                    print(f"  {parity_label}:")
                    print(f"    N = {int(model_het.nobs):,}, Voice β = {voice_c:.4f}, Hier β = {hier_c:.4f}")
                else:
                    print(f"  {parity_label}: Insufficient N ({len(sub)})")

            results["heterogeneity_parity"] = {"median_milex": median_milex}
        else:
            print(f"  ⚠ milex_ratio not available for heterogeneity analysis")
    except Exception as e:
        print(f"  ⚠ Heterogeneity analysis failed: {e}")

    # =========================================================================
    # REPLICATION CHECKLIST
    # =========================================================================
    print(_subhead("REPLICATION CHECKLIST"))
    print(f"""
  ═══════════════════════════════════════════════════════════════════════════
  GANNON REPLICATION STATUS
  ═══════════════════════════════════════════════════════════════════════════

  ✓ Sample: ATOP-only offense/defense dyads (D4)
  ✓ Window: 1980-2010 (DCAD aligned)
  ✓ N = {len(analysis):,} dyad-years, {analysis['dyad_id'].nunique():,} dyads
  ✓ Inst coding: Leeds & Anac (2005) from ATOP provisions
  ✓ SUBORD classified as hierarchical (D2)
  ✓ Reference category: Uninstitutionalized (D3)
  ✓ Governance as nominal 3-category (D1)
  ✓ Controls: {', '.join(controls) if controls else 'none'} (gdp_ratio on LEVELS)
  ✓ FE: Dyad + Decade
  ✓ SE: Clustered by dyad

  EXPLICIT DEVIATIONS FROM GANNON:
    D1: Alliance governance treated as 3-category NOMINAL (not ordinal)
    D2: SUBORD classified as HIERARCHICAL
    D3: Reference category is UNINSTITUTIONALIZED (not DCA-only)
    D4: MAIN tests use ATOP-only; UNION is robustness only
""")

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
        ("H2_ordinal", "Inst depth -> div labor? (suppl.)"),
        ("H2A", "Voice > uninst? (vs uninst ref)"),
        ("H2B", "Hier > uninst? (vs uninst ref)"),
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
