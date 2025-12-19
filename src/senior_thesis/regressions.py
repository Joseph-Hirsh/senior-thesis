"""Regression analyses for ideology and military specialization."""
from __future__ import annotations

import pandas as pd
import statsmodels.formula.api as smf

from senior_thesis.config import Paths, get_controls
from senior_thesis.data_prep import prep_master


def _build_formula(ideology_var: str) -> str:
    """Construct regression formula with fixed effects and controls."""
    predictors = [ideology_var] + get_controls() + ["C(country_code_cow)", "C(year)"]
    return "spec_y ~ " + " + ".join(predictors)


def run_lag_sensitivity() -> pd.DataFrame:
    """
    Run lag sensitivity analysis and return results as a DataFrame.

    Tests ideology lags from 1-20 years and shows coefficient, standard error,
    and significance for each lag specification.

    Returns:
        DataFrame with columns: Lag, N, Coef, SE, p-value, Sig
    """
    paths = Paths()
    df = prep_master(paths)

    all_controls = get_controls()

    # Run regressions and collect results
    results = []

    print("\n" + "="*80)
    print("LAG SENSITIVITY ANALYSIS: Right-of-Center Ideology (Lags 1-20)")
    print("Specification: Country FE + Year FE + Controls")
    print("="*80 + "\n")

    for lag in range(1, 21):
        analysis_data = df.dropna(
            subset=["spec_y", f"right_of_center_lag{lag}"] + all_controls + ["country_code_cow", "year"]
        )

        model = smf.ols(_build_formula(f"right_of_center_lag{lag}"), data=analysis_data)
        result = model.fit(cov_type="cluster", cov_kwds={"groups": analysis_data["country_code_cow"]})

        # Extract coefficient, standard error, and p-value
        ideology_var = f"right_of_center_lag{lag}"
        coef = result.params[ideology_var]
        se = result.bse[ideology_var]
        pval = result.pvalues[ideology_var]

        # Determine significance stars
        sig = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.10 else ""

        results.append({
            "Lag": lag,
            "N": int(result.nobs),
            "Coef": coef,
            "SE": se,
            "p-value": pval,
            "Sig": sig
        })

        print(f"Lag {lag:>2}: N={int(result.nobs):>4} | Coef={coef:>7.4f} | SE={se:>6.4f} | p={pval:>6.4f} {sig}")

    # Create summary DataFrame
    results_df = pd.DataFrame(results)

    # Save to CSV
    output_path = "results/lag_sensitivity.csv"
    results_df.to_csv(output_path, index=False)

    # Display summary
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    print(results_df.to_string(index=False))

    print("\nSignificance levels: *** p<0.01, ** p<0.05, * p<0.10")
    print("\nAll models include:")
    print("  - Country fixed effects")
    print("  - Year fixed effects")
    print("  - GDP (Log)")
    print("  - CINC")
    print("  - Interstate War (5 yr Lag)")
    print("  - Allies' Mil Spend (Log)")
    print("  - Allies' CINC Ratio")
    print("\nStandard errors clustered by country.")
    print(f"\n✓ Saved results to: {output_path}")

    return results_df


def main() -> None:
    """Run regression analyses from command line."""
    run_lag_sensitivity()


if __name__ == "__main__":
    main()
