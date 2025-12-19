"""Simplified regression output showing only key coefficients."""
from __future__ import annotations

import pandas as pd
import statsmodels.formula.api as smf

from senior_thesis.config import Paths
from senior_thesis.analysis import _prep_master


def run_simple_regressions() -> None:
    """Run lag sensitivity regressions and display key results in a table."""
    paths = Paths()
    df = _prep_master(paths)

    # Controls
    base_controls = ["lngdp", "cinc", "war5_lag"]
    alliance_controls = ["ln_milex_allies", "cinc_allies_ratio"]
    all_controls = base_controls + alliance_controls

    # Formula helper
    def fml(ideo: str) -> str:
        rhs = [ideo] + all_controls + ["C(country_code_cow)", "C(year)"]
        return "spec_y ~ " + " + ".join(rhs)

    # Store results
    results = []

    print("\n" + "="*80)
    print("LAG SENSITIVITY ANALYSIS: Binary Right-of-Center Ideology (Lags 1-20)")
    print("Specification: Year FE + GDP (Log) + CINC + War (5yr lag) + Allies' Mil Spend (Log) + Allies' CINC Ratio")
    print("="*80 + "\n")

    for lag in range(1, 21):
        df_lag = df.dropna(subset=["spec_y", f"right_of_center_lag{lag}"] + all_controls + ["country_code_cow", "year"])

        model = smf.ols(fml(f"right_of_center_lag{lag}"), data=df_lag)
        res = model.fit(cov_type="cluster", cov_kwds={"groups": df_lag["country_code_cow"]})

        # Extract key coefficient
        coef = res.params[f"right_of_center_lag{lag}"]
        se = res.bse[f"right_of_center_lag{lag}"]
        pval = res.pvalues[f"right_of_center_lag{lag}"]

        results.append({
            "Lag": lag,
            "N": int(res.nobs),
            "Coef": coef,
            "SE": se,
            "p-value": pval,
            "Sig": "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.10 else ""
        })

        print(f"Lag {lag}: N={int(res.nobs):>4} | Coef={coef:>7.4f} | SE={se:>6.4f} | p={pval:>6.4f} {results[-1]['Sig']}")

    # Create summary table
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)

    results_df = pd.DataFrame(results)
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


if __name__ == "__main__":
    run_simple_regressions()
