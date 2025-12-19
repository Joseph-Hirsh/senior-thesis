"""Visualize how ideology effects on specialization change across lags."""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.formula.api as smf

from senior_thesis.config import Paths
from senior_thesis.analysis import _prep_master


def plot_lag_effects(max_lag: int = 20, include_lag0: bool = True) -> None:
    """
    Plot coefficient estimates across different lag specifications.

    Args:
        max_lag: Maximum lag to test (default 20)
        include_lag0: Whether to include contemporary (t=0) ideology
    """
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

    # Start from lag 0 if requested, else lag 1
    start_lag = 0 if include_lag0 else 1

    print("\n" + "="*80)
    print(f"Estimating ideology effects for lags {start_lag}-{max_lag}...")
    print("="*80 + "\n")

    for lag in range(start_lag, max_lag + 1):
        # Handle lag 0 (contemporary) vs lagged ideology
        if lag == 0:
            ideo_var = "right_of_center"
        else:
            ideo_var = f"right_of_center_lag{lag}"

        # Drop missing values
        df_lag = df.dropna(subset=["spec_y", ideo_var] + all_controls + ["country_code_cow", "year"])

        # Run regression
        model = smf.ols(fml(ideo_var), data=df_lag)
        res = model.fit(cov_type="cluster", cov_kwds={"groups": df_lag["country_code_cow"]})

        # Extract key statistics
        coef = res.params[ideo_var]
        se = res.bse[ideo_var]
        pval = res.pvalues[ideo_var]

        # 95% confidence interval
        ci_lower = coef - 1.96 * se
        ci_upper = coef + 1.96 * se

        results.append({
            "lag": lag,
            "coef": coef,
            "se": se,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "pval": pval,
            "n": int(res.nobs)
        })

        sig = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.10 else ""
        print(f"Lag {lag:>2}: N={int(res.nobs):>4} | Coef={coef:>7.4f} | SE={se:>6.4f} | p={pval:>6.4f} {sig}")

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 7))

    # Plot coefficient point estimates
    ax.plot(results_df["lag"], results_df["coef"],
            marker='o', linewidth=2, markersize=6,
            color='#2E86AB', label='Coefficient estimate')

    # Add 95% confidence interval shading
    ax.fill_between(results_df["lag"],
                     results_df["ci_lower"],
                     results_df["ci_upper"],
                     alpha=0.3, color='#2E86AB', label='95% CI')

    # Add horizontal line at zero
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.7)

    # Highlight significant effects (p < 0.05)
    sig_results = results_df[results_df["pval"] < 0.05]
    if not sig_results.empty:
        ax.scatter(sig_results["lag"], sig_results["coef"],
                  color='red', s=100, zorder=5, marker='*',
                  label='Significant at p<0.05')

    # Labels and title
    ax.set_xlabel("Years Since Party Acquired Power (Lag)", fontsize=12, fontweight='bold')
    ax.set_ylabel("Effect on Trade Specialization", fontsize=12, fontweight='bold')
    ax.set_title("Dynamic Effect of Right-Wing Ideology on Trade Specialization\n" +
                 "How the relationship evolves over time after a party comes to power",
                 fontsize=14, fontweight='bold', pad=20)

    # Grid and legend
    ax.grid(True, alpha=0.3, linestyle=':')
    ax.legend(loc='best', frameon=True, shadow=True)

    # Set x-axis to show integer lags
    ax.set_xticks(range(start_lag, max_lag + 1, 2))

    plt.tight_layout()

    # Save figure
    output_path = "data/lag_effects_plot.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Plot saved to: {output_path}")

    # Also save results table
    csv_path = "data/lag_effects_results.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"✓ Results table saved to: {csv_path}")

    # Close the plot to free memory
    plt.close()

    # Print summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    print(f"\nPeak effect at lag {results_df.loc[results_df['coef'].abs().idxmax(), 'lag']:.0f}")
    print(f"Peak coefficient: {results_df['coef'].abs().max():.4f}")
    print(f"\nNumber of significant effects (p<0.05): {(results_df['pval'] < 0.05).sum()}/{len(results_df)}")
    print(f"Number of significant effects (p<0.10): {(results_df['pval'] < 0.10).sum()}/{len(results_df)}")


if __name__ == "__main__":
    plot_lag_effects(max_lag=20, include_lag0=True)
