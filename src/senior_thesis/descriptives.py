"""
Descriptive statistics and visualizations for thesis chapter.

Produces summary statistics tables and figures for:
- Country-year dataset (H1: ideology -> specialization)
- Dyad-year dataset (H2/H2A/H2B: alliance institutions -> specialization)
- Type-depth diagnostics (collinearity assessment)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

from senior_thesis.config import Paths, COUNTRY_CONTROLS, DYAD_CONTROLS, INST_LABELS


def _setup_style() -> None:
    """Configure matplotlib style for publication-quality figures."""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "figure.titlesize": 14,
        "legend.fontsize": 10,
    })


def _summary_table(df: pd.DataFrame, variables: list[str]) -> pd.DataFrame:
    """Compute summary statistics (N, Mean, SD, Min, Max) for variables."""
    available = [v for v in variables if v in df.columns]
    stats_df = df[available].describe().T[["count", "mean", "std", "min", "max"]]
    stats_df.columns = ["N", "Mean", "SD", "Min", "Max"]
    return stats_df.round(3)


def country_year_descriptives(paths: Paths) -> None:
    """Generate descriptive statistics and figures for country-year dataset."""
    print("\n[3a] Country-year descriptives...")

    df = pd.read_csv(paths.country_year_csv)
    _setup_style()

    # Summary statistics
    key_vars = ["spec_y", "rile", "right_of_center"] + COUNTRY_CONTROLS
    stats_df = _summary_table(df, key_vars)
    stats_df.to_csv(paths.tables_dir / "summary_country_year.csv")
    print(f"  Saved: summary_country_year.csv")
    print(stats_df.to_string())

    # Figure 1: RILE distribution
    fig, ax = plt.subplots(figsize=(7, 4.5))
    df["rile"].dropna().hist(bins=30, ax=ax, color="steelblue", edgecolor="white", alpha=0.8)
    ax.axvline(10, color="firebrick", linestyle="--", linewidth=1.5, label="Right (RILE=10)")
    ax.axvline(-10, color="navy", linestyle="--", linewidth=1.5, label="Left (RILE=-10)")
    ax.set_xlabel("RILE Score")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of Ruling Party Ideology")
    ax.legend(frameon=True, fancybox=False)
    plt.tight_layout()
    plt.savefig(paths.figures_dir / "fig1_rile_distribution.png", dpi=200)
    plt.close()

    # Figure 2: Specialization distribution
    fig, ax = plt.subplots(figsize=(7, 4.5))
    df["spec_y"].dropna().hist(bins=30, ax=ax, color="steelblue", edgecolor="white", alpha=0.8)
    ax.set_xlabel("Specialization Index (Standardized)")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of Military Specialization")
    plt.tight_layout()
    plt.savefig(paths.figures_dir / "fig2_specialization_distribution.png", dpi=200)
    plt.close()

    # Figure 3: Ideology vs Specialization scatter
    fig, ax = plt.subplots(figsize=(7, 5))
    valid = df.dropna(subset=["rile", "spec_y"])
    ax.scatter(valid["rile"], valid["spec_y"], alpha=0.25, s=15, color="steelblue")
    z = np.polyfit(valid["rile"], valid["spec_y"], 1)
    x_line = np.linspace(valid["rile"].min(), valid["rile"].max(), 100)
    ax.plot(x_line, np.polyval(z, x_line), color="firebrick", linewidth=2, label="Linear fit")
    ax.set_xlabel("RILE Score (Left-Right)")
    ax.set_ylabel("Specialization Index")
    ax.set_title("Ideology vs Military Specialization (H1 Preview)")
    ax.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(paths.figures_dir / "fig3_ideology_specialization.png", dpi=200)
    plt.close()

    # Figure 4: Time series
    fig, ax = plt.subplots(figsize=(8, 4))
    yearly = df.groupby("year")["spec_y"].mean()
    ax.plot(yearly.index, yearly.values, marker="o", markersize=4, color="steelblue")
    ax.set_xlabel("Year")
    ax.set_ylabel("Mean Specialization")
    ax.set_title("Average Military Specialization Over Time")
    plt.tight_layout()
    plt.savefig(paths.figures_dir / "fig4_specialization_timeseries.png", dpi=200)
    plt.close()

    print("  Saved: fig1-fig4 (country-year)")


def dyad_year_descriptives(paths: Paths) -> None:
    """Generate descriptive statistics and figures for dyad-year dataset."""
    print("\n[3b] Dyad-year descriptives...")

    df = pd.read_csv(paths.dyad_year_csv)
    _setup_style()

    # Summary statistics
    key_vars = ["spec_dyad_mean", "spec_a", "spec_b", "Depth.score", "inst",
                "hierarchical", "voice_driven", "rile_dyad_mean", "depth_within_type"] + DYAD_CONTROLS
    stats_df = _summary_table(df, key_vars)
    stats_df.to_csv(paths.tables_dir / "summary_dyad_year.csv")
    print(f"  Saved: summary_dyad_year.csv")
    print(stats_df.to_string())

    # Add institution labels
    df["inst_label"] = df["inst"].map(INST_LABELS)
    order = ["Uninstitutionalized", "Voice-Driven", "Hierarchical"]

    # Figure 5: Specialization by institution type
    fig, ax = plt.subplots(figsize=(7, 5))
    valid = df.dropna(subset=["spec_dyad_mean", "inst_label"])
    sns.boxplot(data=valid, x="inst_label", y="spec_dyad_mean", order=order, ax=ax,
                hue="inst_label", palette="Blues", legend=False)
    ax.set_xlabel("Alliance Institution Type")
    ax.set_ylabel("Mean Partner Specialization")
    ax.set_title("Partner Specialization by Alliance Type (H2A/H2B Preview)")
    plt.tight_layout()
    plt.savefig(paths.figures_dir / "fig5_specialization_by_inst.png", dpi=200)
    plt.close()

    # Figure 6: Depth vs Specialization scatter
    fig, ax = plt.subplots(figsize=(7, 5))
    valid = df.dropna(subset=["Depth.score", "spec_dyad_mean"])
    ax.scatter(valid["Depth.score"], valid["spec_dyad_mean"], alpha=0.25, s=15, color="steelblue")
    z = np.polyfit(valid["Depth.score"], valid["spec_dyad_mean"], 1)
    x_line = np.linspace(valid["Depth.score"].min(), valid["Depth.score"].max(), 100)
    ax.plot(x_line, np.polyval(z, x_line), color="firebrick", linewidth=2, label="Linear fit")
    ax.set_xlabel("Alliance Depth Score")
    ax.set_ylabel("Mean Partner Specialization")
    ax.set_title("Alliance Depth vs Partner Specialization (H2 Preview)")
    ax.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(paths.figures_dir / "fig6_depth_specialization.png", dpi=200)
    plt.close()

    # Figure 7: Depth by institution type (diagnostic)
    fig, ax = plt.subplots(figsize=(7, 5))
    valid = df.dropna(subset=["Depth.score", "inst_label"])
    sns.boxplot(data=valid, x="inst_label", y="Depth.score", order=order, ax=ax,
                hue="inst_label", palette="Oranges", legend=False)
    ax.set_xlabel("Alliance Institution Type")
    ax.set_ylabel("Depth Score")
    ax.set_title("Alliance Depth by Institution Type (Type-Depth Overlap)")
    plt.tight_layout()
    plt.savefig(paths.figures_dir / "fig7_depth_by_inst.png", dpi=200)
    plt.close()

    # Figure 8: Active dyads over time
    fig, ax = plt.subplots(figsize=(8, 4))
    yearly = df.groupby("year").size()
    ax.plot(yearly.index, yearly.values, marker="o", markersize=4, color="steelblue")
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of Active Dyad-Years")
    ax.set_title("Alliance Dyad Coverage Over Time")
    plt.tight_layout()
    plt.savefig(paths.figures_dir / "fig8_dyads_timeseries.png", dpi=200)
    plt.close()

    # Institution breakdown table
    breakdown = df.groupby("inst").agg(n_dyads=("atopid", "nunique"), n_dyad_years=("year", "count"))
    breakdown.index = [INST_LABELS.get(i, i) for i in breakdown.index]
    breakdown.to_csv(paths.tables_dir / "inst_breakdown.csv")
    print(f"  Saved: inst_breakdown.csv")
    print(breakdown)

    # Correlation matrix
    corr_vars = ["spec_dyad_mean", "Depth.score", "rile_dyad_mean", "hierarchical", "voice_driven"]
    corr = df[corr_vars].corr().round(3)
    corr.to_csv(paths.tables_dir / "correlation_matrix.csv")
    print(f"  Saved: correlation_matrix.csv")

    print("  Saved: fig5-fig8 (dyad-year)")


def type_depth_diagnostics(paths: Paths) -> pd.DataFrame:
    """
    Run type-depth collinearity diagnostics.

    Returns DataFrame with correlation, R², and VIF statistics.
    """
    print("\n[3c] Type-depth diagnostics...")

    df = pd.read_csv(paths.dyad_year_csv)
    complete = df.dropna(subset=["Depth.score", "inst", "spec_dyad_mean"])

    # 1. Correlation
    corr, pval = stats.pearsonr(complete["inst"], complete["Depth.score"])
    print(f"  Correlation (inst, Depth.score): r={corr:.3f}, p={pval:.2e}")

    # 2. R² from regressing Depth on type dummies
    complete = complete.copy()
    complete["hier_dum"] = (complete["inst"] == 3).astype(float)
    complete["voice_dum"] = (complete["inst"] == 2).astype(float)
    X = sm.add_constant(complete[["hier_dum", "voice_dum"]].values)
    y = complete["Depth.score"].values
    r2 = sm.OLS(y, X).fit().rsquared
    print(f"  R² (Depth ~ type dummies): {r2:.3f}")

    # 3. VIFs
    X_vif = complete[["Depth.score", "hier_dum", "voice_dum"]].values
    X_vif = np.column_stack([np.ones(X_vif.shape[0]), X_vif])
    vif_depth = variance_inflation_factor(X_vif, 1)
    vif_hier = variance_inflation_factor(X_vif, 2)
    vif_voice = variance_inflation_factor(X_vif, 3)
    print(f"  VIFs: Depth={vif_depth:.2f}, hierarchical={vif_hier:.2f}, voice={vif_voice:.2f}")

    # Decision
    max_vif = max(vif_depth, vif_hier, vif_voice)
    if r2 > 0.50 or max_vif > 10:
        status = "SUBSTANTIAL_OVERLAP"
        recommendation = "Use separate models as primary specifications"
    elif r2 < 0.30 and max_vif < 5:
        status = "MODEST_OVERLAP"
        recommendation = "Can include both in joint model"
    else:
        status = "MODERATE_OVERLAP"
        recommendation = "Use residualized depth for robustness"

    print(f"  Status: {status}")
    print(f"  Recommendation: {recommendation}")

    # Save diagnostics
    diag = pd.DataFrame({
        "metric": ["correlation_r", "correlation_p", "r_squared", "vif_depth", "vif_hierarchical", "vif_voice", "status"],
        "value": [f"{corr:.4f}", f"{pval:.2e}", f"{r2:.4f}", f"{vif_depth:.2f}", f"{vif_hier:.2f}", f"{vif_voice:.2f}", status]
    })
    diag.to_csv(paths.tables_dir / "type_depth_diagnostics.csv", index=False)
    print(f"  Saved: type_depth_diagnostics.csv")

    return diag


def run_all() -> None:
    """Generate all descriptive statistics and figures."""
    paths = Paths()

    # Ensure output directories exist
    paths.figures_dir.mkdir(parents=True, exist_ok=True)
    paths.tables_dir.mkdir(parents=True, exist_ok=True)

    country_year_descriptives(paths)
    dyad_year_descriptives(paths)
    type_depth_diagnostics(paths)

    print("\n[Done] All descriptives generated.")


if __name__ == "__main__":
    run_all()
