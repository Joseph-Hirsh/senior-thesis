"""
Descriptive statistics and visualizations for thesis chapter.

Produces summary statistics tables and figures for:
- H1: Ideology -> Specialization (country-year)
- H2/H2A/H2B: Alliance institutions -> Specialization (dyad-year)
"""
from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor

from senior_thesis.config import (
    Paths,
    COUNTRY_CONTROLS,
    DYAD_CONTROLS,
    INST_LABELS,
    FIGURE_DPI,
    RILE_RIGHT_THRESHOLD,
    RILE_LEFT_THRESHOLD,
    load_dataset,
    setup_logging,
)

__all__ = [
    "h1_descriptives",
    "h2_descriptives",
    "run_all",
]

logger = logging.getLogger("senior_thesis")


def _setup_style() -> None:
    """Configure matplotlib style for publication-quality figures."""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "figure.titlesize": 14,
            "legend.fontsize": 10,
        }
    )


@contextmanager
def _save_figure(
    path: Path,
    figsize: tuple[float, float] = (7, 5),
    dpi: int = FIGURE_DPI,
) -> Generator[tuple[Figure, Axes], None, None]:
    """
    Context manager for figure creation and saving.

    Args:
        path: Output path for the figure
        figsize: Figure size in inches (width, height)
        dpi: Resolution for saved figure

    Yields:
        Tuple of (Figure, Axes) for plotting

    Example:
        with _save_figure(out / "plot.png") as (fig, ax):
            ax.plot(x, y)
            ax.set_xlabel("X")
    """
    fig, ax = plt.subplots(figsize=figsize)
    try:
        yield fig, ax
    finally:
        plt.tight_layout()
        plt.savefig(path, dpi=dpi)
        plt.close(fig)


def _summary_table(df: pd.DataFrame, variables: list[str]) -> pd.DataFrame:
    """Compute summary statistics (N, Mean, SD, Min, Max) for variables."""
    available = [v for v in variables if v in df.columns]
    stats_df = df[available].describe().T[["count", "mean", "std", "min", "max"]]
    stats_df.columns = ["N", "Mean", "SD", "Min", "Max"]
    return stats_df.round(3)


def h1_descriptives(paths: Paths) -> None:
    """
    Generate descriptive statistics and figures for H1.

    Outputs to results/h1/:
    - summary_stats.csv
    - rile_distribution.png
    - specialization_distribution.png
    - ideology_vs_specialization.png
    - specialization_timeseries.png
    """
    logger.info("[H1] Generating descriptives...")

    df = load_dataset(paths.country_year_csv)
    _setup_style()
    out = paths.h1_dir

    # Summary statistics
    key_vars = ["spec_y", "rile", "right_of_center"] + COUNTRY_CONTROLS
    stats_df = _summary_table(df, key_vars)
    stats_df.to_csv(out / "summary_stats.csv")
    logger.info("Saved: summary_stats.csv")
    logger.info(f"\n{stats_df.to_string()}")

    # Figure 1: RILE distribution (using config thresholds)
    with _save_figure(out / "rile_distribution.png", figsize=(7, 4.5)) as (fig, ax):
        df["rile"].dropna().hist(
            bins=30, ax=ax, color="steelblue", edgecolor="white", alpha=0.8
        )
        ax.axvline(
            RILE_RIGHT_THRESHOLD,
            color="firebrick",
            linestyle="--",
            linewidth=1.5,
            label=f"Right (RILE={RILE_RIGHT_THRESHOLD:.0f})",
        )
        ax.axvline(
            RILE_LEFT_THRESHOLD,
            color="navy",
            linestyle="--",
            linewidth=1.5,
            label=f"Left (RILE={RILE_LEFT_THRESHOLD:.0f})",
        )
        ax.set_xlabel("RILE Score")
        ax.set_ylabel("Frequency")
        ax.set_title("Distribution of Ruling Party Ideology")
        ax.legend(frameon=True, fancybox=False)

    # Figure 2: Specialization distribution
    with _save_figure(out / "specialization_distribution.png", figsize=(7, 4.5)) as (
        fig,
        ax,
    ):
        df["spec_y"].dropna().hist(
            bins=30, ax=ax, color="steelblue", edgecolor="white", alpha=0.8
        )
        ax.set_xlabel("Specialization Index (Standardized)")
        ax.set_ylabel("Frequency")
        ax.set_title("Distribution of Military Specialization")

    # Figure 3: Ideology vs Specialization scatter
    with _save_figure(out / "ideology_vs_specialization.png") as (fig, ax):
        valid = df.dropna(subset=["rile", "spec_y"])
        ax.scatter(valid["rile"], valid["spec_y"], alpha=0.25, s=15, color="steelblue")
        z = np.polyfit(valid["rile"], valid["spec_y"], 1)
        x_line = np.linspace(valid["rile"].min(), valid["rile"].max(), 100)
        ax.plot(
            x_line, np.polyval(z, x_line), color="firebrick", linewidth=2, label="Linear fit"
        )
        ax.set_xlabel("RILE Score (Left-Right)")
        ax.set_ylabel("Specialization Index")
        ax.set_title("Ideology vs Military Specialization")
        ax.legend(frameon=True)

    # Figure 4: Time series
    with _save_figure(out / "specialization_timeseries.png", figsize=(8, 4)) as (
        fig,
        ax,
    ):
        yearly = df.groupby("year")["spec_y"].mean()
        ax.plot(yearly.index, yearly.values, marker="o", markersize=4, color="steelblue")
        ax.set_xlabel("Year")
        ax.set_ylabel("Mean Specialization")
        ax.set_title("Average Military Specialization Over Time")

    logger.info("Saved: 4 figures")


def h2_descriptives(paths: Paths) -> None:
    """
    Generate descriptive statistics and figures for H2/H2A/H2B.

    Outputs to results/h2/:
    - summary_stats.csv
    - depth_vs_specialization.png
    - specialization_by_inst.png
    - depth_by_inst.png
    - dyads_timeseries.png
    - inst_breakdown.csv
    - correlation_matrix.csv
    - type_depth_diagnostics.csv
    """
    logger.info("[H2] Generating descriptives...")

    df = load_dataset(paths.dyad_year_csv)
    _setup_style()
    out = paths.h2_dir

    # Summary statistics
    key_vars = [
        "spec_dyad_mean",
        "spec_a",
        "spec_b",
        "Depth.score",
        "inst",
        "hierarchical",
        "voice_driven",
        "rile_dyad_mean",
        "depth_within_type",
    ] + DYAD_CONTROLS
    stats_df = _summary_table(df, key_vars)
    stats_df.to_csv(out / "summary_stats.csv")
    logger.info("Saved: summary_stats.csv")
    logger.info(f"\n{stats_df.to_string()}")

    # Add institution labels
    df = df.copy()
    df["inst_label"] = df["inst"].map(INST_LABELS)
    order = ["Uninstitutionalized", "Voice-Driven", "Hierarchical"]

    # Figure: Depth vs Specialization scatter (H2)
    with _save_figure(out / "depth_vs_specialization.png") as (fig, ax):
        valid = df.dropna(subset=["Depth.score", "spec_dyad_mean"])
        ax.scatter(
            valid["Depth.score"],
            valid["spec_dyad_mean"],
            alpha=0.25,
            s=15,
            color="steelblue",
        )
        z = np.polyfit(valid["Depth.score"], valid["spec_dyad_mean"], 1)
        x_line = np.linspace(valid["Depth.score"].min(), valid["Depth.score"].max(), 100)
        ax.plot(
            x_line, np.polyval(z, x_line), color="firebrick", linewidth=2, label="Linear fit"
        )
        ax.set_xlabel("Alliance Depth Score")
        ax.set_ylabel("Mean Partner Specialization")
        ax.set_title("Alliance Depth vs Partner Specialization (H2)")
        ax.legend(frameon=True)

    # Figure: Specialization by institution type (H2A/H2B)
    with _save_figure(out / "specialization_by_inst.png") as (fig, ax):
        valid = df.dropna(subset=["spec_dyad_mean", "inst_label"])
        sns.boxplot(
            data=valid,
            x="inst_label",
            y="spec_dyad_mean",
            order=order,
            ax=ax,
            hue="inst_label",
            palette="Blues",
            legend=False,
        )
        ax.set_xlabel("Alliance Institution Type")
        ax.set_ylabel("Mean Partner Specialization")
        ax.set_title("Partner Specialization by Alliance Type (H2A/H2B)")

    # Figure: Depth by institution type (diagnostic)
    with _save_figure(out / "depth_by_inst.png") as (fig, ax):
        valid = df.dropna(subset=["Depth.score", "inst_label"])
        sns.boxplot(
            data=valid,
            x="inst_label",
            y="Depth.score",
            order=order,
            ax=ax,
            hue="inst_label",
            palette="Oranges",
            legend=False,
        )
        ax.set_xlabel("Alliance Institution Type")
        ax.set_ylabel("Depth Score")
        ax.set_title("Alliance Depth by Institution Type (Collinearity Check)")

    # Figure: Active dyads over time
    with _save_figure(out / "dyads_timeseries.png", figsize=(8, 4)) as (fig, ax):
        yearly = df.groupby("year").size()
        ax.plot(yearly.index, yearly.values, marker="o", markersize=4, color="steelblue")
        ax.set_xlabel("Year")
        ax.set_ylabel("Number of Active Dyad-Years")
        ax.set_title("Alliance Dyad Coverage Over Time")

    # Institution breakdown table
    breakdown = df.groupby("inst").agg(
        n_dyads=("atopid", "nunique"), n_dyad_years=("year", "count")
    )
    breakdown.index = [INST_LABELS.get(i, i) for i in breakdown.index]
    breakdown.to_csv(out / "inst_breakdown.csv")
    logger.info("Saved: inst_breakdown.csv")
    logger.info(f"\n{breakdown}")

    # Correlation matrix
    corr_vars = [
        "spec_dyad_mean",
        "Depth.score",
        "rile_dyad_mean",
        "hierarchical",
        "voice_driven",
    ]
    corr = df[corr_vars].corr().round(3)
    corr.to_csv(out / "correlation_matrix.csv")
    logger.info("Saved: correlation_matrix.csv")

    # Type-depth diagnostics
    _type_depth_diagnostics(df, out)

    logger.info("Saved: 4 figures")


def _type_depth_diagnostics(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    """Run type-depth collinearity diagnostics."""
    logger.info("[Diagnostic] Type-depth collinearity...")

    complete = df.dropna(subset=["Depth.score", "inst", "spec_dyad_mean"])

    # 1. Correlation
    corr, pval = stats.pearsonr(complete["inst"], complete["Depth.score"])
    logger.info(f"Correlation (inst, Depth.score): r={corr:.3f}, p={pval:.2e}")

    # 2. R² from regressing Depth on type dummies
    # Use hierarchical/voice_driven columns already in dataset (no duplication)
    complete = complete.copy()
    X = sm.add_constant(complete[["hierarchical", "voice_driven"]].values.astype(float))
    y = complete["Depth.score"].values
    r2 = sm.OLS(y, X).fit().rsquared
    logger.info(f"R² (Depth ~ type dummies): {r2:.3f}")

    # 3. VIFs
    X_vif = complete[["Depth.score", "hierarchical", "voice_driven"]].values.astype(float)
    X_vif = np.column_stack([np.ones(X_vif.shape[0]), X_vif])
    vif_depth = variance_inflation_factor(X_vif, 1)
    vif_hier = variance_inflation_factor(X_vif, 2)
    vif_voice = variance_inflation_factor(X_vif, 3)
    logger.info(
        f"VIFs: Depth={vif_depth:.2f}, hierarchical={vif_hier:.2f}, voice={vif_voice:.2f}"
    )

    # Decision
    max_vif = max(vif_depth, vif_hier, vif_voice)
    if r2 > 0.50 or max_vif > 10:
        status = "SUBSTANTIAL_OVERLAP"
    elif r2 < 0.30 and max_vif < 5:
        status = "MODEST_OVERLAP"
    else:
        status = "MODERATE_OVERLAP"
    logger.info(f"Status: {status}")

    # Save diagnostics
    diag = pd.DataFrame(
        {
            "metric": [
                "correlation_r",
                "correlation_p",
                "r_squared",
                "vif_depth",
                "vif_hierarchical",
                "vif_voice",
                "status",
            ],
            "value": [
                f"{corr:.4f}",
                f"{pval:.2e}",
                f"{r2:.4f}",
                f"{vif_depth:.2f}",
                f"{vif_hier:.2f}",
                f"{vif_voice:.2f}",
                status,
            ],
        }
    )
    diag.to_csv(out_dir / "type_depth_diagnostics.csv", index=False)
    logger.info("Saved: type_depth_diagnostics.csv")

    return diag


def run_all() -> None:
    """Generate all descriptive statistics and figures."""
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

    h1_descriptives(paths)
    h2_descriptives(paths)

    logger.info("All descriptives generated.")


if __name__ == "__main__":
    run_all()
