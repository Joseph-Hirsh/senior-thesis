"""
Descriptive statistics and visualizations for thesis chapter.

Produces summary statistics tables and figures for:
- H1: Ideology -> Specialization (country-year)
- H2/H2A/H2B: Alliance institutions -> Specialization (dyad-year)
"""
from __future__ import annotations

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
)

__all__ = [
    "h1_descriptives",
    "h2_descriptives",
    "run_all",
]


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
    print("\n  [H1] Generating descriptives...")

    df = load_dataset(paths.country_year_csv)
    _setup_style()
    out = paths.h1_dir

    # Summary statistics
    key_vars = ["spec_y", "rile", "right_of_center"] + COUNTRY_CONTROLS
    stats_df = _summary_table(df, key_vars)
    stats_df.to_csv(out / "summary_stats.csv")
    print("  Saved: summary_stats.csv")

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

    print("  Saved: 4 figures")


def h2_descriptives(paths: Paths) -> None:
    """
    Generate descriptive statistics and figures for H2 (alliance type -> division of labor).

    Outputs to results/h2/:
    - summary_stats.csv
    - div_labor_by_inst.png
    - dyads_timeseries.png
    - inst_breakdown.csv
    - correlation_matrix.csv
    """
    print("\n  [H2] Generating descriptives...")

    df = load_dataset(paths.dyad_year_csv)
    _setup_style()
    out = paths.h2_dir

    # Summary statistics
    key_vars = [
        "div_labor",
        "inst",
        "hierarchical",
        "voice_driven",
    ] + DYAD_CONTROLS
    stats_df = _summary_table(df, key_vars)
    stats_df.to_csv(out / "summary_stats.csv")
    print("  Saved: summary_stats.csv")

    # Add institution labels
    df = df.copy()
    df["inst_label"] = df["inst"].map(INST_LABELS)
    order = ["Uninstitutionalized", "Voice-Driven", "Hierarchical"]

    # Figure: Division of Labor by institution type (H2)
    with _save_figure(out / "div_labor_by_inst.png") as (fig, ax):
        valid = df.dropna(subset=["div_labor", "inst_label"])
        sns.boxplot(
            data=valid,
            x="inst_label",
            y="div_labor",
            order=order,
            ax=ax,
            hue="inst_label",
            palette="Blues",
            legend=False,
        )
        ax.set_xlabel("Alliance Institution Type")
        ax.set_ylabel("Division of Labor")
        ax.set_title("Division of Labor by Alliance Type (H2)")

    # Figure: Active dyads over time
    with _save_figure(out / "dyads_timeseries.png", figsize=(8, 4)) as (fig, ax):
        yearly = df.groupby("year").size()
        ax.plot(yearly.index, yearly.values, marker="o", markersize=4, color="steelblue")
        ax.set_xlabel("Year")
        ax.set_ylabel("Number of Active Dyad-Years")
        ax.set_title("Alliance Dyad Coverage Over Time")

    # Institution breakdown table
    # After dyad-year collapse, use dyad_id for counting unique dyads
    breakdown = df.groupby("inst").agg(
        n_dyads=("dyad_id", "nunique"), n_dyad_years=("year", "count")
    )
    breakdown.index = [INST_LABELS.get(i, i) for i in breakdown.index]
    breakdown.to_csv(out / "inst_breakdown.csv")
    print("  Saved: inst_breakdown.csv")

    # Correlation matrix
    corr_vars = [
        "div_labor",
        "hierarchical",
        "voice_driven",
    ] + DYAD_CONTROLS
    corr = df[corr_vars].corr().round(3)
    corr.to_csv(out / "correlation_matrix.csv")
    print("  Saved: correlation_matrix.csv")

    print("  Saved: 2 figures")


def run_all() -> None:
    """Generate all descriptive statistics and figures."""
    paths = Paths()

    # Validate input files
    missing = paths.validate()
    if missing:
        print(f"Error: Missing input files: {missing}")
        return

    # Ensure output directories exist
    paths.h1_dir.mkdir(parents=True, exist_ok=True)
    paths.h2_dir.mkdir(parents=True, exist_ok=True)

    h1_descriptives(paths)
    h2_descriptives(paths)

    print("\n  All descriptives generated.")


if __name__ == "__main__":
    run_all()
