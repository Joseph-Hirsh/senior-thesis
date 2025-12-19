"""
Visualize the relationship between political ideology and military specialization.

This script creates three complementary, publication-friendly visualizations:

1) Binned RILE plot (continuous ideology):
   - x-axis: bins of RILE (quantiles by default)
   - y-axis: mean specialization (optionally residualized on controls + country FE + year FE)
   - error bars: 95% CI of the mean within each bin

2) RoC vs LoC dot-whisker (binary ideology):
   - compares mean specialization between Right-of-Center and Left-of-Center governments
   - (optionally residualized)
   - whiskers: 95% CI

3) Marginal effect line (continuous ideology regression):
   - fits: specialization ~ RILE + controls + country FE + year FE
   - plots implied ideology component: beta * (RILE - RILE_ref)
   - confidence band from clustered SE (by country)
   - NOTE: With FE, levels are absorbed; this plot is best interpreted as relative differences across RILE.

Outputs (default):
- results/viz_rile_binned.png
- results/viz_roc_loc.png
- results/viz_rile_marginal_effect.png
- plus CSVs alongside each plot for reproducibility.

Usage:
------
python -m senior_thesis.visualizations
python -m senior_thesis.visualizations --no-residualize
python -m senior_thesis.visualizations --bins 10 --bin-method quantile
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

from senior_thesis.config import Paths, get_controls
from senior_thesis.data_prep import prep_master
from senior_thesis.audit import sha256_file


BinMethod = Literal["quantile", "equalwidth"]


@dataclass(frozen=True)
class Settings:
    residualize: bool = True
    bins: int = 5
    bin_method: BinMethod = "quantile"
    min_bin_n: int = 25
    seed: int = 42


def _residualize_on_controls_and_fe(df: pd.DataFrame, y_col: str, controls: list[str]) -> pd.Series:
    """
    Residualize y on controls + country FE + year FE. (NO ideology in this residualization.)
    Returns a Series aligned to df.index with NaN where residual cannot be computed.
    """
    use = df[[y_col, "country_code_cow", "year"] + controls].copy()
    use = use.dropna(subset=[y_col, "country_code_cow", "year"] + controls).copy()

    rhs = controls + ["C(country_code_cow)", "C(year)"]
    formula = f"{y_col} ~ " + " + ".join(rhs)

    model = smf.ols(formula, data=use)
    res = model.fit()

    out = pd.Series(np.nan, index=df.index, name=f"{y_col}_resid")
    out.loc[use.index] = res.resid
    return out


def _make_bins(x: pd.Series, bins: int, method: BinMethod) -> pd.Series:
    """
    Create categorical bins for x.
    - quantile: pd.qcut (equal counts)
    - equalwidth: pd.cut (equal width)
    """
    x = pd.to_numeric(x, errors="coerce")
    if method == "quantile":
        # duplicates="drop" prevents errors when many identical values
        return pd.qcut(x, q=bins, duplicates="drop")
    if method == "equalwidth":
        return pd.cut(x, bins=bins)
    raise ValueError(f"Unknown bin method: {method}")


def _mean_ci(series: pd.Series) -> tuple[float, float, float, int]:
    """
    Mean and 95% CI for the mean using normal approximation.
    Returns (mean, ci_low, ci_high, n).
    """
    s = pd.to_numeric(series, errors="coerce").dropna()
    n = int(len(s))
    if n == 0:
        return (np.nan, np.nan, np.nan, 0)
    m = float(s.mean())
    se = float(s.std(ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    ci_low = m - 1.96 * se if np.isfinite(se) else np.nan
    ci_high = m + 1.96 * se if np.isfinite(se) else np.nan
    return (m, ci_low, ci_high, n)


def _plot_binned_rile(df: pd.DataFrame, y: str, settings: Settings) -> pd.DataFrame:
    """
    Visualization 1: binned RILE -> specialization.
    Produces plot + returns the binned summary table.
    """
    d = df[["rile", y]].copy()
    d["rile"] = pd.to_numeric(d["rile"], errors="coerce")
    d[y] = pd.to_numeric(d[y], errors="coerce")
    d = d.dropna(subset=["rile", y]).copy()

    d["bin"] = _make_bins(d["rile"], bins=settings.bins, method=settings.bin_method)
    d = d.dropna(subset=["bin"]).copy()

    # Summarize each bin
    rows = []
    for b, sub in d.groupby("bin", observed=True):
        mean, lo, hi, n = _mean_ci(sub[y])
        if n < settings.min_bin_n:
            continue
        # Use bin midpoint for x-position
        if hasattr(b, "left") and hasattr(b, "right"):
            x_mid = float((b.left + b.right) / 2)
            label = f"{b.left:.1f} to {b.right:.1f}"
        else:
            # qcut categories are also intervals; but keep fallback
            x_mid = float(sub["rile"].mean())
            label = str(b)
        rows.append(
            {"bin": label, "x_mid": x_mid, "mean": mean, "ci_low": lo, "ci_high": hi, "n": n}
        )

    out = pd.DataFrame(rows).sort_values("x_mid").reset_index(drop=True)

    # Plot
    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.errorbar(out["x_mid"], out["mean"], yerr=[out["mean"] - out["ci_low"], out["ci_high"] - out["mean"]],
                fmt="o-", capsize=4)
    ax.axhline(0, linestyle=":", linewidth=1)
    ax.set_xlabel("RILE (binned; plotted at bin midpoint)")
    ax.set_ylabel("Military specialization" + (" (residualized)" if settings.residualize else ""))
    ax.set_title("Specialization across ideological space (binned RILE)")
    ax.grid(True, linestyle=":", alpha=0.4)

    plt.tight_layout()
    png_path = "results/viz_rile_binned.png"
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close()

    csv_path = "results/viz_rile_binned.csv"
    out.to_csv(csv_path, index=False)

    print(f"\n✓ Saved binned RILE plot: {png_path}")
    print(f"✓ Saved binned table:     {csv_path}")

    return out


def _plot_roc_loc(df: pd.DataFrame, y: str, settings: Settings) -> pd.DataFrame:
    """
    Visualization 2: RoC vs LoC dot-whisker.
    """
    d = df[["right_of_center", y]].copy()
    d["right_of_center"] = pd.to_numeric(d["right_of_center"], errors="coerce")
    d[y] = pd.to_numeric(d[y], errors="coerce")
    d = d.dropna(subset=["right_of_center", y]).copy()
    d = d[d["right_of_center"].isin([0, 1])].copy()

    labels = {0: "LoC (RILE ≤ -10)", 1: "RoC (RILE ≥ 10)"}
    rows = []
    for v in [0, 1]:
        sub = d[d["right_of_center"] == v]
        mean, lo, hi, n = _mean_ci(sub[y])
        rows.append({"group": labels[v], "mean": mean, "ci_low": lo, "ci_high": hi, "n": n})
    out = pd.DataFrame(rows)

    # Plot
    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(out))
    ax.errorbar(x, out["mean"],
                yerr=[out["mean"] - out["ci_low"], out["ci_high"] - out["mean"]],
                fmt="o", capsize=6)
    ax.axhline(0, linestyle=":", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(out["group"], rotation=0)
    ax.set_ylabel("Military specialization" + (" (residualized)" if settings.residualize else ""))
    ax.set_title("Mean specialization under Left- vs Right-of-Center governments")
    ax.grid(True, axis="y", linestyle=":", alpha=0.4)

    # Add N labels
    for i, n in enumerate(out["n"]):
        ax.text(i, out.loc[i, "mean"], f"  n={int(n)}", va="center")

    plt.tight_layout()
    png_path = "results/viz_roc_loc.png"
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close()

    csv_path = "results/viz_roc_loc.csv"
    out.to_csv(csv_path, index=False)

    print(f"\n✓ Saved RoC vs LoC plot: {png_path}")
    print(f"✓ Saved RoC vs LoC table:{csv_path}")

    return out


def _plot_marginal_effect(df: pd.DataFrame, y_col: str, settings: Settings) -> pd.DataFrame:
    """
    Visualization 3: marginal effect line from a FE regression:
        y ~ RILE + controls + country FE + year FE

    We then plot beta * (RILE - RILE_ref) so the line is anchored at 0 at RILE_ref (median by default).

    This avoids misleading "levels" when FE absorb intercepts.
    """
    controls = get_controls()

    use = df[[y_col, "rile", "country_code_cow", "year"] + controls].copy()
    use["rile"] = pd.to_numeric(use["rile"], errors="coerce")
    use[y_col] = pd.to_numeric(use[y_col], errors="coerce")
    use = use.dropna(subset=[y_col, "rile", "country_code_cow", "year"] + controls).copy()

    # FE regression with clustered SE by country
    rhs = ["rile"] + controls + ["C(country_code_cow)", "C(year)"]
    formula = f"{y_col} ~ " + " + ".join(rhs)
    model = smf.ols(formula, data=use)
    res = model.fit(cov_type="cluster", cov_kwds={"groups": use["country_code_cow"]})

    beta = float(res.params["rile"])
    se = float(res.bse["rile"])
    print("\n--- FE regression for marginal effect plot ---")
    print(f"N = {int(res.nobs)}")
    print(f"beta(RILE) = {beta:.6f}, SE(cluster country) = {se:.6f}")

    # Range for plotting
    r_min = float(np.nanpercentile(use["rile"], 1))
    r_max = float(np.nanpercentile(use["rile"], 99))
    grid = np.linspace(r_min, r_max, 200)
    r_ref = float(np.nanmedian(use["rile"]))

    eff = beta * (grid - r_ref)
    eff_low = (beta - 1.96 * se) * (grid - r_ref)
    eff_high = (beta + 1.96 * se) * (grid - r_ref)

    out = pd.DataFrame(
        {
            "rile": grid,
            "rile_ref": r_ref,
            "effect": eff,
            "ci_low": eff_low,
            "ci_high": eff_high,
            "beta": beta,
            "se": se,
            "n": int(res.nobs),
        }
    )

    # Plot
    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.plot(out["rile"], out["effect"])
    ax.fill_between(out["rile"], out["ci_low"], out["ci_high"], alpha=0.2)
    ax.axhline(0, linestyle=":", linewidth=1)
    ax.axvline(r_ref, linestyle="--", linewidth=1)
    ax.set_xlabel("RILE (continuous)")
    ax.set_ylabel("Implied ideology component: β · (RILE − RILE_ref)")
    ax.set_title("Estimated association of ideology with specialization (FE regression)")
    ax.grid(True, linestyle=":", alpha=0.4)

    plt.tight_layout()
    png_path = "results/viz_rile_marginal_effect.png"
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close()

    csv_path = "results/viz_rile_marginal_effect.csv"
    out.to_csv(csv_path, index=False)

    print(f"\n✓ Saved marginal effect plot: {png_path}")
    print(f"✓ Saved marginal effect table:{csv_path}")

    return out


def run(settings: Settings) -> None:
    paths = Paths()
    print("\nRaw file hash (master CSV):", sha256_file(paths.master_out))

    df = prep_master(paths)

    # Choose outcome for plots 1 & 2: raw or residualized
    y_base = "spec_y"
    if settings.residualize:
        print("\nResidualizing specialization on controls + country FE + year FE (for plots 1 & 2)...")
        df["spec_plot"] = _residualize_on_controls_and_fe(df, y_base, get_controls())
        y_for_simple_plots = "spec_plot"
    else:
        df["spec_plot"] = pd.to_numeric(df[y_base], errors="coerce")
        y_for_simple_plots = "spec_plot"

    # Plot 1: binned RILE
    _plot_binned_rile(df, y=y_for_simple_plots, settings=settings)

    # Plot 2: RoC vs LoC
    _plot_roc_loc(df, y=y_for_simple_plots, settings=settings)

    # Plot 3: marginal effect from FE regression using RAW spec_y (not residualized),
    # because the model itself partials out controls + FE.
    _plot_marginal_effect(df, y_col="spec_y", settings=settings)

    # Small metadata dump
    meta = pd.DataFrame(
        [
            {
                "residualize_plots_1_2": settings.residualize,
                "bins": settings.bins,
                "bin_method": settings.bin_method,
                "min_bin_n": settings.min_bin_n,
                "seed": settings.seed,
            }
        ]
    )
    meta_path = "results/viz_ideology_specialization_meta.csv"
    meta.to_csv(meta_path, index=False)
    print(f"\n✓ Saved meta: {meta_path}")


def _parse_args() -> Settings:
    p = argparse.ArgumentParser(description="Create ideology-specialization visualizations.")
    p.add_argument("--no-residualize", action="store_true", help="Use raw specialization for plots 1 & 2.")
    p.add_argument("--bins", type=int, default=5, help="Number of RILE bins for the binned plot.")
    p.add_argument(
        "--bin-method",
        type=str,
        default="quantile",
        choices=["quantile", "equalwidth"],
        help="Binning rule for RILE.",
    )
    p.add_argument("--min-bin-n", type=int, default=25, help="Minimum observations per bin.")
    p.add_argument("--seed", type=int, default=42, help="Random seed (reserved for future extensions).")
    args = p.parse_args()

    return Settings(
        residualize=not args.no_residualize,
        bins=args.bins,
        bin_method=args.bin_method,  # type: ignore[arg-type]
        min_bin_n=args.min_bin_n,
        seed=args.seed,
    )


def main() -> None:
    settings = _parse_args()
    run(settings)


if __name__ == "__main__":
    main()
