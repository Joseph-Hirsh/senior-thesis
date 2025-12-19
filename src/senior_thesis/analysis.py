"""Main dispatcher for running various analyses."""
from __future__ import annotations

# Backward compatibility: maintain old import name
from senior_thesis.data_prep import prep_master as _prep_master


def run_all_analyses(
    include_descriptive: bool = True,
    include_visualizations: bool = True,
    include_regressions: bool = True,
) -> None:
    """
    Run comprehensive analysis pipeline.

    This is the main orchestrator that calls various analysis modules
    based on the flags provided.

    Args:
        include_descriptive: Run descriptive statistics and coverage checks
        include_visualizations: Create ideology-specialization visualizations
        include_regressions: Run regression analyses
    """
    if include_descriptive:
        from senior_thesis.descriptive import run_descriptive_analysis
        print("\n" + "="*80)
        print("RUNNING DESCRIPTIVE ANALYSIS")
        print("="*80)
        run_descriptive_analysis()

    if include_visualizations:
        from senior_thesis.visualizations import run, Settings
        print("\n" + "="*80)
        print("RUNNING VISUALIZATIONS")
        print("="*80)
        run(Settings())

    if include_regressions:
        from senior_thesis.regressions import run_lag_sensitivity
        print("\n" + "="*80)
        print("RUNNING REGRESSIONS")
        print("="*80)
        run_lag_sensitivity()


def main() -> None:
    """Run all analyses by default."""
    run_all_analyses()


if __name__ == "__main__":
    main()
