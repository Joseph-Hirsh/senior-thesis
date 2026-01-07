"""
Command-line interface for the thesis analysis pipeline.

Usage:
    thesisizer              # Run full pipeline
    thesisizer --build      # Build datasets only
    thesisizer --desc       # Descriptives only
    thesisizer --reg        # Regressions only
    thesisizer --help       # Show help
"""
from __future__ import annotations

import argparse
import sys
import time

from senior_thesis.config import Paths


def _header(text: str) -> None:
    print(f"\n{'=' * 60}\n {text}\n{'=' * 60}")


def _run_build() -> None:
    _header("BUILD DATASETS")
    from senior_thesis.build_datasets import build_all
    build_all()


def _run_desc() -> None:
    _header("DESCRIPTIVE STATISTICS")
    from senior_thesis.descriptives import run_all
    run_all()


def _run_reg() -> None:
    _header("REGRESSION ANALYSES")
    from senior_thesis.regressions import run_all
    run_all()


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="thesisizer",
        description="Military Specialization & Alliance Institutions Study",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  thesisizer           Run full pipeline (build + desc + reg)
  thesisizer --build   Build datasets only
  thesisizer --desc    Generate descriptives and figures
  thesisizer --reg     Run regression analyses
        """,
    )
    parser.add_argument("--build", action="store_true", help="Build datasets")
    parser.add_argument("--desc", action="store_true", help="Generate descriptives")
    parser.add_argument("--reg", action="store_true", help="Run regressions")
    parser.add_argument("--version", action="version", version="thesisizer 1.0.0")

    args = parser.parse_args()

    start = time.time()

    # If no flags, run everything
    run_all = not (args.build or args.desc or args.reg)

    if run_all:
        print("\n" + "=" * 60)
        print(" THESISIZER: Full Analysis Pipeline")
        print("=" * 60)
        print("\nHypotheses:")
        print("  H1:  Ideology → Specialization")
        print("  H2:  Alliance Depth → Partner Specialization")
        print("  H2A: Voice-Driven > Uninstitutionalized")
        print("  H2B: Hierarchical > Voice-Driven")

    if args.build or run_all:
        _run_build()

    if args.desc or run_all:
        _run_desc()

    if args.reg or run_all:
        _run_reg()

    elapsed = time.time() - start
    paths = Paths()

    print(f"\n{'=' * 60}")
    print(f" COMPLETE ({elapsed:.1f}s)")
    print(f"{'=' * 60}")
    print(f"\nOutputs:")
    print(f"  {paths.country_year_csv}")
    print(f"  {paths.dyad_year_csv}")
    print(f"  {paths.tables_dir}/")
    print(f"  {paths.figures_dir}/")

    return 0


if __name__ == "__main__":
    sys.exit(main())
