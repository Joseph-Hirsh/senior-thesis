"""
Command-line interface for the thesis analysis pipeline.

Usage:
    thesisizer              # Run full pipeline
    thesisizer --build      # Build datasets only
    thesisizer --h1         # Run H1 analyses (ideology -> specialization)
    thesisizer --h2         # Run H2/H2A/H2B analyses (alliance -> specialization)
    thesisizer --help       # Show help
"""
from __future__ import annotations

import argparse
import sys
import time

from senior_thesis.config import Paths

__all__ = ["main"]


def _header(text: str) -> None:
    """Print a section header."""
    print(f"\n{'=' * 60}\n {text}\n{'=' * 60}")


def _run_build() -> None:
    """Build all datasets."""
    _header("BUILD DATASETS")
    from senior_thesis.build_datasets import build_all

    build_all()


def _run_h1_analysis(paths: Paths) -> None:
    """Run H1 analyses."""
    from senior_thesis.hypotheses import run_h1

    run_h1(paths)


def _run_h2_analysis(paths: Paths) -> None:
    """Run H2/H2A/H2B analyses."""
    from senior_thesis.hypotheses import run_h2

    run_h2(paths)


def main() -> int:
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        prog="thesisizer",
        description="Military Specialization & Alliance Institutions Study",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  thesisizer           Run full pipeline (build + h1 + h2)
  thesisizer --build   Build datasets only
  thesisizer --h1      Run H1: Ideology -> Specialization
  thesisizer --h2      Run H2/H2A/H2B: Alliance -> Specialization

Hypotheses:
  H1:  Right-of-center ideology -> less specialization
  H2:  Alliance depth -> more partner specialization
  H2A: Voice-driven > Uninstitutionalized
  H2B: Hierarchical > Voice-driven
        """,
    )
    parser.add_argument("--build", action="store_true", help="Build datasets only")
    parser.add_argument("--h1", action="store_true", help="H1: Ideology -> Specialization")
    parser.add_argument("--h2", action="store_true", help="H2/H2A/H2B: Alliance -> Specialization")
    parser.add_argument("--version", action="version", version="thesisizer 1.0.0")

    args = parser.parse_args()

    start = time.time()
    paths = Paths()

    # Validate input files before running
    missing = paths.validate()
    if missing:
        print(f"Error: Missing input files:")
        for f in missing:
            print(f"  - {f}")
        return 1

    # Determine what to run (use do_ prefix to avoid confusion with functions)
    do_build = args.build
    do_h1 = args.h1
    do_h2 = args.h2

    # If no flags, run everything
    do_all = not (do_build or do_h1 or do_h2)

    if do_all:
        print("\n" + "=" * 60)
        print(" THESISIZER: Full Analysis Pipeline")
        print("=" * 60)
        print("\nHypotheses:")
        print("  H1:  Ideology -> Specialization")
        print("  H2:  Alliance Depth -> Partner Specialization")
        print("  H2A: Voice-Driven > Uninstitutionalized")
        print("  H2B: Hierarchical > Voice-Driven")

    # Run requested analyses
    if do_build or do_all:
        _run_build()

    if do_h1 or do_all:
        _run_h1_analysis(paths)

    if do_h2 or do_all:
        _run_h2_analysis(paths)

    elapsed = time.time() - start

    print(f"\n{'=' * 60}")
    print(f" COMPLETE ({elapsed:.1f}s)")
    print(f"{'=' * 60}")
    print(f"\nOutputs:")
    if do_build or do_all:
        print(f"  {paths.country_year_csv}")
        print(f"  {paths.dyad_year_csv}")
    if do_h1 or do_all:
        print(f"  {paths.h1_dir}/")
    if do_h2 or do_all:
        print(f"  {paths.h2_dir}/")

    return 0


if __name__ == "__main__":
    sys.exit(main())
