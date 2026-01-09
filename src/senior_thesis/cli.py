"""
Command-line interface for the thesis analysis pipeline.

Usage:
    thesisizer              # Run full pipeline (build + h1 + h2 + h3)
    thesisizer --build      # Build datasets only
    thesisizer --h1         # Run H1 analyses
    thesisizer --h2         # Run H2 analyses
    thesisizer --h3         # Run H3 analyses
    thesisizer --audit      # Run attrition audit (optional)
    thesisizer --log        # Save output to timestamped log file
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

from senior_thesis.config import Paths

__all__ = ["main"]


class TeeOutput:
    """Write output to both stdout and a file."""

    def __init__(self, filepath: Path):
        self.terminal = sys.stdout
        self.log = open(filepath, "w", encoding="utf-8")

    def write(self, message: str) -> None:
        self.terminal.write(message)
        self.log.write(message)

    def flush(self) -> None:
        self.terminal.flush()
        self.log.flush()

    def close(self) -> None:
        self.log.close()


def _banner() -> None:
    """Print the startup banner."""
    print()
    print("╔════════════════════════════════════════════════════════════╗")
    print("║     MILITARY SPECIALIZATION & ALLIANCE INSTITUTIONS        ║")
    print("║                    Senior Thesis Analysis                  ║")
    print("╚════════════════════════════════════════════════════════════╝")


def main() -> int:
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        prog="thesisizer",
        description="Military Specialization & Alliance Institutions Study",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Hypotheses:
  H1:  Ideology → Specialization (right-of-center → less specialization)
  H2:  Alliance Depth → Division of Labor (H2A: voice > uninst, H2B: hier > voice)
  H3:  Ideological Similarity → Division of Labor
        """,
    )
    parser.add_argument("--build", action="store_true", help="Build datasets only")
    parser.add_argument("--h1", action="store_true", help="H1: Ideology → Specialization")
    parser.add_argument("--h2", action="store_true", help="H2: Alliance → Division of Labor")
    parser.add_argument("--h3", action="store_true", help="H3: Ideology Similarity → Division of Labor")
    parser.add_argument("--audit", action="store_true", help="Run attrition & missingness audit")
    parser.add_argument("--log", action="store_true", help="Save output to timestamped log file")
    parser.add_argument("--version", action="version", version="thesisizer 1.0.0")

    args = parser.parse_args()
    start = time.time()
    paths = Paths()

    # Set up logging to file if requested
    tee = None
    log_path = None
    if args.log:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = paths.results_dir / f"thesisizer_{timestamp}.log"
        tee = TeeOutput(log_path)
        sys.stdout = tee

    try:
        # Validate input files
        missing = paths.validate()
        if missing:
            print("Error: Missing input files:")
            for f in missing:
                print(f"  - {f}")
            return 1

        # Determine what to run
        do_build = args.build
        do_h1 = args.h1
        do_h2 = args.h2
        do_h3 = args.h3
        do_audit = args.audit
        do_all = not any([do_build, do_h1, do_h2, do_h3, do_audit])

        _banner()

        if log_path:
            print(f"\n  Logging to: {log_path}")

        # Build datasets
        if do_build or do_all:
            print("\n" + "=" * 60)
            print(" BUILD DATASETS")
            print("=" * 60)
            from senior_thesis.build_datasets import build_all
            build_all()

        # Run hypotheses
        if do_h1 or do_all:
            from senior_thesis.hypotheses import run_h1
            run_h1(paths)

        if do_h2 or do_all:
            from senior_thesis.hypotheses import run_h2
            run_h2(paths)

        if do_h3 or do_all:
            from senior_thesis.hypotheses import run_h3
            run_h3(paths)

        # Print summary table if any hypotheses were run
        if do_h1 or do_h2 or do_h3 or do_all:
            from senior_thesis.regressions import print_summary_table
            print_summary_table()

        # Audit (only when explicitly requested)
        if do_audit:
            from senior_thesis.attrition_audit import run_full_audit
            run_full_audit(paths)

        # Summary
        elapsed = time.time() - start
        print()
        print("═" * 60)
        print(f" COMPLETE ({elapsed:.1f}s)")
        print("═" * 60)
        print("\nOutputs:")
        if do_build or do_all:
            print(f"  Datasets: {paths.country_year_csv.parent}/")
        if do_h1 or do_all:
            print(f"  H1: {paths.h1_dir}/")
        if do_h2 or do_all:
            print(f"  H2: {paths.h2_dir}/")
        if do_h3 or do_all:
            print(f"  H3: {paths.h3_dir}/")
        if do_audit:
            print(f"  Audit: {paths.h1_dir.parent / 'audit'}/")
        if log_path:
            print(f"  Log: {log_path}")

        return 0

    finally:
        # Restore stdout and close log file
        if tee:
            sys.stdout = tee.terminal
            tee.close()


if __name__ == "__main__":
    sys.exit(main())
