# Military Specialization and Alliance Institutions

A senior thesis examining the effects of ruling party ideology and alliance institutional design on military specialization among democratic states.

## Quick Start

```bash
# Install with uv
uv sync

# Run full analysis pipeline
uv run thesisizer
```

## Usage

```bash
thesisizer              # Run full pipeline (~5 seconds)
thesisizer --build      # Build datasets only
thesisizer --desc       # Generate descriptives and figures
thesisizer --reg        # Run regression analyses
thesisizer --help       # Show help
```

## Research Questions

| ID | Hypothesis | Test |
|----|------------|------|
| H1 | Right-of-center parties → less specialization | Country-year |
| H2 | Alliance depth → more partner specialization | Dyad-year |
| H2A | Voice-driven > Uninstitutionalized | Dyad-year |
| H2B | Hierarchical > Voice-driven | Dyad-year |

## Output

| Type | Location | Contents |
|------|----------|----------|
| Datasets | `results/` | `master_country_year.csv`, `master_dyad_year.csv` |
| Tables | `results/tables/` | Summary stats, regression results |
| Figures | `results/figures/` | 8 publication-ready PNGs |

## Project Structure

```
src/senior_thesis/
├── cli.py            # thesisizer command
├── config.py         # Paths, controls
├── build_datasets.py # Dataset construction
├── descriptives.py   # Stats + figures
└── regressions.py    # Hypothesis tests
```

## Data Sources

| File | Source | Contents |
|------|--------|----------|
| `03_DF-full.rds` | Gannon | Specialization index |
| `MPDataset_MPDS2025a.csv` | Manifesto | Party ideology |
| `partiestoanallianceR&R.dta` | Rapport & Rathbun | Alliance institutions |
| `atop5_1m.csv` | ATOP 5.1 | Alliance membership |
| `AllianceDataScoreJCR_RR.csv` | Benson & Clinton | Alliance depth |

## Requirements

- Python ≥3.10
- [uv](https://github.com/astral-sh/uv) (recommended) or pip

## Author

Joseph Hirsh
