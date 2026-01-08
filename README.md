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
thesisizer              # Run full pipeline (build + h1 + h2)
thesisizer --build      # Build datasets only
thesisizer --h1         # Run H1: Ideology -> Specialization
thesisizer --h2         # Run H2/H2A/H2B: Alliance -> Specialization
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
| H1 Results | `results/h1/` | Summary stats, figures, regression tables |
| H2 Results | `results/h2/` | Summary stats, figures, regression tables |

## Project Structure

```
src/senior_thesis/
├── cli.py            # thesisizer command
├── config.py         # Paths, controls, formulas
├── utils.py          # Data validation
├── build_datasets.py # Dataset construction
├── descriptives.py   # Stats + figures
├── regressions.py    # Hypothesis tests
└── hypotheses.py     # Analysis orchestration
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
