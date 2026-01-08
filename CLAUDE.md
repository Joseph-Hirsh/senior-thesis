# Claude Context for Senior Thesis Project

## Project Overview

This is a political science senior thesis examining how **ruling party ideology** and **alliance institutional design** affect **military specialization** among democratic states.

## Research Questions & Hypotheses

| ID | Hypothesis | Expected Effect | Unit of Analysis |
|----|------------|-----------------|------------------|
| H1 | Right-of-center parties → less specialization | β < 0 | Country-year |
| H2 | Alliance depth → more partner specialization | β > 0 | Dyad-year |
| H2A | Voice-driven > Uninstitutionalized | β > 0 | Dyad-year |
| H2B | Hierarchical > Voice-driven | β_hier > β_voice | Dyad-year |

## Key Design Decisions

### Why Two Datasets?
- **Country-year** (H1): Tests individual state policy choices driven by ruling party ideology
- **Dyad-year** (H2/H2A/H2B): Tests specialization within alliances, which is a property of the relationship

### Type-Depth Collinearity
Alliance type (`inst`) and depth (`Depth.score`) have r≈0.83, R²≈0.71. They cannot be naively combined:
- **Primary models**: Run separately (model_h2 for depth, model_h2ab for type)
- **Robustness**: Use `depth_within_type` (residualized depth)

### Sample Restrictions
All analyses restricted to **democracies** because:
1. R&R dataset contains only democratic dyads
2. Manifesto Project covers democracies
3. Ruling party ideology more meaningful in competitive democracies

## Data Sources

| File | Contents | Key Variables |
|------|----------|---------------|
| `03_DF-full.rds` | Gannon specialization | `spec_stand`, `country_code_cow`, `year` |
| `MPDataset_MPDS2025a.csv` | Manifesto ideology | `rile`, `country`, `edate` |
| `partiestoanallianceR&R.dta` | R&R alliance data | `inst`, `state_a`, `state_b`, controls |
| `atop5_1m.csv` | ATOP membership | `atopid`, `member`, `yrent`, `yrexit` |
| `AllianceDataScoreJCR_RR.csv` | Benson & Clinton | `Depth.score`, `atopid` |

## Code Structure

```
src/senior_thesis/
├── __init__.py       # Package initialization
├── cli.py            # Entry point: thesisizer command
├── config.py         # Paths, controls, formulas, constants
├── utils.py          # Data validation helpers
├── build_datasets.py # Dataset construction
├── descriptives.py   # Summary stats + figures
├── regressions.py    # Regression models
└── hypotheses.py     # Orchestrates analyses by hypothesis
```

## Running the Pipeline

```bash
uv run thesisizer              # Full pipeline (build + h1 + h2)
uv run thesisizer --build      # Build datasets only
uv run thesisizer --h1         # Run H1 analyses (descriptives + regressions)
uv run thesisizer --h2         # Run H2/H2A/H2B analyses
```

## Configuration (config.py)

Key exports:
- `Paths`: Dataclass with all file paths (uses absolute paths from package root)
- `COUNTRY_CONTROLS`: `["lngdp", "cinc", "war5_lag"]`
- `DYAD_CONTROLS`: R&R control variables
- `FORMULAS`: Regression formula templates with `{controls}` placeholder
- `VARIABLE_MAP`: Source column → analysis variable name mapping
- `RILE_RIGHT_THRESHOLD` / `RILE_LEFT_THRESHOLD`: ±10.0
- `get_available_controls()`: Filter controls to those in DataFrame
- `load_dataset()`: Cached CSV loading

## Key Variables

### Country-Year (H1)
- `spec_y`: Standardized specialization (DV)
- `rile`: Left-right ideology score (-100 to +100)
- `right_of_center`: Binary (1 if RILE≥10, 0 if RILE≤-10)
- Controls: `lngdp`, `cinc`, `war5_lag`

### Dyad-Year (H2/H2A/H2B)
- `spec_dyad_mean`: (spec_a + spec_b) / 2 (DV)
- `Depth.score`: Benson & Clinton alliance depth
- `inst`: 1=uninst, 2=voice, 3=hierarchical
- `hierarchical`, `voice_driven`: Binary dummies
- `depth_within_type`: Residualized depth (robustness)
- `rile_dyad_mean`: Mean partner ideology
- Controls from R&R: `coldwar`, `tot_rivals`, `totmids2`, `s_un_glo`, `undist`, `jntdem`, `priorviol`, `symm`, `lncprtio`, `priorviol_x_symm`

## Model Specifications

### model_h1 (H1)
```
spec_y ~ right_of_center + lngdp + cinc + war5_lag + C(country_code_cow) + C(year)
```
Clustered SEs by country.

### model_h2 (H2)
```
spec_dyad_mean ~ Depth_score + rile_dyad_mean + [R&R controls] + C(year)
```
Clustered SEs by atopid.

### model_h2ab (H2A/H2B)
```
spec_dyad_mean ~ hierarchical + voice_driven + rile_dyad_mean + [R&R controls] + C(year)
```
Reference: uninstitutionalized (inst=1).

### model_robustness
```
spec_dyad_mean ~ hierarchical + voice_driven + depth_within_type + rile_dyad_mean + [R&R controls] + C(year)
```

## Common Tasks

### Adding a new control variable
1. Add to `COUNTRY_CONTROLS` or `DYAD_CONTROLS` in `config.py`
2. Ensure it exists in the source data or create it in `build_datasets.py`

### Adding a new figure
1. Add to `h1_descriptives()` or `h2_descriptives()` in `descriptives.py`
2. Use `_save_figure()` context manager

### Adding a new model
1. Add function in `regressions.py` following existing pattern
2. Call it from the appropriate `run_h1()` or `run_h2()` in `hypotheses.py`

### Adding a new formula
1. Add to `FORMULAS` dict in `config.py`
2. Use `FORMULAS["key"].format(controls=controls)` in the model function

## Gotchas

- `Depth.score` has a period in the name → rename to `Depth_score` in formulas (done in-memory)
- R&R `inst` variable: 1=uninst, 2=voice, 3=hierarchical (NOT 0/1/2)
- Some ATOP exit years are 0 (meaning still active) → treat as 2014 (YEAR_END)
- Specialization data runs 1970-2014, limit dyad expansion accordingly
- Paths are absolute (resolved from package root via `_ROOT`)
- Use `paths.validate()` to check all input files exist before running
