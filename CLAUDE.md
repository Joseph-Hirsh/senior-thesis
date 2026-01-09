# Claude Context for Senior Thesis Project

## Project Overview

This is a political science senior thesis examining how **ruling party ideology** and **alliance institutional design** affect **military specialization** among democratic states.

## Research Questions & Hypotheses

| ID | Hypothesis | Expected Effect | Unit of Analysis |
|----|------------|-----------------|------------------|
| H1 | Right-of-center parties → less specialization | β < 0 | Country-year |
| H2 | Alliance type → division of labor | (see H2A/H2B) | Dyad-year |
| H2A | Voice-driven > Uninstitutionalized | β > 0 | Dyad-year |
| H2B | Hierarchical > Voice-driven | β_hier > β_voice | Dyad-year |

## Statistical Best Practices Implemented

### H1 Lag Specifications
- **H1 tests 10 lag specifications** (1-10 years)
- Each lag tests a distinct causal mechanism (different procurement lead times)
- No multiple comparison correction applied (lags are independent tests)

### Proper Hypothesis Testing
- **H2B Wald test** uses proper covariance matrix via `model.wald_test()`
- This accounts for correlation between hierarchical and voice_driven coefficients
- Previous implementation incorrectly assumed independence (overstated SE)

### Clustering Structure
- **H1**: Clustered by country (accounts for serial correlation)
- **H2**: Clustered by alliance (atopid)
- Within-dyad correlation handled by dyad fixed effects

## Key Design Decisions

### Why Two Datasets?
- **Country-year** (H1): Tests individual state policy choices driven by ruling party ideology
- **Dyad-year** (H2): Tests division of labor between alliance partners based on institution type

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
| `rDMC_wide_v1.rds` | Military capabilities (69 types) | `ccode`, `year`, + 69 technology columns |
| `division_of_labor.csv` | Pairwise portfolio dissimilarity | `state_a`, `state_b`, `year`, `div_labor` |

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
uv run thesisizer --h2         # Run H2 analyses
```

## Configuration (config.py)

Key exports:
- `Paths`: Dataclass with all file paths (uses absolute paths from package root)
- `COUNTRY_CONTROLS`: `["lngdp", "cinc", "war5_lag"]`
- `DYAD_CONTROLS`: Dyad-level control variables
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
- `roc_lag{N}`: Lagged right_of_center (N = 1-10 years)
- `in_alliance`: Binary indicator (1 if country in any alliance that year)
- `in_hierarchical`: Binary (1 if in hierarchical alliance)
- `in_voice`: Binary (1 if in voice-driven alliance)
- `in_uninst`: Binary (1 if in uninstitutionalized alliance)
- Controls: `lngdp`, `cinc`, `war5_lag`, `in_alliance`

### Dyad-Year (H2)
- `div_labor`: Pairwise portfolio complementarity (DV, 0=identical portfolios, 1=fully dissimilar)
- `inst`: Alliance institutionalization (nominal, 1=uninst, 2=voice, 3=hierarchical)
- `hierarchical`, `voice_driven`: Binary dummies (reference: uninstitutionalized)
- `rile_dyad_mean`: Mean partner ideology
- Partner controls: `lngdp_a`, `lngdp_b`, `cinc_a`, `cinc_b` (time-varying country characteristics)

---

## DETAILED DATA FLOW DOCUMENTATION

This section documents exactly how data flows from raw files to final analysis, so you can verify correctness.

### Division of Labor Calculation (Niche Width Measure)

The dependent variable `div_labor` measures the **weighted pairwise dissimilarity** of military portfolios between two states in a given year, based on the rDMC (relative Defense Military Capabilities) data.

**Source data**: `rDMC_wide_v1.rds` - Contains counts of 69 military technology categories (e.g., aircraft_attack, submarines_ballistic, helicopters_transport) for each country-year.

**Formula** (from Gannon):

1. **Convert to proportions**: For each country i in year t, compute p_im = (count of technology m) / (total count of all technologies)
2. **Compute similarity**: θ_ij = Σ_m min(p_im, p_jm) — the sum of minimum proportions across all technologies
3. **Convert to dissimilarity**: div_labor = 1 - θ_ij

**Interpretation**:
- **θ_ij = 1** (similarity): Both states have identical technology proportions → div_labor = 0
- **θ_ij = 0** (dissimilarity): States have entirely non-overlapping portfolios → div_labor = 1
- Higher div_labor = greater complementarity = your partner has capabilities you lack (and vice versa)

**Why this measure?** The measure is weighted by the abundance of each technology, so sharing common equipment (e.g., main battle tanks) contributes less to similarity than sharing rare capabilities (e.g., ICBMs). This accounts for the wide differences in availability of each technology.

**Distribution** (all dyad-years, 1970-2014):
- N = 483,109 dyad-years
- Mean = 0.55, Median = 0.52, Std = 0.22
- Range: [0, 1]

**File**: `assets/datasets/division_of_labor.csv` contains pre-computed div_labor for all country pairs.

### Alliance Institutionalization Coding (Leeds & Anac 2005)

Uses ATOP treaty provisions. This is a 3-category **NOMINAL** outcome (not ordinal) reflecting distinct governance modes. Hierarchy dominates voice; voice dominates absence.

**Variables used**: INTCOM, MILCON, BASE, SUBORD, ORGAN1, ORGPURP1, ORGAN2, ORGPURP2, MILAID, CONTRIB

**Helper**: `mil_org_present` = (ORGAN1 in {1,2,3} & ORGPURP1==1) OR (ORGAN2 in {1,2,3} & ORGPURP2==1)

**Categories** (applied in order, mutually exclusive):

1. **Hierarchical** (inst=3): Authority, command, or structural control provisions:
   - `INTCOM == 1` (integrated command in peacetime and wartime)
   - `MILCON == 3` (common defense policy: doctrine, training, procurement, joint planning)
   - `BASE > 0` (joint or unilateral troop placement / basing)
   - `SUBORD in {1,2}` (explicit subordination of forces during conflict)

2. **Voice-driven** (inst=2): Coordination/consultation mechanisms (if NOT hierarchical):
   - `MILCON == 2` (peacetime military consultation)
   - `mil_org_present` (formal military coordinating organization)
   - `MILAID in {3,4}` (training and/or technology transfer)
   - `CONTRIB == 1` (specified troop/supply/funding contributions)

3. **Uninstitutionalized** (inst=1): None of the above provisions

**Missing values**: Treated as "provision absent" (equivalent to 0) following ATOP coding conventions.

**CRITICAL**: ATOP codes some provisions at the member level (e.g., BASE varies by member). We aggregate by alliance using `groupby("atopid").max()` — if ANY member has a provision, the alliance has it.

### Dyad-Year Panel Construction

1. **Load ATOP membership** (`atop5_1m.csv`): Contains member × alliance × phase records
2. **Consolidate membership**: Group by (atopid, member), take min(yrent), max(yrexit)
3. **Build all dyads**: Self-join members on atopid, keep state_a < state_b
4. **Compute dyad period**: dyad_start = max(yrent_a, yrent_b), dyad_end = min(yrexit_a, yrexit_b)
5. **Expand to dyad-years**: Create one row per (atopid, state_a, state_b, year)
6. **Merge institutionalization**: Join inst from `_build_institutionalization()`
7. **Merge div_labor**: Join from `division_of_labor.csv` on (state_a, state_b, year)

**Output**: `results/master_dyad_year.csv` with 153,766 dyad-years

---

## DIAGNOSTIC FINDINGS (IMPORTANT!)

### Raw Data Shows Unexpected Pattern

Simple means of div_labor by alliance type (NO controls, NO fixed effects):

| Alliance Type | N | Mean div_labor |
|---------------|---|----------------|
| Uninstitutionalized | 25,228 | 0.481 |
| Voice-driven | 60,056 | 0.517 |
| Hierarchical | 42,880 | 0.472 |

**Key observation**: Hierarchical alliances have LOWER div_labor than uninstitutionalized!

### Alliance vs Non-Alliance Comparison

| Group | N | Mean div_labor |
|-------|---|----------------|
| Not in any alliance | 401,454 | 0.558 |
| In any alliance | 81,655 | 0.508 |
| In hierarchical alliance | 36,125 | 0.475 |
| NATO specifically | 7,941 | 0.545 |

**Key observation**: Alliance members have LESS division of labor than non-members! This suggests alliances lead to portfolio CONVERGENCE, not specialization.

### Why This Might Happen

1. **Selection effect**: Countries with similar military needs form alliances together
2. **Convergence effect**: Alliance members coordinate procurement and converge over time
3. **Measurement issue**: div_labor measures portfolio similarity, not role specialization
4. **NATO exception**: NATO dyads DO show high div_labor (0.545), but other hierarchical alliances don't

### Within-Dyad Variation

For the 1,268 dyads that appear in BOTH hierarchical and non-hierarchical alliances:

| Alliance Type | Mean div_labor (within these dyads) |
|---------------|-------------------------------------|
| Uninstitutionalized | 0.424 |
| Voice-driven | 0.488 |
| Hierarchical | 0.473 |

This is the variation used by country-FE models. Even within the same dyads, hierarchical years don't have higher div_labor than voice-driven years.

---

## Model Specifications

### model_h1 (H1)
```
spec_y ~ roc_lag{N} + lngdp + cinc + war5_lag + C(country_code_cow) + C(year)
```
Tests lags N = 1 to 10 years (defense procurement takes time to materialize).
Clustered SEs by country.

### model_h1_event_study (H1 Event Study)
Runs **separate** event studies for:
- **Transitions to right** (left → right): expect negative post-transition effects
- **Transitions to left** (right → left): expect positive post-transition effects

```
spec_y ~ event_time_dummies + [controls] + C(country_code_cow) + C(year)
```
Window: t-5 to t+5 around transition. Reference period: t=-1. Tests parallel trends.

### model_h2 (H2)
```
div_labor ~ hierarchical + voice_driven + contiguous + gdp_ratio + cinc_ratio + C(dyad_id) + C(decade)
```
- **DV**: div_labor (pairwise portfolio dissimilarity)
- **IVs**: hierarchical, voice_driven (reference: uninstitutionalized)
- **Controls**: contiguous, gdp_ratio, cinc_ratio
- **FEs**: dyad_id (state_a_state_b pair), decade
- **Clustering**: atopid (alliance-level)

Tests H2A (voice_driven > 0) and H2B (hierarchical > voice_driven).

#### Why Dyad FE (Not Country FE)?

Following Gannon (2023), we use **dyad fixed effects** instead of country FE. This matters enormously:

| FE Specification | hierarchical coef | p-value | Result |
|------------------|-------------------|---------|--------|
| Country FE (state_a + state_b) | 0.002 | 0.82 | NULL |
| Dyad FE (dyad_id) | 0.015 | 0.0001 | SIGNIFICANT*** |

**Why the difference?**
1. **Country FE** identifies from variation across different dyads involving the same countries (between-dyad + within-dyad variation)
2. **Dyad FE** identifies ONLY from within-dyad variation over time (dyads that change institution type)
3. About 38% of dyads (1,636 of 4,337) change institution type, providing identifying variation for dyad FE
4. Dyad FE controls for all time-invariant dyad characteristics (geography, history, baseline complementarity)
5. The positive effect comes from: when dyads become MORE hierarchical, their division of labor INCREASES

### model_h2_event_study (H2 Event Study)
```
div_labor ~ event_time_dummies + C(state_a) + C(state_b) + C(year)
```
Tests whether **dyad-level division of labor** changes around alliance formation.
Uses full division_of_labor data (all country pairs) to track same dyads before/after entering alliance.

---

## CRITICAL CONCEPTUAL DISTINCTION

**Division of Labor** and **Specialization** are DIFFERENT concepts at DIFFERENT units of analysis:

| Concept | Definition | Unit | Variable |
|---------|------------|------|----------|
| **Division of Labor** | Complementarity between TWO partners | Dyad-year | `div_labor` |
| **Specialization** | Concentration of ONE country's portfolio | Country-year | `spec_y` |

### Division of Labor (Dyad-Level)
- Requires TWO entities - a single country cannot have "division of labor"
- Measures how COMPLEMENTARY two portfolios are
- `div_labor = 1 - Σ_m min(p_im, p_jm)` (pairwise dissimilarity)
- High div_labor = partners have DIFFERENT capabilities (one has tanks, other has aircraft)
- **H2 tests this**: Does alliance type predict div_labor?

### Specialization (Country-Level)
- Property of a SINGLE entity
- Measures how NARROW/FOCUSED a portfolio is
- High spec_y = country concentrates on few capability types
- **H1 tests this**: Does ideology predict spec_y?

### Why This Matters
You CANNOT test "division of labor" at the country level because it's inherently a relationship between TWO entities. The correct setup is:
- **H1**: Ideology → Specialization (country-year, DV = spec_y)
- **H2**: Alliance type → Division of Labor (dyad-year, DV = div_labor)

---

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

- R&R `inst` variable: 1=uninst, 2=voice, 3=hierarchical (NOT 0/1/2)
- Some ATOP exit years are 0 (meaning still active) → treat as 2014 (YEAR_END)
- Specialization data runs 1970-2014, limit dyad expansion accordingly
- Paths are absolute (resolved from package root via `_ROOT`)
- Use `paths.validate()` to check all input files exist before running
- ATOP provisions vary by member — use `groupby().max()` not `drop_duplicates()`
