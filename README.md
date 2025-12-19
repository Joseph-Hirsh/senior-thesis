# Senior Thesis: Military Specialization and Government Ideology

## Reproducible Data Pipeline for Country-Year Master Dataset

This repository contains a fully reproducible Python pipeline for constructing a country-year master dataset used in a political science thesis examining the relationship between government ideology (Right-of-Center vs Left-of-Center) and military specialization outcomes.

---

## Quick Start

### Prerequisites

- Python 3.13+ (tested on 3.13)
- Required packages (see `requirements.txt`)

### Installation

```bash
# Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run Full Pipeline

```bash
# Run all stages in sequence
python run_pipeline.py

# Or run individual stages
python stage0_load_and_inspect.py
python stage1_normalize_keys.py
python stage2_build_ideology_panel.py
python stage3_build_rr_aggregates.py
python stage4_merge_master.py
```

### Output

The pipeline produces:
- **`data/master_country_year.csv`** - Final master dataset (7,203 country-years × 160 variables)
- **`data/ideology_panel_country_year.csv`** - Intermediate ideology panel
- **`data/rr_aggregates_country_year.csv`** - Intermediate R&R alliance aggregates

---

## Project Structure

```
senior-thesis/
├── README.md                           # This file
├── DATA_DICTIONARY.md                  # Variable definitions and sources
├── requirements.txt                    # Python dependencies
├── run_pipeline.py                     # Master runner (executes all stages)
│
├── Pipeline Stages (run in order):
├── stage0_load_and_inspect.py         # Load datasets, print schemas
├── stage1_normalize_keys.py           # Normalize keys, audit uniqueness
├── stage2_build_ideology_panel.py     # Build government ideology panel
├── stage3_build_rr_aggregates.py      # Aggregate R&R alliance data
├── stage4_merge_master.py             # Merge into final master dataset
│
└── data/
    ├── Input datasets:
    ├── 03_DF-full.rds                  # Military specialization (spine/DV)
    ├── MPDataset_MPDS2025a.csv         # Manifesto Project (ideology)
    ├── partiestoanallianceR&R.dta      # R&R alliance dataset (dyad-year)
    │
    └── Output datasets:
        ├── master_country_year.csv           # FINAL OUTPUT
        ├── ideology_panel_country_year.csv   # Intermediate
        └── rr_aggregates_country_year.csv    # Intermediate
```

---

## Data Pipeline Overview

### Unit of Analysis
**Country-year** (one row per COW country code per year)

### Key Variables

**Dependent Variable:**
- `spec_stand` - Standardized military specialization measure

**Independent Variable (Primary):**
- `right_of_center_lag2` - Binary indicator for government ideology (2-year lag)
  - `1` if RILE[t-2] ≥ 10 (Right-of-Center)
  - `0` if RILE[t-2] ≤ -10 (Left-of-Center)
  - `NaN` if -10 < RILE[t-2] < 10 (centrist/unclear)
  - **NOTE**: Uses 2-year lag to account for policy implementation delays and force structure stickiness

**Controls:**
- Economic: `lngdppc_WDI_full`, `lngdp_WDI_full`
- Demographic: `lnpop_WDI_full`
- Political: `polity2_P4`, `democracy`
- Military: `cinc_MC`, `NATOmem_MEM`
- Geographic: `coastline_log`, `land_boundary_length_log`

**Alliance Variables (R&R):**
- `dyad_count` - Number of alliance dyads for country-year
- `rr_tot_rivals`, `rr_totmids2`, `rr_totmids5` - Conflict measures
- `rr_coldwar`, `rr_inst`, `rr_jntdem` - Alliance characteristics

### Data Sources

1. **Military Specialization Dataset** (`03_DF-full.rds`)
   - Source: IISS Military Balance
   - Coverage: 180 countries, 1970-2014
   - Role: Spine/DV dataset

2. **Manifesto Project Dataset** (`MPDataset_MPDS2025a.csv`)
   - Source: Manifesto Project Database 2025a
   - Coverage: 66 countries, 1920-2025 (elections)
   - Role: Government ideology (RILE scores)

3. **R&R Alliance Dataset** (`partiestoanallianceR&R.dta`)
   - Source: Parties to an Alliance (R&R)
   - Coverage: 64 countries, 1946-2003 (dyad-year)
   - Role: Alliance characteristics for robustness tests

---

## Pipeline Stages

### Stage 0: Load and Inspect
**Script:** `stage0_load_and_inspect.py`

- Loads all three input datasets
- Prints full schemas (columns, dtypes, non-null counts)
- Verifies data accessibility

**Output:** Console inspection only

---

### Stage 1: Normalize Keys and Audit Uniqueness
**Script:** `stage1_normalize_keys.py`

**Operations:**
1. Convert all keys to `int64` (from `float64`)
2. Build Manifesto → COW country code mapping (67 countries)
3. Extract election year from `edate` (fallback to `date` if missing)
4. Verify R&R state codes are COW codes (2=USA, 200=UK, etc.)
5. Audit key uniqueness (country-year combinations)
6. Print coverage statistics and dataset overlap

**Key Construction:**
- Military specialization: `country_code_cow`, `year` (already unique)
- Manifesto: Map `country` → `country_code_cow`, extract `election_year` from `edate`
- R&R: Verify `state_a`, `state_b` are COW codes; rename `yrent` → `year`

**Output:** Console audit reports

---

### Stage 2: Build Government Ideology Panel
**Script:** `stage2_build_ideology_panel.py`

**Operations:**
1. **Identify governing party** (largest by seat share)
   - Use `absseat / totseats` for seat share
   - Fallback to `pervote` if seat data missing
   - Select party with highest share for each country-election

2. **Extract RILE score** for governing party

3. **Build country-year panel** with forward-fill
   - Use military specialization spine (7,203 country-years)
   - Merge election data
   - Forward-fill RILE and party name between elections (within each country)

4. **Create RoC/LoC indicator**
   - `right_of_center = 1` if `rile >= 10`
   - `right_of_center = 0` if `rile <= -10`
   - `right_of_center = NaN` otherwise (centrist)

**Coverage:**
- RILE: 1,890/7,203 (26.2%)
- right_of_center: 1,119/7,203 (15.5%)

**Output:** `data/ideology_panel_country_year.csv`

---

### Stage 3: Build R&R Alliance Aggregates
**Script:** `stage3_build_rr_aggregates.py`

**Operations:**
1. **Handle duplicates**
   - Found: Czech/Slovakia 1993 (split year)
   - Resolution: Average duplicate observations

2. **Stack dyads** (dyad-year → monadic)
   - Create two rows per dyad: one for `state_a`, one for `state_b`
   - 165 dyads → 330 monadic observations

3. **Aggregate to country-year**
   - Group by `country_code_cow`, `year`
   - Aggregation rule: **mean** for all alliance features
   - Add `dyad_count` (number of contributing dyads)

4. **Rename variables** with `rr_` prefix
   - Prevents confusion in merged dataset
   - Example: `tot_rivals` → `rr_tot_rivals`

**Coverage:**
- 196 country-years
- 64 countries
- 21 years overlap with spine (1970-2014)

**Output:** `data/rr_aggregates_country_year.csv`

---

### Stage 4: Merge into Master Dataset
**Script:** `stage4_merge_master.py`

**Operations:**
1. **Merge ideology panel** (left join on spine)
   - All 7,203 spine rows preserved
   - Ideology added where available

2. **Merge R&R aggregates** (left join on spine)
   - R&R data added where available
   - Most rows have NaN (expected - sparse coverage)

3. **Audit merges**
   - Check match rates
   - Verify no duplicate keys
   - Print coverage by variable

4. **Assert key uniqueness** (CRITICAL)
   - Fails loudly if duplicates found
   - Verified: 0 duplicates ✓

**Output:** `data/master_country_year.csv` (7,203 rows × 160 columns)

---

## Coverage Summary

### Overall Dataset
- **Total:** 7,203 country-years
- **Countries:** 180
- **Period:** 1970-2014

### Variable Coverage

| Variable Category | Coverage | Notes |
|------------------|----------|-------|
| **DV (spec_stand)** | 90.0% | 6,481/7,203 |
| **Ideology (rile)** | 26.2% | 1,890/7,203 |
| **RoC indicator** | 15.5% | 1,119/7,203 (386 Right, 733 Left) |
| **GDP per capita** | 96.1% | 6,920/7,203 |
| **Population** | 99.0% | 7,129/7,203 |
| **Polity** | 93.8% | 6,755/7,203 |
| **R&R alliance** | 2.1% | 151/7,203 |

### Complete Observations
- **Baseline models** (DV + RoC + controls): **1,052 observations** (14.6%)
- **With R&R** (DV + RoC + R&R + controls): **54 observations** (0.7%)

### Coverage Over Time
| Year | Countries | With DV | With RoC | With R&R |
|------|-----------|---------|----------|----------|
| 1975 | 148 | 114 | 16 | 0 |
| 1985 | 151 | 140 | 19 | 0 |
| 1995 | 170 | 162 | 30 | 13 |
| 2005 | 171 | 161 | 34 | 0 |

---

## Data Quality Checks

The pipeline includes comprehensive audits:

✓ **Key uniqueness** - No duplicate country-years
✓ **Variable coding** - RoC indicator correctly coded from RILE
✓ **Forward-fill logic** - Ideology carries forward only between elections
✓ **Merge integrity** - All merges preserve spine structure
✓ **Value ranges** - All variables within expected bounds

---

## Notes for Replication

### Important Decisions (Documented for Transparency)

1. **Governing party identification**
   - Uses **largest party** by seat share (not necessarily majority or coalition)
   - Seat share preferred over vote share (more direct representation of power)
   - For presidential systems: still uses largest legislative party

2. **Ideology forward-fill**
   - RILE score **carries forward** within each country until next election
   - Does NOT carry forward across countries
   - Missing elections → gaps in coverage

3. **R&R aggregation**
   - Dyads stacked: each dyad contributes to BOTH member states
   - Aggregation: simple **mean** across all dyads
   - Transparent, symmetric, and reproducible

4. **Missing data handling**
   - No imputation (preserves missing patterns for analysis)
   - Left joins preserve all spine observations
   - Coverage varies by variable (documented above)

### Critical Specification Choices

1. **Lagged ideology (2 years)**
   - **Why**: Force structure takes 2-5 years to change (procurement, training, reorganization)
   - **Implication**: Current-year ideology would create simultaneity bias
   - **Evidence**: Defense budget cycles, organizational inertia
   - **Robustness**: Analysis tests lags 1-5 years to show sensitivity
   - See `LAGGED_IDEOLOGY_FIX.md` for detailed justification

### Known Limitations

1. **Ideology coverage limited to democracies**
   - Manifesto Project only covers multi-party elections
   - Authoritarian regimes have no RILE scores
   - Coverage improves over time (democratization)
   - **External validity**: Results generalize to democracies only, not all states

2. **R&R alliance data is sparse**
   - Only 21 years overlap with spine (1970-2014)
   - Small N for mechanism tests (use cautiously)
   - Sufficient for robustness checks, not primary analysis

3. **Germany 1990 excluded**
   - R&R has pre-unification data (1990)
   - Spine starts unified Germany (1991)
   - Merge correctly excludes this mismatch

4. **Causal identification assumptions**
   - Fixed effects require **parallel trends** (not directly tested yet)
   - Selection into democracy may be endogenous
   - Omitted variable bias still possible despite controls

---

## Citation

If you use this dataset or pipeline, please cite the original data sources:

1. **Military Specialization Data**: [Add citation]
2. **Manifesto Project Database 2025a**: Lehmann, Pola, et al. (2025)
3. **Parties to an Alliance (R&R)**: [Add citation]

---

## Contact

For questions about this pipeline:
- Joseph Hirsh
- [Add email/institution]

---

## License

[Add license information]

---

## Changelog

- **2025-01-XX**: Initial pipeline creation
  - Built 5-stage reproducible pipeline
  - Comprehensive documentation and auditing
  - Verified data quality checks
