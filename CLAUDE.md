# Military Specialization & Alliance Institutions Study

## Executive Summary

This project examines how **domestic politics** and **international institutions** shape military force structure among democratic states. It tests three hypotheses:

1. **H1**: Right-of-center ruling parties lead to *less* military specialization (broader portfolios)
2. **H2**: More institutionalized alliances lead to *greater* division of labor between partners
3. **H3**: Ideologically similar alliance partners exhibit *greater* division of labor

The analysis uses panel data methods (two-way fixed effects, clustered standard errors) to identify effects while controlling for unobserved heterogeneity.

---

## Gannon Replication + Explicit Deviations

This analysis follows **Gannon (2023)** methodology as closely as possible, with **four explicit deviations** (D1-D4) that are theoretically motivated and clearly documented.

### Explicit Deviations from Gannon

| Deviation | Description | Rationale |
|-----------|-------------|-----------|
| **D1** | Alliance governance is treated as a **3-category NOMINAL** concept (not ordinal) | Categories represent qualitatively different governance modes, not additive levels |
| **D2** | `SUBORD` is classified as **HIERARCHICAL** | Explicit subordination of forces indicates authority-based governance |
| **D3** | Main contrasts are **voice > uninst** and **hier > uninst** (reference = uninstitutionalized, not DCA-only) | Tests theory that ATOP treaty provisions matter, not just any security tie |
| **D4** | **MAIN** tests use **ATOP-only** sample; **UNION** (ATOP + DCAD) is **robustness only** | ATOP provides cleaner institutionalization coding from treaty provisions |

### What Aligns with Gannon

- Dyad-year unit of analysis with max(inst) collapse
- Dyad FE + Decade FE specification
- Clustered standard errors by dyad
- Bounded parity ratios for controls (computed on LEVELS per Gannon)
- Division of labor measure (portfolio dissimilarity)
- 1980-2010 window (DCAD alignment)

---

## Data Sources

### Input Datasets

| File | Source | Contents | Coverage |
|------|--------|----------|----------|
| `03_DF-full.rds` | Gannon (2023) | Military specialization scores | 1970-2014 |
| `MPDataset_MPDS2025a.csv` | Manifesto Project | Party ideology (RILE scores) | 1945-2023 |
| `atop5_1m.csv` | ATOP v5 | Alliance membership & provisions | 1815-2018 |
| `division_of_labor.csv` | Computed from rDMC | Pairwise portfolio dissimilarity | 1970-2014 |
| `contdird.csv` | COW | Direct contiguity | 1816-2016 |
| `NMC-60-wsupplementary.csv` | COW NMC 6.0 | Military expenditure | 1816-2016 |
| `DCAD-v1.0-dyadic.csv` | DCAD | Defense Cooperation Agreements | **1980-2010 only** |

### CRITICAL: DCAD Coverage Window

**DCAD only covers 1980-2010.** Outside this window, DCA status is **UNKNOWN**, not absent.

- **Years 1980-2010**: `any_dca_link` = 0 or 1 (observed)
- **Years < 1980 or > 2010**: `any_dca_link` = NA (unknown status)

The primary H2/H3 analyses use the **1980-2010 aligned sample** where DCAD is observed. Legacy specifications using 1970-2014 treat out-of-window DCA as NA.

### Constructed Datasets

```
results/
├── master_country_year.csv                  # H1 analysis dataset (1970-2014)
├── master_dyad_year.csv                     # Full dyad-year panel (1970-2014, DCA=NA outside 1980-2010)
├── master_dyad_year_h3.csv                  # H3 dataset with ideology vars (1970-2014)
├── master_dyad_year_gannon_1980_2010.csv    # ATOP-only aligned sample (1980-2010)
└── master_dyad_year_gannon_union_1980_2010.csv  # PRIMARY: ATOP OR DCAD aligned (1980-2010)
```

**Key distinction (per D4):**
- `master_dyad_year.csv`: Full panel, but `any_dca_link = NA` outside 1980-2010
- `master_dyad_year_gannon_1980_2010.csv`: **MAIN SAMPLE** — ATOP-only dyads (1980-2010)
- `master_dyad_year_gannon_union_1980_2010.csv`: **ROBUSTNESS ONLY** — Dyads in ATOP OR DCAD (1980-2010)

### H2 Sample Methodology (per D4)

**MAIN TESTS use ATOP-only sample** (`master_dyad_year_gannon_1980_2010.csv`):

**Sample Construction:**
1. **ATOP offense/defense pacts only** — excludes neutrality, non-aggression, consultation-only
2. Window restricted to **1980-2010** for DCAD alignment
3. Institutionalization coded from ATOP treaty provisions (Leeds & Anac 2005)

**Institutionalization Coding (within ATOP-only sample):**
- `inst = 1` = Uninstitutionalized (reference category per D3)
- `inst = 2` = Voice-driven
- `inst = 3` = Hierarchical (includes SUBORD per D2)

For regressions, recode to `inst_012` where uninst=0, voice=1, hier=2:
```python
df["inst_012"] = df["inst"] - 1  # 1→0, 2→1, 3→2
```

**Main H2 Regression Specification (per D3):**
```
div_labor ~ voice + hier + controls + C(dyad_id) + C(decade)
```
- Reference: Uninstitutionalized ATOP (inst_012=0)
- H2A test: β_voice > 0
- H2B test: β_hier > 0

**ROBUSTNESS: UNION Sample** (`master_dyad_year_gannon_union_1980_2010.csv`):

The UNION sample includes ATOP OR DCAD dyad-years and uses `vertical_integration`:
- `vertical_integration = 0` = DCA-only (no ATOP treaty)
- `vertical_integration = 1` = Uninstitutionalized ATOP
- `vertical_integration = 2` = Voice-driven ATOP
- `vertical_integration = 3` = Hierarchical ATOP

**IMPORTANT**: In UNION robustness tests, DCA-only is kept as a DISTINCT category (not pooled with uninst ATOP).

**Multiple Alliances:**
- When a dyad shares multiple ATOP alliances in the same year, take the **maximum** institutionalization score
- This follows the logic that the most institutionalized arrangement sets the coordination ceiling

---

## Variable Definitions

### Dependent Variables

#### Specialization Index (H1)

**Variable**: `spec_y`

**Source**: Gannon (2023), computed from rDMC military technology counts

**Interpretation**: Higher values = more concentrated military portfolio (fewer capability types). The variable is standardized in the source data.

#### Division of Labor (H2, H3)

**Variable**: `div_labor`

**Formula**:
```
For countries i and j with portfolio shares p_im and p_jm across M technology types:
  overlap = Σ min(p_im, p_jm) for m = 1 to M
  div_labor = 1 - overlap
```

**Range**: 0 (identical portfolios) to 1 (non-overlapping portfolios)

**Interpretation**: Higher values = partners have more complementary/different military portfolios.

### Independent Variables

#### Ideology (H1)

**Variable**: `rile` (Manifesto Project Right-Left Index)

**Range**: -100 (extreme left) to +100 (extreme right)

**Lagged version**: `rile_lag5` — the RILE score from 5 years prior. Created by:
```python
df["rile_lag5"] = df.groupby("country_code_cow")["rile"].shift(5)
```

**Binary version**: `right_of_center` — equals 1 if RILE ≥ 10, equals 0 if RILE ≤ -10, missing otherwise.

#### Alliance Institutionalization (H2)

**CRITICAL PROVENANCE CONSTRAINT**: Alliance institutionalization (`inst`) is **ALWAYS recalculated from raw ATOP treaty provisions** using Leeds & Anac (2005) logic. **NEVER use pre-coded inst from external sources.** NO pre-coded alliance-type variables from RR (Rapport-Rathbun) or any external governance dataset are used.

---

##### Variable Type

`inst` is a **THREE-CATEGORY NOMINAL** variable (not ordinal):
- **1** = Uninstitutionalized
- **2** = Voice-driven
- **3** = Hierarchical

Interpret categories as **mutually exclusive governance types**, not additive levels. Authority-based governance dominates voice; voice dominates absence.

##### ATOP Variables Used

**Required (all must be present)**:
`INTCOM`, `MILCON`, `BASE`, `SUBORD`, `ORGAN1`, `ORGPURP1`, `ORGAN2`, `ORGPURP2`, `MILAID`, `CONTRIB`

**Forbidden (must NOT be used)**:
`CONWTIN`, `MEDARB`, `ARMRED`, `ACQTERR`, `DIVGAINS`, `DEMWITH`

##### Helper Variable

```
mil_org_present = (ORGAN1 in {1,2,3} AND ORGPURP1 == 1) OR (ORGAN2 in {1,2,3} AND ORGPURP2 == 1)
```

##### Classification Logic (applied in order)

**(3) HIERARCHICAL** — Code as hierarchical if the alliance includes ANY provision that creates authority, command, or structural control of one member by another:
- `INTCOM == 1` — Integrated command in peacetime and wartime
- `MILCON == 3` — Common defense policy (doctrine, training, procurement, joint planning)
- `BASE > 0` — Joint or unilateral troop placement / basing
- `SUBORD in {1, 2}` — Explicit subordination of one party's forces

*Note*: Hierarchical alliances may also include consultation, coordination, organizations, training, or contribution rules; they are classified as hierarchical because authority overrides voice.

**(2) VOICE-DRIVEN** — Code as voice-driven **ONLY IF** the alliance is NOT hierarchical AND includes at least one coordination or consultation mechanism:
- `MILCON == 2` — Peacetime military consultation
- `mil_org_present` — Formal military coordinating organization present
- `MILAID in {3, 4}` — Training and/or technology transfer
- `CONTRIB == 1` — Specified troop, supply, or funding contributions

**(1) UNINSTITUTIONALIZED** — Code as uninstitutionalized if the alliance includes NONE of the hierarchical or voice-driven provisions above.

##### Missing Value Handling

Missing treaty provisions are treated as **0 (provision absent)** following ATOP coding conventions. This is explicitly documented behavior, not silent coercion.

##### Code Implementation

The inst variable is calculated in `_build_institutionalization()` in `build_datasets.py`. The function includes defensive assertions that:
1. Verify all required ATOP provision columns exist
2. Reject any pre-coded `inst` column in the input
3. Warn if forbidden columns are present
4. Assert output values are only {1, 2, 3}

---

**In regressions**: Binary dummies `hierarchical` (=1 if inst==3) and `voice_driven` (=1 if inst==2). Reference category is uninstitutionalized.

**Provenance assertion**: The pipeline includes fail-fast checks (`_assert_no_forbidden_columns`) that will raise `ValueError` if any RR-derived columns are detected in the output datasets.

#### Ideological Distance (H3)

**Variable**: `ideo_dist`

**Formula**: `ideo_dist = |rile_a - rile_b|`

**Lagged version**: `ideo_dist_lag5` — created by:
```python
df["ideo_dist_lag5"] = df.groupby("dyad_id")["ideo_dist"].shift(5)
```

### Control Variables

#### Country-Level (H1)

| Variable | Definition |
|----------|------------|
| `lngdp` | Log GDP from World Development Indicators |
| `cinc` | COW Composite Index of National Capability |
| `war5_lag` | Binary: interstate war in past 5 years |
| `in_hierarchical` | Binary: in any hierarchical alliance |
| `in_voice` | Binary: in any voice-driven alliance |
| `in_uninst` | Binary: in any uninstitutionalized alliance |

#### Dyad-Level (H2, H3)

| Variable | Definition |
|----------|------------|
| `contiguous` | Binary: land contiguity (COW conttype = 1) |
| `gdp_ratio` | **PRIMARY** — `min(gdp_level_a, gdp_level_b) / max(gdp_level_a, gdp_level_b)` — bounded in (0, 1] |
| `milex_ratio` | `min(milex_a, milex_b) / max(milex_a, milex_b)` — bounded in (0, 1] |
| `lngdp_ratio` | **LEGACY** — `min(lngdp_a, lngdp_b) / max(lngdp_a, lngdp_b)` — kept for backward compatibility |

**CRITICAL: Ratio Computation on LEVELS**

Following Gannon (2023), parity ratios must be computed on **LEVELS**, not logs:

```python
# PRIMARY: gdp_ratio on GDP LEVELS (Gannon style)
gdp_min = panel[["gdp_level_a", "gdp_level_b"]].min(axis=1)
gdp_max = panel[["gdp_level_a", "gdp_level_b"]].max(axis=1)
valid_mask = (gdp_min > 0) & (gdp_max > 0)
panel["gdp_ratio"] = np.nan
panel.loc[valid_mask, "gdp_ratio"] = gdp_min[valid_mask] / gdp_max[valid_mask]

# Assertion: ratios must be in (0, 1]
assert ((panel["gdp_ratio"].dropna() > 0) & (panel["gdp_ratio"].dropna() <= 1)).all()

# Similarly for milex_ratio
milex_min = panel[["milex_a", "milex_b"]].min(axis=1)
milex_max = panel[["milex_a", "milex_b"]].max(axis=1)
valid_mask = (milex_min > 0) & (milex_max > 0)
panel["milex_ratio"] = np.nan
panel.loc[valid_mask, "milex_ratio"] = milex_min[valid_mask] / milex_max[valid_mask]
```

**Interpretation**: Ratio = 1 means equal partners; ratio → 0 means highly asymmetric. This bounded [0, 1] formulation is symmetric (invariant to dyad ordering) and avoids extreme values.

---

## Data Lineage: Variable Sources

**CRITICAL**: Alliance institutionalization is coded from ATOP provisions only (Leeds & Anac logic). No pre-coded alliance-type variables from RR (or any external governance dataset) are used.

| Variable | Source | Derivation |
|----------|--------|------------|
| `inst` / `inst_max` | ATOP v5 | Leeds & Anac (2005) rules → max(inst) across shared alliances |
| `inst_012` | Derived | `= inst - 1` (0=uninst, 1=voice, 2=hier) for ATOP-only regressions |
| `hierarchical` | Derived | `= 1 if inst == 3` |
| `voice_driven` | Derived | `= 1 if inst == 2` |
| `vertical_integration` | Derived | `= max(inst)` for ATOP, `= 0` for DCA-only (UNION sample only) |
| `any_dca_link` | DCAD v1.0 | Binary: DCA agreement exists (1980-2010 only) |
| `any_atop_link` | ATOP v5 | Binary: share ATOP offense/defense pact |
| `div_labor` | Computed from rDMC | Pairwise portfolio dissimilarity |
| `contiguous` | COW contdird | Binary: land contiguity |
| `gdp_level` | World Development Indicators | Raw GDP (for ratio computation) |
| `gdp_ratio` | Derived | **PRIMARY** — Bounded min/max ratio on GDP LEVELS |
| `lngdp_ratio` | Derived | **LEGACY** — Bounded min/max ratio on log GDP |
| `milex_ratio` | COW NMC 6.0 | Bounded min/max military expenditure ratio |
| `rile` | Manifesto Project | Left-right ideology score |
| `ideo_dist` | Derived | `= |rile_a - rile_b|` |

---

## Statistical Tests: Detailed Procedures

### H1: Ideology → Specialization

#### Primary Test: `model_h1_master()`

**Research Question**: Does a country becoming more right-leaning correspond to less military specialization?

**Estimation Procedure**:
1. Load `master_country_year.csv`
2. Define required variables: `spec_y`, `rile_lag5`, `country_code_cow`, `year`, plus available controls (`lngdp`, `cinc`, `war5_lag`)
3. Drop rows with any missing values in required variables (listwise deletion)
4. Estimate OLS regression using `statsmodels.formula.api.ols()`

**Exact Formula**:
```
spec_y ~ rile_lag5 + lngdp + cinc + war5_lag + C(country_code_cow) + C(year)
```

**Standard Errors**: Clustered by `country_code_cow` using:
```python
model.fit(cov_type="cluster", cov_kwds={"groups": analysis["country_code_cow"]})
```

**Hypothesis Test**: Two-sided t-test on coefficient of `rile_lag5`. Expected: β < 0.

**Output**: `results/h1/model_h1_master.csv` containing coefficient, SE, t-stat, p-value, 95% CI, N, N_countries, R².

---

#### Robustness A: Placebo Lead Test

**Purpose**: Test for reverse causality. If ideology causes specialization (with a lag), then FUTURE ideology should NOT predict CURRENT specialization.

**Formula**:
```
spec_y ~ rile_lead1 + lngdp + cinc + war5_lag + C(country_code_cow) + C(year)
```

Where `rile_lead1` is ideology from the NEXT year:
```python
df["rile_lead1"] = df.groupby("country_code_cow")["rile"].shift(-1)
```

**Interpretation**: If p-value < 0.10, the placebo FAILS (suggests confounding or reverse causality). If p-value ≥ 0.10, the placebo PASSES.

---

#### Robustness B: Binary Ideology

**Purpose**: Test whether results hold with a binary left/right classification instead of continuous RILE.

**Formula**:
```
spec_y ~ roc_lag5 + lngdp + cinc + war5_lag + C(country_code_cow) + C(year)
```

Where `roc_lag5` is a 5-year lag of `right_of_center`:
```python
df["roc_lag5"] = df.groupby("country_code_cow")["right_of_center"].shift(5)
```

---

#### Robustness C: Control Sensitivity

**Purpose**: Test whether results are sensitive to control variable inclusion.

**Specifications compared**:
1. No controls: `spec_y ~ rile_lag5 + C(country_code_cow) + C(year)`
2. With controls: `spec_y ~ rile_lag5 + lngdp + cinc + war5_lag + C(country_code_cow) + C(year)`
3. Extended: `spec_y ~ rile_lag5 + lngdp + cinc + war5_lag + in_alliance + C(country_code_cow) + C(year)`

---

#### Event Study: `model_h1_event_study()`

**Purpose**: Test whether specialization changes systematically around ideology transitions.

**Event Definition**: A "transition to right" occurs when a country's RILE changes from negative to non-negative. A "transition to left" is the opposite.

**Procedure**:
1. Identify all ideology transitions in the data
2. For countries with transitions, use only their FIRST transition
3. Compute `event_time = year - event_year`
4. Keep observations where `event_time` is between -5 and +5
5. Create dummy variables for each event time (excluding t = -1 as reference)
6. Estimate:
```
spec_y ~ t_m5 + t_m4 + t_m3 + t_m2 + t_p0 + t_p1 + t_p2 + t_p3 + t_p4 + t_p5
         + controls + C(country_code_cow) + C(year)
```

**Standard Errors**: Clustered by country.

**Interpretation**:
- Pre-period coefficients (t < -1) test parallel trends. If significant, the design may be compromised.
- Post-period coefficients (t ≥ 0) estimate the treatment effect at each time point relative to t = -1.

---

#### Difference-in-Differences: `model_h1_did()`

**Purpose**: A more parsimonious estimate of the average post-transition effect.

**Formula**:
```
spec_y ~ post + controls + C(country_code_cow) + C(year)
```

Where `post = 1` if `event_time >= 0`, else 0.

---

### H2: Alliance Type → Division of Labor

#### Primary Test: Categorical Specification (per D1, D3, D4)

**Research Question**: Do more institutionalized alliances promote greater division of labor between partners?

**SAMPLE (per D4)**: ATOP-only offense/defense dyad-years, 1980-2010 (`master_dyad_year_gannon_1980_2010.csv`)

**Estimation Procedure**:
1. Load `master_dyad_year_gannon_1980_2010.csv` (ATOP-only sample)
2. Recode inst: `inst_012 = inst - 1` (0=uninst, 1=voice, 2=hier)
3. Create dummies: `voice = (inst_012 == 1)`, `hier = (inst_012 == 2)`
4. Reference category: Uninstitutionalized (inst_012 == 0) per D3
5. Drop rows with any missing values (listwise deletion)
6. Estimate OLS regression

**Exact Formula (per D3)**:
```
div_labor ~ voice + hier + contiguous + gdp_ratio + milex_ratio + C(dyad_id) + C(decade)
```

Note: `gdp_ratio` is computed on GDP LEVELS per Gannon.

**Standard Errors**: Clustered by `dyad_id` using:
```python
model.fit(cov_type="cluster", cov_kwds={"groups": analysis["dyad_id"]})
```

**Hypothesis Tests (per D3)**:
- **H2A**: t-test on `voice`. Expected: β > 0 (voice-driven > uninstitutionalized)
- **H2B**: t-test on `hier`. Expected: β > 0 (hierarchical > uninstitutionalized)
- **Supplementary**: Wald test on `hier - voice = 0` to compare hier vs voice

The Wald test uses the model covariance matrix for proper SE of differences:
```python
model.wald_test("hier - voice = 0", scalar=True)
# OR compute from covariance matrix:
cov_matrix = model.cov_params()
var_diff = cov_matrix.loc["hier", "hier"] + cov_matrix.loc["voice", "voice"] - 2 * cov_matrix.loc["hier", "voice"]
se_diff = np.sqrt(var_diff)
```

**Output**: `results/h2/model_h2_primary.csv`, `results/h2/h2_main_sample_ids.csv`

---

#### Supplementary: Ordinal Specification (per D1)

**Per D1 (deviation)**: This ordinal test is SUPPLEMENTARY only. The categorical test above is the primary theory test.

**Formula**:
```
div_labor ~ inst_012 + contiguous + gdp_ratio + milex_ratio + C(dyad_id) + C(decade)
```

**Interpretation**: Effect of moving up one institutionalization level (0→1 or 1→2).

---

#### Robustness: UNION Sample (per D4)

**UNION sample is for ROBUSTNESS only**. Uses `master_dyad_year_gannon_union_1980_2010.csv` with DCA-only as reference:

**Formula**:
```
div_labor ~ uninst + voice + hier + contiguous + gdp_ratio + milex_ratio + C(dyad_id) + C(decade)
```

**IMPORTANT**: Uninstitutionalized ATOP (vi=1) and DCA-only (vi=0) are kept as DISTINCT categories, never pooled.

---

#### Diagnostic: Placebo Test

**Purpose**: Test for reverse causality. If future institutionalization predicts current div_labor, suggests confounding.

**Formula**:
```
div_labor ~ inst_012_lead1 + controls + C(dyad_id) + C(decade)
```

Where `inst_012_lead1` is institutionalization from the NEXT year. Expected: β ≈ 0 (p ≥ 0.10 = PASS).

---

#### Heterogeneity: Capability Asymmetry

**Purpose**: Test whether institutionalization effects differ by partner capability parity.

Split sample by `milex_ratio` median:
- High parity (milex_ratio ≥ median): Similar capabilities
- Low parity (milex_ratio < median): Asymmetric capabilities

Estimate categorical specification separately for each subsample.

---

### H3: Ideological Symmetry → Division of Labor

#### Primary Test: `model_h3_master()`

**Research Question**: Does ideological similarity between alliance partners promote greater division of labor?

**Two Specifications**:

**Minimal**:
```
div_labor ~ ideo_dist_lag5 + C(dyad_id) + C(year)
```

**Full** (with Gannon-style controls):
```
div_labor ~ ideo_dist_lag5 + contiguous + gdp_ratio + milex_ratio + C(dyad_id) + C(year)
```

Note: Uses `gdp_ratio` (computed on GDP LEVELS per Gannon), not `lngdp_ratio`.

**Key Difference from H2**: H3 uses year FE (not decade FE) because ideological changes operate at a faster timescale than institutional changes.

**Standard Errors**: Clustered by `dyad_id`.

**Hypothesis Test**: t-test on `ideo_dist_lag5`. Expected: β < 0 (greater distance = less division of labor, i.e., similarity promotes coordination).

**Output**: `results/h3/model_h3_master.csv`

---

#### Robustness A: Categorical Symmetry

**Purpose**: Test with binary indicator instead of continuous distance.

**Formula**:
```
div_labor ~ same_bucket_10_lag5 + controls + C(dyad_id) + C(year)
```

Where `same_bucket_10_lag5 = 1` if both partners are in the same ideology bucket (both RoC, both LoC, or both Moderate) as of 5 years prior.

---

#### Robustness B: Alliance Type Controls

**Purpose**: Test whether H3 effect persists after controlling for alliance institutionalization.

**Formula**:
```
div_labor ~ ideo_dist_lag5 + hierarchical + voice_driven + contiguous + lngdp_ratio + milex_ratio
            + C(dyad_id) + C(year)
```

---

#### Robustness C: Exploratory Bucket Effects

**Purpose**: Explore whether specific ideology pairings (both RoC, both LoC, both Moderate) have distinct effects.

**Formula**:
```
div_labor ~ both_RoC_lag5 + both_LoC_lag5 + both_Mod_lag5 + controls + C(dyad_id) + C(year)
```

---

## Critical Data Processing Steps

### Dyad-Year Collapse (H2) — Safe Collapse with Invariance Checks

**Problem**: Some dyads share multiple alliances in the same year, creating duplicate rows.

**Solution**: Following Gannon (2023), we collapse to true dyad-years with **invariance checks**:

**Step 1: Identify column types**
- **Dyad-year invariant columns**: Must be identical across alliance rows (e.g., `div_labor`, `contiguous`, `lngdp_ratio`, `milex_ratio`, `has_dca`)
- **Alliance-row columns**: Can vary across rows (e.g., `atopid`, `inst`)

**Step 2: Check invariance before collapse**
```python
# For each invariant column, verify nunique(dropna=False) <= 1 within each dyad-year
_check_invariance(panel, dyad_year_invariant_cols, ["state_a", "state_b", "year"])
# Raises ValueError with details if any column varies
```

**Step 3: Collapse with safe aggregation**
```python
# Sort so most institutionalized alliance is first
panel = panel.sort_values(["state_a", "state_b", "year", "inst"], ascending=[True, True, True, False])

# Aggregation rules:
agg_dict = {
    "inst": "max",                    # Most institutionalized alliance wins
    "atopid": ["first", "nunique"],   # atopid_max_inst, n_shared_alliances
}
# Invariant columns: "first" (ONLY after invariance check passes)
for col in check_cols:
    agg_dict[col] = "first"
```

**Step 4: Post-collapse assertions**
- No duplicates on (state_a, state_b, year) AND (dyad_id, year)
- `n_shared_alliances` is integer >= 0

**Result**: Unique key is now (state_a, state_b, year) rather than (atopid, state_a, state_b, year). The invariance check ensures no silent data corruption from unsafe "first" aggregation.

### Ideology Lag Construction

**Country-level (H1)**:
```python
df = df.sort_values(["country_code_cow", "year"])
df["rile_lag5"] = df.groupby("country_code_cow")["rile"].shift(5)
```

**Dyad-level (H3)**:
```python
df["ideo_dist"] = (df["rile_a"] - df["rile_b"]).abs()
df = df.sort_values(["dyad_id", "year"])
df["ideo_dist_lag5"] = df.groupby("dyad_id")["ideo_dist"].shift(5)
```

### Bounded Ratio Construction

**CRITICAL: Compute on LEVELS, not logs (per Gannon)**

```python
# PRIMARY: GDP ratio on LEVELS (Gannon style)
gdp_min = df[["gdp_level_a", "gdp_level_b"]].min(axis=1)
gdp_max = df[["gdp_level_a", "gdp_level_b"]].max(axis=1)
valid_mask = (gdp_min > 0) & (gdp_max > 0)
df["gdp_ratio"] = np.nan
df.loc[valid_mask, "gdp_ratio"] = gdp_min[valid_mask] / gdp_max[valid_mask]

# Military expenditure ratio (bounded in (0,1])
milex_min = df[["milex_a", "milex_b"]].min(axis=1)
milex_max = df[["milex_a", "milex_b"]].max(axis=1)
valid_mask = (milex_min > 0) & (milex_max > 0)
df["milex_ratio"] = np.nan
df.loc[valid_mask, "milex_ratio"] = milex_min[valid_mask] / milex_max[valid_mask]

# ASSERTIONS: Ratios must be in (0, 1]
assert ((df["gdp_ratio"].dropna() > 0) & (df["gdp_ratio"].dropna() <= 1)).all()
assert ((df["milex_ratio"].dropna() > 0) & (df["milex_ratio"].dropna() <= 1)).all()

# LEGACY: lngdp_ratio (kept for backward compatibility, NOT used in main regressions)
lngdp_min = df[["lngdp_a", "lngdp_b"]].min(axis=1)
lngdp_max = df[["lngdp_a", "lngdp_b"]].max(axis=1)
df["lngdp_ratio"] = lngdp_min / lngdp_max.replace(0, np.nan)
```

---

## Fixed Effects and Clustering Summary

| Test | Unit FE | Time FE | Clustering |
|------|---------|---------|------------|
| H1 primary | Country (`C(country_code_cow)`) | Year (`C(year)`) | Country |
| H1 event study | Country | Year | Country |
| H2 primary | Dyad (`C(dyad_id)`) | Decade (`C(decade)`) | Dyad |
| H2 event study | State A + State B | Year | Dyad |
| H3 primary | Dyad (`C(dyad_id)`) | Year (`C(year)`) | Dyad |

---

## Output Files

### H1 Outputs (`results/h1/`)

| File | Contents |
|------|----------|
| `model_h1_master.csv` | Primary specification: rile_lag5 coefficient, SE, p-value, CI, N |
| `model_h1_robustness_placebo.csv` | Placebo test with rile_lead1 |
| `model_h1_robustness_binary.csv` | Binary ideology (roc_lag5) |
| `model_h1_robustness_sensitivity.csv` | Control sensitivity comparison |
| `model_h1_event_study_to_right.csv` | Event study coefficients for left→right transitions |
| `model_h1_event_study_to_left.csv` | Event study coefficients for right→left transitions |

### H2 Outputs (`results/h2/`)

| File | Contents |
|------|----------|
| `model_h2_type.csv` | Primary specification: hierarchical, voice_driven coefficients |
| `model_h2_event_study.csv` | Event study around alliance formation |

### H3 Outputs (`results/h3/`)

| File | Contents |
|------|----------|
| `model_h3_master.csv` | Minimal and Full specifications |
| `model_h3_robustness_categorical.csv` | Categorical symmetry (same_bucket_10_lag5) |
| `model_h3_robustness_with_inst.csv` | With alliance type controls |
| `model_h3_robustness_exploratory.csv` | Both_RoC, Both_LoC, Both_Mod effects |

### Audit Outputs (`results/audit/`)

| File | Contents |
|------|----------|
| `sample_attrition.csv` | Stage-by-stage sample flow |
| `alignment_distribution.csv` | DCA/ATOP indicator distribution |
| `variable_coverage.csv` | Missingness for key variables |
| `inst_distribution.csv` | Institution type distribution |
| `gannon_controls_summary.csv` | Summary statistics for control variables |

---

## Running the Analysis

```bash
# Full pipeline
uv run thesisizer

# Individual components
uv run thesisizer --build    # Build datasets only
uv run thesisizer --h1       # H1 analyses
uv run thesisizer --h2       # H2 analyses
uv run thesisizer --h3       # H3 analyses
```

---

## Code Structure

```
src/senior_thesis/
├── cli.py            # Entry point (thesisizer command)
├── config.py         # Paths, constants, control variable lists
├── build_datasets.py # Dataset construction
├── descriptives.py   # Summary statistics and figures
├── regressions.py    # All regression functions
└── hypotheses.py     # Orchestration (run_h1, run_h2, run_h3)
```

---

## Notes on Inference

### What the Tests Identify

- **H1**: Effect is identified from within-country changes in ideology over time, controlling for time-invariant country characteristics and common year shocks.

- **H2**: Effect is identified from dyads that change their institutional arrangement over time. Dyads that never change institution type contribute only to fixed effect estimation, not to the treatment effect.

- **H3**: Effect is identified from within-dyad changes in partner ideologies over time.

### Sample Restrictions

All analyses are restricted to democratic states because:
1. Manifesto Project covers only democracies
2. Ruling party ideology is most meaningful in competitive democracies
3. ATOP alliance sample focuses on democratic dyads

### Standard Error Clustering

Clustering accounts for serial correlation within clusters. The choice of cluster level reflects the unit at which observations are not independent:
- H1: Countries have persistent characteristics that create correlation across years
- H2/H3: Dyads have persistent relationships that create correlation across years
