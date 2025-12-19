# Critical Fix: Lagged Ideology Specification

## Problem Identified

The original analysis used **current-year ideology** to predict **current-year military specialization**. This is fundamentally misspecified for three reasons:

### 1. Force Structure is Sticky
Military procurement, training, and organizational structure take **2-5 years** to meaningfully change. A government elected in year *t* cannot realistically alter force composition by year *t*.

### 2. Simultaneity Bias
Measuring both ideology and specialization in the same year creates:
- **Reverse causality**: Military structure might affect electoral outcomes
- **Common shocks**: External threats could influence both elections and military spending in the same year

### 3. Policy Implementation Lag
Even with political will:
- Defense budgets are planned 1-2 years in advance
- Weapons procurement has multi-year lead times
- Organizational restructuring requires time for training, doctrine changes, etc.

---

## Solution: Lagged Ideology

The analysis now uses **2-year lagged ideology** as the primary specification:

```
spec_stand[t] = β₀ + β₁·ideology[t-2] + controls[t] + country_FE + year_FE + ε[t]
```

### Why 2-Year Lag?

- **1 year**: Too short for procurement and structural changes
- **2 years**: Captures policy implementation after first budget cycle
- **3+ years**: Risks measuring long-run effects that may reflect multiple government transitions

---

## What Changed in the Code

### 1. Added Lag Creation (`analysis.py:39-46`)

```python
# Create 1-5 year lags of ideology (grouped by country)
for lag in range(1, 6):
    df[f"rile_lag{lag}"] = df.groupby("country_code_cow", sort=False)["rile"].shift(lag)
    df[f"right_of_center_lag{lag}"] = df.groupby("country_code_cow", sort=False)["right_of_center"].shift(lag)
```

**Critical**: Lags are computed **within countries** using `groupby()` to prevent cross-country contamination.

### 2. Updated Primary Models

All models now use `right_of_center_lag2` or `rile_lag2` instead of current-year ideology.

### 3. Added Lag Sensitivity Analysis

The analysis now tests **all lag structures (1-5 years)** to show robustness:

```python
for lag in range(1, 6):
    # Test binary and continuous ideology at each lag
    _run_ols(..., f"right_of_center_lag{lag}", ...)
    _run_ols(..., f"rile_lag{lag}", ...)
```

This allows you to:
- Show whether effects emerge immediately or with delay
- Identify the "correct" lag structure empirically
- Demonstrate robustness to specification choices

---

## Implications for Results

### Sample Size
- **Original**: 916 observations with current-year ideology
- **2-year lag**: 891 observations (small reduction due to losing first 2 years per country)

### Interpretation
- **Before**: "Right-wing governments have more specialized militaries"
  - **Problem**: Could be reverse causality or simultaneity

- **After**: "Countries become more specialized 2 years after electing right-wing governments"
  - **Better**: Temporal ordering supports causal interpretation
  - **Still limited by**: Selection bias, omitted variables, parallel trends assumption

---

## Preliminary Results (2-Year Lag Specification)

From the test run with cubic time trends + allies' military spending:

```
right_of_center_lag2:  β = 0.162,  SE = 0.070,  p = 0.020
```

**Interpretation**: Countries with right-wing governments (RILE ≥ 10) show specialization scores **0.16 standard deviations higher** two years later, controlling for GDP, CINC, democracy, war, alliance spending, country FE, and cubic time trends.

**Magnitude**: Small but statistically significant at p < 0.05 level.

---

## Next Steps for Robustness

1. **Examine lag sensitivity results**
   - Do effects appear at lag 1? Lag 3? Lag 5?
   - Is there a clear pattern or are results noisy across lags?

2. **Event study analysis** (recommended)
   - Plot coefficients for lags -5 to +5 around ideological switches
   - Check for pre-trends (should be flat before switch)
   - Visualize dynamic effects (how effect evolves over time)

3. **Granger causality test**
   - Does ideology[t-k] predict specialization[t] conditional on specialization[t-1]?
   - Formal test of temporal precedence

4. **Discuss why 2 years**
   - Review political science literature on policy implementation lags
   - Consider defense budget cycles in your sample countries
   - Interview with defense policy experts (if feasible)

---

## Literature Support

**Policy Implementation Lags:**
- Clark, David H. (2000). "Trading Butter for Guns: Domestic Imperatives for Foreign Policy Substitution." *Journal of Conflict Resolution* 44(1): 7-31.
- Fordham, Benjamin O. (2002). "Another Look at 'Parties, Voters, and the Use of Force Abroad.'" *Journal of Conflict Resolution* 46(4): 572-596.

**Defense Procurement Timelines:**
- Dombrowski, Peter, and Eugene Gholz (2006). *Buying Military Transformation: Technological Innovation and the Defense Industry*. Columbia University Press.

**Force Structure Stickiness:**
- Avant, Deborah D. (1994). *Political Institutions and Military Change*. Cornell University Press.

---

## Bottom Line

Using **lagged ideology is not optional** — it's essential for credible causal inference. The original specification with current-year ideology would be rejected by reviewers at any serious political science journal.

The 2-year lag:
- ✅ Respects temporal ordering (cause precedes effect)
- ✅ Allows time for policy implementation
- ✅ Reduces simultaneity bias
- ✅ Aligns with institutional realities of defense budgeting

Combined with country and year fixed effects, this substantially strengthens (but doesn't fully establish) causal claims.
