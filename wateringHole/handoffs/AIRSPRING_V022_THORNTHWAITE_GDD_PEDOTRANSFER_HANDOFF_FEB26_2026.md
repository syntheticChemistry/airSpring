# airSpring V022 — Thornthwaite ET₀ + GDD + Pedotransfer Evolution Handoff

**Date**: 2026-02-26
**From**: airSpring v0.4.8
**To**: ToadStool / BarraCuda team
**ToadStool pin**: S68 (`f0feb226`)
**License**: AGPL-3.0-or-later
**Covers**: Three new experiment buildouts completing the ET₀ method portfolio, connecting thermal time to crop phenology, and adding continuous soil hydraulic property estimation

**airSpring**: 491 Rust tests + 570 validation + 1393 atlas checks, 27 binaries, 594/594 Python, 0 clippy errors

---

## Executive Summary

- **Exp 021**: Thornthwaite (1948) monthly ET₀ — temperature-only heat-index method. Python 23/23, Rust 50/50. Completes the 4-method ET₀ portfolio (PM, PT, HG, Thornthwaite).
- **Exp 022**: Growing Degree Days (GDD) + crop phenology — McMaster & Wilhelm (1997). Python 33/33, Rust 26/26. Bridges thermal time to Kc assignment, replacing fixed calendar dates with GDD-driven stages.
- **Exp 023**: Saxton & Rawls (2006) pedotransfer functions — continuous θ_wp, θ_fc, θ_s, Ksat from sand/clay/OM%. Python 70/70, Rust 58/58. Replaces texture-class lookup tables with regression equations for field-specific precision.
- **22 paper reproductions** validated with open data, 75/75 cross-validation, 11 Tier A GPU modules wired.
- **toadStool action**: Three new `batched_elementwise_f64` candidates — monthly ET₀ batch, GDD scan/accumulate, and pedotransfer per-sample regression.

---

## Part 1: New Domain Knowledge for BarraCuda

### Thornthwaite Monthly ET₀ — The 4th Method

Thornthwaite (1948) is the simplest ET₀ method, requiring only monthly mean temperature:

```
I = Σ (Tᵢ/5)^1.514          (annual heat index, 12 months)
a = 6.75e-7·I³ − 7.71e-5·I² + 1.792e-2·I + 0.49239
PET = 16 · (10·T/I)^a         (mm/month, unadjusted)
PET_adj = PET · (N/12) · (d/30)  (daylight and month-length correction)
```

**Why this matters for BarraCuda**: Thornthwaite is monthly-scale, meaning a full year is 12 elements — ideal for `batched_elementwise_f64` where each work item processes one station-year. The heat index I is a reduction over 12 months, then the exponent `a` and 12 monthly PET values are elementwise. This two-phase pattern (reduce → map) is a natural fit for ToadStool's streaming dispatch.

**Cross-method portfolio**: airSpring now validates 4 ET₀ methods:

| Method | Temporal | Inputs | Complexity | BarraCuda Primitive |
|--------|----------|--------|:----------:|---------------------|
| Penman-Monteith | Daily | T, RH, wind, Rs | High | `BatchedEt0` (GPU-FIRST) |
| Priestley-Taylor | Daily | T, Rs | Medium | `batched_elementwise_f64` op=PT |
| Hargreaves-Samani | Daily | Tmax, Tmin | Low | `batched_elementwise_f64` op=6 |
| Thornthwaite | Monthly | T_mean | Lowest | `batched_elementwise_f64` op=TH |

### Growing Degree Days — Thermal Time for Crop Scheduling

GDD accumulation links temperature to crop phenology. Two methods validated:

```
Method 1 (avg):   GDD = max(0, (Tmax + Tmin)/2 − Tbase)
Method 2 (clamp): GDD = max(0, (min(Tmax,Tceil) + max(Tmin,Tbase))/2 − Tbase)
```

Cumulative GDD determines crop growth stage → Kc value, replacing fixed calendar dates.

**Why this matters for BarraCuda**: GDD accumulation is a prefix-sum (scan) operation — ToadStool already has `wgsl_scan_f64` for this. The `kc_from_gdd` interpolation is a piecewise-linear lookup that can be implemented as a small LUT shader. This connects the physical ET₀ computation directly to crop water demand.

**Validated crop parameters**:

| Crop | Tbase (°C) | Maturity GDD | Kc_ini | Kc_mid | Kc_end |
|------|:----------:|:------------:|:------:|:------:|:------:|
| Corn | 10 | 2700 | 0.30 | 1.20 | 0.60 |
| Soybean | 10 | 2600 | 0.40 | 1.15 | 0.50 |
| Winter wheat | 0 | 2100 | 0.70 | 1.15 | 0.25 |
| Alfalfa | 5 | 800 | 0.40 | 1.20 | 1.15 |

### Saxton-Rawls Pedotransfer — Continuous Soil Properties

Saxton & Rawls (2006) regression equations estimate hydraulic properties from sand%, clay%, and organic matter%:

```
θ_wp = f(S, C, OM)     (wilting point at -1500 kPa)
θ_fc = f(S, C, OM)     (field capacity at -33 kPa)
θ_s  = f(S, C, OM)     (saturation/porosity)
Ksat = 1930 · (θ_s - θ_fc)^(3 - λ)
```

**Why this matters for BarraCuda**: These are pure arithmetic regressions — 3 inputs → 5 outputs per soil sample. For regional-scale simulations (100+ stations, each with unique soil), this is an embarrassingly parallel `batched_elementwise_f64` workload. Replaces the discrete texture-class lookup (`SoilTexture::hydraulic_properties()`) with continuous estimates.

**Validated across 8 USDA texture classes**: sand, loamy sand, sandy loam, loam, silt loam, clay loam, silty clay, clay. All satisfy physical ordering constraints (θ_wp < θ_fc < θ_s) and published ranges.

---

## Part 2: New Rust Code for Absorption Consideration

### `eco::evapotranspiration` — Thornthwaite (6 new pub functions)

```rust
pub fn monthly_heat_index_term(tmean_c: f64) -> f64
pub fn annual_heat_index(monthly_temps: &[f64; 12]) -> f64
pub fn thornthwaite_exponent(heat_index: f64) -> f64
pub fn thornthwaite_unadjusted_et0(tmean_c: f64, heat_index: f64, exponent_a: f64) -> f64
pub fn mean_daylight_hours_for_month(latitude_deg: f64, month_index: usize) -> f64
pub fn thornthwaite_monthly_et0(monthly_temps: &[f64; 12], latitude_deg: f64) -> [f64; 12]
```

Reuses existing `daylight_hours()` from FAO-56 module. The `[f64; 12]` return type is stack-allocated — no heap allocation for the common case.

### `eco::crop` — GDD + Phenology (6 new pub functions + 1 struct)

```rust
pub struct GddCropParams { tbase, tceil, maturity_gdd, kc_stages_gdd, kc_values }
pub fn gdd_avg(tmax: f64, tmin: f64, tbase: f64) -> f64
pub fn gdd_clamp(tmax: f64, tmin: f64, tbase: f64, tceil: f64) -> f64
pub fn accumulated_gdd_avg(daily_tmax: &[f64], daily_tmin: &[f64], tbase: f64) -> Vec<f64>
pub fn accumulated_gdd_clamp(daily_tmax: &[f64], daily_tmin: &[f64], tbase: f64, tceil: f64) -> Vec<f64>
pub fn kc_from_gdd(cum_gdd: f64, stages_gdd: &[f64], kc_values: &[f64]) -> f64
```

Plus `CropType::gdd_params()` returning GDD parameters for corn, soybean, wheat, alfalfa.

### `eco::soil_moisture` — Saxton-Rawls (1 new pub function + 2 structs)

```rust
pub struct SaxtonRawlsInput { sand: f64, clay: f64, om_pct: f64 }
pub struct SaxtonRawlsResult { theta_wp, theta_fc, theta_s, ksat_mm_hr, lambda: f64 }
pub fn saxton_rawls(input: &SaxtonRawlsInput) -> SaxtonRawlsResult
```

Internal helper functions (`sr_theta_1500`, `sr_theta_33`, `sr_theta_s_33`, etc.) match the published Saxton-Rawls regression coefficients exactly, validated against Python to 1e-4 tolerance per moisture parameter and 0.5 mm/hr for Ksat.

### Validation Binaries

| Binary | Checks | Key Validations |
|--------|:------:|-----------------|
| `validate_thornthwaite` | 50/50 | Heat index, exponent, 2-station monthly ET₀, monotonicity, edge cases, seasonal pattern |
| `validate_gdd` | 26/26 | Analytical avg/clamp, season accumulation (corn/alfalfa), Kc from GDD, method comparison |
| `validate_pedotransfer` | 58/58 | Loam intermediates, 8 USDA textures, physical ordering, OM sensitivity |

---

## Part 3: Evolution Path Review — All 22 Papers

### Python → BarraCuda CPU → BarraCuda GPU → metalForge

| # | Paper | Python | CPU Binary | GPU Path | metalForge |
|---|-------|:------:|:----------:|:--------:|:----------:|
| 1 | FAO-56 PM ET₀ | 64/64 | `validate_et0` 31/31 | `BatchedEt0` **GPU-FIRST** | metrics (absorbed S64) |
| 2 | Soil sensor calibration | 36/36 | `validate_soil` 40/40 | `fit_ridge` | regression (absorbed S66) |
| 3 | IoT irrigation pipeline | 24/24 | `validate_iot` 11/11 | `StreamSmoother` | moving_window (absorbed S66) |
| 4 | Water balance scheduling | 18/18 | `validate_water_balance` 13/13 | `BatchedWB` **GPU-STEP** | hydrology (absorbed S66) |
| 5 | Real data 100 stations | R²=0.967 | `validate_real_data` 23/23 | All Tier A | All (absorbed) |
| 6 | Dual Kc (Kcb+Ke) | 63/63 | `validate_dual_kc` 61/61 | Tier B (op=8) | hydrology |
| 7 | Regional ET₀ intercomp | 61/61 | `validate_regional_et0` 61/61 | `BatchedEt0` at scale | metrics |
| 8 | Cover crops + no-till | 40/40 | `validate_cover_crop` 40/40 | Tier B (op=8 + mulch) | hydrology |
| 9 | Richards equation | 14/14 | `validate_richards` 15/15 | `BatchedRichards` **WIRED** | VG (absorbed S40) |
| 10 | Biochar isotherms | 14/14 | `validate_biochar` 14/14 | `fit_*_nm` **WIRED** | isotherm (absorbed S64) |
| 11 | 60-year water balance | 10/10 | `validate_long_term_wb` 11/11 | `BatchedEt0` + `BatchedWB` | hydrology |
| 12 | Yield response | 32/32 | `validate_yield` 32/32 | Tier B (elementwise) | yield_response |
| 13 | CW2D Richards | 24/24 | `validate_cw2d` 24/24 | `BatchedRichards` | VG (CW2D media) |
| 14 | Irrigation scheduling | 25/25 | `validate_scheduling` 28/28 | `BatchedWB` + `BatchedEt0` | hydrology |
| 15 | Lysimeter ET | 26/26 | `validate_lysimeter` 25/25 | `BatchedEt0` (ground truth) | metrics |
| 16 | ET₀ sensitivity | 23/23 | `validate_sensitivity` 23/23 | `BatchedEt0` (perturbation) | metrics |
| 17 | Priestley-Taylor | 32/32 | `validate_priestley_taylor` 32/32 | Tier B (op=PT) | evapotranspiration |
| 18 | 3-method intercomp | 36/36 | `validate_et0_intercomparison` 36/36 | All 3 methods at scale | evapotranspiration |
| **19** | **Thornthwaite ET₀** | **23/23** | `validate_thornthwaite` **50/50** | **batch reduce→map** | evapotranspiration |
| **20** | **Growing Degree Days** | **33/33** | `validate_gdd` **26/26** | **scan + LUT** | crop phenology |
| **21** | **Pedotransfer (SR2006)** | **70/70** | `validate_pedotransfer` **58/58** | **elementwise 3→5** | soil_moisture |

### GPU Promotion Roadmap (updated)

| Priority | Candidate | BarraCuda Primitive | Inputs→Outputs | Notes |
|:--------:|-----------|--------------------:|:--------------:|-------|
| **P1** | PT ET₀ batch | `batched_elementwise_f64` op=PT | 4→1 | Simplest daily ET₀ |
| **P1** | HG ET₀ batch | `batched_elementwise_f64` op=6 | 3→1 | Temperature-only |
| **P1** | Pedotransfer batch | `batched_elementwise_f64` op=SR | 3→5 | Per-sample regression |
| **P2** | GDD accumulate | `wgsl_scan_f64` prefix-sum | N days→N cumulative | Prefix-sum pattern |
| **P2** | Kc from GDD | Piecewise-linear LUT | 1+stages→1 | Small uniform buffer |
| **P2** | Thornthwaite monthly | reduce→map (heat index then 12 PET) | 12→12+1 | Two-phase pattern |
| **P3** | Ensemble ET₀ | Multi-op dispatch (PM+PT+HG+TH) | N×varies | Method uncertainty |

---

## Part 4: Open Data Confirmation

All 22 papers use exclusively open data and systems:

| Data Source | Papers Using It | Access |
|-------------|:---------------:|--------|
| FAO-56 (open literature) | 1, 4, 6, 8, 12, 14, 16, 17, 20 | Free — published equations and tables |
| Open-Meteo ERA5 | 5, 7, 11, 18, 19 | Free — no API key, no account |
| Dong published tables | 2, 3, 13, 15 | Free — published in journals |
| Saxton & Rawls (2006) | 21 | Free — published regression coefficients |
| McMaster & Wilhelm (1997) | 20 | Free — published GDD formulas |
| Thornthwaite (1948) | 19 | Free — published equations |
| USDA Web Soil Survey | 21 | Free — public soil data |
| HYDRUS CW2D parameters | 13 | Free — published media parameters |

**Zero institutional access required. Zero synthetic data in the default pipeline.**

---

## Part 5: What ToadStool / BarraCuda Should Evolve

### From This Handoff

1. **Pedotransfer GPU op** — `batched_elementwise_f64` op=SR.
   3 inputs (sand, clay, OM%) → 5 outputs (θ_wp, θ_fc, θ_s, Ksat, λ).
   Pure arithmetic regression — the ideal candidate for demonstrating multi-output elementwise dispatch.
   airSpring has 58 validated test cases across 8 USDA texture classes.

2. **GDD scan** — `wgsl_scan_f64` prefix-sum over daily GDD values.
   ToadStool already has scan primitives; GDD accumulation maps directly.
   Combined with `kc_from_gdd`, this enables GPU-driven crop phenology tracking.

3. **Thornthwaite batch** — Two-phase GPU pattern: reduce 12 monthly temps → heat index I,
   then map 12 months → 12 PET values. Tests the reduce→map composition pattern.

### From Accumulated Learning (22 Papers)

4. **`ValidationHarness::to_json()`** — Machine-readable CI output.
   At 570 validation + 1393 atlas = 1963 total checks, structured output would
   enable automated regression detection across all springs.

5. **Columnar binary I/O** — CSV parsing remains the bottleneck at 100-station scale.
   A `barracuda::io` binary columnar format would eliminate parse overhead for GPU dispatch.

6. **`ComputeScheduler` integration** — airSpring's pipeline is the ideal proof-of-concept
   for metalForge mixed hardware: CPU for Richards PDE (branching) + GPU for batch ET₀ (parallel)
   + pedotransfer (embarrassingly parallel per soil sample).

---

## Part 6: Quality Gates

| Gate | Value |
|------|-------|
| ToadStool pin | S68 (`f0feb226`) |
| `cargo test --lib` | 491/491 PASS |
| Validation binaries | 27/27 PASS (570 checks) |
| Atlas checks | 1393/1393 PASS |
| Python baselines | 594/594 PASS (22 experiments) |
| `cargo clippy -- -D warnings` | 0 errors |
| `cargo fmt --check` | Clean |
| Cross-validation | 75/75 MATCH (tol=1e-5) |
| P0 blockers | None |

---

## Part 7: File Manifest

| File | Purpose |
|------|---------|
| `barracuda/src/eco/evapotranspiration.rs` | Thornthwaite monthly ET₀ (6 pub fns + 8 tests) |
| `barracuda/src/eco/crop.rs` | GDD accumulation + Kc from GDD (6 pub fns + 6 tests) |
| `barracuda/src/eco/soil_moisture.rs` | Saxton-Rawls pedotransfer (1 pub fn + 5 tests) |
| `barracuda/src/bin/validate_thornthwaite.rs` | 50/50 Thornthwaite validation |
| `barracuda/src/bin/validate_gdd.rs` | 26/26 GDD validation |
| `barracuda/src/bin/validate_pedotransfer.rs` | 58/58 pedotransfer validation |
| `control/thornthwaite/thornthwaite_et0.py` | Python Thornthwaite control + benchmark |
| `control/gdd/growing_degree_days.py` | Python GDD control + benchmark |
| `control/pedotransfer/saxton_rawls.py` | Python Saxton-Rawls control + benchmark |
| `barracuda/EVOLUTION_READINESS.md` | Updated: 27 binaries, 491 tests, v0.4.8 |
| `specs/PAPER_REVIEW_QUEUE.md` | Updated: 22 completed papers |

---

*airSpring v0.4.8 → ToadStool S68. 22 papers reproduced, 594/594 Python + 491 Rust tests
+ 570 validation + 1393 atlas checks. 27 binaries. All open data. Thornthwaite + GDD + pedotransfer
ready for GPU promotion. Evolution path: Python baseline → BarraCuda CPU → BarraCuda GPU → metalForge
mixed hardware. Write→Absorb→Lean cycle continues. AGPL-3.0-or-later.*
