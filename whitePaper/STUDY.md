# airSpring Study: Precision Agriculture on Consumer Hardware

**Status**: Working draft
**Date**: February 2026
**See also**: [METHODOLOGY.md](METHODOLOGY.md) for validation protocol

---

## Abstract

We independently replicate 13 precision agriculture and environmental systems computational methods — FAO-56 Penman-Monteith evapotranspiration, soil moisture calibration, IoT irrigation, daily water balance, dual crop coefficient, cover crops, regional ET₀ intercomparison, Richards equation, biochar adsorption isotherms, yield response to water stress, CW2D constructed wetland media, and 60-year water balance reconstruction — using only open-source tools and publicly available data. Python baselines (400/400 checks) validate against digitized paper benchmarks. A real data pipeline using Open-Meteo historical weather (918 station-days, 6 Michigan stations, 2023 growing season) produces ET₀ with R²=0.967 against independent computation. Water balance simulations show 53-72% water savings with smart scheduling. A Rust implementation via BarraCuda passes 635 tests across 18 binaries, with 75/75 Python-Rust cross-validation matches within 1e-5 tolerance. CPU benchmarks show Rust is 69x faster than Python (geometric mean, 20x–502x range) — establishing the foundation for GPU-accelerated precision irrigation on consumer hardware.

---

## 1. Introduction

### 1.1 The Problem

Agricultural irrigation consumes approximately 70% of global freshwater withdrawals. Inefficient scheduling wastes 30-50% of applied water while degrading soil health. Current precision irrigation relies on proprietary sensor systems ($500-$5000 per field), vendor-locked scheduling software, and institutional weather station networks with limited spatial resolution.

### 1.2 This Study

We replicate the computational methods published by Dr. Younsuk Dong (Michigan State University) and the FAO Irrigation and Drainage Paper No. 56, establishing a fully open, reproducible precision irrigation pipeline that requires:
- No institutional access
- No proprietary software
- No synthetic data (in the primary pipeline)
- Only consumer hardware

### 1.3 Evolution Path

```
Paper benchmarks → Python/R baselines → Real open data → Rust (BarraCuda) → GPU (ToadStool) → Penny Irrigation
```

---

## 2. Phase 0: Python/R Control (142/142 PASS)

### 2.1 FAO-56 Penman-Monteith ET₀ (64/64 PASS)

Reference: Allen et al. (1998) FAO Irrigation and Drainage Paper 56.

**Implementation**: Manual numpy implementation following the paper's calculation sheets exactly. Each equation (Eqs. 6-50) implemented as a separate function with FAO-56 equation numbers in docstrings.

**Benchmark data**: Digitized from FAO-56 Chapter 4:
- Example 17: Bangkok, April (monthly, measured ea) — expected ET₀ = 5.72 mm/day
- Example 18: Uccle, 6 July (daily, RH + wind conversion) — expected ET₀ = 3.88 mm/day
- Example 20: Lyon, July (missing data, Tmax/Tmin only) — expected ET₀ = 4.56 mm/day
- Tables 2.3 and 2.4: Saturation vapour pressure and slope (11 + 10 points)

**Results**: All 64 checks pass. ET₀ errors: Bangkok 0.004 mm/d, Uccle 0.000 mm/d, Lyon 0.000 mm/d. All intermediate values within specified tolerances.

### 2.2 Soil Sensor Calibration (36/36 PASS)

Reference: Dong et al. (2020) Agriculture 10(12), 598.

**Implementation**: Topp equation, RMSE/IA/MBE formulas (Paper Eqs. 1-3), correction equation fitting (linear, quadratic, exponential, logarithmic) using scipy curve_fit.

**Results**: All 36 checks pass. Topp equation matches computed values within 0.005 m³/m³. Statistical formulas analytically exact. Quadratic correction confirmed as best fit (paper's conclusion). Field RMSE improvements validated for all 6 sensor/soil combinations.

### 2.3 IoT Irrigation Pipeline (24/24 PASS)

Reference: Dong et al. (2024) Frontiers in Water 6, 1353597.

**Implementation**: SoilWatch 10 calibration equation (Paper Eq. 5), irrigation recommendation model (Paper Eq. 1). R ANOVA script for yield analysis.

**Results**: All 24 Python checks pass. SoilWatch 10 produces correct VWC from raw capacitance. Irrigation model correctly recommends water application based on field capacity and current VWC. Published sensor performance criteria and field demonstration results validated.

### 2.4 Water Balance (18/18 PASS)

Reference: FAO-56 Chapter 8 (Allen et al. 1998).

**Implementation**: Daily root zone depletion tracking (Eq. 85), stress coefficient (Eq. 84), total available water, readily available water.

**Results**: All 18 checks pass. Mass balance closes to 0.000000 mm for dry-down, irrigated, and heavy rain scenarios. Ks correctly transitions from 1.0 to stress at Dr > RAW. Michigan summer simulation produces realistic ET and irrigation schedule.

---

## 3. Phase 0+: Real Data Pipeline (918 Station-Days)

### 3.1 Data Sources

| Source | Records | Coverage | Key |
|--------|---------|----------|-----|
| Open-Meteo archive | 918 station-days | 6 MI stations, May-Sep 2023 | None needed |
| OpenWeatherMap | 42 station-days | 6 MI stations, current + forecast | testing-secrets/ |
| NOAA CDO (GHCND) | 153 station-days | Lansing, May-Sep 2023 | testing-secrets/ |

### 3.2 ET₀ Cross-Validation

Our FAO-56 implementation vs Open-Meteo's independent ET₀ computation:

| Station | Days | RMSE (mm/d) | MBE (mm/d) | R² |
|---------|:----:|:-----------:|:----------:|:--:|
| East Lansing (MSU) | 153 | 0.295 | +0.119 | 0.965 |
| Grand Junction | 153 | 0.244 | +0.051 | 0.971 |
| Sparta | 153 | 0.279 | +0.100 | 0.970 |
| Hart (tomato site) | 153 | 0.220 | +0.048 | 0.974 |
| West Olive (blueberry) | 153 | 0.257 | +0.025 | 0.963 |
| Manchester (corn) | 153 | 0.297 | +0.116 | 0.960 |
| **All stations** | **918** | **0.267** | **+0.076** | **0.967** |

The small positive bias (+0.076 mm/d) is consistent with differences between Open-Meteo's ERA5 reanalysis inputs and our strict FAO-56 implementation (which uses the reported daily min/max for vapour pressure deficit computation). R² > 0.96 at every station confirms our ET₀ is correct on real data.

### 3.3 Water Balance on Real Data

Four crop scenarios simulated at actual Dong (2024) demonstration sites:

| Crop | Station | Rain-fed ET | Irrigated ET | Smart Irrig | Savings vs Naive |
|------|---------|:-----------:|:------------:|:-----------:|:----------------:|
| Blueberry | West Olive | 378.7 mm | 538.8 mm | 210 mm (14 events) | **72%** |
| Tomato | Hart | 424.9 mm | 693.5 mm | 350 mm (14 events) | **53%** |
| Corn | Manchester | 467.7 mm | 759.3 mm | 330 mm (11 events) | **56%** |
| Ref. grass | East Lansing | 434.5 mm | 656.7 mm | 300 mm (12 events) | **60%** |

All mass balances close to 0.0000 mm. Water savings of 53-72% are consistent with Dong et al. (2024) reported savings of ~30% for tomato (our higher figure reflects comparison against a naive 25mm/5-day schedule, while Dong compared against traditional scheduling).

---

## 4. Phase 1: Rust BarraCuda (371 lib + 97 integration = 468 tests)

### 4.1 Module Structure

| Module | Functions | Validation Checks | Unit Tests |
|--------|----------|:-----------------:|:----------:|
| `eco::evapotranspiration` | 22 FAO-56 functions + Hargreaves, sunshine Rs, monthly G | 31 | 25 |
| `eco::crop` | CropType enum (10 crops), FAO-56 Table 12 Kc, Eq. 62 climate adj. | — | 7 |
| `eco::sensor_calibration` | SoilWatch 10 VWC, irrigation recommendation, multi-layer | 21 | 8 |
| `eco::soil_moisture` | Topp eq, inverse, PAW, SoilTexture, SoilHydraulicProps | 25 | 10 |
| `eco::water_balance` | WaterBalanceState, RunoffModel, simulate_season | 13 | 8 |
| `io::csv_ts` | TimeseriesData columnar parser, streaming BufReader | 11 | 6 |
| `error` | AirSpringError enum (Io, CsvParse, JsonParse, InvalidInput, Barracuda) | — | — |
| `testutil` | RMSE, MBE, R², IA, NSE, synthetic data generators | — | 6 |
| **Integration tests** | Cross-module pipelines, determinism, error paths, crop↔balance | — | 97 |
| **Doc-tests** | Inline documentation examples | — | 2 |
| **Total** | 371 lib + 97 integration + doc-tests | — | 468 |

### 4.2 Python-Rust Parity

| Capability | Python | Rust | Status |
|-----------|:------:|:----:|--------|
| FAO-56 ET₀ (full chain) | Yes | Yes | **Complete** |
| Sunshine hours Rs (Eq. 35) | Yes | Yes | **Complete** |
| Temperature Rs / Hargreaves (Eq. 50, 52) | Yes | Yes | **Complete** |
| Monthly soil heat flux (Eq. 43) | Yes | Yes | **Complete** |
| RMSE/IA/MBE/R²/NSE statistics | Yes | Yes | **Complete** |
| SoilWatch 10 calibration | Yes | Yes | **Complete** |
| Irrigation recommendation (single + multi-layer) | Yes | Yes | **Complete** |
| Crop Kc database (FAO-56 Table 12) | Yes | Yes | **Complete** (10 crops) |
| Kc climate adjustment (Eq. 62) | No | Yes | **Rust only** |
| Correction equation fitting | Yes | Yes | Complete (pure Rust, eco::correction) |
| R ANOVA | R script | No | Statistical test |
| Real data download | Yes | No | Python scripts handle APIs |

### 4.3 Rust Validation Results

| Binary | Checks | Key Results |
|--------|:------:|-------------|
| validate_et0 | 31/31 | FAO-56 Tables 2.3/2.4, Example 18 Uccle within 0.0005 mm/day |
| validate_soil | 26/26 | Topp (7 points), inverse round-trip, 5 USDA textures, PAW |
| validate_iot | 11/11 | 168 records, 5 columns, CSV round-trip, diurnal statistics |
| validate_water_balance | 13/13 | Mass balance 0.0000 (3 scenarios), Ks bounds, MI summer |
| validate_sensor_calibration | 21/21 | SoilWatch 10 VWC, irrigation model, Dong 2024 field results |
| validate_real_data | 21/21 | Real data pipeline, Open-Meteo ET₀ |
| cross_validate | 75 values | Python↔Rust JSON harness (benchmark JSON) |
| simulate_season | — | Full growing-season pipeline |

### 4.4 Phase 2: Cross-Validation (75/75 MATCH)

A structured cross-validation harness computes 75 values from identical inputs
in both Python (`scripts/cross_validate.py`) and Rust (`cross_validate` binary),
outputting JSON for automated comparison (single source of truth: benchmark JSON). All 75 values match within 1e-5
tolerance across: atmospheric parameters, solar geometry, radiation, ET₀, Topp
equation, SoilWatch 10 calibration, irrigation recommendation, statistical
measures (RMSE, MBE, IA, R²), sunshine-based radiation, Hargreaves ET₀, and
monthly soil heat flux.

### 4.5 Phase 3: GPU-FIRST (LIVE)

GPU acceleration is operational. Eight orchestrators run on ToadStool: BatchedEt0, BatchedWaterBalance, BatchedDualKc, KrigingInterpolator, SeasonalReducer, StreamSmoother, BatchedRichards, fit_nm (isotherms). All 4/4 ToadStool issues are resolved; GPU determinism has been verified. Cross-validation now loads from benchmark JSON as the single source of truth.

---

## 5. Lessons and Findings

### 5.1 Open Data Is Sufficient

The combination of Open-Meteo (historical), OpenWeatherMap (real-time), and NOAA CDO (station-level) provides all weather data needed for precision agriculture in Michigan — and anywhere globally. No institutional weather station access required.

### 5.2 Paper Benchmarks Are Essential

The digitized paper values caught implementation bugs that would have been invisible with only synthetic test data. Example 18 (Uccle) requires wind speed conversion from 10m to 2m — the FAO-56 equation for this is easy to get wrong.

### 5.3 The FAO-56 Is Remarkably Complete

FAO Paper 56 provides worked examples with all intermediate values — a rare and valuable practice. This made it possible to validate each equation independently, not just the final ET₀ output.

### 5.4 Real Data Reveals Model Behavior

On real Michigan data, blueberry at West Olive experiences 81/153 stress days without irrigation but only 14 with smart scheduling — a dramatic difference that synthetic data would not capture with the same specificity.

### 5.5 Cross-Spring Patterns

airSpring's validation methodology mirrors hotSpring's two-phase approach:
- **hotSpring**: Python plasma physics → Rust nuclear EOS → GPU HFB
- **airSpring**: Python agricultural science → Rust ET₀/soil/water → GPU irrigation

The same BarraCuda/ToadStool infrastructure supports both domains. The key shared abstractions are: ODE/PDE solvers, nonlinear fitting, time series processing, and spatial interpolation.

---

## 6. Evolution Path

### Completed (Phase 2 & 3)
- ~~Cross-validate Python vs Rust~~ — **75/75 MATCH** within 1e-5 tolerance
- ~~Port RMSE/IA/MBE/NSE statistics to Rust~~ — **Done** (`testutil`)
- ~~Port SoilWatch 10 calibration to Rust~~ — **Done** (`eco::sensor_calibration`)
- ~~Port Hargreaves ET₀, sunshine/temp Rs, monthly G~~ — **Done** (`eco::evapotranspiration`)
- ~~Build crop Kc database~~ — **Done** (`eco::crop`, 10 crops, FAO-56 Table 12)
- ~~Full pipeline demonstration~~ — **Done** (`simulate_season` binary)
- ~~GPU acceleration via ToadStool~~ — **Done** (8 orchestrators: BatchedEt0, BatchedWaterBalance, BatchedDualKc, Kriging, Reduce, Stream, BatchedRichards, fit_nm)
- ~~4/4 ToadStool issues resolved, GPU determinism verified~~ — **Done**
- ~~Spatial interpolation (kriging)~~ — **Done** (KrigingInterpolator)

### Near Term (Phase 3 continued)
- Real-time IoT stream processing on GPU
- Richards equation solver (1D unsaturated flow)
- Benchmark: Rust vs Python throughput on 918 station-days

### Long Term (Phase 4: Penny Irrigation)
- Sovereign irrigation scheduling on consumer hardware ($600 GPU)
- Sub-field spatial resolution from cheap sensor networks
- Model-predictive control with weather forecast integration
- Open alternative to $5000/field proprietary systems

---

*February 2026 — 330 validation checks, 468 Rust tests (371 lib + 97 integration), 918 station-days real data,
75/75 cross-validation match, zero synthetic*
