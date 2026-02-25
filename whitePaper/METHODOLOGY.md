# Validation Methodology

**Purpose**: Document the multi-phase validation approach used in the airSpring study
**See also**: [STUDY.md](STUDY.md) for complete results

---

## 1. Three-Phase Approach

Every published workload passes through three validation phases:

### Phase 0: Python/R Control (Paper Replication)

Reproduce published results using the same open-source tools the original authors used. This establishes:
- **Correctness**: Our implementations match the published equations
- **Performance baseline**: Traditional scientific Python stack (numpy, scipy, pandas)
- **Benchmark data**: Digitized values from paper tables/figures stored as JSON

Phase 0 uses manual implementations following the papers exactly. The FAO-56 Penman-Monteith is implemented equation-by-equation from Allen et al. (1998). Soil sensor calibration follows Dong et al. (2020) Tables 3-4. The R ANOVA matches Dong et al. (2024)'s use of R v4.3.1.

### Phase 0+: Real Open Data Pipeline

Apply the validated methods to real weather observations from public APIs. This establishes:
- **Open data viability**: Public APIs provide sufficient data for precision agriculture
- **Cross-validation**: Our ET₀ compared against independent computations (Open-Meteo)
- **Scalability**: Real data enables sweeps across years, stations, and seasons

Phase 0+ uses zero synthetic data. Three open data sources:
1. **Open-Meteo** (free, no key): 80+ years of historical weather at 10km resolution
2. **OpenWeatherMap** (free key): Real-time current + 5-day forecasts
3. **NOAA CDO** (free token): GHCND daily observations from US weather stations

### Phase 1: BarraCuda Execution (Rust Evolution)

Re-implement the same computations in pure Rust using BarraCuda. Compare:
- **Accuracy**: Same ET₀, soil moisture, water balance outputs as Python
- **Throughput**: Evaluations per second
- **Dependencies**: BarraCuda target is minimal external dependencies (barracuda, serde, serde_json)
- **Reproducibility**: Deterministic results
- **GPU readiness**: Architecture suitable for ToadStool GPU acceleration
- **Code quality**: Zero clippy pedantic/nursery warnings, proper error types, idiomatic Rust
- **Binaries**: 8 validation binaries (validate_et0, validate_soil, validate_iot, validate_water_balance, validate_sensor_calibration, validate_real_data, cross_validate, simulate_season)

### Phase 2: Cross-Validation (Python↔Rust)

Verify that Python and Rust produce identical results for identical inputs:
- Structured harness computes values from fixed inputs in both languages
- JSON output enables automated comparison
- Tolerance: 1e-5 for all floating-point values
- Covers: atmospheric, solar, radiation, ET₀, Topp, SoilWatch 10, irrigation, statistics, Hargreaves, sunshine Rs, monthly G

---

## 2. Workloads

### 2.1 FAO-56 Penman-Monteith ET₀ (Experiment 001)

**Source**: Allen, R.G., Pereira, L.S., Raes, D., Smith, M. (1998). FAO Irrigation and Drainage Paper 56.
**Reference data**: FAO-56 Chapter 4 Examples 17, 18, 20; Tables 2.3, 2.4

**Phase 0**: Implement all FAO-56 component equations (Eqs. 6-50) in Python with numpy. Validate against 3 worked examples covering monthly (Bangkok), daily (Uccle), and missing-data (Lyon) scenarios. Check all intermediate values (Delta, gamma, es, ea, Ra, Rs, Rn, etc.).

**Phase 0+**: Compute ET₀ for 6 Michigan agricultural stations using 2023 growing season data (153 days each, 918 station-days total). Cross-check against Open-Meteo's independent ET₀ computation.

**Phase 1**: Implement in Rust (`eco::evapotranspiration`). Validate against same FAO-56 tables.

**Acceptance criteria**:
- Phase 0: ET₀ within 0.15 mm/day of FAO-56 published examples
- Phase 0+: R² > 0.95 vs Open-Meteo ET₀ across all stations
- Phase 1: Identical outputs to Python for same inputs

### 2.2 Soil Sensor Calibration (Experiment 002)

**Source**: Dong et al. (2020) "Performance evaluation of soil moisture sensors in coarse- and fine-textured Michigan agricultural soils" Agriculture 10(12), 598
**Reference data**: Paper Tables 3, 4; Topp et al. (1980) equation

**Phase 0**: Implement Topp equation, statistical metrics (RMSE, IA, MBE, R²), and correction equation fitting (linear, quadratic, exponential, logarithmic). Validate against 8 published epsilon-theta points, 7 analytical statistical tests, factory calibration criteria, and field improvement RMSE values.

**Phase 1**: Implement Topp equation and soil properties in Rust (`eco::soil_moisture`).

**Acceptance criteria**:
- Phase 0: Topp equation within 0.005 m³/m³ of computed values
- Phase 0: RMSE/IA/MBE formulas analytically exact for known test vectors
- Phase 1: Topp outputs match Python to machine precision

### 2.3 IoT Irrigation Pipeline (Experiment 003)

**Source**: Dong et al. (2024) "Implementation of an In-Field IoT System for Precision Irrigation Management" Frontiers in Water 6, 1353597
**Reference data**: Paper Eq. 5 (SoilWatch 10 calibration), Table 2, field demonstration results
**Statistical tool**: R v4.3.1 (matching paper)

**Phase 0**: Implement SoilWatch 10 calibration equation and irrigation recommendation model. Validate against published sensor performance metrics and field demonstration yield/weight data. Write R ANOVA script matching the paper's statistical method.

**Phase 1**: Implement CSV time series parser in Rust (`io::csv_ts`).

**Acceptance criteria**:
- Phase 0: SoilWatch 10 equation reproduces published RC→VWC mapping
- Phase 0: Irrigation model produces correct recommendations for known inputs
- Phase 1: CSV parsing + column statistics match Python output

### 2.4 Water Balance (Experiment 004)

**Source**: FAO-56 Chapter 8 (Allen et al. 1998)
**Reference data**: FAO-56 Eqs. 84-85 (stress coefficient, daily depletion)

**Phase 0**: Implement daily soil water balance tracking (TAW, RAW, Ks, Dr). Validate mass conservation (inflow = outflow + storage change). Run dry-down, irrigated, and heavy rain scenarios.

**Phase 0+**: Simulate 4 crop scenarios (blueberry, tomato, corn, reference grass) at actual Dong (2024) demonstration sites using real 2023 Michigan weather data.

**Phase 1**: Implement water balance in Rust (`eco::water_balance`).

**Acceptance criteria**:
- Phase 0: Mass balance closure |error| < 0.001 mm for all scenarios
- Phase 0: Ks correctly bounded [0,1] with proper stress onset at Dr > RAW
- Phase 0+: Water savings consistent with Dong (2024) published results (30%+ for tomato)
- Phase 1: Mass balance closure matches Python

---

## 3. Comparison Protocol

1. Digitize benchmark data from published papers into JSON files (`benchmark_*.json`)
2. Implement equations in Python following the paper's notation exactly
3. Validate against digitized benchmarks with explicit tolerances
4. Download real open data from public APIs (no synthetic unless API unavailable)
5. Run validated methods on real data, cross-check against independent sources
6. Implement in Rust, validate against same benchmark JSON files
7. Cross-validate: Python and Rust produce identical outputs for identical inputs
8. Report per-experiment check counts, error metrics, and data provenance
9. All results reproducible from `bash scripts/run_all_baselines.sh`

---

## 4. Hardware

All experiments run on a single consumer workstation:

| Component | Specification |
|-----------|--------------|
| CPU | Intel i9-12900K (8P+8E, 24 threads) |
| RAM | 64 GB DDR5-4800 |
| OS | Pop!_OS 22.04 (Linux 6.17) |
| Python | 3.x (numpy, scipy, pandas, requests) |
| R | 4.x (planned — for ANOVA matching paper) |
| Rust | stable (rustc) |
| GPU | RTX 4070 (SHADER_F64 confirmed) — Phase 3 LIVE (8 orchestrators) |

---

## 5. Open Data Sources

| Source | Type | Key Required | Coverage | Used For |
|--------|------|:------------:|----------|----------|
| Open-Meteo | Historical reanalysis | No | Global, 80+ years, 10km | ET₀ computation, water balance |
| OpenWeatherMap | Current + forecast | Yes (free) | Global, real-time + 5 days | Current conditions monitoring |
| NOAA CDO | GHCND observations | Yes (free) | US, 100+ years, station-level | Historical station validation |
| USDA Web Soil Survey | Soil properties | No | US counties | Soil hydraulic parameters |
| FAO Paper 56 | Reference tables | No | Universal | Equation validation |

---

## 6. Acceptance Criteria Summary

| Experiment | Phase 0 Target | Phase 0+ Target | Phase 1 Target | Phase 2 Target |
|------------|---------------|-----------------|----------------|----------------|
| FAO-56 ET₀ | ET₀ ±0.15 mm/d of FAO examples | R² > 0.95 vs Open-Meteo | Match Python outputs | ≤1e-5 tolerance |
| Soil Sensors | Topp ±0.005 m³/m³; stats exact | — | Match Python outputs | ≤1e-5 tolerance |
| IoT Pipeline | SoilWatch 10 + irrigation correct | — | CSV stats + calibration match | ≤1e-5 tolerance |
| Water Balance | Mass balance < 0.001 mm | Savings per Dong (2024) | Mass balance match Python | ≤1e-5 tolerance |

### Grand Total: 330 Validation Checks Pass + 468 Rust Tests + 918 Real Data Points

| Phase | Checks | Description |
|-------|:------:|-------------|
| Phase 0 (Python control) | 142 | 64 ET₀ + 36 soil + 24 IoT + 18 water balance |
| Phase 1 (Rust BarraCuda) | 327 | 31 ET₀ + 26 soil + 11 IoT + 13 water balance + 21 sensor + 23 real data + dual Kc + cover crop + regional ET₀ + Richards + biochar + long-term WB |
| Phase 1 (Rust tests) | 468 | 371 lib + 97 integration + doc-tests |
| Phase 2 (Cross-validation) | 75 | Python↔Rust identical outputs (tol=1e-5), loads from benchmark JSON |
| **Total** | **330** | **All pass** |
| Phase 0+ (Real data) | 918 station-days | R²=0.967, 4 crop water balance, zero synthetic |

---

## 7. Software Versions

| Component | Version | Notes |
|-----------|---------|-------|
| Python | 3.10+ | numpy, scipy, pandas, requests |
| numpy | 1.24+ | Core numerical computing |
| scipy | 1.10+ | Curve fitting (calibration) |
| pandas | 2.0+ | Data handling |
| pyet | 1.4+ | FAO-56 PM cross-reference |
| R | 4.3.1 (paper match) | One-way ANOVA (planned) |
| Rust | stable (1.77+) | BarraCuda, zero unsafe |
| serde | 1.x | Rust serialization |
| serde_json | 1.x | Benchmark JSON + cross-validation |
| OS | Pop!_OS 22.04 | Linux 6.17 |
