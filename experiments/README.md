# airSpring Experiments

**Updated**: February 25, 2026
**Status**: 11 experiments, 344/344 Python + 371 Rust lib tests + 97 integration + GPU wired (8 orchestrators)

---

## Experiment Index

| Exp | Name | Track | Status | Baseline Tool | Rust Modules Validated | Checks |
|:---:|------|-------|:------:|---------------|------------------------|:------:|
| 001 | FAO-56 Penman-Monteith ET₀ | Irrigation | **Complete** | Python (FAO-56 Chapter 2/4) | `eco::evapotranspiration` | 64+31 |
| 002 | Soil sensor calibration (Dong 2020) | Soil | **Complete** | Python (Dong 2020) | `eco::soil_moisture`, `eco::correction` | 36+26 |
| 003 | IoT irrigation pipeline (Dong 2024) | IoT | **Complete** | Python + R ANOVA | `io::csv_ts`, `eco::sensor_calibration` | 24+11 |
| 004 | Water balance scheduling (FAO-56 Ch 8) | Irrigation | **Complete** | Python (FAO-56 Ch 8) | `eco::water_balance` | 18+13 |
| 005 | Real data pipeline (918 station-days) | Integration | **Complete** | Python + Open-Meteo API | All modules | R²=0.967+21 |

| 006 | HYDRUS Richards Equation (van Genuchten-Mualem) | Environmental | **Complete** | Python + Rust CPU | `eco::richards` | 14+15 |
| 007 | Biochar Adsorption Isotherms (Kumari et al. 2025) | Environmental | **Complete** | Python + Rust CPU | `eco::isotherm` | 14+14 |
| 009 | FAO-56 Dual Kc (Allen 1998 Ch 7) | Irrigation | **Complete** | Python + Rust CPU | `eco::dual_kc` | 63+61 |
| 010 | Regional ET₀ Intercomparison (6 MI stations) | Precision Ag | **Complete** | Python + Rust CPU | `eco::evapotranspiration` | 61+61 |
| 011 | Cover Crop Dual Kc + No-Till (FAO-56 Ch 11) | Irrigation | **Complete** | Python + Rust CPU | `eco::dual_kc` (mulch) | 40+40 |
| 015 | 60-Year Water Balance (Wooster OH, ERA5) | Integration | **Complete** | Python + Rust CPU | `eco::water_balance`, `eco::evapotranspiration` | 10+11 |

**Total**: 344 Python checks + 371 Rust lib + 97 integration tests + 75 cross-validation values + 8 GPU orchestrators

---

## Experiment Protocol

Each experiment follows the same multi-phase protocol:

### Phase 0: Python Control
1. Digitize paper benchmarks into `control/*/benchmark_*.json`
2. Implement in Python using the paper's equations
3. Validate against benchmarks with quantitative checks
4. Record all checks in `CONTROL_EXPERIMENT_STATUS.md`

### Phase 0+: Real Open Data
1. Download real weather/soil data (Open-Meteo, NOAA, USDA)
2. Run the validated Python pipeline on real data
3. Compare against independent computations (e.g., Open-Meteo ET₀)

### Phase 1: Rust BarraCuda
1. Implement the same algorithms in Rust using BarraCuda primitives
2. Write validation binary that loads `benchmark_*.json` and checks results
3. Run `cargo test` (unit + integration)

### Phase 2: Cross-Validation
1. Both Python and Rust emit 75 intermediate values to JSON
2. `scripts/cross_validate.py` diffs them (tolerance: 1e-5)
3. All 75 values must match (includes Richards VG, isotherm predictions)

### Phase 3: GPU Evolution
1. Wire CPU modules to GPU orchestrators via ToadStool primitives
2. Verify GPU results match CPU baselines
3. Measure speedup and throughput

---

## Experiment Details

### Exp 001: FAO-56 Penman-Monteith ET₀

**Paper**: Allen et al. (1998) *Crop evapotranspiration: Guidelines for computing crop water requirements.* FAO Irrigation and Drainage Paper No. 56.

**Control**: `control/fao56/penman_monteith.py` — 64/64 checks against digitized Table 2.3-2.8, Example 17-20 benchmarks.

**Rust**: `barracuda/src/eco/evapotranspiration.rs` — 23 FAO-56 functions + Hargreaves ET₀. `validate_et0` binary: 31/31 checks.

**GPU**: `gpu::et0::BatchedEt0` via `BatchedElementwiseF64::fao56_et0_batch()` — GPU-FIRST dispatch. 12.5M ops/sec at N=10,000.

**Key Result**: Bangkok 5.72, Uccle 3.88, Lyon 4.56 mm/day match paper exactly.

### Exp 002: Soil Sensor Calibration (Dong 2020)

**Paper**: Dong et al. (2020) *Soil moisture sensor performance and corrections for Michigan agricultural soils.* Agriculture 10(12), 598.

**Control**: `control/soil_sensors/calibration_dong2020.py` — 36/36 checks. Topp equation, RMSE/IA/MBE, four correction models (linear, quadratic, exponential, logarithmic).

**Rust**: `barracuda/src/eco/soil_moisture.rs`, `eco/correction.rs` — 7 soil textures, 4 correction fits + ridge regression via `barracuda::linalg::ridge`. `validate_soil` binary: 26/26 checks.

### Exp 003: IoT Irrigation Pipeline (Dong 2024)

**Paper**: Dong et al. (2024) *In-field IoT-based soil moisture monitoring and irrigation scheduling.* Frontiers in Water 6, 1353597.

**Control**: `control/iot_irrigation/calibration_dong2024.py` — 24/24 checks. SoilWatch 10 calibration, irrigation recommendation model.

**Rust**: `barracuda/src/io/csv_ts.rs`, `eco/sensor_calibration.rs` — streaming columnar parser + SoilWatch 10 VWC. `validate_iot`: 11/11 checks.

**GPU**: `gpu::stream::StreamSmoother` via `MovingWindowStats` (wetSpring S28+) — IoT stream smoothing with 24-hour sliding window. 32.4M elem/sec.

### Exp 004: Water Balance Scheduling (FAO-56 Ch 8)

**Paper**: Allen et al. (1998) *FAO-56 Chapter 8 — Daily soil water balance.*

**Control**: `control/water_balance/fao56_water_balance.py` — 18/18 checks. Mass balance (0.0000 mm error), Ks stress, TAW/RAW, deep percolation.

**Rust**: `barracuda/src/eco/water_balance.rs` — `WaterBalanceState`, `RunoffModel`, `simulate_season()`. `validate_water_balance`: 13/13 checks.

**GPU**: `gpu::water_balance::BatchedWaterBalance` via `water_balance_batch()` — GPU step dispatch.

### Exp 005: Real Data Pipeline (918 Station-Days)

**Data**: 6 Michigan agricultural weather stations, 2023 growing season, downloaded from Open-Meteo ERA5 archive (free, no API key, 80+ year history).

**Control**: `control/fao56/compute_et0_real_data.py` — ET₀ computed for each station-day.

**Validation**: R²=0.967 against Open-Meteo's independent ET₀ computation. RMSE 0.295 mm/day (East Lansing).

**Rust**: `validate_real_data` binary — 23/23 checks. 4 crops × rainfed + irrigated scenarios. Capability-based station discovery (filesystem/env var). Mass balance verified for all scenarios.

---

### Exp 009: FAO-56 Dual Crop Coefficient (Allen 1998 Ch 7)

**Paper**: Allen et al. (1998) *FAO-56 Chapter 7 — ETc: Dual crop coefficient.*

**Control**: `control/dual_kc/dual_crop_coefficient.py` — 63/63 checks. Basal Kc
(Table 17), soil evaporation (Eqs 69-74), evaporation layer water balance, REW/TEW
(Table 19), multi-day simulations (bare soil drydown, corn mid-season).

**Benchmark**: `control/dual_kc/benchmark_dual_kc.json` — 10 crops Kcb values, 11
soil types REW/TEW, equation test vectors, integration scenarios.

**Key Result**: Dual Kc separates transpiration (Kcb) from soil evaporation (Ke).
Under full canopy cover (corn mid-season), ETc/ET₀ ≈ Kcb because Ke → 0. Under
bare soil, Ke dominates and declines as surface dries (stage 1 → stage 2).

---

### Exp 011: Cover Crop Dual Kc + No-Till Mulch Effects

**Papers**: Allen et al. (1998) *FAO-56 Ch 7 + Ch 11*; Islam & Reeder (2014) *ISWCR*.

**Control**: `control/dual_kc/cover_crop_dual_kc.py` — 40/40 checks. Cover crop Kcb
values (5 crops), no-till mulch reduction (5 residue levels), Islam et al. soil
observations (SOC, bulk density, infiltration, AWC), rye→corn transition simulation,
no-till ET savings (39.6% during initial stage).

**Benchmark**: `control/dual_kc/benchmark_cover_crop_kc.json` — Cover crop Kcb
(cereal rye, crimson clover, winter wheat, hairy vetch, tillage radish), no-till
mulch factors (0.25–1.0), Islam et al. Brandt farm observations, rye→corn phases.

**Key Result**: No-till with heavy residue (mulch_factor=0.40) reduces bare soil
evaporation by ~40% during the initial growth stage. The mulch effect diminishes
during mid-season when transpiration (Kcb) dominates. Connects to baseCamp
Sub-thesis 06 (no-till → Anderson geometry coupling).

---

### Exp 010: Regional ET₀ Intercomparison — 6 Michigan Stations

**Paper**: Regional ET₀ intercomparison across Michigan microclimates.

**Control**: `control/regional_et0/regional_et0_intercomparison.py` — 61/61 checks.
Per-station FAO-56 PM ET₀ vs Open-Meteo ERA5, seasonal totals, spatial variability,
geographic consistency, 15-station-pair temporal correlations.

**Data**: Open-Meteo Historical Weather API (ERA5 reanalysis), 2023 growing season
(May 1 – Sep 30), 6 stations × 153 days = 918 station-days.

**Key Results**: R² > 0.96 all stations, RMSE < 0.33 mm/day. Grand mean ET₀ =
4.27 mm/day. Season totals 633–677 mm (matches MSU Enviro-weather references).
Cross-station correlation r = 0.80–0.96 (geographic proximity drives correlation).
Spatial CV = 2.0% (tight clustering expected for Lower Michigan stations).

---

### Exp 006: HYDRUS Richards Equation (Dong 2019 / van Genuchten 1980)

**Paper**: Richards (1931), van Genuchten (1980), Dong et al. (2019) J Sustainable Water 5(4):04019005

**Control**: `control/richards/richards_1d.py` — 14/14 checks. Van Genuchten retention, Mualem conductivity, sand infiltration, silt loam drainage, steady-state flux.

**Rust**: `barracuda/src/eco/richards.rs` — Implicit Euler + Picard iteration, Thomas algorithm. `validate_richards` binary: 15/15 checks.

**GPU**: `gpu::richards::BatchedRichards` wired to `barracuda::pde::richards::solve_richards` (Tier B). Cross-validates eco::richards (implicit Euler) against upstream (Crank-Nicolson) for physical reasonableness.

### Exp 007: Biochar Adsorption Isotherms (Kumari et al. 2025)

**Paper**: Kumari, Dong & Safferman (2025) Applied Water Science 15(7):162

**Control**: `control/biochar/biochar_isotherms.py` — 14/14 checks. Langmuir and Freundlich isotherm fitting for P adsorption on wood and sugar beet biochar.

**Rust**: `barracuda/src/eco/isotherm.rs` — Linearized least squares fitting. `validate_biochar` binary: 14/14 checks.

**GPU**: `gpu::isotherm::fit_langmuir_nm` / `fit_freundlich_nm` wired to `barracuda::optimize::nelder_mead` (Tier B). Linearized LS as initial guess → NM refinement matches scipy.curve_fit.

### Exp 015: 60-Year Water Balance Reconstruction

**Data**: Open-Meteo ERA5 archive, 1960-2023, Wooster OH (OSU OARDC)

**Control**: `control/long_term_wb/long_term_water_balance.py` — 10/10 checks. 64 growing seasons, decade trends, climate signal detection.

**Rust**: Existing `eco::water_balance` + `eco::evapotranspiration::hargreaves_et0`. `validate_long_term_wb` binary: 11/11 checks.

**GPU**: `BatchedEt0` + `BatchedWaterBalance` at 64-year scale (already wired)

---

## Naming Convention

Experiments follow `NNN_name` format:
- `001`–`005`: Baseline reproduction (FAO-56, soil, IoT, water balance, real data)
- `006`–`007`: Richards equation, biochar isotherms (Track 2)
- `009`–`011`: Dual Kc, regional ET₀, cover crops + no-till
- `015`: Long-term water balance reconstruction (see `specs/PAPER_REVIEW_QUEUE.md`)

## Results

Benchmark data is stored in `control/*/benchmark_*.json` (digitized paper values)
and used by both Python control scripts and Rust validation binaries as ground
truth. Cross-validation outputs are produced by `cross_validate` (Rust) and
`scripts/cross_validate.py` (Python).
