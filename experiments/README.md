# airSpring Experiments

**Updated**: March 1, 2026
**Status**: 56 experiments, 1237/1237 Python + 636 lib + 57 forge tests + 60 barracuda + 5 forge binaries + 78/78 pure GPU pipeline + 104/104 metalForge mixed-hardware + 75/75 cross-validation + 15 Tier A GPU orchestrators (ops 5-8 GPU-first via ToadStool S70+) + seasonal pipeline (GPU Stages 1-2) + NPU→GPU PCIe bypass + NUCLEUS atomics + biomeOS graph execution + **Rust 21.0× faster than Python** (18/18 parity)

---

## Experiment Index

| Exp | Name | Track | Status | Baseline Tool | Rust Modules Validated | Checks |
|:---:|------|-------|:------:|---------------|------------------------|:------:|
| 001 | FAO-56 Penman-Monteith ET₀ | Irrigation | **Complete** | Python (FAO-56 Chapter 2/4) | `eco::evapotranspiration` | 64+31 |
| 002 | Soil sensor calibration (Dong 2020) | Soil | **Complete** | Python (Dong 2020) | `eco::soil_moisture`, `eco::correction` | 36+26 |
| 003 | IoT irrigation pipeline (Dong 2024) | IoT | **Complete** | Python + R ANOVA | `io::csv_ts`, `eco::sensor_calibration` | 24+11 |
| 004 | Water balance scheduling (FAO-56 Ch 8) | Irrigation | **Complete** | Python (FAO-56 Ch 8) | `eco::water_balance` | 18+13 |
| 005 | Real data pipeline (918 station-days) | Integration | **Complete** | Python + Open-Meteo API | All modules | R²=0.967+21 |
| 006 | HYDRUS Richards Equation (VG-Mualem) | Environmental | **Complete** | Python + Rust CPU | `eco::richards` | 14+15 |
| 007 | Biochar Adsorption Isotherms (Kumari 2025) | Environmental | **Complete** | Python + Rust CPU | `eco::isotherm` | 14+14 |
| 009 | FAO-56 Dual Kc (Allen 1998 Ch 7) | Irrigation | **Complete** | Python + Rust CPU | `eco::dual_kc` | 63+61 |
| 010 | Regional ET₀ Intercomparison (6 MI stations) | Precision Ag | **Complete** | Python + Rust CPU | `eco::evapotranspiration` | 61+61 |
| 011 | Cover Crop Dual Kc + No-Till (FAO-56 Ch 11) | Irrigation | **Complete** | Python + Rust CPU | `eco::dual_kc` (mulch) | 40+40 |
| 015 | 60-Year Water Balance (Wooster OH, ERA5) | Integration | **Complete** | Python + Rust CPU | `eco::water_balance`, Hargreaves | 10+11 |
| 008 | Yield Response to Water Stress (FAO-56 Ch 10) | Irrigation | **Complete** | Python + Rust CPU | `eco::yield_response` | 32+32 |
| 012 | CW2D Richards Extension (Dong 2019) | Environmental | **Complete** | Python + Rust CPU | `eco::richards` (CW2D media) | 24+24 |
| 014 | Irrigation Scheduling Optimization | Precision Ag | **Complete** | Python + Rust CPU | `eco::water_balance`, `eco::yield_response` | 25+28 |
| 016 | Lysimeter ET Direct Measurement | IoT | **Complete** | Python + Rust CPU | mass→ET, temp compensation | 26+25 |
| 017 | ET₀ Sensitivity Analysis (OAT) | Precision Ag | **Complete** | Python + Rust CPU | `eco::evapotranspiration` | 23+23 |
| 018 | Michigan Crop Water Atlas (100 stations) | Integration | **Active** | Python + Rust CPU | All `eco::` + `yield_response` | 1354/1354 |
| 019 | Priestley-Taylor ET₀ (1972) | Precision Ag | **Complete** | Python + Rust CPU | `eco::evapotranspiration` (PT) | 32+32 |
| 020 | ET₀ 3-Method Intercomparison (PM/PT/HG) | Integration | **Complete** | Python + Rust CPU | `eco::evapotranspiration` (all 3) | 36+36 |
| 021 | Thornthwaite Monthly ET₀ (1948) | Precision Ag | **Complete** | Python + Rust CPU | `eco::evapotranspiration` (Thornthwaite) | 23+50 |
| 022 | Growing Degree Days (GDD) | Precision Ag | **Complete** | Python + Rust CPU | `eco::crop` (gdd_avg, kc_from_gdd) | 33+26 |
| 023 | Pedotransfer Functions (Saxton-Rawls 2006) | Soil | **Complete** | Python + Rust CPU | `eco::soil_moisture` (saxton_rawls) | 70+58 |
| 024 | NASS Yield Validation (Stewart 1977 pipeline) | Irrigation | **Complete** | Python + Rust CPU | `eco::yield_response` + `eco::water_balance` | 41+40 |
| 025 | Forecast Scheduling Hindcast | Precision Ag | **Complete** | Python + Rust CPU | `eco::water_balance` + `eco::yield_response` | 19+19 |
| 026 | USDA SCAN Soil Moisture (Richards 1D) | Soil | **Complete** | Python + Rust CPU | `eco::richards` + `eco::van_genuchten` | 34+34 |
| 027 | Multi-Crop Water Budget (5 Michigan crops) | Integration | **Complete** | Python + Rust CPU | `eco::water_balance` + `eco::dual_kc` + `eco::yield_response` | 47+47 |
| 028 | NPU Edge Inference (AKD1000 live) | IoT/NPU | **Complete** | Rust + metalForge | `npu` (akida-driver) + forge dispatch | 35+21 |
| 029 | Funky NPU for Agricultural IoT | IoT/NPU | **Complete** | Rust + AKD1000 | streaming, evolution, multi-crop, LOCOMOS | 32/32 |
| 029b | High-Cadence NPU Streaming Pipeline | IoT/NPU | **Complete** | Rust + AKD1000 | 1-min cadence, burst, fusion, ensemble, hot-swap | 28/28 |
| 030 | AmeriFlux Eddy Covariance ET (Baldocchi 2003) | Precision Ag | **Complete** | Python + Rust CPU | `eco::evapotranspiration` | 27+27 |
| 031 | Hargreaves-Samani Temperature ET₀ (1985) | Precision Ag | **Complete** | Python + Rust CPU | `eco::evapotranspiration` (Hargreaves) | 24+24 |
| 032 | Ecological Diversity Indices | Integration | **Complete** | Python + Rust CPU | `eco::diversity` | 22+22 |
| 033 | Makkink (1957) Radiation-Based ET₀ | Precision Ag | **Complete** | Python + Rust CPU | `eco::evapotranspiration` (Makkink) | 21+16 |
| 034 | Turc (1961) Temperature-Radiation ET₀ | Precision Ag | **Complete** | Python + Rust CPU | `eco::evapotranspiration` (Turc) | 22+17 |
| 035 | Hamon (1961) Temperature-Based PET | Precision Ag | **Complete** | Python + Rust CPU | `eco::evapotranspiration` (Hamon) | 20+19 |
| 036 | biomeOS Neural API Round-Trip Parity | Integration | **Complete** | Python + Rust CPU | JSON serialization, metalForge Neural dispatch | 14+29 |
| 037 | ET₀ Ensemble Consensus (6-Method) | Precision Ag | **Complete** | Python + Rust CPU | `eco::evapotranspiration` (ensemble) | 9+17 |
| 038 | Pedotransfer → Richards Coupled Simulation | Soil Physics | **Complete** | Python + Rust CPU | `soil_moisture` + `richards` + `van_genuchten` | 29+32 |
| 039 | Cross-Method ET₀ Bias Correction | Precision Ag | **Complete** | Python + Rust CPU | `eco::evapotranspiration` (bias factors) | 24+24 |
| 040 | CPU vs GPU Parity Validation | GPU Portability | **Complete** | Python + Rust (BatchedEt0) | `gpu::et0`, `gpu::water_balance` | 22+26 |
| 041 | metalForge Mixed-Hardware Dispatch | Mixed Hardware | **Complete** | Python + Rust (forge) | `dispatch::route`, `workloads`, `substrate` | 14+18 |
| 042 | Seasonal Batch ET₀ at GPU Scale | GPU Batch | **Complete** | Python + Rust (BatchedEt0) | `gpu::et0` (365×4 station-days) | 18+21 |
| 043 | Titan V GPU Live Dispatch | GPU Live | **Complete** | Rust (Titan V GV100) | `gpu::et0` (live WGSL shader, 10K batch) | 24 |
| 044 | metalForge Live Hardware Probe | Mixed HW | **Complete** | Rust (probe + dispatch) | RTX 4070 + Titan V + AKD1000 + i9-12900K | 17 |
| 045 | Anderson Soil-Moisture Coupling | Cross-Spring | **Complete** | Python + Rust CPU | `eco::anderson` (θ→S_e→d_eff→QS regime) | 55+95 |
| 046 | Atlas Stream Real Data Validation | Integration | **Complete** | Rust (80yr Open-Meteo) | `gpu::atlas_stream` + `gpu::seasonal_pipeline` | 73/73 |
| 047 | GPU Math Portability Validation | GPU Portability | **Complete** | Python + Rust (all 13 GPU modules) | All `gpu::*` orchestrators (13 modules, 46 checks) | 21+46 |
| 048 | NCBI 16S + Soil Moisture Anderson Coupling | Cross-Spring | **Complete** | Python + Rust CPU | `eco::anderson` + `eco::et` + `eco::water_balance` + NCBI | 14+29 |
| 049 | Blaney-Criddle (1950) Temperature PET | Precision Ag | **Complete** | Python + Rust CPU | `eco::evapotranspiration` (Blaney-Criddle) | 18+18 |
| 050 | SCS Curve Number Runoff (USDA 1972) | Hydrology | **Complete** | Python + Rust CPU | `eco::runoff` (SCS-CN, AMC) | 38+38 |
| 051 | Green-Ampt (1911) Infiltration | Soil Physics | **Complete** | Python + Rust CPU | `eco::infiltration` (Newton-Raphson) | 37+37 |
| 052 | SCS-CN + Green-Ampt Coupled Runoff-Infiltration | Hydrology | **Complete** | Python + Rust CPU | `eco::runoff` + `eco::infiltration` | 292+292 |
| 053 | Van Genuchten Inverse Parameter Estimation | Soil Physics | **Complete** | Python + Rust CPU | `eco::van_genuchten` + `barracuda::optimize::brent` | 84+84 |
| 054 | Full-Season Irrigation Water Budget Audit | Integration | **Complete** | Python + Rust CPU | `eco::evapotranspiration` + `eco::water_balance` + `eco::yield_response` | 34+34 |
| 055 | Barracuda Pure GPU Workload Validation | GPU | **Complete** | Rust GPU (Titan V) | `gpu::et0` + `gpu::seasonal_pipeline` + `gpu::hargreaves` + `gpu::kc_climate` | 78 |
| 056 | Mixed-Hardware Pipeline + NUCLEUS Atomics | metalForge | **Complete** | Rust (synthetic) | `metalForge::pipeline` + `metalForge::nucleus` + `metalForge::graph` | 43 |

**Grand Total**: 1237 Python + **618 lib + 31 forge tests** + 1498/1498 atlas + 33/33 cross-validation + 11 Tier A + 4 Tier B GPU orchestrators + seasonal pipeline + Titan V GPU live (24/24) + AKD1000 NPU live (95/95) + metalForge (5 substrates, 18 workloads, 29/29 cross-system) + GPU math portability (46/46) + NCBI 16S coupling (14+29) + coupled runoff-infiltration (292/292) + VG inverse (84/84) + full-season WB audit (34/34) + 59 binaries + 30/30 cross-spring benchmarks (6 Springs) + ToadStool S68 synced

---

## Test Breakdown (v0.5.4)

| Category | Tests | Source |
|----------|:-----:|--------|
| Barracuda lib (unit + doc) | 618 | `cargo test --lib` (incl. Tier B orchestrators, seasonal pipeline, atlas stream, anderson, diversity, mc\_et0, NPU, Makkink/Turc/Hamon) |
| Barracuda validation binaries | 59 | `validate_*`, `bench_*`, `cross_validate`, `simulate_season` |
| Forge | 31 | `metalForge/forge/` (substrate, dispatch, probe, workloads, cross-system routing) |
| Forge binaries | 4 | `validate_dispatch`, `validate_live_hardware`, `validate_dispatch_routing` |
| **Total project tests** | **618 lib + 31 forge** | |
| Atlas stream (real data) | 73 | `validate_atlas_stream` (12 stations, 4800 crop-year results) |
| Atlas checks | 1393 | `validate_atlas` (100 stations × 13 checks each) |
| GPU live checks | 24 | `validate_gpu_live` (Titan V WGSL dispatch) |
| Cross-system routing | 29 | `validate_dispatch` (18 workloads × dispatch checks) |
| Hardware probe checks | 17 | `validate_live_hardware` (5 substrates) |
| CPU vs Python parity | 8/8 | `bench_cpu_vs_python` (25.9× geometric mean speedup) |

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

**Rust**: `barracuda/src/eco/soil_moisture.rs`, `eco/correction.rs` — 7 soil textures, 4 correction fits + ridge regression via `barracuda::linalg::ridge`. `validate_soil` binary: 40/40 checks.

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
evaporation by ~40% during the initial growth stage.

---

### Exp 010: Regional ET₀ Intercomparison — 6 Michigan Stations

**Paper**: Regional ET₀ intercomparison across Michigan microclimates.

**Control**: `control/regional_et0/regional_et0_intercomparison.py` — 61/61 checks.
Per-station FAO-56 PM ET₀ vs Open-Meteo ERA5, seasonal totals, spatial variability,
geographic consistency, 15-station-pair temporal correlations.

**Data**: Open-Meteo Historical Weather API (ERA5 reanalysis), 2023 growing season
(May 1 – Sep 30), 6 stations × 153 days = 918 station-days.

**Key Results**: R² > 0.96 all stations, RMSE < 0.33 mm/day. Grand mean ET₀ =
4.27 mm/day. Season totals 633–677 mm. Cross-station correlation r = 0.80–0.96.

---

### Exp 006: HYDRUS Richards Equation (Dong 2019 / van Genuchten 1980)

**Paper**: Richards (1931), van Genuchten (1980), Dong et al. (2019) J Sustainable Water 5(4):04019005

**Control**: `control/richards/richards_1d.py` — 14/14 checks. Van Genuchten retention, Mualem conductivity, sand infiltration, silt loam drainage, steady-state flux.

**Rust**: `barracuda/src/eco/richards.rs` — Implicit Euler + Picard iteration, Thomas algorithm. `validate_richards` binary: 15/15 checks.

**GPU**: `gpu::richards::BatchedRichards` wired to `barracuda::pde::richards::solve_richards` (Tier B). Cross-validates eco::richards (implicit Euler) against upstream (Crank-Nicolson).

### Exp 007: Biochar Adsorption Isotherms (Kumari et al. 2025)

**Paper**: Kumari, Dong & Safferman (2025) Applied Water Science 15(7):162

**Control**: `control/biochar/biochar_isotherms.py` — 14/14 checks. Langmuir and Freundlich isotherm fitting for P adsorption on wood and sugar beet biochar.

**Rust**: `barracuda/src/eco/isotherm.rs` — Linearized least squares + grid refinement. `validate_biochar` binary: 14/14 checks.

**GPU**: `gpu::isotherm::fit_langmuir_nm` / `fit_freundlich_nm` wired to `barracuda::optimize::nelder_mead`. Linearized LS as initial guess → NM refinement matches scipy.curve_fit.

### Exp 015: 60-Year Water Balance Reconstruction

**Data**: Open-Meteo ERA5 archive, 1960-2023, Wooster OH (OSU OARDC)

**Control**: `control/long_term_wb/long_term_water_balance.py` — 10/10 checks. 64 growing seasons, decade trends, climate signal detection.

**Rust**: Existing `eco::water_balance` + `eco::evapotranspiration::hargreaves_et0`. `validate_long_term_wb` binary: 11/11 checks.

**GPU**: `BatchedEt0` + `BatchedWaterBalance` at 64-year scale (already wired).

### Exp 008: Yield Response to Water Stress (Stewart 1977 / FAO-56 Ch 10)

**Paper**: Stewart (1977); Allen et al. (1998) *FAO-56 Chapter 10*, Table 24 (Doorenbos & Kassam 1979); Ali, Dong & Lavely (2024) Ag Water Mgmt 306:109148

**Control**: `control/yield_response/yield_response.py` — 32/32 checks. Ky table values (7 crops), single-stage Stewart equation (8 analytical), multi-stage product formula (5 analytical), WUE calculations (4 crops), scheduling comparison (3 strategies × 2 metrics + 2 ordering checks).

**Benchmark**: `control/yield_response/benchmark_yield_response.json` — FAO-56 Table 24 Ky values, Stewart equation test vectors, multi-stage product formula, WUE, scheduling scenarios.

**Rust**: `barracuda/src/eco/yield_response.rs` — `yield_ratio_single`, `yield_ratio_multistage`, `water_use_efficiency`, `ky_table` (9 crops). `validate_yield` binary: 32/32 checks. 16 unit tests.

**GPU**: `BatchedElementwiseF64` yield batch (Tier B — ready for GPU promotion).

### Exp 012: CW2D Richards Extension (Dong 2019)

**Paper**: Dong et al. (2019) *Land-based wastewater treatment system modeling using HYDRUS CW2D.* J Sustainable Water 5(4):04019005

**Control**: `control/cw2d/cw2d_richards.py` — 24/24 checks. VG retention curves for 4 CW2D media (gravel, coarse sand, organic, fine gravel), Mualem conductivity, gravel infiltration, organic drainage, mass balance.

**Benchmark**: `control/cw2d/benchmark_cw2d.json` — HYDRUS CW2D standard media parameters (Šimůnek et al. 2012), analytical VG values, solver convergence checks.

**Rust**: Reuses existing `barracuda/src/eco/richards.rs` — validates same solver on extreme VG parameters. `validate_cw2d` binary: 24/24 checks. No new Rust module needed (parameter-driven validation).

**GPU**: Reuses `gpu::richards::BatchedRichards` — CW2D parameters work with existing GPU pipeline.

### Exp 014: Irrigation Scheduling Optimization

**Paper**: Ali, Dong & Lavely (2024) *Irrigation scheduling optimization for corn yield.* Ag Water Mgmt 306:109148

**Control**: `control/scheduling/irrigation_scheduling.py` — 25/25 checks. 5-strategy comparison (rainfed, MAD 50/60/70%, growth-stage), mass balance closure < 1e-13 mm, yield ordering monotonicity, WUE analysis.

**Rust**: `barracuda/src/bin/validate_scheduling.rs` — 28/28 checks. Deterministic weather (sinusoidal ET₀ + periodic rain). Composes `eco::water_balance` + `eco::yield_response` into full scheduling pipeline.

**Key Result**: Growth-stage scheduling achieves best WUE (22.3 kg/ha/mm) by targeting irrigation to critical mid-season period. Full "Penny Irrigation" precursor pipeline.

### Exp 016: Lysimeter ET Direct Measurement

**Paper**: Dong & Hansen (2023) *Affordable weighing lysimeter design.* Smart Ag Tech 4:100147

**Control**: `control/lysimeter/lysimeter_et.py` — 26/26 checks. Mass-to-ET conversion, temperature compensation (α=2.5 g/°C), data quality filtering, load cell calibration (R²=0.9999), diurnal ET pattern, synthetic daily comparison.

**Rust**: `barracuda/src/bin/validate_lysimeter.rs` — 25/25 checks. Deterministic synthetic comparison (r=0.974, RMSE=0.106 mm).

**Key Result**: Direct ET measurement ground truth for equation-based ET₀ calibration.

### Exp 017: ET₀ Sensitivity Analysis

**Paper**: Gong et al. (2006) *Sensitivity of PM ET₀.* Ag Water Mgmt 86:57-63

**Control**: `control/sensitivity/et0_sensitivity.py` — 23/23 checks. OAT ±10% perturbation of 6 variables across 3 climatic zones (humid, arid, tropical). Monotonicity, elasticity bounds, symmetry, multi-site ranking.

**Rust**: `barracuda/src/bin/validate_sensitivity.rs` — 23/23 checks. Uses `barracuda::eco::evapotranspiration` directly.

**Key Result**: Solar radiation and wind consistently dominate across all climates. Complements groundSpring Exp 003 (humidity at 66% of MC variance).

### Exp 018: Michigan Crop Water Atlas

**Data**: Open-Meteo ERA5 archive, 100 Michigan stations, up to 80 years daily weather (free, no API key). See `specs/ATLAS_STATION_LIST.md`.

**Control**: `control/atlas/atlas_water_budget.py` — Runs FAO-56 ET₀ + water balance + Stewart yield response for 10 crops × all available stations.

**Rust**: `barracuda/src/bin/validate_atlas.rs` — **1354/1354 checks** on 100 Michigan stations. Discovers CSVs at runtime, computes ET₀ (R² > 0.96 vs Open-Meteo), runs water balance for 10 crops per station-year, checks mass balance (< 0.01 mm), yield ratio bounds, and aggregate Michigan ET₀ statistics.

**Cross-validation**: Python vs Rust — 690 crop-station yield ratios all within 0.01 (mean diff 0.0003). Mean ET₀ diff 0.133% across matched stations.

**Output**: `data/atlas_results/atlas_station_summary.csv` (100 rows) and `atlas_crop_summary.csv` (1000 rows) — per-station and per-crop seasonal water budgets.

**Scale**: 100 stations × 10 crops × 80 years = 29.2B cell-days. 100-station pilot: 15,300 station-days processed in 141s (release mode). Full 80yr: ~2hr CPU, ~5min GPU (estimated).

**Key Result**: All 100 stations show ET₀ R² > 0.96, mass balance = 0.000 mm, yield ratios 0.99+ with smart irrigation. Statewide mean ET₀ = 640 mm (growing season 2023).

**GPU**: Candidate for `BatchedEt0` + `BatchedWaterBalance` at atlas scale. Kriging interpolation (100 stations → 10km grid) via `gpu::kriging`.

---

## Naming Convention

Experiments follow `NNN_name` format:
- `001`–`005`: Baseline reproduction (FAO-56, soil, IoT, water balance, real data)
- `006`–`007`: Richards equation, biochar isotherms (Track 2)
- `008`: Yield response to water stress (Stewart 1977 / FAO-56 Ch 10)
- `009`–`011`: Dual Kc, regional ET₀, cover crops + no-till
- `012`: CW2D Richards extension (Dong 2019)
- `014`: Irrigation scheduling optimization (Ali, Dong & Lavely 2024)
- `015`: Long-term water balance reconstruction
- `016`: Lysimeter ET measurement (Dong & Hansen 2023)
- `017`: ET₀ sensitivity analysis (Gong 2006 methodology)
- `018`: Michigan Crop Water Atlas (100 stations × 10 crops × 80yr)
- `019`–`020`: Priestley-Taylor ET₀, 3-method intercomparison
- `021`–`023`: Thornthwaite, GDD, pedotransfer functions
- `024`–`026`: NASS yield, forecast scheduling, SCAN soil moisture
- `027`: Multi-crop water budget (5 Michigan crops)
- `028`–`029b`: NPU edge inference trilogy (AKD1000 live)
- `030`: AmeriFlux eddy covariance ET (Baldocchi 2003)
- `031`: Hargreaves-Samani temperature ET₀
- `032`: Ecological diversity indices
- `033`–`035`: Makkink, Turc, Hamon ET₀ methods (completing 7-method portfolio)
- `036`–`044`: biomeOS Neural API, ensemble, pedotransfer-Richards coupling, CPU-GPU parity, metalForge dispatch, seasonal batch, Titan V live, live hardware
- `045`: Anderson soil-moisture coupling (cross-spring)
- `046`: Atlas stream real data validation (80yr Open-Meteo, seasonal pipeline + atlas stream)
- `047`: GPU math portability validation (all 13 GPU modules, CPU vs GPU parity)
- `049`: Blaney-Criddle ET₀
- `050`: SCS Curve Number runoff
- `051`: Green-Ampt infiltration
- `052`: Coupled runoff-infiltration (SCS-CN + Green-Ampt)
- `053`: Van Genuchten inverse parameter estimation
- `054`: Full-season irrigation water budget audit

Gap (013) reserved. See `specs/PAPER_REVIEW_QUEUE.md`.

---

### Exp 033: Makkink (1957) Radiation-Based ET₀

**Paper**: Makkink GF (1957) *Testing the Penman formula by means of lysimeters.* J Inst Water Eng 11:277-288. De Bruin HAR (1987) *From Penman to Makkink.* TNO, The Hague, pp 5-31.

**Control**: `control/makkink/makkink_et0.py` — 21/21 checks. Analytical benchmarks, PM cross-comparison (Xu & Singh 2002 ratio bounds), edge cases, monotonicity, pyet cross-validation.

**Rust**: `barracuda/src/eco/evapotranspiration.rs` — `makkink_et0(tmean_c, rs_mj, elevation_m)`. `validate_makkink` binary: 16/16 checks.

**Equation**: ET₀ = 0.61 × (Δ/(Δ+γ)) × Rs/λ − 0.12 (de Bruin 1987 coefficients).

**Key Result**: Makkink/PM ratio 0.57–0.85 across climate zones. Radiation-only method suitable for KNMI-style networks lacking wind/humidity.

### Exp 034: Turc (1961) Temperature-Radiation ET₀

**Paper**: Turc L (1961) *Évaluation des besoins en eau d'irrigation.* Annales Agronomiques 12:13-49.

**Control**: `control/turc/turc_et0.py` — 22/22 checks. Analytical (RH ≥ 50% and RH < 50% branches), humidity boundary continuity, edge cases, monotonicity, pyet cross-validation (diff < 0.002 mm/d).

**Rust**: `barracuda/src/eco/evapotranspiration.rs` — `turc_et0(tmean_c, rs_mj, rh_pct)`. `validate_turc` binary: 17/17 checks.

**Equation**: ET₀ = 0.013 × T/(T+15) × (23.8846 Rs + 50), with arid correction for RH < 50%.

**Key Result**: Exact match with pyet.turc() (< 0.002 mm/day). The humidity correction multiplier ranges 1.0–1.57x.

### Exp 035: Hamon (1961) Temperature-Based PET

**Paper**: Hamon WR (1961) *Estimating potential evapotranspiration.* J Hydraulics Div ASCE 87(HY3):107-120. Lu J, et al. (2005) *A comparison of six PET methods.* J Am Water Resour Assoc 41(3):621-633.

**Control**: `control/hamon/hamon_pet.py` — 20/20 checks. Analytical, daylight hour computation (FAO-56 Eq. 34), edge cases, monotonicity, pyet rank-correlation (different formulation variant).

**Rust**: `barracuda/src/eco/evapotranspiration.rs` — `hamon_pet(tmean_c, day_length_hours)`, `hamon_pet_from_location(tmean_c, latitude_rad, doy)`. `validate_hamon` binary: 19/19 checks.

**Equation**: PET = 0.1651 × N × RHOSAT (Lu et al. 2005). Minimum data: temperature + latitude.

**Key Result**: Simplest method in portfolio (T + day length only). Rank-correlated with pyet despite coefficient variant difference (~3x). Suitable for data-sparse historical reconstruction.

### Exp 049: Blaney-Criddle (1950) Temperature PET

**Paper**: Blaney & Criddle (1950) *Determining water requirements in irrigated areas from climatological and irrigation data.* USDA SCS-TP 96.

**Control**: `control/blaney_criddle/blaney_criddle.py` — 18/18 checks. Temperature-based PET, monthly k factor, pyet cross-validation.

**Rust**: `barracuda/src/eco/evapotranspiration.rs` — `blaney_criddle_pet()`. `validate_blaney_criddle` binary: 18/18 checks.

**Key Result**: Minimal-input PET method (T + latitude only). Completes temperature-based ET₀ portfolio with Thornthwaite, Hamon, Hargreaves.

### Exp 050: SCS Curve Number Runoff (USDA 1972)

**Paper**: USDA (1972) *National Engineering Handbook Section 4: Hydrology.* SCS Curve Number method for rainfall-runoff.

**Control**: `control/scs_curve_number/scs_curve_number.py` — 38/38 checks. CN lookup (hydrologic soil group × land use), AMC I/II/III, retention S, runoff Q.

**Rust**: `barracuda/src/eco/runoff.rs` — `scs_curve_number()`, `scs_runoff()`, AMC adjustment. `validate_scs_cn` binary: 38/38 checks.

**Key Result**: Industry-standard event-based runoff. Integrates with water balance for irrigation scheduling and flood design.

### Exp 051: Green-Ampt (1911) Infiltration

**Paper**: Green & Ampt (1911) *Studies on soil physics.* J Agric Sci 4(1):1-24. Mein & Larson (1973) *Application of Green-Ampt.* Water Resour Res 9(2):384-394.

**Control**: `control/green_ampt/green_ampt.py` — 37/37 checks. Cumulative infiltration, wetting front suction, Newton-Raphson iteration, ponding time.

**Rust**: `barracuda/src/eco/infiltration.rs` — `green_ampt_infiltration()`, Newton-Raphson solver. `validate_green_ampt` binary: 37/37 checks.

**Key Result**: Physics-based infiltration for event-scale modeling. Complements Richards for rapid storm events.

### Exp 052: SCS-CN + Green-Ampt Coupled Runoff-Infiltration

**Papers**: USDA-SCS (1972) NEH-4; Green & Ampt (1911); Rawls et al. (1983).

**Control**: `control/coupled_runoff_infiltration/coupled_runoff_infiltration.py` — 292/292 checks. Couples SCS Curve Number runoff with Green-Ampt cumulative infiltration across 48 storm × soil × land-use combinations, 80-point conservation sweep, 4 monotonicity checks, and 160 sensitivity perturbations.

**Rust**: `barracuda/src/bin/validate_coupled_runoff.rs` — 292/292 checks. Validates coupled partitioning: rainfall → runoff (SCS-CN) → net rain → infiltration (Green-Ampt) → surface storage, with mass balance ≤ 1e-8 mm.

**Key Result**: Module coupling preserves mass conservation across all 292 scenarios. Demonstrates that `eco::runoff` + `eco::infiltration` compose cleanly for event-scale hydrology.

### Exp 053: Van Genuchten Inverse Parameter Estimation

**Paper**: van Genuchten (1980), Carsel & Parrish (1988) Table 1.

**Control**: `control/vg_inverse/vg_inverse_fitting.py` — 84/84 checks. Forward VG retention θ(h), Mualem hydraulic conductivity K(h), θ→h→θ round-trip via bisection, monotonicity for 7 USDA textures at 5 Se fractions.

**Rust**: `barracuda/src/bin/validate_vg_inverse.rs` — 84/84 checks. Uses `barracuda::optimize::brent` for inverse θ→h root-finding, validates K(h) monotonicity and round-trip accuracy.

**Key Result**: Brent inversion converges for all practical Se fractions. Demonstrates that `barracuda::optimize::brent` serves inverse problems in soil physics.

### Exp 054: Full-Season Irrigation Water Budget Audit

**Papers**: Allen et al. (1998) FAO-56 Ch 2-8; Stewart (1977) yield response.

**Control**: `control/season_water_budget/season_water_budget.py` — 34/34 checks. Synthetic deterministic weather → FAO-56 PM ET₀ → trapezoidal Kc schedule → daily water balance → Stewart yield for 4 crops (corn, soybean, winter wheat, alfalfa). Mass conservation, ETa ≤ ETc, yield 0–1, cross-crop comparison.

**Rust**: `barracuda/src/bin/validate_season_wb.rs` — 34/34 checks. Replicates the complete chain end-to-end. Validates mass balance to 0.1 mm, physical bounds, and crop ordering.

**Key Result**: End-to-end pipeline audit confirms all `eco` modules compose correctly for a full growing season simulation.

## Adding a New Experiment

1. Create benchmark JSON: `control/{name}/benchmark_{name}.json` with provenance
2. Write Python baseline: `control/{name}/{name}.py` with provenance docstring
3. Run baseline: validate against paper, add to `CONTROL_EXPERIMENT_STATUS.md`
4. Write Rust validation binary: `barracuda/src/bin/validate_{name}.rs`
5. Add `[[bin]]` to `barracuda/Cargo.toml`
6. Add row to experiment index above
7. Update counts in README, CHANGELOG, whitePaper docs

## CPU Benchmark: Rust vs Python (25.9× geometric mean, 8/8 parity)

Formal 8-algorithm benchmark with Python timing subprocess and Rust `black_box`.
Same algorithms, same f64 precision, same inputs, same outputs.

| Algorithm | N | Rust (s) | Python (s) | Speedup | Parity |
|-----------|---:|---:|---:|---:|:---:|
| FAO-56 PM ET₀ | 10K | 0.0008 | 0.012 | **15×** | ✓ |
| Hargreaves-Samani | 10K | 0.00001 | 0.001 | **114×** | ✓ |
| Water Balance Step | 10K | 0.00001 | 0.001 | **190×** | ✓ |
| Anderson Coupling | 100K | 0.0002 | 0.023 | **94×** | ✓ |
| Season Sim (153d) | 1K | 0.001 | 0.056 | **44×** | ✓ |
| Shannon Diversity | 10K | 0.0002 | 0.005 | **26×** | ✓ |
| Van Genuchten θ(h) | 100K | 0.002 | 0.015 | **6×** | ✓ |
| Thornthwaite PET | 10K | 0.084 | 0.081 | **1×** | ✓ |

**Geometric mean speedup: 25.9×** (8/8 parity)

Thornthwaite 1× is expected — Rust computes daylight hours per day (365 trig
calls) while Python uses mid-month approximation (12 calls). Higher fidelity = more work.

Reproduce:
```sh
cargo run --release --bin bench_cpu_vs_python
```

---

## GPU Benchmark: CPU vs GPU Orchestrators

The GPU benchmark measures throughput for 11 Tier A wired modules through
ToadStool's shader evolution (774 WGSL shaders, 46+ cross-spring absorptions):

| Orchestrator | CPU (items/s) | Notes |
|---|---:|---|
| Batched ET₀ | 8.6M (N=10K) | `batched_elementwise_f64.wgsl` |
| Seasonal Reduce | 244M elem/s | `fused_map_reduce_f64.wgsl` |
| Stream Smoothing | 33M elem/s | `moving_window.wgsl` (24h window) |
| Kriging (20→500) | 500 targets in 25µs | `kriging_f64.wgsl` |
| Ridge Regression | 5K in 50µs, R²=1.0 | `barracuda::linalg::ridge` |
| Richards PDE (100 nodes) | 36/s | Crank-Nicolson + Thomas |
| VG θ(h) batch (100K) | 37.7M/s | `df64` precision |
| Isotherm NM | 36.6K fits/s | Multi-start Nelder-Mead |

GPU dispatches are CPU-only in this benchmark (no GPU hardware detected).
BarraCuda GPU validation will show identical math with GPU dispatch overhead.

Reproduce:
```sh
cargo run --release --bin bench_airspring_gpu
```

---

## Results

Benchmark data is stored in `control/*/benchmark_*.json` (digitized paper values)
and used by both Python control scripts and Rust validation binaries as ground
truth. Cross-validation outputs are produced by `cross_validate` (Rust) and
`scripts/cross_validate.py` (Python). Benchmark throughput data is in
`scripts/bench_python_results.json` and `scripts/bench_comparison.json`.
