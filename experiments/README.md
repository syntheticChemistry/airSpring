# airSpring Experiments

**Updated**: March 15, 2026
**Status**: 87 experiments, barraCuda 0.3.5 (wgpu 28), v0.8.2, Edition 2024 (rust-version 1.87). 1284/1284 Python + 848 lib + 280 integration + 61 forge + 381/381 validation checks + 146/146 cross-spring evolution + 33/33 cross-validation. **14.3× Rust-vs-Python speedup** (24/24 parity). 21/21 CPU-GPU parity modules. Niche architecture: 41 capabilities, 4 deploy graphs, BYOB niche deployment. All 20 ops upstream (`BatchedElementwiseF64`), `local_dispatch` retired. Deep code quality: zero `#[allow()]` in production (91 binaries cleaned), zero clippy pedantic+nursery warnings, zero unsafe in production AND tests (DI `_with`/`_in` pattern). 57 tolerances in 4 domain submodules. Full validation pipeline green (2026-03-15).

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
| 057 | GPU Ops 5-8 Rewire Validation + Benchmark | GPU | **Complete** | Rust GPU (Titan V) | `BatchedElementwiseF64` ops 0-8 + `gpu::seasonal_pipeline` | 26 |
| 058 | Climate Scenario Analysis | Integration | **Complete** | Rust CPU | `eco::evapotranspiration` + `eco::water_balance` + `eco::yield_response` | 46 |
| 059 | Atlas 80yr Decade Analysis | Integration | **Complete** | Rust CPU + Open-Meteo | `data::open_meteo` + `eco::evapotranspiration` | 102 |
| 060 | NASS Real Yield Comparison | Irrigation | **Complete** | Rust CPU + USDA NASS | `data::usda_nass` + `eco::yield_response` | 99 |
| 061 | Cross-Spring Diversity (NCBI 16S) | Biodiversity | **Complete** | Rust CPU | `eco::diversity` + wetSpring Shannon H' | 63 |
| 062 | NUCLEUS Integration Validation | NUCLEUS | **Complete** | Rust + biomeOS | `airspring_primal` — JSON-RPC parity (9 science methods) | 29 |
| 063 | NUCLEUS Cross-Primal Pipeline | NUCLEUS | **Complete** | Rust + biomeOS + neural-api | ecology domain, cross-primal forwarding, capability.call routing | 28 |
| 064 | Full Dispatch Experiment | GPU+CPU | **Complete** | Rust CPU+GPU | 21 CPU science + 5 GPU domains + batch scaling + absorption audit + mixed pipeline | 51 |
| 065 | biomeOS Graph Experiment | NUCLEUS | **Complete** | Rust + biomeOS | deployment graph topology, 30 capabilities, offline pipeline, GPU parity, evolution manifest | 35 |
| 066 | Tissue Diversity Profiling (Paper 12) | Immunological | **Complete** | Python + Rust CPU | `eco::tissue` — Pielou→Anderson W, regime classification, barrier d_eff, multi-compartment | 30+30 |
| 067 | CytokineBrain Regime Prediction (Paper 12) | Immunological | **Complete** | Python + Rust CPU | `eco::cytokine` — Nautilus reservoir, 3-head AD flare prediction, DriftMonitor | 14+28 |
| 068 | Barrier State Model (Paper 12) | Immunological | **Complete** | Python + Rust CPU | `eco::van_genuchten` + `eco::tissue` — VG θ(h)/K(h) for skin, dimensional promotion | 16+16 |
| 069 | Cross-Species Skin Comparison (Paper 12) | Immunological | **Complete** | Python + Rust CPU | `eco::diversity` + `eco::tissue` — canine/human/feline Anderson, One Health bridge | 19+20 |
| 070 | GPU Streaming Multi-Field Pipeline | GPU Pipeline | **Complete** | Rust CPU + GPU | `SeasonalPipeline::run_multi_field()` — M fields × N days, Stage 3 `gpu_step()`, 6.8M field-days/s | 57 |
| 071 | CPU Parity & Speedup Benchmark | CPU Parity | **Complete** | Rust CPU | All 9 domains (ET₀, HG, PT, WB, Kc, Yield, Diversity, Seasonal, Atlas), 10M ET₀/s, 13K× Python | 34 |
| 072 | Pure GPU End-to-End Multi-Field | Pure GPU | **Complete** | Rust CPU + GPU | All 4 stages on GPU, CPU↔GPU parity, 19.7× dispatch reduction, scaling 1→50 fields | 46 |
| 073 | Cross-Spring Evolution Rewire | GPU Rewire | **Complete** | Rust CPU + GPU | `BrentGpu` VG inverse, `RichardsGpu` Picard, `StatefulPipeline`, hydrology parity, 5-spring provenance, benchmarks | 68 |
| 074 | Paper Chain Validation | Integration | **Complete** | Rust CPU + GPU | Full CPU→GPU→metalForge chain for 28 domains (22 GPU, 6 CPU-only), `validate_paper_chain` binary | 79 |
| 075 | Local GPU Parity Validation | GPU Local | **Complete** | Rust CPU + GPU (wgpu) | 6 local WGSL ops (SCS-CN, Stewart, Makkink, Turc, Hamon, BC), f32 GPU vs f64 CPU parity, batch scaling, edge cases | ALL |
| 076 | NUCLEUS Mixed-Hardware Routing | metalForge | **Complete** | Rust (forge) | 27 workloads, NUCLEUS mesh routing, PCIe P2P bypass, 7-stage pipeline, Tower/Node/Nest atomics, multi-node cross-hop | 60 |
| 077 | Cross-Spring Provenance & CPU↔GPU Benchmark | GPU + Cross-Spring | **Complete** | Rust (barracuda) | CPU vs GPU timing for ET₀/WB/Hargreaves/Kc/VG/GDD/pedotransfer, shader provenance tracking (5 springs), precision lineage validation (hotSpring→all), seasonal pipeline parity, uncertainty (jackknife/bootstrap/diversity) | 32 |
| 078 | Cross-Spring Evolution — Universal Precision | GPU + Cross-Spring | **Complete** | Rust (barracuda) | f64 canonical → compile_shader_universal (6 ops: SCS-CN, Stewart, Makkink, Turc, Hamon, Blaney-Criddle), cross-spring provenance (hotSpring precision, wetSpring bio, groundSpring uncertainty, neuralSpring architecture), f64 compute shader reliability discovery (NVK/Mesa) | — |
| 079 | Monte Carlo ET₀ Uncertainty Propagation | Stochastic/UQ | **Complete** | Python + Rust CPU | `gpu::mc_et0::mc_et0_cpu` — Lehmer LCG + Box-Muller MC sampling, input uncertainty → ET₀ distribution | 47+26 |
| 080 | Bootstrap & Jackknife CI for Seasonal ET₀ | Stochastic/UQ | **Complete** | Python + Rust CPU | `gpu::bootstrap::GpuBootstrap::cpu()`, `gpu::jackknife::GpuJackknife::cpu()` — deterministic bootstrap resampling + jackknife LOO variance | 20+20 |
| 081 | Standardized Precipitation Index (SPI) | Drought/Hydrology | **Complete** | Python + Rust CPU | `eco::drought_index` — gamma MLE, regularized incomplete gamma, normal quantile, multi-scale SPI (1/3/6/12), WMO classification | 20+20 |
| 082 | Cross-Spring Modern Systems Validation | Integration | **Complete** | Rust | `gpu::autocorrelation`, provenance registry, PrecisionRoutingAdvice, special functions, cross-spring shader flows | 36/36 |
| 083 | NUCLEUS Modern Deployment Validation | Integration | **Complete** | Rust | biomeOS NUCLEUS (Tower/Node), primal JSON-RPC (SPI, ACF, gamma_cdf), full pipeline, cross-primal discovery, GPU precision routing | 43/43 |
| 084 | CPU vs GPU Comprehensive Parity | GPU | **Complete** | Rust | All 18 GPU modules: FAO-56, Hargreaves, SCS-CN, Yield, Makkink, Turc, Hamon, Blaney-Criddle, VG θ/K, Thornthwaite, GDD, Pedotransfer, Infiltration, Autocorrelation, Bootstrap, Jackknife, Diversity, Reduce | 21/21 |
| 085 | toadStool Compute Dispatch | Integration | **Complete** | Rust | In-process science dispatch (14 methods), compute.offload flow, cross-primal discovery (7 primals), precision routing, provenance chains | 19/19 |
| 086 | metalForge Mixed Hardware Live NUCLEUS | Hardware | **Complete** | Rust | Live probe (RTX 4070 + Titan V + i9-12900K), NUCLEUS mesh (Tower+Node), 23/27 workload routing, ecology pipeline (3 stages GPU), PCIe bypass, transfer matrix | 17/17 |
| 087 | NUCLEUS Graph Coordination | Integration | **Complete** | Rust | biomeOS TOML graph parsing, DAG validation, capability refs, dependency ordering, prerequisite checks, Tower/Node atomic detection, 7 primals | 22/22 |

**Grand Total**: 1284 Python + **848 lib + 280 integration + 61 forge tests** + 381/381 validation + 146/146 cross-spring evolution + 33/33 cross-validation + 25 Tier A (ops 0-19 upstream) + `local_dispatch` retired + `PrecisionRoutingAdvice` + upstream provenance registry + 4 GPU orchestrators + `BrentGpu` + `RichardsGpu` + seasonal pipeline GPU Stages 1-3 + metalForge 66/66 cross-system + NUCLEUS primal (41 capabilities) + 91 binaries + barraCuda 0.3.5 (wgpu 28, DF64 precision tier) + 14.3× CPU speedup (24/24 parity) + 21/21 CPU-GPU parity modules + 87 experiments (v0.8.2). Exp 084 CPU/GPU 21/21, Exp 085 toadStool 19/19, Exp 086 metalForge NUCLEUS 17/17, Exp 087 Graphs 22/22. Full NUCLEUS mesh: Tower+Node+Nest live. Full validation pipeline green (2026-03-15).

---

## Test Breakdown (v0.8.2)

| Category | Tests | Source |
|----------|:-----:|--------|
| Barracuda lib (unit + doc) | 848 | `cargo test --lib` |
| Barracuda integration | 280 | `cargo test --tests` (barracuda/tests/) |
| Barracuda validation binaries | 91 | `validate_*`, `bench_*`, `cross_validate`, `simulate_season` |
| Forge | 61 | `metalForge/forge/` (substrate, dispatch, probe, workloads, cross-system routing) |
| Forge binaries | 6 | `validate_dispatch`, `validate_live_hardware`, `validate_dispatch_routing`, `validate_mixed_pipeline`, `validate_mixed_nucleus_live`, `validate_nucleus_routing` |
| **Total project tests** | **848 lib + 280 integration + 61 forge** | |
| Validation checks | 381/381 | 10 validation binaries |
| Cross-spring evolution | 146/146 | `bench_cross_spring` (34 provenance entries, 6 origin Springs) |
| Cross-validation | 33/33 | Python↔Rust match (tol=1e-5) |
| CPU vs Python parity | 24/24 | `bench_cpu_vs_python` (20.6× geometric mean speedup) |

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
- `055`–`074`: GPU live, mixed-hardware, NUCLEUS, paper chain, streaming, cross-spring rewire
- `075`: Local GPU parity validation (6 ops via `local_elementwise.wgsl`)
- `076`: NUCLEUS mixed-hardware routing (27 workloads, mesh routing, PCIe bypass)
- `077`: Cross-spring provenance & CPU↔GPU benchmark (5-spring shader provenance, precision lineage)
- `079`: Monte Carlo ET₀ uncertainty propagation (Lehmer LCG + Box-Muller MC sampling)
- `080`: Bootstrap & Jackknife CI for seasonal ET₀ (deterministic resampling)
- `081`: Standardized Precipitation Index (SPI) drought analysis (gamma MLE + normal quantile)
- `082`: Cross-Spring Modern Systems Validation (provenance, autocorrelation, PrecisionRoutingAdvice)
- `083`: NUCLEUS Modern Deployment Validation (biomeOS, Tower/Node, 35 JSON-RPC, SPI/ACF/gamma_cdf)
- `084`: CPU vs GPU Comprehensive Parity (18 modules, all GPU ops, tolerance-aware)
- `085`: toadStool Compute Dispatch (14 methods, compute.offload, 7 primals discovered)
- `086`: metalForge Mixed Hardware Live NUCLEUS (live probe, NUCLEUS mesh, ecology pipeline)
- `087`: NUCLEUS Graph Coordination (TOML graphs, DAG validation, capability refs, Tower+Node)

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

### Exp 079: Monte Carlo ET₀ Uncertainty Propagation

**Paper**: FAO-56 (Allen 1998) ET₀ under parametric input uncertainty; Beven (2009) uncertainty in environmental modeling.

**Control**: `control/mc_et0/mc_et0_propagation.py` — 47/47 checks. Deterministic Lehmer LCG (m=2³¹-1, a=16807) + Box-Muller transform for reproducible MC sampling. Simplified FAO-56 PM ET₀ with Gaussian perturbation of T, RH, u₂, Rₛ. Tests: default N=2000, zero uncertainty, high uncertainty (10% CV), arid/humid climate gradient, convergence at N=500/1000/2000, determinism (two runs identical), parametric CI consistency.

**Benchmark**: `control/mc_et0/benchmark_mc_et0.json` — with provenance (script, commit, date, Python version).

**Rust**: `barracuda/src/bin/validate_mc_et0.rs` — 26/26 checks. Validates `gpu::mc_et0::mc_et0_cpu` against Python benchmark. Tolerance: `MC_ET0_PROPAGATION` (abs=0.15, rel=0.08).

**Key Result**: MC uncertainty propagation confirms ET₀ std dev ~0.2-0.5 mm/day for typical input uncertainty (3-5% CV), validating the stochastic infrastructure for GPU promotion.

### Exp 080: Bootstrap & Jackknife CI for Seasonal ET₀

**Paper**: Efron (1979) bootstrap method; Quenouille-Tukey jackknife; seasonal ET₀ confidence estimation.

**Control**: `control/bootstrap_jackknife/bootstrap_jackknife_et0.py` — 20/20 checks. Generates deterministic synthetic seasonal ET₀ series, implements Lehmer RNG bootstrap resampling (B=1000), jackknife LOO variance. Tests: full season statistics, known analytical values (mean=5.0, SE=1/√10), small sample (n=5), constant data (CI width=0, variance=0).

**Benchmark**: `control/bootstrap_jackknife/benchmark_bootstrap_jackknife.json` — with provenance.

**Rust**: `barracuda/src/bin/validate_bootstrap_jackknife.rs` — 20/20 checks. Validates `gpu::bootstrap::GpuBootstrap::cpu()` and `gpu::jackknife::GpuJackknife::cpu()`. Checks CI bounds, SE, variance, and monotonicity (larger data → tighter CI).

**Key Result**: Bootstrap CI and jackknife variance agree with analytical expectations. Infrastructure validated for GPU-parallel uncertainty quantification.

### Exp 081: Standardized Precipitation Index (SPI)

**Paper**: McKee et al. (1993) SPI definition; Thom (1958) gamma MLE; Edwards & McKee (1997) WMO drought classification.

**Control**: `control/drought_index/drought_index_spi.py` — 20/20 checks. Implements gamma distribution MLE fitting (`Thom` method), regularized incomplete gamma function (series + continued fraction), inverse normal CDF (`norm_ppf`), and multi-scale SPI computation (SPI-1, SPI-3, SPI-6, SPI-12). WMO drought classification (extremely wet → extremely dry). Tests: gamma fit accuracy, SPI range [-3,3], scale monotonicity, dry/wet month classification.

**Benchmark**: `control/drought_index/benchmark_drought_index.json` — with provenance. NaN sanitized to null for JSON compatibility.

**Rust**: `barracuda/src/eco/drought_index.rs` — `DroughtClass`, `GammaParams`, `gamma_mle_fit`, `regularized_gamma_p` (series + continued fraction), `gamma_cdf`, `compute_spi`. Uses `barracuda::special::gamma::ln_gamma`. `validate_drought_index` binary: 20/20 checks.

**Key Result**: SPI correctly identifies drought periods in synthetic precipitation record. Multi-scale analysis (SPI-1 vs SPI-12) reveals different temporal drought signals. GPU-promotable: each grid cell's SPI is independent.

### Exp 082: Cross-Spring Modern Systems Validation

**Goal**: Validate the complete modern upstream integration — barraCuda HEAD,
toadStool S130+, coralReef Phase 10. Exercises provenance registry, cross-spring
matrix, `PrecisionRoutingAdvice`, `regularized_gamma_p` lean, `gpu::autocorrelation`,
special functions, and cross-spring shader flows.

**Phase 1 (Rust — 36/36 PASS):**
- [x] Provenance registry: 28 shaders, 10 evolution events, all 5 springs
- [x] Cross-spring matrix: every spring contributes AND consumes
- [x] `regularized_gamma_p` delegation (v0.7.5 lean from `eco::drought_index`)
- [x] `gpu::autocorrelation` — wraps upstream `AutocorrelationF64`, NVK zero-output CPU fallback
- [x] Special functions: upstream `digamma`, `beta`, `ln_beta`, `norm_ppf`
- [x] `PrecisionRoutingAdvice` from `DevicePrecisionReport`
- [x] airSpring provenance integration (≥5 upstream shaders, evolution report)

**Binary**: `validate_cross_spring_modern`

**Key Result**: Full modern upstream integration validated. `gpu::autocorrelation`
enables cross-spring time-series analysis with NVK-safe CPU fallback.

### Exp 083: NUCLEUS Modern Deployment Validation

**Goal**: End-to-end validation of biomeOS/NUCLEUS integration with v0.7.5
capabilities. Exercises NUCLEUS atomic detection (Tower/Node), primal socket
discovery, JSON-RPC capability enumeration, new science endpoints, full ecology
pipeline, cross-primal discovery, and GPU precision routing.

**Phase 1 (Rust — 43/43 PASS):**
- [x] NUCLEUS atomic detection: Tower (BearDog+Songbird) LIVE, Node (+ToadStool) LIVE
- [x] Primal socket discovery and health check (v0.7.5)
- [x] v0.7.5 capability enumeration: 35 capabilities (SPI, ACF, gamma_cdf + ecology aliases)
- [x] SPI drought index via JSON-RPC: parity direct-Rust vs RPC, upstream provenance
- [x] Autocorrelation via JSON-RPC: cross-spring provenance (hotSpring→neuralSpring→airSpring)
- [x] Gamma CDF via JSON-RPC: upstream `regularized_gamma_p` lean confirmed
- [x] Full ecology pipeline via JSON-RPC: ET₀→water_balance→yield (3 stages)
- [x] Cross-primal discovery: 7 primals in ecosystem
- [x] ToadStool socket detected, provenance IPC graceful fallback
- [x] GPU precision routing: `Df64Only`, `Hybrid` Fp64 strategy

**Binary**: `validate_nucleus_modern`

**Key Result**: biomeOS NUCLEUS integration fully operational. airSpring primal
serves 35 JSON-RPC capabilities with live Tower/Node Atomic. New v0.7.5
endpoints (SPI, autocorrelation, gamma_cdf) all pass parity with direct Rust calls.

### Exp 084: CPU vs GPU Comprehensive Parity

**Goal**: Exhaustive validation of numerical parity between CPU and GPU
implementations across all 18 barraCuda ecological science modules.

**Phase 1 (Rust — 21/21 PASS):**
- [x] FAO-56 Penman-Monteith ET₀: CPU vs GPU (tolerance 2.0 mm/day, schema mismatch path)
- [x] Hargreaves-Samani: CPU vs GPU (tolerance 0.05)
- [x] SCS Curve Number runoff: CPU vs GPU
- [x] Yield response (Ky model): CPU vs GPU
- [x] Simple ET₀ — Makkink: CPU vs GPU
- [x] Simple ET₀ — Turc: CPU vs GPU
- [x] Simple ET₀ — Hamon: CPU vs GPU (tolerance 2.0, daylight formula divergence)
- [x] Simple ET₀ — Blaney-Criddle: CPU vs GPU
- [x] Van Genuchten θ(h): CPU vs GPU
- [x] Van Genuchten K(h): CPU vs GPU
- [x] Thornthwaite PET: CPU vs GPU
- [x] Growing Degree Days: CPU vs GPU
- [x] Pedotransfer (Saxton-Rawls): CPU vs GPU
- [x] Infiltration (Green-Ampt): CPU vs GPU
- [x] Autocorrelation: CPU vs GPU
- [x] Bootstrap CI: CPU vs GPU
- [x] Jackknife CI: CPU vs GPU
- [x] Shannon Diversity: CPU vs GPU
- [x] Fused map-reduce mean: CPU vs GPU
- [x] Fused map-reduce variance: CPU vs GPU
- [x] All modules compile with single binary

**Binary**: `validate_cpu_gpu_comprehensive`

**Key Result**: All 18 GPU modules validated for CPU↔GPU parity. Known
schema-level divergences (FAO-56 vapor pressure path, Hamon daylight formula)
documented with appropriate tolerances. Pure Rust math confirmed consistent
across execution substrates.

### Exp 085: toadStool Compute Dispatch

**Goal**: Validate airSpring in-process science dispatch via JSON-RPC and
toadStool compute offload flow. Tests 14 exposed science methods, cross-primal
discovery, precision routing, and graceful degradation when toadStool is absent.

**Phase 1 (Rust — 19/19 PASS):**
- [x] In-process dispatch: 14 JSON-RPC science methods validated
- [x] ecology.et0_fao56, ecology.water_balance, ecology.yield_response
- [x] science.thornthwaite, science.gdd, science.pedotransfer
- [x] science.spi_drought_index, science.autocorrelation, science.gamma_cdf
- [x] ecology.runoff_scs_cn, ecology.van_genuchten_theta, ecology.van_genuchten_k
- [x] science.bootstrap_ci, science.jackknife_ci
- [x] compute.offload structure validated (toadStool socket detection)
- [x] Cross-primal discovery: 7 primals found (airSpring, barraCuda, toadStool, wetSpring, hotSpring, neuralSpring, groundSpring)
- [x] PrecisionRoutingAdvice: Df64Only/Hybrid routing from DevicePrecisionReport
- [x] Graceful degradation: toadStool absent/stale handled without failure

**Binary**: `validate_toadstool_dispatch`

**Key Result**: airSpring science layer fully accessible via JSON-RPC. All 14
exposed methods return valid results. toadStool compute offload architecture
validated (socket detection, health check, provenance IPC). Graceful fallback
when Node Atomic not running.

### Exp 086: metalForge Mixed Hardware — Live NUCLEUS Mesh

**Goal**: Live hardware probe and NUCLEUS mesh pipeline validation. Tests GPU/CPU/NPU
substrate discovery, NUCLEUS atomic construction (Tower+Node+Nest), capability-based
workload routing, and ecology pipeline dispatch through live hardware.

**Phase 1 (Rust — 17/17 PASS):**
- [x] Live GPU probe: RTX 4070 (F64Native) + Titan V (Df64Only) detected
- [x] Live CPU probe: i9-12900K, 24 cores, x86_64
- [x] NPU probe: graceful absent (no NPU hardware)
- [x] NUCLEUS Mesh construction: Tower (GPU+CPU), Node (Titan V), Nest (CPU)
- [x] Workload routing: 23/27 routed (4 NPU-only unroutable — graceful)
- [x] Ecology pipeline: et0_batch → water_balance_batch → yield_response_surface (3 GPU stages)
- [x] PCIe bypass check: same-node GPU→GPU transfer path confirmed
- [x] CPU roundtrip validation: cross-node transfer matrix computed
- [x] Transfer matrix symmetric and ≥ 0
- [x] CpuCompute capability dispatches to GPU (capability superset)
- [x] Mesh pipeline cross-node hops reported

**Binary**: `validate_mixed_nucleus_live` (metalForge/forge)

**Key Result**: Live hardware inventory feeds NUCLEUS mesh construction. All three
atomic types (Tower, Node, Nest) instantiated from probed substrates. Ecology
pipeline routes through GPU stages with PCIe bypass. Mixed hardware validated for
production dispatch.

### Exp 087: NUCLEUS Graph Coordination via biomeOS

**Goal**: Structural validation of biomeOS deployment graphs (TOML). Parses
graph definitions, validates DAG properties, capability references, dependency
ordering, prerequisite checks, and alignment with discovered NUCLEUS atomics.

**Phase 1 (Rust — 22/22 PASS):**
- [x] airspring_eco_pipeline.toml: parsed, 7 nodes, valid DAG
- [x] cross_primal_soil_microbiome.toml: parsed, 5 nodes, valid DAG
- [x] Graph section metadata present (ID, description)
- [x] DAG acyclicity via topological sort (custom Kahn's algorithm)
- [x] Dependency ordering: fetch_weather before compute_et0 before water_balance
- [x] Prerequisite nodes: check_nestgate and check_toadstool validated
- [x] Capability references match known ecology.* / science.* set
- [x] Cross-primal soil graph: airSpring + wetSpring capabilities
- [x] biomeOS primal discovery: ≥5 primals found
- [x] Tower Atomic detection: live
- [x] Node Atomic detection: live
- [x] Both graphs structurally sound for deployment

**Binary**: `validate_nucleus_graphs`

**Key Result**: biomeOS deployment graphs are well-formed DAGs with correct
capability references and dependency ordering. Topological sort confirms no
cycles. Both ecology and cross-primal pipelines ready for live NUCLEUS dispatch.

---

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
ToadStool's shader evolution (845 WGSL shaders, 46+ cross-spring absorptions):

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
