# airSpring — BarraCuda Requirements

**Last Updated**: March 1, 2026 (v0.5.6 — 636 lib tests, 60 barracuda + 5 forge binaries, 15 Tier A + 3 pipeline GPU orchestrators + pure GPU pipeline (78/78) + mixed-hardware pipeline (104/104 metalForge) + NUCLEUS atomics + biomeOS graph execution, 21.0× CPU speedup across 18 algorithms)
**Purpose**: GPU kernel requirements, evolution status, and compute pipeline planning
**ToadStool HEAD**: `1dd7e338` (S70+++ — ops 5-8 absorbed, seasonal pipeline WGSL, brent GPU shader, cross-spring absorption)

---

## Current Kernel Usage

### Phase 1: Validated in Rust CPU + NPU (45 experiments)

| Kernel / Module | Rust Crate | Checks | Validation |
|----------------|------------|:------:|------------|
| ET₀ Penman-Monteith | `eco::evapotranspiration` | 31/31 | FAO-56 tables, 3 reference cities |
| ET₀ Hargreaves | `eco::evapotranspiration` | — | Temperature-only fallback |
| Soil calibration (Topp eq) | `eco::soil_moisture` | 26/26 | 7 USDA textures, inverse round-trip |
| Correction curve fitting | `eco::correction` | — | Linear, quadratic, exponential, logarithmic, ridge |
| IoT pipeline | `io::csv_ts` | 11/11 | CSV time series streaming parser |
| Water balance | `eco::water_balance` | 13/13 | Mass balance exact (< 1e-10 mm) |
| Sensor calibration | `eco::sensor_calibration` | 21/21 | SoilWatch 10 VWC + irrigation |
| Dual Kc (Kcb+Ke) | `eco::dual_kc` | 61/61 | FAO-56 Ch 7, Tables 17/19, all crop groups |
| Cover crops + mulch | `eco::dual_kc` (cover_crop, mulched_ke) | 40/40 | 5 species, Islam et al., no-till |
| Regional ET₀ | `eco::evapotranspiration` | 61/61 | 6 Michigan stations, Pearson r, CV |
| Richards equation | `eco::richards` | 15/15 | VG retention/K, infiltration, drainage, mass balance |
| Biochar isotherms | `eco::isotherm` | 14/14 | Langmuir/Freundlich R², RL, residuals |
| 60-year water balance | `eco::water_balance` + Hargreaves | 11/11 | Decadal stability, climate trends |
| Real data (capability) | `eco::*` + `io::csv_ts` | 23/23 | Dynamic station discovery |
| Priestley-Taylor ET₀ | `eco::evapotranspiration` | 32/32 | α=1.26, Rn-only method |
| ET₀ 3-method intercomp | `eco::evapotranspiration` | 36/36 | PM vs PT vs HG, 6 MI stations |
| Thornthwaite monthly ET₀ | `eco::evapotranspiration` | 50/50 | Heat index, 2-station monthly |
| Growing Degree Days | `eco::crop` | 26/26 | GDD avg/clamp, Kc from GDD |
| Saxton-Rawls pedotransfer | `eco::soil_moisture` | 58/58 | θ_wp/θ_fc/θ_s/Ksat, 8 textures |
| NASS yield validation | `eco::yield_response` + `eco::water_balance` | 40/40 | Stewart pipeline, 5 crops, MI synthetic |
| Forecast scheduling | `eco::water_balance` + `eco::yield_response` | 19/19 | Forecast vs perfect knowledge |
| SCAN soil moisture | `eco::richards` + `eco::van_genuchten` | 34/34 | VG θ/K, Ks ordering, SCAN ranges |
| Multi-crop water budget | `eco::water_balance` + `eco::dual_kc` + `eco::yield_response` | 47/47 | 5 crops, irrigated/rainfed/dual Kc |
| **NPU edge inference** | **`npu` (feature-gated `akida-driver`)** | **35/35** | **AKD1000 live: crop stress, irrigation, anomaly** |
| **Funky NPU IoT** | **`validate_npu_funky_eco`** | **32/32** | **Streaming, seasonal evolution, multi-crop crosstalk, LOCOMOS power, noise** |
| **High-Cadence NPU** | **`validate_npu_high_cadence`** | **28/28** | **1-min cadence, burst mode, multi-sensor fusion, ensemble, weight hot-swap** |
| **metalForge dispatch** | **`airspring-forge` crate** | **29/29** | **CPU/GPU/NPU routing, 18 eco workloads, cross-system** |
| **Anderson coupling** | **`eco::anderson`** | **55+95** | **θ→S_e→d_eff→QS regime (cross-spring)** |
| Cross-validation harness | `validation` | 75/75 | Python↔Rust match (tol=1e-5) |

### Phase 2: GPU Orchestrators Wired (15 Tier A + 3 pipeline)

| Orchestrator | BarraCuda Primitive | Status | Provenance |
|-------------|--------------------|----|---|
| `gpu::et0::BatchedEt0` | `ops::batched_elementwise_f64` (op=0) | **GPU-FIRST** | hotSpring pow_f64 fix |
| `gpu::water_balance::BatchedWaterBalance` | `ops::batched_elementwise_f64` (op=1) | **GPU-STEP** | Multi-spring |
| `gpu::kriging::KrigingInterpolator` | `ops::kriging_f64::KrigingF64` | **Integrated** | wetSpring |
| `gpu::reduce::SeasonalReducer` | `ops::fused_map_reduce_f64` | **GPU N≥1024** | wetSpring, TS-004 fix |
| `gpu::stream::StreamSmoother` | `ops::moving_window_stats` | **Wired** | wetSpring S28+ |
| `eco::correction::fit_ridge` | `linalg::ridge::ridge_regression` | **Wired** | wetSpring ESN |
| `gpu::richards::BatchedRichards` | `pde::richards::solve_richards` | **Wired** (v0.4.0) | airSpring → upstream |
| `gpu::isotherm::fit_*_nm` | `optimize::nelder_mead` | **Wired** (v0.4.0) | airSpring → upstream |
| `gpu::hargreaves::BatchedHargreaves` | `batched_elementwise_f64` (op=6) | **GPU-FIRST** (v0.5.6) | FAO-56 Eq. 52, S70+ absorbed |
| `gpu::kc_climate::BatchedKcClimate` | `batched_elementwise_f64` (op=7) | **GPU-FIRST** (v0.5.6) | FAO-56 Eq. 62, S70+ absorbed |
| `gpu::dual_kc::BatchedDualKc` | `batched_elementwise_f64` (op=8) | **GPU-FIRST** (v0.5.6) | airSpring v0.5.2, S70+ absorbed |
| `gpu::sensor_calibration::BatchedSensorCal` | `batched_elementwise_f64` (op=5) | **GPU-FIRST** (v0.5.6) | Dong et al. 2024, S70+ absorbed |
| `gpu::seasonal_pipeline::SeasonalPipeline` | Chains ops 0→7→1→yield | **GPU Stages 1-2** (v0.5.6) | ET₀ + Kc GPU dispatch, CPU stages 3-4 |
| `gpu::atlas_stream::AtlasStream` | `UnidirectionalPipeline` (pending) | **GPU+streaming** (v0.5.4) | GPU-capable + callback pattern |
| `gpu::mc_et0::mc_et0_gpu` | `mc_et0_propagate_f64.wgsl` (pending) | **Wired** (v0.5.2, Tier B) | groundSpring xoshiro |

### Phase 2: Stats & Validation

| Primitive | Integration | Status |
|-----------|-------------|--------|
| `stats::pearson_correlation` | `testutil::r_squared` | Working |
| `stats::spearman_correlation` | `testutil::spearman_r` | Working |
| `stats::bootstrap_ci` | `testutil::bootstrap_rmse` | Working |
| `stats::std_dev` | Integration tests | Working |
| `validation::ValidationHarness` | 22 binaries | Absorbed (S59) |

---

## Compute Pipeline: CPU → GPU → metalForge

### Layer 1: BarraCuda CPU (validated, complete)

All algorithms implemented in pure Rust. 628 lib tests, 50+ binaries, 651+ total checks.
This is the baseline for correctness — GPU and metalForge results must match.
CPU benchmarks: 21.0× geometric mean speedup vs Python (18/18 parity across
ET₀, dual Kc, mulched Kc, VG θ, Richards 1D, Langmuir, Freundlich, GDD,
SCS-CN runoff, Green-Ampt, Saxton-Rawls, Priestley-Taylor, yield response,
dual Kc step, Makkink, Blaney-Criddle, Hargreaves, sensor calibration).

```
eco::evapotranspiration → validated daily_et0(), hargreaves_et0()
eco::soil_moisture      → validated topp_equation(), inverse_topp()
eco::water_balance      → validated simulate_season()
eco::dual_kc            → validated simulate_dual_kc(), cover crops, mulched_ke()
eco::correction         → validated fit_linear/quadratic/exponential/logarithmic/ridge()
eco::sensor_calibration → validated soilwatch10_calibrate()
eco::richards           → validated solve_richards_1d(), van_genuchten theta/K
eco::isotherm           → validated fit_langmuir/freundlich(), predictions
eco::crop               → validated gdd_avg/clamp(), accumulated_gdd(), kc_from_gdd()
io::csv_ts              → validated parse(), TimeseriesData
```

### Layer 2: BarraCuda GPU (wired, 15 orchestrators)

GPU dispatch for batch operations. CPU fallback available for all.

```
gpu::et0                → BatchedEt0::gpu()              → fao56_et0_batch()         [op=0, Tier A]
gpu::water_balance      → BatchedWaterBalance::gpu_step() → water_balance_batch()     [op=1, Tier A]
gpu::kriging            → KrigingInterpolator::new()      → KrigingF64               [spatial, Tier A]
gpu::reduce             → SeasonalReducer::new()          → FusedMapReduceF64        [N≥1024, Tier A]
gpu::stream             → StreamSmoother::new()           → MovingWindowStats         [sliding, Tier A]
gpu::richards           → BatchedRichards::solve()        → pde::richards             [Tier A]
gpu::isotherm           → fit_langmuir_nm/freundlich_nm   → optimize::nelder_mead    [Tier A]
eco::correction         → fit_ridge()                     → ridge_regression          [CPU]
gpu::sensor_calibration → BatchedSensorCal::compute_gpu() → batched_elementwise (op=5) [Tier B]
gpu::hargreaves         → BatchedHargreaves::compute_gpu()→ batched_elementwise (op=6) [Tier B]
gpu::kc_climate         → BatchedKcClimate::compute_gpu() → batched_elementwise (op=7) [Tier B]
gpu::dual_kc            → BatchedDualKc::step_gpu()       → batched_elementwise (op=8) [Tier B]
gpu::seasonal_pipeline  → SeasonalPipeline::gpu()/cpu()   → GPU ET₀ + chained 7→1→yield [GPU Stage 1]
gpu::atlas_stream       → AtlasStream::with_gpu()/new()  → GPU pipeline + streaming    [GPU+streaming]
gpu::mc_et0             → mc_et0_gpu()                    → mc_et0_propagate_f64.wgsl  [Tier B]
```

### Layer 3: metalForge Mixed Hardware (staged, 64 tests)

Upstream absorption candidates:

```
forge::metrics       → rmse, mbe, nse, ia, r2                        [→ barracuda::stats::metrics]
forge::regression    → fit_linear, quadratic, exponential, logarithmic [→ barracuda::stats::regression]
forge::moving_window → moving_window_stats_f64                        [→ barracuda::ops]
forge::hydrology     → hargreaves_et0, crop_kc, soil_water_balance    [→ barracuda::ops::hydrology]
forge::van_genuchten → theta, conductivity, capacity                  [ABSORBED → pde::richards]
forge::isotherm      → langmuir, freundlich, fit, separation_factor   [WIRED → optimize]
```

Mixed hardware extensions (future):
- GPU batch metrics (RMSE/R²/IA over N scenario arrays)
- Mixed CPU+GPU pipeline (CPU for control flow, GPU for batch math)
- NPU streaming (real-time IoT on neural accelerator)

---

## Remaining Gaps

### Tier B — Ready to Wire (5 remaining, 6 resolved in v0.5.2)

| Need | Primitive | Status | Effort |
|------|----------|--------|:------:|
| ~~Dual Kc batch (Ke)~~ | `batched_elementwise_f64` (op=8) | **WIRED** (v0.5.2) | — |
| **VG θ/K batch** | `batched_elementwise_f64` (new op) | eco::richards validated | Low |
| **Batch Nelder-Mead** | `NelderMeadGpu` | CPU NM wired via gpu::isotherm | Medium |
| ~~Sensor batch calibration~~ | `batched_elementwise_f64` (op=5) | **WIRED** (v0.5.2) | — |
| ~~Hargreaves ET₀ batch~~ | `batched_elementwise_f64` (op=6) | **WIRED** (v0.5.2) | — |
| ~~Kc climate adjustment~~ | `batched_elementwise_f64` (op=7) | **WIRED** (v0.5.2) | — |
| Richards PDE (GPU) | WGSL van_genuchten_f64 shader | **Wired** via gpu::richards | — |
| Tridiagonal solve | `linalg::tridiagonal_solve_f64` | Available upstream | Low |
| Adaptive ODE (RK45) | `numerical::rk45_solve` | Available upstream | Low |
| Isotherm batch fitting | `NelderMeadGpu` batch | **Wired** via gpu::isotherm fit_*_nm | — |
| m/z tolerance search | `batched_bisection_f64.wgsl` | Cross-spring from wetSpring | Low |

**New in v0.5.2**: Seasonal pipeline (chained), atlas stream, MC ET₀ GPU path — see Phase 2 table.

### Tier C — Needs New Primitive (1 item)

| Need | Description | Complexity |
|------|-------------|:---------:|
| HTTP/JSON client | Open-Meteo, NOAA CDO APIs | Low (not GPU) |

---

## Benchmark Results (CPU baselines, `--release`)

Run `cargo run --release --bin bench_cpu_vs_python` for current numbers.

| Operation | N | Throughput | Source |
|-----------|---|------------|--------|
| ET₀ (FAO-56) | 1M | 12.2M ops/sec | `bench_cpu_vs_python` |
| Dual Kc (Kcb+Ke) | 3,650 | 59M days/sec | `bench_cpu_vs_python` |
| Mulched Kc | 3,650 | 64M days/sec | `bench_cpu_vs_python` |
| VG θ retention | 100K | 36.5M evals/sec | `bench_cpu_vs_python` (v0.4.0) |
| Richards 1D | 10 steps | 3,618 sims/sec | `bench_cpu_vs_python` (v0.4.0) |
| Langmuir fit | 9 pts | 57M fits/sec | `bench_cpu_vs_python` (v0.4.0) |
| Freundlich fit | 9 pts | 1.2M fits/sec | `bench_cpu_vs_python` (v0.4.0) |
| Reduce (seasonal) | 100K | 399M elem/sec | `bench_airspring_gpu` |
| Stream smooth | 8,760 | 32.4M elem/sec | `bench_airspring_gpu` |
| Kriging | 20→500 | — | `bench_airspring_gpu` |
| Ridge regression | 5,000 | R²=1.000 | `bench_airspring_gpu` (CPU-only) |

---

## ToadStool Issues — All RESOLVED

| ID | Summary | Status |
|----|---------|:------:|
| TS-001 | `pow_f64` returns 0.0 for fractional exponents | **RESOLVED** (S54 — H-011) |
| TS-002 | No Rust orchestrator for `batched_elementwise_f64` | **RESOLVED** (S54 — L-011, already present) |
| TS-003 | `acos`/`sin` precision drift in f64 WGSL shaders | **RESOLVED** (S54 — H-012) |
| TS-004 | `FusedMapReduceF64` buffer conflict for N≥1024 | **RESOLVED** (S54 — H-013) |

See `barracuda/src/gpu/evolution_gaps.rs` for the full 26-gap roadmap (v0.5.2).
