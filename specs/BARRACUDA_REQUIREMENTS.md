# airSpring — BarraCuda Requirements

**Last Updated**: February 25, 2026 (v0.4.0 — Richards PDE wired, isotherm NM wired, 8 GPU orchestrators)
**Purpose**: GPU kernel requirements, evolution status, and compute pipeline planning
**ToadStool HEAD**: `02207c4a` (S62+, 608 WGSL shaders)

---

## Current Kernel Usage

### Phase 1: Validated in Rust CPU (11 experiments)

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
| Cross-validation harness | `validation` | 75/75 | Python↔Rust match (tol=1e-5) |

### Phase 2: GPU Orchestrators Wired (8 modules)

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

Note: `gpu::dual_kc::BatchedDualKc` has CPU orchestrator wired (Tier B → pending shader).

### Phase 2: Stats & Validation

| Primitive | Integration | Status |
|-----------|-------------|--------|
| `stats::pearson_correlation` | `testutil::r_squared` | Working |
| `stats::spearman_correlation` | `testutil::spearman_r` | Working |
| `stats::bootstrap_ci` | `testutil::bootstrap_rmse` | Working |
| `stats::std_dev` | Integration tests | Working |
| `validation::ValidationHarness` | 16 binaries | Absorbed (S59) |

---

## Compute Pipeline: CPU → GPU → metalForge

### Layer 1: BarraCuda CPU (validated, complete)

All algorithms implemented in pure Rust. 371 lib + 97 integration tests, 16 binaries.
This is the baseline for correctness — GPU and metalForge results must match.
CPU benchmarks: 12.7M ET₀/s, 36.5M VG θ/s, 59M dual Kc/s, 57M Langmuir fits/s.

```
eco::evapotranspiration → validated daily_et0(), hargreaves_et0()
eco::soil_moisture      → validated topp_equation(), inverse_topp()
eco::water_balance      → validated simulate_season()
eco::dual_kc            → validated simulate_dual_kc(), cover crops, mulched_ke()
eco::correction         → validated fit_linear/quadratic/exponential/logarithmic/ridge()
eco::sensor_calibration → validated soilwatch10_calibrate()
eco::richards           → validated solve_richards_1d(), van_genuchten theta/K
eco::isotherm           → validated fit_langmuir/freundlich(), predictions
io::csv_ts              → validated parse(), TimeseriesData
```

### Layer 2: BarraCuda GPU (wired, 8 orchestrators)

GPU dispatch for batch operations. CPU fallback available for all.

```
gpu::et0           → BatchedEt0::gpu()             → fao56_et0_batch()         [op=0]
gpu::water_balance → BatchedWaterBalance::gpu_step()→ water_balance_batch()     [op=1]
gpu::kriging       → KrigingInterpolator::new()     → KrigingF64               [spatial]
gpu::reduce        → SeasonalReducer::new()         → FusedMapReduceF64        [N≥1024]
gpu::stream        → StreamSmoother::new()          → MovingWindowStats         [sliding]
gpu::richards      → BatchedRichards::solve_upstream()→ pde::richards           [Tier B]
gpu::isotherm      → fit_langmuir_nm/freundlich_nm  → optimize::nelder_mead    [Tier B]
eco::correction    → fit_ridge()                    → ridge_regression          [CPU]
```

### Layer 3: metalForge Mixed Hardware (staged, 53 tests)

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

### Tier B — Ready to Wire (11 items)

| Need | Primitive | Status | Effort |
|------|----------|--------|:------:|
| **Dual Kc batch (Ke)** | `batched_elementwise_f64` (op=8) | CPU orchestrator wired | Low |
| **VG θ/K batch** | `batched_elementwise_f64` (new op) | eco::richards validated | Low |
| **Batch Nelder-Mead** | `NelderMeadGpu` | CPU NM wired via gpu::isotherm | Medium |
| Sensor batch calibration | `batched_elementwise_f64` (op=5) | — | Low |
| Hargreaves ET₀ batch | `batched_elementwise_f64` (op=6) | — | Low |
| Kc climate adjustment | `batched_elementwise_f64` (op=7) | — | Low |
| Richards PDE (GPU) | WGSL van_genuchten_f64 shader | **Wired** via gpu::richards | — |
| Tridiagonal solve | `linalg::tridiagonal_solve_f64` | Available upstream | Low |
| Adaptive ODE (RK45) | `numerical::rk45_solve` | Available upstream | Low |
| Isotherm batch fitting | `NelderMeadGpu` batch | **Wired** via gpu::isotherm fit_*_nm | — |
| m/z tolerance search | `batched_bisection_f64.wgsl` | Cross-spring from wetSpring | Low |

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
| TS-001 | `pow_f64` returns 0.0 for fractional exponents | **RESOLVED** (`0c477306`) |
| TS-002 | No Rust orchestrator for `batched_elementwise_f64` | **RESOLVED** |
| TS-003 | `acos`/`sin` precision drift in f64 WGSL shaders | **RESOLVED** |
| TS-004 | `FusedMapReduceF64` buffer conflict for N≥1024 | **RESOLVED** |

See `barracuda/src/gpu/evolution_gaps.rs` for the full 20-gap roadmap.
