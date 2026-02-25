# airSpring — BarraCuda Requirements

**Last Updated**: February 25, 2026 (v0.3.10 — dual Kc batch added, cover crops, CPU benchmarks)
**Purpose**: GPU kernel requirements, evolution status, and compute pipeline planning
**ToadStool HEAD**: `02207c4a` (S62+, 608 WGSL shaders)

---

## Current Kernel Usage

### Phase 1: Validated in Rust CPU

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
| Real data (capability) | `eco::*` + `io::csv_ts` | 23/23 | Dynamic station discovery |
| Cross-validation harness | `validation` | 65/65 | Python↔Rust match (tol=1e-5) |

### Phase 2: GPU Orchestrators Wired

| Orchestrator | BarraCuda Primitive | Status | Provenance |
|-------------|--------------------|----|---|
| `gpu::et0::BatchedEt0` | `ops::batched_elementwise_f64` (op=0) | **GPU-FIRST** | hotSpring pow_f64 fix |
| `gpu::water_balance::BatchedWaterBalance` | `ops::batched_elementwise_f64` (op=1) | **GPU-STEP** | Multi-spring |
| `gpu::kriging::KrigingInterpolator` | `ops::kriging_f64::KrigingF64` | **Integrated** | wetSpring |
| `gpu::reduce::SeasonalReducer` | `ops::fused_map_reduce_f64` | **GPU N≥1024** | wetSpring, TS-004 fix |
| `gpu::stream::StreamSmoother` | `ops::moving_window_stats` | **Wired** | wetSpring S28+ |
| `eco::correction::fit_ridge` | `linalg::ridge::ridge_regression` | **Wired** | wetSpring ESN |
| `gpu::dual_kc::BatchedDualKc` | CPU path (Tier B → GPU pending) | **CPU-STEP** | airSpring v0.3.10 |

### Phase 2: Stats & Validation

| Primitive | Integration | Status |
|-----------|-------------|--------|
| `stats::pearson_correlation` | `testutil::r_squared` | Working |
| `stats::spearman_correlation` | `testutil::spearman_r` | Working |
| `stats::bootstrap_ci` | `testutil::bootstrap_rmse` | Working |
| `stats::std_dev` | Integration tests | Working |
| `validation::ValidationHarness` | 6 validation binaries | Absorbed (S59) |

---

## Compute Pipeline: CPU → GPU → metalForge

### Layer 1: BarraCuda CPU (validated, complete)

All algorithms implemented in pure Rust. 279 tests, 287 validation checks.
This is the baseline for correctness — GPU and metalForge results must match.
CPU benchmarks: 12.7M ET₀/s, 59M dual Kc/s, 64M mulched Kc/s.

```
eco::evapotranspiration → validated daily_et0()
eco::soil_moisture      → validated topp_equation(), inverse_topp()
eco::water_balance      → validated simulate_season()
eco::dual_kc            → validated simulate_dual_kc(), cover crops, mulched_ke()
eco::correction         → validated fit_linear/quadratic/exponential/logarithmic/ridge()
eco::sensor_calibration → validated soilwatch10_calibrate()
io::csv_ts              → validated parse(), TimeseriesData
```

### Layer 2: BarraCuda GPU (wired, 6 orchestrators)

GPU dispatch for batch operations. CPU fallback available for all.

```
gpu::et0        → BatchedEt0::gpu()        → fao56_et0_batch()        [op=0]
gpu::water_balance → BatchedWaterBalance::gpu_step() → water_balance_batch() [op=1]
gpu::kriging    → KrigingInterpolator::new() → KrigingF64              [spatial]
gpu::reduce     → SeasonalReducer::new()    → FusedMapReduceF64       [N≥1024]
gpu::stream     → StreamSmoother::new()     → MovingWindowStats        [sliding window]
eco::correction → fit_ridge()               → ridge_regression         [CPU linalg]
```

### Layer 3: metalForge Mixed Hardware (staged, 40 tests)

Upstream absorption candidates for `barracuda::stats`:

```
forge::metrics    → rmse, mbe, nash_sutcliffe, index_of_agreement    [→ barracuda::stats::metrics]
forge::regression → fit_linear, fit_quadratic, fit_exponential,       [→ barracuda::stats::regression]
                    fit_logarithmic, fit_all
```

Future metalForge extensions:
- GPU batch metrics (RMSE/R²/IA over N scenario arrays)
- Mixed CPU+GPU pipeline (CPU for control flow, GPU for batch math)
- NPU streaming (future: real-time IoT on neural accelerator)

---

## Remaining Gaps

### Tier B — Ready to Wire (upstream primitive exists)

| Need | Primitive | Purpose | Effort |
|------|----------|---------|:------:|
| 1D Richards equation | `pde::richards::solve_richards` | Unsaturated soil water flow | Medium — **PROMOTED from Tier C** |
| Sensor batch calibration | `batched_elementwise_f64` (op=5) | Batch SoilWatch 10 VWC | Low |
| Hargreaves ET₀ batch | `batched_elementwise_f64` (op=6) | Simpler ET₀ | Low |
| Kc climate adjustment | `batched_elementwise_f64` (op=7) | FAO-56 Eq. 62 | Low |
| Dual Kc batch (Ke) | `batched_elementwise_f64` (op=8) | GPU Ke for M-field batching | Low — orchestrator wired |
| Nonlinear curve fitting | `optimize::nelder_mead`, `NelderMeadGpu` | Correction equations | Medium |
| Tridiagonal solve | `linalg::tridiagonal_solve_f64` | Implicit PDE steps | Low |
| Adaptive ODE (RK45) | `numerical::rk45_solve` | Dynamic soil models | Low |
| m/z tolerance search | `batched_bisection_f64.wgsl` | Cross-spring from wetSpring | Low |

### Tier C — Needs New Primitive

| Need | Description | Complexity | Upstream Support |
|------|-------------|:---------:|------------------|
| HTTP/JSON client | Open-Meteo, NOAA CDO APIs | Low | Not GPU |

**Note (v0.3.10):** Richards equation promoted from Tier C to Tier B. ToadStool now
provides `pde::richards::solve_richards` with van Genuchten-Mualem constitutive
relations, Picard iteration, Crank-Nicolson time-stepping, and Thomas (tridiagonal)
spatial solver. airSpring needs to wire this with domain-specific soil parameters
from `eco::soil_moisture` and validate against HYDRUS benchmarks.

Dual Kc batch (op=8) added. The GPU orchestrator `gpu::dual_kc::BatchedDualKc` is
wired with CPU fallback and M-field batching. Pending: ToadStool shader for GPU Ke
computation. Cover crop species (5) and no-till mulch reduction are CPU-validated.

---

## Benchmark Results (CPU baselines, `--release`)

Run `cargo run --release --bin bench_airspring_gpu` for current numbers.

| Operation | N | Time (µs) | Throughput | Shader |
|-----------|---|-----------|------------|--------|
| ET₀ (FAO-56) | 10,000 | 795 | 12.6M ops/sec | `batched_elementwise_f64.wgsl` |
| Reduce (seasonal) | 100,000 | 251 | 399M elem/sec | `fused_map_reduce_f64.wgsl` |
| Stream smooth | 8,760 (24h) | 270 | 32.4M elem/sec | `moving_window.wgsl` |
| Kriging | 20→500 | 26 | — | `kriging_f64.wgsl` |
| Ridge regression | 5,000 | 48 | R²=1.000 | (CPU-only) |

---

## ToadStool Issues — All RESOLVED

| ID | Summary | Status |
|----|---------|:------:|
| TS-001 | `pow_f64` returns 0.0 for fractional exponents | **RESOLVED** (`0c477306`) |
| TS-002 | No Rust orchestrator for `batched_elementwise_f64` | **RESOLVED** |
| TS-003 | `acos`/`sin` precision drift in f64 WGSL shaders | **RESOLVED** |
| TS-004 | `FusedMapReduceF64` buffer conflict for N≥1024 | **RESOLVED** |

See `barracuda/src/gpu/evolution_gaps.rs` for the full 18-gap roadmap.
