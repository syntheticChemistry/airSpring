# Handoff: airSpring → Toadstool/BarraCuda

**Date:** February 16, 2026 (updated February 25, 2026 — v0.3.7, synced to ToadStool `02207c4a`)
**From:** airSpring (Precision Agriculture validation study)
**To:** Toadstool/BarraCuda core team
**License:** AGPL-3.0-or-later

---

## STATED GOAL: Open Precision Irrigation on Consumer Hardware

Phase 0 and 0+ confirmed: **published agricultural science reproduces with
open tools and open data.** 142/142 Python checks pass against paper
benchmarks. 918 station-days of real Michigan weather produce ET₀ with
R²=0.967. Water balance simulations show 53-72% water savings with smart
scheduling — consistent with published results.

Phase 1 confirmed: **Rust BarraCuda validates the core pipeline.** 123/123
validation checks pass across 8 binaries (ET₀, soil moisture, water balance,
IoT parsing, sensor calibration, real data 4-crop scenarios, season simulation).
253 tests (175 unit + 76 integration + 2 doc-tests) cover cross-module integration,
determinism, error paths, sensor calibration, correction curve fitting,
statistical validation, crop Kc database, barracuda primitive cross-validation,
GPU orchestrators (BatchedEt0, BatchedWaterBalance, kriging, reduce),
GPU-matches-CPU validation, GPU determinism (bit-identical reruns),
evolution gap catalog, and deepened barracuda stats.

Phase 2 confirmed: **Python and Rust produce identical results.** 65/65 values
match within 1e-5 tolerance across atmospheric, solar, radiation, ET₀, Topp,
SoilWatch 10, irrigation, statistical, sunshine Rs, Hargreaves ET₀, monthly
soil heat flux, low-level PM Eq. 6, standalone water balance step, and
correction model (linear, quadratic, exponential, logarithmic) computations.

| Phase | Status | Key Metric |
|-------|--------|------------|
| Phase 0: Paper baselines (Python) | **142/142 PASS** | FAO-56, soil, IoT, water balance |
| Phase 0+: Real data pipeline | **918 station-days** | ET₀ R²=0.967 vs Open-Meteo, 3 API sources |
| Phase 1: Rust validation | **123/123 PASS** | 8 binaries, 293 tests, 0 clippy warnings |
| Phase 2: Cross-validation | **65/65 MATCH** | Python vs Rust identical outputs (tol=1e-5) |
| Phase 3: GPU bridge | **GPU-FIRST** | 6 orchestrators, 4/4 ToadStool issues **RESOLVED** |

---

## What airSpring Brings to BarraCuda

### New Domain: Time Series + Spatial + IoT

hotSpring proved BarraCuda can do clean matrix math (eigensolve, PDE,
optimization). airSpring adds a fundamentally different workload pattern:

| Dimension | hotSpring | airSpring |
|-----------|-----------|-----------|
| Data shape | Dense matrices (12×12 to 50×50) | Long time series (153+ days × N stations) |
| Input rate | Static (AME2020 table) | Streaming (IoT sensors, API feeds) |
| Spatial | Per-nucleus | Per-field-cell (kriging grid) |
| Time coupling | SCF iteration | Daily water balance (sequential) |
| Parallelism | Across nuclei | Across stations, fields, and grid cells |
| I/O pattern | Preload once | Continuous API ingestion |

### BarraCuda Primitives Used

| Primitive | airSpring Integration | Status |
|-----------|----------------------|:------:|
| `barracuda::stats::pearson_correlation` | R² cross-validation in integration tests | **Working** |
| `barracuda::stats::correlation::std_dev` | Statistical cross-validation (sample vs population) | **Working** |
| `barracuda::stats::correlation::spearman_correlation` | Nonparametric rank validation | **Working** |
| `barracuda::stats::correlation::variance` | Sample variance computation | **Working** |
| `barracuda::stats::bootstrap::bootstrap_ci` | RMSE uncertainty quantification | **Working** |
| `barracuda::validation::ValidationHarness` | Structured pass/fail checks (absorbed upstream S59) | **Leaning** |
| `barracuda::ops::kriging_f64::KrigingF64` | Soil moisture spatial interpolation (ordinary kriging) | **Integrated** |
| `barracuda::ops::fused_map_reduce_f64::FusedMapReduceF64` | Seasonal statistics (GPU for N≥1024) | **Integrated** |
| `barracuda::ops::moving_window_stats::MovingWindowStats` | IoT stream smoothing (wetSpring S28+) | **Wired** |
| `barracuda::linalg::ridge::ridge_regression` | Sensor calibration regression (wetSpring ESN) | **Wired** |
| `barracuda::device::WgpuDevice` | f64-capable GPU device for ops dispatch | **Working** |
| `serde` serialization | Benchmark JSON loading (compile-time `include_str!()`) | **Working** |
| f64 arithmetic + `mul_add` | All FAO-56 functions (FMA precision) | **Working** |
| Zero unsafe | `#![deny(unsafe_code)]` pattern inherited from barracuda | **Working** |

### ToadStool Primitives Integrated (Phase 3 — GPU-First)

All `ToadStool` issues resolved. GPU-first architecture with CPU fallback.

| Primitive | ToadStool Source | airSpring Status |
|-----------|-----------------|:----------------:|
| `BatchedElementwiseF64` (op=0) | `barracuda::ops::batched_elementwise_f64` | **GPU-FIRST** (`BatchedEt0::gpu()` → `fao56_et0_batch()`) |
| `BatchedElementwiseF64` (op=1) | `barracuda::ops::batched_elementwise_f64` | **GPU-STEP** (`BatchedWaterBalance::gpu_step()` → `water_balance_batch()`) |
| `KrigingF64` | `barracuda::ops::kriging_f64` | **Integrated** (`KrigingInterpolator`, proper ordinary kriging via LU) |
| `FusedMapReduceF64` | `barracuda::ops::fused_map_reduce_f64` | **GPU N≥1024** (`SeasonalReducer`, TS-004 RESOLVED) |
| `MovingWindowStats` | `barracuda::ops::moving_window_stats` | **Wired** (`StreamSmoother`, wetSpring S28+) |
| `ridge_regression` | `barracuda::linalg::ridge` | **Wired** (`fit_ridge`, wetSpring ESN) |
| `stats::bootstrap_ci` | Uncertainty quantification | **Working** |
| `stats::spearman_correlation` | Nonparametric validation | **Working** |

See `barracuda/src/gpu/evolution_gaps.rs` for the complete 15-gap roadmap
(8 Tier A, 5 Tier B, 2 Tier C). Cross-spring provenance documented in
`specs/CROSS_SPRING_EVOLUTION.md`.

### ToadStool Issues — ALL RESOLVED (commit `0c477306`)

| ID | Summary | Severity | Status |
|----|---------|:--------:|:------:|
| TS-001 | `pow_f64` returns 0.0 for fractional exponents | **Critical** | **RESOLVED** |
| TS-002 | No Rust orchestrator for `batched_elementwise_f64` | **Medium** | **RESOLVED** |
| TS-003 | `acos`/`sin` precision drift in f64 WGSL shaders | **Low** | **RESOLVED** |
| TS-004 | `FusedMapReduceF64` GPU dispatch buffer conflict | **High** | **RESOLVED** |

All GPU paths now use the resolved ToadStool primitives directly. CPU fallbacks
remain available but are no longer the default path.

### Remaining Primitives airSpring Needs

**Tier B** (upstream primitive exists, needs domain wiring):

| Primitive | Upstream Source | Purpose |
|-----------|---------------|---------|
| Sensor calibration batch | `batched_elementwise_f64.wgsl` (op=5) | Batch SoilWatch 10 calibration |
| Hargreaves ET₀ batch | `batched_elementwise_f64.wgsl` (op=6) | Simpler ET₀ (no humidity/wind) |
| Kc climate adjustment batch | `batched_elementwise_f64.wgsl` (op=7) | FAO-56 Eq. 62 |
| Nonlinear curve fitting | `optimize::nelder_mead`, `NelderMeadGpu` | Batch correction eq fitting |
| m/z tolerance search | `batched_bisection_f64.wgsl` | Cross-spring from wetSpring |

**Tier C** (needs new primitive):

| Primitive | Purpose | Complexity |
|-----------|---------|:----------:|
| 1D Richards solver | Unsaturated flow PDE | High — `ops::crank_nicolson` + tridiagonal |
| API client (HTTP + JSON) | Open-Meteo, NOAA CDO | Low — not GPU |

---

## Rust Crate Architecture (v0.3.7)

### Module Map

```
airspring-barracuda/
├── src/
│   ├── lib.rs              # pub mod eco, error, gpu, io, validation, testutil
│   ├── error.rs            # AirSpringError enum (Io, CsvParse, JsonParse, InvalidInput, Barracuda)
│   ├── gpu/
│   │   ├── mod.rs                # ToadStool/BarraCuda GPU bridge — GPU-FIRST architecture
│   │   ├── et0.rs                # BatchedEt0 GPU-first via BatchedElementwiseF64::fao56_et0_batch()
│   │   ├── water_balance.rs      # BatchedWaterBalance GPU-step via water_balance_batch()
│   │   ├── kriging.rs            # KrigingInterpolator + IDW (↔ barracuda::ops::kriging_f64)
│   │   ├── reduce.rs             # SeasonalReducer GPU N≥1024 (↔ barracuda::ops::fused_map_reduce_f64)
│   │   ├── stream.rs             # StreamSmoother (↔ barracuda::ops::moving_window_stats, wetSpring S28+)
│   │   └── evolution_gaps.rs     # 15 gaps (8A+5B+2C), 4/4 ToadStool issues RESOLVED
│   ├── eco/
│   │   ├── mod.rs
│   │   ├── correction.rs         # Sensor correction curve fitting (linear/quad/exp/log/ridge)
│   │   ├── crop.rs               # CropType enum, FAO-56 Table 12 Kc database, Eq. 62 adjustment
│   │   ├── evapotranspiration.rs  # 23 FAO-56 functions + Hargreaves ET₀, low-level PM Eq. 6
│   │   ├── sensor_calibration.rs  # SoilWatch 10 VWC, irrigation recommendation
│   │   ├── soil_moisture.rs       # Topp eq, inverse, SoilTexture, SoilHydraulicProps
│   │   └── water_balance.rs       # WaterBalanceState, RunoffModel, standalone fns, simulate_season
│   ├── io/
│   │   ├── mod.rs
│   │   └── csv_ts.rs       # TimeseriesData (columnar), streaming BufReader parser
│   ├── validation.rs        # Re-exports barracuda::validation::ValidationHarness + JSON utils
│   ├── testutil.rs          # R², RMSE, MBE, IA, NSE, Spearman, bootstrap CI, variance
│   └── bin/
│       ├── validate_et0.rs          # 31 checks (loads benchmark_fao56.json)
│       ├── validate_soil.rs         # 26 checks
│       ├── validate_iot.rs          # 11 checks (round-trip streaming)
│       ├── validate_water_balance.rs  # 13 checks (mass balance, Michigan season)
│       ├── validate_sensor_calibration.rs  # 21 checks (SoilWatch 10, irrigation)
│       ├── validate_real_data.rs    # 15 checks (4 crops × rainfed+irrigated, real Open-Meteo)
│       ├── cross_validate.rs        # Phase 2 JSON output (65 values) for Python↔Rust diff
│       ├── simulate_season.rs      # Full pipeline: crop Kc → soil → ET₀ → water balance → scheduling
│       └── bench_airspring_gpu.rs  # Benchmark all GPU ops with cross-spring provenance
├── tests/
│   ├── eco_integration.rs     # 27 tests (soil, ET₀, crop, water balance, correction)
│   ├── gpu_integration.rs     # 25 tests (orchestrators, evolution gaps, ToadStool, GPU determinism)
│   ├── io_and_errors.rs       # 8 tests (CSV streaming, error types, round-trip)
│   └── stats_integration.rs   # 16 tests (barracuda stats x-val, bootstrap, Spearman)
└── Cargo.toml
```

### Dependencies

```toml
[dependencies]
barracuda = { path = "../../phase1/toadstool/crates/barracuda" }
serde = { version = "1", features = ["derive"] }
serde_json = "1"

[dev-dependencies]
pollster = "0.3"  # Async WgpuDevice creation in GPU integration tests
```

### Quality Gates

| Check | Status |
|-------|:------:|
| `cargo fmt -- --check` | **PASS** |
| `cargo clippy --pedantic --nursery` | **0 warnings** |
| `cargo doc --no-deps` | **0 warnings** |
| `cargo test` | **253/253 PASS** (175 unit + 76 integration + 2 doc-tests) |
| Validation binaries | **123/123 PASS** (7 binaries + 1 JSON output) |
| Cross-validation | **65/65 MATCH** (Python↔Rust, tol=1e-5) |
| Unsafe code | **0** |
| `AirSpringError` | **Proper error type** (replaces `String`) |
| Max LOC per file | **695** (under 1000 limit) |

### Issues Resolved (since v0.1.0)

1. ~~**Typo**: `SoilTexture::SandyCite`~~ → Fixed to `SandyClay`. Regression test added.
2. ~~**Runoff model differs**~~ → Now capability-based `RunoffModel` enum. `None` (FAO-56 default) aligns with Python.
3. ~~**Phantom modules**~~ → `eco::isotherms` and `eco::richards` references removed from lib.rs.
4. ~~**189 clippy warnings**~~ → All resolved. Zero pedantic/nursery warnings.
5. ~~**No benchmark JSON loading**~~ → `validate_et0` now loads `benchmark_fao56.json` at compile time.
6. ~~**Duplicated check() function**~~ → Shared `ValidationRunner` → now **leaning on upstream** `barracuda::validation::ValidationHarness`.
7. ~~**CSV buffered entire file**~~ → Streaming `BufReader` + columnar storage.
8. ~~**Mock data in production code**~~ → Moved to `testutil` module.
9. ~~**barracuda dep unused**~~ → Now uses `barracuda::stats` for cross-validation.

---

## metalForge — Absorption-Ready Extensions (v0.2.0)

Following hotSpring's Write → Validate → Handoff → Absorb → Lean pattern,
airSpring now stages domain primitives in `metalForge/forge/` for upstream
absorption.  Pure Rust, zero dependencies, 40/40 tests pass.

| Module | Functions | Upstream Target | Tests |
|--------|-----------|----------------|:-----:|
| `metrics` | `rmse`, `mbe`, `nash_sutcliffe`, `index_of_agreement`, `coefficient_of_determination` | `barracuda::stats::metrics` | 9 |
| `regression` | `fit_linear`, `fit_quadratic`, `fit_exponential`, `fit_logarithmic`, `fit_all` + `FitResult::predict()` | `barracuda::stats::regression` | 11 |
| `moving_window_f64` | `moving_window_stats` (CPU f64 complement to GPU f32) | `barracuda::ops::moving_window_stats_f64` | 7 |
| `hydrology` | `hargreaves_et0`, `hargreaves_et0_batch`, `crop_coefficient`, `soil_water_balance` | `barracuda::ops::hydrology` | 13 |

**Design**: `FitResult` includes `predict()` and `predict_one()` following the
`RidgeResult::predict()` pattern from `barracuda::linalg::ridge`.  Metrics follow
the `stats::correlation` pattern — standalone functions on `&[f64]`.

**Absorption manifest**: See `metalForge/ABSORPTION_MANIFEST.md` for full
signatures, validation provenance, and post-absorption rewiring plan.

---

## Primal Self-Knowledge

airSpring follows the wateringHole principle of **primal self-knowledge**:
- Zero compile-time coupling to hotSpring or wetSpring
- The only path dependency is `barracuda` (shared compute crate)
- Cross-primal discovery happens at runtime via `wateringHole` protocols (JSON-RPC 2.0)
- Domain-specific knowledge (FAO-56, Topp, water balance) stays in airSpring

---

## GPU Acceleration Mapping (Phase 3)

### Rust Module → WGSL Shader → Pipeline Stage

| Rust Module | GPU Tier | WGSL Shader | Pipeline Stage | Blocking |
|------------|:--------:|-------------|----------------|----------|
| `eco::evapotranspiration::daily_et0` | A (rewire) | `batched_et0.wgsl` | Single dispatch, N station-days | None |
| `eco::water_balance::simulate_season` | A (rewire) | `batched_water_balance.wgsl` | Per-field parallel, sequential days | None |
| `eco::soil_moisture::topp_equation` | A (rewire) | `sensor_calibration.wgsl` | Per-sensor parallel | None |
| `eco::sensor_calibration::soilwatch10_vwc` | A (rewire) | `iot_calibration.wgsl` | Per-reading parallel | None |
| `eco::sensor_calibration::irrigation_recommendation` | A (rewire) | `irrigation_decision.wgsl` | Per-field-layer parallel | None |
| `eco::evapotranspiration::hargreaves_et0` | A (rewire) | `batched_et0.wgsl` | Per station-day (simplified) | None |
| `eco::crop::adjust_kc_for_climate` | A (rewire) | `kc_adjustment.wgsl` | Per-field-crop parallel | None |
| Kriging interpolation | B (adapt) | `kriging_solver.wgsl` | Covariance matrix + Cholesky | Needs `barracuda::linalg` |
| Richards equation (1D) | C (new) | `richards_1d.wgsl` | FD time-stepping | Needs PDE framework |
| IoT stream processing | B (adapt) | `moving_stats.wgsl` | Windowed reduction | Needs `barracuda::timeseries` |

### Tier Definitions

- **Tier A (rewire)**: Pure arithmetic, no dependencies beyond barracuda. Map directly to WGSL compute shaders. Same precision patterns hotSpring validated (f64 FMA, exp, sqrt, trig).
- **Tier B (adapt)**: Requires barracuda primitives (linalg, timeseries). Adapt existing GPU ops to agricultural domain.
- **Tier C (new)**: Requires new barracuda capabilities (PDE solver framework).

---

## Python↔Rust Parity (Phase 2 — Complete)

All Python control functions have been ported to Rust and cross-validated.

| Category | Functions Ported | Rust Module |
|----------|:----------------:|-------------|
| Cross-validation infrastructure | 3 | `validation`, `io::csv_ts`, `cross_validate` binary |
| Statistical methods | 13 | `testutil`, `eco::evapotranspiration`, `eco::crop`, `eco::correction` |
| Sensor calibration | 3 | `eco::sensor_calibration` |
| **Total** | **19** | **65/65 values match** (tol=1e-5) |

---

## Cross-Spring Lessons for ToadStool

### From hotSpring → airSpring

| Lesson | hotSpring Discovery | airSpring Application |
|--------|--------------------|-----------------------|
| f64 GPU works | RTX 4070 SHADER_F64 confirmed | FAO-56 ET₀ needs f64 precision |
| Dispatch overhead matters | 145k dispatches = 16x slower | Batch all station-days in one dispatch |
| Hybrid GPU+Rayon | CPU parallel complements GPU | Cross-station parallelism on CPU |
| Pre-computed buffers | Avoid f32 pow() on GPU | Pre-compute pressure, gamma tables |
| Single-encoder batching | Mega-batch eliminated overhead | Apply same pattern to batched ET₀ |

### From airSpring → ToadStool (New Capabilities)

| Capability | Why airSpring Needs It | Other Springs Benefit |
|-----------|----------------------|----------------------|
| HTTP/JSON data client | Open-Meteo, NOAA CDO, OWM APIs | Any spring needing open data |
| Time series windowed ops | IoT sensor smoothing | wetSpring LC-MS chromatograms |
| CSV streaming parser | Real-time sensor data | Universal utility |
| Spatial interpolation | Soil moisture kriging | wetSpring: sampling site interpolation |
| 1D PDE solver | Richards equation | hotSpring: simpler variant of HFB |

---

## What Stays in airSpring (Domain-Specific)

These encode agricultural physics and should NOT migrate to ToadStool:

| Module | Purpose | Why Domain-Specific |
|--------|---------|---------------------|
| `eco::crop` | FAO-56 Table 12 Kc database + climate adjustment | Crop-specific coefficients |
| `eco::evapotranspiration` | FAO-56 PM, Hargreaves, sunshine Rs, monthly G | Agricultural ET₀ coefficients |
| `eco::sensor_calibration` | SoilWatch 10 VWC + irrigation recommendation | IoT sensor calibration |
| `eco::soil_moisture` | Topp equation + textures | Soil dielectric physics |
| `eco::water_balance` | FAO-56 Chapter 8 | Crop Kc, depletion tracking |
| `error` | `AirSpringError` unified error type | Domain error taxonomy |
| `testutil` | Synthetic IoT data, IA, NSE, R² | Domain-specific stats |
| Benchmark JSONs | Digitized paper values | Domain validation data |

---

## Previously Completed (Confirmed Working)

| Component | Status | Check Count |
|-----------|:------:|:-----------:|
| FAO-56 Penman-Monteith (Python) | PASS | 64 |
| Soil sensor calibration (Python) | PASS | 36 |
| IoT irrigation pipeline (Python) | PASS | 24 |
| Water balance (Python) | PASS | 18 |
| R ANOVA script | Written | Awaiting R |
| Real data: Open-Meteo (6 stations) | PASS | 918 station-days |
| Real data: OpenWeatherMap (6 stations) | PASS | 42 records |
| Real data: NOAA CDO (Lansing) | PASS | 153 days |
| ET₀ cross-check (our vs Open-Meteo) | PASS | R²=0.967 |
| Water balance on real data (4 crops) | PASS | Mass balance 0.0000 |
| Rust ET₀ validation | PASS | 31 |
| Rust soil validation | PASS | 25 |
| Rust IoT validation | PASS | 11 |
| Rust water balance validation | PASS | 13 |
| Rust sensor calibration validation | PASS | 21 |
| Rust real data (4 crops, rainfed+irrigated) | PASS | 15 |
| Rust unit tests | PASS | 175 |
| Rust integration tests | PASS | 76 |
| Rust doc-tests | PASS | 2 |
| Python↔Rust cross-validation | MATCH | 65 (tol=1e-5) |
| GPU orchestrators | **GPU-FIRST** | 6 (et0, water_balance, kriging, reduce, stream, ridge) |
| GPU evolution gaps | DOCUMENTED | 13 (6 integrated, 5 Tier B, 2 Tier C) |
| ToadStool issues | **ALL RESOLVED** | 4/4 (TS-001/002/003/004) |
| `BatchedEt0` → `fao56_et0_batch()` | **GPU-FIRST** | TS-001/002 resolved |
| `BatchedWaterBalance` → `water_balance_batch()` | **GPU-STEP** | TS-002 resolved |
| `KrigingInterpolator` ↔ `KrigingF64` | INTEGRATED | Ordinary kriging via LU |
| `SeasonalReducer` ↔ `FusedMapReduceF64` | **GPU N≥1024** | TS-004 resolved |
| **Total** | **All pass** | **330 + 918 data** |

---

## The Proof

BarraCuda, through airSpring's validation:

- **Rust implements FAO-56 correctly** — 31 ET₀ checks, Example 18 matches to 0.0005 mm/day
- **barracuda primitives integrate** — `barracuda::stats` cross-validates with airSpring computations
- **Open data replaces institutional access** — 918 station-days, 3 free APIs, zero synthetic
- **Water savings are real** — 53-72% vs naive scheduling on real 2023 Michigan weather
- **Zero unsafe, zero warnings, zero unwrap** — `cargo clippy --pedantic --nursery` = 0, `cargo fmt` = PASS, zero `.unwrap()` in production
- **97.2% library coverage** — measured by `cargo llvm-cov` (target was 90%). 7 modules at 100%, all above 89%
- **GPU determinism proven** — 4 explicit tests verify bit-identical outputs on rerun (ET₀, water balance, kriging, reducer)
- **Zero duplication** — `len_f64` unified to crate-level, `ModelType` enum replaces strings, `stress_coefficient` delegates
- **All tolerances named and documented** — every validation threshold is a named `const` with provenance comment
- **Mock isolation** — synthetic data in `testutil`, production code is clean
- **Capability-based design** — `RunoffModel` enum, configurable `BufRead` parser
- **Primal self-knowledge** — zero compile-time coupling to other primals
- **Proper error type** — `AirSpringError` enum replaces ad-hoc `String` errors
- **Complete Python feature parity** — IA, NSE, SoilWatch 10, irrigation, wind conversion, Hargreaves, Kc database all ported
- **Python↔Rust identical** — 65/65 cross-validation values match within 1e-5
- **330 quantitative checks + 293 tests** — zero failures across all experiments
- **Full pipeline demonstrated** — crop Kc → soil → ET₀ → water balance → scheduling (simulate_season binary)
- **4 crop scenarios on real data** — blueberry, tomato, corn, reference grass (rainfed + irrigated)
- **Pure Rust curve fitting** — `eco::correction` replaces `scipy.optimize.curve_fit`
- **GPU-FIRST architecture** — `BatchedEt0` → `fao56_et0_batch()`, `BatchedWaterBalance` → `water_balance_batch()`,
  `KrigingInterpolator` ↔ `KrigingF64`, `SeasonalReducer` ↔ `FusedMapReduceF64` (N≥1024 on GPU),
  `StreamSmoother` ↔ `MovingWindowStats` (IoT smoothing), `fit_ridge` ↔ `ridge_regression` (calibration)
- **4/4 ToadStool issues RESOLVED** — TS-001 (pow_f64), TS-002 (ops module), TS-003 (trig), TS-004 (buffer)
- **15 evolution gaps** — 8 integrated (GPU-first + validation + stream + ridge), 5 Tier B, 2 Tier C
- **Deepened barracuda leaning** — stats (Pearson, Spearman, bootstrap CI, variance, std_dev), validation (ValidationHarness),
  linalg (ridge regression), ops (moving_window_stats)
- **`#[must_use]` on all Result-returning public fns** — proper API hygiene
- **Sovereign Science** — AGPL-3.0, fully reproducible, no institutional access

The path to Penny Irrigation: GPU ET₀ and water balance are now live via
ToadStool. Next: GPU spatial kriging for full field mapping, 1D Richards equation
for soil water flow, and deploy on consumer hardware ($600 GPU) for sub-field
irrigation scheduling.

---

*February 25, 2026 — v0.3.7, synced to ToadStool `02207c4a` (S62+). Phases 0/0+/1/2
complete, Phase 3 GPU-FIRST. 330 checks, 293 tests (253 barracuda + 40 forge), 918
real station-days, 65/65 cross-validation match. Library coverage: 97.2% (llvm-cov).
All 4 ToadStool issues RESOLVED (commit `0c477306`). 6 GPU orchestrators wired:
BatchedEt0, BatchedWaterBalance, KrigingInterpolator, SeasonalReducer, StreamSmoother
(MovingWindowStats), fit_ridge (ridge_regression). metalForge v0.2.0: 4 absorption-ready
modules (metrics, regression, moving_window_f64, hydrology) following hotSpring's
Write → Absorb → Lean pattern. Cross-spring shader evolution documented — 608 WGSL
shaders, 46 cross-spring absorptions. 15 evolution gaps tracked (8A+5B+2C). Pure
Rust + BarraCuda GPU pipeline. Ready for Penny Irrigation deployment.*
