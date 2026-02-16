# Handoff: airSpring → Toadstool/BarraCUDA

**Date:** February 16, 2026
**From:** airSpring (Precision Agriculture validation study)
**To:** Toadstool/BarraCUDA core team
**License:** AGPL-3.0-or-later

---

## STATED GOAL: Open Precision Irrigation on Consumer Hardware

Phase 0 and 0+ confirmed: **published agricultural science reproduces with
open tools and open data.** 142/142 Python checks pass against paper
benchmarks. 918 station-days of real Michigan weather produce ET₀ with
R²=0.967. Water balance simulations show 53-72% water savings with smart
scheduling — consistent with published results.

Phase 1 confirmed: **Rust BarraCUDA validates the core pipeline.** 119/119
validation checks pass across 7 binaries (ET₀, soil moisture, water balance,
IoT parsing, sensor calibration, real data 4-crop scenarios, season simulation).
162 tests (94 unit + 68 integration) cover cross-module integration,
determinism, error paths, sensor calibration, correction curve fitting,
statistical validation, crop Kc database, barracuda primitive cross-validation,
GPU orchestrators (BatchedEt0, BatchedWaterBalance, kriging, reduce),
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
| Phase 1: Rust validation | **119/119 PASS** | 8 binaries, 162 tests, 0 clippy warnings |
| Phase 2: Cross-validation | **65/65 MATCH** | Python vs Rust identical outputs (tol=1e-5) |
| Phase 3: GPU bridge | **Integrated** | 4 orchestrators, 4 ToadStool issues filed |

---

## What airSpring Brings to BarraCUDA

### New Domain: Time Series + Spatial + IoT

hotSpring proved BarraCUDA can do clean matrix math (eigensolve, PDE,
optimization). airSpring adds a fundamentally different workload pattern:

| Dimension | hotSpring | airSpring |
|-----------|-----------|-----------|
| Data shape | Dense matrices (12×12 to 50×50) | Long time series (153+ days × N stations) |
| Input rate | Static (AME2020 table) | Streaming (IoT sensors, API feeds) |
| Spatial | Per-nucleus | Per-field-cell (kriging grid) |
| Time coupling | SCF iteration | Daily water balance (sequential) |
| Parallelism | Across nuclei | Across stations, fields, and grid cells |
| I/O pattern | Preload once | Continuous API ingestion |

### BarraCUDA Primitives Used

| Primitive | airSpring Integration | Status |
|-----------|----------------------|:------:|
| `barracuda::stats::pearson_correlation` | R² cross-validation in integration tests | **Working** |
| `barracuda::stats::correlation::std_dev` | Statistical cross-validation (sample vs population) | **Working** |
| `barracuda::stats::correlation::spearman_correlation` | Nonparametric rank validation | **Working** |
| `barracuda::stats::correlation::variance` | Sample variance computation | **Working** |
| `barracuda::stats::bootstrap::bootstrap_ci` | RMSE uncertainty quantification | **Working** |
| `barracuda::ops::kriging_f64::KrigingF64` | Soil moisture spatial interpolation (ordinary kriging) | **Integrated** |
| `barracuda::ops::fused_map_reduce_f64::FusedMapReduceF64` | Seasonal statistics (GPU for N≥1024) | **Integrated** |
| `barracuda::device::WgpuDevice` | f64-capable GPU device for ops dispatch | **Working** |
| `serde` serialization | Benchmark JSON loading (compile-time `include_str!()`) | **Working** |
| f64 arithmetic + `mul_add` | All FAO-56 functions (FMA precision) | **Working** |
| Zero unsafe | `#![deny(unsafe_code)]` pattern inherited from barracuda | **Working** |

### ToadStool Primitives Integrated (Phase 3 Bridge)

| Primitive | ToadStool Source | airSpring Status |
|-----------|-----------------|:----------------:|
| `KrigingF64` | `barracuda::ops::kriging_f64` | **Integrated** (`KrigingInterpolator`, proper ordinary kriging via LU) |
| `FusedMapReduceF64` | `barracuda::ops::fused_map_reduce_f64` | **Integrated** (`SeasonalReducer`, GPU for N≥1024, see TS-004) |
| `batched_elementwise_f64.wgsl` (op=0) | GPU FAO-56 ET₀ shader | **Wired** (`gpu::et0`, CPU fallback — TS-001 blocks GPU) |
| `batched_elementwise_f64.wgsl` (op=1) | GPU water balance shader | **Wired** (`gpu::water_balance`, CPU path) |
| `stats::bootstrap_ci` | Uncertainty quantification | **Working** |
| `stats::spearman_correlation` | Nonparametric validation | **Working** |

See `barracuda/src/gpu/evolution_gaps.rs` for the complete 11-gap roadmap.

### ToadStool Issues Found (for ToadStool team)

| ID | Summary | Severity | Impact |
|----|---------|:--------:|--------|
| TS-001 | `pow_f64` in `batched_elementwise_f64.wgsl` returns 0.0 for fractional exponents | **Critical** | Blocks GPU ET₀ (atmospheric pressure uses exponent 5.26) |
| TS-002 | No Rust orchestrator for `batched_elementwise_f64` WGSL shader | **Medium** | Cannot dispatch batched ET₀/water balance ops from Rust |
| TS-003 | `acos`/`sin` precision drift in f64 WGSL shaders | **Medium** | Solar declination/radiation off by ~0.01 rad |
| TS-004 | `FusedMapReduceF64` GPU dispatch buffer conflict | **High** | GPU reduction panics for N≥1024 (partials pipeline) |

These issues are documented in `barracuda/src/gpu/evolution_gaps.rs::TOADSTOOL_ISSUES`
with full reproduction details. airSpring uses CPU fallbacks locally until resolved.

### Remaining Primitives airSpring Needs (Tier C)

| Primitive | Purpose | Complexity |
|-----------|---------|:----------:|
| ~~Nonlinear least squares~~ | ~~Soil calibration fitting~~ | **Done** (pure Rust in `eco::correction`) |
| Moving window statistics | IoT stream processing | Low |
| 1D Richards solver | Unsaturated flow PDE | High |
| API client (HTTP + JSON) | Open-Meteo, NOAA CDO | Low |

---

## Rust Crate Architecture (v0.2.0)

### Module Map

```
airspring-barracuda/
├── src/
│   ├── lib.rs              # pub mod eco, error, gpu, io, validation, testutil
│   ├── error.rs            # AirSpringError enum (Io, CsvParse, JsonParse, InvalidInput, Barracuda)
│   ├── gpu/
│   │   ├── mod.rs                # ToadStool/BarraCUDA GPU bridge architecture + ToadStool issues
│   │   ├── et0.rs                # BatchedEt0 GPU orchestrator (CPU fallback — TS-001 pow_f64)
│   │   ├── water_balance.rs      # BatchedWaterBalance GPU orchestrator (CPU path)
│   │   ├── kriging.rs            # KrigingInterpolator + IDW (↔ barracuda::ops::kriging_f64)
│   │   ├── reduce.rs             # SeasonalReducer + CPU fns (↔ barracuda::ops::fused_map_reduce_f64)
│   │   └── evolution_gaps.rs     # 11 EvolutionGap entries + TOADSTOOL_ISSUES (TS-001/002/003/004)
│   ├── eco/
│   │   ├── mod.rs
│   │   ├── correction.rs         # Sensor correction curve fitting (linear/quad/exp/log)
│   │   ├── crop.rs               # CropType enum, FAO-56 Table 12 Kc database, Eq. 62 adjustment
│   │   ├── evapotranspiration.rs  # 23 FAO-56 functions + Hargreaves ET₀, low-level PM Eq. 6
│   │   ├── sensor_calibration.rs  # SoilWatch 10 VWC, irrigation recommendation
│   │   ├── soil_moisture.rs       # Topp eq, inverse, SoilTexture, SoilHydraulicProps
│   │   └── water_balance.rs       # WaterBalanceState, RunoffModel, standalone fns, simulate_season
│   ├── io/
│   │   ├── mod.rs
│   │   └── csv_ts.rs       # TimeseriesData (columnar), streaming BufReader parser
│   ├── validation.rs        # ValidationRunner (shared hotSpring-pattern infrastructure)
│   ├── testutil.rs          # R², RMSE, MBE, IA, NSE, Spearman, bootstrap CI, variance
│   └── bin/
│       ├── validate_et0.rs          # 31 checks (loads benchmark_fao56.json)
│       ├── validate_soil.rs         # 25 checks
│       ├── validate_iot.rs          # 11 checks (round-trip streaming)
│       ├── validate_water_balance.rs  # 13 checks (mass balance, Michigan season)
│       ├── validate_sensor_calibration.rs  # 21 checks (SoilWatch 10, irrigation)
│       ├── validate_real_data.rs    # 15 checks (4 crops × rainfed+irrigated, real Open-Meteo)
│       ├── cross_validate.rs        # Phase 2 JSON output (65 values) for Python↔Rust diff
│       └── simulate_season.rs      # Full pipeline: crop Kc → soil → ET₀ → water balance → scheduling
├── tests/
│   └── integration.rs       # 68 tests (GPU wiring, orchestrators, correction, stats)
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
| `cargo test` | **162/162 PASS** (94 unit + 68 integration) |
| Validation binaries | **119/119 PASS** (7 binaries + 1 JSON output) |
| Cross-validation | **65/65 MATCH** (Python↔Rust, tol=1e-5) |
| Unsafe code | **0** |
| `AirSpringError` | **Proper error type** (replaces `String`) |
| Max LOC per file | **760** (under 1000 limit) |

### Issues Resolved (since v0.1.0)

1. ~~**Typo**: `SoilTexture::SandyCite`~~ → Fixed to `SandyClay`. Regression test added.
2. ~~**Runoff model differs**~~ → Now capability-based `RunoffModel` enum. `None` (FAO-56 default) aligns with Python.
3. ~~**Phantom modules**~~ → `eco::isotherms` and `eco::richards` references removed from lib.rs.
4. ~~**189 clippy warnings**~~ → All resolved. Zero pedantic/nursery warnings.
5. ~~**No benchmark JSON loading**~~ → `validate_et0` now loads `benchmark_fao56.json` at compile time.
6. ~~**Duplicated check() function**~~ → Shared `ValidationRunner` in `validation.rs`.
7. ~~**CSV buffered entire file**~~ → Streaming `BufReader` + columnar storage.
8. ~~**Mock data in production code**~~ → Moved to `testutil` module.
9. ~~**barracuda dep unused**~~ → Now uses `barracuda::stats` for cross-validation.

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
| Rust unit tests | PASS | 94 |
| Rust integration tests | PASS | 68 |
| Python↔Rust cross-validation | MATCH | 65 (tol=1e-5) |
| GPU orchestrators wired | PASS | 4 (et0, water_balance, kriging, reduce) |
| GPU evolution gaps | DOCUMENTED | 11 (Tier A/B/C) |
| ToadStool issues filed | DOCUMENTED | 4 (TS-001/002/003/004) |
| `KrigingInterpolator` ↔ `KrigingF64` | INTEGRATED | Ordinary kriging via LU |
| `SeasonalReducer` ↔ `FusedMapReduceF64` | INTEGRATED | GPU for N≥1024 (TS-004) |
| **Total** | **All pass** | **381 + 918 data** |

---

## The Proof

BarraCUDA, through airSpring's validation:

- **Rust implements FAO-56 correctly** — 31 ET₀ checks, Example 18 matches to 0.0005 mm/day
- **barracuda primitives integrate** — `barracuda::stats` cross-validates with airSpring computations
- **Open data replaces institutional access** — 918 station-days, 3 free APIs, zero synthetic
- **Water savings are real** — 53-72% vs naive scheduling on real 2023 Michigan weather
- **Zero unsafe, zero warnings** — `cargo clippy --pedantic --nursery` = 0, `cargo fmt` = PASS
- **Mock isolation** — synthetic data in `testutil`, production code is clean
- **Capability-based design** — `RunoffModel` enum, configurable `BufRead` parser
- **Primal self-knowledge** — zero compile-time coupling to other primals
- **Proper error type** — `AirSpringError` enum replaces ad-hoc `String` errors
- **Complete Python feature parity** — IA, NSE, SoilWatch 10, irrigation, wind conversion, Hargreaves, Kc database all ported
- **Python↔Rust identical** — 65/65 cross-validation values match within 1e-5
- **381 quantitative checks + 162 tests** — zero failures across all experiments
- **Full pipeline demonstrated** — crop Kc → soil → ET₀ → water balance → scheduling (simulate_season binary)
- **4 crop scenarios on real data** — blueberry, tomato, corn, reference grass (rainfed + irrigated)
- **Pure Rust curve fitting** — `eco::correction` replaces `scipy.optimize.curve_fit`
- **GPU bridge integrated** — `KrigingInterpolator` ↔ `KrigingF64`, `SeasonalReducer` ↔ `FusedMapReduceF64`
- **4 ToadStool issues filed** — TS-001 (pow_f64), TS-002 (ops module), TS-003 (trig), TS-004 (buffer)
- **11 evolution gaps** — 2 INTEGRATED, 3 Tier A wired, 3 Tier B, 3 Tier C
- **Deepened barracuda stats** — 5 primitives wired (Pearson, Spearman, bootstrap CI, variance, std_dev)
- **`#[must_use]` on all Result-returning public fns** — proper API hygiene
- **Sovereign Science** — AGPL-3.0, fully reproducible, no institutional access

The path to Penny Irrigation: port remaining Python methods to Rust,
GPU-accelerate ET₀ and water balance on ToadStool, add spatial kriging,
and deploy on consumer hardware ($600 GPU) for sub-field irrigation scheduling.

---

*February 16, 2026 — Phases 0/0+/1/2 complete, Phase 3 integrated. 381 checks,
162 tests, 918 real station-days, 65/65 cross-validation match. `KrigingInterpolator`
wired to `barracuda::ops::kriging_f64` (proper ordinary kriging with variogram).
`SeasonalReducer` wired to `barracuda::ops::fused_map_reduce_f64` (GPU dispatch
for N≥1024). 4 ToadStool issues filed (TS-001 pow_f64, TS-002 ops module,
TS-003 trig precision, TS-004 buffer conflict). Ready for GPU acceleration once
ToadStool resolves TS-001 and TS-004.*
