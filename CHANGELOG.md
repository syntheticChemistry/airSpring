# Changelog

All notable changes to airSpring follow [Keep a Changelog](https://keepachangelog.com/).

## [0.3.8] - 2026-02-25

### ToadStool Deep Audit — Richards PDE Promoted, Evolution Gaps Reconciled

Deep audit of ToadStool HEAD `02207c4a` (S62+) revealed upstream has
absorbed the Richards PDE solver (`pde::richards::solve_richards` with
van Genuchten-Mualem, Picard + Crank-Nicolson + Thomas). Promoted
from Tier C ("needs new primitive") to Tier B ("wire with domain params").

Also discovered upstream `linalg::tridiagonal_solve_f64` (Thomas algorithm)
and `numerical::rk45_solve` (Dormand-Prince adaptive ODE) — both added as
new Tier B evolution gaps for future soil dynamics work.

Confirmed metalForge candidates (metrics, regression, hydrology,
moving_window_f64) are NOT yet absorbed upstream — pending ToadStool review.

### Changed

- **`evolution_gaps.rs`**: Richards PDE promoted Tier C → Tier B. Added
  `tridiagonal_batch` and `rk45_adaptive` as new Tier B gaps. Gap count
  updated from 15 (8A+5B+2C) to 17 (8A+8B+1C).
- **`specs/BARRACUDA_REQUIREMENTS.md`**: Remaining gaps updated. Richards
  promoted with note on upstream solver capabilities.
- **`specs/CROSS_SPRING_EVOLUTION.md`**: Timeline updated with v0.3.8 audit.
  Gap summary corrected to 8B+1C.
- **`wateringHole/handoffs/V001`**: Version bumped. Richards promotion noted.
  metalForge absorption status clarified.
- **`metalForge/ABSORPTION_MANIFEST.md`**: Explicit "NOT YET ABSORBED" status.
- **Root docs**: Version bumped to v0.3.8. Evolution gap counts updated.
- **`Cargo.toml`**: Version `0.3.7` → `0.3.8`.

## [0.3.7] - 2026-02-25

### metalForge Evolution — Absorption-Ready Extensions

Evolved `airspring-forge` from v0.1.0 (2 modules, 18 tests) to v0.2.0
(4 modules, 40 tests), following hotSpring's Write → Validate → Handoff →
Absorb → Lean pattern for upstream barracuda absorption.

**New forge modules:**
- **`moving_window_f64`**: CPU f64 sliding window statistics (mean, variance,
  min, max). Complements upstream f32 GPU path (wetSpring S28+). 7 tests
  including diurnal temperature smoothing.
- **`hydrology`**: Pure-Rust Hargreaves ET₀, batched ET₀, crop coefficient
  interpolation (FAO-56 Ch. 6), soil water balance (FAO-56 Ch. 8).
  Validated against FAO-56 reference data. 13 tests.
- **`regression` evolved**: Added `FitResult::predict()` and `predict_one()`
  following `RidgeResult::predict()` from `barracuda::linalg::ridge`. Added
  `model` field for self-describing results. 2 new predict tests.
- **`fit_all` evolved**: Now returns `Vec<FitResult>` (was `Vec<(&str, FitResult)>`),
  since `FitResult` carries its own `model` name.

Updated `ABSORPTION_MANIFEST.md` with full signatures, validation provenance,
post-absorption rewiring plan, and absorption procedure matching hotSpring's
format.  Updated root docs, whitePaper, and HANDOFF.

**293 tests** (253 barracuda + 40 forge), **123 validation checks** across 8 binaries.

## [0.3.6] - 2026-02-24

### ToadStool Sync + Validation Rewire + Cross-Spring Evolution

Synced to ToadStool HEAD `02207c4a` (S62+, 50 commits since handoff).
Rewired all 6 validation binaries from local `ValidationRunner` to upstream
`barracuda::validation::ValidationHarness` (absorbed from neuralSpring S59).
Renamed BarraCUDA → BarraCuda throughout (matching ToadStool S42 rename).

**New wiring (cross-spring evolution):**
- `gpu::stream::StreamSmoother` — wraps `MovingWindowStats` (wetSpring S28+) for
  IoT sensor stream smoothing. f64→f32→f64 bridge with CPU fallback.
- `eco::correction::fit_ridge` — wraps `barracuda::linalg::ridge::ridge_regression`
  (wetSpring ESN calibration) for regularized sensor calibration.
- `bench_airspring_gpu` — benchmark binary measuring CPU throughput for all 6 GPU
  orchestrators with cross-spring provenance annotations.
- `specs/CROSS_SPRING_EVOLUTION.md` — full provenance story documenting 608 WGSL
  shaders across 4 Springs (hotSpring 56, wetSpring 25, neuralSpring 20, shared 507).

Evolution gaps updated: `moving_window_stats` and `ridge_regression` promoted from
Tier B to Tier A (wired). 15 total (8 Tier A, 5 Tier B, 2 Tier C).

Deduplicated `len_f64` utility (was copied 4×), evolved stringly-typed
`model_type: &'static str` to `ModelType` enum, delegated duplicated
`stress_coefficient` logic, added 4 GPU determinism tests (bit-identical
verification), and filled coverage gaps. Library coverage: **97.2%** (target 90%).
Added `Copy` to 8 small value types. Fixed wind speed unit bug in
cross-validation. Started **metalForge** — `airspring-forge` crate with
statistical metrics and regression primitives staged for upstream absorption.

**293 tests** (253 barracuda + 40 forge), **123 validation checks** across 8 binaries.
Synced evolution gaps: 15 total (8 Tier A, 5 Tier B, 2 Tier C).

### Added

- **`gpu::stream`** module: `StreamSmoother` wraps ToadStool's `MovingWindowStats`
  (wetSpring S28+ environmental monitoring shader) with f64↔f32 bridge for IoT
  sensor stream smoothing. `smooth_cpu()` CPU fallback. 6 unit tests.
- **`eco::correction::fit_ridge`**: Ridge regression via `barracuda::linalg::ridge`
  (wetSpring ESN calibration). Regularized linear calibration with design matrix
  construction and goodness-of-fit reporting. 3 unit tests.
- **`bench_airspring_gpu`** binary: Benchmarks all GPU orchestrators (ET₀, reduce,
  stream, kriging, ridge) with cross-spring provenance annotations and throughput
  reporting. Measures CPU baselines at multiple problem sizes.
- **`specs/CROSS_SPRING_EVOLUTION.md`**: Full cross-spring shader provenance
  documenting 608 WGSL shaders, 46 absorptions, 4 Spring contributions, and the
  timeline of how hotSpring precision shaders, wetSpring bio/environmental shaders,
  and neuralSpring ML shaders evolved to benefit airSpring's agriculture pipeline.
- **4 GPU determinism tests** in `gpu_integration.rs`:
  `test_gpu_batched_et0_deterministic`, `test_gpu_water_balance_deterministic`,
  `test_gpu_reducer_deterministic`, `test_gpu_kriging_deterministic` — each runs
  identical inputs twice and asserts bit-identical results (`< f64::EPSILON`).
- **6 coverage-filling tests** in `eco/correction.rs`:
  `test_model_type_as_str_and_display`, `test_evaluate_all_model_types`,
  `test_fit_linear_insufficient_points`, `test_fit_quadratic_insufficient_points`,
  `test_fit_exponential_all_negative_y`, `test_fit_logarithmic_all_negative_x`,
  `test_fit_linear_singular`.
- **`metalForge/forge/`**: `airspring-forge` v0.1.0 crate with 18 tests:
  - `metrics` module: `rmse`, `mbe`, `nash_sutcliffe`, `index_of_agreement`,
    `coefficient_of_determination` — absorption target `barracuda::stats::metrics`.
  - `regression` module: `fit_linear`, `fit_quadratic`, `fit_exponential`,
    `fit_logarithmic`, `fit_all` — absorption target `barracuda::stats::regression`.
  - `ABSORPTION_MANIFEST.md` documenting upstream integration procedure.

### Changed

- **`validation.rs`**: Replaced local `ValidationRunner` with re-export of
  `barracuda::validation::ValidationHarness`. Added `banner()` and `section()`
  free functions for airSpring-specific output formatting. JSON utilities
  (`parse_benchmark_json`, `json_f64`) retained as airSpring-specific.
- **All 6 validation binaries** rewired: `check()` → `check_abs()`,
  `check_bool(label, cond, expected)` → `check_bool(label, cond)`,
  `v.section()` → `validation::section()`. Zero-tolerance checks use
  `f64::EPSILON` (upstream `check_abs` uses strict `<` not `<=`).
- **`evolution_gaps.rs`**: Updated to ToadStool HEAD `02207c4a`. Moving window
  stats, Nelder-Mead, ridge regression promoted Tier C → Tier B. Validation
  harness added as Tier A absorbed. Richards PDE upgraded (upstream CN +
  tridiagonal now available). 11 → 13 gaps (6A + 5B + 2C).
- **BarraCUDA → BarraCuda** naming across all docs and code (49 replacements,
  matching ToadStool S42 rename).
- **`lib.rs`**: Added crate-level `pub(crate) const fn len_f64<T>()`.
  Four local copies in `correction.rs`, `csv_ts.rs`, `reduce.rs`, `testutil.rs`
  replaced with `use crate::len_f64`.
- **`eco/correction.rs`**: `model_type: &'static str` evolved to
  `ModelType` enum (`Linear`, `Quadratic`, `Exponential`, `Logarithmic`) with
  `as_str()` and `Display`. `evaluate()` match is now exhaustive (no `_ => NAN`
  dead arm).
- **`eco/water_balance.rs`**: `WaterBalanceState::stress_coefficient()` now
  delegates to the standalone `stress_coefficient()` function, eliminating
  duplicated logic.
- **`Copy` derive** added to 8 small value types: `DailyInput`, `DailyOutput`,
  `Et0Result`, `SoilHydraulicProps`, `SeasonalStats`, `ColumnStats`,
  `SensorReading`, `TargetPoint`. Enables pass-by-value and eliminates
  unnecessary clones.
- **`tests/eco_integration.rs`**: Updated `ModelType` comparison from string
  to enum variant.
- **`Cargo.toml`**: Version `0.3.4` → `0.3.6`.

### Fixed

- **`cross_validate.rs`**: Wind speed was passed as km/h directly to
  `wind_speed_at_2m()` which expects m/s, causing u2 = 7.48 instead of 2.08.
  Added `/ 3.6` conversion. All 65/65 cross-validation values now match Python.
- **`scripts/cross_validate.py`**: Hardcoded inputs replaced with loading from
  `benchmark_fao56.json` (single source of truth), eliminating pre-rounded
  values that caused 1.7e-3 drift.

### Documentation

- **All root docs** updated to v0.3.6: README.md, CONTROL_EXPERIMENT_STATUS.md,
  HANDOFF, CHANGELOG.
- **whitePaper/** updated: README (Phase 3 GPU-FIRST), METHODOLOGY (330 checks),
  STUDY (123/123, 65/65, Phase 3 section).
- **specs/** updated: README (Phase 0-3 complete), BARRACUDA_REQUIREMENTS (correct
  module names, GPU DONE), PAPER_REVIEW_QUEUE (date).
- **`evolution_gaps.rs`**: Updated to v0.3.6, 123/123 checks, GPU determinism note.

### Quality Gates

| Check | Before | After |
|-------|--------|-------|
| `cargo test` | 235 (161+74) | **244** (166+76+2) |
| Library coverage (`llvm-cov`) | ~88% (unit only) | **97.2%** (all tests) |
| GPU determinism | Implicit (GPU vs CPU) | **Explicit** (same input → bit-identical) |
| `len_f64` copies | 4 | **1** (crate-level) |
| `model_type` typing | `&'static str` | **`ModelType` enum** |
| `stress_coefficient` duplication | 2 impls | **1 + delegation** |

## [0.3.4] - 2026-02-17

### Coverage Push & Code Hygiene

Library test coverage raised from 78.3% to 88.2% (56 new unit tests, 105 → 161).
Remaining gap is GPU device-backed paths testable only via integration tests.

Magic numbers extracted to named constants: Topp equation coefficients
(`TOPP_A0`–`TOPP_A3`), Newton-Raphson parameters (`INVERSE_TOPP_MAX_ITER`,
`INVERSE_TOPP_CONVERGENCE`), and kriging distance threshold
(`COLLOCATED_DIST_SQ`). Remaining `#[allow]` in binaries narrowed to inline
per-cast annotations with justification comments. Avoidable `.clone()` calls
eliminated in `validate_real_data.rs`. Test paths migrated from hardcoded
`/tmp/` to `std::env::temp_dir()`. Benchmark JSON files enriched with
`_provenance` metadata blocks. `validate_iot.rs` refactored from monolithic
`main()` into `validate_sensor_stats()` + `validate_csv_round_trip()`.

### Changed

- **`eco/soil_moisture.rs`**: Topp coefficients, Newton-Raphson iteration
  params, and epsilon bounds extracted to 8 named constants with provenance.
- **`gpu/kriging.rs`**: IDW collocated-distance threshold extracted to
  `COLLOCATED_DIST_SQ` constant.
- **`gpu/reduce.rs`**: Added 9 unit tests (empty/single/large/constant
  values, sentinel checks).
- **`gpu/kriging.rs`**: Added 7 unit tests (exponential variogram, closer-
  sensor dominance, multiple targets, variance-at-sensor).
- **`gpu/et0.rs`**: Added 5 unit tests (toadstool conversion, debug format,
  empty GPU, seasonal variation).
- **`gpu/water_balance.rs`**: Added 4 unit tests (to_toadstool, from_state,
  empty step, deep percolation, TAW clamp).
- **`eco/soil_moisture.rs`**: Added 6 unit tests (all textures, Ksat
  ordering, monotonicity, boundary behaviour, clay PAW).
- **`eco/water_balance.rs`**: Added 8 unit tests (runoff model, theta,
  deep percolation, irrigation trigger, standalone functions).
- **`error.rs`**: Added 11 unit tests (Display, Debug, source, From impls).
- **`validation.rs`**: Added 4 unit tests (section, counters, root-level
  JSON, tolerance boundary).
- **`validate_iot.rs`**: Refactored into `validate_sensor_stats()` and
  `validate_csv_round_trip()` helpers; narrowed `#[allow]` to per-cast.
- **`simulate_season.rs`**: Eliminated function-level `#[allow]`;
  `usize→u32` casts now use `u32::try_from().expect()`, `usize→f64` via
  inline `#[allow]` with justification.
- **`validate_real_data.rs`**: Replaced `.clone()` with separate
  `irr_inputs`/`irr_outputs` Vecs built during loop.
- **`tests/io_and_errors.rs`**: `/tmp/` paths replaced with
  `std::env::temp_dir()` for portability.
- **Benchmark JSONs**: All 4 benchmark files (`benchmark_fao56.json`,
  `benchmark_dong2020.json`, `benchmark_dong2024.json`,
  `benchmark_water_balance.json`) enriched with `_provenance` block
  (method, digitized_by, created, validated_by, repository).
- **`Cargo.toml`**: Version `0.3.3` → `0.3.4`.

## [0.3.3] - 2026-02-17

### Lint Hygiene & Structural Refactoring

Centralised `usize → f64` casts behind `len_f64()` helpers, eliminating 13
`#[allow(clippy::cast_precision_loss)]` annotations across `testutil`, `correction`,
`gpu/reduce`, and `csv_ts`. Refactored `cross_validate.rs` from a 226-line `main()`
into 5 focused functions, removing `#[allow(too_many_lines)]`. Refactored
`correction.rs`: renamed single-character variables to descriptive names
(`sx` → `s_x`, `sxy` → `s_cross`), extracted 3×3 Cramer solve into `det3()` +
`cramer_3x3()`, removing all 5 `#[allow]` annotations from `fit_quadratic`.
Removed 3 stale `#[allow(cast_precision_loss)]` from binaries that no longer
had any `as f64` casts. Documented `.unwrap_or()` fallbacks with named constants
(`DEFAULT_TOPP_TOL`, `ES_TOL`, `BANGKOK_DELTA_TOL`).

### Changed

- **`testutil.rs`**: Added `const fn len_f64<T>()` helper; removed 6
  `#[allow(cast_precision_loss)]` from `rmse`, `mbe`, `index_of_agreement`,
  `nash_sutcliffe`, `coefficient_of_determination`, `bootstrap_rmse`.
- **`eco/correction.rs`**: Added `const fn len_f64<T>()`; extracted `det3()` and
  `cramer_3x3()` helpers; renamed variables in `fit_linear` and `fit_quadratic`;
  removed all `#[allow]` from `fit_linear`, `fit_quadratic`, and `goodness_of_fit`.
- **`gpu/reduce.rs`**: Added `const fn len_f64<T>()`; removed 4
  `#[allow(cast_precision_loss)]` from `compute_stats`, `seasonal_mean`,
  `sum_of_squares_from_mean`, `sample_variance`.
- **`io/csv_ts.rs`**: Added `const fn len_f64<T>()`; removed
  `#[allow(cast_precision_loss)]` from `column_stats`.
- **`cross_validate.rs`**: Split monolithic `main()` into `uccle_core()`,
  `uccle_extended()`, `soil_and_sensor_values()`, `water_balance_and_correction()`,
  `merge_into()` — main now 10 lines. Removed dead `UccleInputs` struct.
- **`validate_real_data.rs`**: Removed stale `#[allow(cast_precision_loss)]`.
- **`validate_water_balance.rs`**: Removed stale `#[allow(cast_precision_loss)]`.
- **`validate_et0.rs`**: Narrowed allow from 3 lints to 2 (removed `cast_precision_loss`).
- **`validate_soil.rs`**: Added `DEFAULT_TOPP_TOL` constant for `.unwrap_or()` fallback.
- **`validate_et0.rs`**: `.unwrap_or()` fallbacks now use named constants (`ES_TOL`,
  `BANGKOK_DELTA_TOL`).
- **`Cargo.toml`**: Version `0.3.1` → `0.3.3`.

## [0.3.2] - 2026-02-17

### Hardcoding Elimination & Binary Refactoring

All bare numeric literals in validation binaries evolved to named `const` declarations
with provenance comments. Remaining `panic!()` calls in production code replaced with
`.expect()`. Binary `main()` functions refactored into focused helper functions, removing
all `#[allow(clippy::too_many_lines)]` annotations except where `cast_` lints still apply.
Cargo.toml version synchronized with CHANGELOG. HANDOFF doc updated for 177 tests, 8
binaries, and refactored test layout.

### Changed

- **`validate_sensor_calibration.rs`**: Extracted `validate_soilwatch10()`,
  `validate_irrigation()`, `validate_performance_and_demos()` — removed
  `#[allow(too_many_lines)]` from `main()`. All tolerances named: `EXACT_TOL`,
  `IR_TOL`, `IA_CRITERION`, `P_SIGNIFICANT`, `SAVINGS_TOL`.
- **`validate_real_data.rs`**: Extracted `validate_station_et0()`,
  `validate_scenario()`, `run_irrigated()` — removed `#[allow(too_many_lines)]`
  from `main()`. `panic!()` replaced with `.expect()`.
- **`simulate_season.rs`**: Extracted `SimResult` struct, `simulate_rainfed()`,
  `simulate_smart()`, `generate_weather()` — removed `#[allow(too_many_lines)]`.
  Named constants: `LN_GUARD`, `RAIN_PROBABILITY`, `RAIN_MEAN_MM`, `RAIN_CAP_MM`,
  `MAX_IRRIGATION_MM`.
- **`validate_iot.rs`**: Named constants: `TEMP_MEAN_TOL`, `TEMP_EXTREMES_TOL`,
  `SM1_VALID_MIN/MAX`, `PAR_MAX_TOL`, `ROUNDTRIP_TEMP_TOL`.
- **`validate_et0.rs`**: Named constants: `ES_TOL`, `VPD_TOL`, `RN_TOL`,
  `BANGKOK_ES_TOL`, `BANGKOK_DELTA_TOL`, `BANGKOK_GAMMA_TOL`, `COLD_ET0_TOL`.
- **`validate_water_balance.rs`**: Named constants: `PER_STEP_STRICT`,
  `SIM_MASS_BALANCE_TOL`, `KS_MIDPOINT_TOL`. Removed local `sim_mass_balance_tol`
  variable in favor of module-level `const`.
- **`Cargo.toml`**: Version `0.2.0` → `0.3.1` (synchronized with CHANGELOG).
- **HANDOFF doc**: Updated test counts (177), binary count (8), crate version
  (v0.3.1), and test module layout (4 files replacing `integration.rs`).

### Removed

- 2 `panic!()` calls in production binaries.
- `#[allow(clippy::too_many_lines)]` from `validate_sensor_calibration`, `validate_real_data`,
  and `simulate_season` `main()` functions (moved logic to helper functions).

## [0.3.1] - 2026-02-16

### Deep Debt Resolution & Modern Idiomatic Rust

Comprehensive audit and evolution. All production `.unwrap()` eliminated. Monolithic
integration test (1726 lines) smart-refactored into 4 domain-focused test modules.
Validation binaries evolved to load thresholds from benchmark JSON. Coverage measured
and gaps filled.

**177 tests** (105 unit + 72 integration), **119 validation checks** across 8 binaries.
Library coverage: 90%+ (all eco modules >95%, all GPU modules >90%).

### Added

- **7 unit tests** for `ValidationRunner` (check, check_bool, JSON parsing, path traversal).
- **2 integration tests**: exhaustive soil texture coverage, Ksat ordering.
- **`validate_soil.rs`**: Now loads `benchmark_dong2020.json` for Topp published points
  and tolerance (was hardcoded inline).
- **`validate_water_balance.rs`**: Now loads `benchmark_water_balance.json` for mass
  balance tolerance and Michigan ET range (was hardcoded inline).

### Changed

- **`tests/integration.rs`** (1726 lines) refactored into 4 domain-focused modules:
  - `eco_integration.rs` (534 lines) — FAO-56, water balance, soil, crop, sensors
  - `gpu_integration.rs` (701 lines) — GPU orchestrators, evolution gaps, ToadStool
  - `io_and_errors.rs` (169 lines) — CSV parsing, round-trips, error types
  - `stats_integration.rs` (216 lines) — BarraCuda cross-validation, Spearman, bootstrap
- **`validate_real_data.rs`**: Evolved to use `ValidationRunner` with proper exit codes.
  All `.unwrap()` replaced with `.expect()` with descriptive messages. Thresholds
  extracted to named constants with documented justification.
- **`validate_et0.rs`**: All `json_f64(...).unwrap()` → `.expect("path description")`.
- **`validate_iot.rs`**: All `.unwrap()` → `.expect()` with context.
- **`csv_ts::column_stats`**: Documentation clarifies population statistics (N divisor)
  choice and points to `barracuda::stats` for sample statistics (N−1).
- All binary `const` declarations moved to module level (clippy `items_after_statements`).

### Fixed

- Zero clippy pedantic/nursery warnings.
- No bare `.unwrap()` in any production (non-test) code.
- All validation thresholds sourced from benchmark JSON or named constants.

## [0.3.0] - 2026-02-16

### GPU-First Evolution

Rewired all GPU orchestrators to use resolved `ToadStool` primitives. All four
`ToadStool` issues (TS-001 through TS-004) are **RESOLVED** as of `ToadStool`
commit `0c477306`. airSpring is now GPU-first with CPU fallback.

**168 tests** (98 unit + 70 integration), **119 validation checks** across 8 binaries.

### Added

- **`BatchedEt0::gpu(device)`**: GPU-first ET₀ via `BatchedElementwiseF64::fao56_et0_batch()`.
  `StationDay` input type maps directly to ToadStool shader layout (rh_max/rh_min).
  CPU fallback via `compute()` still available for pre-computed `ea` inputs.
- **`BatchedWaterBalance::with_gpu(device)`**: GPU-backed constructor.
  `gpu_step()` dispatches one timestep across M fields in parallel via
  `BatchedElementwiseF64::water_balance_batch()`. Applies Ks stress coefficient.
- **`FieldDayInput`**: New type for GPU water balance step inputs matching ToadStool layout.
- **`StationDay`**: New type for GPU ET₀ inputs matching ToadStool `StationDayInput`.
- **`IssueStatus` enum**: Tracks resolved/open status of ToadStool issues.
- **2 new GPU-matches-CPU integration tests**: `test_gpu_batched_et0_station_day_gpu_dispatch`,
  `test_gpu_water_balance_gpu_step_dispatch`.
- **4 new unit tests**: `test_station_day_cpu_fallback`, `test_station_day_multiple`,
  `test_gpu_step_cpu_fallback`, `test_gpu_step_clamp`.

### Changed

- **`BatchedEt0`**: Now holds optional `BatchedElementwiseF64` engine. `Backend::Gpu`
  is the new default. Old `compute()` CPU path unchanged.
- **`BatchedWaterBalance`**: Now holds optional `BatchedElementwiseF64` engine.
  CPU season simulation via `simulate_season()` unchanged.
- **`SeasonalReducer`**: TS-004 resolved — GPU dispatch for N≥1024 works without panic.
  Removed `catch_unwind` workaround from large array integration test.
- **`evolution_gaps`**: All 4 ToadStool issues marked RESOLVED with `IssueStatus::Resolved`.
  Evolution gaps updated: 5 Tier A integrated, 3 Tier B, 3 Tier C.
- **Integration tests**: Updated TS issue tests to verify all 4 resolved. Large array
  GPU test now directly asserts (no `catch_unwind`).

### Quality Gates

| Check | Before | After |
|-------|--------|-------|
| `cargo test` | 162 (94+68) | **168** (98+70) |
| GPU orchestrators | 4 (CPU fallback) | **4 GPU-first** |
| ToadStool issues | 4 open | **4/4 resolved** |
| GPU N≥1024 reduce | Panics (TS-004) | **Works** |
| GPU ET₀ | CPU only (TS-001/002) | **GPU dispatch** |
| GPU water balance | CPU only (TS-002) | **GPU step** |

## [0.2.0] - 2026-02-16

### Deep Debt Elimination

Comprehensive audit and remediation of the Rust validation crate. Evolved from
prototype to modern, idiomatic Rust with full validation fidelity.

**Previous**: 189 clippy warnings, formatting failures, phantom modules, loose
tolerances, duplicated code, no integration tests.

**After**: Zero clippy pedantic/nursery warnings, zero formatting issues, zero
doc warnings, 162 tests (94 unit + 68 integration), 119 validation checks across
8 binaries, proper `AirSpringError` type, complete Python feature parity including
Hargreaves ET₀, FAO-56 Kc crop database, sunshine/temperature radiation estimation,
sensor calibration, pure Rust correction curve fitting (replaces scipy), standalone
FAO-56 water balance API, real data validation on Michigan weather (918 station-days),
barracuda primitives actively used (`KrigingInterpolator` ↔ `KrigingF64`,
`SeasonalReducer` ↔ `FusedMapReduceF64`), mocks isolated, 4 GPU orchestrators
integrated with CPU fallback, 4 ToadStool issues filed (TS-001/002/003/004),
11 documented evolution gaps.

### Added

- **Validation infrastructure** (`src/validation.rs`): Shared `ValidationRunner`
  for hotSpring-pattern binaries. Eliminates 4× duplicated `check()` function.
  Includes benchmark JSON loading with `serde_json`.
- **Integration test suite** (`tests/integration.rs`): 14 tests covering:
  - Cross-module integration (ET₀ → water balance, soil texture → water balance)
  - CSV round-trip fidelity (generate → write → stream-parse → compare)
  - Determinism verification (ET₀, water balance, Topp inverse)
  - Error path coverage (empty input, missing columns, nonexistent files)
  - Boundary conditions (arctic, tropical, saturation overflow)
  - Configurable runoff model validation
- **`testutil` module** (`src/testutil.rs`): Synthetic data generation isolated
  from production library code. Includes `r_squared()` (backed by
  `barracuda::stats::pearson_correlation`), `rmse()`, and `mbe()` for
  cross-validation. Mocks no longer pollute the production API.
- **BarraCuda cross-validation**: Integration tests verify airSpring
  computations against `barracuda::stats` primitives (Pearson correlation,
  population vs sample std_dev ratio). Proves the Spring thesis.
- **GPU evolution mapping**: Added Rust Module → WGSL Shader → Pipeline Stage
  mapping in the handoff document with tier classifications (A/B/C).
- **Benchmark JSON integration**: `validate_et0` now loads
  `control/fao56/benchmark_fao56.json` at compile time via `include_str!()`,
  validating against exact published FAO-56 Table 2.3, Table 2.4, and Example
  18 (Uccle daily) values with provenance.
- **Configurable runoff model**: `RunoffModel` enum with `None` (FAO-56 default)
  and `SimpleThreshold` variants. Water balance no longer hardcodes a specific
  runoff formula — capability-based, not assumption-based.
- **Builder pattern**: `WaterBalanceState::with_runoff_model()` for composable
  configuration.
- **`AirSpringError` enum** (`src/error.rs`): Unified error type replacing ad-hoc
  `String` errors. Variants: `Io`, `CsvParse`, `JsonParse`, `InvalidInput`,
  `Barracuda`. Implements `std::error::Error` with proper `source()` chain.
  `From<std::io::Error>` and `From<serde_json::Error>` for `?` ergonomics.
- **`SoilWatch` 10 calibration** (`src/eco/sensor_calibration.rs`): Dong et al.
  (2024) Eq. 5 — VWC from raw analog counts. Horner's method for numerical
  stability. Includes `soilwatch10_vwc()`, `soilwatch10_vwc_vec()`,
  `irrigation_recommendation()`, `SoilLayer`, and `multi_layer_irrigation()`.
  8 unit tests. Ported from `control/iot_irrigation/calibration_dong2024.py`.
- **Index of Agreement** (`testutil::index_of_agreement`): Willmott (1981) IA
  statistic. Ported from `control/soil_sensors/calibration_dong2020.py::compute_ia`.
- **Nash-Sutcliffe Efficiency** (`testutil::nash_sutcliffe`): NSE (Nash &
  Sutcliffe, 1970) for hydrological model evaluation.
- **Coefficient of determination** (`testutil::coefficient_of_determination`):
  SS-based R² (standard regression definition).
- **Wind speed conversion** (`eco::evapotranspiration::wind_speed_at_2m`):
  FAO-56 Eq. 47 — converts anemometer height to standard 2 m reference.
- **14 new integration tests** covering: sensor calibration end-to-end, IA/NSE
  validation, wind speed conversion, error type variants, `std::error::Error`
  trait compliance.
- **`validate_sensor_calibration` binary**: 21 checks validating SoilWatch 10
  calibration equation, irrigation recommendation, sensor performance criteria,
  and field demonstration results against `benchmark_dong2024.json`.
- **Phase 2 cross-validation harness**: `cross_validate` binary (Rust) and
  `scripts/cross_validate.py` (Python) produce JSON output for automated diff.
  **65/65 values match** within 1e-5 tolerance across atmospheric, solar,
  radiation, ET₀, Topp, SoilWatch 10, irrigation, statistical, sunshine Rs,
  Hargreaves ET₀, monthly G, low-level PM, water balance, and correction model
  computations.
- **3 wind speed unit tests** in `evapotranspiration.rs`: 10 m→2 m conversion,
  identity at 2 m, and monotonicity (lower at 2 m than above).
- **Solar radiation from sunshine** (`eco::evapotranspiration::solar_radiation_from_sunshine`):
  FAO-56 Eq. 35 — Ångström formula for Rs from sunshine hours.
- **Solar radiation from temperature** (`eco::evapotranspiration::solar_radiation_from_temperature`):
  FAO-56 Eq. 50 — Hargreaves method for Rs when sunshine data unavailable.
- **Soil heat flux** (`eco::evapotranspiration::soil_heat_flux_monthly`):
  FAO-56 Eq. 43 — monthly soil heat flux G.
- **Hargreaves ET₀** (`eco::evapotranspiration::hargreaves_et0`):
  FAO-56 Eq. 52 — simplified ET₀ requiring only temperature and Ra.
- **Crop coefficient database** (`eco::crop`): `CropType` enum with FAO-56
  Table 12 Kc values for 10 crops (corn, soybean, wheat, alfalfa, tomato,
  potato, sugar beet, dry bean, blueberry, turfgrass). `CropCoefficients`
  struct with `kc_ini`, `kc_mid`, `kc_end`, `root_depth_m`, `depletion_fraction`.
  `adjust_kc_for_climate()` implements FAO-56 Eq. 62. 7 unit tests.
- **Season simulation binary** (`src/bin/simulate_season.rs`): Full pipeline
  demonstration: crop Kc → soil properties → ET₀ → water balance → scheduling.
  Deterministic Michigan summer with Xorshift64 RNG. Compares rainfed vs smart
  irrigation strategies.
- **9 new ET₀ unit tests**: sunshine radiation, temperature radiation, monthly
  soil heat flux (warming + cooling), Hargreaves (range, temperature sensitivity,
  non-negative).
- **4 new integration tests**: crop Kc → water balance pipeline, tomato vs corn
  depletion rate, Hargreaves vs PM cross-check, sunshine radiation → ET₀.
- **GPU acceleration bridge** (`src/gpu/`): ToadStool/BarraCuda GPU bridge module
  documenting the architecture (eco→gpu→ops→shaders) and exposing evolution gaps.
- **`gpu::evolution_gaps`**: 11 structured `EvolutionGap` entries covering Tier A
  (kriging, fused reduce, batched ET₀, batched water balance, bootstrap CI),
  Tier B (pow_f64 precision, acos precision, ops module), and Tier C (Richards
  PDE, nonlinear solver, moving window).
- **Deepened barracuda stats integration**: `testutil` now wraps 5 barracuda
  primitives: `pearson_correlation` (existing), `spearman_correlation` (new),
  `bootstrap_ci` (new), `variance` (new), `std_dev` (new).
- **10 new integration tests**: Spearman rank correlation (monotonic, inverse,
  nonlinear vs Pearson), bootstrap RMSE confidence interval, variance/std_dev
  cross-validation, evolution gap catalog validation (catalogued, unique IDs,
  ET₀ gap, kriging gap).
- **Low-level `fao56_penman_monteith()`** (`eco::evapotranspiration`): Exposes the
  core FAO-56 Eq. 6 for use when intermediates are pre-computed (GPU buffers,
  batch workflows). `daily_et0()` now delegates to this internally.
- **Standalone water balance functions** (`eco::water_balance`):
  `total_available_water()`, `readily_available_water()`, `stress_coefficient()`,
  `daily_water_balance_step()` — match Python control API for direct comparison.
- **Correction models** (`eco::correction`): Pure Rust sensor calibration curve
  fitting — linear, quadratic, exponential, logarithmic models with analytical
  and log-linearized least squares. `fit_correction_equations()` replaces
  `scipy.optimize.curve_fit` with zero external dependencies. 8 unit tests.
- **Real data validation** (`bin/validate_real_data`): Computes ET₀ on real
  Open-Meteo Michigan weather data (6 stations, 918 station-days), cross-validates
  against Open-Meteo's own ET₀ (R² > 0.90), and runs water balance for 4 crop
  scenarios (blueberry, tomato, corn, reference grass) in both rainfed and irrigated
  modes. Mass balance verified for all 8 simulations. Water savings vs naive
  scheduling reported. 15/15 checks pass.
- **GPU orchestrators** (`gpu/`): Four domain-specific wrappers:
  - `gpu::et0::BatchedEt0` — N station-day ET₀ (CPU fallback — TS-001 blocks GPU)
  - `gpu::water_balance::BatchedWaterBalance` — season simulation with mass balance
  - `gpu::kriging::KrigingInterpolator` — ordinary kriging via `barracuda::ops::kriging_f64`
  - `gpu::reduce::SeasonalReducer` — GPU reductions via `barracuda::ops::fused_map_reduce_f64`
- **`ToadStool` issue tracker** (`gpu::evolution_gaps::TOADSTOOL_ISSUES`):
  4 documented issues for next handoff:
  - TS-001 (CRITICAL): `pow_f64` returns 0.0 for non-integer exponents
  - TS-002 (MEDIUM): No Rust `ops` module for `batched_elementwise_f64`
  - TS-003 (LOW): `acos_simple`/`sin_simple` approximation accuracy
  - TS-004 (HIGH): `FusedMapReduceF64` GPU dispatch buffer conflict for N≥1024
- **`KrigingInterpolator`** (`gpu::kriging`): Wraps `barracuda::ops::kriging_f64::KrigingF64`
  for proper ordinary kriging with variogram-based covariance and LU solve.
  `fit_variogram()` for empirical variogram fitting. Replaces IDW fallback.
- **`SeasonalReducer`** (`gpu::reduce`): Wraps
  `barracuda::ops::fused_map_reduce_f64::FusedMapReduceF64` for GPU-accelerated
  sum/max/min/sum-of-squares and derived stats. GPU dispatch for N≥1024, CPU
  fallback for smaller arrays. GPU path has known TS-004 buffer conflict.
- **`pollster` dev-dependency**: For async `WgpuDevice` creation in integration tests.
- **13 new integration tests**: Low-level PM matches daily_et0, standalone
  TAW/RAW/Ks cross-validation, correction model pipeline, GPU orchestrator
  mass conservation, kriging interpolation, seasonal stats, ToadStool issue
  documentation verification.
- **9 new integration tests**: GPU wiring — `KrigingInterpolator` at-sensor,
  midpoint, empty inputs, variogram fitting; `SeasonalReducer` sum, max/min,
  compute_stats, large array GPU dispatch (TS-004), empty.
- **Expanded cross-validation**: 53 → 65 values (added low-level PM, standalone
  water balance functions, correction model evaluation). All 65/65 match.

### Changed

- **CSV parser rewritten for streaming**: Replaced `std::fs::read_to_string`
  (buffers entire file) with `std::io::BufReader` (streams line-by-line).
  Added `parse_csv_reader<R: BufRead>()` for any `BufRead` source.
- **Columnar storage**: Replaced per-record `HashMap<String, f64>` with
  `Vec<Vec<f64>>` column-major layout. Column access via `column()` now returns
  `&[f64]` (zero-copy slice) instead of allocating a new `Vec<f64>`.
- **Idiomatic Rust throughout**:
  - `#[must_use]` on all pure functions and `Result`-returning public functions
  - `const fn` on `hydraulic_properties()`, `len()`, `is_empty()`,
    `num_columns()`, `passed()`, `total()`, `with_runoff_model()`
  - `f64::mul_add()` for FMA precision in all numerical expressions
  - `f64::midpoint()` for symmetric averages (FAO-56 Eqs. 12, 39)
  - `.to_radians()` for latitude conversion (was manual `* PI / 180.0`)
  - `Self::` in all match arms
  - `#[derive(Default)]` with `#[default]` attribute
  - `f64::from(u32)` replacing `as f64` casts
- **Validation binaries**: Load benchmark JSON with exact published inputs and
  tight tolerances. Example 18 Uccle ET₀ now matches within 0.0005 mm/day
  (was 0.5 mm tolerance with different inputs).
- **Runoff model alignment**: `RunoffModel::None` matches Python baseline's
  `RO = 0` (FAO-56 Ch. 8 default). Previous hardcoded `(P−20)×0.2` formula
  removed.
- **Error handling**: `csv_ts`, `validation`, and `testutil` modules migrated
  from `Result<T, String>` to `Result<T, AirSpringError>` with proper error
  variant taxonomy and `?` operator ergonomics.

### Fixed

- **`SandyCite` → `SandyClay`**: Typo in `SoilTexture` enum (public API).
  Regression test added.
- **189 → 0 clippy warnings**: Resolved all pedantic and nursery lints.
- **`cargo fmt`**: All files now pass `cargo fmt -- --check`.
- **`cargo doc`**: Zero warnings. Fixed unescaped `<f64>` HTML tag in doc
  comments and added backticks to all function parameter references.
- **Tolerance justification**: All tolerances documented with source (FAO-56
  Table rounding, Tetens coefficient approximation, etc.).

### Removed

- **`rayon` dependency**: Was declared but never used. Will be re-added when
  parallel computation (batched ET₀, spatial kriging) is implemented.
- **Phantom module references**: `eco::isotherms` and `eco::richards` were
  documented in `lib.rs` but never implemented. Removed from module docs.
- **Duplicated `check()` function**: Was copy-pasted across 4 validation
  binaries. Replaced with shared `ValidationRunner`.

### Quality Gates

| Check | Before | After |
|-------|--------|-------|
| `cargo fmt -- --check` | FAIL | PASS |
| `cargo clippy --pedantic --nursery` | 189 warnings | 0 warnings |
| `cargo doc --no-deps` | 1 warning | 0 warnings |
| `cargo test` | ~30 unit | 162 (94 unit + 68 integration) |
| Validation checks | 70/70 | 119/119 (8 binaries) |
| Cross-validation | N/A | 65/65 MATCH (Python↔Rust) |
| Library coverage | N/A | 96%+ (tarpaulin, excl. validation runner) |
| Error handling | `String` | `AirSpringError` enum |
| Lines per file (max) | N/A | 760 (under 1000 limit) |
| Zero unsafe | Yes | Yes |

## [0.1.0] - 2026-02-16

### Added

- Initial Rust validation crate with ET₀, soil moisture, water balance, CSV
  parser, and 4 validation binaries.
- 70/70 validation checks passing.
- Python/R control baselines: 142/142 PASS.
