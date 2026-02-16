# Changelog

All notable changes to airSpring follow [Keep a Changelog](https://keepachangelog.com/).

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
- **BarraCUDA cross-validation**: Integration tests verify airSpring
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
- **GPU acceleration bridge** (`src/gpu/`): ToadStool/BarraCUDA GPU bridge module
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
