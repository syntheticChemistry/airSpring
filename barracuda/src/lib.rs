// SPDX-License-Identifier: AGPL-3.0-or-later

//! airSpring `BarraCuda` — Ecological & Agricultural Science Pipelines
//!
//! Rust implementations validated against FAO-56, HYDRUS, and published
//! field data from Dr. Younsuk Dong (MSU Biosystems & Agricultural Engineering).
//!
//! # Track 1: Precision Agriculture
//! - [`eco::correction`] — Sensor correction curve fitting (linear, quadratic, exponential, logarithmic)
//! - [`eco::crop`] — FAO-56 Table 12 crop coefficient (`Kc`) database (10 crops) + climate adjustment
//! - [`eco::dual_kc`] — FAO-56 Ch 7+11 dual Kc (Kcb + Ke), cover crops, no-till mulch
//! - [`eco::evapotranspiration`] — FAO-56 Penman-Monteith (low-level + high-level) + Hargreaves ET₀
//! - [`eco::isotherm`] — Langmuir and Freundlich isotherm fitting for biochar adsorption
//! - [`eco::sensor_calibration`] — `SoilWatch` 10 VWC calibration + irrigation recommendation
//! - [`eco::soil_moisture`] — Dielectric sensor calibration (Topp equation)
//! - [`eco::water_balance`] — Field-scale water budget (standalone + stateful APIs)
//! - [`eco::richards`] — 1D Richards equation solver (van Genuchten-Mualem hydraulics)
//!
//! # GPU Acceleration (all `ToadStool` issues RESOLVED — S54, Feb 2026)
//!
//! Cross-spring shader provenance: hotSpring precision math (`pow_f64`,
//! `acos_f64`, `math_f64.wgsl`), wetSpring bio primitives (`moving_window`,
//! `kriging_f64`), neuralSpring optimizer (`nelder_mead`, `ValidationHarness`).
//!
//! - [`gpu::et0`] — **GPU-first** batched ET₀ via `BatchedElementwiseF64::fao56_et0_batch()`
//! - [`gpu::water_balance`] — **GPU-step** + CPU season via `BatchedElementwiseF64::water_balance_batch()`
//! - [`gpu::kriging`] — Soil moisture spatial interpolation (`KrigingInterpolator` ↔ `KrigingF64`)
//! - [`gpu::reduce`] — **GPU** for N≥1024 (`SeasonalReducer` ↔ `FusedMapReduceF64`, TS-004 S54)
//! - [`gpu::stream`] — `IoT` stream smoothing (`StreamSmoother` ↔ `MovingWindowStats`)
//! - [`gpu::richards`] — 1D Richards PDE (`BatchedRichards` ↔ `pde::richards::solve_richards`)
//! - [`gpu::isotherm`] — Batch isotherm fitting (`fit_*_nm` ↔ `optimize::nelder_mead`)
//! - [`gpu::evolution_gaps`] — Living roadmap, 4/4 `ToadStool` issues resolved
//!
//! # I/O
//! - [`io::csv_ts`] — Time series CSV streaming parser for `IoT` sensor data
//!
//! # Error Handling
//! - [`error`] — [`error::AirSpringError`] unified error type (replaces `String` errors)
//!
//! # Validation & Testing
//! - [`validation`] — Shared infrastructure for hotSpring-pattern validation binaries
//! - [`tolerances`] — Centralized validation tolerances (`barracuda::tolerances` pattern, S52)
//! - [`testutil`] — Synthetic data generators, `IA`, `NSE`, `RMSE`, `MBE`, R², Spearman, bootstrap CI
//!
//! # `BarraCuda` Integration (`ToadStool` S62+, 650+ WGSL shaders)
//!
//! Directly uses `barracuda` primitives for:
//! - `tolerances::Tolerance` + `check()` → centralized validation (S52, neuralSpring pattern)
//! - `stats::pearson_correlation` → R² in [`testutil::r_squared`]
//! - `stats::spearman_correlation` → nonparametric validation in [`testutil::spearman_r`]
//! - `stats::bootstrap_ci` → uncertainty quantification in [`testutil::bootstrap_rmse`]
//! - `stats::std_dev` → cross-validation in integration tests
//! - `linalg::ridge::ridge_regression` → calibration regression in [`eco::correction::fit_ridge`]
//! - `ops::batched_elementwise_f64` → GPU ET₀ + water balance (hotSpring `math_f64.wgsl`)
//! - `ops::fused_map_reduce_f64` → seasonal stats (wetSpring Shannon/Simpson, TS-004 S54)
//! - `ops::kriging_f64` → spatial interpolation (wetSpring geostatistics)
//! - `ops::moving_window_stats` → `IoT` stream smoothing (wetSpring monitoring)
//! - `pde::richards::solve_richards` → upstream Richards PDE (airSpring → `ToadStool` S40)
//! - `optimize::nelder_mead` → nonlinear isotherm fitting (`neuralSpring` S62)
//! - `validation::ValidationHarness` → 16 validation binaries (`neuralSpring` → `ToadStool` S59)

pub mod eco;
pub mod error;
pub mod gpu;
pub mod io;
pub mod testutil;
pub mod tolerances;
pub mod validation;

/// Convert a slice length to `f64` for use in statistical denominators.
///
/// For typical sample sizes (< 2^53 elements), the cast from `usize` to `f64`
/// is exact.
#[inline]
pub(crate) const fn len_f64<T>(slice: &[T]) -> f64 {
    slice.len() as f64
}
