#![warn(clippy::pedantic)]
#![allow(
    clippy::module_name_repetitions,
    clippy::must_use_candidate,
    clippy::return_self_not_must_use
)]

//! airSpring `BarraCuda` — Ecological & Agricultural Science Pipelines
//!
//! Rust implementations validated against FAO-56, HYDRUS, and published
//! field data from Dr. Younsuk Dong (MSU Biosystems & Agricultural Engineering).
//!
//! # Track 1: Precision Agriculture
//! - [`eco::correction`] — Sensor correction curve fitting (linear, quadratic, exponential, logarithmic)
//! - [`eco::crop`] — FAO-56 Table 12 crop coefficient (`Kc`) database (10 crops) + climate adjustment
//! - [`eco::evapotranspiration`] — FAO-56 Penman-Monteith (low-level + high-level) + Hargreaves ET₀
//! - [`eco::sensor_calibration`] — `SoilWatch` 10 VWC calibration + irrigation recommendation
//! - [`eco::soil_moisture`] — Dielectric sensor calibration (Topp equation)
//! - [`eco::water_balance`] — Field-scale water budget (standalone + stateful APIs)
//!
//! # GPU Acceleration (all `ToadStool` issues RESOLVED — `0c477306`)
//! - [`gpu::et0`] — **GPU-first** batched ET₀ via `BatchedElementwiseF64::fao56_et0_batch()`
//! - [`gpu::water_balance`] — **GPU-step** + CPU season via `BatchedElementwiseF64::water_balance_batch()`
//! - [`gpu::kriging`] — Soil moisture spatial interpolation (`KrigingInterpolator` ↔ `KrigingF64`)
//! - [`gpu::reduce`] — **GPU** for N≥1024 (`SeasonalReducer` ↔ `FusedMapReduceF64`, TS-004 resolved)
//! - [`gpu::stream`] — `IoT` stream smoothing (`StreamSmoother` ↔ `MovingWindowStats`, wetSpring S28+)
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
//! - [`testutil`] — Synthetic data generators, `IA`, `NSE`, `RMSE`, `MBE`, R², Spearman, bootstrap CI
//!
//! # `BarraCuda` Integration
//!
//! Directly uses `barracuda` primitives for:
//! - `stats::pearson_correlation` → R² in [`testutil::r_squared`]
//! - `stats::spearman_correlation` → nonparametric validation in [`testutil::spearman_r`]
//! - `stats::bootstrap_ci` → uncertainty quantification in [`testutil::bootstrap_rmse`]
//! - `stats::std_dev` → cross-validation in integration tests
//! - `linalg::ridge::ridge_regression` → calibration regression in [`eco::correction::fit_ridge`]
//! - `ops::moving_window_stats` → `IoT` stream smoothing in [`gpu::stream::StreamSmoother`]
//! - `validation::ValidationHarness` → validation binaries

pub mod eco;
pub mod error;
pub mod gpu;
pub mod io;
pub mod testutil;
pub mod validation;

/// Convert a slice length to `f64` for use in statistical denominators.
///
/// For typical sample sizes (< 2^53 elements), the cast from `usize` to `f64`
/// is exact. This centralises the `as f64` cast so individual call sites
/// do not need `#[allow(clippy::cast_precision_loss)]`.
#[inline]
#[allow(clippy::cast_precision_loss)]
pub(crate) const fn len_f64<T>(slice: &[T]) -> f64 {
    slice.len() as f64
}
