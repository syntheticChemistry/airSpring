//! airSpring `BarraCUDA` — Ecological & Agricultural Science Pipelines
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
//! # GPU Acceleration
//! - [`gpu::et0`] — Batched ET₀ orchestrator (CPU fallback, GPU blocked on `ToadStool` `pow_f64`)
//! - [`gpu::water_balance`] — Batched season simulation with mass balance tracking
//! - [`gpu::kriging`] — Soil moisture spatial interpolation (`KrigingInterpolator` ↔ `KrigingF64`)
//! - [`gpu::reduce`] — Seasonal aggregation statistics (`SeasonalReducer` ↔ `FusedMapReduceF64`)
//! - [`gpu::evolution_gaps`] — Living roadmap + `ToadStool` issue tracker (TS-001/002/003/004)
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
//! # `BarraCUDA` Stats Integration
//!
//! Directly uses [`barracuda::stats`] for:
//! - `pearson_correlation` → R² in [`testutil::r_squared`]
//! - `spearman_correlation` → nonparametric validation in [`testutil::spearman_r`]
//! - `bootstrap_ci` → uncertainty quantification in [`testutil::bootstrap_rmse`]
//! - `std_dev` → cross-validation in integration tests

pub mod eco;
pub mod error;
pub mod gpu;
pub mod io;
pub mod testutil;
pub mod validation;
