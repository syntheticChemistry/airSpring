// SPDX-License-Identifier: AGPL-3.0-or-later
//! Validation and test utilities.
//!
//! Synthetic data generators for testing parsers, statistics, and validation
//! binaries. These produce deterministic data with known properties so that
//! computed results can be verified analytically.
//!
//! # Production use
//!
//! This module is NOT for production data ingestion. It exists to support:
//! - Unit tests in [`crate::io::csv_ts`]
//! - Validation binary `validate_iot`
//! - Integration tests in `tests/integration.rs`
//!
//! # Submodules
//!
//! - [`generators`] — Synthetic `IoT` sensor data
//! - [`stats`] — Validation metrics (RMSE, MBE, IA, NSE, R², Pearson, Spearman)
//! - [`bootstrap`] — Bootstrap confidence intervals

pub mod bootstrap;
pub mod generators;
pub mod stats;

pub use bootstrap::bootstrap_rmse;
pub use generators::generate_synthetic_iot_data;
pub use stats::{
    coefficient_of_determination, dot, hit_rate, index_of_agreement, l2_norm, mbe, mean,
    nash_sutcliffe, pearson_r, percentile, r_squared, rmse, spearman_r, std_deviation, variance,
};
