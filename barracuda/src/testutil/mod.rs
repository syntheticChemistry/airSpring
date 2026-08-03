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
#[cfg(test)]
pub mod env_guard;
pub mod generators;
pub mod stats;

pub use bootstrap::bootstrap_rmse;
#[cfg(test)]
pub use env_guard::EnvGuard;
pub use generators::generate_synthetic_iot_data;
pub use stats::{
    coefficient_of_determination, dot, hit_rate, index_of_agreement, l2_norm, mbe, mean,
    nash_sutcliffe, pearson_r, percentile, r_squared, rmse, spearman_r, std_deviation, variance,
};

/// Acquire an f64-capable GPU device or skip the test with a message.
///
/// Expands to a `let`-binding: the device is bound to the provided identifier.
///
/// ```rust,ignore
/// gpu_or_skip!(device);
/// // `device` is now an Arc<WgpuDevice>
/// ```
#[cfg(test)]
macro_rules! gpu_or_skip {
    ($dev:ident) => {
        let Some($dev) = $crate::gpu::device_info::try_f64_device() else {
            eprintln!("SKIP: No f64-capable GPU — {}", ::std::module_path!());
            return;
        };
    };
}

#[cfg(test)]
pub(crate) use gpu_or_skip;
