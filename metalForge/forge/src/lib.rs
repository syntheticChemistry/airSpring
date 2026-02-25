// SPDX-License-Identifier: AGPL-3.0-or-later
//! airspring-forge — domain primitives for upstream absorption.
//!
//! This crate stages general-purpose statistical, regression, signal processing,
//! and hydrology functions from airSpring's validated pipeline.  Implementations
//! are pure Rust with zero external dependencies, structured to mirror the
//! `BarraCuda` module layout for direct absorption.
//!
//! # Absorption targets
//!
//! | Module | Upstream target | Status |
//! |--------|----------------|--------|
//! | [`metrics`] | `barracuda::stats::metrics` | Ready |
//! | [`regression`] | `barracuda::stats::regression` | Ready |
//! | [`moving_window_f64`] | `barracuda::ops::moving_window_stats_f64` | Ready |
//! | [`hydrology`] | `barracuda::ops::hydrology` | Ready |
//! | [`van_genuchten`] | `barracuda::pde::richards` | Ready (ABSORBED upstream) |
//! | [`isotherm`] | `barracuda::optimize` (curve fitting) | Ready |
//!
//! # Design philosophy
//!
//! Following hotSpring's Write → Validate → Handoff → Absorb → Lean pattern:
//! once `ToadStool` absorbs a module, airSpring rewires to `use barracuda::*`
//! and deletes the local code.
//!
//! # Provenance
//!
//! All functions validated against published benchmarks:
//! - Metrics: FAO-56 real data pipeline (918 station-days, R²=0.967)
//! - Metrics: Dong et al. (2020) soil sensor calibration (36/36 checks)
//! - Regression: Dong et al. (2020) correction equations (linear, quad, exp, log)
//! - Hydrology: FAO-56 (Allen et al. 1998), 918 station-days
//! - Moving window: validated against upstream GPU f32 path (wetSpring S28+)
//! - Van Genuchten: Carsel & Parrish (1988), HYDRUS baseline
//! - Isotherm: Kumari, Dong & Safferman (2025), Langmuir/Freundlich models
//! - Cross-validated: Python↔Rust 75/75 match within 1e-5

pub mod hydrology;
pub mod isotherm;
pub mod metrics;
pub mod moving_window_f64;
pub mod regression;
pub mod van_genuchten;

pub use metrics::ForgeError;

/// Convert a slice length to `f64` for use in statistical denominators.
///
/// For typical sample sizes (< 2^53 elements), the cast is exact.
#[inline]
#[allow(clippy::cast_precision_loss)]
const fn len_f64<T>(slice: &[T]) -> f64 {
    slice.len() as f64
}
