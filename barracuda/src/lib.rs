// SPDX-License-Identifier: AGPL-3.0-or-later
#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used, clippy::expect_used)]
#![allow(clippy::cast_precision_loss)]

//! airSpring `BarraCuda` — Ecological & Agricultural Science Pipelines
//!
//! Rust implementations validated against FAO-56, HYDRUS, and published
//! field data from Dr. Younsuk Dong (MSU Biosystems & Agricultural Engineering).
//!
//! # Precision Agriculture
//! - [`eco::correction`] — Sensor correction curve fitting (linear, quadratic, exponential, logarithmic)
//! - [`eco::crop`] — FAO-56 Table 12 crop coefficient (`Kc`) database (10 crops) + climate adjustment
//! - [`eco::dual_kc`] — FAO-56 Ch 7+11 dual Kc (Kcb + Ke), cover crops, no-till mulch
//! - [`eco::evapotranspiration`] — FAO-56 Penman-Monteith (low-level + high-level) + Hargreaves + Priestley-Taylor
//! - [`eco::simple_et0`] — Simplified ET₀ methods: Makkink, Turc, Hamon, Blaney-Criddle
//! - [`eco::isotherm`] — Langmuir and Freundlich isotherm fitting for biochar adsorption
//! - [`eco::sensor_calibration`] — `SoilWatch` 10 VWC calibration + irrigation recommendation
//! - [`eco::soil_moisture`] — Dielectric sensor calibration (Topp equation)
//! - [`eco::water_balance`] — Field-scale water budget (standalone + stateful APIs)
//! - [`eco::richards`] — 1D Richards equation solver (van Genuchten-Mualem hydraulics)
//!
//! # GPU Acceleration
//!
//! All GPU modules use `BarraCuda` f64-precision primitives with CPU fallback.
//!
//! - [`gpu::device_info`] — Precision-aware device probing, `Fp64Strategy`, cross-spring provenance
//! - [`gpu::et0`] — Batched ET₀ via `BatchedElementwiseF64::fao56_et0_batch()`
//! - [`gpu::water_balance`] — GPU-step + CPU season via `BatchedElementwiseF64::water_balance_batch()`
//! - [`gpu::kriging`] — Soil moisture spatial interpolation (`KrigingInterpolator` ↔ `KrigingF64`)
//! - [`gpu::reduce`] — GPU for N≥1024 (`SeasonalReducer` ↔ `FusedMapReduceF64`)
//! - [`gpu::stream`] — `IoT` stream smoothing (`StreamSmoother` ↔ `MovingWindowStats`)
//! - [`gpu::richards`] — 1D Richards PDE (`BatchedRichards` ↔ `pde::richards::solve_richards`)
//! - [`gpu::isotherm`] — Batch isotherm fitting (`fit_*_nm` ↔ `optimize::nelder_mead`)
//! - [`gpu::atlas_stream`] — Multi-station streaming with [`gpu::atlas_stream::MonitoredAtlasStream`] drift detection
//! - [`gpu::evolution_gaps`] — Living roadmap
//!
//! # Immunological Anderson (Paper 12)
//! - [`eco::tissue`] — Tissue diversity profiling: Pielou evenness → Anderson disorder W
//! - [`eco::cytokine`] — `CytokineBrain`: 3-head Nautilus reservoir (IL-31 propagation,
//!   tissue disorder, barrier state) for AD flare regime prediction
//!
//! # Nautilus Brain
//! - [`nautilus`] — Evolutionary reservoir computing via `bingocube-nautilus` for agricultural
//!   regime prediction (ET₀, soil moisture, crop stress), with drift detection and cross-station
//!   shell transfer
//!
//! # I/O
//! - [`io::csv_ts`] — Time series CSV streaming parser for `IoT` sensor data
//!
//! # Inter-Primal Communication (IPC)
//! - [`ipc::provenance`] — Provenance trio integration via biomeOS `capability.call`
//!   (rhizoCrypt + loamSpine + sweetGrass) with graceful degradation
//!
//! # Error Handling
//! - [`error`] — [`error::AirSpringError`] unified error type
//!
//! # Validation & Testing
//! - [`validation`] — Shared infrastructure for validation binaries
//! - [`tolerances`] — Centralized validation tolerances (`barracuda::tolerances` pattern)
//! - [`testutil`] — Synthetic data generators, `IA`, `NSE`, `RMSE`, `MBE`, R², Spearman, bootstrap CI
//!
//! # `BarraCuda` Primitives Used
//!
//! Capabilities consumed from the shared `barracuda` substrate (discovered at build time):
//! - **Statistics**: `pearson_correlation`, `spearman_correlation`, `bootstrap_ci`, `std_dev`, `norm_ppf`
//! - **Linear algebra**: `ridge_regression` → sensor calibration
//! - **GPU ops**: `batched_elementwise_f64` (ET₀, WB, VG, Thornthwaite, GDD, pedotransfer),
//!   `fused_map_reduce_f64` (seasonal stats), `kriging_f64` (spatial interpolation),
//!   `moving_window_stats` (`IoT` smoothing), `jackknife_mean`, `bootstrap_mean`, `diversity_fusion`
//! - **PDE solvers**: `pde::richards::solve_richards` → upstream Richards equation
//! - **Optimizers**: `nelder_mead` (isotherm fitting), `brent` (VG pressure head inversion)
//! - **Validation**: `ValidationHarness`, `tolerances::Tolerance`
//! - **Nautilus**: `bingocube-nautilus` (evolutionary reservoir computing, drift monitoring)

pub mod biomeos;
pub mod data;
pub mod eco;
pub mod error;
pub mod gpu;
pub mod io;
pub mod ipc;
pub mod nautilus;
#[cfg(feature = "npu")]
pub mod npu;
pub mod primal_science;
pub mod rpc;
#[cfg(any(test, feature = "testutil"))]
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
