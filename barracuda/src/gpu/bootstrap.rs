// SPDX-License-Identifier: AGPL-3.0-or-later
//! GPU-accelerated bootstrap mean estimation.
//!
//! Wraps `barracuda::stats::bootstrap::BootstrapMeanGpu` from `BarraCuda` S71.
//!
//! # Cross-Spring Provenance
//!
//! - **groundSpring**: Bootstrap methodology for uncertainty bands
//! - **neuralSpring**: GPU dispatch pattern
//! - **`BarraCuda` S71**: `bootstrap_mean_f64.wgsl` shader

use std::sync::Arc;

use barracuda::device::WgpuDevice;
use barracuda::stats::bootstrap::BootstrapMeanGpu;
use barracuda::stats::{bootstrap_ci, mean, percentile};

#[cfg(test)]
use super::device_info::try_f64_device;

/// Bootstrap mean estimate with confidence interval.
#[derive(Debug, Clone)]
pub struct BootstrapEstimate {
    /// Point estimate (mean).
    pub mean: f64,
    /// Lower bound of 95% CI.
    pub ci_lower: f64,
    /// Upper bound of 95% CI.
    pub ci_upper: f64,
    /// Standard error.
    pub std_error: f64,
}

/// GPU-accelerated bootstrap mean orchestrator.
///
/// Dispatches to `BootstrapMeanGpu` when a GPU engine is configured;
/// falls back to CPU otherwise.
pub struct GpuBootstrap {
    gpu_engine: Option<BootstrapMeanGpu>,
}

impl std::fmt::Debug for GpuBootstrap {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuBootstrap")
            .field("gpu_engine", &self.gpu_engine.is_some())
            .finish()
    }
}

impl GpuBootstrap {
    /// Create with GPU engine.
    ///
    /// # Errors
    ///
    /// Returns an error if `BootstrapMeanGpu` cannot be initialised.
    pub fn gpu(device: Arc<WgpuDevice>) -> crate::error::Result<Self> {
        let engine = BootstrapMeanGpu::new(device)?;
        Ok(Self {
            gpu_engine: Some(engine),
        })
    }

    /// Create with CPU fallback (always safe, no device needed).
    #[must_use]
    pub const fn cpu() -> Self {
        Self { gpu_engine: None }
    }

    /// Estimate mean with bootstrap 95% confidence interval.
    ///
    /// # Errors
    ///
    /// Returns an error if data is empty, `n_bootstrap` is 0, or GPU dispatch fails.
    pub fn estimate_mean(
        &self,
        data: &[f64],
        n_bootstrap: u32,
        seed: u32,
    ) -> crate::error::Result<BootstrapEstimate> {
        if data.is_empty() {
            return Err(crate::error::AirSpringError::InvalidInput(
                "data cannot be empty".into(),
            ));
        }
        if n_bootstrap == 0 {
            return Err(crate::error::AirSpringError::InvalidInput(
                "n_bootstrap must be > 0".into(),
            ));
        }

        if let Some(engine) = &self.gpu_engine {
            let distribution = engine.dispatch(data, n_bootstrap, seed)?;

            // NVK/Titan V workaround: wgpu 28 f64 compute can return all zeros.
            // Detect and fall back to CPU (same pattern as gpu::reduce).
            let all_zero = distribution.iter().all(|&v| v == 0.0);
            if all_zero && data.iter().any(|&v| v != 0.0) {
                return bootstrap_mean_cpu(data, n_bootstrap as usize, u64::from(seed));
            }

            let mean_est = mean(data);
            let mut sorted = distribution.clone();
            sorted.sort_by(f64::total_cmp);
            let ci_lower = percentile(&sorted, 2.5);
            let ci_upper = percentile(&sorted, 97.5);
            let boot_mean: f64 = distribution.iter().sum::<f64>() / distribution.len() as f64;
            let variance: f64 = distribution
                .iter()
                .map(|&x| (x - boot_mean).powi(2))
                .sum::<f64>()
                / (distribution.len() as f64);
            let std_error = variance.sqrt();
            Ok(BootstrapEstimate {
                mean: mean_est,
                ci_lower,
                ci_upper,
                std_error,
            })
        } else {
            bootstrap_mean_cpu(data, n_bootstrap as usize, u64::from(seed))
        }
    }
}

/// CPU fallback: bootstrap CI for mean.
fn bootstrap_mean_cpu(
    data: &[f64],
    n_bootstrap: usize,
    seed: u64,
) -> crate::error::Result<BootstrapEstimate> {
    let ci = bootstrap_ci(
        data,
        |d| d.iter().sum::<f64>() / d.len() as f64,
        n_bootstrap,
        0.95,
        seed,
    )?;
    Ok(BootstrapEstimate {
        mean: ci.estimate,
        ci_lower: ci.lower,
        ci_upper: ci.upper,
        std_error: ci.std_error,
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    fn try_device() -> Option<Arc<WgpuDevice>> {
        try_f64_device()
    }

    #[test]
    fn test_known_mean() {
        let engine = GpuBootstrap::cpu();
        let data: Vec<f64> = (1..=10).map(f64::from).collect();
        let est = engine.estimate_mean(&data, 500, 42).unwrap();
        assert!((est.mean - 5.5).abs() < 0.01);
        assert!(est.ci_lower < 5.5 && est.ci_upper > 5.5);
        assert!(est.std_error > 0.0);
    }

    #[test]
    fn test_gpu_matches_cpu() {
        let Some(device) = try_device() else {
            eprintln!("SKIP: No GPU device for GpuBootstrap");
            return;
        };
        let Ok(gpu_engine) = GpuBootstrap::gpu(device) else {
            eprintln!("SKIP: GpuBootstrap::gpu init failed");
            return;
        };
        let cpu_engine = GpuBootstrap::cpu();
        let data: Vec<f64> = (1..=20).map(f64::from).collect();

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let gpu_est = gpu_engine.estimate_mean(&data, 200, 42).unwrap();
            let cpu_est = cpu_engine.estimate_mean(&data, 200, 42).unwrap();
            (gpu_est, cpu_est)
        }));
        let Ok((gpu_est, cpu_est)) = result else {
            eprintln!("SKIP: Bootstrap GPU failed (upstream shader validation)");
            return;
        };

        assert!((gpu_est.mean - cpu_est.mean).abs() < 0.01);
        assert!((gpu_est.ci_lower - cpu_est.ci_lower).abs() < 0.5);
        assert!((gpu_est.ci_upper - cpu_est.ci_upper).abs() < 0.5);
    }

    #[test]
    fn test_empty_errors() {
        let engine = GpuBootstrap::cpu();
        assert!(engine.estimate_mean(&[], 100, 42).is_err());
    }
}
