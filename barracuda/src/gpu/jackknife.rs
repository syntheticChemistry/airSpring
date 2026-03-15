// SPDX-License-Identifier: AGPL-3.0-or-later
//! GPU-accelerated jackknife variance estimation.
//!
//! Wraps `barracuda::stats::jackknife::JackknifeMeanGpu` from `BarraCuda` S71.
//! Computes leave-one-out jackknife mean variance via dedicated WGSL shader.
//!
//! # Cross-Spring Provenance
//!
//! - **groundSpring**: Uncertainty quantification methodology
//! - **neuralSpring**: GPU dispatch pattern
//! - **`BarraCuda` S71**: `jackknife_mean_f64.wgsl` shader

use std::sync::Arc;

use barracuda::device::WgpuDevice;
use barracuda::stats::jackknife::{JackknifeMeanGpu, jackknife_mean_variance};

#[cfg(test)]
use super::device_info::try_f64_device;

/// Jackknife estimate result.
#[derive(Debug, Clone, Copy)]
pub struct JackknifeEstimate {
    /// Jackknife mean.
    pub mean: f64,
    /// Jackknife variance.
    pub variance: f64,
    /// Standard error (sqrt of variance).
    pub std_error: f64,
}

/// GPU-accelerated jackknife mean variance orchestrator.
///
/// Dispatches to `JackknifeMeanGpu` when a GPU engine is configured;
/// falls back to CPU otherwise.
pub struct GpuJackknife {
    gpu_engine: Option<JackknifeMeanGpu>,
}

impl std::fmt::Debug for GpuJackknife {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuJackknife")
            .field("gpu_engine", &self.gpu_engine.is_some())
            .finish()
    }
}

impl GpuJackknife {
    /// Create with GPU engine.
    ///
    /// # Errors
    ///
    /// Returns an error if `JackknifeMeanGpu` cannot be initialised.
    pub fn gpu(device: Arc<WgpuDevice>) -> crate::error::Result<Self> {
        let engine = JackknifeMeanGpu::new(device)?;
        Ok(Self {
            gpu_engine: Some(engine),
        })
    }

    /// Create with CPU fallback (always safe, no device needed).
    #[must_use]
    pub const fn cpu() -> Self {
        Self { gpu_engine: None }
    }

    /// Estimate jackknife mean and variance.
    ///
    /// # Errors
    ///
    /// Returns an error if data has fewer than 2 observations or GPU dispatch fails.
    pub fn estimate(&self, data: &[f64]) -> crate::error::Result<JackknifeEstimate> {
        if data.len() < 2 {
            return Err(crate::error::AirSpringError::InvalidInput(
                "jackknife requires at least 2 observations".into(),
            ));
        }
        if let Some(engine) = &self.gpu_engine {
            let result = engine.dispatch(data)?;
            Ok(JackknifeEstimate {
                mean: result.estimate,
                variance: result.variance,
                std_error: result.std_error,
            })
        } else {
            jackknife_cpu(data)
        }
    }
}

/// CPU fallback: leave-one-out jackknife mean variance.
fn jackknife_cpu(data: &[f64]) -> crate::error::Result<JackknifeEstimate> {
    jackknife_mean_variance(data)
        .map(|r| JackknifeEstimate {
            mean: r.estimate,
            variance: r.variance,
            std_error: r.std_error,
        })
        .ok_or_else(|| {
            crate::error::AirSpringError::InvalidInput(
                "jackknife requires at least 2 observations".into(),
            )
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
    fn test_small_sample() {
        let engine = GpuJackknife::cpu();
        let data = [1.0, 2.0, 3.0, 4.0, 5.0];
        let est = engine.estimate(&data).unwrap();
        assert!((est.mean - 3.0).abs() < 0.001);
        assert!(est.variance > 0.0);
        assert!(est.std_error > 0.0);
    }

    #[test]
    fn test_gpu_matches_cpu() {
        let Some(device) = try_device() else {
            eprintln!("SKIP: No GPU device for GpuJackknife");
            return;
        };
        let gpu_engine = GpuJackknife::gpu(device).unwrap();
        let cpu_engine = GpuJackknife::cpu();

        let data = [3.2, 3.5, 3.1, 3.8, 3.6, 3.3, 3.7, 3.4, 3.9, 3.0];

        let gpu_est = gpu_engine.estimate(&data).unwrap();
        let cpu_est = cpu_engine.estimate(&data).unwrap();

        assert!((gpu_est.mean - cpu_est.mean).abs() < 0.001);
        assert!((gpu_est.variance - cpu_est.variance).abs() < 0.001);
        assert!((gpu_est.std_error - cpu_est.std_error).abs() < 0.001);
    }

    #[test]
    fn test_empty_errors() {
        let engine = GpuJackknife::cpu();
        assert!(engine.estimate(&[]).is_err());
    }

    #[test]
    fn test_single_errors() {
        let engine = GpuJackknife::cpu();
        assert!(engine.estimate(&[1.0]).is_err());
    }
}
