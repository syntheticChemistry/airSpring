// SPDX-License-Identifier: AGPL-3.0-or-later
//! GPU-accelerated Growing Degree Days (GDD) batch computation.
//!
//! Wraps `BatchedElementwiseF64` op 12 (Gdd) from `ToadStool` S79.
//! Uses `execute_with_aux` for `T_base`. CPU: `(T_mean - T_base).max(0.0)`.

use std::sync::Arc;

use barracuda::device::WgpuDevice;
use barracuda::ops::batched_elementwise_f64::{BatchedElementwiseF64, Op};

#[cfg(test)]
use super::device_info::try_f64_device;

/// Batched GDD GPU orchestrator.
///
/// Dispatches to `BatchedElementwiseF64` op 12 when a GPU engine is configured;
/// falls back to CPU otherwise.
pub struct BatchedGdd {
    gpu_engine: Option<BatchedElementwiseF64>,
}

impl std::fmt::Debug for BatchedGdd {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BatchedGdd")
            .field("gpu_engine", &self.gpu_engine.is_some())
            .finish()
    }
}

impl BatchedGdd {
    /// Create with GPU engine (dispatches to op 12).
    ///
    /// # Errors
    ///
    /// Returns an error if `BatchedElementwiseF64` cannot be initialised.
    pub fn gpu(device: Arc<WgpuDevice>) -> crate::error::Result<Self> {
        let engine = BatchedElementwiseF64::new(device)?;
        Ok(Self {
            gpu_engine: Some(engine),
        })
    }

    /// Create with CPU fallback (always safe, no device needed).
    #[must_use]
    pub const fn cpu() -> Self {
        Self { gpu_engine: None }
    }

    /// Compute GDD for a batch of mean temperatures.
    ///
    /// GDD = max(0, `T_mean` - `T_base`). Uses `execute_with_aux` for `T_base`.
    ///
    /// # Errors
    ///
    /// Returns an error if GPU dispatch fails.
    pub fn compute_gpu(&self, tmean_values: &[f64], tbase: f64) -> crate::error::Result<Vec<f64>> {
        if tmean_values.is_empty() {
            return Ok(Vec::new());
        }
        if let Some(engine) = &self.gpu_engine {
            Ok(engine.execute_with_aux(tmean_values, tmean_values.len(), Op::Gdd, tbase)?)
        } else {
            Ok(compute_gdd_cpu(tmean_values, tbase))
        }
    }
}

/// CPU fallback: GDD = max(0, `T_mean` - `T_base`).
#[must_use]
pub fn compute_gdd_cpu(tmean_values: &[f64], tbase: f64) -> Vec<f64> {
    tmean_values.iter().map(|&t| (t - tbase).max(0.0)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn try_device() -> Option<Arc<WgpuDevice>> {
        try_f64_device()
    }

    #[test]
    fn test_cpu_gdd_positive() {
        let engine = BatchedGdd::cpu();
        let result = engine.compute_gpu(&[25.0], 10.0).unwrap();
        assert_eq!(result.len(), 1);
        assert!((result[0] - 15.0).abs() < 0.001, "GDD(25,10)=15");
    }

    #[test]
    fn test_cpu_gdd_zero_below_base() {
        let engine = BatchedGdd::cpu();
        let result = engine.compute_gpu(&[5.0], 10.0).unwrap();
        assert_eq!(result.len(), 1);
        assert!(result[0].abs() < 0.001, "GDD(5,10)=0");
    }

    #[test]
    fn test_gpu_matches_cpu() {
        let Some(device) = try_device() else {
            eprintln!("SKIP: No GPU device for BatchedGdd");
            return;
        };
        let gpu_engine = BatchedGdd::gpu(device).unwrap();
        let cpu_engine = BatchedGdd::cpu();

        let tmeans = [5.0, 10.0, 15.0, 20.0, 25.0, 30.0];
        let tbase = 10.0;

        let gpu_result = gpu_engine.compute_gpu(&tmeans, tbase).unwrap();
        let cpu_result = cpu_engine.compute_gpu(&tmeans, tbase).unwrap();

        assert_eq!(gpu_result.len(), cpu_result.len());
        for (g, c) in gpu_result.iter().zip(&cpu_result) {
            assert!((g - c).abs() < 0.001, "GPU GDD={g:.4} vs CPU GDD={c:.4}");
        }
    }

    #[test]
    fn test_empty_batch() {
        let engine = BatchedGdd::cpu();
        let result = engine.compute_gpu(&[], 10.0).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_gdd_cpu_batch() {
        let cpu = compute_gdd_cpu(&[0.0, 10.0, 20.0], 10.0);
        assert_eq!(cpu, [0.0, 0.0, 10.0]);
    }
}
