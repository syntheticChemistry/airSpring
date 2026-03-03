// SPDX-License-Identifier: AGPL-3.0-or-later
//! Batched SCS-CN runoff computation — GPU-local via `local_elementwise.wgsl`.
//!
//! The SCS-CN equation `Q = (P − Ia)² / (P − Ia + S)` is embarrassingly parallel
//! across fields/events. This module provides a batched orchestrator with both
//! CPU fallback and GPU dispatch via `LocalElementwise` (f32 WGSL, op=0).
//!
//! # Cross-Spring Provenance
//!
//! | Primitive | Origin | Status |
//! |-----------|--------|--------|
//! | SCS-CN equation | USDA-SCS (1972) NEH-4 | **GPU-local** (f32 WGSL) |
//! | AMC adjustment | Chow et al. (1988) | CPU fallback |
//! | GPU dispatch | `local_elementwise.wgsl` op=0 | **Live** (v0.6.8) |
//!
//! # GPU Path
//!
//! SCS-CN runs on GPU via `LocalElementwise` (wgpu direct, f32 precision).
//! `ToadStool` will absorb into canonical f64 shader for full precision.

use std::sync::Arc;

use barracuda::device::WgpuDevice;

use crate::eco::runoff;
use crate::gpu::local_dispatch::{LocalElementwise, LocalOp};

/// Single runoff computation request.
#[derive(Debug, Clone, Copy)]
pub struct RunoffInput {
    /// Precipitation (mm).
    pub precip_mm: f64,
    /// Curve number (1–100).
    pub cn: f64,
    /// Initial abstraction ratio (default 0.2).
    pub ia_ratio: f64,
}

/// Result of batched runoff computation.
#[derive(Debug)]
pub struct BatchedRunoffResult {
    /// Computed runoff Q (mm) for each input.
    pub runoff_mm: Vec<f64>,
    /// Potential retention S (mm) for each input.
    pub retention_mm: Vec<f64>,
}

/// Batched SCS-CN runoff orchestrator.
///
/// CPU path uses `eco::runoff` directly. GPU path dispatches via
/// `LocalElementwise` (f32 WGSL shader, pending `ToadStool` absorption to f64).
#[derive(Debug)]
pub struct BatchedRunoff;

/// GPU-backed SCS-CN runoff dispatcher.
///
/// Uses `local_elementwise.wgsl` op 0 for GPU-parallel computation.
/// f32 precision on GPU; `ToadStool` absorption upgrades to f64.
pub struct GpuRunoff {
    dispatcher: LocalElementwise,
}

impl std::fmt::Debug for GpuRunoff {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuRunoff").finish()
    }
}

impl GpuRunoff {
    /// Create a GPU-backed runoff solver.
    ///
    /// # Errors
    ///
    /// Returns an error if WGSL shader compilation fails.
    pub fn new(device: Arc<WgpuDevice>) -> crate::error::Result<Self> {
        Ok(Self {
            dispatcher: LocalElementwise::new(device)?,
        })
    }

    /// Batch compute SCS-CN runoff on GPU.
    ///
    /// # Errors
    ///
    /// Returns an error if GPU dispatch fails.
    pub fn compute(&self, inputs: &[RunoffInput]) -> crate::error::Result<Vec<f64>> {
        let a: Vec<f64> = inputs.iter().map(|i| i.precip_mm).collect();
        let b: Vec<f64> = inputs.iter().map(|i| i.cn).collect();
        let c: Vec<f64> = inputs.iter().map(|i| i.ia_ratio).collect();
        self.dispatcher.dispatch(LocalOp::ScsCnRunoff, &a, &b, &c)
    }

    /// Batch compute with standard Ia ratio (0.2) and uniform CN.
    ///
    /// # Errors
    ///
    /// Returns an error if GPU dispatch fails.
    pub fn compute_uniform(&self, precip_mm: &[f64], cn: f64) -> crate::error::Result<Vec<f64>> {
        let b = vec![cn; precip_mm.len()];
        let c = vec![0.2; precip_mm.len()];
        self.dispatcher
            .dispatch(LocalOp::ScsCnRunoff, precip_mm, &b, &c)
    }
}

impl BatchedRunoff {
    /// Batch compute SCS-CN runoff for N precipitation events.
    ///
    /// Each input specifies precipitation, curve number, and Ia ratio.
    #[must_use]
    pub fn compute(inputs: &[RunoffInput]) -> BatchedRunoffResult {
        let mut runoff_mm = Vec::with_capacity(inputs.len());
        let mut retention_mm = Vec::with_capacity(inputs.len());

        for inp in inputs {
            let s = runoff::potential_retention(inp.cn);
            let q = runoff::scs_cn_runoff(inp.precip_mm, inp.cn, inp.ia_ratio);
            runoff_mm.push(q);
            retention_mm.push(s);
        }

        BatchedRunoffResult {
            runoff_mm,
            retention_mm,
        }
    }

    /// Batch compute with standard Ia ratio (0.2) and uniform CN.
    #[must_use]
    pub fn compute_uniform(precip_mm: &[f64], cn: f64) -> Vec<f64> {
        precip_mm
            .iter()
            .map(|&p| runoff::scs_cn_runoff_standard(p, cn))
            .collect()
    }

    /// Batch compute with AMC-I adjustment (dry conditions).
    #[must_use]
    pub fn compute_amc_dry(precip_mm: &[f64], cn_ii: f64) -> Vec<f64> {
        let cn_i = runoff::amc_cn_dry(cn_ii);
        Self::compute_uniform(precip_mm, cn_i)
    }

    /// Batch compute with AMC-III adjustment (wet conditions).
    #[must_use]
    pub fn compute_amc_wet(precip_mm: &[f64], cn_ii: f64) -> Vec<f64> {
        let cn_iii = runoff::amc_cn_wet(cn_ii);
        Self::compute_uniform(precip_mm, cn_iii)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_matches_cpu() {
        let device = crate::gpu::device_info::try_f64_device();
        let Some(device) = device else {
            eprintln!("SKIP: no GPU for GpuRunoff");
            return;
        };
        let gpu_solver = GpuRunoff::new(device).unwrap();
        let inputs = vec![
            RunoffInput { precip_mm: 50.0, cn: 75.0, ia_ratio: 0.2 },
            RunoffInput { precip_mm: 100.0, cn: 85.0, ia_ratio: 0.2 },
            RunoffInput { precip_mm: 25.0, cn: 65.0, ia_ratio: 0.2 },
            RunoffInput { precip_mm: 0.0, cn: 80.0, ia_ratio: 0.2 },
        ];
        let gpu = gpu_solver.compute(&inputs).unwrap();
        let cpu = BatchedRunoff::compute(&inputs);

        for (i, (g, c)) in gpu.iter().zip(&cpu.runoff_mm).enumerate() {
            let tol = c.abs().mul_add(1e-3, 1e-4);
            assert!((g - c).abs() < tol, "Runoff[{i}] GPU={g:.4} CPU={c:.4}");
        }
    }

    #[test]
    fn test_batch_compute() {
        let inputs = vec![
            RunoffInput { precip_mm: 50.0, cn: 75.0, ia_ratio: 0.2 },
            RunoffInput { precip_mm: 100.0, cn: 85.0, ia_ratio: 0.2 },
            RunoffInput { precip_mm: 25.0, cn: 65.0, ia_ratio: 0.2 },
        ];
        let result = BatchedRunoff::compute(&inputs);
        assert_eq!(result.runoff_mm.len(), 3);
        assert!(result.runoff_mm[1] > result.runoff_mm[0]);
        for &q in &result.runoff_mm {
            assert!(q >= 0.0);
        }
    }

    #[test]
    fn test_uniform_monotonic() {
        let precips: Vec<f64> = (0..20).map(|i| f64::from(i) * 10.0).collect();
        let qs = BatchedRunoff::compute_uniform(&precips, 80.0);
        for w in qs.windows(2) {
            assert!(w[1] >= w[0] - 1e-10, "Q not monotonic");
        }
    }

    #[test]
    fn test_amc_ordering() {
        let precips = [50.0, 100.0];
        let dry = BatchedRunoff::compute_amc_dry(&precips, 75.0);
        let normal = BatchedRunoff::compute_uniform(&precips, 75.0);
        let wet = BatchedRunoff::compute_amc_wet(&precips, 75.0);
        for i in 0..precips.len() {
            assert!(dry[i] <= normal[i] + 1e-6, "dry should produce less runoff");
            assert!(normal[i] <= wet[i] + 1e-6, "wet should produce more runoff");
        }
    }

    #[test]
    fn test_empty() {
        let result = BatchedRunoff::compute(&[]);
        assert!(result.runoff_mm.is_empty());
    }
}
