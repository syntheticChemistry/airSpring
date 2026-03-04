// SPDX-License-Identifier: AGPL-3.0-or-later
//! GPU-accelerated Green-Ampt infiltration via batched Brent root-finding.
//!
//! Solves the implicit Green-Ampt equation `F − Ks·t − ψ·Δθ·ln(1 + F/(ψ·Δθ)) = 0`
//! for many (time, soil) pairs simultaneously on GPU using `BrentGpu::solve_green_ampt`.
//!
//! # Cross-Spring Provenance
//!
//! | Primitive | Origin | Shader |
//! |-----------|--------|--------|
//! | `BrentGpu` Green-Ampt | airSpring V045 → S83 | `brent_f64.wgsl` |
//! | Implicit GA equation | Green & Ampt (1911) | hotSpring precision math |
//! | Rawls soil parameters | Rawls et al. (1983) | CPU constants |

use std::sync::Arc;

use barracuda::device::WgpuDevice;
use barracuda::optimize::brent_gpu::BrentGpu;

use crate::eco::infiltration::{self, GreenAmptParams};

#[cfg(test)]
use super::device_info::try_f64_device;

/// Batched GPU Green-Ampt infiltration orchestrator.
///
/// Uses `BrentGpu::solve_green_ampt()` to solve the implicit GA cumulative
/// infiltration equation for many time steps in parallel.
pub struct BatchedInfiltration {
    device: Arc<WgpuDevice>,
}

impl std::fmt::Debug for BatchedInfiltration {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BatchedInfiltration")
            .field("device", &"Arc<WgpuDevice>")
            .finish()
    }
}

impl BatchedInfiltration {
    /// Create a GPU-backed infiltration solver.
    #[must_use]
    pub const fn new(device: Arc<WgpuDevice>) -> Self {
        Self { device }
    }

    /// Batch compute cumulative infiltration F(t) for multiple time values.
    ///
    /// The WGSL `green_ampt_residual` computes `t(F) = F/Ks − ψΔθ/Ks · ln(1 + F/ψΔθ)`,
    /// then Brent finds F such that `t(F) = target_time`.
    ///
    /// # Errors
    ///
    /// Returns an error if the GPU Brent dispatch fails.
    pub fn cumulative_gpu(
        &self,
        params: &GreenAmptParams,
        times_hr: &[f64],
    ) -> crate::error::Result<Vec<f64>> {
        if times_hr.is_empty() {
            return Ok(Vec::new());
        }

        let brent = BrentGpu::new(Arc::clone(&self.device), 100, 1e-10)?;

        let psi_dt = params.psi_cm * params.delta_theta;
        let ks = params.ks_cm_hr;

        let lower = vec![1e-8; times_hr.len()];
        let upper: Vec<f64> = times_hr
            .iter()
            .map(|&t| {
                ks.mul_add(t, (2.0 * ks * psi_dt * t).sqrt())
                    .mul_add(3.0, 1.0)
            })
            .collect();

        let result = brent.solve_green_ampt(&lower, &upper, times_hr, ks, psi_dt)?;
        Ok(result.roots)
    }

    /// Batch compute infiltration rate f(t) from cumulative F values.
    ///
    /// `f = Ks × (1 + ψΔθ/F)` — pure CPU arithmetic on GPU-produced F values.
    #[must_use]
    pub fn rate_from_cumulative(params: &GreenAmptParams, f_cumulative: &[f64]) -> Vec<f64> {
        f_cumulative
            .iter()
            .map(|&f_cm| infiltration::infiltration_rate(params, f_cm))
            .collect()
    }

    /// Batch compute F(t) and f(t) series on GPU.
    ///
    /// # Errors
    ///
    /// Returns an error if GPU Brent dispatch fails.
    pub fn series_gpu(
        &self,
        params: &GreenAmptParams,
        times_hr: &[f64],
    ) -> crate::error::Result<Vec<(f64, f64)>> {
        let f_values = self.cumulative_gpu(params, times_hr)?;
        let rates = Self::rate_from_cumulative(params, &f_values);
        Ok(f_values.into_iter().zip(rates).collect())
    }
}

/// CPU fallback for batched cumulative infiltration.
#[must_use]
pub fn cumulative_cpu(params: &GreenAmptParams, times_hr: &[f64]) -> Vec<f64> {
    times_hr
        .iter()
        .map(|&t| infiltration::cumulative_infiltration(params, t))
        .collect()
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    fn try_device() -> Option<Arc<WgpuDevice>> {
        try_f64_device()
    }

    #[test]
    fn test_gpu_matches_cpu_sandy_loam() {
        let Some(device) = try_device() else {
            eprintln!("SKIP: No GPU device for BatchedInfiltration");
            return;
        };
        let solver = BatchedInfiltration::new(device);
        let params = GreenAmptParams {
            delta_theta: 0.312,
            ..GreenAmptParams::SANDY_LOAM
        };
        let times = [0.1, 0.5, 1.0, 2.0, 4.0, 8.0];

        let gpu = solver.cumulative_gpu(&params, &times).unwrap();
        let cpu = cumulative_cpu(&params, &times);

        assert_eq!(gpu.len(), cpu.len());
        for (i, (g, c)) in gpu.iter().zip(&cpu).enumerate() {
            assert!((g - c).abs() < 0.5, "F[{i}] GPU={g:.4} vs CPU={c:.4}");
        }
    }

    #[test]
    fn test_gpu_matches_cpu_clay() {
        let Some(device) = try_device() else {
            eprintln!("SKIP: No GPU device for BatchedInfiltration (clay)");
            return;
        };
        let solver = BatchedInfiltration::new(device);
        let params = GreenAmptParams {
            delta_theta: 0.285,
            ..GreenAmptParams::CLAY
        };
        let times = [0.5, 1.0, 2.0, 4.0, 8.0];

        let gpu = solver.cumulative_gpu(&params, &times).unwrap();
        let cpu = cumulative_cpu(&params, &times);

        for (i, (g, c)) in gpu.iter().zip(&cpu).enumerate() {
            assert!((g - c).abs() < 0.5, "Clay F[{i}] GPU={g:.4} vs CPU={c:.4}");
        }
    }

    #[test]
    fn test_series_gpu() {
        let Some(device) = try_device() else {
            eprintln!("SKIP: No GPU device for series");
            return;
        };
        let solver = BatchedInfiltration::new(device);
        let params = GreenAmptParams::LOAM;
        let times = [0.5, 1.0, 2.0];

        let series = solver.series_gpu(&params, &times).unwrap();
        assert_eq!(series.len(), 3);
        for (i, &(f_cum, rate)) in series.iter().enumerate() {
            assert!(f_cum > 0.0, "F[{i}] should be positive");
            assert!(rate > 0.0, "rate[{i}] should be positive");
        }
    }

    #[test]
    fn test_gpu_monotonic() {
        let Some(device) = try_device() else {
            eprintln!("SKIP: No GPU device for monotonic");
            return;
        };
        let solver = BatchedInfiltration::new(device);
        let params = GreenAmptParams {
            delta_theta: 0.312,
            ..GreenAmptParams::SANDY_LOAM
        };
        let times = [0.1, 0.5, 1.0, 2.0, 4.0, 8.0, 24.0];

        let gpu = solver.cumulative_gpu(&params, &times).unwrap();
        for w in gpu.windows(2) {
            assert!(w[1] >= w[0] - 1e-6, "F not monotonic: {} > {}", w[0], w[1]);
        }
    }

    #[test]
    fn test_empty_input() {
        let Some(device) = try_device() else {
            eprintln!("SKIP: No GPU device");
            return;
        };
        let solver = BatchedInfiltration::new(device);
        let result = solver.cumulative_gpu(&GreenAmptParams::LOAM, &[]).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_cpu_fallback() {
        let params = GreenAmptParams::SANDY_LOAM;
        let cpu = cumulative_cpu(&params, &[0.5, 1.0]);
        assert_eq!(cpu.len(), 2);
        assert!(cpu[0] > 0.0);
        assert!(cpu[1] > cpu[0]);
    }
}
