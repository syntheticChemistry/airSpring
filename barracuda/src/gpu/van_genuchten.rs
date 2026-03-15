// SPDX-License-Identifier: AGPL-3.0-or-later
//! GPU-accelerated van Genuchten hydraulics — θ(h), K(h), and θ→h inverse.
//!
//! Forward: `BatchedElementwiseF64` ops 9/10 (`BarraCuda` S79).
//! Inverse: `BrentGpu::solve_vg_inverse` (`BarraCuda` S83, `brent_f64.wgsl`).
//!
//! # Cross-Spring Provenance
//!
//! | Primitive | Origin | Shader |
//! |-----------|--------|--------|
//! | θ(h) op=9 | airSpring V039 → S76 | `batched_elementwise_f64.wgsl` |
//! | K(h) op=10 | airSpring V039 → S76 | `batched_elementwise_f64.wgsl` |
//! | θ→h inverse | airSpring V045 → S83 | `brent_f64.wgsl` (hotSpring precision math) |

use std::sync::Arc;

use barracuda::device::WgpuDevice;
use barracuda::ops::batched_elementwise_f64::{
    BatchedElementwiseF64, Op, van_genuchten_k_cpu, van_genuchten_theta_cpu,
};
use barracuda::optimize::brent_gpu::BrentGpu;

#[cfg(test)]
use super::device_info::try_f64_device;

/// Batched van Genuchten θ(h) and K(h) GPU orchestrator.
///
/// Dispatches to `BatchedElementwiseF64` ops 9 and 10 when a GPU engine
/// is configured; falls back to CPU otherwise.
pub struct BatchedVanGenuchten {
    device: Arc<WgpuDevice>,
    engine: BatchedElementwiseF64,
}

impl std::fmt::Debug for BatchedVanGenuchten {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BatchedVanGenuchten")
            .field("device", &"Arc<WgpuDevice>")
            .finish()
    }
}

impl BatchedVanGenuchten {
    /// Create with GPU engine (dispatches to ops 9 and 10).
    ///
    /// # Errors
    ///
    /// Returns an error if `BatchedElementwiseF64` cannot be initialised.
    pub fn gpu(device: Arc<WgpuDevice>) -> crate::error::Result<Self> {
        let engine = BatchedElementwiseF64::new(device.clone())?;
        Ok(Self { device, engine })
    }

    /// Batch compute θ(h) for N pressure heads with shared VG parameters.
    ///
    /// Input layout per row (stride 5): `[theta_r, theta_s, alpha, n, h]`.
    ///
    /// # Errors
    ///
    /// Returns an error if GPU dispatch fails.
    pub fn compute_theta_gpu(
        &self,
        theta_r: f64,
        theta_s: f64,
        alpha: f64,
        n: f64,
        h_values: &[f64],
    ) -> crate::error::Result<Vec<f64>> {
        if h_values.is_empty() {
            return Ok(Vec::new());
        }
        let packed = Self::pack_theta_input(theta_r, theta_s, alpha, n, h_values);
        Ok(self
            .engine
            .execute(&packed, h_values.len(), Op::VanGenuchtenTheta)?)
    }

    /// Batch compute K(h) for N pressure heads with shared VG parameters.
    ///
    /// Input layout per row (stride 7): `[K_s, theta_r, theta_s, alpha, n, l, h]`.
    ///
    /// # Errors
    ///
    /// Returns an error if GPU dispatch fails.
    #[expect(
        clippy::too_many_arguments,
        reason = "van Genuchten K(θ) requires soil params, residual/saturated moisture, and GPU config"
    )]
    pub fn compute_k_gpu(
        &self,
        k_s: f64,
        theta_r: f64,
        theta_s: f64,
        alpha: f64,
        n: f64,
        l: f64,
        h_values: &[f64],
    ) -> crate::error::Result<Vec<f64>> {
        if h_values.is_empty() {
            return Ok(Vec::new());
        }
        let packed = Self::pack_k_input(k_s, theta_r, theta_s, alpha, n, l, h_values);
        Ok(self
            .engine
            .execute(&packed, h_values.len(), Op::VanGenuchtenK)?)
    }

    /// Batch compute inverse θ→h for N target moisture values using GPU Brent root-finding.
    ///
    /// Uses `BrentGpu::solve_vg_inverse` (S83, `brent_f64.wgsl`) — hotSpring precision
    /// math (`pow_f64`, `exp_f64`) applied to airSpring soil hydraulics.
    ///
    /// Returns `None` roots as `f64::NAN` for targets outside [θr, θs].
    ///
    /// # Errors
    ///
    /// Returns an error if GPU Brent dispatch fails.
    pub fn compute_inverse_gpu(
        &self,
        theta_r: f64,
        theta_s: f64,
        alpha: f64,
        n_vg: f64,
        theta_targets: &[f64],
    ) -> crate::error::Result<Vec<f64>> {
        if theta_targets.is_empty() {
            return Ok(Vec::new());
        }

        let brent = BrentGpu::new(Arc::clone(&self.device), 100, 1e-10)?;

        let lower = vec![-1e6; theta_targets.len()];
        let upper = vec![-1e-6; theta_targets.len()];

        let result =
            brent.solve_vg_inverse(&lower, &upper, theta_targets, theta_r, theta_s, alpha, n_vg)?;
        Ok(result.roots)
    }

    fn pack_theta_input(
        theta_r: f64,
        theta_s: f64,
        alpha: f64,
        n: f64,
        h_values: &[f64],
    ) -> Vec<f64> {
        let mut data = Vec::with_capacity(h_values.len() * 5);
        for &h in h_values {
            data.push(theta_r);
            data.push(theta_s);
            data.push(alpha);
            data.push(n);
            data.push(h);
        }
        data
    }

    fn pack_k_input(
        k_s: f64,
        theta_r: f64,
        theta_s: f64,
        alpha: f64,
        n: f64,
        l: f64,
        h_values: &[f64],
    ) -> Vec<f64> {
        let mut data = Vec::with_capacity(h_values.len() * 7);
        for &h in h_values {
            data.push(k_s);
            data.push(theta_r);
            data.push(theta_s);
            data.push(alpha);
            data.push(n);
            data.push(l);
            data.push(h);
        }
        data
    }
}

/// CPU fallback for θ(h) batch.
#[must_use]
pub fn compute_theta_cpu(
    theta_r: f64,
    theta_s: f64,
    alpha: f64,
    n: f64,
    h_values: &[f64],
) -> Vec<f64> {
    h_values
        .iter()
        .map(|&h| van_genuchten_theta_cpu(theta_r, theta_s, alpha, n, h))
        .collect()
}

/// CPU fallback for K(h) batch.
#[must_use]
pub fn compute_k_cpu(
    k_s: f64,
    theta_r: f64,
    theta_s: f64,
    alpha: f64,
    n: f64,
    l: f64,
    h_values: &[f64],
) -> Vec<f64> {
    h_values
        .iter()
        .map(|&h| van_genuchten_k_cpu(k_s, theta_r, theta_s, alpha, n, l, h))
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
    fn test_gpu_theta_matches_cpu() {
        let Some(device) = try_device() else {
            eprintln!("SKIP: No GPU device for BatchedVanGenuchten");
            return;
        };
        let vg = BatchedVanGenuchten::gpu(device).unwrap();
        let theta_r = 0.065;
        let theta_s = 0.41;
        let alpha = 0.075;
        let n = 1.89;
        let h_values = [-500.0, -100.0, -50.0, -10.0, -1.0, 0.0];

        let gpu = vg
            .compute_theta_gpu(theta_r, theta_s, alpha, n, &h_values)
            .unwrap();
        let cpu = compute_theta_cpu(theta_r, theta_s, alpha, n, &h_values);

        assert_eq!(gpu.len(), cpu.len());
        for (g, c) in gpu.iter().zip(&cpu) {
            assert!((g - c).abs() < 0.01, "GPU θ={g:.6} vs CPU θ={c:.6}");
        }
    }

    #[test]
    fn test_gpu_k_matches_cpu() {
        let Some(device) = try_device() else {
            eprintln!("SKIP: No GPU device for BatchedVanGenuchten");
            return;
        };
        let vg = BatchedVanGenuchten::gpu(device).unwrap();
        let k_s = 10.0;
        let theta_r = 0.065;
        let theta_s = 0.41;
        let alpha = 0.075;
        let n = 1.89;
        let l = 0.5;
        let h_values = [-500.0, -100.0, -50.0, -10.0, -1.0, 0.0];

        let gpu = vg
            .compute_k_gpu(k_s, theta_r, theta_s, alpha, n, l, &h_values)
            .unwrap();
        let cpu = compute_k_cpu(k_s, theta_r, theta_s, alpha, n, l, &h_values);

        assert_eq!(gpu.len(), cpu.len());
        for (g, c) in gpu.iter().zip(&cpu) {
            assert!((g - c).abs() < 0.1, "GPU K={g:.6} vs CPU K={c:.6}");
        }
    }

    #[test]
    fn test_gpu_inverse_matches_cpu() {
        let Some(device) = try_device() else {
            eprintln!("SKIP: No GPU device for BatchedVanGenuchten inverse");
            return;
        };
        let vg = BatchedVanGenuchten::gpu(device).unwrap();
        let theta_r = 0.065;
        let theta_s = 0.41;
        let alpha = 0.075;
        let n = 1.89;

        let h_values = [-500.0, -100.0, -50.0, -10.0, -1.0];
        let theta_targets = compute_theta_cpu(theta_r, theta_s, alpha, n, &h_values);

        let gpu_h = vg
            .compute_inverse_gpu(theta_r, theta_s, alpha, n, &theta_targets)
            .unwrap();
        assert_eq!(gpu_h.len(), h_values.len());
        for (i, (&got, &expected)) in gpu_h.iter().zip(h_values.iter()).enumerate() {
            assert!(
                (got - expected).abs() < 1.0,
                "GPU inverse h[{i}]={got:.2} vs expected={expected:.2}"
            );
        }
    }

    #[test]
    fn test_theta_bounded() {
        let theta_r = 0.045;
        let theta_s = 0.43;
        let alpha = 0.145;
        let n = 2.68;
        let h_values = [-10_000.0, -100.0, -1.0, 0.0, 5.0];

        let cpu = compute_theta_cpu(theta_r, theta_s, alpha, n, &h_values);
        for &theta in &cpu {
            assert!(
                theta >= theta_r && theta <= theta_s,
                "θ={theta} should be in [θr={theta_r}, θs={theta_s}]"
            );
        }
    }

    #[test]
    fn test_k_non_negative() {
        let k_s = 712.8;
        let theta_r = 0.045;
        let theta_s = 0.43;
        let alpha = 0.145;
        let n = 2.68;
        let l = 0.5;
        let h_values = [-10_000.0, -500.0, -100.0, -10.0, 0.0];

        let cpu = compute_k_cpu(k_s, theta_r, theta_s, alpha, n, l, &h_values);
        for &k in &cpu {
            assert!(k >= 0.0, "K={k} should be non-negative");
        }
    }

    #[test]
    fn test_theta_cpu_empty() {
        let result = compute_theta_cpu(0.065, 0.41, 0.075, 1.89, &[]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_k_cpu_empty() {
        let result = compute_k_cpu(10.0, 0.065, 0.41, 0.075, 1.89, 0.5, &[]);
        assert!(result.is_empty());
    }
}
