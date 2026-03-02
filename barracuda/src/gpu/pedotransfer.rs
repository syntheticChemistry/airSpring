// SPDX-License-Identifier: AGPL-3.0-or-later
//! GPU-accelerated pedotransfer polynomial evaluation (Horner form).
//!
//! Wraps `BatchedElementwiseF64` op 13 (`PedotransferPolynomial`) from `ToadStool` S79.
//! Evaluates 5th-degree polynomials `a0 + x*(a1 + x*(a2 + x*(a3 + x*(a4 + x*a5))))`.
//!
//! Used for Saxton-Rawls (2006) pedotransfer functions that predict soil hydraulic
//! properties from basic soil texture/OM data.
//!
//! # Cross-Spring Provenance
//!
//! - **airSpring**: Domain need (pedotransfer for soil hydraulics)
//! - **neuralSpring**: Batch orchestrator pattern
//! - **`ToadStool` S79**: WGSL shader, f64 canonical Horner evaluation

use std::sync::Arc;

use barracuda::device::WgpuDevice;
use barracuda::ops::batched_elementwise_f64::{BatchedElementwiseF64, Op};

#[cfg(test)]
use super::device_info::try_f64_device;

/// Single pedotransfer polynomial input: coefficients a0..a5 and evaluation point x.
#[derive(Debug, Clone, Copy)]
pub struct PedotransferInput {
    /// Polynomial coefficients [a0, a1, a2, a3, a4, a5].
    pub coeffs: [f64; 6],
    /// Evaluation point.
    pub x: f64,
}

/// Batched pedotransfer polynomial GPU orchestrator.
///
/// Dispatches to `BatchedElementwiseF64` op 13 when a GPU engine is configured;
/// falls back to CPU otherwise.
pub struct BatchedPedotransfer {
    gpu_engine: Option<BatchedElementwiseF64>,
}

impl std::fmt::Debug for BatchedPedotransfer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BatchedPedotransfer")
            .field("gpu_engine", &self.gpu_engine.is_some())
            .finish()
    }
}

impl BatchedPedotransfer {
    /// Create with GPU engine (dispatches to op 13).
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

    /// Compute polynomial values for a batch of inputs.
    ///
    /// Input layout per row (stride 7): `[a0, a1, a2, a3, a4, a5, x]`.
    ///
    /// # Errors
    ///
    /// Returns an error if GPU dispatch fails.
    pub fn compute(&self, inputs: &[PedotransferInput]) -> crate::error::Result<Vec<f64>> {
        if inputs.is_empty() {
            return Ok(Vec::new());
        }
        if let Some(engine) = &self.gpu_engine {
            let packed = Self::pack_inputs(inputs);
            Ok(engine.execute(&packed, inputs.len(), Op::PedotransferPolynomial)?)
        } else {
            Ok(compute_pedotransfer_cpu(inputs))
        }
    }

    fn pack_inputs(inputs: &[PedotransferInput]) -> Vec<f64> {
        let mut data = Vec::with_capacity(inputs.len() * 7);
        for inp in inputs {
            data.extend_from_slice(&inp.coeffs);
            data.push(inp.x);
        }
        data
    }
}

/// CPU fallback: Horner form evaluation.
#[must_use]
pub fn compute_pedotransfer_cpu(inputs: &[PedotransferInput]) -> Vec<f64> {
    inputs
        .iter()
        .map(|inp| horner_eval(&inp.coeffs, inp.x))
        .collect()
}

/// Horner form: a0 + x*(a1 + x*(a2 + x*(a3 + x*(a4 + x*a5)))).
#[must_use]
fn horner_eval(coeffs: &[f64; 6], x: f64) -> f64 {
    let [a0, a1, a2, a3, a4, a5] = *coeffs;
    a5.mul_add(x, a4)
        .mul_add(x, a3)
        .mul_add(x, a2)
        .mul_add(x, a1)
        .mul_add(x, a0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn try_device() -> Option<Arc<WgpuDevice>> {
        try_f64_device()
    }

    #[test]
    fn test_positive_coefficients() {
        let engine = BatchedPedotransfer::cpu();
        let inputs = [PedotransferInput {
            coeffs: [1.0, 2.0, 3.0, 0.0, 0.0, 0.0],
            x: 2.0,
        }];
        let result = engine.compute(&inputs).unwrap();
        assert_eq!(result.len(), 1);
        // 1 + 2*2 + 3*4 = 1 + 4 + 12 = 17
        assert!((result[0] - 17.0).abs() < 0.001);
    }

    #[test]
    fn test_zero_polynomial() {
        let engine = BatchedPedotransfer::cpu();
        let inputs = [PedotransferInput {
            coeffs: [0.0; 6],
            x: 42.0,
        }];
        let result = engine.compute(&inputs).unwrap();
        assert_eq!(result.len(), 1);
        assert!(result[0].abs() < 0.001);
    }

    #[test]
    fn test_gpu_matches_cpu() {
        let Some(device) = try_device() else {
            eprintln!("SKIP: No GPU device for BatchedPedotransfer");
            return;
        };
        let gpu_engine = BatchedPedotransfer::gpu(device).unwrap();
        let cpu_engine = BatchedPedotransfer::cpu();

        let inputs = [
            PedotransferInput {
                coeffs: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                x: 0.5,
            },
            PedotransferInput {
                coeffs: [0.065, 0.1, -0.01, 0.0, 0.0, 0.0],
                x: 0.15,
            },
        ];

        let gpu_result = gpu_engine.compute(&inputs).unwrap();
        let cpu_result = cpu_engine.compute(&inputs).unwrap();

        assert_eq!(gpu_result.len(), cpu_result.len());
        for (g, c) in gpu_result.iter().zip(&cpu_result) {
            assert!((g - c).abs() < 0.001, "GPU={g:.6} vs CPU={c:.6}");
        }
    }

    #[test]
    fn test_empty_batch() {
        let engine = BatchedPedotransfer::cpu();
        let result = engine.compute(&[]).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_cpu_horner_known() {
        let inputs = [PedotransferInput {
            coeffs: [2.0, 3.0, 0.0, 0.0, 0.0, 0.0],
            x: 4.0,
        }];
        let cpu = compute_pedotransfer_cpu(&inputs);
        assert_eq!(cpu.len(), 1);
        assert!((cpu[0] - 14.0).abs() < 0.001);
    }
}
