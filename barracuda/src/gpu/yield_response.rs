// SPDX-License-Identifier: AGPL-3.0-or-later
//! Batched Stewart yield response — GPU-local via `local_elementwise.wgsl`.
//!
//! Implements the Stewart (1977) yield-water production function
//! `Ya/Ymax = 1 − Ky × (1 − ETa/ETc)` in batch for many fields simultaneously.
//!
//! # Cross-Spring Provenance
//!
//! | Primitive | Origin | Status |
//! |-----------|--------|--------|
//! | Stewart equation | Stewart (1977) + FAO-56 Ch 10 | **GPU-local** (f32 WGSL) |
//! | Multi-stage product | Doorenbos & Kassam (1979) | CPU fallback |
//! | GPU dispatch | `local_elementwise.wgsl` op=1 | **Live** (v0.6.8) |

use std::sync::Arc;

use barracuda::device::WgpuDevice;

use crate::eco::yield_response;
use crate::gpu::local_dispatch::{LocalElementwise, LocalOp};

/// Single field yield computation input.
#[derive(Debug, Clone, Copy)]
pub struct YieldInput {
    /// Yield response factor (FAO-56 Table 24).
    pub ky: f64,
    /// Actual evapotranspiration (mm).
    pub et_actual: f64,
    /// Crop evapotranspiration (mm).
    pub et_crop: f64,
}

/// Batched Stewart yield response orchestrator.
///
/// CPU path uses `eco::yield_response` directly. GPU path dispatches via
/// `LocalElementwise` (f32 WGSL shader, pending `ToadStool` absorption to f64).
#[derive(Debug)]
pub struct BatchedYieldResponse;

/// GPU-backed Stewart yield response dispatcher.
///
/// Uses `local_elementwise.wgsl` op 1 for GPU-parallel computation.
pub struct GpuYieldResponse {
    dispatcher: LocalElementwise,
}

impl std::fmt::Debug for GpuYieldResponse {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuYieldResponse").finish()
    }
}

impl GpuYieldResponse {
    /// Create a GPU-backed yield response solver.
    ///
    /// # Errors
    ///
    /// Returns an error if WGSL shader compilation fails.
    pub fn new(device: Arc<WgpuDevice>) -> crate::error::Result<Self> {
        Ok(Self {
            dispatcher: LocalElementwise::new(device)?,
        })
    }

    /// Batch compute yield ratios on GPU.
    ///
    /// # Errors
    ///
    /// Returns an error if GPU dispatch fails.
    pub fn compute(&self, inputs: &[YieldInput]) -> crate::error::Result<Vec<f64>> {
        let ky: Vec<f64> = inputs.iter().map(|i| i.ky).collect();
        let ratio: Vec<f64> = inputs
            .iter()
            .map(|i| {
                if i.et_crop > 0.0 {
                    i.et_actual / i.et_crop
                } else {
                    1.0
                }
            })
            .collect();
        let zeros = vec![0.0; inputs.len()];
        let results = self
            .dispatcher
            .dispatch(LocalOp::StewartYield, &ky, &ratio, &zeros)?;
        Ok(results.iter().map(|&v| v.clamp(0.0, 1.0)).collect())
    }

    /// Batch compute with uniform Ky across fields.
    ///
    /// # Errors
    ///
    /// Returns an error if GPU dispatch fails.
    pub fn compute_uniform(
        &self,
        ky: f64,
        et_actual: &[f64],
        et_crop: &[f64],
    ) -> crate::error::Result<Vec<f64>> {
        let kys = vec![ky; et_actual.len()];
        let ratios: Vec<f64> = et_actual
            .iter()
            .zip(et_crop)
            .map(|(&a, &c)| if c > 0.0 { a / c } else { 1.0 })
            .collect();
        let zeros = vec![0.0; et_actual.len()];
        let results = self
            .dispatcher
            .dispatch(LocalOp::StewartYield, &kys, &ratios, &zeros)?;
        Ok(results.iter().map(|&v| v.clamp(0.0, 1.0)).collect())
    }
}

impl BatchedYieldResponse {
    /// Batch compute yield ratios for N fields.
    ///
    /// Returns `Ya/Ymax` clamped to `[0, 1]` for each field.
    #[must_use]
    pub fn compute(inputs: &[YieldInput]) -> Vec<f64> {
        inputs
            .iter()
            .map(|inp| {
                let ratio = if inp.et_crop > 0.0 {
                    inp.et_actual / inp.et_crop
                } else {
                    1.0
                };
                yield_response::yield_ratio_single(inp.ky, ratio).clamp(0.0, 1.0)
            })
            .collect()
    }

    /// Batch compute yield ratios with uniform Ky across fields.
    #[must_use]
    pub fn compute_uniform(ky: f64, et_actual: &[f64], et_crop: &[f64]) -> Vec<f64> {
        et_actual
            .iter()
            .zip(et_crop)
            .map(|(&eta, &etc)| {
                let ratio = if etc > 0.0 { eta / etc } else { 1.0 };
                yield_response::yield_ratio_single(ky, ratio).clamp(0.0, 1.0)
            })
            .collect()
    }

    /// Batch compute water use efficiency: Ya / `ETa`.
    #[must_use]
    pub fn water_use_efficiency(yield_ratios: &[f64], et_actual: &[f64]) -> Vec<f64> {
        yield_ratios
            .iter()
            .zip(et_actual)
            .map(|(&yr, &eta)| if eta > 0.0 { yr / eta } else { 0.0 })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_matches_cpu() {
        let Some(device) = crate::gpu::device_info::try_f64_device() else {
            eprintln!("SKIP: no GPU for GpuYieldResponse");
            return;
        };
        let gpu_solver = GpuYieldResponse::new(device).unwrap();
        let inputs = vec![
            YieldInput {
                ky: 1.25,
                et_actual: 500.0,
                et_crop: 600.0,
            },
            YieldInput {
                ky: 0.85,
                et_actual: 400.0,
                et_crop: 600.0,
            },
            YieldInput {
                ky: 1.0,
                et_actual: 600.0,
                et_crop: 600.0,
            },
        ];
        let gpu = gpu_solver.compute(&inputs).unwrap();
        let cpu = BatchedYieldResponse::compute(&inputs);

        for (i, (g, c)) in gpu.iter().zip(&cpu).enumerate() {
            let tol = c.abs().mul_add(1e-3, 1e-4);
            assert!((g - c).abs() < tol, "Yield[{i}] GPU={g:.6} CPU={c:.6}");
        }
    }

    #[test]
    fn test_batch_compute() {
        let inputs = vec![
            YieldInput {
                ky: 1.25,
                et_actual: 500.0,
                et_crop: 600.0,
            },
            YieldInput {
                ky: 0.85,
                et_actual: 400.0,
                et_crop: 600.0,
            },
            YieldInput {
                ky: 1.0,
                et_actual: 600.0,
                et_crop: 600.0,
            },
        ];
        let ratios = BatchedYieldResponse::compute(&inputs);
        assert_eq!(ratios.len(), 3);
        assert!(ratios[0] < 1.0, "stressed should reduce yield");
        assert!(ratios[2] > 0.99, "no stress → full yield");
    }

    #[test]
    fn test_uniform() {
        let eta = [400.0, 450.0, 500.0, 550.0, 600.0];
        let etc = [600.0; 5];
        let ratios = BatchedYieldResponse::compute_uniform(1.25, &eta, &etc);
        for w in ratios.windows(2) {
            assert!(w[1] >= w[0] - 1e-10, "more ETa → better yield");
        }
    }

    #[test]
    fn test_wue() {
        let ratios = [0.9, 0.8, 0.7];
        let eta = [500.0, 450.0, 400.0];
        let wue = BatchedYieldResponse::water_use_efficiency(&ratios, &eta);
        assert_eq!(wue.len(), 3);
        for &w in &wue {
            assert!(w > 0.0);
        }
    }
}
