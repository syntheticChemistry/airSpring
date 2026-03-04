// SPDX-License-Identifier: AGPL-3.0-or-later
//! Batched simple ET₀ methods — Makkink, Turc, Hamon, Blaney-Criddle.
//!
//! GPU-local batch interface for four data-sparse ET₀ methods via
//! `local_elementwise_f64.wgsl` (ops 2-5). Complements the full FAO-56 PM
//! (`gpu::et0`) and Hargreaves (`gpu::hargreaves`).
//!
//! # Cross-Spring Provenance
//!
//! | Method | Origin | Status |
//! |--------|--------|--------|
//! | Makkink (1957) | KNMI / de Bruin (1987) | **GPU-universal** (f64 canonical op=2) |
//! | Turc (1961) | Turc (1961) Eq. 1-2 | **GPU-universal** (f64 canonical op=3) |
//! | Hamon (1961) | Lu et al. (2005) | **GPU-universal** (f64 canonical op=4) |
//! | Blaney-Criddle (1950) | FAO-24, USDA-SCS | **GPU-universal** (f64 canonical op=5) |
//! | GPU dispatch | `local_elementwise_f64.wgsl` ops 2-5 | **Live** (v0.6.9, f64 canonical) |

use std::sync::Arc;

use barracuda::device::WgpuDevice;

use crate::eco::simple_et0;
use crate::gpu::local_dispatch::{LocalElementwise, LocalOp};

/// Makkink input: (`tmean_c`, `rs_mj`, `elevation_m`).
#[derive(Debug, Clone, Copy)]
pub struct MakkinkInput {
    pub tmean_c: f64,
    pub rs_mj: f64,
    pub elevation_m: f64,
}

/// Turc input: (`tmean_c`, `rs_mj`, `rh_pct`).
#[derive(Debug, Clone, Copy)]
pub struct TurcInput {
    pub tmean_c: f64,
    pub rs_mj: f64,
    pub rh_pct: f64,
}

/// Hamon input: (`tmean_c`, `latitude_rad`, `doy`).
#[derive(Debug, Clone, Copy)]
pub struct HamonInput {
    pub tmean_c: f64,
    pub latitude_rad: f64,
    pub doy: u32,
}

/// Blaney-Criddle input: (`tmean_c`, `latitude_rad`, `doy`).
#[derive(Debug, Clone, Copy)]
pub struct BlaneyCriddleInput {
    pub tmean_c: f64,
    pub latitude_rad: f64,
    pub doy: u32,
}

/// Batched simple ET₀ orchestrator (CPU path).
#[derive(Debug)]
pub struct BatchedSimpleEt0;

/// GPU-backed simple ET₀ dispatcher for all four methods.
///
/// Uses `local_elementwise_f64.wgsl` ops 2-5 via `compile_shader_universal` (f64 canonical → f32 on consumer GPUs).
pub struct GpuSimpleEt0 {
    dispatcher: LocalElementwise,
}

impl std::fmt::Debug for GpuSimpleEt0 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuSimpleEt0").finish()
    }
}

impl GpuSimpleEt0 {
    /// Create a GPU-backed simple ET₀ solver.
    ///
    /// # Errors
    ///
    /// Returns an error if WGSL shader compilation fails.
    pub fn new(device: Arc<WgpuDevice>) -> crate::error::Result<Self> {
        Ok(Self {
            dispatcher: LocalElementwise::new(device)?,
        })
    }

    /// Batch Makkink ET₀ on GPU.
    ///
    /// # Errors
    ///
    /// Returns an error if GPU dispatch fails.
    pub fn makkink(&self, inputs: &[MakkinkInput]) -> crate::error::Result<Vec<f64>> {
        let a: Vec<f64> = inputs.iter().map(|i| i.tmean_c).collect();
        let b: Vec<f64> = inputs.iter().map(|i| i.rs_mj).collect();
        let c: Vec<f64> = inputs.iter().map(|i| i.elevation_m).collect();
        self.dispatcher.dispatch(LocalOp::MakkinkEt0, &a, &b, &c)
    }

    /// Batch Turc ET₀ on GPU.
    ///
    /// # Errors
    ///
    /// Returns an error if GPU dispatch fails.
    pub fn turc(&self, inputs: &[TurcInput]) -> crate::error::Result<Vec<f64>> {
        let a: Vec<f64> = inputs.iter().map(|i| i.tmean_c).collect();
        let b: Vec<f64> = inputs.iter().map(|i| i.rs_mj).collect();
        let c: Vec<f64> = inputs.iter().map(|i| i.rh_pct).collect();
        self.dispatcher.dispatch(LocalOp::TurcEt0, &a, &b, &c)
    }

    /// Batch Hamon PET on GPU.
    ///
    /// # Errors
    ///
    /// Returns an error if GPU dispatch fails.
    pub fn hamon(&self, inputs: &[HamonInput]) -> crate::error::Result<Vec<f64>> {
        let a: Vec<f64> = inputs.iter().map(|i| i.tmean_c).collect();
        let b: Vec<f64> = inputs.iter().map(|i| i.latitude_rad).collect();
        let c: Vec<f64> = inputs.iter().map(|i| f64::from(i.doy)).collect();
        self.dispatcher.dispatch(LocalOp::HamonPet, &a, &b, &c)
    }

    /// Batch Blaney-Criddle ET₀ on GPU.
    ///
    /// # Errors
    ///
    /// Returns an error if GPU dispatch fails.
    pub fn blaney_criddle(&self, inputs: &[BlaneyCriddleInput]) -> crate::error::Result<Vec<f64>> {
        let a: Vec<f64> = inputs.iter().map(|i| i.tmean_c).collect();
        let b: Vec<f64> = inputs.iter().map(|i| i.latitude_rad).collect();
        let c: Vec<f64> = inputs.iter().map(|i| f64::from(i.doy)).collect();
        self.dispatcher
            .dispatch(LocalOp::BlaneyCriddleEt0, &a, &b, &c)
    }
}

impl BatchedSimpleEt0 {
    /// Batch Makkink ET₀ (mm/day).
    #[must_use]
    pub fn makkink(inputs: &[MakkinkInput]) -> Vec<f64> {
        inputs
            .iter()
            .map(|i| simple_et0::makkink_et0(i.tmean_c, i.rs_mj, i.elevation_m))
            .collect()
    }

    /// Batch Turc ET₀ (mm/day).
    #[must_use]
    pub fn turc(inputs: &[TurcInput]) -> Vec<f64> {
        inputs
            .iter()
            .map(|i| simple_et0::turc_et0(i.tmean_c, i.rs_mj, i.rh_pct))
            .collect()
    }

    /// Batch Hamon PET (mm/day).
    #[must_use]
    pub fn hamon(inputs: &[HamonInput]) -> Vec<f64> {
        inputs
            .iter()
            .map(|i| simple_et0::hamon_pet_from_location(i.tmean_c, i.latitude_rad, i.doy))
            .collect()
    }

    /// Batch Blaney-Criddle ET₀ (mm/day).
    #[must_use]
    pub fn blaney_criddle(inputs: &[BlaneyCriddleInput]) -> Vec<f64> {
        inputs
            .iter()
            .map(|i| simple_et0::blaney_criddle_from_location(i.tmean_c, i.latitude_rad, i.doy))
            .collect()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_makkink_parity() {
        let Some(device) = crate::gpu::device_info::try_f64_device() else {
            eprintln!("SKIP: no GPU for GpuSimpleEt0");
            return;
        };
        let gpu_solver = GpuSimpleEt0::new(device).unwrap();
        let inputs = vec![
            MakkinkInput {
                tmean_c: 20.0,
                rs_mj: 15.0,
                elevation_m: 100.0,
            },
            MakkinkInput {
                tmean_c: 30.0,
                rs_mj: 25.0,
                elevation_m: 0.0,
            },
        ];
        let gpu = gpu_solver.makkink(&inputs).unwrap();
        let cpu = BatchedSimpleEt0::makkink(&inputs);
        for (i, (g, c)) in gpu.iter().zip(&cpu).enumerate() {
            let tol = c.abs().mul_add(2e-3, 0.01);
            assert!((g - c).abs() < tol, "Makkink[{i}] GPU={g:.4} CPU={c:.4}");
        }
    }

    #[test]
    fn test_gpu_turc_parity() {
        let Some(device) = crate::gpu::device_info::try_f64_device() else {
            eprintln!("SKIP: no GPU");
            return;
        };
        let gpu_solver = GpuSimpleEt0::new(device).unwrap();
        let inputs = vec![
            TurcInput {
                tmean_c: 20.0,
                rs_mj: 15.0,
                rh_pct: 70.0,
            },
            TurcInput {
                tmean_c: 25.0,
                rs_mj: 20.0,
                rh_pct: 40.0,
            },
        ];
        let gpu = gpu_solver.turc(&inputs).unwrap();
        let cpu = BatchedSimpleEt0::turc(&inputs);
        for (i, (g, c)) in gpu.iter().zip(&cpu).enumerate() {
            let tol = c.abs().mul_add(2e-3, 0.01);
            assert!((g - c).abs() < tol, "Turc[{i}] GPU={g:.4} CPU={c:.4}");
        }
    }

    #[test]
    fn test_gpu_hamon_parity() {
        let Some(device) = crate::gpu::device_info::try_f64_device() else {
            eprintln!("SKIP: no GPU");
            return;
        };
        let gpu_solver = GpuSimpleEt0::new(device).unwrap();
        let lat_rad = 42.7_f64.to_radians();
        let inputs = vec![
            HamonInput {
                tmean_c: 20.0,
                latitude_rad: lat_rad,
                doy: 180,
            },
            HamonInput {
                tmean_c: 10.0,
                latitude_rad: lat_rad,
                doy: 90,
            },
        ];
        let gpu = gpu_solver.hamon(&inputs).unwrap();
        let cpu = BatchedSimpleEt0::hamon(&inputs);
        for (i, (g, c)) in gpu.iter().zip(&cpu).enumerate() {
            let tol = c.abs().mul_add(5e-3, 0.02);
            assert!((g - c).abs() < tol, "Hamon[{i}] GPU={g:.4} CPU={c:.4}");
        }
    }

    #[test]
    fn test_gpu_blaney_criddle_parity() {
        let Some(device) = crate::gpu::device_info::try_f64_device() else {
            eprintln!("SKIP: no GPU");
            return;
        };
        let gpu_solver = GpuSimpleEt0::new(device).unwrap();
        let lat_rad = 42.7_f64.to_radians();
        let inputs = vec![
            BlaneyCriddleInput {
                tmean_c: 25.0,
                latitude_rad: lat_rad,
                doy: 180,
            },
            BlaneyCriddleInput {
                tmean_c: 5.0,
                latitude_rad: lat_rad,
                doy: 15,
            },
        ];
        let gpu = gpu_solver.blaney_criddle(&inputs).unwrap();
        let cpu = BatchedSimpleEt0::blaney_criddle(&inputs);
        for (i, (g, c)) in gpu.iter().zip(&cpu).enumerate() {
            let tol = c.abs().mul_add(5e-3, 0.02);
            assert!((g - c).abs() < tol, "BC[{i}] GPU={g:.4} CPU={c:.4}");
        }
    }

    #[test]
    fn test_makkink_batch() {
        let inputs = vec![
            MakkinkInput {
                tmean_c: 20.0,
                rs_mj: 15.0,
                elevation_m: 100.0,
            },
            MakkinkInput {
                tmean_c: 30.0,
                rs_mj: 25.0,
                elevation_m: 0.0,
            },
        ];
        let results = BatchedSimpleEt0::makkink(&inputs);
        assert_eq!(results.len(), 2);
        for &et0 in &results {
            assert!(et0 > 0.0 && et0 < 10.0, "ET₀={et0}");
        }
    }

    #[test]
    fn test_turc_batch() {
        let inputs = vec![
            TurcInput {
                tmean_c: 20.0,
                rs_mj: 15.0,
                rh_pct: 70.0,
            },
            TurcInput {
                tmean_c: 25.0,
                rs_mj: 20.0,
                rh_pct: 60.0,
            },
        ];
        let results = BatchedSimpleEt0::turc(&inputs);
        assert_eq!(results.len(), 2);
        for &et0 in &results {
            assert!(et0 > 0.0 && et0 < 10.0, "ET₀={et0}");
        }
    }

    #[test]
    fn test_hamon_batch() {
        let lat_rad = 42.7_f64.to_radians();
        let inputs = vec![
            HamonInput {
                tmean_c: 20.0,
                latitude_rad: lat_rad,
                doy: 180,
            },
            HamonInput {
                tmean_c: 10.0,
                latitude_rad: lat_rad,
                doy: 90,
            },
        ];
        let results = BatchedSimpleEt0::hamon(&inputs);
        assert_eq!(results.len(), 2);
        assert!(results[0] > results[1], "warmer + longer days → more ET");
    }

    #[test]
    fn test_blaney_criddle_batch() {
        let lat_rad = 42.7_f64.to_radians();
        let inputs = vec![
            BlaneyCriddleInput {
                tmean_c: 25.0,
                latitude_rad: lat_rad,
                doy: 180,
            },
            BlaneyCriddleInput {
                tmean_c: 5.0,
                latitude_rad: lat_rad,
                doy: 15,
            },
        ];
        let results = BatchedSimpleEt0::blaney_criddle(&inputs);
        assert_eq!(results.len(), 2);
        assert!(results[0] > results[1], "summer > winter ET₀");
    }

    #[test]
    fn test_empty_batches() {
        assert!(BatchedSimpleEt0::makkink(&[]).is_empty());
        assert!(BatchedSimpleEt0::turc(&[]).is_empty());
        assert!(BatchedSimpleEt0::hamon(&[]).is_empty());
        assert!(BatchedSimpleEt0::blaney_criddle(&[]).is_empty());
    }
}
