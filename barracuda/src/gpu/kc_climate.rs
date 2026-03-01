// SPDX-License-Identifier: AGPL-3.0-or-later
//! Batched Kc climate adjustment GPU orchestrator (Tier A — op=7 absorbed).
//!
//! FAO-56 Eq. 62 adjusts tabulated crop coefficients for local wind and humidity:
//!
//! ```text
//! Kc_adj = Kc_table + [0.04(u2 - 2) - 0.004(RH_min - 45)] × (h/3)^0.3
//! ```
//!
//! # Two API Levels
//!
//! | API | Device? | Backend |
//! |-----|:-------:|---------|
//! | [`BatchedKcClimate::compute`] | No | CPU via `eco::crop::adjust_kc_for_climate` |
//! | [`BatchedKcClimate::compute_gpu`] | Yes | **GPU** via `BatchedElementwiseF64` op=7 (`ToadStool` S70+) |
//!
//! # GPU Dispatch
//!
//! The CPU path is fully validated against FAO-56 Eq. 62. The GPU path
//! dispatches to `ToadStool` `BatchedElementwiseF64` op=7 (stride=4:
//! `[kc_table, u2, rh_min, crop_height_m]`), absorbed in S70+.
//!
//! # Reference
//!
//! Allen RG et al. (1998) FAO Irrigation and Drainage Paper 56, Eq. 62.

use std::sync::Arc;

use barracuda::device::WgpuDevice;
use barracuda::ops::batched_elementwise_f64::{BatchedElementwiseF64, Op};

use crate::eco::crop;

/// Per-day input for batched Kc climate adjustment.
///
/// Stride-4 layout for future GPU shader: `[kc_table, u2, rh_min, crop_height_m]`.
#[derive(Debug, Clone, Copy)]
pub struct KcClimateDay {
    /// Tabulated Kc from FAO-56 Table 12.
    pub kc_table: f64,
    /// Mean wind speed at 2 m (m/s).
    pub u2: f64,
    /// Mean minimum relative humidity (%).
    pub rh_min: f64,
    /// Crop height (m).
    pub crop_height_m: f64,
}

/// Backend selection for batched Kc climate adjustment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Backend {
    /// Validated CPU path (always available).
    #[default]
    Cpu,
    /// GPU path via `BatchedElementwiseF64` op=7 (`ToadStool` S70+ absorbed).
    Gpu,
}

/// Result from a batched Kc climate adjustment computation.
#[derive(Debug, Clone)]
pub struct BatchedKcClimateResult {
    /// Adjusted Kc values, one per input row.
    pub kc_values: Vec<f64>,
    /// Which backend was actually used.
    pub backend_used: Backend,
}

/// Batched Kc climate adjustment orchestrator.
///
/// Computes FAO-56 Eq. 62 for N station-days in a single call.
/// GPU dispatch via `BatchedElementwiseF64` op=7 (absorbed in `ToadStool` S70+).
/// Falls back to CPU when no GPU device is configured.
pub struct BatchedKcClimate {
    backend: Backend,
    gpu_engine: Option<BatchedElementwiseF64>,
}

impl std::fmt::Debug for BatchedKcClimate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BatchedKcClimate")
            .field("backend", &self.backend)
            .field("gpu_engine", &self.gpu_engine.is_some())
            .finish()
    }
}

impl BatchedKcClimate {
    /// Create with GPU engine (Tier A — dispatches to op=7 shader).
    ///
    /// # Errors
    ///
    /// Returns an error if `BatchedElementwiseF64` cannot be initialised.
    pub fn gpu(device: Arc<WgpuDevice>) -> crate::error::Result<Self> {
        let engine = BatchedElementwiseF64::new(device)?;
        Ok(Self {
            backend: Backend::Gpu,
            gpu_engine: Some(engine),
        })
    }

    /// Returns a reference to the GPU engine, if available.
    /// Used for `ToadStool` GPU dispatch when the shader is wired.
    #[must_use]
    pub const fn gpu_engine(&self) -> Option<&BatchedElementwiseF64> {
        self.gpu_engine.as_ref()
    }

    /// Create with CPU fallback (always safe, no device needed).
    #[must_use]
    pub const fn cpu() -> Self {
        Self {
            backend: Backend::Cpu,
            gpu_engine: None,
        }
    }

    /// Compute Kc climate adjustment for a batch of station-days.
    ///
    /// Dispatches to GPU via `BatchedElementwiseF64` op=7 when a GPU engine
    /// is configured; falls back to CPU otherwise.
    ///
    /// # Errors
    ///
    /// Returns an error if the GPU dispatch fails irrecoverably.
    pub fn compute_gpu(
        &self,
        inputs: &[KcClimateDay],
    ) -> crate::error::Result<BatchedKcClimateResult> {
        if let Some(engine) = &self.gpu_engine {
            let packed = Self::pack_gpu_input(inputs);
            let kc_values = engine.execute(&packed, inputs.len(), Op::KcClimateAdjust)?;
            Ok(BatchedKcClimateResult {
                kc_values,
                backend_used: Backend::Gpu,
            })
        } else {
            let kc_values = Self::compute_cpu_batch(inputs);
            Ok(BatchedKcClimateResult {
                kc_values,
                backend_used: Backend::Cpu,
            })
        }
    }

    /// Pack inputs into stride-4 GPU layout: `[kc_table, u2, rh_min, crop_height_m]`.
    ///
    /// Ready for `ToadStool` op=7 absorption — produces the flat `f64` array
    /// that `BatchedElementwiseF64::execute` expects.
    #[must_use]
    pub fn pack_gpu_input(inputs: &[KcClimateDay]) -> Vec<f64> {
        let mut data = Vec::with_capacity(inputs.len() * 4);
        for d in inputs {
            data.push(d.kc_table);
            data.push(d.u2);
            data.push(d.rh_min);
            data.push(d.crop_height_m);
        }
        data
    }

    /// Compute Kc climate adjustment using the validated CPU path.
    #[must_use]
    pub fn compute(&self, inputs: &[KcClimateDay]) -> BatchedKcClimateResult {
        BatchedKcClimateResult {
            kc_values: Self::compute_cpu_batch(inputs),
            backend_used: Backend::Cpu,
        }
    }

    fn compute_cpu_batch(inputs: &[KcClimateDay]) -> Vec<f64> {
        inputs
            .iter()
            .map(|day| {
                crop::adjust_kc_for_climate(day.kc_table, day.u2, day.rh_min, day.crop_height_m)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_day() -> KcClimateDay {
        KcClimateDay {
            kc_table: 1.20,
            u2: 2.0,
            rh_min: 45.0,
            crop_height_m: 2.0,
        }
    }

    #[test]
    fn single_day() {
        let engine = BatchedKcClimate::cpu();
        let result = engine.compute(&[sample_day()]);
        assert_eq!(result.kc_values.len(), 1);
        assert!(
            (result.kc_values[0] - 1.20).abs() < 0.001,
            "no adjustment at standard conditions: {:.3}",
            result.kc_values[0]
        );
        assert_eq!(result.backend_used, Backend::Cpu);
    }

    #[test]
    fn windy_drier() {
        let engine = BatchedKcClimate::cpu();
        let day = KcClimateDay {
            kc_table: 1.20,
            u2: 4.0,
            rh_min: 30.0,
            crop_height_m: 2.0,
        };
        let result = engine.compute(&[day]);
        assert!(
            result.kc_values[0] > 1.20,
            "higher wind/lower RH increases Kc: {:.3}",
            result.kc_values[0]
        );
    }

    #[test]
    fn multiple_days() {
        let engine = BatchedKcClimate::cpu();
        let inputs: Vec<KcClimateDay> = (0..100)
            .map(|i| KcClimateDay {
                rh_min: f64::from(i).mul_add(0.1, 40.0),
                ..sample_day()
            })
            .collect();
        let result = engine.compute(&inputs);
        assert_eq!(result.kc_values.len(), 100);
        for &val in &result.kc_values {
            assert!(val >= 0.0, "Kc should be non-negative: {val}");
        }
    }

    #[test]
    fn empty_batch() {
        let engine = BatchedKcClimate::cpu();
        let result = engine.compute(&[]);
        assert!(result.kc_values.is_empty());
    }

    #[test]
    fn deterministic() {
        let engine = BatchedKcClimate::cpu();
        let inputs = vec![sample_day(); 50];
        let r1 = engine.compute(&inputs);
        let r2 = engine.compute(&inputs);
        for (a, b) in r1.kc_values.iter().zip(&r2.kc_values) {
            assert!((a - b).abs() < f64::EPSILON);
        }
    }

    #[test]
    fn compute_gpu_cpu_fallback() {
        let engine = BatchedKcClimate::cpu();
        let result = engine.compute_gpu(&[sample_day()]).unwrap();
        assert_eq!(result.kc_values.len(), 1);
        assert!((result.kc_values[0] - 1.20).abs() < 0.001);
        assert_eq!(result.backend_used, Backend::Cpu);
    }

    fn try_device() -> Option<std::sync::Arc<barracuda::device::WgpuDevice>> {
        pollster::block_on(barracuda::device::WgpuDevice::new_f64_capable())
            .ok()
            .map(std::sync::Arc::new)
    }

    #[test]
    fn compute_gpu_device_dispatch() {
        let Some(device) = try_device() else {
            eprintln!("SKIP: No GPU device for BatchedKcClimate");
            return;
        };
        let engine = BatchedKcClimate::gpu(device).unwrap();
        let result = engine.compute_gpu(&[sample_day()]).unwrap();
        assert_eq!(result.kc_values.len(), 1);
        assert!(
            (result.kc_values[0] - 1.20).abs() < 0.01,
            "GPU Kc = {:.4}",
            result.kc_values[0]
        );
        assert_eq!(result.backend_used, Backend::Gpu);
    }

    #[test]
    fn compute_gpu_matches_cpu() {
        let Some(device) = try_device() else {
            eprintln!("SKIP: No GPU device for BatchedKcClimate");
            return;
        };
        let gpu_engine = BatchedKcClimate::gpu(device).unwrap();
        let cpu_engine = BatchedKcClimate::cpu();
        let inputs: Vec<KcClimateDay> = (0..50)
            .map(|i| KcClimateDay {
                rh_min: f64::from(i as u32).mul_add(0.5, 30.0),
                ..sample_day()
            })
            .collect();
        let gpu_result = gpu_engine.compute_gpu(&inputs).unwrap();
        let cpu_result = cpu_engine.compute(&inputs);
        for (g, c) in gpu_result.kc_values.iter().zip(&cpu_result.kc_values) {
            assert!(
                (g - c).abs() < 0.01,
                "GPU {g:.6} vs CPU {c:.6}"
            );
        }
    }

    #[test]
    fn debug_format() {
        let engine = BatchedKcClimate::cpu();
        let dbg = format!("{engine:?}");
        assert!(dbg.contains("BatchedKcClimate"));
        assert!(dbg.contains("Cpu"));
    }

    #[test]
    fn default_backend_is_cpu() {
        assert_eq!(Backend::default(), Backend::Cpu);
    }
}
