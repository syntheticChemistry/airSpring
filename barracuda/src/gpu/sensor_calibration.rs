// SPDX-License-Identifier: AGPL-3.0-or-later
//! Batched `SoilWatch` 10 sensor calibration GPU orchestrator (Tier A — op=5 absorbed).
//!
//! Converts raw analog counts to volumetric water content (cm³/cm³) via Dong et al. 2024 Eq. 5:
//!
//! ```text
//! VWC = ((2×10⁻¹³ × RC − 4×10⁻⁹) × RC + 4×10⁻⁵) × RC − 0.0677
//! ```
//!
//! # Two API Levels
//!
//! | API | Device? | Backend |
//! |-----|:-------:|---------|
//! | [`BatchedSensorCal::compute`] | No | CPU via `eco::sensor_calibration::soilwatch10_vwc` |
//! | [`BatchedSensorCal::compute_gpu`] | Yes | **GPU** via `BatchedElementwiseF64` op=5 (`BarraCuda` S70+) |
//!
//! # Reference
//!
//! Dong J, Werling B, Cao R, Li B (2024) "Implementation of an In-Field `IoT` System
//! for Precision Irrigation Management" *Frontiers in Water* 6, 1353597.

use std::sync::Arc;

use barracuda::device::WgpuDevice;
use barracuda::ops::batched_elementwise_f64::{BatchedElementwiseF64, Op};

use crate::eco::sensor_calibration;

/// Per-sensor input for batched `SoilWatch` 10 calibration.
///
/// Stride-1 layout for future GPU shader: `[raw_count]`.
#[derive(Debug, Clone, Copy)]
pub struct SensorReading {
    /// Raw analog count from `SoilWatch` 10 at 3.3 V.
    pub raw_count: f64,
}

/// Backend selection for batched sensor calibration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Backend {
    /// Validated CPU path (always available).
    #[default]
    Cpu,
    /// GPU path via `BatchedElementwiseF64` op=5 (`BarraCuda` S70+ absorbed).
    Gpu,
}

/// Result from a batched sensor calibration computation.
#[derive(Debug, Clone)]
pub struct BatchedSensorCalResult {
    /// VWC values (cm³/cm³), one per input.
    pub vwc_values: Vec<f64>,
    /// Which backend was actually used.
    pub backend_used: Backend,
}

/// Batched `SoilWatch` 10 sensor calibration orchestrator.
///
/// Computes VWC for N sensor readings in a single call.
/// GPU dispatch via `BatchedElementwiseF64` op=5 (absorbed in `BarraCuda` S70+).
/// Falls back to CPU when no GPU device is configured.
pub struct BatchedSensorCal {
    backend: Backend,
    gpu_engine: Option<BatchedElementwiseF64>,
}

impl std::fmt::Debug for BatchedSensorCal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BatchedSensorCal")
            .field("backend", &self.backend)
            .field("gpu_engine", &self.gpu_engine.is_some())
            .finish()
    }
}

impl BatchedSensorCal {
    /// Create with GPU engine (Tier A — dispatches to op=5 shader).
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
    /// Used for `BarraCuda` GPU dispatch when the shader is wired.
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

    /// Compute VWC for a batch of sensor readings.
    ///
    /// Dispatches to GPU via `BatchedElementwiseF64` op=5 when a GPU engine
    /// is configured; falls back to CPU otherwise.
    ///
    /// # Errors
    ///
    /// Returns an error if the GPU dispatch fails irrecoverably.
    pub fn compute_gpu(
        &self,
        inputs: &[SensorReading],
    ) -> crate::error::Result<BatchedSensorCalResult> {
        if let Some(engine) = &self.gpu_engine {
            let packed = Self::pack_gpu_input(inputs);
            let vwc_values = engine.execute(&packed, inputs.len(), Op::SensorCalibration)?;
            Ok(BatchedSensorCalResult {
                vwc_values,
                backend_used: Backend::Gpu,
            })
        } else {
            let vwc_values = compute_cpu_batch(inputs);
            Ok(BatchedSensorCalResult {
                vwc_values,
                backend_used: Backend::Cpu,
            })
        }
    }

    /// Pack inputs into stride-1 GPU layout: `[raw_count]`.
    ///
    /// Ready for `BarraCuda` op=5 absorption — produces the flat `f64` array
    /// that `BatchedElementwiseF64::execute` expects.
    #[must_use]
    pub fn pack_gpu_input(inputs: &[SensorReading]) -> Vec<f64> {
        inputs.iter().map(|r| r.raw_count).collect()
    }

    /// Compute VWC using the validated CPU path.
    #[must_use]
    pub fn compute(&self, inputs: &[SensorReading]) -> BatchedSensorCalResult {
        BatchedSensorCalResult {
            vwc_values: compute_cpu_batch(inputs),
            backend_used: Backend::Cpu,
        }
    }
}

fn compute_cpu_batch(inputs: &[SensorReading]) -> Vec<f64> {
    inputs
        .iter()
        .map(|r| sensor_calibration::soilwatch10_vwc(r.raw_count))
        .collect()
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn reference_point() {
        let engine = BatchedSensorCal::cpu();
        let result = engine.compute(&[SensorReading {
            raw_count: 10_000.0,
        }]);
        assert!(
            (result.vwc_values[0] - 0.1323).abs() < 1e-4,
            "VWC(10000) = {}, expected ~0.1323",
            result.vwc_values[0]
        );
    }

    #[test]
    fn matches_scalar() {
        let raw = 15_000.0;
        let scalar = sensor_calibration::soilwatch10_vwc(raw);
        let engine = BatchedSensorCal::cpu();
        let batched = engine.compute(&[SensorReading { raw_count: raw }]);
        assert!(
            (batched.vwc_values[0] - scalar).abs() < f64::EPSILON,
            "Batched {} != scalar {scalar}",
            batched.vwc_values[0]
        );
    }

    #[test]
    fn multiple_readings() {
        let engine = BatchedSensorCal::cpu();
        let inputs: Vec<SensorReading> = (0..50)
            .map(|i| SensorReading {
                raw_count: f64::from(i).mul_add(500.0, 5_000.0),
            })
            .collect();
        let result = engine.compute(&inputs);
        assert_eq!(result.vwc_values.len(), 50);
    }

    #[test]
    fn empty_batch() {
        let engine = BatchedSensorCal::cpu();
        let result = engine.compute(&[]);
        assert!(result.vwc_values.is_empty());
    }

    #[test]
    fn deterministic() {
        let engine = BatchedSensorCal::cpu();
        let inputs = vec![
            SensorReading {
                raw_count: 10_000.0,
            },
            SensorReading {
                raw_count: 20_000.0,
            },
        ];
        let r1 = engine.compute(&inputs);
        let r2 = engine.compute(&inputs);
        for (a, b) in r1.vwc_values.iter().zip(&r2.vwc_values) {
            assert!((a - b).abs() < f64::EPSILON);
        }
    }

    #[test]
    fn monotonic_in_valid_range() {
        let engine = BatchedSensorCal::cpu();
        let low = engine.compute(&[SensorReading { raw_count: 5_000.0 }]);
        let high = engine.compute(&[SensorReading {
            raw_count: 20_000.0,
        }]);
        assert!(
            low.vwc_values[0] < high.vwc_values[0],
            "VWC(5000)={} should be < VWC(20000)={}",
            low.vwc_values[0],
            high.vwc_values[0]
        );
    }

    #[test]
    fn compute_gpu_cpu_fallback() {
        let engine = BatchedSensorCal::cpu();
        let result = engine
            .compute_gpu(&[SensorReading {
                raw_count: 10_000.0,
            }])
            .unwrap();
        assert_eq!(result.vwc_values.len(), 1);
        assert!((result.vwc_values[0] - 0.1323).abs() < 1e-4);
        assert_eq!(result.backend_used, Backend::Cpu);
    }

    fn try_device() -> Option<std::sync::Arc<barracuda::device::WgpuDevice>> {
        barracuda::device::test_pool::tokio_block_on(
            barracuda::device::WgpuDevice::new_f64_capable(),
        )
        .ok()
        .map(std::sync::Arc::new)
    }

    #[test]
    fn compute_gpu_device_dispatch() {
        let Some(device) = try_device() else {
            eprintln!("SKIP: No GPU device for BatchedSensorCal");
            return;
        };
        let engine = BatchedSensorCal::gpu(device).unwrap();
        let result = engine
            .compute_gpu(&[SensorReading {
                raw_count: 10_000.0,
            }])
            .unwrap();
        assert_eq!(result.vwc_values.len(), 1);
        assert!(
            (result.vwc_values[0] - 0.1323).abs() < 0.01,
            "GPU VWC(10000) = {}, expected ~0.1323",
            result.vwc_values[0]
        );
        assert_eq!(result.backend_used, Backend::Gpu);
    }

    #[test]
    fn compute_gpu_matches_cpu() {
        let Some(device) = try_device() else {
            eprintln!("SKIP: No GPU device for BatchedSensorCal");
            return;
        };
        let gpu_engine = BatchedSensorCal::gpu(device).unwrap();
        let cpu_engine = BatchedSensorCal::cpu();
        let inputs: Vec<SensorReading> = (0..50)
            .map(|i| SensorReading {
                raw_count: f64::from(i).mul_add(500.0, 5_000.0),
            })
            .collect();
        let gpu_result = gpu_engine.compute_gpu(&inputs).unwrap();
        let cpu_result = cpu_engine.compute(&inputs);
        for (g, c) in gpu_result.vwc_values.iter().zip(&cpu_result.vwc_values) {
            assert!((g - c).abs() < 0.01, "GPU {g:.6} vs CPU {c:.6}");
        }
    }

    #[test]
    fn debug_format() {
        let engine = BatchedSensorCal::cpu();
        let dbg = format!("{engine:?}");
        assert!(dbg.contains("BatchedSensorCal"));
        assert!(dbg.contains("Cpu"));
    }

    #[test]
    fn default_backend_is_cpu() {
        assert_eq!(Backend::default(), Backend::Cpu);
    }
}
