// SPDX-License-Identifier: AGPL-3.0-or-later
//! GPU-accelerated Thornthwaite (1948) monthly ET₀ batch computation.
//!
//! Wraps `BatchedElementwiseF64` op 11 (`ThornthwaiteEt0`) from `ToadStool` S79.
//! CPU reference: `barracuda::ops::batched_elementwise_f64::cpu_ref::thornthwaite_et0_cpu`.

use std::sync::Arc;

use barracuda::device::WgpuDevice;
use barracuda::ops::batched_elementwise_f64::{thornthwaite_et0_cpu, BatchedElementwiseF64, Op};

#[cfg(test)]
use super::device_info::try_f64_device;

/// Per-month input for batched Thornthwaite ET₀.
///
/// Stride-5 layout: `[heat_index_I, exponent_a, daylight_hours_N, days_in_month_d, T_mean]`.
#[derive(Debug, Clone, Copy)]
pub struct ThornthwaiteInput {
    /// Annual heat index I.
    pub heat_index: f64,
    /// Thornthwaite exponent a.
    pub exponent_a: f64,
    /// Mean daylight hours for the month.
    pub daylight_hours: f64,
    /// Days in the month.
    pub days_in_month: f64,
    /// Mean monthly temperature (°C).
    pub tmean: f64,
}

/// Batched Thornthwaite ET₀ GPU orchestrator.
///
/// Dispatches to `BatchedElementwiseF64` op 11 when a GPU engine is configured;
/// falls back to CPU otherwise.
pub struct BatchedThornthwaite {
    gpu_engine: Option<BatchedElementwiseF64>,
}

impl std::fmt::Debug for BatchedThornthwaite {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BatchedThornthwaite")
            .field("gpu_engine", &self.gpu_engine.is_some())
            .finish()
    }
}

impl BatchedThornthwaite {
    /// Create with GPU engine (dispatches to op 11).
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

    /// Compute Thornthwaite ET₀ for a batch of monthly inputs.
    ///
    /// Returns ET₀ values in mm/month. Falls back to CPU when no GPU engine.
    ///
    /// # Errors
    ///
    /// Returns an error if GPU dispatch fails.
    pub fn compute_gpu(
        &self,
        monthly_inputs: &[ThornthwaiteInput],
    ) -> crate::error::Result<Vec<f64>> {
        if monthly_inputs.is_empty() {
            return Ok(Vec::new());
        }
        if let Some(engine) = &self.gpu_engine {
            let packed = Self::pack_input(monthly_inputs);
            Ok(engine.execute(&packed, monthly_inputs.len(), Op::ThornthwaiteEt0)?)
        } else {
            Ok(compute_thornthwaite_cpu(monthly_inputs))
        }
    }

    fn pack_input(inputs: &[ThornthwaiteInput]) -> Vec<f64> {
        let mut data = Vec::with_capacity(inputs.len() * 5);
        for i in inputs {
            data.push(i.heat_index);
            data.push(i.exponent_a);
            data.push(i.daylight_hours);
            data.push(i.days_in_month);
            data.push(i.tmean);
        }
        data
    }
}

/// CPU fallback for Thornthwaite ET₀ batch.
#[must_use]
pub fn compute_thornthwaite_cpu(inputs: &[ThornthwaiteInput]) -> Vec<f64> {
    inputs
        .iter()
        .map(|i| {
            thornthwaite_et0_cpu(
                i.heat_index,
                i.exponent_a,
                i.daylight_hours,
                i.days_in_month,
                i.tmean,
            )
        })
        .collect()
}

#[cfg(test)]
#[expect(
    clippy::suboptimal_flops,
    reason = "test code matches reference Thornthwaite formula term-by-term"
)]
mod tests {
    use super::*;

    fn try_device() -> Option<Arc<WgpuDevice>> {
        try_f64_device()
    }

    fn sample_input() -> ThornthwaiteInput {
        ThornthwaiteInput {
            heat_index: 100.0,
            exponent_a: 0.16,
            daylight_hours: 12.0,
            days_in_month: 30.0,
            tmean: 20.0,
        }
    }

    #[test]
    fn test_cpu_et0_positive() {
        let engine = BatchedThornthwaite::cpu();
        let result = engine.compute_gpu(&[sample_input()]).unwrap();
        assert_eq!(result.len(), 1);
        assert!(
            result[0] > 10.0 && result[0] < 50.0,
            "ET₀={} mm/month expected 10-50",
            result[0]
        );
    }

    #[test]
    fn test_gpu_matches_cpu() {
        let Some(device) = try_device() else {
            eprintln!("SKIP: No GPU device for BatchedThornthwaite");
            return;
        };
        let gpu_engine = BatchedThornthwaite::gpu(device).unwrap();
        let cpu_engine = BatchedThornthwaite::cpu();

        let inputs: Vec<ThornthwaiteInput> = (0..12)
            .map(|m| ThornthwaiteInput {
                heat_index: 100.0,
                exponent_a: 0.16,
                daylight_hours: 10.0 + f64::from(m) * 0.5,
                days_in_month: if m == 1 { 28.0 } else { 30.0 },
                tmean: 5.0 + f64::from(m) * 2.0,
            })
            .collect();

        let gpu_result = gpu_engine.compute_gpu(&inputs).unwrap();
        let cpu_result = cpu_engine.compute_gpu(&inputs).unwrap();

        assert_eq!(gpu_result.len(), cpu_result.len());
        for (g, c) in gpu_result.iter().zip(&cpu_result) {
            assert!((g - c).abs() < 1.0, "GPU ET₀={g:.4} vs CPU ET₀={c:.4}");
        }
    }

    #[test]
    fn test_empty_batch() {
        let engine = BatchedThornthwaite::cpu();
        let result = engine.compute_gpu(&[]).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_freezing_zero() {
        let engine = BatchedThornthwaite::cpu();
        let cold = ThornthwaiteInput {
            tmean: -5.0,
            ..sample_input()
        };
        let result = engine.compute_gpu(&[cold]).unwrap();
        assert_eq!(result.len(), 1);
        assert!(result[0].abs() < f64::EPSILON, "T≤0 should give ET₀≈0");
    }
}
