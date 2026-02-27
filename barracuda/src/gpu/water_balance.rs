// SPDX-License-Identifier: AGPL-3.0-or-later
//! Batched water balance — GPU + CPU orchestrator.
//!
//! Dispatches N field-day water balance steps. Two paths:
//!
//! - **GPU**: [`BatchedWaterBalance::gpu_step`] dispatches a single timestep across
//!   M fields in parallel via [`BatchedElementwiseF64::water_balance_batch()`].
//! - **CPU**: [`BatchedWaterBalance::simulate_season`] runs the full FAO-56 Ch. 8
//!   sequential model with `RunoffModel`, mass balance tracking, percolation, and
//!   irrigation trigger detection.
//!
//! # GPU vs CPU
//!
//! The GPU path computes the core depletion update for M independent fields
//! simultaneously (one workgroup per field). The CPU path handles the sequential
//! day-by-day simulation for a single field including full bookkeeping.
//!
//! Use GPU when you have M >> 1 fields at the same timestep.
//! Use CPU for a single field's season simulation.

use std::sync::Arc;

use barracuda::device::WgpuDevice;
use barracuda::ops::batched_elementwise_f64::{self as bef64, BatchedElementwiseF64};

use crate::eco::water_balance::{self as wb, DailyInput, DailyOutput, WaterBalanceState};

/// One field's daily state for the GPU water balance step.
///
/// Maps to `ToadStool` `WaterBalanceInput`: `(dr_prev, P, I, ETc, TAW, RAW, p)`.
#[derive(Debug, Clone, Copy)]
pub struct FieldDayInput {
    /// Previous day's root zone depletion (mm).
    pub dr_prev: f64,
    /// Precipitation (mm).
    pub precipitation: f64,
    /// Irrigation applied (mm).
    pub irrigation: f64,
    /// Crop evapotranspiration `ETc` = Kc × ET₀ (mm).
    pub etc: f64,
    /// Total available water (mm).
    pub taw: f64,
    /// Readily available water (mm).
    pub raw: f64,
    /// Depletion fraction p.
    pub p: f64,
}

impl FieldDayInput {
    /// Convert to `ToadStool` `WaterBalanceInput` tuple.
    #[must_use]
    pub const fn to_toadstool(self) -> bef64::WaterBalanceInput {
        (
            self.dr_prev,
            self.precipitation,
            self.irrigation,
            self.etc,
            self.taw,
            self.raw,
            self.p,
        )
    }
}

/// Batched water balance orchestrator — GPU + CPU.
pub struct BatchedWaterBalance {
    /// Soil parameters for the field being simulated (CPU season path).
    state: WaterBalanceState,
    /// GPU engine (optional).
    gpu_engine: Option<BatchedElementwiseF64>,
}

impl std::fmt::Debug for BatchedWaterBalance {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BatchedWaterBalance")
            .field("state", &self.state)
            .field("gpu_engine", &self.gpu_engine.is_some())
            .finish()
    }
}

/// Season-level summary from batched simulation.
#[derive(Debug, Clone)]
pub struct SeasonSummary {
    /// Total actual ET over the season (mm).
    pub total_actual_et: f64,
    /// Total precipitation (mm).
    pub total_precipitation: f64,
    /// Total deep percolation (mm).
    pub total_deep_percolation: f64,
    /// Number of days with water stress (Ks < 1).
    pub stress_days: usize,
    /// Mass balance error (mm) — should be < 0.01.
    pub mass_balance_error: f64,
    /// Final root zone depletion (mm).
    pub final_depletion: f64,
    /// Daily outputs for detailed analysis.
    pub daily_outputs: Vec<DailyOutput>,
}

impl BatchedWaterBalance {
    /// Create a new CPU-only batched water balance for a field.
    #[must_use]
    pub fn new(fc: f64, wp: f64, root_depth_mm: f64, p: f64) -> Self {
        Self {
            state: WaterBalanceState::new(fc, wp, root_depth_mm, p),
            gpu_engine: None,
        }
    }

    /// Create a GPU-backed batched water balance.
    ///
    /// # Errors
    ///
    /// Returns an error if `BatchedElementwiseF64` cannot be initialised.
    pub fn with_gpu(
        fc: f64,
        wp: f64,
        root_depth_mm: f64,
        p: f64,
        device: Arc<WgpuDevice>,
    ) -> crate::error::Result<Self> {
        let engine = BatchedElementwiseF64::new(device)
            .map_err(|e| crate::error::AirSpringError::Barracuda(format!("{e}")))?;
        Ok(Self {
            state: WaterBalanceState::new(fc, wp, root_depth_mm, p),
            gpu_engine: Some(engine),
        })
    }

    /// Create from an existing state (e.g., mid-season restart).
    #[must_use]
    pub const fn from_state(state: WaterBalanceState) -> Self {
        Self {
            state,
            gpu_engine: None,
        }
    }

    /// GPU step: compute one timestep across M fields in parallel.
    ///
    /// Returns the new depletion for each field. This is the GPU-accelerated
    /// core depletion update without full bookkeeping (no mass balance, no
    /// runoff model — those stay on CPU).
    ///
    /// Falls back to CPU if no GPU engine is available.
    ///
    /// # Errors
    ///
    /// Returns an error if GPU dispatch fails.
    pub fn gpu_step(&self, fields: &[FieldDayInput]) -> crate::error::Result<Vec<f64>> {
        self.gpu_engine.as_ref().map_or_else(
            || {
                // CPU fallback: simplified depletion update matching shader logic.
                // The shader applies Ks internally: ETc_actual = Ks * ETc
                Ok(fields
                    .iter()
                    .map(|f| {
                        let ks = if f.dr_prev > f.raw {
                            (f.taw - f.dr_prev) / (f.taw - f.raw)
                        } else {
                            1.0
                        };
                        let dr = f.dr_prev - f.precipitation - f.irrigation + ks * f.etc;
                        dr.clamp(0.0, f.taw)
                    })
                    .collect())
            },
            |engine| {
                let inputs: Vec<bef64::WaterBalanceInput> =
                    fields.iter().map(|f| f.to_toadstool()).collect();
                engine
                    .water_balance_batch(&inputs)
                    .map_err(|e| crate::error::AirSpringError::Barracuda(format!("{e}")))
            },
        )
    }

    /// Run a full season simulation on CPU (sequential, full bookkeeping).
    #[must_use]
    pub fn simulate_season(&self, daily_inputs: &[DailyInput]) -> SeasonSummary {
        let initial_depletion = self.state.depletion;
        let (final_state, outputs) = wb::simulate_season(&self.state, daily_inputs);

        let total_actual_et: f64 = outputs.iter().map(|o| o.actual_et).sum();
        let total_precipitation: f64 = daily_inputs.iter().map(|d| d.precipitation).sum();
        let total_deep_percolation: f64 = outputs.iter().map(|o| o.deep_percolation).sum();
        let stress_days = outputs.iter().filter(|o| o.ks < 1.0).count();
        let mass_balance_error = wb::mass_balance_check(
            daily_inputs,
            &outputs,
            initial_depletion,
            final_state.depletion,
        );

        SeasonSummary {
            total_actual_et,
            total_precipitation,
            total_deep_percolation,
            stress_days,
            mass_balance_error,
            final_depletion: final_state.depletion,
            daily_outputs: outputs,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_season_mass_balance() {
        let engine = BatchedWaterBalance::new(0.30, 0.10, 500.0, 0.5);
        let inputs: Vec<DailyInput> = (0..60)
            .map(|day| DailyInput {
                precipitation: if day % 5 == 0 { 10.0 } else { 0.0 },
                irrigation: 0.0,
                et0: 4.0,
                kc: 1.0,
            })
            .collect();

        let summary = engine.simulate_season(&inputs);
        assert!(
            summary.mass_balance_error < 0.01,
            "Mass balance error: {}",
            summary.mass_balance_error
        );
        assert_eq!(summary.daily_outputs.len(), 60);
    }

    #[test]
    fn test_season_stress_detection() {
        let engine = BatchedWaterBalance::new(0.30, 0.10, 500.0, 0.5);
        let inputs: Vec<DailyInput> = (0..30)
            .map(|_| DailyInput {
                precipitation: 0.0,
                irrigation: 0.0,
                et0: 6.0,
                kc: 1.2,
            })
            .collect();

        let summary = engine.simulate_season(&inputs);
        assert!(summary.stress_days > 0, "Should detect water stress");
        assert!(
            summary.final_depletion > 0.0,
            "Final depletion should be positive"
        );
    }

    #[test]
    fn test_season_irrigated_no_stress() {
        let engine = BatchedWaterBalance::new(0.30, 0.10, 500.0, 0.5);
        let inputs: Vec<DailyInput> = (0..30)
            .map(|_| DailyInput {
                precipitation: 0.0,
                irrigation: 50.0,
                et0: 5.0,
                kc: 1.0,
            })
            .collect();

        let summary = engine.simulate_season(&inputs);
        assert_eq!(
            summary.stress_days, 0,
            "Over-irrigated should have no stress"
        );
    }

    #[test]
    fn test_gpu_step_cpu_fallback() {
        let engine = BatchedWaterBalance::new(0.30, 0.10, 500.0, 0.5);
        let fields = vec![
            FieldDayInput {
                dr_prev: 20.0, // Dr < RAW → Ks = 1.0
                precipitation: 5.0,
                irrigation: 0.0,
                etc: 4.0,
                taw: 100.0,
                raw: 50.0,
                p: 0.5,
            },
            FieldDayInput {
                dr_prev: 50.0, // Dr = RAW → Ks = 1.0
                precipitation: 0.0,
                irrigation: 25.0,
                etc: 6.0,
                taw: 120.0,
                raw: 48.0,
                p: 0.4,
            },
        ];
        let results = engine.gpu_step(&fields).unwrap();
        assert_eq!(results.len(), 2);
        // Field 0: Ks=1.0 (Dr=20 < RAW=50), Dr_new = 20 - 5 - 0 + 1.0*4 = 19
        assert!((results[0] - 19.0).abs() < 1e-10, "Field 0: {}", results[0]);
        // Field 1: Ks = (120-50)/(120-48) = 70/72 ≈ 0.972, ETc_a = 0.972*6 = 5.833
        // Dr_new = 50 - 0 - 25 + 5.833 = 30.833
        let ks1 = (120.0 - 50.0) / (120.0 - 48.0);
        let expected1 = 50.0 - 0.0 - 25.0 + ks1 * 6.0;
        assert!(
            (results[1] - expected1).abs() < 1e-10,
            "Field 1: {} expected {}",
            results[1],
            expected1
        );
    }

    #[test]
    fn test_gpu_step_clamp() {
        let engine = BatchedWaterBalance::new(0.30, 0.10, 500.0, 0.5);
        // Heavy rain drives depletion negative → clamp to 0
        let fields = vec![FieldDayInput {
            dr_prev: 5.0,
            precipitation: 50.0,
            irrigation: 0.0,
            etc: 3.0,
            taw: 100.0,
            raw: 50.0,
            p: 0.5,
        }];
        let results = engine.gpu_step(&fields).unwrap();
        assert!(
            (results[0]).abs() < 1e-10,
            "Should clamp to 0: {}",
            results[0]
        );
    }

    #[test]
    fn test_field_day_input_to_toadstool() {
        let f = FieldDayInput {
            dr_prev: 20.0,
            precipitation: 5.0,
            irrigation: 10.0,
            etc: 4.0,
            taw: 100.0,
            raw: 50.0,
            p: 0.5,
        };
        let tt = f.to_toadstool();
        assert!((tt.0 - 20.0).abs() < f64::EPSILON);
        assert!((tt.6 - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_from_state() {
        let state = WaterBalanceState::new(0.30, 0.10, 500.0, 0.5);
        let engine = BatchedWaterBalance::from_state(state);
        let dbg = format!("{engine:?}");
        assert!(dbg.contains("BatchedWaterBalance"));
    }

    #[test]
    fn test_gpu_step_empty() {
        let engine = BatchedWaterBalance::new(0.30, 0.10, 500.0, 0.5);
        let results = engine.gpu_step(&[]).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_season_deep_percolation() {
        let engine = BatchedWaterBalance::new(0.30, 0.10, 500.0, 0.5);
        let inputs: Vec<DailyInput> = (0..10)
            .map(|_| DailyInput {
                precipitation: 30.0,
                irrigation: 0.0,
                et0: 2.0,
                kc: 1.0,
            })
            .collect();
        let summary = engine.simulate_season(&inputs);
        assert!(
            summary.total_deep_percolation > 0.0,
            "Heavy rain should cause deep percolation"
        );
    }

    #[test]
    fn test_gpu_step_clamp_to_taw() {
        let engine = BatchedWaterBalance::new(0.30, 0.10, 500.0, 0.5);
        // High ET, no water → depletion maxes at TAW
        let fields = vec![FieldDayInput {
            dr_prev: 95.0,
            precipitation: 0.0,
            irrigation: 0.0,
            etc: 20.0,
            taw: 100.0,
            raw: 50.0,
            p: 0.5,
        }];
        let results = engine.gpu_step(&fields).unwrap();
        assert!(
            results[0] <= 100.0 + 1e-10,
            "Clamped to TAW: {}",
            results[0]
        );
    }

    #[test]
    fn test_simulate_season_single_day() {
        let engine = BatchedWaterBalance::new(0.30, 0.10, 500.0, 0.5);
        let inputs = vec![DailyInput {
            precipitation: 5.0,
            irrigation: 10.0,
            et0: 4.0,
            kc: 1.0,
        }];
        let summary = engine.simulate_season(&inputs);
        assert_eq!(summary.daily_outputs.len(), 1);
        assert!(summary.mass_balance_error < 0.01);
        assert!((summary.total_precipitation - 5.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_simulate_season_empty_input() {
        let engine = BatchedWaterBalance::new(0.30, 0.10, 500.0, 0.5);
        let summary = engine.simulate_season(&[]);
        assert!(summary.daily_outputs.is_empty());
        assert!((summary.total_actual_et).abs() < 1e-10);
        assert!((summary.total_precipitation).abs() < 1e-10);
        assert!((summary.mass_balance_error).abs() < 1e-10);
    }

    #[test]
    fn test_season_mass_balance_explicit() {
        let engine = BatchedWaterBalance::new(0.30, 0.10, 500.0, 0.5);
        let inputs: Vec<DailyInput> = (0..20)
            .map(|day| DailyInput {
                precipitation: if day % 3 == 0 { 8.0 } else { 0.0 },
                irrigation: if day == 10 { 30.0 } else { 0.0 },
                et0: 3.5,
                kc: 1.0,
            })
            .collect();
        let summary = engine.simulate_season(&inputs);
        assert!(
            summary.mass_balance_error < 0.01,
            "Mass balance error: {}",
            summary.mass_balance_error
        );
        let total_irrigation: f64 = inputs.iter().map(|d| d.irrigation).sum();
        let inflow = summary.total_precipitation + total_irrigation;
        let total_runoff: f64 = summary.daily_outputs.iter().map(|o| o.runoff).sum();
        let outflow = summary.total_actual_et + summary.total_deep_percolation + total_runoff;
        let storage_change = 0.0 - summary.final_depletion;
        let residual = (inflow - outflow - storage_change).abs();
        assert!(
            residual < 0.01,
            "Explicit mass balance residual: {residual}"
        );
    }

    fn try_device() -> Option<std::sync::Arc<barracuda::device::WgpuDevice>> {
        pollster::block_on(barracuda::device::WgpuDevice::new_f64_capable())
            .ok()
            .map(std::sync::Arc::new)
    }

    #[test]
    fn test_gpu_step_device_empty() {
        let Some(device) = try_device() else {
            eprintln!("SKIP: No GPU device for BatchedWaterBalance");
            return;
        };
        let engine = BatchedWaterBalance::with_gpu(0.30, 0.10, 500.0, 0.5, device).unwrap();
        let results = engine.gpu_step(&[]).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_gpu_step_device_single_field() {
        let Some(device) = try_device() else {
            eprintln!("SKIP: No GPU device for BatchedWaterBalance");
            return;
        };
        let engine = BatchedWaterBalance::with_gpu(0.30, 0.10, 500.0, 0.5, device).unwrap();
        let fields = vec![FieldDayInput {
            dr_prev: 20.0,
            precipitation: 5.0,
            irrigation: 0.0,
            etc: 4.0,
            taw: 100.0,
            raw: 50.0,
            p: 0.5,
        }];
        let results = engine.gpu_step(&fields).unwrap();
        assert_eq!(results.len(), 1);
        let expected = 1.0f64.mul_add(4.0, 20.0 - 5.0 - 0.0);
        assert!(
            (results[0] - expected).abs() < 1e-6,
            "GPU {} vs expected {}",
            results[0],
            expected
        );
    }

    #[test]
    fn test_gpu_step_device_matches_cpu() {
        let Some(device) = try_device() else {
            eprintln!("SKIP: No GPU device for BatchedWaterBalance");
            return;
        };
        let gpu_engine = BatchedWaterBalance::with_gpu(0.30, 0.10, 500.0, 0.5, device).unwrap();
        let cpu_engine = BatchedWaterBalance::new(0.30, 0.10, 500.0, 0.5);
        let fields: Vec<FieldDayInput> = (0..100)
            .map(|i| FieldDayInput {
                dr_prev: f64::from(i) * 0.5,
                precipitation: 2.0,
                irrigation: 0.0,
                etc: 5.0,
                taw: 100.0,
                raw: 50.0,
                p: 0.5,
            })
            .collect();
        let gpu_results = gpu_engine.gpu_step(&fields).unwrap();
        let cpu_results = cpu_engine.gpu_step(&fields).unwrap();
        assert_eq!(gpu_results.len(), cpu_results.len());
        for (g, c) in gpu_results.iter().zip(&cpu_results) {
            assert!((g - c).abs() < 1e-6, "GPU {g} vs CPU {c}");
        }
    }

    #[test]
    fn test_gpu_step_device_large_batch() {
        let Some(device) = try_device() else {
            eprintln!("SKIP: No GPU device for BatchedWaterBalance");
            return;
        };
        let engine = BatchedWaterBalance::with_gpu(0.30, 0.10, 500.0, 0.5, device).unwrap();
        let fields: Vec<FieldDayInput> = (0..500)
            .map(|i| FieldDayInput {
                dr_prev: (f64::from(i) % 80.0),
                precipitation: 1.0,
                irrigation: 0.0,
                etc: 4.0,
                taw: 100.0,
                raw: 50.0,
                p: 0.5,
            })
            .collect();
        let results = engine.gpu_step(&fields).unwrap();
        assert_eq!(results.len(), 500);
        for &dr in &results {
            assert!(
                (0.0..=100.0 + 1e-6).contains(&dr),
                "Depletion out of range: {dr}"
            );
        }
    }

    #[test]
    fn test_with_gpu_debug_format() {
        let Some(device) = try_device() else {
            eprintln!("SKIP: No GPU device for BatchedWaterBalance");
            return;
        };
        let engine = BatchedWaterBalance::with_gpu(0.30, 0.10, 500.0, 0.5, device).unwrap();
        let dbg = format!("{engine:?}");
        assert!(dbg.contains("BatchedWaterBalance"));
        assert!(dbg.contains("true"));
    }
}
