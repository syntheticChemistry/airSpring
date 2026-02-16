//! Batched water balance GPU orchestrator with CPU fallback.
//!
//! Dispatches N field-day water balance steps. Currently uses the validated
//! CPU path. The GPU shader (`batched_elementwise_f64.wgsl` op=1) implements
//! a simplified water balance update; the CPU path uses the full FAO-56 Ch. 8
//! model with `RunoffModel` capability.
//!
//! # GPU vs CPU Differences
//!
//! The shader's op=1 implements: `Dr_new = Dr_prev − P − I + Ks × Kc × ET₀`
//! with clamping at 0 and TAW. The CPU path additionally supports:
//! - `RunoffModel` (threshold-based runoff estimation)
//! - Mass balance tracking
//! - Deep percolation bookkeeping
//! - Irrigation trigger detection
//!
//! When the GPU path becomes available, it will be used for the core depletion
//! update, with runoff/mass-balance handled on the CPU side.

use crate::eco::water_balance::{self as wb, DailyInput, DailyOutput, WaterBalanceState};

/// Batched water balance orchestrator.
#[derive(Debug)]
pub struct BatchedWaterBalance {
    /// Soil parameters for the field being simulated.
    state: WaterBalanceState,
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
    /// Create a new batched water balance for a field.
    #[must_use]
    pub fn new(fc: f64, wp: f64, root_depth_mm: f64, p: f64) -> Self {
        Self {
            state: WaterBalanceState::new(fc, wp, root_depth_mm, p),
        }
    }

    /// Create from an existing state (e.g., mid-season restart).
    #[must_use]
    pub const fn from_state(state: WaterBalanceState) -> Self {
        Self { state }
    }

    /// Run a full season simulation.
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
        // Dry season — no rain, no irrigation, high ET₀
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
        // Heavy irrigation — should prevent stress
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
}
