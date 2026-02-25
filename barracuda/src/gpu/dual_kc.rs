// SPDX-License-Identifier: AGPL-3.0-or-later
//! Batched dual Kc — GPU + CPU orchestrator.
//!
//! Dispatches M fields' dual Kc computations (`Ke` + `ETc`) in parallel.
//!
//! - **GPU**: Per-timestep `Ke` batch across M independent fields via
//!   `BatchedElementwiseF64` (Tier B — pending `ToadStool` primitive absorption).
//! - **CPU**: Sequential multi-day simulation using validated `eco::dual_kc`.
//!
//! # GPU readiness
//!
//! The CPU path is fully validated against FAO-56 Ch 7+11. The GPU interface
//! is wired and ready for `ToadStool` absorption. Once `ToadStool` adds a
//! `dual_kc_ke_batch` shader operation, the GPU path activates automatically.
//!
//! # Mulch support
//!
//! Both CPU and GPU paths support optional mulch factors for no-till systems
//! (FAO-56 Ch 11). Pass `mulch_factor = 1.0` for conventional tillage.

use crate::eco::dual_kc::{self, DualKcInput, DualKcOutput, EvaporationLayerState};

/// Per-field configuration for batched dual Kc.
#[derive(Debug, Clone, Copy)]
pub struct FieldDualKcConfig {
    /// Basal crop coefficient for the current growth stage.
    pub kcb: f64,
    /// Maximum Kc (climate-adjusted via FAO-56 Eq. 72).
    pub kc_max: f64,
    /// Fraction of soil surface exposed to wetting and evaporation.
    pub few: f64,
    /// Mulch reduction factor (1.0 = bare, 0.25 = full mulch).
    pub mulch_factor: f64,
    /// Current evaporation layer state.
    pub state: EvaporationLayerState,
}

/// Result from a batched dual Kc computation across M fields.
#[derive(Debug, Clone)]
pub struct BatchedDualKcResult {
    /// Per-field outputs (one `DualKcOutput` per field).
    pub outputs: Vec<DualKcOutput>,
    /// Updated evaporation layer states (one per field).
    pub states: Vec<EvaporationLayerState>,
}

/// Batched dual Kc orchestrator — M fields in parallel.
///
/// Computes `Ke` and `ETc` for M independent fields sharing the same
/// daily weather (ET₀, precipitation, irrigation). Each field has
/// its own crop parameters and evaporation layer state.
#[derive(Debug)]
pub struct BatchedDualKc {
    configs: Vec<FieldDualKcConfig>,
}

impl BatchedDualKc {
    /// Create a new batched dual Kc orchestrator for M fields.
    #[must_use]
    pub fn new(configs: Vec<FieldDualKcConfig>) -> Self {
        Self { configs }
    }

    /// Compute one timestep across all fields (CPU path).
    ///
    /// Each field independently computes `Ke` and `ETc` using the shared
    /// daily weather input and its own crop/soil parameters.
    #[must_use]
    pub fn step_cpu(&mut self, input: &DualKcInput) -> BatchedDualKcResult {
        let mut outputs = Vec::with_capacity(self.configs.len());
        let mut states = Vec::with_capacity(self.configs.len());

        for config in &mut self.configs {
            let (day_outputs, new_state) = if (config.mulch_factor - 1.0).abs() < f64::EPSILON {
                dual_kc::simulate_dual_kc(
                    std::slice::from_ref(input),
                    config.kcb,
                    config.kc_max,
                    config.few,
                    &config.state,
                )
            } else {
                dual_kc::simulate_dual_kc_mulched(
                    std::slice::from_ref(input),
                    config.kcb,
                    config.kc_max,
                    config.few,
                    config.mulch_factor,
                    &config.state,
                )
            };

            outputs.push(day_outputs[0]);
            config.state = new_state;
            states.push(new_state);
        }

        BatchedDualKcResult { outputs, states }
    }

    /// Run a multi-day season simulation across all fields.
    ///
    /// Returns per-field seasonal totals (`ETc`, `Ke`, etc.).
    #[must_use]
    pub fn simulate_season(&mut self, inputs: &[DualKcInput]) -> Vec<SeasonFieldSummary> {
        let mut summaries: Vec<SeasonFieldSummary> = self
            .configs
            .iter()
            .map(|_| SeasonFieldSummary::default())
            .collect();

        for input in inputs {
            let result = self.step_cpu(input);
            for (i, out) in result.outputs.iter().enumerate() {
                summaries[i].total_etc += out.etc;
                summaries[i].total_ke += out.ke;
                summaries[i].days += 1;
                summaries[i].final_de = out.de;
            }
        }

        summaries
    }
}

/// Seasonal summary for one field.
#[derive(Debug, Clone, Default)]
pub struct SeasonFieldSummary {
    /// Total crop evapotranspiration (mm).
    pub total_etc: f64,
    /// Total soil evaporation component (mm as Ke × ET₀ equivalent).
    pub total_ke: f64,
    /// Number of simulated days.
    pub days: usize,
    /// Final evaporation layer depletion (mm).
    pub final_de: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn silt_loam_state() -> EvaporationLayerState {
        EvaporationLayerState {
            de: 0.0,
            tew: 22.5,
            rew: 9.0,
        }
    }

    #[test]
    fn test_single_field_matches_eco() {
        let inputs: Vec<DualKcInput> = [5.0, 5.5, 4.8]
            .iter()
            .map(|&et0| DualKcInput {
                et0,
                precipitation: 0.0,
                irrigation: 0.0,
            })
            .collect();

        let state = silt_loam_state();
        let (eco_out, _) = dual_kc::simulate_dual_kc(&inputs, 1.15, 1.20, 0.05, &state);

        let mut batch = BatchedDualKc::new(vec![FieldDualKcConfig {
            kcb: 1.15,
            kc_max: 1.20,
            few: 0.05,
            mulch_factor: 1.0,
            state,
        }]);

        for (i, inp) in inputs.iter().enumerate() {
            let result = batch.step_cpu(inp);
            assert!(
                (result.outputs[0].etc - eco_out[i].etc).abs() < 1e-10,
                "day {i}: batch {:.6} != eco {:.6}",
                result.outputs[0].etc,
                eco_out[i].etc
            );
        }
    }

    #[test]
    fn test_mulched_field_less_et() {
        let inputs: Vec<DualKcInput> = (0..30)
            .map(|i| DualKcInput {
                et0: 5.0,
                precipitation: if i == 0 { 10.0 } else { 0.0 },
                irrigation: 0.0,
            })
            .collect();

        let mut batch = BatchedDualKc::new(vec![
            FieldDualKcConfig {
                kcb: 0.15,
                kc_max: 1.20,
                few: 1.0,
                mulch_factor: 1.0,
                state: silt_loam_state(),
            },
            FieldDualKcConfig {
                kcb: 0.15,
                kc_max: 1.20,
                few: 1.0,
                mulch_factor: 0.40,
                state: silt_loam_state(),
            },
        ]);

        let summaries = batch.simulate_season(&inputs);
        assert!(
            summaries[1].total_etc < summaries[0].total_etc,
            "Mulched ({:.1}) < bare ({:.1})",
            summaries[1].total_etc,
            summaries[0].total_etc
        );
    }

    #[test]
    fn test_multi_field_independence() {
        let input = DualKcInput {
            et0: 5.0,
            precipitation: 0.0,
            irrigation: 0.0,
        };

        let mut batch = BatchedDualKc::new(vec![
            FieldDualKcConfig {
                kcb: 1.15,
                kc_max: 1.20,
                few: 0.05,
                mulch_factor: 1.0,
                state: silt_loam_state(),
            },
            FieldDualKcConfig {
                kcb: 0.15,
                kc_max: 1.20,
                few: 1.0,
                mulch_factor: 1.0,
                state: silt_loam_state(),
            },
        ]);

        let result = batch.step_cpu(&input);
        assert!(
            (result.outputs[0].ke - result.outputs[1].ke).abs() > 0.01,
            "Different Kcb should partition Ke differently: {:.4} vs {:.4}",
            result.outputs[0].ke,
            result.outputs[1].ke
        );
    }

    #[test]
    fn test_season_simulation_days() {
        let inputs: Vec<DualKcInput> = (0..180)
            .map(|_| DualKcInput {
                et0: 5.0,
                precipitation: 0.0,
                irrigation: 0.0,
            })
            .collect();

        let mut batch = BatchedDualKc::new(vec![FieldDualKcConfig {
            kcb: 1.15,
            kc_max: 1.20,
            few: 0.05,
            mulch_factor: 1.0,
            state: silt_loam_state(),
        }]);

        let summaries = batch.simulate_season(&inputs);
        assert_eq!(summaries[0].days, 180);
        assert!(summaries[0].total_etc > 0.0);
    }

    #[test]
    fn test_empty_fields() {
        let mut batch = BatchedDualKc::new(vec![]);
        let input = DualKcInput {
            et0: 5.0,
            precipitation: 0.0,
            irrigation: 0.0,
        };
        let result = batch.step_cpu(&input);
        assert!(result.outputs.is_empty());
    }
}
