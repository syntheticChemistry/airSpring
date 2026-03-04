// SPDX-License-Identifier: AGPL-3.0-or-later
//! Batched dual Kc — GPU + CPU orchestrator.
//!
//! Dispatches M fields' dual Kc computations (`Ke` + `ETc`) in parallel.
//!
//! - **GPU**: Per-timestep `Ke` batch across M independent fields via
//!   `BatchedElementwiseF64` op=8 (`BarraCuda` S70+ absorbed).
//! - **CPU**: Sequential multi-day simulation using validated `eco::dual_kc`.
//!
//! # GPU dispatch
//!
//! The CPU path is fully validated against FAO-56 Ch 7+11. The GPU path
//! dispatches `Ke` computation to `BarraCuda` `BatchedElementwiseF64` op=8
//! (stride=9, absorbed in S70+). State updates remain on CPU.
//!
//! # Mulch support
//!
//! Both CPU and GPU paths support optional mulch factors for no-till systems
//! (FAO-56 Ch 11). Pass `mulch_factor = 1.0` for conventional tillage.

use std::sync::Arc;

use barracuda::device::WgpuDevice;
use barracuda::ops::batched_elementwise_f64::{BatchedElementwiseF64, Op};

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
///
/// # GPU Dispatch (Tier A — op=8 absorbed in `BarraCuda` S70+)
///
/// The GPU path packs per-field state into stride-9 vectors:
/// `[kcb, kc_max, few, mulch_factor, de_prev, rew, tew, p_eff, et0]`
/// and dispatches to `BatchedElementwiseF64` op=8 for `Ke` computation.
/// Falls back to CPU when no GPU device is configured.
pub struct BatchedDualKc {
    configs: Vec<FieldDualKcConfig>,
    gpu_engine: Option<BatchedElementwiseF64>,
}

impl std::fmt::Debug for BatchedDualKc {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BatchedDualKc")
            .field("fields", &self.configs.len())
            .field("gpu_engine", &self.gpu_engine.is_some())
            .finish()
    }
}

impl BatchedDualKc {
    /// Returns a reference to the GPU engine, if available.
    /// Used for `BarraCuda` GPU dispatch when the shader is wired.
    #[must_use]
    pub const fn gpu_engine(&self) -> Option<&BatchedElementwiseF64> {
        self.gpu_engine.as_ref()
    }

    /// Create a new batched dual Kc orchestrator for M fields (CPU only).
    #[expect(
        clippy::missing_const_for_fn,
        reason = "method may gain runtime logic when GPU dispatch is wired"
    )]
    #[must_use]
    pub fn new(configs: Vec<FieldDualKcConfig>) -> Self {
        Self {
            configs,
            gpu_engine: None,
        }
    }

    /// Create with GPU engine (Tier B — currently falls back to CPU).
    ///
    /// # Errors
    ///
    /// Returns an error if `BatchedElementwiseF64` cannot be initialised.
    pub fn with_gpu(
        configs: Vec<FieldDualKcConfig>,
        device: Arc<WgpuDevice>,
    ) -> crate::error::Result<Self> {
        let engine = BatchedElementwiseF64::new(device)?;
        Ok(Self {
            configs,
            gpu_engine: Some(engine),
        })
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

    /// Compute one timestep across all fields (GPU path).
    ///
    /// Dispatches M-field `Ke` computations to GPU via `BatchedElementwiseF64`
    /// op=8. The GPU returns raw `Ke` values; state update remains on CPU to
    /// maintain the sequential dependency chain of `de_prev`.
    ///
    /// # Errors
    ///
    /// Returns an error if the GPU dispatch fails irrecoverably.
    pub fn step_gpu(&mut self, input: &DualKcInput) -> crate::error::Result<BatchedDualKcResult> {
        if let Some(engine) = &self.gpu_engine {
            let packed = self.pack_gpu_timestep(input);
            let ke_values = engine.execute(&packed, self.configs.len(), Op::DualKcKe)?;

            let mut outputs = Vec::with_capacity(self.configs.len());
            let mut states = Vec::with_capacity(self.configs.len());

            for (i, config) in self.configs.iter_mut().enumerate() {
                let ke = ke_values[i];
                let kc_act = config.kcb + ke;
                let etc = kc_act * input.et0;
                let p_eff = (input.precipitation + input.irrigation).max(0.0);
                let de_new = (config.state.de - p_eff + etc).clamp(0.0, config.state.tew);

                let out = DualKcOutput {
                    ke,
                    etc,
                    kr: if config.state.de <= config.state.rew {
                        1.0
                    } else {
                        ((config.state.tew - config.state.de)
                            / (config.state.tew - config.state.rew))
                            .clamp(0.0, 1.0)
                    },
                    de: de_new,
                };
                config.state.de = de_new;
                outputs.push(out);
                states.push(config.state);
            }

            Ok(BatchedDualKcResult { outputs, states })
        } else {
            Ok(self.step_cpu(input))
        }
    }

    /// Pack per-field state for one timestep into stride-9 GPU layout.
    ///
    /// `[kcb, kc_max, few, mulch_factor, de_prev, rew, tew, p_eff, et0]` per field.
    /// Ready for `BarraCuda` op=8 absorption.
    #[must_use]
    pub fn pack_gpu_timestep(&self, input: &DualKcInput) -> Vec<f64> {
        let mut data = Vec::with_capacity(self.configs.len() * 9);
        let p_eff = (input.precipitation + input.irrigation).max(0.0);
        for cfg in &self.configs {
            data.push(cfg.kcb);
            data.push(cfg.kc_max);
            data.push(cfg.few);
            data.push(cfg.mulch_factor);
            data.push(cfg.state.de);
            data.push(cfg.state.rew);
            data.push(cfg.state.tew);
            data.push(p_eff);
            data.push(input.et0);
        }
        data
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
    #![allow(clippy::float_cmp)]
    #![allow(clippy::expect_used, clippy::unwrap_used)]

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
    fn test_step_gpu_fallback_matches_cpu() {
        let configs = vec![FieldDualKcConfig {
            kcb: 1.15,
            kc_max: 1.20,
            few: 0.05,
            mulch_factor: 1.0,
            state: silt_loam_state(),
        }];

        let input = DualKcInput {
            et0: 5.0,
            precipitation: 0.0,
            irrigation: 0.0,
        };

        let mut cpu_batch = BatchedDualKc::new(configs.clone());
        let cpu_result = cpu_batch.step_cpu(&input);

        let mut gpu_batch = BatchedDualKc::new(configs);
        let gpu_result = gpu_batch.step_gpu(&input).unwrap();

        assert!(
            (gpu_result.outputs[0].etc - cpu_result.outputs[0].etc).abs() < f64::EPSILON,
            "GPU fallback {:.6} != CPU {:.6}",
            gpu_result.outputs[0].etc,
            cpu_result.outputs[0].etc,
        );
    }

    #[test]
    fn test_debug_format() {
        let batch = BatchedDualKc::new(vec![FieldDualKcConfig {
            kcb: 1.15,
            kc_max: 1.20,
            few: 0.05,
            mulch_factor: 1.0,
            state: silt_loam_state(),
        }]);
        let dbg = format!("{batch:?}");
        assert!(dbg.contains("BatchedDualKc"));
        assert!(dbg.contains("false"));
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

    #[test]
    fn test_gpu_engine_none_for_cpu_only() {
        let batch = BatchedDualKc::new(vec![FieldDualKcConfig {
            kcb: 1.15,
            kc_max: 1.20,
            few: 0.05,
            mulch_factor: 1.0,
            state: silt_loam_state(),
        }]);
        assert!(batch.gpu_engine().is_none());
    }

    #[test]
    fn test_pack_gpu_timestep_format() {
        let batch = BatchedDualKc::new(vec![
            FieldDualKcConfig {
                kcb: 1.0,
                kc_max: 1.2,
                few: 0.1,
                mulch_factor: 1.0,
                state: EvaporationLayerState {
                    de: 5.0,
                    rew: 9.0,
                    tew: 22.5,
                },
            },
            FieldDualKcConfig {
                kcb: 0.5,
                kc_max: 1.2,
                few: 0.2,
                mulch_factor: 0.4,
                state: EvaporationLayerState {
                    de: 0.0,
                    rew: 9.0,
                    tew: 22.5,
                },
            },
        ]);
        let input = DualKcInput {
            et0: 4.0,
            precipitation: 2.0,
            irrigation: 3.0,
        };
        let packed = batch.pack_gpu_timestep(&input);
        assert_eq!(packed.len(), 2 * 9);
        let p_eff = 5.0_f64;
        assert!((packed[0] - 1.0).abs() < f64::EPSILON);
        assert!((packed[3] - 1.0).abs() < f64::EPSILON);
        assert!((packed[4] - 5.0).abs() < f64::EPSILON);
        assert!((packed[7] - p_eff).abs() < f64::EPSILON);
        assert!((packed[8] - 4.0).abs() < f64::EPSILON);
        assert!((packed[9] - 0.5).abs() < f64::EPSILON);
        assert!((packed[12] - 0.4).abs() < f64::EPSILON);
    }

    #[test]
    fn test_simulate_season_empty_inputs() {
        let mut batch = BatchedDualKc::new(vec![FieldDualKcConfig {
            kcb: 1.15,
            kc_max: 1.20,
            few: 0.05,
            mulch_factor: 1.0,
            state: silt_loam_state(),
        }]);
        let summaries = batch.simulate_season(&[]);
        assert_eq!(summaries.len(), 1);
        assert_eq!(summaries[0].days, 0);
        assert_eq!(summaries[0].total_etc, 0.0);
        assert_eq!(summaries[0].total_ke, 0.0);
    }

    #[test]
    fn test_step_gpu_no_device_uses_cpu_path() {
        let configs = vec![FieldDualKcConfig {
            kcb: 1.15,
            kc_max: 1.20,
            few: 0.05,
            mulch_factor: 1.0,
            state: silt_loam_state(),
        }];
        let mut batch = BatchedDualKc::new(configs);
        let input = DualKcInput {
            et0: 5.0,
            precipitation: 0.0,
            irrigation: 0.0,
        };
        let result = batch.step_gpu(&input).expect("step_gpu should not fail");
        assert_eq!(result.outputs.len(), 1);
        assert!(result.outputs[0].etc > 0.0);
    }
}
