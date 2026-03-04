// SPDX-License-Identifier: AGPL-3.0-or-later
//! Multi-field seasonal simulation — atlas-scale GPU dispatch.
//!
//! Extends [`SeasonalPipeline`] to run M fields × N days with GPU-parallel
//! water balance. Stages 1-2 (ET₀ + Kc) are batched across all fields in
//! single GPU dispatches; Stage 3 (water balance) dispatches M fields per
//! day via [`crate::gpu::water_balance::BatchedWaterBalance`].

use crate::eco::water_balance::{self as wb, DailyInput, DailyOutput, WaterBalanceState};
use crate::eco::yield_response;

use super::{Backend, CropConfig, SeasonResult, SeasonalPipeline};

/// Multi-field season result for atlas-scale GPU dispatch.
#[derive(Debug, Clone)]
pub struct MultiFieldResult {
    /// Per-field results.
    pub fields: Vec<SeasonResult>,
    /// Number of GPU water-balance dispatches (1 per day across all fields).
    pub gpu_wb_dispatches: usize,
    /// Whether GPU WB was used (vs CPU fallback).
    pub gpu_wb_used: bool,
}

impl SeasonalPipeline {
    /// Run a multi-field seasonal simulation with GPU-parallel water balance.
    ///
    /// For M fields × N days:
    /// - Stage 1: GPU batch all M×N station-days → ET₀ (single dispatch)
    /// - Stage 2: GPU batch all M×N Kc adjustments (single dispatch)
    /// - Stage 3: For each day, GPU batch M fields' depletion update (`gpu_step`)
    /// - Stage 4: CPU yield response per field (trivial arithmetic)
    ///
    /// # Panics
    ///
    /// Panics if `weather_per_field` slices have unequal lengths.
    ///
    /// # Errors
    ///
    /// Returns an error if GPU water balance dispatch fails irrecoverably.
    pub fn run_multi_field(
        &self,
        weather_per_field: &[&[super::WeatherDay]],
        configs: &[CropConfig],
    ) -> crate::error::Result<MultiFieldResult> {
        use crate::gpu::water_balance::BatchedWaterBalance;

        let m = weather_per_field.len();
        if m == 0 || m != configs.len() {
            return Ok(MultiFieldResult {
                fields: Vec::new(),
                gpu_wb_dispatches: 0,
                gpu_wb_used: false,
            });
        }

        let n_days = weather_per_field[0].len();
        for w in weather_per_field {
            assert_eq!(w.len(), n_days, "All fields must have same number of days");
        }

        let et0_per_field: Vec<Vec<f64>> = weather_per_field
            .iter()
            .map(|w| self.compute_et0_batch(w))
            .collect();
        let kc_per_field: Vec<Vec<f64>> = weather_per_field
            .iter()
            .zip(configs)
            .map(|(w, c)| self.compute_kc_batch(w, c))
            .collect();

        let wb_gpu = if self.backend == Backend::Cpu {
            None
        } else {
            BatchedWaterBalance::gpu_only().ok()
        };
        let gpu_wb_used = wb_gpu.is_some();
        let mut gpu_wb_dispatches = 0_usize;

        let mut states: Vec<WaterBalanceState> = configs
            .iter()
            .map(|c| {
                let kc = c.crop_type.coefficients();
                let root_mm = kc.root_depth_m * 1000.0;
                WaterBalanceState::new(
                    c.field_capacity,
                    c.wilting_point,
                    root_mm,
                    c.irrigation_trigger,
                )
            })
            .collect();

        let mut wb_outputs: Vec<Vec<DailyOutput>> =
            (0..m).map(|_| Vec::with_capacity(n_days)).collect();
        let mut wb_inputs: Vec<Vec<DailyInput>> =
            (0..m).map(|_| Vec::with_capacity(n_days)).collect();
        let mut total_irr: Vec<f64> = vec![0.0; m];

        for day in 0..n_days {
            let (field_inputs, daily_inputs_per_field) = build_day_inputs(
                m,
                day,
                &states,
                configs,
                &kc_per_field,
                &et0_per_field,
                weather_per_field,
                &mut total_irr,
            );

            if let Some(ref wb_engine) = wb_gpu {
                let dr_new = wb_engine.gpu_step(&field_inputs)?;
                gpu_wb_dispatches += 1;
                apply_gpu_wb_outputs(
                    day,
                    &dr_new,
                    &mut states,
                    &mut wb_outputs,
                    &kc_per_field,
                    &et0_per_field,
                );
            } else {
                for (f, state) in states.iter_mut().enumerate() {
                    wb_outputs[f].push(state.step(&daily_inputs_per_field[f]));
                }
            }

            for (f, di) in daily_inputs_per_field.into_iter().enumerate() {
                wb_inputs[f].push(di);
            }
        }

        let fields = collect_field_results(
            m,
            n_days,
            &et0_per_field,
            &wb_outputs,
            &wb_inputs,
            &total_irr,
            &states,
            configs,
            weather_per_field,
        );
        Ok(MultiFieldResult {
            fields,
            gpu_wb_dispatches,
            gpu_wb_used,
        })
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "daily input construction requires weather, soil, crop, and state parameters"
)]
fn build_day_inputs(
    m: usize,
    day: usize,
    states: &[WaterBalanceState],
    configs: &[CropConfig],
    kc_per_field: &[Vec<f64>],
    et0_per_field: &[Vec<f64>],
    weather_per_field: &[&[super::WeatherDay]],
    total_irr: &mut [f64],
) -> (
    Vec<crate::gpu::water_balance::FieldDayInput>,
    Vec<DailyInput>,
) {
    use crate::gpu::water_balance::FieldDayInput;
    let mut field_inputs = Vec::with_capacity(m);
    let mut daily_inputs = Vec::with_capacity(m);

    for (f, state) in states.iter().enumerate() {
        let irr = if state.depletion > state.raw {
            configs[f].irrigation_depth_mm
        } else {
            0.0
        };
        total_irr[f] += irr;
        let etc = kc_per_field[f][day] * et0_per_field[f][day];

        field_inputs.push(FieldDayInput {
            dr_prev: state.depletion,
            precipitation: weather_per_field[f][day].precipitation,
            irrigation: irr,
            etc,
            taw: state.taw,
            raw: state.raw,
            p: configs[f].irrigation_trigger,
        });
        daily_inputs.push(DailyInput {
            precipitation: weather_per_field[f][day].precipitation,
            irrigation: irr,
            et0: et0_per_field[f][day],
            kc: kc_per_field[f][day],
        });
    }
    (field_inputs, daily_inputs)
}

fn apply_gpu_wb_outputs(
    day: usize,
    dr_new: &[f64],
    states: &mut [WaterBalanceState],
    wb_outputs: &mut [Vec<DailyOutput>],
    kc_per_field: &[Vec<f64>],
    et0_per_field: &[Vec<f64>],
) {
    for (f, &dr) in dr_new.iter().enumerate() {
        let ks = if states[f].depletion > states[f].raw {
            (states[f].taw - states[f].depletion) / (states[f].taw - states[f].raw)
        } else {
            1.0
        };
        let etc = kc_per_field[f][day] * et0_per_field[f][day];
        wb_outputs[f].push(DailyOutput {
            actual_et: ks * etc,
            etc,
            ks,
            depletion: dr,
            runoff: 0.0,
            deep_percolation: 0.0,
            needs_irrigation: dr > states[f].raw,
        });
        states[f].depletion = dr;
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "result collection aggregates across multiple field dimensions"
)]
fn collect_field_results(
    m: usize,
    n_days: usize,
    et0_per_field: &[Vec<f64>],
    wb_outputs: &[Vec<DailyOutput>],
    wb_inputs: &[Vec<DailyInput>],
    total_irr: &[f64],
    states: &[WaterBalanceState],
    configs: &[CropConfig],
    weather_per_field: &[&[super::WeatherDay]],
) -> Vec<SeasonResult> {
    let mut fields = Vec::with_capacity(m);
    for f in 0..m {
        let total_actual_et: f64 = wb_outputs[f].iter().map(|o| o.actual_et).sum();
        let total_etc: f64 = wb_outputs[f].iter().map(|o| o.etc).sum();
        let eta_etc = if total_etc > 0.0 {
            total_actual_et / total_etc
        } else {
            1.0
        };
        let yield_ratio = yield_response::clamp_yield_ratio(yield_response::yield_ratio_single(
            configs[f].ky,
            eta_etc,
        ));
        let mass_balance_error =
            wb::mass_balance_check(&wb_inputs[f], &wb_outputs[f], 0.0, states[f].depletion);
        fields.push(SeasonResult {
            n_days,
            total_et0: et0_per_field[f].iter().sum(),
            total_actual_et,
            total_precipitation: weather_per_field[f].iter().map(|w| w.precipitation).sum(),
            total_irrigation: total_irr[f],
            stress_days: wb_outputs[f].iter().filter(|o| o.ks < 1.0).count(),
            mass_balance_error,
            yield_ratio,
            et0_daily: et0_per_field[f].clone(),
            actual_et_daily: wb_outputs[f].iter().map(|o| o.actual_et).collect(),
        });
    }
    fields
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::eco::crop::CropType;

    fn make_weather_day(precip: f64) -> super::super::WeatherDay {
        super::super::WeatherDay {
            tmax: 30.0,
            tmin: 15.0,
            rh_max: 80.0,
            rh_min: 40.0,
            wind_2m: 2.0,
            solar_rad: 22.0,
            precipitation: precip,
            elevation: 200.0,
            latitude_deg: 42.5,
            day_of_year: 180,
        }
    }

    #[test]
    fn build_day_inputs_constructs_matching_pairs() {
        let configs = [CropConfig::standard(CropType::Corn)];
        let states = [WaterBalanceState::new(0.30, 0.12, 900.0, 0.55)];
        let kc = vec![vec![1.15]];
        let et0 = vec![vec![5.0]];
        let w = [make_weather_day(2.0)];
        let weather: &[&[super::super::WeatherDay]] = &[&w];
        let mut total_irr = vec![0.0];

        let (field_inputs, daily_inputs) =
            build_day_inputs(1, 0, &states, &configs, &kc, &et0, weather, &mut total_irr);

        assert_eq!(field_inputs.len(), 1);
        assert_eq!(daily_inputs.len(), 1);
        assert!((daily_inputs[0].et0 - 5.0).abs() < 1e-10);
        assert!((daily_inputs[0].kc - 1.15).abs() < 1e-10);
        assert!((daily_inputs[0].precipitation - 2.0).abs() < 1e-10);
    }

    #[test]
    fn apply_gpu_wb_outputs_updates_states() {
        let mut states = vec![WaterBalanceState::new(0.30, 0.12, 900.0, 0.55)];
        let mut wb_outputs: Vec<Vec<DailyOutput>> = vec![Vec::new()];
        let kc = vec![vec![1.0]];
        let et0 = vec![vec![5.0]];
        let dr_new = [20.0];

        apply_gpu_wb_outputs(0, &dr_new, &mut states, &mut wb_outputs, &kc, &et0);

        assert_eq!(wb_outputs[0].len(), 1);
        assert!((states[0].depletion - 20.0).abs() < 1e-10);
        assert!((wb_outputs[0][0].etc - 5.0).abs() < 1e-10);
    }

    #[test]
    fn apply_gpu_wb_outputs_stress_reduces_actual_et() {
        let mut states = vec![WaterBalanceState::new(0.30, 0.12, 900.0, 0.55)];
        states[0].depletion = states[0].raw + 10.0;
        let mut wb_outputs: Vec<Vec<DailyOutput>> = vec![Vec::new()];
        let kc = vec![vec![1.0]];
        let et0 = vec![vec![5.0]];

        apply_gpu_wb_outputs(
            0,
            &[states[0].depletion],
            &mut states,
            &mut wb_outputs,
            &kc,
            &et0,
        );

        assert!(wb_outputs[0][0].ks < 1.0, "ks should be < 1 under stress");
        assert!(wb_outputs[0][0].actual_et < wb_outputs[0][0].etc);
    }

    #[test]
    fn multi_field_result_empty_fields() {
        let pipeline = SeasonalPipeline::cpu();
        let result = pipeline.run_multi_field(&[], &[]).unwrap();
        assert!(result.fields.is_empty());
        assert!(!result.gpu_wb_used);
    }

    #[test]
    fn multi_field_mass_balance_conservation() {
        let pipeline = SeasonalPipeline::cpu();
        let weather: Vec<super::super::WeatherDay> = (150..=180)
            .map(|doy| super::super::WeatherDay {
                tmax: 28.0 + f64::from(doy % 5),
                tmin: 14.0,
                rh_max: 80.0,
                rh_min: 40.0,
                wind_2m: 2.0,
                solar_rad: 20.0,
                precipitation: if doy % 7 == 0 { 8.0 } else { 0.0 },
                elevation: 200.0,
                latitude_deg: 42.5,
                day_of_year: doy,
            })
            .collect();
        let configs = [CropConfig::standard(CropType::Corn)];
        let result = pipeline.run_multi_field(&[&weather], &configs).unwrap();

        assert_eq!(result.fields.len(), 1);
        assert!(
            result.fields[0].mass_balance_error.abs() < 1.0,
            "mass balance error {} mm exceeds 1 mm",
            result.fields[0].mass_balance_error
        );
    }
}
