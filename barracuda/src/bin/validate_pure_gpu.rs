// SPDX-License-Identifier: AGPL-3.0-or-later
//! Exp 042: Pure GPU Workload Validation.
//!
//! Validates the complete agricultural pipeline through the GPU orchestration
//! layer: ET₀ → Kc Adjust → Water Balance → Yield Response.
//!
//! Currently dispatches via CPU fallback (Tier B). When `BarraCuda` absorbs
//! all ops (0, 1, 5-8), this binary demonstrates end-to-end GPU execution
//! without CPU round-trips.
//!
//! # Provenance
//!
//! Expected values are derived from the deterministic synthetic weather
//! generator `growing_season_weather()` (DOY 121–273, sinusoidal patterns)
//! processed through the CPU-validated eco modules. Thresholds are physical
//! plausibility bounds, not Python baseline matches:
//!
//! - `VWC(10000) ≈ 0.1323` — Topp equation at `raw_count=10000` (Exp 005)
//! - `KcClimate ≈ 1.20` — FAO-56 Table 12 standard conditions
//! - `total_et0 > 400 mm` — 153-day Michigan growing season physical minimum
//! - `yield_ratio ∈ [0.5, 1.0]` — Stewart equation physical range
//!
//! # Validated Stages
//!
//! | Stage | Op | Module | Status |
//! |-------|----|--------|--------|
//! | ET₀ (Penman-Monteith) | 0 | `gpu::et0` | GPU-first |
//! | Water balance | 1 | `gpu::water_balance` | GPU-step |
//! | Sensor calibration | 5 | `gpu::sensor_calibration` | Tier B |
//! | Hargreaves ET₀ | 6 | `gpu::hargreaves` | Tier B |
//! | Kc climate adjustment | 7 | `gpu::kc_climate` | Tier B |
//! | Dual Kc (Ke batch) | 8 | `gpu::dual_kc` | Tier B |
//!
//! Provenance: `BarraCuda` GPU pipeline validation

use airspring_barracuda::eco::crop::CropType;
use airspring_barracuda::gpu::et0::{BatchedEt0, StationDay};
use airspring_barracuda::gpu::hargreaves::{BatchedHargreaves, HargreavesDay};
use airspring_barracuda::gpu::kc_climate::{BatchedKcClimate, KcClimateDay};
use airspring_barracuda::gpu::seasonal_pipeline::{
    CropConfig, SeasonResult, SeasonalPipeline, WeatherDay,
};
use airspring_barracuda::gpu::sensor_calibration::{BatchedSensorCal, SensorReading};
use airspring_barracuda::validation::{self, ValidationHarness};

fn growing_season_weather() -> Vec<WeatherDay> {
    let phase = 2.0 * std::f64::consts::PI / 365.0;
    (121..=273)
        .map(|doy| {
            let d = f64::from(doy);
            let sin_phase = (phase.mul_add(d - 196.0, 0.4)).sin();
            let tmax = 2.5_f64.mul_add(sin_phase, 27.5);
            let tmin = 2.0_f64.mul_add(sin_phase, 16.0);
            let rh_max = 7.5_f64.mul_add(sin_phase, 77.5);
            let rh_min = 7.5_f64.mul_add(sin_phase, 52.5);
            let solar_rad = 3.0_f64.mul_add(sin_phase, 21.0);
            let precipitation = if doy % 7 == 0 { 8.0 } else { 0.0 };
            WeatherDay {
                tmax,
                tmin,
                rh_max,
                rh_min,
                wind_2m: 2.0,
                solar_rad,
                precipitation,
                elevation: 250.0,
                latitude_deg: 42.5,
                day_of_year: doy,
            }
        })
        .collect()
}

fn validate_batched_et0(v: &mut ValidationHarness) {
    validation::section("BatchedEt0");
    let weather = growing_season_weather();
    let station_days: Vec<StationDay> = weather
        .iter()
        .map(|w| StationDay {
            tmax: w.tmax,
            tmin: w.tmin,
            rh_max: w.rh_max,
            rh_min: w.rh_min,
            wind_2m: w.wind_2m,
            rs: w.solar_rad,
            elevation: w.elevation,
            latitude: w.latitude_deg,
            doy: w.day_of_year,
        })
        .collect();
    let engine = BatchedEt0::cpu();
    let result = engine.compute_gpu(&station_days).expect("compute_gpu");
    v.check_bool(
        "BatchedEt0 batch length",
        result.et0_values.len() == station_days.len(),
    );
    let all_positive = result.et0_values.iter().all(|&e| e > 0.0);
    v.check_bool("all ET₀ > 0", all_positive);
    let summer_et0: f64 = result.et0_values[31..92].iter().sum();
    let early_et0: f64 = result.et0_values[0..31].iter().sum();
    v.check_lower("summer ET₀ > early ET₀", summer_et0, early_et0);
}

fn validate_hargreaves(v: &mut ValidationHarness) {
    validation::section("BatchedHargreaves");
    let weather = growing_season_weather();
    let days: Vec<HargreavesDay> = weather
        .iter()
        .map(|w| HargreavesDay {
            tmax: w.tmax,
            tmin: w.tmin,
            latitude_deg: w.latitude_deg,
            day_of_year: w.day_of_year,
        })
        .collect();
    let engine = BatchedHargreaves::cpu();
    let result = engine.compute(&days);
    v.check_bool(
        "BatchedHargreaves batch length",
        result.et0_values.len() == days.len(),
    );
    let all_positive = result.et0_values.iter().all(|&e| e > 0.0);
    v.check_bool("all HG ET₀ > 0", all_positive);
    let summer: f64 = result.et0_values[31..92].iter().sum();
    let early: f64 = result.et0_values[0..31].iter().sum();
    v.check_lower("HG summer > early", summer, early);
}

fn validate_kc_climate(v: &mut ValidationHarness) {
    validation::section("BatchedKcClimate");
    let inputs = vec![
        KcClimateDay {
            kc_table: 1.20,
            u2: 2.0,
            rh_min: 45.0,
            crop_height_m: 2.0,
        },
        KcClimateDay {
            kc_table: 1.15,
            u2: 4.0,
            rh_min: 30.0,
            crop_height_m: 2.0,
        },
    ];
    let engine = BatchedKcClimate::cpu();
    let result = engine.compute(&inputs);
    v.check_bool("KcClimate batch length", result.kc_values.len() == 2);
    v.check_abs("KcClimate standard", result.kc_values[0], 1.20, 0.01);
    v.check_lower(
        "KcClimate windy > standard",
        result.kc_values[1],
        result.kc_values[0],
    );
}

fn validate_sensor_cal(v: &mut ValidationHarness) {
    validation::section("BatchedSensorCal");
    let engine = BatchedSensorCal::cpu();
    let result = engine.compute(&[SensorReading {
        raw_count: 10_000.0,
    }]);
    v.check_bool("SensorCal single", result.vwc_values.len() == 1);
    v.check_abs("VWC(10000) ≈ 0.1323", result.vwc_values[0], 0.1323, 1e-3);
}

fn validate_seasonal_pipeline(v: &mut ValidationHarness) {
    validation::section("SeasonalPipeline");
    let weather = growing_season_weather();
    let pipeline = SeasonalPipeline::cpu();
    let config = CropConfig::standard(CropType::Corn);
    let result: SeasonResult = pipeline.run_season(&weather, &config);
    v.check_abs("n_days", result.n_days as f64, 153.0, f64::EPSILON);
    v.check_lower("total_et0 > 400 mm", result.total_et0, 400.0);
    v.check_bool(
        "yield_ratio in [0.5, 1.0]",
        result.yield_ratio >= 0.5 && result.yield_ratio <= 1.0,
    );
    v.check_abs("mass_balance < 0.1 mm", result.mass_balance_error, 0.0, 0.1);
    v.check_bool("stress_days > 0", result.stress_days > 0);
}

fn main() {
    validation::init_tracing();
    validation::banner("Exp 042: Pure GPU Workload Validation");

    let mut v = ValidationHarness::new("Pure GPU Pipeline");
    validate_batched_et0(&mut v);
    validate_hargreaves(&mut v);
    validate_kc_climate(&mut v);
    validate_sensor_cal(&mut v);
    validate_seasonal_pipeline(&mut v);

    v.finish();
}
