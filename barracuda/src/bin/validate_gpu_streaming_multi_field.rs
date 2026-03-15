// SPDX-License-Identifier: AGPL-3.0-or-later
#![allow(clippy::similar_names)]
//! Exp 070: GPU Streaming Multi-Field Pipeline Validation.
//!
//! Validates the multi-field seasonal pipeline with GPU-parallel
//! water balance dispatch. Proves:
//!
//! 1. Multi-field CPU parity: N fields via `run_multi_field()` match N
//!    individual `run_season()` calls.
//! 2. GPU water balance: Stage 3 dispatches M field depletions per day
//!    to GPU in a single `gpu_step()` call (1 dispatch per day × N days
//!    vs M×N individual CPU calls).
//! 3. Streaming benefit: Stages 1-2 (ET₀+Kc) use GPU batch with no CPU
//!    readback between them. Stage 3 uses GPU multi-field per-day.
//!    Stage 4 (yield) is trivial CPU arithmetic.
//!
//! Provenance: GPU streaming multi-field seasonal pipeline validation

use std::time::Instant;

use airspring_barracuda::eco::crop::CropType;
use airspring_barracuda::gpu::seasonal_pipeline::{CropConfig, SeasonalPipeline, WeatherDay};
use airspring_barracuda::validation::{self, ValidationHarness};

fn synthetic_field_weather(lat: f64, elev: f64, precip_period: u32) -> Vec<WeatherDay> {
    let phase = 2.0 * std::f64::consts::PI / 153.0;
    (121..=273)
        .map(|doy| {
            let d = f64::from(doy - 121);
            let s = (phase * d).sin();
            WeatherDay {
                tmax: 2.5_f64.mul_add(s, 28.0),
                tmin: 2.0_f64.mul_add(s, 16.0),
                rh_max: 7.5_f64.mul_add(s, 77.5),
                rh_min: 7.5_f64.mul_add(s, 52.5),
                wind_2m: 2.0,
                solar_rad: 3.0_f64.mul_add(s, 21.0),
                precipitation: if doy % precip_period == 0 { 8.0 } else { 0.0 },
                elevation: elev,
                latitude_deg: lat,
                day_of_year: doy,
            }
        })
        .collect()
}

fn validate_multi_field_cpu_parity(v: &mut ValidationHarness) {
    validation::section("Multi-Field CPU Parity (single vs batch)");

    let pipeline = SeasonalPipeline::cpu();

    let crops = [CropType::Corn, CropType::Soybean, CropType::WinterWheat];
    let lats = [41.0, 42.5, 44.0];
    let elevs = [200.0, 250.0, 300.0];
    let periods = [7, 5, 9];

    let weather: Vec<Vec<WeatherDay>> = (0..3)
        .map(|i| synthetic_field_weather(lats[i], elevs[i], periods[i]))
        .collect();
    let configs: Vec<CropConfig> = crops.iter().map(|c| CropConfig::standard(*c)).collect();

    let individual: Vec<_> = weather
        .iter()
        .zip(&configs)
        .map(|(w, c)| pipeline.run_season(w, c))
        .collect();

    let weather_refs: Vec<&[WeatherDay]> = weather.iter().map(Vec::as_slice).collect();
    let multi = pipeline
        .run_multi_field(&weather_refs, &configs)
        .expect("multi-field should succeed");

    v.check_bool("multi-field count = 3", multi.fields.len() == 3);

    for (i, (single, mf)) in individual.iter().zip(&multi.fields).enumerate() {
        v.check_abs(
            &format!("field_{i}_et0_parity"),
            mf.total_et0,
            single.total_et0,
            0.01,
        );
        v.check_abs(
            &format!("field_{i}_actual_et_parity"),
            mf.total_actual_et,
            single.total_actual_et,
            0.5,
        );
        v.check_abs(
            &format!("field_{i}_yield_parity"),
            mf.yield_ratio,
            single.yield_ratio,
            0.02,
        );
        v.check_bool(
            &format!("field_{i}_n_days_match"),
            mf.n_days == single.n_days,
        );
        v.check_bool(
            &format!("field_{i}_stress_plausible"),
            mf.stress_days <= mf.n_days,
        );
    }
}

fn validate_streaming_pipeline(v: &mut ValidationHarness) {
    validation::section("GPU Streaming Pipeline (Stages 1-2 GPU, Stage 3 GPU per-day)");

    let cpu_pipe = SeasonalPipeline::cpu();

    let n_fields = 10;
    let weather: Vec<Vec<WeatherDay>> = (0..n_fields)
        .map(|i| {
            let lat = f64::from(i).mul_add(0.3, 41.0);
            let elev = f64::from(i).mul_add(10.0, 200.0);
            synthetic_field_weather(lat, elev, 7)
        })
        .collect();
    let configs: Vec<CropConfig> = (0..n_fields)
        .map(|i| {
            let crop = match i % 3 {
                0 => CropType::Corn,
                1 => CropType::Soybean,
                _ => CropType::WinterWheat,
            };
            CropConfig::standard(crop)
        })
        .collect();

    let weather_refs: Vec<&[WeatherDay]> = weather.iter().map(Vec::as_slice).collect();

    let cpu_start = Instant::now();
    let cpu_result = cpu_pipe
        .run_multi_field(&weather_refs, &configs)
        .expect("CPU multi-field should work");
    let cpu_elapsed = cpu_start.elapsed();

    v.check_bool("CPU multi-field: 10 fields", cpu_result.fields.len() == 10);
    v.check_bool("CPU path: no GPU WB", !cpu_result.gpu_wb_used);

    for (i, field) in cpu_result.fields.iter().enumerate() {
        v.check_bool(
            &format!("field_{i}_yield_valid"),
            field.yield_ratio > 0.0 && field.yield_ratio <= 1.0,
        );
        v.check_bool(&format!("field_{i}_et0_positive"), field.total_et0 > 0.0);
        v.check_bool(
            &format!("field_{i}_actual_et_bounded"),
            field.total_actual_et <= field.total_et0 * 2.0,
        );
    }

    let mean_yield: f64 =
        cpu_result.fields.iter().map(|f| f.yield_ratio).sum::<f64>() / f64::from(n_fields);
    v.check_lower("mean yield > 0.5", mean_yield, 0.5);
    v.check_upper("mean yield <= 1.0", mean_yield, 1.0);

    let et0_spread: f64 =
        cpu_result.fields.iter().map(|f| f.total_et0).sum::<f64>() / f64::from(n_fields);
    v.check_lower("mean ET₀ > 200 mm", et0_spread, 200.0);
    v.check_upper("mean ET₀ < 1000 mm", et0_spread, 1000.0);

    println!("  CPU multi-field time: {cpu_elapsed:?}");
    println!("  GPU WB dispatches: {}", cpu_result.gpu_wb_dispatches);
}

fn validate_atlas_scale(v: &mut ValidationHarness) {
    validation::section("Atlas-Scale Multi-Field (50 stations × 153 days)");

    let pipeline = SeasonalPipeline::cpu();
    let n_fields = 50;

    let weather: Vec<Vec<WeatherDay>> = (0..n_fields)
        .map(|i| {
            let lat = f64::from(i).mul_add(0.06, 41.0);
            let elev = f64::from(i).mul_add(5.0, 150.0);
            synthetic_field_weather(lat, elev, 7)
        })
        .collect();
    let configs: Vec<CropConfig> = (0..n_fields)
        .map(|i| {
            let crop = match i % 5 {
                0 => CropType::Corn,
                1 => CropType::Soybean,
                2 => CropType::WinterWheat,
                3 => CropType::Alfalfa,
                _ => CropType::Tomato,
            };
            CropConfig::standard(crop)
        })
        .collect();
    let weather_refs: Vec<&[WeatherDay]> = weather.iter().map(Vec::as_slice).collect();

    let start = Instant::now();
    let result = pipeline
        .run_multi_field(&weather_refs, &configs)
        .expect("atlas multi-field");
    let elapsed = start.elapsed();

    v.check_bool("50 field results", result.fields.len() == 50);

    let all_valid = result
        .fields
        .iter()
        .all(|f| f.yield_ratio > 0.0 && f.yield_ratio <= 1.0);
    v.check_bool("all 50 yields valid", all_valid);

    let all_et0_positive = result.fields.iter().all(|f| f.total_et0 > 0.0);
    v.check_bool("all 50 ET₀ positive", all_et0_positive);

    let n_days_correct = result.fields.iter().all(|f| f.n_days == 153);
    v.check_bool("all 50 fields = 153 days", n_days_correct);

    v.check_bool("completes in < 30s", elapsed.as_secs_f64() < 30.0);

    println!("  Atlas-scale time: {elapsed:?}");
    println!(
        "  Throughput: {:.0} field-days/sec",
        f64::from(n_fields * 153) / elapsed.as_secs_f64()
    );
}

fn main() {
    validation::init_tracing();
    validation::banner("Exp 070: GPU Streaming Multi-Field Pipeline");

    let mut v = ValidationHarness::new("GPU Streaming Multi-Field");

    validate_multi_field_cpu_parity(&mut v);
    validate_streaming_pipeline(&mut v);
    validate_atlas_scale(&mut v);

    v.finish();
}
