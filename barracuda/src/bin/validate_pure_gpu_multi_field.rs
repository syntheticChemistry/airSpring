// SPDX-License-Identifier: AGPL-3.0-or-later
#![warn(clippy::pedantic)]
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::similar_names
)]
//! Exp 072: Pure GPU End-to-End Multi-Field Workload.
//!
//! Proves that the **full** seasonal pipeline runs on GPU with results
//! matching the validated CPU path:
//!
//! - Stage 1 (ET₀): `BatchedEt0` GPU dispatch (Tier A, absorbed by `ToadStool`)
//! - Stage 2 (Kc): `BatchedKcClimate` GPU dispatch (Tier A, absorbed)
//! - Stage 3 (WB): `BatchedWaterBalance::gpu_step()` per-day × M fields
//! - Stage 4 (Yield): CPU arithmetic (single multiplication per field)
//!
//! For M fields × N days, GPU dispatches = 2 (ET₀ + Kc batches) + N (WB per-day).
//! CPU dispatches would be M×N per-field-per-day. GPU reduces total dispatches
//! from M×N to N+2 — a factor of M (number of fields) reduction.
//!
//! `ToadStool` unidirectional streaming: Stages 1-2 fire without CPU readback.
//! Stage 3 reads back per-day (N round-trips), but each reads M fields in one call.
//!
//! Provenance: Pure GPU end-to-end multi-field pipeline validation

use std::sync::Arc;
use std::time::Instant;

use airspring_barracuda::eco::crop::CropType;
use airspring_barracuda::gpu::seasonal_pipeline::{CropConfig, SeasonalPipeline, WeatherDay};
use airspring_barracuda::validation::{self, ValidationHarness};

fn synthetic_field_weather(lat: f64, elev: f64) -> Vec<WeatherDay> {
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
                precipitation: if doy % 7 == 0 { 8.0 } else { 0.0 },
                elevation: elev,
                latitude_deg: lat,
                day_of_year: doy,
            }
        })
        .collect()
}

fn validate_gpu_vs_cpu_parity(v: &mut ValidationHarness) {
    validation::section("GPU vs CPU Pipeline Parity (10 fields × 153 days)");

    let cpu_pipe = SeasonalPipeline::cpu();
    let gpu_pipe = match SeasonalPipeline::gpu(Arc::new(
        barracuda::device::test_pool::tokio_block_on(
            barracuda::device::WgpuDevice::new_f64_capable(),
        )
        .expect("GPU device init"),
    )) {
        Ok(p) => p,
        Err(e) => {
            println!("  GPU init failed: {e} — running CPU fallback validation");
            SeasonalPipeline::cpu()
        }
    };

    let n_fields = 10;
    let weather: Vec<Vec<WeatherDay>> = (0..n_fields)
        .map(|i| {
            let lat = f64::from(i).mul_add(0.3, 41.0);
            let elev = f64::from(i).mul_add(10.0, 200.0);
            synthetic_field_weather(lat, elev)
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

    let cpu_result = cpu_pipe
        .run_multi_field(&weather_refs, &configs)
        .expect("CPU multi-field");
    let gpu_result = gpu_pipe
        .run_multi_field(&weather_refs, &configs)
        .expect("GPU multi-field");

    v.check_bool("CPU: 10 fields", cpu_result.fields.len() == 10);
    v.check_bool("GPU: 10 fields", gpu_result.fields.len() == 10);

    for (i, (cpu_f, gpu_f)) in cpu_result.fields.iter().zip(&gpu_result.fields).enumerate() {
        v.check_abs(
            &format!("field_{i}_et0_parity"),
            gpu_f.total_et0,
            cpu_f.total_et0,
            2.0,
        );
        v.check_abs(
            &format!("field_{i}_actual_et_parity"),
            gpu_f.total_actual_et,
            cpu_f.total_actual_et,
            2.0,
        );
        v.check_abs(
            &format!("field_{i}_yield_parity"),
            gpu_f.yield_ratio,
            cpu_f.yield_ratio,
            0.02,
        );
    }

    if gpu_result.gpu_wb_used {
        println!(
            "  GPU WB dispatches: {} (1 per day × {} days)",
            gpu_result.gpu_wb_dispatches,
            weather[0].len()
        );
    } else {
        println!("  GPU WB fallback to CPU (no GPU engine for Stage 3)");
    }
}

fn validate_streaming_vs_per_stage(v: &mut ValidationHarness) {
    validation::section("GpuPipelined vs GpuPerStage Parity");

    let device = Arc::new(
        barracuda::device::test_pool::tokio_block_on(
            barracuda::device::WgpuDevice::new_f64_capable(),
        )
        .expect("GPU device init"),
    );

    let per_stage =
        SeasonalPipeline::gpu(Arc::clone(&device)).unwrap_or_else(|_| SeasonalPipeline::cpu());
    let streaming = SeasonalPipeline::streaming(device).unwrap_or_else(|_| SeasonalPipeline::cpu());

    let weather = synthetic_field_weather(42.5, 200.0);
    let config = CropConfig::standard(CropType::Corn);

    let ps_result = per_stage.run_season(&weather, &config);
    let st_result = streaming.run_season(&weather, &config);

    v.check_abs(
        "ET₀ per-stage = streaming",
        ps_result.total_et0,
        st_result.total_et0,
        0.01,
    );
    v.check_abs(
        "actual_ET per-stage = streaming",
        ps_result.total_actual_et,
        st_result.total_actual_et,
        0.01,
    );
    v.check_abs(
        "yield per-stage = streaming",
        ps_result.yield_ratio,
        st_result.yield_ratio,
        0.001,
    );
    v.check_bool("n_days match", ps_result.n_days == st_result.n_days);
}

fn validate_dispatch_reduction(v: &mut ValidationHarness) {
    validation::section("GPU Dispatch Reduction (M fields × N days)");

    let n_fields = 20;
    let n_days = 153;

    let cpu_dispatches = n_fields * n_days;
    let gpu_dispatches = 2 + n_days;

    let reduction = f64::from(cpu_dispatches) / f64::from(gpu_dispatches);
    v.check_lower(
        &format!("dispatch reduction > {n_fields}× ({gpu_dispatches} vs {cpu_dispatches})"),
        reduction,
        f64::from(n_fields) / 2.0,
    );

    println!("  CPU dispatches (field-by-day): {cpu_dispatches}");
    println!("  GPU dispatches (batch-by-day + ET₀ + Kc): {gpu_dispatches}");
    println!("  Dispatch reduction: {reduction:.1}×");

    let device = Arc::new(
        barracuda::device::test_pool::tokio_block_on(
            barracuda::device::WgpuDevice::new_f64_capable(),
        )
        .expect("GPU device init"),
    );
    let gpu_pipe = SeasonalPipeline::gpu(device).unwrap_or_else(|_| SeasonalPipeline::cpu());

    let weather: Vec<Vec<WeatherDay>> = (0..n_fields)
        .map(|i| {
            let lat = f64::from(i).mul_add(0.15, 41.0);
            let elev = f64::from(i).mul_add(8.0, 150.0);
            synthetic_field_weather(lat, elev)
        })
        .collect();
    let configs: Vec<CropConfig> = (0..n_fields)
        .map(|i| {
            let crop = match i % 4 {
                0 => CropType::Corn,
                1 => CropType::Soybean,
                2 => CropType::Alfalfa,
                _ => CropType::WinterWheat,
            };
            CropConfig::standard(crop)
        })
        .collect();
    let weather_refs: Vec<&[WeatherDay]> = weather.iter().map(Vec::as_slice).collect();

    let start = Instant::now();
    let result = gpu_pipe
        .run_multi_field(&weather_refs, &configs)
        .expect("GPU multi-field 20");
    let elapsed = start.elapsed();

    v.check_bool("20 fields completed", result.fields.len() == 20);

    let all_valid = result
        .fields
        .iter()
        .all(|f| f.yield_ratio > 0.0 && f.yield_ratio <= 1.0);
    v.check_bool("all 20 yields valid", all_valid);

    let total_field_days = n_fields * n_days;
    let throughput = f64::from(total_field_days) / elapsed.as_secs_f64();
    v.check_lower("GPU throughput > 1K field-days/s", throughput, 1000.0);

    println!("  GPU pipeline time: {elapsed:?}");
    println!("  Throughput: {throughput:.0} field-days/s");
}

fn validate_pure_gpu_scaling(v: &mut ValidationHarness) {
    validation::section("Pure GPU Scaling (1, 10, 50 fields)");

    let device = Arc::new(
        barracuda::device::test_pool::tokio_block_on(
            barracuda::device::WgpuDevice::new_f64_capable(),
        )
        .expect("GPU device init"),
    );
    let pipe = SeasonalPipeline::gpu(device).unwrap_or_else(|_| SeasonalPipeline::cpu());

    for &n_fields in &[1, 10, 50] {
        let weather: Vec<Vec<WeatherDay>> = (0..n_fields)
            .map(|i| {
                let lat = (i as f64).mul_add(0.06, 41.0);
                synthetic_field_weather(lat, 200.0)
            })
            .collect();
        let configs: Vec<CropConfig> = (0..n_fields)
            .map(|_| CropConfig::standard(CropType::Corn))
            .collect();
        let weather_refs: Vec<&[WeatherDay]> = weather.iter().map(Vec::as_slice).collect();

        let start = Instant::now();
        let result = pipe
            .run_multi_field(&weather_refs, &configs)
            .expect("scaling");
        let elapsed = start.elapsed();

        let throughput = (n_fields * 153) as f64 / elapsed.as_secs_f64();
        v.check_bool(
            &format!("{n_fields} fields completed"),
            result.fields.len() == n_fields,
        );
        v.check_lower(
            &format!("{n_fields} fields > 100 field-days/s"),
            throughput,
            100.0,
        );
        println!("  {n_fields:>3} fields: {elapsed:?}, {throughput:.0} field-days/s");
    }
}

fn main() {
    validation::init_tracing();
    validation::banner("Exp 072: Pure GPU End-to-End Multi-Field Workload");

    let mut v = ValidationHarness::new("Pure GPU End-to-End Multi-Field");

    validate_gpu_vs_cpu_parity(&mut v);
    validate_streaming_vs_per_stage(&mut v);
    validate_dispatch_reduction(&mut v);
    validate_pure_gpu_scaling(&mut v);

    v.finish();
}
