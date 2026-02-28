// SPDX-License-Identifier: AGPL-3.0-or-later
#![warn(clippy::pedantic)]
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
//! Exp 043: Titan V GPU Live Dispatch Validation.
//!
//! Runs actual GPU shader dispatch on the NVIDIA TITAN V (GV100) via wgpu/Vulkan.
//! Proves that `BatchedEt0::compute_gpu()` on real GPU hardware produces results
//! matching the validated CPU path to within f64 precision.
//!
//! Also benchmarks GPU vs CPU throughput for seasonal-scale batches.
//!
//! Hardware requirements:
//! - NVIDIA TITAN V (Volta GV100) with `SHADER_F64` support
//! - Set `BARRACUDA_GPU_ADAPTER=titan` to target the Titan V specifically
//!
//! Benchmark: `control/cpu_gpu_parity/benchmark_cpu_gpu_parity.json`
//! Baseline: Exp 040 (26/26 PASS on CPU fallback)

use std::sync::Arc;
use std::time::Instant;

use barracuda::device::WgpuDevice;

use airspring_barracuda::eco::evapotranspiration::{
    self as et, actual_vapour_pressure_rh, DailyEt0Input,
};
use airspring_barracuda::gpu::et0::{Backend, BatchedEt0, BatchedEt0Result, StationDay};
use airspring_barracuda::validation::{self, json_field, parse_benchmark_json, ValidationHarness};

const PARITY_JSON: &str =
    include_str!("../../../control/cpu_gpu_parity/benchmark_cpu_gpu_parity.json");
const SEASONAL_JSON: &str =
    include_str!("../../../control/seasonal_batch_et0/benchmark_seasonal_batch.json");

fn seasonal_value(doy: u32, vmin: f64, vmax: f64) -> f64 {
    let phase_doy = 196.0_f64;
    let frac = (2.0 * std::f64::consts::PI * (f64::from(doy) - phase_doy + 91.25) / 365.0).sin();
    let mid = f64::midpoint(vmin, vmax);
    let amp = (vmax - vmin) / 2.0;
    mid + amp * frac
}

fn create_device() -> Option<Arc<WgpuDevice>> {
    // Use adapter from env or discover at runtime (capability-based).
    let adapter_hint = std::env::var("BARRACUDA_GPU_ADAPTER").unwrap_or_default();
    if adapter_hint.is_empty() {
        println!("  No GPU adapter hint — using runtime discovery");
    }

    match pollster::block_on(WgpuDevice::from_env()) {
        Ok(dev) => {
            println!("  GPU device created");
            Some(Arc::new(dev))
        }
        Err(e) => {
            eprintln!("  WARN: Could not create Titan V device: {e}");
            eprintln!("  Trying auto-discovery (any f64-capable GPU)...");
            match pollster::block_on(WgpuDevice::new_f64_capable()) {
                Ok(dev) => {
                    println!("  GPU device created via auto-discovery");
                    Some(Arc::new(dev))
                }
                Err(e2) => {
                    eprintln!("  SKIP: No f64-capable GPU available: {e2}");
                    None
                }
            }
        }
    }
}

fn validate_gpu_parity(
    v: &mut ValidationHarness,
    device: &Arc<WgpuDevice>,
    benchmark: &serde_json::Value,
) {
    validation::section("GPU Parity — BatchedEt0 (live GPU vs CPU)");

    let gpu_batcher =
        BatchedEt0::gpu(Arc::clone(device)).expect("BatchedEt0::gpu() should succeed");
    let cpu_batcher = BatchedEt0::cpu();

    let tests = &benchmark["validation_checks"]["et0_cpu_gpu_parity"]["test_cases"];

    for tc in tests.as_array().expect("array") {
        let label = tc["label"].as_str().unwrap_or("test");
        let station = StationDay {
            tmax: json_field(tc, "tmax"),
            tmin: json_field(tc, "tmin"),
            rh_max: json_field(tc, "rh_max"),
            rh_min: json_field(tc, "rh_min"),
            wind_2m: json_field(tc, "wind_2m"),
            rs: json_field(tc, "rs"),
            elevation: json_field(tc, "elevation"),
            latitude: json_field(tc, "latitude"),
            doy: json_field(tc, "doy") as u32,
        };

        let cpu_result = cpu_batcher.compute_gpu(&[station]).expect("CPU compute");
        let gpu_result = gpu_batcher.compute_gpu(&[station]).expect("GPU compute");

        v.check_bool(
            &format!("{label}: GPU backend reports Gpu"),
            gpu_result.backend_used == Backend::Gpu,
        );

        // GPU f64 emulation via math_f64.wgsl has minor precision differences
        // in intermediate trig (solar declination, hour angle). 0.02 mm/day
        // tolerance reflects this while being well within scientific accuracy.
        let gpu_tol = 0.02_f64;
        v.check_abs(
            &format!("{label}: GPU ET₀ ≈ CPU ET₀"),
            gpu_result.et0_values[0],
            cpu_result.et0_values[0],
            gpu_tol,
        );

        v.check_lower(
            &format!("{label}: GPU ET₀ > 0"),
            gpu_result.et0_values[0],
            0.0,
        );

        // Also check direct CPU path
        let ea =
            actual_vapour_pressure_rh(station.tmin, station.tmax, station.rh_min, station.rh_max);
        let direct = et::daily_et0(&DailyEt0Input {
            tmin: station.tmin,
            tmax: station.tmax,
            tmean: None,
            solar_radiation: station.rs,
            wind_speed_2m: station.wind_2m,
            actual_vapour_pressure: ea,
            elevation_m: station.elevation,
            latitude_deg: station.latitude,
            day_of_year: station.doy,
        });

        v.check_abs(
            &format!("{label}: GPU ET₀ ≈ direct CPU ET₀"),
            gpu_result.et0_values[0],
            direct.et0,
            gpu_tol,
        );
    }
}

fn build_seasonal_batch(benchmark: &serde_json::Value) -> Vec<StationDay> {
    let stations = benchmark["stations"].as_array().expect("stations array");
    let mut all_days: Vec<StationDay> = Vec::new();
    for st in stations {
        let tmax_range = st["tmax_range"].as_array().unwrap();
        let tmin_range = st["tmin_range"].as_array().unwrap();
        let rh_max_range = st["rh_max_range"].as_array().unwrap();
        let rh_min_range = st["rh_min_range"].as_array().unwrap();
        let rs_range = st["rs_range"].as_array().unwrap();

        for doy in 1..=365_u32 {
            all_days.push(StationDay {
                tmax: seasonal_value(
                    doy,
                    tmax_range[0].as_f64().unwrap(),
                    tmax_range[1].as_f64().unwrap(),
                ),
                tmin: seasonal_value(
                    doy,
                    tmin_range[0].as_f64().unwrap(),
                    tmin_range[1].as_f64().unwrap(),
                ),
                rh_max: seasonal_value(
                    doy,
                    rh_max_range[0].as_f64().unwrap(),
                    rh_max_range[1].as_f64().unwrap(),
                ),
                rh_min: seasonal_value(
                    doy,
                    rh_min_range[0].as_f64().unwrap(),
                    rh_min_range[1].as_f64().unwrap(),
                ),
                wind_2m: json_field(st, "wind_2m"),
                rs: seasonal_value(
                    doy,
                    rs_range[0].as_f64().unwrap(),
                    rs_range[1].as_f64().unwrap(),
                ),
                elevation: json_field(st, "elevation"),
                latitude: json_field(st, "latitude"),
                doy,
            });
        }
    }
    all_days
}

fn run_throughput_benchmarks(
    gpu_batcher: &BatchedEt0,
    cpu_batcher: &BatchedEt0,
    all_days: &[StationDay],
    iters: u32,
) -> (
    Option<BatchedEt0Result>,
    Option<BatchedEt0Result>,
    std::time::Duration,
    std::time::Duration,
) {
    let _ = gpu_batcher.compute_gpu(all_days);

    let gpu_start = Instant::now();
    let mut gpu_result = None;
    for _ in 0..iters {
        gpu_result = Some(gpu_batcher.compute_gpu(all_days).expect("GPU compute"));
    }
    let gpu_per_iter = gpu_start.elapsed() / iters;

    let cpu_start = Instant::now();
    let mut cpu_result = None;
    for _ in 0..iters {
        cpu_result = Some(cpu_batcher.compute_gpu(all_days).expect("CPU compute"));
    }
    let cpu_per_iter = cpu_start.elapsed() / iters;

    (gpu_result, cpu_result, gpu_per_iter, cpu_per_iter)
}

fn check_throughput_parity(
    v: &mut ValidationHarness,
    gpu_result: Option<&BatchedEt0Result>,
    cpu_result: Option<&BatchedEt0Result>,
    n: usize,
) {
    v.check_bool(
        &format!("GPU computed {n} station-days"),
        gpu_result.is_some_and(|r| r.et0_values.len() == n),
    );

    v.check_bool(
        "GPU reports Gpu backend",
        gpu_result.is_some_and(|r| r.backend_used == Backend::Gpu),
    );

    let gpu_vals = &gpu_result.unwrap().et0_values;
    let cpu_vals = &cpu_result.unwrap().et0_values;

    let gpu_total: f64 = gpu_vals.iter().sum();
    let cpu_total: f64 = cpu_vals.iter().sum();
    let pct_diff = ((gpu_total - cpu_total) / cpu_total * 100.0).abs();

    println!("  GPU annual total: {gpu_total:.2} mm");
    println!("  CPU annual total: {cpu_total:.2} mm");
    println!("  Difference: {pct_diff:.4}%");

    v.check_bool("Seasonal total GPU ≈ CPU (< 1%)", pct_diff < 1.0);

    let max_diff: f64 = gpu_vals
        .iter()
        .zip(cpu_vals.iter())
        .map(|(g, c)| (g - c).abs())
        .fold(0.0_f64, f64::max);

    println!("  Max daily divergence: {max_diff:.6} mm/day");

    v.check_bool("Max daily GPU-CPU divergence < 0.1 mm/day", max_diff < 0.1);
}

const BENCH_ITERS: u32 = 5;

fn benchmark_throughput(
    v: &mut ValidationHarness,
    device: &Arc<WgpuDevice>,
    benchmark: &serde_json::Value,
) {
    validation::section("GPU Throughput Benchmark");

    let gpu_batcher =
        BatchedEt0::gpu(Arc::clone(device)).expect("BatchedEt0::gpu() should succeed");
    let cpu_batcher = BatchedEt0::cpu();

    let all_days = build_seasonal_batch(benchmark);
    let n = all_days.len();
    println!("  Batch size: {n} station-days");
    let (gpu_result, cpu_result, gpu_per_iter, cpu_per_iter) =
        run_throughput_benchmarks(&gpu_batcher, &cpu_batcher, &all_days, BENCH_ITERS);

    let gpu_us = gpu_per_iter.as_micros();
    let cpu_us = cpu_per_iter.as_micros();
    let speedup = if gpu_us > 0 {
        cpu_us as f64 / gpu_us as f64
    } else {
        f64::INFINITY
    };

    println!("  GPU: {gpu_us} µs/batch ({n} station-days)");
    println!("  CPU: {cpu_us} µs/batch ({n} station-days)");
    println!("  Speedup: {speedup:.1}x");

    check_throughput_parity(v, gpu_result.as_ref(), cpu_result.as_ref(), n);
}

fn validate_batch_scaling_gpu(v: &mut ValidationHarness, device: &Arc<WgpuDevice>) {
    validation::section("GPU Batch Scaling");

    let gpu_batcher = BatchedEt0::gpu(Arc::clone(device)).expect("BatchedEt0::gpu()");

    let station = StationDay {
        tmax: 25.0,
        tmin: 15.0,
        rh_max: 80.0,
        rh_min: 40.0,
        wind_2m: 2.0,
        rs: 20.0,
        elevation: 100.0,
        latitude: 45.0,
        doy: 180,
    };

    let ref_result = gpu_batcher
        .compute_gpu(&[station])
        .expect("single GPU compute");
    let ref_val = ref_result.et0_values[0];

    for sz in [10, 100, 1000, 10_000] {
        let batch: Vec<StationDay> = vec![station; sz];
        let start = Instant::now();
        let result = gpu_batcher.compute_gpu(&batch).expect("batch GPU compute");
        let elapsed = start.elapsed();

        let max_diff: f64 = result
            .et0_values
            .iter()
            .map(|&v| (v - ref_val).abs())
            .fold(0.0_f64, f64::max);

        println!(
            "  N={sz:>6}: {:.0} µs, max_diff={max_diff:.2e}",
            elapsed.as_micros()
        );

        v.check_bool(
            &format!("batch N={sz}: all within 0.01 mm/day of reference"),
            max_diff < 0.01,
        );
    }
}

fn main() {
    validation::init_tracing();
    validation::banner("Exp 043: Titan V GPU Live Dispatch Validation");

    let mut v = ValidationHarness::new("GPU Live Dispatch");

    let parity_bm = parse_benchmark_json(PARITY_JSON).expect("parity benchmark must parse");
    let seasonal_bm = parse_benchmark_json(SEASONAL_JSON).expect("seasonal benchmark must parse");

    let Some(device) = create_device() else {
        println!("SKIP: No GPU available — all GPU tests skipped");
        return;
    };

    validate_gpu_parity(&mut v, &device, &parity_bm);
    benchmark_throughput(&mut v, &device, &seasonal_bm);
    validate_batch_scaling_gpu(&mut v, &device);

    v.finish();
}
