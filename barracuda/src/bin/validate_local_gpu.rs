// SPDX-License-Identifier: AGPL-3.0-or-later
#![warn(clippy::pedantic)]
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
//! Exp 075: Local GPU Compute Parity Validation.
//!
//! Validates that airSpring-evolved WGSL shaders produce results matching
//! the validated CPU paths for all 6 local operations:
//!
//! 1. SCS-CN runoff (USDA-SCS 1972)
//! 2. Stewart yield response (Stewart 1977, FAO-56 Ch 10)
//! 3. Makkink ET₀ (Makkink 1957, de Bruin 1987)
//! 4. Turc ET₀ (Turc 1961)
//! 5. Hamon PET (Hamon 1961, Lu et al. 2005)
//! 6. Blaney-Criddle ET₀ (Blaney & Criddle 1950, FAO-24)
//!
//! GPU path: `local_elementwise.wgsl` (f32) dispatched via `LocalElementwise`.
//! CPU path: `eco::runoff`, `eco::yield_response`, `eco::simple_et0` (f64).
//!
//! Parity tolerance: relative 0.5% + absolute floor (f32 precision).
//! `ToadStool` absorption upgrades to f64 for exact parity.

use airspring_barracuda::gpu::device_info::try_f64_device;
use airspring_barracuda::gpu::local_dispatch::{LocalElementwise, LocalOp};
use airspring_barracuda::gpu::runoff::{BatchedRunoff, GpuRunoff, RunoffInput};
use airspring_barracuda::gpu::simple_et0::{
    BatchedSimpleEt0, BlaneyCriddleInput, GpuSimpleEt0, HamonInput, MakkinkInput, TurcInput,
};
use airspring_barracuda::gpu::yield_response::{
    BatchedYieldResponse, GpuYieldResponse, YieldInput,
};
use airspring_barracuda::validation::ValidationHarness;

fn main() {
    tracing_subscriber::fmt::init();
    let mut v = ValidationHarness::new("Exp 075: Local GPU Compute Parity");

    let Some(device) = try_f64_device() else {
        println!("SKIP: No GPU device available — cannot validate local GPU parity");
        v.check_bool("gpu_device_available", false);
        v.finish();
    };

    let le = LocalElementwise::new(device.clone()).expect("shader compilation");
    let gpu_runoff = GpuRunoff::new(device.clone()).expect("GpuRunoff");
    let gpu_yield = GpuYieldResponse::new(device.clone()).expect("GpuYieldResponse");
    let gpu_et0 = GpuSimpleEt0::new(device).expect("GpuSimpleEt0");

    validate_scs_cn(&mut v, &le, &gpu_runoff);
    validate_stewart(&mut v, &le, &gpu_yield);
    validate_makkink(&mut v, &gpu_et0);
    validate_turc(&mut v, &gpu_et0);
    validate_hamon(&mut v, &gpu_et0);
    validate_blaney_criddle(&mut v, &gpu_et0);
    validate_batch_scaling(&mut v, &le);
    validate_edge_cases(&mut v, &le);

    v.finish();
}

fn parity_ok(gpu: f64, cpu: f64, rel_tol: f64, abs_tol: f64) -> bool {
    (gpu - cpu).abs() < cpu.abs().mul_add(rel_tol, abs_tol)
}

fn validate_scs_cn(v: &mut ValidationHarness, le: &LocalElementwise, gpu: &GpuRunoff) {
    println!("\n── SCS-CN Runoff ──");

    let inputs = vec![
        RunoffInput { precip_mm: 50.0, cn: 75.0, ia_ratio: 0.2 },
        RunoffInput { precip_mm: 100.0, cn: 85.0, ia_ratio: 0.2 },
        RunoffInput { precip_mm: 25.0, cn: 65.0, ia_ratio: 0.2 },
        RunoffInput { precip_mm: 200.0, cn: 90.0, ia_ratio: 0.2 },
        RunoffInput { precip_mm: 10.0, cn: 50.0, ia_ratio: 0.2 },
        RunoffInput { precip_mm: 0.0, cn: 80.0, ia_ratio: 0.2 },
        RunoffInput { precip_mm: 75.0, cn: 98.0, ia_ratio: 0.05 },
    ];
    let cpu_result = BatchedRunoff::compute(&inputs);
    let gpu_result = gpu.compute(&inputs).expect("GPU dispatch");

    for (i, (g, c)) in gpu_result.iter().zip(&cpu_result.runoff_mm).enumerate() {
        let ok = parity_ok(*g, *c, 5e-3, 0.01);
        println!("  SCS-CN[{i}]: CPU={c:.4} GPU={g:.4} |Δ|={:.6}", (g - c).abs());
        v.check_bool(&format!("SCS_CN_{i}"), ok);
    }

    let p: Vec<f64> = inputs.iter().map(|i| i.precip_mm).collect();
    let cn: Vec<f64> = inputs.iter().map(|i| i.cn).collect();
    let ia: Vec<f64> = inputs.iter().map(|i| i.ia_ratio).collect();
    let raw = le.dispatch(LocalOp::ScsCnRunoff, &p, &cn, &ia).expect("raw dispatch");
    v.check_bool(
        "SCS_CN_raw_matches_typed",
        raw.iter().zip(&gpu_result).all(|(a, b)| (a - b).abs() < 1e-6),
    );
}

fn validate_stewart(v: &mut ValidationHarness, le: &LocalElementwise, gpu: &GpuYieldResponse) {
    println!("\n── Stewart Yield Response ──");

    let inputs = vec![
        YieldInput { ky: 1.25, et_actual: 500.0, et_crop: 600.0 },
        YieldInput { ky: 0.85, et_actual: 400.0, et_crop: 600.0 },
        YieldInput { ky: 1.0, et_actual: 600.0, et_crop: 600.0 },
        YieldInput { ky: 1.50, et_actual: 300.0, et_crop: 600.0 },
        YieldInput { ky: 0.40, et_actual: 550.0, et_crop: 600.0 },
    ];
    let cpu = BatchedYieldResponse::compute(&inputs);
    let gpu_result = gpu.compute(&inputs).expect("GPU dispatch");

    for (i, (g, c)) in gpu_result.iter().zip(&cpu).enumerate() {
        let ok = parity_ok(*g, *c, 1e-3, 1e-4);
        println!("  Stewart[{i}]: CPU={c:.6} GPU={g:.6} |Δ|={:.8}", (g - c).abs());
        v.check_bool(&format!("Stewart_{i}"), ok);
    }

    let ky: Vec<f64> = inputs.iter().map(|i| i.ky).collect();
    let ratio: Vec<f64> = inputs.iter().map(|i| i.et_actual / i.et_crop).collect();
    let zeros = vec![0.0; inputs.len()];
    let raw = le.dispatch(LocalOp::StewartYield, &ky, &ratio, &zeros).expect("raw");
    let raw_clamped: Vec<f64> = raw.iter().map(|&r| r.clamp(0.0, 1.0)).collect();
    v.check_bool(
        "Stewart_raw_matches_typed",
        raw_clamped.iter().zip(&gpu_result).all(|(a, b)| (a - b).abs() < 1e-6),
    );
}

fn validate_makkink(v: &mut ValidationHarness, gpu: &GpuSimpleEt0) {
    println!("\n── Makkink ET₀ ──");

    let inputs = vec![
        MakkinkInput { tmean_c: 20.0, rs_mj: 15.0, elevation_m: 100.0 },
        MakkinkInput { tmean_c: 30.0, rs_mj: 25.0, elevation_m: 0.0 },
        MakkinkInput { tmean_c: 10.0, rs_mj: 8.0, elevation_m: 500.0 },
        MakkinkInput { tmean_c: 25.0, rs_mj: 20.0, elevation_m: 50.0 },
    ];
    let cpu = BatchedSimpleEt0::makkink(&inputs);
    let gpu_result = gpu.makkink(&inputs).expect("GPU dispatch");

    for (i, (g, c)) in gpu_result.iter().zip(&cpu).enumerate() {
        let ok = parity_ok(*g, *c, 5e-3, 0.01);
        println!("  Makkink[{i}]: CPU={c:.4} GPU={g:.4} |Δ|={:.6}", (g - c).abs());
        v.check_bool(&format!("Makkink_{i}"), ok);
    }
}

fn validate_turc(v: &mut ValidationHarness, gpu: &GpuSimpleEt0) {
    println!("\n── Turc ET₀ ──");

    let inputs = vec![
        TurcInput { tmean_c: 20.0, rs_mj: 15.0, rh_pct: 70.0 },
        TurcInput { tmean_c: 25.0, rs_mj: 20.0, rh_pct: 40.0 },
        TurcInput { tmean_c: 30.0, rs_mj: 25.0, rh_pct: 55.0 },
        TurcInput { tmean_c: 15.0, rs_mj: 10.0, rh_pct: 80.0 },
    ];
    let cpu = BatchedSimpleEt0::turc(&inputs);
    let gpu_result = gpu.turc(&inputs).expect("GPU dispatch");

    for (i, (g, c)) in gpu_result.iter().zip(&cpu).enumerate() {
        let ok = parity_ok(*g, *c, 5e-3, 0.01);
        println!("  Turc[{i}]: CPU={c:.4} GPU={g:.4} |Δ|={:.6}", (g - c).abs());
        v.check_bool(&format!("Turc_{i}"), ok);
    }
}

fn validate_hamon(v: &mut ValidationHarness, gpu: &GpuSimpleEt0) {
    println!("\n── Hamon PET ──");

    let lat_rad = 42.7_f64.to_radians();
    let inputs = vec![
        HamonInput { tmean_c: 20.0, latitude_rad: lat_rad, doy: 180 },
        HamonInput { tmean_c: 10.0, latitude_rad: lat_rad, doy: 90 },
        HamonInput { tmean_c: 30.0, latitude_rad: lat_rad, doy: 200 },
        HamonInput { tmean_c: 5.0, latitude_rad: lat_rad, doy: 60 },
    ];
    let cpu = BatchedSimpleEt0::hamon(&inputs);
    let gpu_result = gpu.hamon(&inputs).expect("GPU dispatch");

    for (i, (g, c)) in gpu_result.iter().zip(&cpu).enumerate() {
        let ok = parity_ok(*g, *c, 1e-2, 0.02);
        println!("  Hamon[{i}]: CPU={c:.4} GPU={g:.4} |Δ|={:.6}", (g - c).abs());
        v.check_bool(&format!("Hamon_{i}"), ok);
    }
}

fn validate_blaney_criddle(v: &mut ValidationHarness, gpu: &GpuSimpleEt0) {
    println!("\n── Blaney-Criddle ET₀ ──");

    let lat_rad = 42.7_f64.to_radians();
    let inputs = vec![
        BlaneyCriddleInput { tmean_c: 25.0, latitude_rad: lat_rad, doy: 180 },
        BlaneyCriddleInput { tmean_c: 5.0, latitude_rad: lat_rad, doy: 15 },
        BlaneyCriddleInput { tmean_c: 20.0, latitude_rad: lat_rad, doy: 120 },
        BlaneyCriddleInput { tmean_c: 15.0, latitude_rad: lat_rad, doy: 270 },
    ];
    let cpu = BatchedSimpleEt0::blaney_criddle(&inputs);
    let gpu_result = gpu.blaney_criddle(&inputs).expect("GPU dispatch");

    for (i, (g, c)) in gpu_result.iter().zip(&cpu).enumerate() {
        let ok = parity_ok(*g, *c, 1e-2, 0.02);
        println!("  BC[{i}]: CPU={c:.4} GPU={g:.4} |Δ|={:.6}", (g - c).abs());
        v.check_bool(&format!("BC_{i}"), ok);
    }
}

#[allow(clippy::many_single_char_names)]
fn validate_batch_scaling(v: &mut ValidationHarness, le: &LocalElementwise) {
    println!("\n── Batch Scaling ──");

    let n = 10_000;
    let a: Vec<f64> = (0..n).map(|i| (i as f64).mul_add(0.02, 10.0)).collect();
    let b: Vec<f64> = (0..n).map(|i| (i as f64).mul_add(0.004, 60.0)).collect();
    let c: Vec<f64> = vec![0.2; n];

    let gpu = le.dispatch(LocalOp::ScsCnRunoff, &a, &b, &c).expect("10K batch");
    v.check_bool("batch_10K_len", gpu.len() == n);

    let all_finite = gpu.iter().all(|r| r.is_finite());
    v.check_bool("batch_10K_finite", all_finite);

    let monotonic = gpu.windows(2).all(|w| w[1] >= w[0] - 0.1);
    v.check_bool("batch_10K_monotonic", monotonic);

    println!("  10K batch: {} results, all finite={all_finite}, monotonic={monotonic}", gpu.len());
}

fn validate_edge_cases(v: &mut ValidationHarness, le: &LocalElementwise) {
    println!("\n── Edge Cases ──");

    let empty = le.dispatch(LocalOp::ScsCnRunoff, &[], &[], &[]).expect("empty");
    v.check_bool("empty_dispatch", empty.is_empty());

    let single = le
        .dispatch(LocalOp::StewartYield, &[1.0], &[0.9], &[0.0])
        .expect("single");
    v.check_bool("single_element", single.len() == 1);

    let zero_p = le
        .dispatch(LocalOp::ScsCnRunoff, &[0.0], &[80.0], &[0.2])
        .expect("zero P");
    v.check_bool("zero_precip_zero_runoff", zero_p[0].abs() < 0.01);

    let full_yield = le
        .dispatch(LocalOp::StewartYield, &[1.0], &[1.0], &[0.0])
        .expect("full yield");
    v.check_bool("full_et_full_yield", (full_yield[0] - 1.0).abs() < 0.01);

    println!("  empty={}, single={:.4}, zero_p={:.4}, full={:.4}",
        empty.len(), single[0], zero_p[0], full_yield[0]);
}
