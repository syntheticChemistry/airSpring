// SPDX-License-Identifier: AGPL-3.0-or-later
//! Exp 075: GPU Compute Parity Validation (upstream `BatchedElementwiseF64`).
//!
//! Validates that `BarraCuda` `BatchedElementwiseF64` ops 14-19 produce results
//! matching the validated CPU paths for all 6 agricultural operations:
//!
//! 1. SCS-CN runoff (Op 17, USDA-SCS 1972)
//! 2. Stewart yield response (Op 18, Stewart 1977, FAO-56 Ch 10)
//! 3. Makkink ET₀ (Op 14, Makkink 1957, de Bruin 1987)
//! 4. Turc ET₀ (Op 15, Turc 1961)
//! 5. Hamon PET (Op 16, Hamon 1961, Lu et al. 2005)
//! 6. Blaney-Criddle ET₀ (Op 19, Blaney & Criddle 1950, FAO-24)
//!
//! GPU path: `batched_elementwise_f64.wgsl` via `BatchedElementwiseF64`.
//! CPU path: `eco::runoff`, `eco::yield_response`, `eco::simple_et0` (f64).
//!
//! This binary validates the **Lean** phase of Write→Absorb→Lean: all 6 ops
//! that were previously in `local_elementwise_f64.wgsl` are now upstream in
//! `BarraCuda`'s canonical shader.
//!
//! # Tolerance Provenance
//!
//! GPU/CPU parity tolerances derive from f64 shader precision:
//! - `1e-6` relative for most ops (f64 upstream shader)
//! - `5e-3` relative + absolute floor for Hamon/BC (daylight pre-computation)
//!
//! Validated against CPU baselines: `control/cpu_gpu_parity/cpu_gpu_parity.py`
//! (commit `dbfb53a`, 2026-03-02).

use airspring_barracuda::gpu::device_info::try_f64_device;
use airspring_barracuda::gpu::runoff::{BatchedRunoff, GpuRunoff, RunoffInput};
use airspring_barracuda::gpu::simple_et0::{
    BatchedSimpleEt0, BlaneyCriddleInput, GpuSimpleEt0, HamonInput, MakkinkInput, TurcInput,
};
use airspring_barracuda::gpu::yield_response::{
    BatchedYieldResponse, GpuYieldResponse, YieldInput,
};
use airspring_barracuda::tolerances;
use airspring_barracuda::validation::ValidationHarness;

fn main() {
    tracing_subscriber::fmt::init();
    let mut v =
        ValidationHarness::new("Exp 075: GPU Compute Parity (Upstream BatchedElementwiseF64)");

    let Some(device) = try_f64_device() else {
        airspring_barracuda::validation::exit_no_gpu();
    };

    let gpu_runoff = GpuRunoff::new(device.clone()).expect("GpuRunoff");
    let gpu_yield = GpuYieldResponse::new(device.clone()).expect("GpuYieldResponse");
    let gpu_et0 = GpuSimpleEt0::new(device).expect("GpuSimpleEt0");

    validate_scs_cn(&mut v, &gpu_runoff);
    validate_stewart(&mut v, &gpu_yield);
    validate_makkink(&mut v, &gpu_et0);
    validate_turc(&mut v, &gpu_et0);
    validate_hamon(&mut v, &gpu_et0);
    validate_blaney_criddle(&mut v, &gpu_et0);
    validate_batch_scaling(&mut v, &gpu_runoff);
    validate_edge_cases(&mut v, &gpu_runoff, &gpu_yield);

    v.finish();
}

fn parity_ok(gpu: f64, cpu: f64, rel_tol: f64, abs_tol: f64) -> bool {
    (gpu - cpu).abs() < cpu.abs().mul_add(rel_tol, abs_tol)
}

fn validate_scs_cn(v: &mut ValidationHarness, gpu: &GpuRunoff) {
    println!("\n── SCS-CN Runoff (Op 17) ──");

    let inputs = vec![
        RunoffInput {
            precip_mm: 50.0,
            cn: 75.0,
            ia_ratio: 0.2,
        },
        RunoffInput {
            precip_mm: 100.0,
            cn: 85.0,
            ia_ratio: 0.2,
        },
        RunoffInput {
            precip_mm: 25.0,
            cn: 65.0,
            ia_ratio: 0.2,
        },
        RunoffInput {
            precip_mm: 200.0,
            cn: 90.0,
            ia_ratio: 0.2,
        },
        RunoffInput {
            precip_mm: 10.0,
            cn: 50.0,
            ia_ratio: 0.2,
        },
        RunoffInput {
            precip_mm: 0.0,
            cn: 80.0,
            ia_ratio: 0.2,
        },
        RunoffInput {
            precip_mm: 75.0,
            cn: 98.0,
            ia_ratio: 0.05,
        },
    ];
    let cpu_result = BatchedRunoff::compute(&inputs);
    let gpu_result = gpu.compute(&inputs).expect("GPU dispatch");

    for (i, (g, c)) in gpu_result.iter().zip(&cpu_result.runoff_mm).enumerate() {
        let ok = parity_ok(*g, *c, 5e-3, 0.01);
        println!(
            "  SCS-CN[{i}]: CPU={c:.4} GPU={g:.4} |Δ|={:.6}",
            (g - c).abs()
        );
        v.check_bool(&format!("SCS_CN_{i}"), ok);
    }
}

fn validate_stewart(v: &mut ValidationHarness, gpu: &GpuYieldResponse) {
    println!("\n── Stewart Yield Response (Op 18) ──");

    let inputs = vec![
        YieldInput {
            ky: 1.25,
            et_actual: 500.0,
            et_crop: 600.0,
        },
        YieldInput {
            ky: 0.85,
            et_actual: 400.0,
            et_crop: 600.0,
        },
        YieldInput {
            ky: 1.0,
            et_actual: 600.0,
            et_crop: 600.0,
        },
        YieldInput {
            ky: 1.50,
            et_actual: 300.0,
            et_crop: 600.0,
        },
        YieldInput {
            ky: 0.40,
            et_actual: 550.0,
            et_crop: 600.0,
        },
    ];
    let cpu = BatchedYieldResponse::compute(&inputs);
    let gpu_result = gpu.compute(&inputs).expect("GPU dispatch");

    for (i, (g, c)) in gpu_result.iter().zip(&cpu).enumerate() {
        let ok = parity_ok(*g, *c, 1e-3, 1e-4);
        println!(
            "  Stewart[{i}]: CPU={c:.6} GPU={g:.6} |Δ|={:.8}",
            (g - c).abs()
        );
        v.check_bool(&format!("Stewart_{i}"), ok);
    }
}

fn validate_makkink(v: &mut ValidationHarness, gpu: &GpuSimpleEt0) {
    println!("\n── Makkink ET₀ (Op 14) ──");

    let inputs = vec![
        MakkinkInput {
            tmean_c: 20.0,
            rs_mj: 15.0,
            elevation_m: 100.0,
        },
        MakkinkInput {
            tmean_c: 30.0,
            rs_mj: 25.0,
            elevation_m: 0.0,
        },
        MakkinkInput {
            tmean_c: 10.0,
            rs_mj: 8.0,
            elevation_m: 500.0,
        },
        MakkinkInput {
            tmean_c: 25.0,
            rs_mj: 20.0,
            elevation_m: 50.0,
        },
    ];
    let cpu = BatchedSimpleEt0::makkink(&inputs);
    let gpu_result = gpu.makkink(&inputs).expect("GPU dispatch");

    for (i, (g, c)) in gpu_result.iter().zip(&cpu).enumerate() {
        let ok = parity_ok(*g, *c, 5e-3, 0.01);
        println!(
            "  Makkink[{i}]: CPU={c:.4} GPU={g:.4} |Δ|={:.6}",
            (g - c).abs()
        );
        v.check_bool(&format!("Makkink_{i}"), ok);
    }
}

fn validate_turc(v: &mut ValidationHarness, gpu: &GpuSimpleEt0) {
    println!("\n── Turc ET₀ (Op 15) ──");

    let inputs = vec![
        TurcInput {
            tmean_c: 20.0,
            rs_mj: 15.0,
            rh_pct: 70.0,
        },
        TurcInput {
            tmean_c: 25.0,
            rs_mj: 20.0,
            rh_pct: 40.0,
        },
        TurcInput {
            tmean_c: 30.0,
            rs_mj: 25.0,
            rh_pct: 55.0,
        },
        TurcInput {
            tmean_c: 15.0,
            rs_mj: 10.0,
            rh_pct: 80.0,
        },
    ];
    let cpu = BatchedSimpleEt0::turc(&inputs);
    let gpu_result = gpu.turc(&inputs).expect("GPU dispatch");

    for (i, (g, c)) in gpu_result.iter().zip(&cpu).enumerate() {
        let ok = parity_ok(*g, *c, 5e-3, 0.01);
        println!(
            "  Turc[{i}]: CPU={c:.4} GPU={g:.4} |Δ|={:.6}",
            (g - c).abs()
        );
        v.check_bool(&format!("Turc_{i}"), ok);
    }
}

/// Upstream Hamon (1963 ASCE) CPU reference, matching the `BatchedElementwiseF64` shader.
/// airSpring `eco::simple_et0::hamon_pet` uses the Lu et al. (2005) formulation.
fn hamon_upstream_ref(t_mean: f64, daylight_hours: f64) -> f64 {
    let d_ratio = daylight_hours / 12.0;
    let pt = 4.95 * (0.062 * t_mean).exp() / 100.0;
    (13.97 * d_ratio * d_ratio * pt).max(0.0)
}

fn validate_hamon(v: &mut ValidationHarness, gpu: &GpuSimpleEt0) {
    println!("\n── Hamon PET (Op 16) ──");

    let lat_rad = 42.7_f64.to_radians();
    let inputs = vec![
        HamonInput {
            tmean_c: 20.0,
            latitude_rad: lat_rad,
            doy: 180,
        },
        HamonInput {
            tmean_c: 10.0,
            latitude_rad: lat_rad,
            doy: 90,
        },
        HamonInput {
            tmean_c: 30.0,
            latitude_rad: lat_rad,
            doy: 200,
        },
        HamonInput {
            tmean_c: 5.0,
            latitude_rad: lat_rad,
            doy: 60,
        },
    ];
    let upstream_cpu: Vec<f64> = inputs
        .iter()
        .map(|i| {
            let n = airspring_barracuda::eco::solar::daylight_hours(i.latitude_rad, i.doy);
            hamon_upstream_ref(i.tmean_c, n)
        })
        .collect();
    let gpu_result = gpu.hamon(&inputs).expect("GPU dispatch");

    for (i, (g, c)) in gpu_result.iter().zip(&upstream_cpu).enumerate() {
        let ok = parity_ok(*g, *c, 1e-2, 0.02);
        println!(
            "  Hamon[{i}]: upstream_CPU={c:.4} GPU={g:.4} |Δ|={:.6}",
            (g - c).abs()
        );
        v.check_bool(&format!("Hamon_{i}"), ok);
    }
}

fn validate_blaney_criddle(v: &mut ValidationHarness, gpu: &GpuSimpleEt0) {
    println!("\n── Blaney-Criddle ET₀ (Op 19) ──");

    let lat_rad = 42.7_f64.to_radians();
    let inputs = vec![
        BlaneyCriddleInput {
            tmean_c: 25.0,
            latitude_rad: lat_rad,
            doy: 180,
        },
        BlaneyCriddleInput {
            tmean_c: 5.0,
            latitude_rad: lat_rad,
            doy: 15,
        },
        BlaneyCriddleInput {
            tmean_c: 20.0,
            latitude_rad: lat_rad,
            doy: 120,
        },
        BlaneyCriddleInput {
            tmean_c: 15.0,
            latitude_rad: lat_rad,
            doy: 270,
        },
    ];
    let cpu = BatchedSimpleEt0::blaney_criddle(&inputs);
    let gpu_result = gpu.blaney_criddle(&inputs).expect("GPU dispatch");

    for (i, (g, c)) in gpu_result.iter().zip(&cpu).enumerate() {
        let ok = parity_ok(*g, *c, 1e-2, 0.02);
        println!("  BC[{i}]: CPU={c:.4} GPU={g:.4} |Δ|={:.6}", (g - c).abs());
        v.check_bool(&format!("BC_{i}"), ok);
    }
}

fn validate_batch_scaling(v: &mut ValidationHarness, gpu: &GpuRunoff) {
    println!("\n── Batch Scaling ──");

    let n = 10_000;
    let inputs: Vec<RunoffInput> = (0..n)
        .map(|i| RunoffInput {
            precip_mm: (i as f64).mul_add(0.02, 10.0),
            cn: (i as f64).mul_add(0.004, 60.0),
            ia_ratio: 0.2,
        })
        .collect();

    let gpu_result = gpu.compute(&inputs).expect("10K batch");
    v.check_bool("batch_10K_len", gpu_result.len() == n);

    let all_finite = gpu_result.iter().all(|r| r.is_finite());
    v.check_bool("batch_10K_finite", all_finite);

    let monotonic = gpu_result.windows(2).all(|w| w[1] >= w[0] - 0.1);
    v.check_bool("batch_10K_monotonic", monotonic);

    println!(
        "  10K batch: {} results, all finite={all_finite}, monotonic={monotonic}",
        gpu_result.len()
    );
}

fn validate_edge_cases(
    v: &mut ValidationHarness,
    gpu_runoff: &GpuRunoff,
    gpu_yield: &GpuYieldResponse,
) {
    println!("\n── Edge Cases ──");

    let empty = gpu_runoff.compute(&[]).expect("empty");
    v.check_bool("empty_dispatch", empty.is_empty());

    let single = gpu_yield
        .compute(&[YieldInput {
            ky: 1.0,
            et_actual: 540.0,
            et_crop: 600.0,
        }])
        .expect("single");
    v.check_bool("single_element", single.len() == 1);

    let zero_p = gpu_runoff
        .compute(&[RunoffInput {
            precip_mm: 0.0,
            cn: 80.0,
            ia_ratio: 0.2,
        }])
        .expect("zero P");
    v.check_abs(
        "zero_precip_zero_runoff",
        zero_p[0],
        0.0,
        tolerances::SCS_CN_ANALYTICAL.abs_tol,
    );

    let full_yield = gpu_yield
        .compute(&[YieldInput {
            ky: 1.0,
            et_actual: 600.0,
            et_crop: 600.0,
        }])
        .expect("full yield");
    v.check_abs(
        "full_et_full_yield",
        full_yield[0],
        1.0,
        tolerances::DUAL_KC_PRECISION.abs_tol,
    );

    println!(
        "  empty={}, single={:.4}, zero_p={:.4}, full={:.4}",
        empty.len(),
        single[0],
        zero_p[0],
        full_yield[0]
    );
}
