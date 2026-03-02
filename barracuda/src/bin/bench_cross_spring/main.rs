// SPDX-License-Identifier: AGPL-3.0-or-later
//! Cross-Spring Provenance Benchmark
//!
//! Exercises all airSpring GPU paths, benchmarks them against CPU baselines,
//! and reports the cross-spring shader lineage for each primitive.

#![warn(clippy::pedantic)]
#![allow(clippy::cast_precision_loss)]

mod data;
mod helpers;
mod impls;

use std::sync::Arc;

use barracuda::device::WgpuDevice;

use airspring_barracuda::gpu::device_info;

use helpers::{print_device_report, print_provenance_report, print_summary, run_bench};
use impls::{
    bench_anderson_chain, bench_anderson_regimes, bench_blaney_criddle, bench_bray_curtis_matrix,
    bench_crop_kc_stage, bench_diversity_alpha, bench_et0_cpu, bench_et0_gpu, bench_et0_parity,
    bench_green_ampt_ponding, bench_green_ampt_soils, bench_hargreaves_batch,
    bench_isotherm_global, bench_isotherm_nm, bench_kc_from_gdd, bench_mc_et0, bench_reduce_gpu,
    bench_richards_cn_diffusion, bench_richards_cpu, bench_richards_upstream, bench_scs_cn,
    bench_scs_cn_amc, bench_shannon_frequencies, bench_stream_cpu, bench_stream_gpu,
    bench_wb_cpu_season, bench_wb_gpu_step,
};

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter("info")
        .with_target(false)
        .init();

    println!("═══════════════════════════════════════════════════════════════");
    println!("  airSpring Cross-Spring Provenance Benchmark (v0.5.6)");
    println!("  ToadStool S70+ — Ops 5-8 Absorption, GPU-First Dispatch");
    println!("═══════════════════════════════════════════════════════════════\n");

    print_provenance_report();

    let device = device_info::try_f64_device();
    print_device_report(device.as_ref());

    let (pass, fail) = run_all_benchmarks(device.as_ref());
    print_summary(pass, fail);

    std::process::exit(i32::from(fail > 0));
}

fn run_all_benchmarks(device: Option<&Arc<WgpuDevice>>) -> (u32, u32) {
    let mut pass = 0u32;
    let mut fail = 0u32;

    println!("── Benchmark Results ────────────────────────────────────────\n");

    run_et0_benchmarks(device, &mut pass, &mut fail);
    run_hargreaves_benchmarks(&mut pass, &mut fail);
    run_ops_5_8_gpu_benchmarks(device, &mut pass, &mut fail);
    run_water_balance_benchmarks(device, &mut pass, &mut fail);
    run_reduce_benchmarks(device, &mut pass, &mut fail);
    run_stream_benchmarks(device, &mut pass, &mut fail);
    run_richards_benchmarks(&mut pass, &mut fail);
    run_isotherm_benchmarks(&mut pass, &mut fail);
    run_mc_et0_benchmarks(&mut pass, &mut fail);
    run_diversity_benchmarks(&mut pass, &mut fail);
    run_crop_kc_benchmarks(&mut pass, &mut fail);
    run_anderson_benchmarks(&mut pass, &mut fail);
    run_blaney_criddle_benchmarks(&mut pass, &mut fail);
    run_scs_cn_benchmarks(&mut pass, &mut fail);
    run_green_ampt_benchmarks(&mut pass, &mut fail);

    (pass, fail)
}

fn run_et0_benchmarks(device: Option<&Arc<WgpuDevice>>, pass: &mut u32, fail: &mut u32) {
    bench_suite!(
        pass,
        fail,
        ("ET₀ CPU baseline (N=365)", "hotSpring math_f64", || {
            bench_et0_cpu(365)
        }),
        ("ET₀ CPU batch (N=10000)", "hotSpring math_f64", || {
            bench_et0_cpu(10_000)
        }),
    );
    if let Some(dev) = device {
        bench_suite!(
            pass,
            fail,
            ("ET₀ GPU (N=365)", "hotSpring→ToadStool→GPU", || {
                bench_et0_gpu(dev, 365)
            }),
            ("ET₀ GPU (N=10000)", "hotSpring→ToadStool→GPU", || {
                bench_et0_gpu(dev, 10_000)
            }),
            (
                "ET₀ CPU↔GPU parity (N=200)",
                "cross-spring validation",
                || bench_et0_parity(dev, 200)
            ),
        );
    }
}

fn run_water_balance_benchmarks(device: Option<&Arc<WgpuDevice>>, pass: &mut u32, fail: &mut u32) {
    run_bench(
        pass,
        fail,
        "Water Balance CPU season (180d)",
        "airSpring domain",
        || bench_wb_cpu_season(180),
    );
    if let Some(dev) = device {
        run_bench(
            pass,
            fail,
            "Water Balance GPU step (N=500)",
            "airSpring→ToadStool→GPU",
            || bench_wb_gpu_step(dev, 500),
        );
    }
}

fn run_reduce_benchmarks(device: Option<&Arc<WgpuDevice>>, pass: &mut u32, fail: &mut u32) {
    if let Some(dev) = device {
        run_bench(
            pass,
            fail,
            "Seasonal Reduce GPU (N=2000)",
            "wetSpring→ToadStool→GPU",
            || bench_reduce_gpu(dev, 2000),
        );
    }
}

fn run_stream_benchmarks(device: Option<&Arc<WgpuDevice>>, pass: &mut u32, fail: &mut u32) {
    run_bench(
        pass,
        fail,
        "Stream Smoothing CPU (N=500, w=24)",
        "wetSpring moving_window",
        || bench_stream_cpu(500, 24),
    );
    if let Some(dev) = device {
        run_bench(
            pass,
            fail,
            "Stream Smoothing GPU (N=500, w=24)",
            "wetSpring→ToadStool→GPU",
            || bench_stream_gpu(dev, 500, 24),
        );
    }
}

fn run_richards_benchmarks(pass: &mut u32, fail: &mut u32) {
    bench_suite!(
        pass,
        fail,
        (
            "Richards CPU (sand, 0.1d)",
            "airSpring→ToadStool S40",
            bench_richards_cpu
        ),
        (
            "Richards upstream CN (sand, 0.1d)",
            "hotSpring CN f64 S62",
            bench_richards_upstream
        ),
        (
            "Richards CN diffusion (sand, 0.1d)",
            "hotSpring CN f64 S62",
            bench_richards_cn_diffusion
        ),
    );
}

fn run_isotherm_benchmarks(pass: &mut u32, fail: &mut u32) {
    bench_suite!(
        pass,
        fail,
        (
            "Isotherm NM (Langmuir, wood char)",
            "neuralSpring nelder_mead",
            bench_isotherm_nm
        ),
        (
            "Isotherm Global (LHS, 8 starts)",
            "neuralSpring multi_start_NM",
            bench_isotherm_global
        ),
    );
}

fn run_hargreaves_benchmarks(pass: &mut u32, fail: &mut u32) {
    bench_suite!(
        pass,
        fail,
        (
            "Hargreaves batch CPU (N=365)",
            "airSpring→ToadStool S66 hydrology",
            || bench_hargreaves_batch(365)
        ),
        (
            "Hargreaves batch CPU (N=10000)",
            "airSpring→ToadStool S66 hydrology",
            || bench_hargreaves_batch(10_000)
        ),
    );
}

#[allow(clippy::too_many_lines)]
fn run_ops_5_8_gpu_benchmarks(device: Option<&Arc<WgpuDevice>>, pass: &mut u32, fail: &mut u32) {
    use barracuda::ops::batched_elementwise_f64::{self as bef64, BatchedElementwiseF64, Op};

    let Some(dev) = device else {
        println!("  [SKIP] Ops 5-8 GPU — no f64 device\n");
        return;
    };
    let engine = match BatchedElementwiseF64::new(Arc::clone(dev)) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("  [SKIP] Ops 5-8 GPU init failed: {e}\n");
            return;
        }
    };

    run_bench(
        pass,
        fail,
        "SensorCal GPU (op=5, N=2000)",
        "airSpring→ToadStool S70+",
        || {
            let data: Vec<f64> = (0..2000)
                .map(|i| (f64::from(i) / 2000.0).mul_add(20_000.0, 5_000.0))
                .collect();
            let gpu = engine
                .execute(&data, 2000, Op::SensorCalibration)
                .expect("GPU engine initialization");
            let cpu: Vec<f64> = data
                .iter()
                .map(|&r| {
                    (2e-13_f64.mul_add(r, -4e-9))
                        .mul_add(r, 4e-5)
                        .mul_add(r, -0.0677)
                })
                .collect();
            let max_err = gpu
                .iter()
                .zip(&cpu)
                .map(|(g, c)| (g - c).abs())
                .fold(0.0_f64, f64::max);
            println!("    max_err={max_err:.2e}");
            max_err < 0.01
        },
    );

    run_bench(
        pass,
        fail,
        "Hargreaves GPU (op=6, N=2000)",
        "airSpring+hotSpring→S70+",
        || {
            let mut data = Vec::with_capacity(2000 * 4);
            for i in 0..2000 {
                data.push(25.0 + (f64::from(i) % 10.0));
                data.push(10.0 + (f64::from(i) % 5.0));
                data.push((42.0_f64).to_radians());
                data.push(100.0 + (f64::from(i) % 265.0));
            }
            let gpu = engine
                .execute(&data, 2000, Op::HargreavesEt0)
                .expect("GPU engine initialization");
            let cpu: Vec<f64> = data
                .chunks(4)
                .map(|c| bef64::hargreaves_et0_cpu(c[0], c[1], c[2], c[3]))
                .collect();
            let max_err = gpu
                .iter()
                .zip(&cpu)
                .map(|(g, c)| (g - c).abs())
                .fold(0.0_f64, f64::max);
            println!("    max_err={max_err:.4} mm/day");
            max_err < 0.1 && gpu.iter().all(|&x| x > 0.0)
        },
    );

    run_bench(
        pass,
        fail,
        "Kc Climate GPU (op=7, N=2000)",
        "airSpring FAO-56→S70+",
        || {
            let mut data = Vec::with_capacity(2000 * 4);
            for i in 0..2000 {
                data.push((f64::from(i) % 5.0).mul_add(0.1, 0.8));
                data.push(1.5 + (f64::from(i) % 4.0));
                data.push(30.0 + (f64::from(i) % 30.0));
                data.push((f64::from(i) % 4.0).mul_add(0.5, 0.5));
            }
            let gpu = engine
                .execute(&data, 2000, Op::KcClimateAdjust)
                .expect("GPU engine initialization");
            let cpu: Vec<f64> = data
                .chunks(4)
                .map(|c| bef64::kc_climate_adjust_cpu(c[0], c[1], c[2], c[3]))
                .collect();
            let max_err = gpu
                .iter()
                .zip(&cpu)
                .map(|(g, c)| (g - c).abs())
                .fold(0.0_f64, f64::max);
            println!("    max_err={max_err:.2e}");
            max_err < 0.01
        },
    );

    run_bench(
        pass,
        fail,
        "DualKc Ke GPU (op=8, N=1000)",
        "airSpring FAO-56 Ch7+11→S70+",
        || {
            let mut data = Vec::with_capacity(1000 * 9);
            for i in 0..1000 {
                data.push((f64::from(i) % 10.0).mul_add(0.1, 0.15));
                data.push(1.20);
                data.push((f64::from(i) % 10.0).mul_add(0.09, 0.05));
                data.push(if i % 3 == 0 { 0.4 } else { 1.0 });
                data.push(f64::from(i) % 15.0);
                data.push(9.0);
                data.push(22.5);
                data.push(if i % 4 == 0 { 5.0 } else { 0.0 });
                data.push(5.0);
            }
            let gpu = engine
                .execute(&data, 1000, Op::DualKcKe)
                .expect("GPU engine initialization");
            let all_valid = gpu.iter().all(|&x| (0.0..1.5).contains(&x));
            println!("    all Ke in [0, 1.5): {all_valid}");
            all_valid
        },
    );

    run_bench(
        pass,
        fail,
        "GPU↔CPU parity (op=5-8 sweep)",
        "cross-spring validation",
        || {
            let vwc = engine
                .execute(&[10_000.0], 1, Op::SensorCalibration)
                .expect("GPU engine initialization")[0];
            let kc = engine
                .execute(&[1.20, 2.0, 45.0, 2.0], 1, Op::KcClimateAdjust)
                .expect("GPU engine initialization")[0];
            let hg = engine
                .execute(&[30.0, 15.0, 0.733_f64, 187.0], 1, Op::HargreavesEt0)
                .expect("GPU engine initialization")[0];
            let vwc_ok = (vwc - 0.1323).abs() < 0.01;
            let kc_ok = (kc - 1.20).abs() < 0.001;
            let hg_ok = hg > 0.0 && hg < 12.0;
            println!("    VWC(10k)={vwc:.4} Kc(std)={kc:.4} HG={hg:.3}");
            vwc_ok && kc_ok && hg_ok
        },
    );
}

fn run_mc_et0_benchmarks(pass: &mut u32, fail: &mut u32) {
    run_bench(
        pass,
        fail,
        "MC ET₀ CPU (N=5000, parametric CI)",
        "groundSpring MC + hotSpring norm_ppf",
        bench_mc_et0,
    );
}

fn run_diversity_benchmarks(pass: &mut u32, fail: &mut u32) {
    bench_suite!(
        pass,
        fail,
        (
            "Diversity alpha (5-species mix)",
            "wetSpring→ToadStool S64 bio",
            bench_diversity_alpha
        ),
        (
            "Bray-Curtis matrix (20 samples)",
            "wetSpring→ToadStool S64 bio",
            bench_bray_curtis_matrix
        ),
        (
            "Shannon from frequencies (pre-norm)",
            "wetSpring→ToadStool S66",
            bench_shannon_frequencies
        ),
    );
}

fn run_crop_kc_benchmarks(pass: &mut u32, fail: &mut u32) {
    bench_suite!(
        pass,
        fail,
        (
            "Crop Kc stage interpolation (180d)",
            "airSpring→ToadStool S66 hydrology",
            bench_crop_kc_stage
        ),
        (
            "Kc from GDD (corn season)",
            "airSpring domain (FAO-56 Table 12)",
            bench_kc_from_gdd
        ),
    );
}

fn run_anderson_benchmarks(pass: &mut u32, fail: &mut u32) {
    bench_suite!(
        pass,
        fail,
        (
            "Anderson coupling chain (10K θ)",
            "groundSpring→airSpring cross-spring",
            bench_anderson_chain
        ),
        (
            "Anderson regime classification",
            "groundSpring spectral→airSpring eco",
            bench_anderson_regimes
        ),
    );
}

fn run_blaney_criddle_benchmarks(pass: &mut u32, fail: &mut u32) {
    run_bench(
        pass,
        fail,
        "Blaney-Criddle ET₀ (10K days)",
        "airSpring ET₀ (8th method)",
        bench_blaney_criddle,
    );
}

fn run_scs_cn_benchmarks(pass: &mut u32, fail: &mut u32) {
    bench_suite!(
        pass,
        fail,
        (
            "SCS-CN runoff (10K events)",
            "airSpring hydrology",
            bench_scs_cn
        ),
        (
            "SCS-CN AMC adjustment",
            "airSpring hydrology",
            bench_scs_cn_amc
        ),
    );
}

fn run_green_ampt_benchmarks(pass: &mut u32, fail: &mut u32) {
    bench_suite!(
        pass,
        fail,
        (
            "Green-Ampt infiltration (7 soils)",
            "airSpring soil physics",
            bench_green_ampt_soils
        ),
        (
            "Green-Ampt ponding time",
            "airSpring soil physics",
            bench_green_ampt_ponding
        ),
    );
}
