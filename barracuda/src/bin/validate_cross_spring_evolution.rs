// SPDX-License-Identifier: AGPL-3.0-or-later
//! Cross-Spring Evolution Validation — `BarraCuda` Universal Precision Benchmark
//!
//! Validates the modern `BarraCuda` GPU pipeline with cross-spring provenance
//! tracking. Documents when and where each primitive evolved to benefit the
//! ecosystem.
//!
//! # Cross-Spring Shader Provenance
//!
//! | Primitive | Origin Spring | Session | What It Enables |
//! |-----------|---------------|---------|-----------------|
//! | `math_f64.wgsl` (exp, log, pow, sin, cos, acos) | hotSpring | S54 | Full f64 precision for all shaders |
//! | `df64_core.wgsl` + `df64_transcendentals.wgsl` | hotSpring | S58-S71 | ~48-bit on consumer GPUs |
//! | `compile_shader_universal` | neuralSpring | S68 | One source → any precision |
//! | `Fp64Strategy` / `GpuDriverProfile` | hotSpring | S58 | Auto precision per hardware |
//! | `batched_elementwise_f64.wgsl` | airSpring+wetSpring | S42-S79 | 14 domain ops at f64 |
//! | `mc_et0_propagate_f64.wgsl` | groundSpring | S64 | MC uncertainty GPU kernel |
//! | `bootstrap_mean_f64.wgsl` | groundSpring | S71 | GPU bootstrap CI |
//! | `jackknife_mean_f64.wgsl` | groundSpring | S71 | GPU jackknife variance |
//! | `diversity_fusion_f64.wgsl` | wetSpring | S70 | Shannon+Simpson GPU |
//! | `kriging_f64.wgsl` | wetSpring | S42 | Spatial interpolation |
//! | `brent_f64.wgsl` | neuralSpring | S83 | Root-finding (VG inverse) |
//! | `local_elementwise_f64.wgsl` | airSpring | V0.6.9 | 6 agri ops, f64 canonical |
//!
//! # Architecture
//!
//! ```text
//! f64 canonical WGSL source (math is universal)
//!        │
//!        ▼
//! compile_shader_universal(source, precision, label)
//!        │
//!  ┌─────┼─────┐
//!  F64   F32   Df64
//!  │     │     │
//!  Titan RTX   Arc
//!  V     4070  A770
//! ```

#![allow(
    clippy::cast_precision_loss,
    clippy::suboptimal_flops,
    clippy::cast_lossless
)]

use std::sync::Arc;
use std::time::Instant;

use airspring_barracuda::gpu::device_info;
use airspring_barracuda::gpu::local_dispatch::{LocalElementwise, LocalOp};
use airspring_barracuda::validation;
use barracuda::validation::ValidationHarness;

fn main() {
    validation::init_tracing();
    validation::banner("Exp 078: Cross-Spring Evolution — Universal Precision Benchmark");

    let mut v = ValidationHarness::new("cross_spring_evolution");

    // ── Device & Precision Detection ─────────────────────────────────
    validation::section("Device & Precision Discovery");

    let Some(device) = device_info::try_f64_device() else {
        println!("  No GPU available — exiting gracefully.");
        barracuda::validation::exit_no_gpu();
    };

    let report = device_info::probe_device(&device);
    println!("{report}");

    let le = match LocalElementwise::new(Arc::clone(&device)) {
        Ok(le) => le,
        Err(e) => {
            println!("  Shader compilation failed: {e}");
            barracuda::validation::exit_no_gpu();
        }
    };

    println!("  Compiled precision: {:?}", le.precision());
    v.check_bool("shader_compiled", true);

    // ── Cross-Spring Provenance ──────────────────────────────────────
    validation::section("Cross-Spring Shader Provenance");
    print_provenance();

    // ── Op 0: SCS-CN Runoff (USDA TR-55) ────────────────────────────
    bench_scs_cn(&mut v, &le);

    // ── Op 1: Stewart Yield (Doorenbos & Kassam 1979) ────────────────
    bench_stewart(&mut v, &le);

    // ── Ops 2-5: Simple ET₀ Methods ─────────────────────────────────
    bench_makkink(&mut v, &le);
    bench_turc(&mut v, &le);
    bench_hamon(&mut v, &le);
    bench_blaney_criddle(&mut v, &le);

    // ── Batch Scaling ────────────────────────────────────────────────
    bench_scaling(&mut v, &le);

    // ── Summary ──────────────────────────────────────────────────────
    validation::section("Summary");
    println!("  Precision: {:?}", le.precision());
    println!("  Architecture: f64 canonical → compile_shader_universal → GPU");

    v.finish();
}

fn print_provenance() {
    println!("  ┌─────────────────────────────────────────────────────────────┐");
    println!("  │  Cross-Spring Shader Evolution Provenance                   │");
    println!("  ├─────────────────────────────────────────────────────────────┤");
    println!("  │  hotSpring   → precision: math_f64.wgsl (pow, acos, sin)   │");
    println!("  │               → Fp64Strategy, GpuDriverProfile (S58)       │");
    println!("  │               → df64_core.wgsl, df64_transcendentals (S71) │");
    println!("  │  wetSpring   → bio: diversity_fusion_f64.wgsl (S70)        │");
    println!("  │               → spatial: kriging_f64.wgsl (S42)            │");
    println!("  │  groundSpring→ uncertainty: mc_et0_propagate_f64.wgsl (S64)│");
    println!("  │               → stats: bootstrap/jackknife_mean_f64 (S71)  │");
    println!("  │  neuralSpring→ architecture: compile_shader_universal (S68) │");
    println!("  │               → optimizer: brent_f64.wgsl (S83)            │");
    println!("  │  airSpring   → domain: local_elementwise_f64.wgsl (V0.6.9) │");
    println!("  │               → 14 batched_elementwise_f64 ops (S42-S79)   │");
    println!("  └─────────────────────────────────────────────────────────────┘");
}

fn bench_scs_cn(v: &mut ValidationHarness, le: &LocalElementwise) {
    validation::section("Op 0: SCS-CN Runoff (airSpring → f64 canonical)");
    let n = 1000;
    let p: Vec<f64> = (0..n).map(|i| 10.0 + (i as f64) * 0.1).collect();
    let cn: Vec<f64> = (0..n).map(|i| 60.0 + (i as f64) * 0.03).collect();
    let ia = vec![0.2; n];

    let start = Instant::now();
    let gpu = le
        .dispatch(LocalOp::ScsCnRunoff, &p, &cn, &ia)
        .expect("SCS-CN GPU dispatch");
    let gpu_us = start.elapsed().as_micros();

    let start = Instant::now();
    let cpu: Vec<f64> = p
        .iter()
        .zip(&cn)
        .zip(&ia)
        .map(|((&pp, &cc), &ii)| airspring_barracuda::eco::runoff::scs_cn_runoff(pp, cc, ii))
        .collect();
    let cpu_us = start.elapsed().as_micros();

    let max_err = max_rel_error(&gpu, &cpu);
    println!("  N={n}, GPU={gpu_us}µs, CPU={cpu_us}µs, max_rel_err={max_err:.2e}");

    let tol = precision_tol(le, 1e-10, 1e-3);
    v.check_rel("scs_cn_parity", max_err, 0.0, tol);
}

fn bench_stewart(v: &mut ValidationHarness, le: &LocalElementwise) {
    validation::section("Op 1: Stewart Yield (airSpring → f64 canonical)");
    let n = 500;
    let ky: Vec<f64> = (0..n).map(|i| 0.5 + (i as f64) * 0.003).collect();
    let ratio: Vec<f64> = (0..n).map(|i| 0.3 + (i as f64) * 0.0014).collect();
    let zeros = vec![0.0; n];

    let start = Instant::now();
    let gpu = le
        .dispatch(LocalOp::StewartYield, &ky, &ratio, &zeros)
        .expect("Stewart GPU dispatch");
    let gpu_us = start.elapsed().as_micros();

    let start = Instant::now();
    let cpu: Vec<f64> = ky
        .iter()
        .zip(&ratio)
        .map(|(&k, &r)| airspring_barracuda::eco::yield_response::yield_ratio_single(k, r))
        .collect();
    let cpu_us = start.elapsed().as_micros();

    let max_err = max_rel_error(&gpu, &cpu);
    println!("  N={n}, GPU={gpu_us}µs, CPU={cpu_us}µs, max_rel_err={max_err:.2e}");

    let tol = precision_tol(le, 1e-12, 1e-4);
    v.check_rel("stewart_parity", max_err, 0.0, tol);
}

fn bench_makkink(v: &mut ValidationHarness, le: &LocalElementwise) {
    validation::section("Op 2: Makkink ET₀ (airSpring → f64 canonical)");
    let n = 500;
    let t: Vec<f64> = (0..n).map(|i| 5.0 + (i as f64) * 0.05).collect();
    let rs: Vec<f64> = (0..n).map(|i| 8.0 + (i as f64) * 0.04).collect();
    let elev = vec![150.0; n];

    let start = Instant::now();
    let gpu = le
        .dispatch(LocalOp::MakkinkEt0, &t, &rs, &elev)
        .expect("Makkink GPU dispatch");
    let gpu_us = start.elapsed().as_micros();

    let start = Instant::now();
    let cpu: Vec<f64> = t
        .iter()
        .zip(&rs)
        .zip(&elev)
        .map(|((&tt, &rr), &ee)| airspring_barracuda::eco::simple_et0::makkink_et0(tt, rr, ee))
        .collect();
    let cpu_us = start.elapsed().as_micros();

    let max_err = max_rel_error(&gpu, &cpu);
    println!("  N={n}, GPU={gpu_us}µs, CPU={cpu_us}µs, max_rel_err={max_err:.2e}");

    let tol = precision_tol(le, 1e-8, 5e-3);
    v.check_rel("makkink_parity", max_err, 0.0, tol);
}

fn bench_turc(v: &mut ValidationHarness, le: &LocalElementwise) {
    validation::section("Op 3: Turc ET₀ (airSpring → f64 canonical)");
    let n = 500;
    let t: Vec<f64> = (0..n).map(|i| 10.0 + (i as f64) * 0.04).collect();
    let rs: Vec<f64> = (0..n).map(|i| 10.0 + (i as f64) * 0.02).collect();
    let rh: Vec<f64> = (0..n).map(|i| 30.0 + (i as f64) * 0.1).collect();

    let start = Instant::now();
    let gpu = le
        .dispatch(LocalOp::TurcEt0, &t, &rs, &rh)
        .expect("Turc GPU dispatch");
    let gpu_us = start.elapsed().as_micros();

    let start = Instant::now();
    let cpu: Vec<f64> = t
        .iter()
        .zip(&rs)
        .zip(&rh)
        .map(|((&tt, &rr), &hh)| airspring_barracuda::eco::simple_et0::turc_et0(tt, rr, hh))
        .collect();
    let cpu_us = start.elapsed().as_micros();

    let max_err = max_rel_error(&gpu, &cpu);
    println!("  N={n}, GPU={gpu_us}µs, CPU={cpu_us}µs, max_rel_err={max_err:.2e}");

    let tol = precision_tol(le, 1e-8, 5e-3);
    v.check_rel("turc_parity", max_err, 0.0, tol);
}

fn bench_hamon(v: &mut ValidationHarness, le: &LocalElementwise) {
    validation::section("Op 4: Hamon PET (airSpring → f64 canonical)");
    let n = 365;
    let t = vec![20.0; n];
    let lat = vec![42.7_f64.to_radians(); n];
    let doy: Vec<f64> = (1..=n).map(|d| d as f64).collect();

    let start = Instant::now();
    let gpu = le
        .dispatch(LocalOp::HamonPet, &t, &lat, &doy)
        .expect("Hamon GPU dispatch");
    let gpu_us = start.elapsed().as_micros();

    let start = Instant::now();
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let cpu: Vec<f64> = t
        .iter()
        .zip(&lat)
        .zip(&doy)
        .map(|((&tt, &ll), &dd)| {
            airspring_barracuda::eco::simple_et0::hamon_pet_from_location(tt, ll, dd as u32)
        })
        .collect();
    let cpu_us = start.elapsed().as_micros();

    let max_err = max_rel_error(&gpu, &cpu);
    println!("  N={n}, GPU={gpu_us}µs, CPU={cpu_us}µs, max_rel_err={max_err:.2e}");

    let tol = precision_tol(le, 1e-6, 1e-2);
    v.check_rel("hamon_parity", max_err, 0.0, tol);
}

fn bench_blaney_criddle(v: &mut ValidationHarness, le: &LocalElementwise) {
    validation::section("Op 5: Blaney-Criddle ET₀ (airSpring → f64 canonical)");
    let n = 365;
    let t = vec![22.0; n];
    let lat = vec![35.0_f64.to_radians(); n];
    let doy: Vec<f64> = (1..=n).map(|d| d as f64).collect();

    let start = Instant::now();
    let gpu = le
        .dispatch(LocalOp::BlaneyCriddleEt0, &t, &lat, &doy)
        .expect("BC GPU dispatch");
    let gpu_us = start.elapsed().as_micros();

    let start = Instant::now();
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let cpu: Vec<f64> = t
        .iter()
        .zip(&lat)
        .zip(&doy)
        .map(|((&tt, &ll), &dd)| {
            airspring_barracuda::eco::simple_et0::blaney_criddle_from_location(tt, ll, dd as u32)
        })
        .collect();
    let cpu_us = start.elapsed().as_micros();

    let max_err = max_rel_error(&gpu, &cpu);
    println!("  N={n}, GPU={gpu_us}µs, CPU={cpu_us}µs, max_rel_err={max_err:.2e}");

    let tol = precision_tol(le, 1e-6, 1e-2);
    v.check_rel("blaney_criddle_parity", max_err, 0.0, tol);
}

fn bench_scaling(v: &mut ValidationHarness, le: &LocalElementwise) {
    validation::section("Batch Scaling (N=100, 1K, 10K, 100K)");
    for &n in &[100, 1_000, 10_000, 100_000] {
        let p: Vec<f64> = (0..n).map(|i| 20.0 + (i as f64) * 0.001).collect();
        let cn = vec![75.0; n];
        let ia = vec![0.2; n];

        let start = Instant::now();
        let gpu = le
            .dispatch(LocalOp::ScsCnRunoff, &p, &cn, &ia)
            .expect("scaling dispatch");
        let elapsed = start.elapsed().as_micros();

        let cpu_ref = airspring_barracuda::eco::runoff::scs_cn_runoff(p[0], cn[0], ia[0]);
        let err = (gpu[0] - cpu_ref).abs();
        println!("  N={n:>6}: {elapsed:>6}µs, first_err={err:.2e}");
        v.check_bool(&format!("scaling_n{n}"), gpu.len() == n);
    }
}

fn max_rel_error(gpu: &[f64], cpu: &[f64]) -> f64 {
    gpu.iter().zip(cpu).fold(0.0_f64, |acc, (&g, &c)| {
        if c.abs() < 1e-15 {
            acc.max((g - c).abs())
        } else {
            acc.max(((g - c) / c).abs())
        }
    })
}

const fn precision_tol(le: &LocalElementwise, f64_tol: f64, f32_tol: f64) -> f64 {
    match le.precision() {
        barracuda::shaders::precision::Precision::F64 => f64_tol,
        _ => f32_tol,
    }
}
