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
//! | `batched_elementwise_f64.wgsl` | airSpring+wetSpring | S42-S87 | 20 domain ops at f64 |
//! | `mc_et0_propagate_f64.wgsl` | groundSpring | S64 | MC uncertainty GPU kernel |
//! | `bootstrap_mean_f64.wgsl` | groundSpring | S71 | GPU bootstrap CI |
//! | `jackknife_mean_f64.wgsl` | groundSpring | S71 | GPU jackknife variance |
//! | `diversity_fusion_f64.wgsl` | wetSpring | S70 | Shannon+Simpson GPU |
//! | `kriging_f64.wgsl` | wetSpring | S42 | Spatial interpolation |
//! | `brent_f64.wgsl` | neuralSpring | S83 | Root-finding (VG inverse) |
//!
//! Note: `local_elementwise_f64.wgsl` (6 airSpring agri ops) has been fully
//! absorbed upstream into `batched_elementwise_f64.wgsl` ops 14-19.
//! airSpring now **leans** on the upstream primitive.
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
    clippy::cast_lossless,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]

use std::time::Instant;

use airspring_barracuda::eco;
use airspring_barracuda::gpu::device_info;
use airspring_barracuda::gpu::runoff::{GpuRunoff, RunoffInput};
use airspring_barracuda::gpu::simple_et0::{
    BlaneyCriddleInput, GpuSimpleEt0, HamonInput, MakkinkInput, TurcInput,
};
use airspring_barracuda::gpu::yield_response::{GpuYieldResponse, YieldInput};
use airspring_barracuda::validation;
use barracuda::validation::ValidationHarness;

fn main() {
    validation::init_tracing();
    validation::banner(
        "Exp 078: Cross-Spring Evolution — Upstream BatchedElementwiseF64 Benchmark",
    );

    let mut v = ValidationHarness::new("cross_spring_evolution");

    // ── Device & Precision Detection ─────────────────────────────────
    validation::section("Device & Precision Discovery");

    let Some(device) = device_info::try_f64_device() else {
        println!("  No GPU available — exiting gracefully.");
        barracuda::validation::exit_no_gpu();
    };

    let report = device_info::probe_device(&device);
    println!("{report}");

    let gpu_runoff = match GpuRunoff::new(device.clone()) {
        Ok(g) => g,
        Err(e) => {
            println!("  GPU executor init failed: {e}");
            barracuda::validation::exit_no_gpu();
        }
    };
    let gpu_yield = GpuYieldResponse::new(device.clone()).expect("GpuYieldResponse");
    let gpu_et0 = GpuSimpleEt0::new(device).expect("GpuSimpleEt0");

    v.check_bool("executor_created", true);

    // ── Precision Routing Advice ────────────────────────────────────
    validation::section("PrecisionRoutingAdvice (toadStool S128)");
    println!("  Routing: {:?}", report.precision_routing);
    match report.precision_routing {
        barracuda::device::driver_profile::PrecisionRoutingAdvice::F64Native => {
            println!("  → Full native f64 compute + shared-memory reductions");
        }
        barracuda::device::driver_profile::PrecisionRoutingAdvice::F64NativeNoSharedMem => {
            println!("  → Native f64 compute OK, shared-memory reductions broken");
            println!("  → Reductions route through DF64 or scalar f64");
        }
        barracuda::device::driver_profile::PrecisionRoutingAdvice::Df64Only => {
            println!("  → DF64 (f32-pair, ~48-bit) for all f64-class work");
        }
        barracuda::device::driver_profile::PrecisionRoutingAdvice::F32Only => {
            println!("  → f32 only — not suitable for science pipelines");
        }
    }

    // ── Upstream Provenance Registry ─────────────────────────────────
    validation::section("Upstream Provenance Registry");
    let upstream = device_info::upstream_airspring_provenance();
    println!("  airSpring consumes {} upstream shaders", upstream.len());
    for s in &upstream {
        println!("    {} [{}] — {}", s.path, s.category, s.evolution_note);
    }
    v.check_bool("upstream_registry_non_empty", !upstream.is_empty());

    // ── Cross-Spring Provenance ──────────────────────────────────────
    validation::section("Cross-Spring Shader Provenance (local)");
    print_provenance();

    // ── Op 17: SCS-CN Runoff (USDA TR-55) ────────────────────────────
    bench_scs_cn(&mut v, &gpu_runoff);

    // ── Op 18: Stewart Yield (Doorenbos & Kassam 1979) ────────────────
    bench_stewart(&mut v, &gpu_yield);

    // ── Ops 14-16, 19: Simple ET₀ Methods ────────────────────────────
    bench_makkink(&mut v, &gpu_et0);
    bench_turc(&mut v, &gpu_et0);
    bench_hamon(&mut v, &gpu_et0);
    bench_blaney_criddle(&mut v, &gpu_et0);

    // ── Batch Scaling ────────────────────────────────────────────────
    bench_scaling(&mut v, &gpu_runoff);

    // ── Summary ──────────────────────────────────────────────────────
    validation::section("Summary");
    println!("  Architecture: f64 canonical → BatchedElementwiseF64 → GPU");
    println!("  All 6 local ops absorbed upstream (ops 14-19).");
    println!("  Write → Absorb → Lean cycle: complete.");

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
    println!("  │  airSpring   → 6 ops absorbed into upstream ops 14-19      │");
    println!("  │               → 20 batched_elementwise_f64 ops total       │");
    println!("  └─────────────────────────────────────────────────────────────┘");
}

fn bench_scs_cn(v: &mut ValidationHarness, gpu: &GpuRunoff) {
    validation::section("Op 17: SCS-CN Runoff (BatchedElementwiseF64)");
    let n = 1000;
    let inputs: Vec<RunoffInput> = (0..n)
        .map(|i| RunoffInput {
            precip_mm: 10.0 + (i as f64) * 0.1,
            cn: 60.0 + (i as f64) * 0.03,
            ia_ratio: 0.2,
        })
        .collect();

    let start = Instant::now();
    let gpu_result = gpu.compute(&inputs).expect("SCS-CN GPU dispatch");
    let gpu_us = start.elapsed().as_micros();

    let start = Instant::now();
    let cpu: Vec<f64> = inputs
        .iter()
        .map(|i| eco::runoff::scs_cn_runoff(i.precip_mm, i.cn, i.ia_ratio))
        .collect();
    let cpu_us = start.elapsed().as_micros();

    let max_err = max_rel_error(&gpu_result, &cpu);
    println!("  N={n}, GPU={gpu_us}µs, CPU={cpu_us}µs, max_rel_err={max_err:.2e}");

    let tol = 1e-3;
    v.check_rel("scs_cn_parity", max_err, 0.0, tol);
}

fn bench_stewart(v: &mut ValidationHarness, gpu: &GpuYieldResponse) {
    validation::section("Op 18: Stewart Yield (BatchedElementwiseF64)");
    let n = 500;
    let inputs: Vec<YieldInput> = (0..n)
        .map(|i| YieldInput {
            ky: 0.5 + (i as f64) * 0.003,
            et_actual: (0.3 + (i as f64) * 0.0014) * 600.0,
            et_crop: 600.0,
        })
        .collect();

    let start = Instant::now();
    let gpu_result = gpu.compute(&inputs).expect("Stewart GPU dispatch");
    let gpu_us = start.elapsed().as_micros();

    let start = Instant::now();
    let cpu: Vec<f64> = inputs
        .iter()
        .map(|i| {
            let ratio = if i.et_crop > 0.0 {
                i.et_actual / i.et_crop
            } else {
                1.0
            };
            eco::yield_response::yield_ratio_single(i.ky, ratio).clamp(0.0, 1.0)
        })
        .collect();
    let cpu_us = start.elapsed().as_micros();

    let max_err = max_rel_error(&gpu_result, &cpu);
    println!("  N={n}, GPU={gpu_us}µs, CPU={cpu_us}µs, max_rel_err={max_err:.2e}");

    let tol = 1e-4;
    v.check_rel("stewart_parity", max_err, 0.0, tol);
}

fn bench_makkink(v: &mut ValidationHarness, gpu: &GpuSimpleEt0) {
    validation::section("Op 14: Makkink ET₀ (BatchedElementwiseF64)");
    let n = 500;
    let inputs: Vec<MakkinkInput> = (0..n)
        .map(|i| MakkinkInput {
            tmean_c: 5.0 + (i as f64) * 0.05,
            rs_mj: 8.0 + (i as f64) * 0.04,
            elevation_m: 150.0,
        })
        .collect();

    let start = Instant::now();
    let gpu_result = gpu.makkink(&inputs).expect("Makkink GPU dispatch");
    let gpu_us = start.elapsed().as_micros();

    let start = Instant::now();
    let cpu: Vec<f64> = inputs
        .iter()
        .map(|i| eco::simple_et0::makkink_et0(i.tmean_c, i.rs_mj, i.elevation_m))
        .collect();
    let cpu_us = start.elapsed().as_micros();

    let max_err = max_rel_error(&gpu_result, &cpu);
    println!("  N={n}, GPU={gpu_us}µs, CPU={cpu_us}µs, max_rel_err={max_err:.2e}");

    let tol = 5e-3;
    v.check_rel("makkink_parity", max_err, 0.0, tol);
}

fn bench_turc(v: &mut ValidationHarness, gpu: &GpuSimpleEt0) {
    validation::section("Op 15: Turc ET₀ (BatchedElementwiseF64)");
    let n = 500;
    let inputs: Vec<TurcInput> = (0..n)
        .map(|i| TurcInput {
            tmean_c: 10.0 + (i as f64) * 0.04,
            rs_mj: 10.0 + (i as f64) * 0.02,
            rh_pct: 30.0 + (i as f64) * 0.1,
        })
        .collect();

    let start = Instant::now();
    let gpu_result = gpu.turc(&inputs).expect("Turc GPU dispatch");
    let gpu_us = start.elapsed().as_micros();

    let start = Instant::now();
    let cpu: Vec<f64> = inputs
        .iter()
        .map(|i| eco::simple_et0::turc_et0(i.tmean_c, i.rs_mj, i.rh_pct))
        .collect();
    let cpu_us = start.elapsed().as_micros();

    let max_err = max_rel_error(&gpu_result, &cpu);
    println!("  N={n}, GPU={gpu_us}µs, CPU={cpu_us}µs, max_rel_err={max_err:.2e}");

    let tol = 5e-3;
    v.check_rel("turc_parity", max_err, 0.0, tol);
}

/// Upstream Hamon (1963 ASCE) CPU reference, matching the `BatchedElementwiseF64` shader.
fn hamon_upstream_ref(t_mean: f64, daylight_hours: f64) -> f64 {
    let d_ratio = daylight_hours / 12.0;
    let pt = 4.95 * (0.062 * t_mean).exp() / 100.0;
    (13.97 * d_ratio * d_ratio * pt).max(0.0)
}

fn bench_hamon(v: &mut ValidationHarness, gpu: &GpuSimpleEt0) {
    validation::section("Op 16: Hamon PET (BatchedElementwiseF64)");
    let n = 365;
    let lat_rad = 42.7_f64.to_radians();
    let inputs: Vec<HamonInput> = (1..=n)
        .map(|d| HamonInput {
            tmean_c: 20.0,
            latitude_rad: lat_rad,
            doy: d as u32,
        })
        .collect();

    let start = Instant::now();
    let gpu_result = gpu.hamon(&inputs).expect("Hamon GPU dispatch");
    let gpu_us = start.elapsed().as_micros();

    let start = Instant::now();
    let cpu: Vec<f64> = inputs
        .iter()
        .map(|i| {
            let dlh = eco::solar::daylight_hours(i.latitude_rad, i.doy);
            hamon_upstream_ref(i.tmean_c, dlh)
        })
        .collect();
    let cpu_us = start.elapsed().as_micros();

    let max_err = max_rel_error(&gpu_result, &cpu);
    println!("  N={n}, GPU={gpu_us}µs, CPU={cpu_us}µs, max_rel_err={max_err:.2e}");

    let tol = 1e-2;
    v.check_rel("hamon_parity", max_err, 0.0, tol);
}

fn bench_blaney_criddle(v: &mut ValidationHarness, gpu: &GpuSimpleEt0) {
    validation::section("Op 19: Blaney-Criddle ET₀ (BatchedElementwiseF64)");
    let n = 365;
    let lat_rad = 35.0_f64.to_radians();
    let inputs: Vec<BlaneyCriddleInput> = (1..=n)
        .map(|d| BlaneyCriddleInput {
            tmean_c: 22.0,
            latitude_rad: lat_rad,
            doy: d as u32,
        })
        .collect();

    let start = Instant::now();
    let gpu_result = gpu.blaney_criddle(&inputs).expect("BC GPU dispatch");
    let gpu_us = start.elapsed().as_micros();

    let start = Instant::now();
    let cpu: Vec<f64> = inputs
        .iter()
        .map(|i| eco::simple_et0::blaney_criddle_from_location(i.tmean_c, i.latitude_rad, i.doy))
        .collect();
    let cpu_us = start.elapsed().as_micros();

    let max_err = max_rel_error(&gpu_result, &cpu);
    println!("  N={n}, GPU={gpu_us}µs, CPU={cpu_us}µs, max_rel_err={max_err:.2e}");

    let tol = 1e-2;
    v.check_rel("blaney_criddle_parity", max_err, 0.0, tol);
}

fn bench_scaling(v: &mut ValidationHarness, gpu: &GpuRunoff) {
    validation::section("Batch Scaling (N=100, 1K, 10K, 100K)");
    for &n in &[100, 1_000, 10_000, 100_000] {
        let inputs: Vec<RunoffInput> = (0..n)
            .map(|i| RunoffInput {
                precip_mm: 20.0 + (i as f64) * 0.001,
                cn: 75.0,
                ia_ratio: 0.2,
            })
            .collect();

        let start = Instant::now();
        let gpu_result = gpu.compute(&inputs).expect("scaling dispatch");
        let elapsed = start.elapsed().as_micros();

        let cpu_ref = eco::runoff::scs_cn_runoff(inputs[0].precip_mm, 75.0, 0.2);
        let err = (gpu_result[0] - cpu_ref).abs();
        println!("  N={n:>6}: {elapsed:>6}µs, first_err={err:.2e}");
        v.check_bool(&format!("scaling_n{n}"), gpu_result.len() == n);
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
