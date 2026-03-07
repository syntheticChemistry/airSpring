// SPDX-License-Identifier: AGPL-3.0-or-later
//! v0.7.2+ modern upstream integration: ops 14-19, `PrecisionRoutingAdvice`,
//! upstream provenance registry, and cross-spring shader evolution timeline.

use std::time::Instant;

use airspring_barracuda::gpu::device_info;
use airspring_barracuda::gpu::runoff::{GpuRunoff, RunoffInput};
use airspring_barracuda::gpu::simple_et0::{
    BlaneyCriddleInput, GpuSimpleEt0, HamonInput, MakkinkInput, TurcInput,
};
use airspring_barracuda::gpu::yield_response::{GpuYieldResponse, YieldInput};
use barracuda::device::driver_profile::PrecisionRoutingAdvice;
use barracuda::validation::ValidationHarness;

pub fn bench_modern_upstream(v: &mut ValidationHarness) {
    println!("\n═══════════════════════════════════════════════════════════════");
    println!("  v0.7.3 Modern Upstream — PrecisionRouting + Ops 14-19");
    println!("  Write → Absorb → Lean cycle complete");
    println!("═══════════════════════════════════════════════════════════════");

    bench_upstream_provenance_registry(v);
    bench_precision_routing(v);
    bench_ops_14_19_gpu(v);
}

fn bench_upstream_provenance_registry(v: &mut ValidationHarness) {
    use barracuda::shaders::provenance::SpringDomain::{
        AirSpring, GroundSpring, HotSpring, NeuralSpring, WetSpring,
    };

    println!("\n── Upstream Provenance Registry (barraCuda shaders::provenance) ─");
    let t0 = Instant::now();

    let air_shaders = device_info::upstream_airspring_provenance();
    let n = air_shaders.len();
    v.check_lower("airSpring consumes upstream shaders", n as f64, 5.0);
    println!("  airSpring consumes {n} upstream shaders:");
    for s in &air_shaders {
        println!(
            "    {} [{}] — from {} (created: {}, absorbed: {})",
            s.path, s.category, s.origin, s.created, s.absorbed
        );
    }

    let matrix = device_info::upstream_cross_spring_matrix();
    let air_receives: usize = matrix
        .iter()
        .filter(|((_, to), _)| *to == AirSpring)
        .map(|(_, count)| count)
        .sum();
    let air_gives: usize = matrix
        .iter()
        .filter(|((from, _), _)| *from == AirSpring)
        .map(|(_, count)| count)
        .sum();
    println!("  Cross-spring flows: airSpring receives {air_receives}, gives {air_gives}");
    v.check_lower(
        "airSpring receives from other springs",
        air_receives as f64,
        1.0,
    );

    let cross = barracuda::shaders::provenance::cross_spring_shaders();
    v.check_lower(
        "total cross-spring shaders in ecosystem",
        cross.len() as f64,
        10.0,
    );
    println!("  Ecosystem: {} cross-spring shaders total", cross.len());

    println!("\n  Cross-Spring Dependency Matrix:");
    println!(
        "  {:>13} | {:>3} {:>3} {:>3} {:>3} {:>3}",
        "from\\to", "hot", "wet", "nrl", "air", "gnd"
    );
    println!(
        "  {:-<13}-+-{:-<3}-{:-<3}-{:-<3}-{:-<3}-{:-<3}",
        "", "", "", "", "", ""
    );
    let domains = [HotSpring, WetSpring, NeuralSpring, AirSpring, GroundSpring];
    for from in &domains {
        let row: Vec<String> = domains
            .iter()
            .map(|to| {
                if from == to {
                    " — ".to_string()
                } else {
                    format!("{:>3}", matrix.get(&(*from, *to)).copied().unwrap_or(0))
                }
            })
            .collect();
        println!("  {:>13} | {}", from, row.join(" "));
    }

    println!("  Upstream provenance: {:.1?}", t0.elapsed());
}

fn bench_precision_routing(v: &mut ValidationHarness) {
    println!("\n── PrecisionRoutingAdvice (toadStool S128) ─────────────────────");
    let t0 = Instant::now();

    let Some(device) = device_info::try_f64_device() else {
        println!("  No GPU — skipping precision routing benchmark");
        v.check_bool("precision_routing: no GPU (skip)", true);
        return;
    };

    let report = device_info::probe_device(&device);
    println!("{report}");

    let advice = report.precision_routing;
    println!("\n  Dispatch routing for this device:");
    match advice {
        PrecisionRoutingAdvice::F64Native => {
            println!("    → F64Native: full f64 compute + shared-memory reductions");
            println!("    → All 20 BatchedElementwiseF64 ops run at native f64");
            println!("    → Welford/correlation reductions use var<workgroup> f64");
        }
        PrecisionRoutingAdvice::F64NativeNoSharedMem => {
            println!(
                "    → F64NativeNoSharedMem: native f64 compute, but shared-mem returns zeros"
            );
            println!("    → BatchedElementwiseF64 ops: OK (no workgroup accumulators)");
            println!("    → Reductions (Welford, correlation): use DF64 or scalar path");
        }
        PrecisionRoutingAdvice::Df64Only => {
            println!("    → Df64Only: use DF64 (f32-pair, ~48-bit mantissa)");
            println!("    → All ops compile via compile_shader_universal → Df64");
            println!("    → Adequate for FAO-56 ET₀ (only needs ~6 significant digits)");
        }
        PrecisionRoutingAdvice::F32Only => {
            println!("    → F32Only: f32 only, no f64 support");
            println!("    → Edge/inference workloads only, not suitable for science pipelines");
        }
    }

    v.check_bool(&format!("precision_routing: {advice:?}"), true);
    println!("  Precision routing: {:.1?}", t0.elapsed());
}

fn bench_ops_14_19_gpu(v: &mut ValidationHarness) {
    println!("\n── Ops 14-19: Absorbed airSpring Agri Ops (BatchedElementwiseF64) ─");

    let Some(device) = device_info::try_f64_device() else {
        println!("  No GPU — skipping ops 14-19 benchmark");
        v.check_bool("ops_14_19: no GPU (skip)", true);
        return;
    };

    let gpu_runoff = match GpuRunoff::new(device.clone()) {
        Ok(g) => g,
        Err(e) => {
            println!("  GpuRunoff init failed: {e}");
            v.check_bool("ops_14_19: init failed", false);
            return;
        }
    };
    let gpu_yield = match GpuYieldResponse::new(device.clone()) {
        Ok(g) => g,
        Err(e) => {
            println!("  GpuYieldResponse init failed: {e}");
            v.check_bool("ops_14_19: init failed", false);
            return;
        }
    };
    let gpu_et0 = match GpuSimpleEt0::new(device) {
        Ok(g) => g,
        Err(e) => {
            println!("  GpuSimpleEt0 init failed: {e}");
            v.check_bool("ops_14_19: init failed", false);
            return;
        }
    };

    bench_op14_makkink(v, &gpu_et0);
    bench_op15_turc(v, &gpu_et0);
    bench_op16_hamon(v, &gpu_et0);
    bench_op17_scs_cn(v, &gpu_runoff);
    bench_op18_stewart(v, &gpu_yield);
    bench_op19_blaney_criddle(v, &gpu_et0);
    bench_scaling_modern(v, &gpu_runoff);
}

fn bench_op14_makkink(v: &mut ValidationHarness, gpu: &GpuSimpleEt0) {
    let n = 1000_i32;
    let inputs: Vec<MakkinkInput> = (0..n)
        .map(|i| {
            let fi = f64::from(i);
            MakkinkInput {
                tmean_c: fi.mul_add(0.03, 5.0),
                rs_mj: fi.mul_add(0.02, 8.0),
                elevation_m: 150.0,
            }
        })
        .collect();

    let t = Instant::now();
    let result = gpu.makkink(&inputs).expect("Makkink GPU");
    let us = t.elapsed().as_micros();

    let cpu: Vec<f64> = inputs
        .iter()
        .map(|i| {
            airspring_barracuda::eco::simple_et0::makkink_et0(i.tmean_c, i.rs_mj, i.elevation_m)
        })
        .collect();

    let err = max_rel_error(&result, &cpu);
    println!(
        "  Op 14 Makkink:       N={n:>5}, {us:>6}µs, err={err:.2e} [airSpring→upstream v0.7.2]"
    );
    v.check_rel("op14_makkink_parity", err, 0.0, 5e-3);
}

fn bench_op15_turc(v: &mut ValidationHarness, gpu: &GpuSimpleEt0) {
    let n = 1000_i32;
    let inputs: Vec<TurcInput> = (0..n)
        .map(|i| {
            let fi = f64::from(i);
            TurcInput {
                tmean_c: fi.mul_add(0.02, 10.0),
                rs_mj: fi.mul_add(0.015, 10.0),
                rh_pct: fi.mul_add(0.06, 30.0),
            }
        })
        .collect();

    let t = Instant::now();
    let result = gpu.turc(&inputs).expect("Turc GPU");
    let us = t.elapsed().as_micros();

    let cpu: Vec<f64> = inputs
        .iter()
        .map(|i| airspring_barracuda::eco::simple_et0::turc_et0(i.tmean_c, i.rs_mj, i.rh_pct))
        .collect();

    let err = max_rel_error(&result, &cpu);
    println!(
        "  Op 15 Turc:          N={n:>5}, {us:>6}µs, err={err:.2e} [airSpring→upstream v0.7.2]"
    );
    v.check_rel("op15_turc_parity", err, 0.0, 5e-3);
}

fn bench_op16_hamon(v: &mut ValidationHarness, gpu: &GpuSimpleEt0) {
    let n = 365_u32;
    let lat_rad = 42.7_f64.to_radians();
    let inputs: Vec<HamonInput> = (1..=n)
        .map(|d| HamonInput {
            tmean_c: 20.0,
            latitude_rad: lat_rad,
            doy: d,
        })
        .collect();

    let t = Instant::now();
    let result = gpu.hamon(&inputs).expect("Hamon GPU");
    let us = t.elapsed().as_micros();

    let cpu: Vec<f64> = inputs
        .iter()
        .map(|i| {
            let dlh = airspring_barracuda::eco::solar::daylight_hours(i.latitude_rad, i.doy);
            hamon_upstream_ref(i.tmean_c, dlh)
        })
        .collect();

    let err = max_rel_error(&result, &cpu);
    println!("  Op 16 Hamon (1963):  N={n:>5}, {us:>6}µs, err={err:.2e} [upstream ASCE formula]");
    v.check_rel("op16_hamon_parity", err, 0.0, 1e-2);
}

fn bench_op17_scs_cn(v: &mut ValidationHarness, gpu: &GpuRunoff) {
    let n = 1000_i32;
    let inputs: Vec<RunoffInput> = (0..n)
        .map(|i| {
            let fi = f64::from(i);
            RunoffInput {
                precip_mm: fi.mul_add(0.1, 10.0),
                cn: fi.mul_add(0.03, 60.0),
                ia_ratio: 0.2,
            }
        })
        .collect();

    let t = Instant::now();
    let result = gpu.compute(&inputs).expect("SCS-CN GPU");
    let us = t.elapsed().as_micros();

    let cpu: Vec<f64> = inputs
        .iter()
        .map(|i| airspring_barracuda::eco::runoff::scs_cn_runoff(i.precip_mm, i.cn, i.ia_ratio))
        .collect();

    let err = max_rel_error(&result, &cpu);
    println!(
        "  Op 17 SCS-CN:        N={n:>5}, {us:>6}µs, err={err:.2e} [airSpring→upstream v0.7.2]"
    );
    v.check_rel("op17_scs_cn_parity", err, 0.0, 1e-3);
}

fn bench_op18_stewart(v: &mut ValidationHarness, gpu: &GpuYieldResponse) {
    let n = 1000_i32;
    let inputs: Vec<YieldInput> = (0..n)
        .map(|i| {
            let fi = f64::from(i);
            YieldInput {
                ky: fi.mul_add(0.001, 0.5),
                et_actual: fi.mul_add(0.0007, 0.3) * 600.0,
                et_crop: 600.0,
            }
        })
        .collect();

    let t = Instant::now();
    let result = gpu.compute(&inputs).expect("Stewart GPU");
    let us = t.elapsed().as_micros();

    let cpu: Vec<f64> = inputs
        .iter()
        .map(|i| {
            let ratio = if i.et_crop > 0.0 {
                i.et_actual / i.et_crop
            } else {
                1.0
            };
            airspring_barracuda::eco::yield_response::yield_ratio_single(i.ky, ratio)
                .clamp(0.0, 1.0)
        })
        .collect();

    let err = max_rel_error(&result, &cpu);
    println!(
        "  Op 18 Stewart:       N={n:>5}, {us:>6}µs, err={err:.2e} [airSpring→upstream v0.7.2]"
    );
    v.check_rel("op18_stewart_parity", err, 0.0, 1e-4);
}

fn bench_op19_blaney_criddle(v: &mut ValidationHarness, gpu: &GpuSimpleEt0) {
    let n = 365_u32;
    let lat_rad = 35.0_f64.to_radians();
    let inputs: Vec<BlaneyCriddleInput> = (1..=n)
        .map(|d| BlaneyCriddleInput {
            tmean_c: 22.0,
            latitude_rad: lat_rad,
            doy: d,
        })
        .collect();

    let t = Instant::now();
    let result = gpu.blaney_criddle(&inputs).expect("BC GPU");
    let us = t.elapsed().as_micros();

    let cpu: Vec<f64> = inputs
        .iter()
        .map(|i| {
            airspring_barracuda::eco::simple_et0::blaney_criddle_from_location(
                i.tmean_c,
                i.latitude_rad,
                i.doy,
            )
        })
        .collect();

    let err = max_rel_error(&result, &cpu);
    println!(
        "  Op 19 Blaney-Criddle:N={n:>5}, {us:>6}µs, err={err:.2e} [airSpring→upstream v0.7.2]"
    );
    v.check_rel("op19_blaney_criddle_parity", err, 0.0, 1e-2);
}

fn bench_scaling_modern(v: &mut ValidationHarness, gpu: &GpuRunoff) {
    println!("\n  Batch Scaling (modern upstream dispatch):");
    for &n in &[100_usize, 1_000, 10_000, 100_000] {
        let inputs: Vec<RunoffInput> = (0..n)
            .map(|i| {
                #[allow(clippy::cast_precision_loss)]
                let fi = i as f64;
                RunoffInput {
                    precip_mm: fi.mul_add(0.001, 20.0),
                    cn: 75.0,
                    ia_ratio: 0.2,
                }
            })
            .collect();

        let t = Instant::now();
        let result = gpu.compute(&inputs).expect("scaling");
        let us = t.elapsed().as_micros();

        v.check_bool(&format!("scaling_{n}"), result.len() == n);
        println!("    N={n:>6}: {us:>6}µs ({} results)", result.len());
    }
}

fn hamon_upstream_ref(t_mean: f64, daylight_hours: f64) -> f64 {
    let d_ratio = daylight_hours / 12.0;
    let pt = 4.95 * (0.062 * t_mean).exp() / 100.0;
    (13.97 * d_ratio * d_ratio * pt).max(0.0)
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
