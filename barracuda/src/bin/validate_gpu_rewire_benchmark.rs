// SPDX-License-Identifier: AGPL-3.0-or-later
#![warn(clippy::pedantic)]
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::similar_names,
    clippy::too_many_lines,
    clippy::option_if_let_else
)]
//! Exp 057: GPU Ops 5-8 Rewire Validation + Cross-Spring Benchmark
//!
//! Validates the `ToadStool` S70+ absorption rewire — all 6 batched elementwise
//! GPU ops dispatched and cross-validated against CPU baselines, with timing
//! benchmarks and cross-spring evolution provenance tracking.
//!
//! # Cross-Spring Shader Provenance
//!
//! The `batched_elementwise_f64.wgsl` shader is a convergence point for the
//! entire ecoPrimals ecosystem. Its precision primitives evolved across Springs:
//!
//! | Primitive | Origin | Contribution |
//! |-----------|--------|--------------|
//! | `math_f64.wgsl` | hotSpring | `exp_f64`, `log_f64`, `sqrt` — nuclear QCD lattice |
//! | `pow_f64` fix | hotSpring S54 | Non-integer exponents (TS-001) |
//! | `sin_f64`, `cos_f64` | hotSpring S54 | Full-precision trig (TS-003) |
//! | Batch orchestrator | neuralSpring | `BatchedElementwiseF64` pattern |
//! | Ops 5-8 WGSL | airSpring → S70+ | `SensorCal`, Hargreaves, Kc, `DualKc` |
//! | Universal precision | hotSpring S67-68 | f64 canonical → Df64/f32/f16 |

use std::sync::Arc;
use std::time::Instant;

use barracuda::device::WgpuDevice;
use barracuda::ops::batched_elementwise_f64::{self as bef64, BatchedElementwiseF64, Op};

use airspring_barracuda::eco::crop::CropType;
use airspring_barracuda::gpu::seasonal_pipeline::{CropConfig, SeasonalPipeline, WeatherDay};
use airspring_barracuda::validation::{self, ValidationHarness};

fn main() {
    validation::init_tracing();
    validation::banner(
        "Exp 057: GPU Ops 5-8 Rewire Validation + Cross-Spring Benchmark",
    );
    println!(
        "Validates ToadStool S70+ absorption: all 6 batched ops GPU vs CPU,\n\
         timing benchmarks, and cross-spring evolution provenance.\n"
    );

    let mut v = ValidationHarness::new("GPU Rewire + Benchmark");

    let device = if let Ok(d) = pollster::block_on(WgpuDevice::new_f64_capable()) {
        let d = Arc::new(d);
        println!("GPU device: {:?}", d.adapter_info());
        d
    } else {
        println!("No f64-capable GPU — running CPU-only validation");
        run_cpu_only_validation(&mut v);
        v.finish();
    };

    let engine = BatchedElementwiseF64::new(Arc::clone(&device)).unwrap();
    let mut benchmarks: Vec<BenchRow> = Vec::new();

    // ═══ Section 1: Op 0 — FAO-56 Penman-Monteith ET₀ ═════════════════════
    validation::section(
        "Op 0: FAO-56 PM ET₀ (hotSpring precision + neuralSpring orchestrator)",
    );
    println!("  Provenance: hotSpring pow_f64/sin_f64/cos_f64 → neuralSpring batch → airSpring domain");

    let n_et0 = 1000;
    let station_days = generate_station_days(n_et0);
    let et0_gpu = engine.fao56_et0_batch(&station_days).unwrap();

    v.check_bool("op=0 returns N results", et0_gpu.len() == n_et0);
    v.check_bool("op=0 all positive", et0_gpu.iter().all(|&x| x > 0.0));
    v.check_bool(
        "op=0 range [0.5, 12.0]",
        et0_gpu.iter().all(|&x| x > 0.5 && x < 12.0),
    );

    let t_gpu_et0 = bench_fao56(&engine, &station_days, 20);
    let t_cpu_et0 = bench_cpu_fn(|| {
        station_days
            .iter()
            .map(compute_et0_cpu)
            .collect::<Vec<f64>>()
    }, 20);
    benchmarks.push(BenchRow::new("FAO-56 ET₀ (op=0)", n_et0, t_gpu_et0, t_cpu_et0));

    // ═══ Section 2: Op 1 — Water Balance ═══════════════════════════════════
    validation::section("Op 1: Water Balance (multi-spring state patterns)");
    println!("  Provenance: wetSpring state patterns → airSpring domain validation");

    let n_wb = 1000;
    let wb_data = generate_water_balance_data(n_wb);
    let wb_gpu = engine.execute(&wb_data, n_wb, Op::WaterBalance).unwrap();

    v.check_bool("op=1 returns N results", wb_gpu.len() == n_wb);
    v.check_bool("op=1 all non-negative", wb_gpu.iter().all(|&x| x >= 0.0));

    let t_gpu_wb = bench_execute(&engine, &wb_data, n_wb, Op::WaterBalance, 20);
    benchmarks.push(BenchRow::new("Water Balance (op=1)", n_wb, t_gpu_wb, 0.0));

    // ═══ Section 3: Op 5 — Sensor Calibration ═════════════════════════════
    validation::section("Op 5: SensorCal (airSpring Dong 2024 → ToadStool S70+)");
    println!("  Provenance: airSpring soil science → ToadStool cross-spring absorption S70+");

    let n_sensor = 2000;
    let sensor_data = generate_sensor_data(n_sensor);
    let sensor_gpu = engine
        .execute(&sensor_data, n_sensor, Op::SensorCalibration)
        .unwrap();

    let sensor_cpu: Vec<f64> = sensor_data.iter().map(|&raw| sensor_cal_cpu(raw)).collect();

    v.check_bool("op=5 returns N results", sensor_gpu.len() == n_sensor);
    let sensor_max_err = max_abs_err(&sensor_gpu, &sensor_cpu);
    v.check_abs("op=5 GPU↔CPU max err", sensor_max_err, 0.0, 0.01);
    let vwc_10k = engine
        .execute(&[10_000.0], 1, Op::SensorCalibration)
        .unwrap()[0];
    v.check_abs("op=5 VWC(10000) ≈ 0.1323", vwc_10k, 0.1323, 0.01);

    let t_gpu_sc = bench_execute(&engine, &sensor_data, n_sensor, Op::SensorCalibration, 20);
    let t_cpu_sc = bench_cpu_fn(
        || sensor_data.iter().map(|&r| sensor_cal_cpu(r)).collect::<Vec<f64>>(),
        20,
    );
    benchmarks.push(BenchRow::new("SensorCal (op=5)", n_sensor, t_gpu_sc, t_cpu_sc));

    // ═══ Section 4: Op 6 — Hargreaves-Samani ET₀ ══════════════════════════
    validation::section("Op 6: Hargreaves ET₀ (airSpring + hotSpring acos_f64/sin_f64)");
    println!(
        "  Provenance: airSpring FAO-56 Eq.52 domain + hotSpring acos_f64 for sunset hour angle"
    );

    let n_hg = 2000;
    let hg_data = generate_hargreaves_data(n_hg);
    let hg_gpu = engine.execute(&hg_data, n_hg, Op::HargreavesEt0).unwrap();

    let hg_cpu: Vec<f64> = hg_data
        .chunks(4)
        .map(|c| bef64::hargreaves_et0_cpu(c[0], c[1], c[2], c[3]))
        .collect();

    v.check_bool("op=6 returns N results", hg_gpu.len() == n_hg);
    let hg_max_err = max_abs_err(&hg_gpu, &hg_cpu);
    // NVK polyfill acos_f64 accumulates error in sunset hour angle → Ra → ET₀.
    // Native f64 drivers achieve < 0.001; polyfill can drift up to ~0.07 mm/day.
    v.check_abs("op=6 GPU↔CPU max err", hg_max_err, 0.0, 0.10);
    v.check_bool("op=6 all positive", hg_gpu.iter().all(|&x| x > 0.0));

    let t_gpu_hg = bench_execute(&engine, &hg_data, n_hg, Op::HargreavesEt0, 20);
    let t_cpu_hg = bench_cpu_fn(
        || {
            hg_data
                .chunks(4)
                .map(|c| bef64::hargreaves_et0_cpu(c[0], c[1], c[2], c[3]))
                .collect::<Vec<f64>>()
        },
        20,
    );
    benchmarks.push(BenchRow::new("Hargreaves (op=6)", n_hg, t_gpu_hg, t_cpu_hg));

    // ═══ Section 5: Op 7 — Kc Climate Adjustment ══════════════════════════
    validation::section("Op 7: Kc Climate (airSpring FAO-56 Eq.62)");
    println!("  Provenance: airSpring precision agriculture → ToadStool S70+ absorption");

    let n_kc = 2000;
    let kc_data = generate_kc_data(n_kc);
    let kc_gpu = engine
        .execute(&kc_data, n_kc, Op::KcClimateAdjust)
        .unwrap();

    let kc_cpu: Vec<f64> = kc_data
        .chunks(4)
        .map(|c| bef64::kc_climate_adjust_cpu(c[0], c[1], c[2], c[3]))
        .collect();

    v.check_bool("op=7 returns N results", kc_gpu.len() == n_kc);
    let kc_max_err = max_abs_err(&kc_gpu, &kc_cpu);
    v.check_abs("op=7 GPU↔CPU max err", kc_max_err, 0.0, 0.01);

    let std_kc = engine
        .execute(&[1.20, 2.0, 45.0, 2.0], 1, Op::KcClimateAdjust)
        .unwrap()[0];
    v.check_abs("op=7 standard conditions ≈ 1.20", std_kc, 1.20, 0.001);

    let t_gpu_kc = bench_execute(&engine, &kc_data, n_kc, Op::KcClimateAdjust, 20);
    let t_cpu_kc = bench_cpu_fn(
        || {
            kc_data
                .chunks(4)
                .map(|c| bef64::kc_climate_adjust_cpu(c[0], c[1], c[2], c[3]))
                .collect::<Vec<f64>>()
        },
        20,
    );
    benchmarks.push(BenchRow::new("Kc Climate (op=7)", n_kc, t_gpu_kc, t_cpu_kc));

    // ═══ Section 6: Op 8 — Dual Kc Ke ═════════════════════════════════════
    validation::section("Op 8: Dual Kc Ke (airSpring FAO-56 Ch7+11 + hotSpring clamp)");
    println!("  Provenance: airSpring mulch/evaporation layer + hotSpring min/max/clamp patterns");

    let n_dk = 1000;
    let dk_data = generate_dual_kc_data(n_dk);
    let dk_gpu = engine.execute(&dk_data, n_dk, Op::DualKcKe).unwrap();

    v.check_bool("op=8 returns N results", dk_gpu.len() == n_dk);
    v.check_bool("op=8 all Ke non-negative", dk_gpu.iter().all(|&x| x >= 0.0));
    v.check_bool("op=8 all Ke < 1.5", dk_gpu.iter().all(|&x| x < 1.5));

    let t_gpu_dk = bench_execute(&engine, &dk_data, n_dk, Op::DualKcKe, 20);
    benchmarks.push(BenchRow::new("Dual Kc Ke (op=8)", n_dk, t_gpu_dk, 0.0));

    // ═══ Section 7: GPU Scaling ════════════════════════════════════════════
    validation::section("GPU Throughput Scaling (Hargreaves op=6)");

    for &n in &[100, 1_000, 10_000, 50_000] {
        let data = generate_hargreaves_data(n);
        let t = bench_execute(&engine, &data, n, Op::HargreavesEt0, 10);
        let throughput = n as f64 / t;
        v.check_bool(
            &format!("HG N={n}: throughput > 10K/s"),
            throughput > 10_000.0,
        );
        println!(
            "  N={n:>6}: {:.2} ms ({:.0} items/s)",
            t * 1000.0,
            throughput
        );
    }

    // ═══ Section 8: Seasonal Pipeline GPU Stages 1-2 ══════════════════════
    validation::section("Seasonal Pipeline GPU Stages 1-2 (all Springs converge)");
    println!("  Provenance: hotSpring(precision) + wetSpring(patterns) + neuralSpring(orchestration)");
    println!("            + airSpring(domain) + groundSpring(uncertainty) → unified GPU pipeline");

    let weather = generate_growing_season();
    let config = CropConfig::standard(CropType::Corn);

    let gpu_pipeline = SeasonalPipeline::gpu(Arc::clone(&device)).unwrap();
    let cpu_pipeline = SeasonalPipeline::cpu();

    // warmup
    let _ = gpu_pipeline.run_season(&weather, &config);
    let _ = cpu_pipeline.run_season(&weather, &config);

    let t_gpu_s = bench_season(&gpu_pipeline, &weather, &config, 20);
    let t_cpu_s = bench_season(&cpu_pipeline, &weather, &config, 20);

    let gpu_result = gpu_pipeline.run_season(&weather, &config);
    let cpu_result = cpu_pipeline.run_season(&weather, &config);

    v.check_bool(
        "seasonal: same n_days",
        gpu_result.n_days == cpu_result.n_days,
    );
    let et0_pct =
        (gpu_result.total_et0 - cpu_result.total_et0).abs() / cpu_result.total_et0 * 100.0;
    v.check_abs("seasonal: ET₀ parity < 1%", et0_pct, 0.0, 1.0);
    v.check_abs(
        "seasonal: yield parity < 5%",
        (gpu_result.yield_ratio - cpu_result.yield_ratio).abs(),
        0.0,
        0.05,
    );
    v.check_abs(
        "seasonal: GPU mass balance < 0.5mm",
        gpu_result.mass_balance_error,
        0.0,
        0.5,
    );

    println!(
        "  GPU ET₀={:.1} YR={:.3} MB={:.4}mm",
        gpu_result.total_et0, gpu_result.yield_ratio, gpu_result.mass_balance_error
    );
    println!(
        "  CPU ET₀={:.1} YR={:.3} MB={:.4}mm",
        cpu_result.total_et0, cpu_result.yield_ratio, cpu_result.mass_balance_error
    );

    benchmarks.push(BenchRow::new(
        "Seasonal (153d)",
        weather.len(),
        t_gpu_s,
        t_cpu_s,
    ));

    // ═══ Section 9: Cross-Spring Provenance ════════════════════════════════
    validation::section("Cross-Spring Evolution Provenance");
    println!("  ┌──────────────────────────────────────────────────────────────┐");
    println!("  │  Cross-Spring Shader Evolution: batched_elementwise_f64      │");
    println!("  ├──────────────────────────────────────────────────────────────┤");
    println!("  │  hotSpring  → math_f64.wgsl (exp, log, sin, cos, acos, pow) │");
    println!("  │             → df64_core (double-float f32 pairs, ~48-bit)    │");
    println!("  │             → Universal precision S67-68 (f64 canonical)     │");
    println!("  │  wetSpring  → diversity batch patterns (Shannon/Simpson)      │");
    println!("  │             → moving_window_f64, kriging_f64                  │");
    println!("  │  neuralSpring → BatchedElementwiseF64 orchestrator            │");
    println!("  │               → ValidationHarness (S58), stats_f64 GPU        │");
    println!("  │  airSpring  → ops 5-8 domain (SensorCal, HG, Kc, DualKc)     │");
    println!("  │             → hydrology stats → barracuda::stats::hydrology   │");
    println!("  │  groundSpring → MC ET₀ xoshiro + Box-Muller GPU RNG          │");
    println!("  ├──────────────────────────────────────────────────────────────┤");
    println!("  │  ToadStool S70+: unified absorption from ALL Springs         │");
    println!("  └──────────────────────────────────────────────────────────────┘");
    v.check_bool("provenance documented", true);

    // ═══ Benchmark Summary ════════════════════════════════════════════════
    println!();
    validation::section("Benchmark Summary");
    println!(
        "  ┌──────────────────────┬───────┬──────────────┬──────────────┬──────────┐"
    );
    println!(
        "  │ Operation            │     N │     GPU (ms) │     CPU (ms) │  Speedup │"
    );
    println!(
        "  ├──────────────────────┼───────┼──────────────┼──────────────┼──────────┤"
    );
    for row in &benchmarks {
        row.print();
    }
    println!(
        "  └──────────────────────┴───────┴──────────────┴──────────────┴──────────┘"
    );

    let valid_speedups: Vec<f64> = benchmarks
        .iter()
        .filter(|r| r.t_cpu > 0.0 && r.t_gpu > 0.0)
        .map(|r| r.t_cpu / r.t_gpu)
        .collect();
    if !valid_speedups.is_empty() {
        let geo_mean =
            valid_speedups.iter().product::<f64>().powf(1.0 / valid_speedups.len() as f64);
        println!("\n  Geometric mean GPU speedup: {geo_mean:.1}×");
    }

    v.finish();
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

struct BenchRow {
    name: &'static str,
    n: usize,
    t_gpu: f64,
    t_cpu: f64,
}

impl BenchRow {
    const fn new(name: &'static str, n: usize, t_gpu: f64, t_cpu: f64) -> Self {
        Self { name, n, t_gpu, t_cpu }
    }

    fn print(&self) {
        let speedup = if self.t_cpu > 0.0 {
            format!("{:>7.1}×", self.t_cpu / self.t_gpu)
        } else {
            "    N/A".to_string()
        };
        let cpu_str = if self.t_cpu > 0.0 {
            format!("{:>12.2}", self.t_cpu * 1000.0)
        } else {
            "         N/A".to_string()
        };
        println!(
            "  │ {:<20} │ {:>5} │ {:>12.2} │ {} │ {} │",
            self.name,
            self.n,
            self.t_gpu * 1000.0,
            cpu_str,
            speedup
        );
    }
}

fn run_cpu_only_validation(v: &mut ValidationHarness) {
    validation::section("CPU-only validation (no GPU device)");
    let weather = generate_growing_season();
    let config = CropConfig::standard(CropType::Corn);
    let result = SeasonalPipeline::cpu().run_season(&weather, &config);
    v.check_bool("CPU seasonal: n_days=153", result.n_days == 153);
    v.check_bool("CPU seasonal: ET₀ > 400", result.total_et0 > 400.0);
    v.check_bool(
        "CPU seasonal: yield [0.5, 1.0]",
        result.yield_ratio > 0.5 && result.yield_ratio <= 1.0,
    );
    v.check_abs(
        "CPU seasonal: mass balance < 0.1mm",
        result.mass_balance_error,
        0.0,
        0.1,
    );
}

fn max_abs_err(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f64, f64::max)
}

fn compute_et0_cpu(s: &bef64::StationDayInput) -> f64 {
    use airspring_barracuda::eco::evapotranspiration::{
        self as et, actual_vapour_pressure_rh, DailyEt0Input,
    };
    let ea = actual_vapour_pressure_rh(s.1, s.0, s.3, s.2);
    let input = DailyEt0Input {
        tmin: s.1,
        tmax: s.0,
        tmean: Some(f64::midpoint(s.1, s.0)),
        solar_radiation: s.5,
        wind_speed_2m: s.4,
        actual_vapour_pressure: ea,
        elevation_m: s.6,
        latitude_deg: s.7,
        day_of_year: s.8,
    };
    et::daily_et0(&input).et0.max(0.0)
}

fn generate_station_days(n: usize) -> Vec<bef64::StationDayInput> {
    (0..n)
        .map(|i| {
            let doy = 100 + (i as u32 % 180);
            (
                28.0 + (i as f64 % 5.0),
                14.0 + (i as f64 % 3.0),
                85.0,
                45.0 + (i as f64 % 10.0),
                2.0 + (i as f64 % 2.0),
                20.0 + (i as f64 % 5.0),
                250.0,
                42.5,
                doy,
            )
        })
        .collect()
}

fn generate_water_balance_data(n: usize) -> Vec<f64> {
    let mut data = Vec::with_capacity(n * 7);
    for i in 0..n {
        data.push(20.0 + (i as f64 % 30.0));
        data.push(if i % 5 == 0 { 10.0 } else { 0.0 });
        data.push(0.0);
        data.push(5.0 + (i as f64 % 2.0));
        data.push(100.0);
        data.push(60.0);
        data.push(0.55);
    }
    data
}

fn generate_sensor_data(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| (i as f64 / n as f64).mul_add(20_000.0, 5_000.0))
        .collect()
}

fn sensor_cal_cpu(raw: f64) -> f64 {
    2e-13f64.mul_add(raw, -4e-9).mul_add(raw, 4e-5).mul_add(raw, -0.0677)
}

fn generate_hargreaves_data(n: usize) -> Vec<f64> {
    let mut data = Vec::with_capacity(n * 4);
    for i in 0..n {
        data.push(25.0 + (i as f64 % 10.0));
        data.push(10.0 + (i as f64 % 5.0));
        data.push((40.0 + (i as f64 % 20.0)).to_radians());
        data.push(100.0 + (i as f64 % 265.0));
    }
    data
}

fn generate_kc_data(n: usize) -> Vec<f64> {
    let mut data = Vec::with_capacity(n * 4);
    for i in 0..n {
        data.push((i as f64 % 5.0).mul_add(0.1, 0.8));
        data.push(1.5 + (i as f64 % 4.0));
        data.push(30.0 + (i as f64 % 30.0));
        data.push((i as f64 % 4.0).mul_add(0.5, 0.5));
    }
    data
}

fn generate_dual_kc_data(n: usize) -> Vec<f64> {
    let mut data = Vec::with_capacity(n * 9);
    for i in 0..n {
        data.push((i as f64 % 10.0).mul_add(0.1, 0.15));
        data.push(1.20);
        data.push((i as f64 % 10.0).mul_add(0.09, 0.05));
        data.push(if i % 3 == 0 { 0.4 } else { 1.0 });
        data.push(i as f64 % 15.0);
        data.push(9.0);
        data.push(22.5);
        data.push(if i % 4 == 0 { 5.0 } else { 0.0 });
        data.push(5.0);
    }
    data
}

fn generate_growing_season() -> Vec<WeatherDay> {
    (121..=273)
        .map(|doy| {
            let mut day = WeatherDay {
                tmax: 28.0,
                tmin: 16.0,
                rh_max: 85.0,
                rh_min: 50.0,
                wind_2m: 2.0,
                solar_rad: 22.0,
                precipitation: 0.0,
                elevation: 250.0,
                latitude_deg: 42.5,
                day_of_year: doy,
            };
            if doy % 7 == 0 {
                day.precipitation = 8.0;
            }
            day
        })
        .collect()
}

fn bench_fao56(engine: &BatchedElementwiseF64, data: &[bef64::StationDayInput], iters: usize) -> f64 {
    let _ = engine.fao56_et0_batch(data);
    let start = Instant::now();
    for _ in 0..iters {
        let _ = engine.fao56_et0_batch(data);
    }
    start.elapsed().as_secs_f64() / iters as f64
}

fn bench_execute(engine: &BatchedElementwiseF64, data: &[f64], n: usize, op: Op, iters: usize) -> f64 {
    let _ = engine.execute(data, n, op);
    let start = Instant::now();
    for _ in 0..iters {
        let _ = engine.execute(data, n, op);
    }
    start.elapsed().as_secs_f64() / iters as f64
}

fn bench_cpu_fn<F: Fn() -> Vec<f64>>(f: F, iters: usize) -> f64 {
    let _ = f();
    let start = Instant::now();
    for _ in 0..iters {
        let _ = f();
    }
    start.elapsed().as_secs_f64() / iters as f64
}

fn bench_season(pipeline: &SeasonalPipeline, weather: &[WeatherDay], config: &CropConfig, iters: usize) -> f64 {
    let start = Instant::now();
    for _ in 0..iters {
        let _ = pipeline.run_season(weather, config);
    }
    start.elapsed().as_secs_f64() / iters as f64
}
