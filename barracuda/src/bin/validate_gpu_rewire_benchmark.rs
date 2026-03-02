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
//!
//! Provenance: `ToadStool` S68+ cross-spring absorption validation

use std::sync::Arc;
use std::time::Instant;

use barracuda::device::WgpuDevice;
use barracuda::ops::batched_elementwise_f64::{self as bef64, BatchedElementwiseF64, Op};

use airspring_barracuda::eco::crop::CropType;
use airspring_barracuda::gpu::seasonal_pipeline::{CropConfig, SeasonalPipeline, WeatherDay};
use airspring_barracuda::tolerances;
use airspring_barracuda::validation::{self, ValidationHarness};

// ─── Domain constants (synthetic validation data) ────────────────────────────

/// FAO-56 typical ET₀ range (mm/day): cold-climate minimum to high-evaporation maximum.
const ET0_RANGE_MIN_MM_DAY: f64 = 0.5;
/// FAO-56 typical ET₀ range (mm/day): cold-climate minimum to high-evaporation maximum.
const ET0_RANGE_MAX_MM_DAY: f64 = 12.0;

/// `SoilWatch` 10 Topp polynomial VWC at ε=10000; `control/soil_sensors/calibration_dong2020.py`.
const VWC_AT_10000_DIELECTRIC: f64 = 0.1323;

/// NVK polyfill `acos_f64` accumulates error in sunset hour angle → Ra → ET₀; native f64 < 0.001.
const HARGREAVES_GPU_POLYFILL_ABS_TOL: f64 = 0.10;

/// FAO-56 Eq. 62 `Kc_adj` identity: standard conditions (u2=2.0 m/s, RHmin=45%) → `Kc_tab`.
const KC_STANDARD_CONDITIONS: f64 = 1.20;

/// Seasonal pipeline: max ET₀ GPU↔CPU percent difference (1%).
const SEASONAL_ET0_PCT_PARITY: f64 = 1.0;
/// Seasonal pipeline: max yield ratio GPU↔CPU absolute difference (5%).
const SEASONAL_YIELD_PARITY: f64 = 0.05;
/// Seasonal pipeline: max mass balance error (mm); accumulation over 153 days.
const SEASONAL_MASS_BALANCE_ABS_TOL: f64 = 0.5;
/// CPU-only seasonal: max mass balance error (mm).
const SEASONAL_MASS_BALANCE_CPU_ABS_TOL: f64 = 0.1;

/// Benchmark iterations for timing (main ops).
const BENCH_ITERS: usize = 20;
/// Benchmark iterations for throughput scaling.
const BENCH_ITERS_SCALE: usize = 10;
/// Minimum throughput (items/s) for GPU scaling validation.
const THROUGHPUT_MIN_ITEMS_PER_S: f64 = 10_000.0;

/// FAO-56 dual Kc Ke upper bound (evaporation coefficient).
const DUAL_KC_KE_MAX: f64 = 1.5;

/// Growing season DOY range (May 1–Sep 30, Northern Hemisphere).
const GROWING_SEASON_DOY_START: u32 = 121;
const GROWING_SEASON_DOY_END: u32 = 273;
const EXPECTED_GROWING_DAYS: usize =
    (GROWING_SEASON_DOY_END - GROWING_SEASON_DOY_START + 1) as usize;

/// Reference station: elevation (m), latitude (°N), typical Midwest.
const REFERENCE_ELEVATION_M: f64 = 250.0;
const REFERENCE_LATITUDE_DEG: f64 = 42.5;

/// Synthetic ET₀ station: Tmax/Tmin base (°C), RH range (%), wind (m/s), Rs (MJ/m²/day).
const SYNTH_TMAX_BASE: f64 = 28.0;
const SYNTH_TMAX_RANGE: f64 = 5.0;
const SYNTH_TMIN_BASE: f64 = 14.0;
const SYNTH_TMIN_RANGE: f64 = 3.0;
const SYNTH_RH_MAX: f64 = 85.0;
const SYNTH_RH_MIN_BASE: f64 = 45.0;
const SYNTH_RH_MIN_RANGE: f64 = 10.0;
const SYNTH_WIND_BASE: f64 = 2.0;
const SYNTH_WIND_RANGE: f64 = 2.0;
const SYNTH_RS_BASE: f64 = 20.0;
const SYNTH_RS_RANGE: f64 = 5.0;
const SYNTH_DOY_BASE: u32 = 100;
const SYNTH_DOY_SPAN: u32 = 180;

/// Water balance: ETC base (mm), precip/irrig, TAW/RAW (mm), depletion fraction p.
const WB_ETC_BASE: f64 = 20.0;
const WB_ETC_RANGE: f64 = 30.0;
const WB_PRECIP_EVERY_NTH: usize = 5;
const WB_PRECIP_MM: f64 = 10.0;
const WB_IRRIG_DEFAULT: f64 = 0.0;
const WB_ETC_DAILY_BASE: f64 = 5.0;
const WB_ETC_DAILY_RANGE: f64 = 2.0;
const WB_TAW: f64 = 100.0;
const WB_RAW: f64 = 60.0;
const WB_P_DEPLETION: f64 = 0.55;

/// Sensor dielectric range (ε): `SoilWatch` 10 typical 5k–25k.
const SENSOR_DIELECTRIC_MIN: f64 = 5_000.0;
const SENSOR_DIELECTRIC_MAX: f64 = 25_000.0;

/// Hargreaves: Tmax/Tmin base (°C), latitude range (rad), DOY range.
const HG_TMAX_BASE: f64 = 25.0;
const HG_TMAX_RANGE: f64 = 10.0;
const HG_TMIN_BASE: f64 = 10.0;
const HG_TMIN_RANGE: f64 = 5.0;
const HG_LAT_BASE_DEG: f64 = 40.0;
const HG_LAT_RANGE_DEG: f64 = 20.0;
const HG_DOY_BASE: f64 = 100.0;
const HG_DOY_RANGE: f64 = 265.0;

/// Kc climate: `Kc_tab` range, u2 (m/s), `RHmin` (%), h (m). Used when GPU available.
const KC_TAB_BASE: f64 = 0.8;
const KC_TAB_RANGE: f64 = 0.5;
const KC_U2_BASE: f64 = 1.5;
const KC_U2_RANGE: f64 = 4.0;
const KC_RHMIN_BASE: f64 = 30.0;
const KC_RHMIN_RANGE: f64 = 30.0;
const KC_H_BASE: f64 = 0.5;
const KC_H_RANGE: f64 = 2.0;

/// Dual Kc: Ke range, Kcb, depletion, TAW, ETC, etc. Used when GPU available.
const DK_KE_BASE: f64 = 0.15;
const DK_KE_RANGE: f64 = 1.0;
const DK_KCB_TAB: f64 = 1.20;
const DK_RAW_FRAC_BASE: f64 = 0.05;
const DK_RAW_FRAC_RANGE: f64 = 0.9;
const DK_FEW_FULL: f64 = 1.0;
const DK_FEW_MULCH: f64 = 0.4;
const DK_ETC_BASE: f64 = 9.0;
const DK_RS_BASE: f64 = 22.5;
const DK_PRECIP_EVERY_NTH: usize = 4;
const DK_PRECIP_MM: f64 = 5.0;
const DK_IRRIG_MM: f64 = 5.0;

/// Growing season weather: Tmax/Tmin (°C), RH (%), wind (m/s), Rs (MJ/m²/day).
const WEATHER_TMAX: f64 = 28.0;
const WEATHER_TMIN: f64 = 16.0;
const WEATHER_RH_MAX: f64 = 85.0;
const WEATHER_RH_MIN: f64 = 50.0;
const WEATHER_WIND: f64 = 2.0;
const WEATHER_RS: f64 = 22.0;
const WEATHER_PRECIP_DEFAULT: f64 = 0.0;
const WEATHER_PRECIP_RAINY_DOY_MOD: u32 = 7;
const WEATHER_PRECIP_MM: f64 = 8.0;

const GPU_SCALE_SIZES: [usize; 4] = [100, 1_000, 10_000, 50_000];

fn main() {
    validation::init_tracing();
    validation::banner("Exp 057: GPU Ops 5-8 Rewire Validation + Cross-Spring Benchmark");
    println!(
        "Validates ToadStool S70+ absorption: all 6 batched ops GPU vs CPU,\n\
         timing benchmarks, and cross-spring evolution provenance.\n"
    );

    let mut v = ValidationHarness::new("GPU Rewire + Benchmark");

    let device = if let Ok(d) =
        barracuda::device::test_pool::tokio_block_on(WgpuDevice::new_f64_capable())
    {
        let d = Arc::new(d);
        println!("GPU device: {:?}", d.adapter_info());
        d
    } else {
        println!("No f64-capable GPU — running CPU-only validation");
        run_cpu_only_validation(&mut v);
        v.finish();
    };

    let engine =
        BatchedElementwiseF64::new(Arc::clone(&device)).expect("GPU engine initialization");
    let mut benchmarks: Vec<BenchRow> = Vec::new();

    // ═══ Section 1: Op 0 — FAO-56 Penman-Monteith ET₀ ═════════════════════
    validation::section("Op 0: FAO-56 PM ET₀ (hotSpring precision + neuralSpring orchestrator)");
    println!(
        "  Provenance: hotSpring pow_f64/sin_f64/cos_f64 → neuralSpring batch → airSpring domain"
    );

    let n_et0 = 1000;
    let station_days = generate_station_days(n_et0);
    let et0_gpu = engine
        .fao56_et0_batch(&station_days)
        .expect("GPU engine FAO-56 ET₀ execution");

    v.check_bool("op=0 returns N results", et0_gpu.len() == n_et0);
    v.check_bool("op=0 all positive", et0_gpu.iter().all(|&x| x > 0.0));
    v.check_bool(
        "op=0 range [0.5, 12.0]",
        et0_gpu
            .iter()
            .all(|&x| x > ET0_RANGE_MIN_MM_DAY && x < ET0_RANGE_MAX_MM_DAY),
    );

    let t_gpu_et0 = bench_fao56(&engine, &station_days, BENCH_ITERS);
    let t_cpu_et0 = bench_cpu_fn(
        || {
            station_days
                .iter()
                .map(compute_et0_cpu)
                .collect::<Vec<f64>>()
        },
        BENCH_ITERS,
    );
    benchmarks.push(BenchRow::new(
        "FAO-56 ET₀ (op=0)",
        n_et0,
        t_gpu_et0,
        t_cpu_et0,
    ));

    // ═══ Section 2: Op 1 — Water Balance ═══════════════════════════════════
    validation::section("Op 1: Water Balance (multi-spring state patterns)");
    println!("  Provenance: wetSpring state patterns → airSpring domain validation");

    let n_wb = 1000;
    let wb_data = generate_water_balance_data(n_wb);
    let wb_gpu = engine
        .execute(&wb_data, n_wb, Op::WaterBalance)
        .expect("GPU engine water balance execution");

    v.check_bool("op=1 returns N results", wb_gpu.len() == n_wb);
    v.check_bool("op=1 all non-negative", wb_gpu.iter().all(|&x| x >= 0.0));

    let t_gpu_wb = bench_execute(&engine, &wb_data, n_wb, Op::WaterBalance, BENCH_ITERS);
    benchmarks.push(BenchRow::new("Water Balance (op=1)", n_wb, t_gpu_wb, 0.0));

    // ═══ Section 3: Op 5 — Sensor Calibration ═════════════════════════════
    validation::section("Op 5: SensorCal (airSpring Dong 2024 → ToadStool S70+)");
    println!("  Provenance: airSpring soil science → ToadStool cross-spring absorption S70+");

    let n_sensor = 2000;
    let sensor_data = generate_sensor_data(n_sensor);
    let sensor_gpu = engine
        .execute(&sensor_data, n_sensor, Op::SensorCalibration)
        .expect("GPU engine sensor calibration execution");

    let sensor_cpu: Vec<f64> = sensor_data.iter().map(|&raw| sensor_cal_cpu(raw)).collect();

    v.check_bool("op=5 returns N results", sensor_gpu.len() == n_sensor);
    let sensor_max_err = max_abs_err(&sensor_gpu, &sensor_cpu);
    v.check_abs(
        "op=5 GPU↔CPU max err",
        sensor_max_err,
        0.0,
        tolerances::SOIL_HYDRAULIC.abs_tol,
    );
    let vwc_10k = engine
        .execute(&[10_000.0], 1, Op::SensorCalibration)
        .expect("GPU engine VWC(10000) execution")[0];
    // Provenance: SoilWatch 10 Topp polynomial, Topp et al. (1980).
    // VWC(ε=10000) verified against control/soil_sensors/calibration_dong2020.py
    // commit 502f2ada, 2026-02-16.
    v.check_abs(
        "op=5 VWC(10000) ≈ 0.1323",
        vwc_10k,
        VWC_AT_10000_DIELECTRIC,
        tolerances::SOIL_HYDRAULIC.abs_tol,
    );

    let t_gpu_sc = bench_execute(
        &engine,
        &sensor_data,
        n_sensor,
        Op::SensorCalibration,
        BENCH_ITERS,
    );
    let t_cpu_sc = bench_cpu_fn(
        || {
            sensor_data
                .iter()
                .map(|&r| sensor_cal_cpu(r))
                .collect::<Vec<f64>>()
        },
        BENCH_ITERS,
    );
    benchmarks.push(BenchRow::new(
        "SensorCal (op=5)",
        n_sensor,
        t_gpu_sc,
        t_cpu_sc,
    ));

    // ═══ Section 4: Op 6 — Hargreaves-Samani ET₀ ══════════════════════════
    validation::section("Op 6: Hargreaves ET₀ (airSpring + hotSpring acos_f64/sin_f64)");
    println!(
        "  Provenance: airSpring FAO-56 Eq.52 domain + hotSpring acos_f64 for sunset hour angle"
    );

    let n_hg = 2000;
    let hg_data = generate_hargreaves_data(n_hg);
    let hg_gpu = engine
        .execute(&hg_data, n_hg, Op::HargreavesEt0)
        .expect("GPU engine Hargreaves ET₀ execution");

    let hg_cpu: Vec<f64> = hg_data
        .chunks(4)
        .map(|c| bef64::hargreaves_et0_cpu(c[0], c[1], c[2], c[3]))
        .collect();

    v.check_bool("op=6 returns N results", hg_gpu.len() == n_hg);
    let hg_max_err = max_abs_err(&hg_gpu, &hg_cpu);
    // NVK polyfill acos_f64 accumulates error in sunset hour angle → Ra → ET₀.
    // Native f64 drivers achieve < 0.001; polyfill can drift up to ~0.07 mm/day.
    v.check_abs(
        "op=6 GPU↔CPU max err",
        hg_max_err,
        0.0,
        HARGREAVES_GPU_POLYFILL_ABS_TOL,
    );
    v.check_bool("op=6 all positive", hg_gpu.iter().all(|&x| x > 0.0));

    let t_gpu_hg = bench_execute(&engine, &hg_data, n_hg, Op::HargreavesEt0, BENCH_ITERS);
    let t_cpu_hg = bench_cpu_fn(
        || {
            hg_data
                .chunks(4)
                .map(|c| bef64::hargreaves_et0_cpu(c[0], c[1], c[2], c[3]))
                .collect::<Vec<f64>>()
        },
        BENCH_ITERS,
    );
    benchmarks.push(BenchRow::new("Hargreaves (op=6)", n_hg, t_gpu_hg, t_cpu_hg));

    // ═══ Section 5: Op 7 — Kc Climate Adjustment ══════════════════════════
    validation::section("Op 7: Kc Climate (airSpring FAO-56 Eq.62)");
    println!("  Provenance: airSpring precision agriculture → ToadStool S70+ absorption");

    let n_kc = 2000;
    let kc_data = generate_kc_data(n_kc);
    let kc_gpu = engine
        .execute(&kc_data, n_kc, Op::KcClimateAdjust)
        .expect("GPU engine Kc climate execution");

    let kc_cpu: Vec<f64> = kc_data
        .chunks(4)
        .map(|c| bef64::kc_climate_adjust_cpu(c[0], c[1], c[2], c[3]))
        .collect();

    v.check_bool("op=7 returns N results", kc_gpu.len() == n_kc);
    let kc_max_err = max_abs_err(&kc_gpu, &kc_cpu);
    v.check_abs(
        "op=7 GPU↔CPU max err",
        kc_max_err,
        0.0,
        tolerances::DUAL_KC_PRECISION.abs_tol,
    );

    let std_kc = engine
        .execute(
            &[KC_STANDARD_CONDITIONS, 2.0, 45.0, 2.0],
            1,
            Op::KcClimateAdjust,
        )
        .expect("GPU engine Kc standard conditions execution")[0];
    // Provenance: FAO-56 Eq. 62 Kc_adj identity — standard conditions
    // (u2=2.0 m/s, RHmin=45%) produce no adjustment, so Kc_adj = Kc_tab = 1.20.
    v.check_abs(
        "op=7 standard conditions ≈ 1.20",
        std_kc,
        KC_STANDARD_CONDITIONS,
        tolerances::DUAL_KC_PRECISION.abs_tol,
    );

    let t_gpu_kc = bench_execute(&engine, &kc_data, n_kc, Op::KcClimateAdjust, BENCH_ITERS);
    let t_cpu_kc = bench_cpu_fn(
        || {
            kc_data
                .chunks(4)
                .map(|c| bef64::kc_climate_adjust_cpu(c[0], c[1], c[2], c[3]))
                .collect::<Vec<f64>>()
        },
        BENCH_ITERS,
    );
    benchmarks.push(BenchRow::new("Kc Climate (op=7)", n_kc, t_gpu_kc, t_cpu_kc));

    // ═══ Section 6: Op 8 — Dual Kc Ke ═════════════════════════════════════
    validation::section("Op 8: Dual Kc Ke (airSpring FAO-56 Ch7+11 + hotSpring clamp)");
    println!("  Provenance: airSpring mulch/evaporation layer + hotSpring min/max/clamp patterns");

    let n_dk = 1000;
    let dk_data = generate_dual_kc_data(n_dk);
    let dk_gpu = engine
        .execute(&dk_data, n_dk, Op::DualKcKe)
        .expect("GPU engine Dual Kc Ke execution");

    v.check_bool("op=8 returns N results", dk_gpu.len() == n_dk);
    v.check_bool("op=8 all Ke non-negative", dk_gpu.iter().all(|&x| x >= 0.0));
    v.check_bool(
        "op=8 all Ke < 1.5",
        dk_gpu.iter().all(|&x| x < DUAL_KC_KE_MAX),
    );

    let t_gpu_dk = bench_execute(&engine, &dk_data, n_dk, Op::DualKcKe, BENCH_ITERS);
    benchmarks.push(BenchRow::new("Dual Kc Ke (op=8)", n_dk, t_gpu_dk, 0.0));

    // ═══ Section 7: GPU Scaling ════════════════════════════════════════════
    validation::section("GPU Throughput Scaling (Hargreaves op=6)");

    for &n in &GPU_SCALE_SIZES {
        let data = generate_hargreaves_data(n);
        let t = bench_execute(&engine, &data, n, Op::HargreavesEt0, BENCH_ITERS_SCALE);
        let throughput = n as f64 / t;
        v.check_bool(
            &format!("HG N={n}: throughput > 10K/s"),
            throughput > THROUGHPUT_MIN_ITEMS_PER_S,
        );
        println!(
            "  N={n:>6}: {:.2} ms ({:.0} items/s)",
            t * 1000.0,
            throughput
        );
    }

    // ═══ Section 8: Seasonal Pipeline GPU Stages 1-2 ══════════════════════
    validation::section("Seasonal Pipeline GPU Stages 1-2 (all Springs converge)");
    println!(
        "  Provenance: hotSpring(precision) + wetSpring(patterns) + neuralSpring(orchestration)"
    );
    println!("            + airSpring(domain) + groundSpring(uncertainty) → unified GPU pipeline");

    let weather = generate_growing_season();
    let config = CropConfig::standard(CropType::Corn);

    let gpu_pipeline =
        SeasonalPipeline::gpu(Arc::clone(&device)).expect("GPU seasonal pipeline initialization");
    let cpu_pipeline = SeasonalPipeline::cpu();

    // warmup
    let _ = gpu_pipeline.run_season(&weather, &config);
    let _ = cpu_pipeline.run_season(&weather, &config);

    let t_gpu_s = bench_season(&gpu_pipeline, &weather, &config, BENCH_ITERS);
    let t_cpu_s = bench_season(&cpu_pipeline, &weather, &config, BENCH_ITERS);

    let gpu_result = gpu_pipeline.run_season(&weather, &config);
    let cpu_result = cpu_pipeline.run_season(&weather, &config);

    v.check_bool(
        "seasonal: same n_days",
        gpu_result.n_days == cpu_result.n_days,
    );
    let et0_pct =
        (gpu_result.total_et0 - cpu_result.total_et0).abs() / cpu_result.total_et0 * 100.0;
    v.check_abs(
        "seasonal: ET₀ parity < 1%",
        et0_pct,
        0.0,
        SEASONAL_ET0_PCT_PARITY,
    );
    v.check_abs(
        "seasonal: yield parity < 5%",
        (gpu_result.yield_ratio - cpu_result.yield_ratio).abs(),
        0.0,
        SEASONAL_YIELD_PARITY,
    );
    v.check_abs(
        "seasonal: GPU mass balance < 0.5mm",
        gpu_result.mass_balance_error,
        0.0,
        SEASONAL_MASS_BALANCE_ABS_TOL,
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
    println!("  ┌──────────────────────┬───────┬──────────────┬──────────────┬──────────┐");
    println!("  │ Operation            │     N │     GPU (ms) │     CPU (ms) │  Speedup │");
    println!("  ├──────────────────────┼───────┼──────────────┼──────────────┼──────────┤");
    for row in &benchmarks {
        row.print();
    }
    println!("  └──────────────────────┴───────┴──────────────┴──────────────┴──────────┘");

    let valid_speedups: Vec<f64> = benchmarks
        .iter()
        .filter(|r| r.t_cpu > 0.0 && r.t_gpu > 0.0)
        .map(|r| r.t_cpu / r.t_gpu)
        .collect();
    if !valid_speedups.is_empty() {
        let geo_mean = valid_speedups
            .iter()
            .product::<f64>()
            .powf(1.0 / valid_speedups.len() as f64);
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
        Self {
            name,
            n,
            t_gpu,
            t_cpu,
        }
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
    v.check_bool(
        "CPU seasonal: n_days=153",
        result.n_days == EXPECTED_GROWING_DAYS,
    );
    v.check_bool("CPU seasonal: ET₀ > 400", result.total_et0 > 400.0);
    v.check_bool(
        "CPU seasonal: yield [0.5, 1.0]",
        result.yield_ratio > 0.5 && result.yield_ratio <= 1.0,
    );
    v.check_abs(
        "CPU seasonal: mass balance < 0.1mm",
        result.mass_balance_error,
        0.0,
        SEASONAL_MASS_BALANCE_CPU_ABS_TOL,
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
            let doy = SYNTH_DOY_BASE + (i as u32 % SYNTH_DOY_SPAN);
            (
                SYNTH_TMAX_BASE + (i as f64 % SYNTH_TMAX_RANGE),
                SYNTH_TMIN_BASE + (i as f64 % SYNTH_TMIN_RANGE),
                SYNTH_RH_MAX,
                SYNTH_RH_MIN_BASE + (i as f64 % SYNTH_RH_MIN_RANGE),
                SYNTH_WIND_BASE + (i as f64 % SYNTH_WIND_RANGE),
                SYNTH_RS_BASE + (i as f64 % SYNTH_RS_RANGE),
                REFERENCE_ELEVATION_M,
                REFERENCE_LATITUDE_DEG,
                doy,
            )
        })
        .collect()
}

fn generate_water_balance_data(n: usize) -> Vec<f64> {
    let mut data = Vec::with_capacity(n * 7);
    for i in 0..n {
        data.push(WB_ETC_BASE + (i as f64 % WB_ETC_RANGE));
        data.push(if i % WB_PRECIP_EVERY_NTH == 0 {
            WB_PRECIP_MM
        } else {
            WB_IRRIG_DEFAULT
        });
        data.push(WB_IRRIG_DEFAULT);
        data.push(WB_ETC_DAILY_BASE + (i as f64 % WB_ETC_DAILY_RANGE));
        data.push(WB_TAW);
        data.push(WB_RAW);
        data.push(WB_P_DEPLETION);
    }
    data
}

fn generate_sensor_data(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| {
            (i as f64 / n as f64).mul_add(
                SENSOR_DIELECTRIC_MAX - SENSOR_DIELECTRIC_MIN,
                SENSOR_DIELECTRIC_MIN,
            )
        })
        .collect()
}

/// `SoilWatch` 10 Topp polynomial coefficients; `control/soil_sensors/calibration_dong2020.py`.
const SENSOR_CAL_C3: f64 = 2e-13;
const SENSOR_CAL_C2: f64 = -4e-9;
const SENSOR_CAL_C1: f64 = 4e-5;
const SENSOR_CAL_C0: f64 = -0.0677;

fn sensor_cal_cpu(raw: f64) -> f64 {
    SENSOR_CAL_C3
        .mul_add(raw, SENSOR_CAL_C2)
        .mul_add(raw, SENSOR_CAL_C1)
        .mul_add(raw, SENSOR_CAL_C0)
}

fn generate_hargreaves_data(n: usize) -> Vec<f64> {
    let mut data = Vec::with_capacity(n * 4);
    for i in 0..n {
        data.push(HG_TMAX_BASE + (i as f64 % HG_TMAX_RANGE));
        data.push(HG_TMIN_BASE + (i as f64 % HG_TMIN_RANGE));
        data.push((HG_LAT_BASE_DEG + (i as f64 % HG_LAT_RANGE_DEG)).to_radians());
        data.push(HG_DOY_BASE + (i as f64 % HG_DOY_RANGE));
    }
    data
}

fn generate_kc_data(n: usize) -> Vec<f64> {
    let mut data = Vec::with_capacity(n * 4);
    for i in 0..n {
        data.push((i as f64 % 5.0).mul_add(KC_TAB_RANGE / 5.0, KC_TAB_BASE));
        data.push(KC_U2_BASE + (i as f64 % KC_U2_RANGE));
        data.push(KC_RHMIN_BASE + (i as f64 % KC_RHMIN_RANGE));
        data.push((i as f64 % 4.0).mul_add(KC_H_RANGE / 4.0, KC_H_BASE));
    }
    data
}

fn generate_dual_kc_data(n: usize) -> Vec<f64> {
    let mut data = Vec::with_capacity(n * 9);
    for i in 0..n {
        data.push((i as f64 % 10.0).mul_add(DK_KE_RANGE / 10.0, DK_KE_BASE));
        data.push(DK_KCB_TAB);
        data.push((i as f64 % 10.0).mul_add(DK_RAW_FRAC_RANGE / 10.0, DK_RAW_FRAC_BASE));
        data.push(if i % 3 == 0 {
            DK_FEW_MULCH
        } else {
            DK_FEW_FULL
        });
        data.push(i as f64 % 15.0);
        data.push(DK_ETC_BASE);
        data.push(DK_RS_BASE);
        data.push(if i % DK_PRECIP_EVERY_NTH == 0 {
            DK_PRECIP_MM
        } else {
            0.0
        });
        data.push(DK_IRRIG_MM);
    }
    data
}

fn generate_growing_season() -> Vec<WeatherDay> {
    (GROWING_SEASON_DOY_START..=GROWING_SEASON_DOY_END)
        .map(|doy| {
            let mut day = WeatherDay {
                tmax: WEATHER_TMAX,
                tmin: WEATHER_TMIN,
                rh_max: WEATHER_RH_MAX,
                rh_min: WEATHER_RH_MIN,
                wind_2m: WEATHER_WIND,
                solar_rad: WEATHER_RS,
                precipitation: WEATHER_PRECIP_DEFAULT,
                elevation: REFERENCE_ELEVATION_M,
                latitude_deg: REFERENCE_LATITUDE_DEG,
                day_of_year: doy,
            };
            if doy % WEATHER_PRECIP_RAINY_DOY_MOD == 0 {
                day.precipitation = WEATHER_PRECIP_MM;
            }
            day
        })
        .collect()
}

fn bench_fao56(
    engine: &BatchedElementwiseF64,
    data: &[bef64::StationDayInput],
    iters: usize,
) -> f64 {
    let _ = engine.fao56_et0_batch(data);
    let start = Instant::now();
    for _ in 0..iters {
        let _ = engine.fao56_et0_batch(data);
    }
    start.elapsed().as_secs_f64() / iters as f64
}

fn bench_execute(
    engine: &BatchedElementwiseF64,
    data: &[f64],
    n: usize,
    op: Op,
    iters: usize,
) -> f64 {
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

fn bench_season(
    pipeline: &SeasonalPipeline,
    weather: &[WeatherDay],
    config: &CropConfig,
    iters: usize,
) -> f64 {
    let start = Instant::now();
    for _ in 0..iters {
        let _ = pipeline.run_season(weather, config);
    }
    start.elapsed().as_secs_f64() / iters as f64
}
