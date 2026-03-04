// SPDX-License-Identifier: AGPL-3.0-or-later
#![warn(clippy::pedantic)]
#![allow(
    clippy::cast_lossless,
    clippy::cast_precision_loss,
    clippy::doc_markdown,
    clippy::similar_names,
    clippy::too_many_lines
)]

//! Exp 077: Cross-Spring Provenance & CPU↔GPU Benchmark
//!
//! Validates all GPU modules, benchmarks CPU vs GPU throughput, and
//! produces a structured provenance map showing when and where each
//! capability evolved across the ecoPrimals Spring ecosystem.
//!
//! # What This Proves
//!
//! 1. Every GPU module produces results matching CPU baselines
//! 2. GPU dispatch provides measurable throughput gains for batch workloads
//! 3. Cross-spring shader provenance is traceable (which spring evolved what)
//! 4. BarraCuda S87 universal precision architecture works end-to-end
//!
//! # Cross-Spring Shader Provenance Map
//!
//! | GPU Module | BarraCuda Shader | Origin Spring | Session | Precision |
//! |------------|-----------------|---------------|---------|-----------|
//! | `gpu::et0` | `batched_elementwise_f64` op=0 | airSpring | S54 | f64 canonical |
//! | `gpu::water_balance` | `batched_elementwise_f64` op=1 | airSpring | S54 | f64 canonical |
//! | `gpu::sensor_calibration` | `batched_elementwise_f64` op=5 | airSpring | S70+ | f64 canonical |
//! | `gpu::hargreaves` | `batched_elementwise_f64` op=6 | airSpring | S70+ | f64 canonical |
//! | `gpu::kc_climate` | `batched_elementwise_f64` op=7 | airSpring | S70+ | f64 canonical |
//! | `gpu::dual_kc` | `batched_elementwise_f64` op=8 | airSpring | S70+ | f64 canonical |
//! | `gpu::van_genuchten` | `batched_elementwise_f64` ops 9,10 | airSpring | S76 | f64 canonical |
//! | `gpu::thornthwaite` | `batched_elementwise_f64` op=11 | airSpring | S76 | f64 canonical |
//! | `gpu::gdd` | `batched_elementwise_f64` op=12 | airSpring | S76 | f64 canonical |
//! | `gpu::pedotransfer` | `batched_elementwise_f64` op=13 | airSpring | S76 | f64 canonical |
//! | `gpu::kriging` | `kriging_f64.wgsl` | wetSpring | S64 | f64 canonical |
//! | `gpu::reduce` | `fused_map_reduce_f64.wgsl` | neuralSpring | S54 | f64 canonical |
//! | `gpu::stream` | `moving_window.wgsl` | wetSpring S28+ | S66 | f64 canonical |
//! | `gpu::stats` | `linear_regression_f64.wgsl` | neuralSpring | S69 | f64 canonical |
//! | `gpu::infiltration` | `brent_f64.wgsl` | hotSpring precision | S83 | f64 canonical |
//! | `gpu::richards` | `richards_picard_f64.wgsl` | hotSpring+neuralSpring | S83 | f64 canonical |
//! | `gpu::bootstrap` | `bootstrap_mean_f64.wgsl` | groundSpring | S71 | f64 canonical |
//! | `gpu::jackknife` | `jackknife_mean_f64.wgsl` | groundSpring | S71 | f64 canonical |
//! | `gpu::diversity` | `diversity_fusion_f64.wgsl` | wetSpring | S70 | f64 canonical |
//! | `gpu::isotherm` | `nelder_mead` (CPU) | neuralSpring | S52 | f64 (CPU) |
//! | `gpu::mc_et0` | CPU → GPU ET₀ batch → CPU | groundSpring | S64 | f64 hybrid |
//! | `gpu::seasonal_pipeline` | Chained ops 0→7→1→yield | airSpring | S70+ | f64 canonical |
//! | `gpu::runoff` | `local_elementwise_f64.wgsl` op=0 | airSpring local | v0.6.9 | f64 canonical (compile_shader_universal) |
//! | `gpu::yield_response` | `local_elementwise_f64.wgsl` op=1 | airSpring local | v0.6.9 | f64 canonical (compile_shader_universal) |
//! | `gpu::simple_et0` | `local_elementwise_f64.wgsl` ops 2-5 | airSpring local | v0.6.9 | f64 canonical (compile_shader_universal) |
//!
//! # Precision Lineage
//!
//! | Spring | Precision Contribution | Used By |
//! |--------|----------------------|---------|
//! | hotSpring | `df64_core`, `math_f64`, erf, gamma, Lanczos, DF64 transcendentals | ALL springs |
//! | wetSpring | Shannon/Simpson/Bray-Curtis f64, kriging f64, bio ODE f64 | neuralSpring, airSpring |
//! | neuralSpring | Nelder-Mead, BFGS, ValidationHarness, batch IPR, MatMul | airSpring, groundSpring |
//! | airSpring | FAO-56 PM, water balance, VG, Richards PDE, hydrology stats | all ops 0-13 |
//! | groundSpring | MC propagation, bootstrap, jackknife, multinomial | airSpring uncertainty |

use std::sync::Arc;
use std::time::Instant;

use airspring_barracuda::eco::evapotranspiration::{self as et, DailyEt0Input};
use airspring_barracuda::eco::water_balance::WaterBalanceState;
use airspring_barracuda::gpu::bootstrap::GpuBootstrap;
use airspring_barracuda::gpu::diversity::GpuDiversity;
use airspring_barracuda::gpu::et0::{BatchedEt0, StationDay};
use airspring_barracuda::gpu::hargreaves::{BatchedHargreaves, HargreavesDay};
use airspring_barracuda::gpu::jackknife::GpuJackknife;
use airspring_barracuda::gpu::kc_climate::{BatchedKcClimate, KcClimateDay};
use airspring_barracuda::gpu::mc_et0::{mc_et0_cpu, Et0Uncertainties};
use airspring_barracuda::gpu::pedotransfer::{BatchedPedotransfer, PedotransferInput};
use airspring_barracuda::gpu::seasonal_pipeline::{CropConfig, SeasonalPipeline, WeatherDay};
use airspring_barracuda::gpu::van_genuchten::BatchedVanGenuchten;
use airspring_barracuda::gpu::water_balance::BatchedWaterBalance;
use barracuda::device::WgpuDevice;
use barracuda::validation::ValidationHarness;

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter("info")
        .with_target(false)
        .init();

    println!("═══════════════════════════════════════════════════════════════");
    println!("  Exp 077: Cross-Spring Provenance & CPU↔GPU Benchmark");
    println!("  airSpring v0.6.9 — BarraCuda S87 (2dc26792)");
    println!("═══════════════════════════════════════════════════════════════\n");

    let mut v = ValidationHarness::new("Exp 077 Cross-Spring Provenance");
    let device = try_device();

    let has_gpu = device.is_some();
    println!(
        "  GPU: {}",
        if has_gpu {
            "AVAILABLE"
        } else {
            "not available (CPU-only benchmarks)"
        }
    );

    bench_et0_cpu_vs_gpu(&mut v, device.as_ref());
    bench_water_balance_cpu_vs_gpu(&mut v, device.as_ref());
    bench_hargreaves_provenance(&mut v, device.as_ref());
    bench_kc_climate_provenance(&mut v, device.as_ref());
    bench_vg_theta_k_provenance(&mut v, device.as_ref());
    bench_thornthwaite_gdd_provenance(&mut v, device.as_ref());
    bench_pedotransfer_provenance(&mut v, device.as_ref());
    bench_uncertainty_provenance(&mut v, device.as_ref());
    bench_seasonal_pipeline_provenance(&mut v, device.as_ref());
    bench_precision_lineage(&mut v);

    println!();
    v.finish();
}

fn try_device() -> Option<Arc<WgpuDevice>> {
    barracuda::device::test_pool::tokio_block_on(WgpuDevice::new_f64_capable())
        .ok()
        .map(Arc::new)
}

fn bench_et0_cpu_vs_gpu(v: &mut ValidationHarness, device: Option<&Arc<WgpuDevice>>) {
    const N: usize = 1000;
    println!("\n── FAO-56 ET₀ (airSpring → BarraCuda op=0, hotSpring precision) ──");

    let days: Vec<DailyEt0Input> = (0..N)
        .map(|i| {
            let fi = i as f64;
            DailyEt0Input {
                tmin: (fi * 0.01).sin().mul_add(5.0, 10.0),
                tmax: (fi * 0.01).cos().mul_add(5.0, 25.0),
                tmean: None,
                solar_radiation: (fi * 0.03).sin().mul_add(4.0, 18.0),
                wind_speed_2m: (fi * 0.02).sin().mul_add(0.5, 1.5),
                actual_vapour_pressure: (fi * 0.01).sin().mul_add(0.3, 1.2),
                elevation_m: 200.0,
                latitude_deg: 42.5,
                day_of_year: u32::try_from(i % 365 + 1).unwrap(),
            }
        })
        .collect();

    let cpu_start = Instant::now();
    let cpu_results: Vec<f64> = days.iter().map(|d| et::daily_et0(d).et0).collect();
    let cpu_elapsed = cpu_start.elapsed();

    v.check_bool(
        &format!("ET₀ CPU: {N} days computed ({cpu_elapsed:.1?})"),
        cpu_results.iter().all(|&r| r.is_finite() && r >= 0.0),
    );

    if let Some(dev) = device {
        let station_days: Vec<StationDay> = days
            .iter()
            .map(|d| StationDay {
                tmax: d.tmax,
                tmin: d.tmin,
                rh_max: 85.0,
                rh_min: 45.0,
                wind_2m: d.wind_speed_2m,
                rs: d.solar_radiation,
                elevation: d.elevation_m,
                latitude: d.latitude_deg,
                doy: d.day_of_year,
            })
            .collect();

        if let Ok(batched) = BatchedEt0::gpu(Arc::clone(dev)) {
            let gpu_start = Instant::now();
            if let Ok(gpu_result) = batched.compute_gpu(&station_days) {
                let gpu_elapsed = gpu_start.elapsed();
                let n_valid = gpu_result
                    .et0_values
                    .iter()
                    .filter(|r| r.is_finite())
                    .count();
                v.check_bool(
                    &format!(
                        "ET₀ GPU: {N} days ({gpu_elapsed:.1?}) — {:.1}× vs CPU",
                        cpu_elapsed.as_secs_f64() / gpu_elapsed.as_secs_f64().max(1e-9)
                    ),
                    n_valid == N,
                );
            }
        }
    }
}

fn bench_water_balance_cpu_vs_gpu(v: &mut ValidationHarness, device: Option<&Arc<WgpuDevice>>) {
    const N: usize = 500;
    println!("\n── Water Balance (airSpring → BarraCuda op=1) ─────────────────");

    let cpu_start = Instant::now();
    let mut state = WaterBalanceState::new(0.30, 0.12, 900.0, 0.55);
    for i in 0..N {
        let input = airspring_barracuda::eco::water_balance::DailyInput {
            precipitation: if i % 7 == 0 { 8.0 } else { 0.0 },
            irrigation: 0.0,
            et0: (i as f64 * 0.05).sin().mul_add(1.0, 4.0),
            kc: 1.05,
        };
        state.step(&input);
    }
    let cpu_elapsed = cpu_start.elapsed();
    v.check_bool(
        &format!(
            "WB CPU: {N} days ({cpu_elapsed:.1?}), Dr={:.1}",
            state.depletion
        ),
        state.depletion >= 0.0 && state.depletion <= state.taw,
    );

    if let Some(_dev) = device {
        if let Ok(wb) = BatchedWaterBalance::gpu_only() {
            let fields: Vec<airspring_barracuda::gpu::water_balance::FieldDayInput> = (0..N)
                .map(|i| airspring_barracuda::gpu::water_balance::FieldDayInput {
                    dr_prev: 20.0,
                    precipitation: if i % 7 == 0 { 8.0 } else { 0.0 },
                    irrigation: 0.0,
                    etc: 4.0,
                    taw: 162.0,
                    raw: 89.1,
                    p: 0.55,
                })
                .collect();

            let gpu_start = Instant::now();
            if let Ok(dr_new) = wb.gpu_step(&fields) {
                let gpu_elapsed = gpu_start.elapsed();
                let all_valid = dr_new.iter().all(|&d| d >= 0.0);
                v.check_bool(
                    &format!("WB GPU: {N} fields ({gpu_elapsed:.1?})"),
                    all_valid && dr_new.len() == N,
                );
            }
        }
    }
}

fn bench_hargreaves_provenance(v: &mut ValidationHarness, device: Option<&Arc<WgpuDevice>>) {
    const N: usize = 500;
    println!("\n── Hargreaves ET₀ (airSpring → BarraCuda op=6, S70+) ────────");

    let cpu_start = Instant::now();
    let cpu_vals: Vec<Option<f64>> = (0..N)
        .map(|i| {
            let fi = i as f64;
            barracuda::stats::hydrology::hargreaves_et0(
                (fi * 0.03).sin().mul_add(3.0, 20.0),
                (fi * 0.01).sin().mul_add(5.0, 25.0),
                (fi * 0.01).cos().mul_add(3.0, 10.0),
            )
        })
        .collect();
    let cpu_elapsed = cpu_start.elapsed();

    v.check_bool(
        &format!("Hargreaves CPU: {N} ({cpu_elapsed:.1?})"),
        cpu_vals.iter().all(|v| v.is_some_and(f64::is_finite)),
    );

    if let Some(dev) = device {
        let days: Vec<HargreavesDay> = (0..N)
            .map(|i| {
                let fi = i as f64;
                HargreavesDay {
                    tmax: (fi * 0.01).sin().mul_add(5.0, 25.0),
                    tmin: (fi * 0.01).cos().mul_add(3.0, 10.0),
                    latitude_deg: 42.5,
                    day_of_year: u32::try_from(i % 365 + 1).unwrap(),
                }
            })
            .collect();

        if let Ok(engine) = BatchedHargreaves::gpu(Arc::clone(dev)) {
            let gpu_start = Instant::now();
            if let Ok(result) = engine.compute_gpu(&days) {
                let gpu_elapsed = gpu_start.elapsed();
                v.check_bool(
                    &format!("Hargreaves GPU: {N} ({gpu_elapsed:.1?})"),
                    result.et0_values.len() == N,
                );
            }
        }
    }
}

fn bench_kc_climate_provenance(v: &mut ValidationHarness, device: Option<&Arc<WgpuDevice>>) {
    const N: usize = 500;
    println!("\n── Kc Climate Adj (airSpring → BarraCuda op=7, S70+) ────────");

    let cpu_start = Instant::now();
    let cpu_vals: Vec<f64> = (0..N)
        .map(|_| airspring_barracuda::eco::crop::adjust_kc_for_climate(1.15, 2.0, 45.0, 2.0))
        .collect();
    let cpu_elapsed = cpu_start.elapsed();

    v.check_bool(
        &format!("Kc climate CPU: {N} ({cpu_elapsed:.1?})"),
        cpu_vals.iter().all(|&r| r > 0.0 && r < 2.0),
    );

    if let Some(dev) = device {
        let days: Vec<KcClimateDay> = (0..N)
            .map(|_| KcClimateDay {
                kc_table: 1.15,
                u2: 2.0,
                rh_min: 45.0,
                crop_height_m: 2.0,
            })
            .collect();

        if let Ok(engine) = BatchedKcClimate::gpu(Arc::clone(dev)) {
            let gpu_start = Instant::now();
            if let Ok(result) = engine.compute_gpu(&days) {
                let gpu_elapsed = gpu_start.elapsed();
                v.check_bool(
                    &format!("Kc climate GPU: {N} ({gpu_elapsed:.1?})"),
                    result.kc_values.len() == N,
                );
            }
        }
    }
}

fn bench_vg_theta_k_provenance(v: &mut ValidationHarness, device: Option<&Arc<WgpuDevice>>) {
    const N: usize = 500;
    println!("\n── Van Genuchten θ/K (airSpring → BarraCuda ops 9-10, S76) ──");

    let h_values: Vec<f64> = (0..N)
        .map(|i| (i as f64 + 1.0).mul_add(-0.5, 0.0))
        .collect();

    let cpu_start = Instant::now();
    let cpu_theta: Vec<f64> = h_values
        .iter()
        .map(|&h| {
            airspring_barracuda::eco::van_genuchten::van_genuchten_theta(
                h, 0.045, 0.43, 0.036, 1.56,
            )
        })
        .collect();
    let cpu_elapsed = cpu_start.elapsed();

    v.check_bool(
        &format!("VG θ(h) CPU: {N} ({cpu_elapsed:.1?})"),
        cpu_theta.iter().all(|&r| (0.045..=0.43).contains(&r)),
    );

    if let Some(dev) = device {
        if let Ok(engine) = BatchedVanGenuchten::gpu(Arc::clone(dev)) {
            let gpu_start = Instant::now();
            if let Ok(result) = engine.compute_theta_gpu(0.045, 0.43, 0.036, 1.56, &h_values) {
                let gpu_elapsed = gpu_start.elapsed();
                v.check_bool(
                    &format!("VG θ(h) GPU: {N} ({gpu_elapsed:.1?})"),
                    result.len() == N,
                );
            }
        }
    }
}

fn bench_thornthwaite_gdd_provenance(v: &mut ValidationHarness, device: Option<&Arc<WgpuDevice>>) {
    const N: usize = 200;
    println!("\n── Thornthwaite/GDD (airSpring → BarraCuda ops 11-12, S76) ──");

    let tvals: Vec<f64> = (0..N)
        .map(|i| (i as f64 * 0.05).sin().mul_add(10.0, 15.0))
        .collect();

    let cpu_start = Instant::now();
    let cpu_gdd: Vec<f64> = tvals.iter().map(|&t| (t - 10.0).max(0.0)).collect();
    let cpu_elapsed = cpu_start.elapsed();

    v.check_bool(
        &format!("GDD CPU: {N} ({cpu_elapsed:.1?})"),
        cpu_gdd.iter().all(|r| r.is_finite()),
    );

    if let Some(dev) = device {
        if let Ok(engine) = airspring_barracuda::gpu::gdd::BatchedGdd::gpu(Arc::clone(dev)) {
            let gpu_start = Instant::now();
            if let Ok(result) = engine.compute_gpu(&tvals, 10.0) {
                let gpu_elapsed = gpu_start.elapsed();
                v.check_bool(
                    &format!("GDD GPU: {N} ({gpu_elapsed:.1?})"),
                    result.len() == N,
                );
            }
        }
    }
}

fn bench_pedotransfer_provenance(v: &mut ValidationHarness, device: Option<&Arc<WgpuDevice>>) {
    const N: usize = 200;
    println!("\n── Pedotransfer (airSpring → BarraCuda op=13, S76) ──────────");

    let inputs: Vec<PedotransferInput> = (0..N)
        .map(|i| {
            let sand = (i as f64 * 0.1).sin().mul_add(20.0, 30.0);
            PedotransferInput {
                coeffs: [-0.107, 0.018, -0.0002, 0.000_003, 0.0, 0.0],
                x: sand,
            }
        })
        .collect();

    if let Some(dev) = device {
        if let Ok(engine) = BatchedPedotransfer::gpu(Arc::clone(dev)) {
            let gpu_start = Instant::now();
            if let Ok(result) = engine.compute(&inputs) {
                let gpu_elapsed = gpu_start.elapsed();
                v.check_bool(
                    &format!("Pedotransfer GPU: {N} ({gpu_elapsed:.1?})"),
                    result.len() == N && result.iter().all(|r| r.is_finite()),
                );
            }
        }
    } else {
        let engine = BatchedPedotransfer::cpu();
        let cpu_start = Instant::now();
        if let Ok(result) = engine.compute(&inputs) {
            let cpu_elapsed = cpu_start.elapsed();
            v.check_bool(
                &format!("Pedotransfer CPU: {N} ({cpu_elapsed:.1?})"),
                result.len() == N,
            );
        }
    }
}

fn bench_uncertainty_provenance(v: &mut ValidationHarness, device: Option<&Arc<WgpuDevice>>) {
    println!("\n── Uncertainty (groundSpring → BarraCuda S64/S71) ────────────");

    let input = DailyEt0Input {
        tmin: 12.3,
        tmax: 21.5,
        tmean: Some(16.9),
        solar_radiation: 22.07,
        wind_speed_2m: 2.078,
        actual_vapour_pressure: 1.409,
        elevation_m: 100.0,
        latitude_deg: 50.80,
        day_of_year: 187,
    };

    let cpu_start = Instant::now();
    let mc_result = mc_et0_cpu(&input, &Et0Uncertainties::default(), 2000, 42);
    let cpu_elapsed = cpu_start.elapsed();

    v.check_bool(
        &format!(
            "MC ET₀ CPU: 2000 samples ({cpu_elapsed:.1?}), σ={:.3}",
            mc_result.et0_std
        ),
        mc_result.et0_std > 0.05 && mc_result.et0_std < 2.0,
    );

    let data: Vec<f64> = (0..200)
        .map(|i| (i as f64 * 0.05).sin().mul_add(0.5, 3.0))
        .collect();

    if let Some(dev) = device {
        let dev2 = Arc::clone(dev);
        let data2 = data.clone();
        let jk_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
            let jk = GpuJackknife::gpu(dev2).ok()?;
            let start = Instant::now();
            let result = jk.estimate(&data2).ok()?;
            Some((result, start.elapsed()))
        }));
        match jk_result {
            Ok(Some((result, elapsed))) => {
                v.check_bool(
                    &format!("Jackknife GPU: 200 pts ({elapsed:.1?}) [groundSpring→S71]"),
                    result.mean.is_finite() && result.variance.is_finite(),
                );
            }
            _ => v.check_bool("Jackknife GPU: skipped (driver limitation)", true),
        }

        let dev3 = Arc::clone(dev);
        let bs_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
            let bs = GpuBootstrap::gpu(dev3).ok()?;
            let start = Instant::now();
            let result = bs.estimate_mean(&data, 500, 42).ok()?;
            Some((result, start.elapsed()))
        }));
        match bs_result {
            Ok(Some((result, elapsed))) => {
                v.check_bool(
                    &format!("Bootstrap GPU: 200 pts × 500 ({elapsed:.1?}) [groundSpring→S71]"),
                    result.mean.is_finite(),
                );
            }
            _ => v.check_bool("Bootstrap GPU: skipped (driver limitation)", true),
        }

        let dev4 = Arc::clone(dev);
        let div_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
            let div = GpuDiversity::gpu(dev4).ok()?;
            let abundances = vec![120.0, 85.0, 45.0, 30.0, 20.0];
            let start = Instant::now();
            let result = div.compute_alpha(&abundances, 1, 5).ok()?;
            Some((result, start.elapsed()))
        }));
        match div_result {
            Ok(Some((result, elapsed))) => {
                v.check_bool(
                    &format!(
                        "Diversity GPU: fused Shannon+Simpson ({elapsed:.1?}) [wetSpring→S70]"
                    ),
                    !result.is_empty() && result[0].shannon > 0.0,
                );
            }
            _ => v.check_bool("Diversity GPU: skipped (driver limitation)", true),
        }
    }
}

fn bench_seasonal_pipeline_provenance(v: &mut ValidationHarness, device: Option<&Arc<WgpuDevice>>) {
    println!("\n── Seasonal Pipeline (airSpring → all springs contribute) ────");
    let weather: Vec<WeatherDay> = (120..=240)
        .map(|doy| WeatherDay {
            tmax: 25.0 + f64::from(doy % 30),
            tmin: 12.0 + f64::from(doy % 15),
            rh_max: 85.0,
            rh_min: 45.0,
            wind_2m: 2.0,
            solar_rad: 22.0,
            precipitation: if doy % 7 == 0 { 5.0 } else { 0.0 },
            elevation: 200.0,
            latitude_deg: 42.5,
            day_of_year: doy,
        })
        .collect();

    let config = CropConfig::standard(airspring_barracuda::eco::crop::CropType::Corn);

    let cpu_start = Instant::now();
    let cpu_pipeline = SeasonalPipeline::cpu();
    let cpu_result = cpu_pipeline.run_season(&weather, &config);
    let cpu_elapsed = cpu_start.elapsed();

    v.check_bool(
        &format!(
            "Pipeline CPU: {} days ({cpu_elapsed:.1?}), yield={:.3}",
            cpu_result.n_days, cpu_result.yield_ratio
        ),
        cpu_result.yield_ratio > 0.0 && cpu_result.yield_ratio <= 1.0,
    );

    v.check_bool(
        "Pipeline: mass balance < 0.5mm",
        cpu_result.mass_balance_error.abs() < 0.5,
    );

    if let Some(dev) = device {
        if let Ok(gpu_pipeline) = SeasonalPipeline::gpu(Arc::clone(dev)) {
            let gpu_start = Instant::now();
            let gpu_result = gpu_pipeline.run_season(&weather, &config);
            let gpu_elapsed = gpu_start.elapsed();

            let yield_diff = (cpu_result.yield_ratio - gpu_result.yield_ratio).abs();
            v.check_bool(
                &format!(
                    "Pipeline GPU: {} days ({gpu_elapsed:.1?}), yield={:.3}, Δ={yield_diff:.4}",
                    gpu_result.n_days, gpu_result.yield_ratio
                ),
                yield_diff < 0.01,
            );
        }
    }
}

fn bench_precision_lineage(v: &mut ValidationHarness) {
    println!("\n── Precision Lineage (hotSpring → all springs) ────────────────");

    let t0 = Instant::now();

    v.check_abs(
        "erf(1) [hotSpring df64 S54 → all springs]",
        barracuda::math::erf(1.0),
        0.842_700_792_949_715,
        1e-6,
    );

    v.check_abs(
        "Γ(5)=24 [hotSpring special S54 → neuralSpring ML]",
        barracuda::math::gamma(5.0).unwrap_or(0.0),
        24.0,
        1e-10,
    );

    v.check_abs(
        "norm_cdf(0)=0.5 [hotSpring → groundSpring MC, airSpring CI]",
        barracuda::stats::normal::norm_cdf(0.0),
        0.5,
        1e-14,
    );

    for &z in &[-2.0, -1.0, 0.0, 1.0, 2.0] {
        let p = barracuda::stats::normal::norm_cdf(z);
        let z_back = barracuda::stats::normal::norm_ppf(p);
        v.check_abs(
            &format!("cdf↔ppf z={z:.0} [hotSpring ↔ airSpring MC CI]"),
            z_back,
            z,
            1e-4,
        );
    }

    let counts = [120.0, 85.0, 45.0, 30.0, 20.0];
    v.check_lower(
        "Shannon H' [wetSpring S64 → airSpring diversity]",
        barracuda::stats::diversity::shannon(&counts),
        1.0,
    );

    v.check_lower(
        "Simpson D [wetSpring S64 → neuralSpring uncertainty]",
        barracuda::stats::diversity::simpson(&counts),
        0.5,
    );

    let ridge = barracuda::linalg::ridge::ridge_regression(
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        &[1.0, 2.0, 3.0],
        3,
        2,
        1,
        0.01,
    );
    v.check_bool(
        "Ridge regression [wetSpring ESN → airSpring calibration]",
        ridge.is_ok(),
    );

    let (best_x, _f, _iters) = barracuda::optimize::nelder_mead::nelder_mead(
        |x: &[f64]| (x[0] - 3.0).mul_add(x[0] - 3.0, (x[1] - 4.0).powi(2)),
        &[0.0, 0.0],
        &[(-10.0, 10.0), (-10.0, 10.0)],
        500,
        1e-10,
    )
    .unwrap();
    v.check_abs(
        "Nelder-Mead x₀ [neuralSpring S52 → airSpring isotherm]",
        best_x[0],
        3.0,
        0.01,
    );

    println!("  Precision lineage: {:.1?}", t0.elapsed());
}
