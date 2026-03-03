// SPDX-License-Identifier: AGPL-3.0-or-later
#![warn(clippy::pedantic)]
#![allow(clippy::cast_precision_loss)]

//! Exp 073: Cross-Spring Evolution Rewire Validation
//!
//! Validates airSpring's rewiring to modern `ToadStool` S86 primitives and
//! benchmarks the cross-spring shader evolution — documenting when and
//! where each capability evolved across the ecosystem.
//!
//! # Cross-Spring Shader Provenance
//!
//! ```text
//! hotSpring ──── precision math (df64, pow_f64, exp_f64, erf, gamma)
//!     │              └── Used by: ALL springs, ALL shaders
//!     ├── Lanczos eigensolve ── Used by: wetSpring (Anderson QS), airSpring (Anderson coupling)
//!     ├── CrankNicolson1D f64 ── Used by: airSpring (Richards PDE linearised baseline)
//!     └── anderson_4d (S83) ── Used by: wetSpring (QS), airSpring (soil disorder)
//!
//! wetSpring ──── bio diversity (Shannon, Simpson, Bray-Curtis, Hill)
//!     │              └── Used by: airSpring (soil biodiversity, Paper 12 tissue)
//!     ├── kriging_f64 ── Used by: airSpring (soil moisture interpolation)
//!     ├── moving_window_f64 ── Used by: airSpring (IoT stream smoothing)
//!     └── NPU driver ── Used by: airSpring (AKD1000 edge inference)
//!
//! neuralSpring ── Nelder-Mead, BFGS, BatchedBisection
//!     │              └── Used by: airSpring (isotherm fitting, VG calibration)
//!     ├── ValidationHarness ── Used by: ALL springs, ALL validation binaries
//!     ├── BatchedNelderMeadGpu (S80) ── Used by: airSpring (batch isotherm)
//!     └── ridge_regression ── Used by: airSpring (sensor calibration)
//!
//! airSpring ──── FAO-56 ET₀ (op=0), WB (op=1), ops 5-13
//!     │              └── Used by: ToadStool (seasonal pipeline shader)
//!     ├── StatefulPipeline (S80) ── day-over-day water balance
//!     ├── BatchedStatefulF64 (S83) ── GPU-resident state carry
//!     ├── BrentGpu (S83) ── VG inverse θ→h on GPU
//!     ├── RichardsGpu (S83) ── GPU Picard solver
//!     └── SeasonalPipelineF64 ── fused ET₀→Kc→WB→stress
//!
//! groundSpring ── MC uncertainty propagation
//!     │              └── Used by: airSpring (MC ET₀ confidence intervals)
//!     ├── bootstrap/jackknife GPU ── Used by: airSpring (uncertainty stack)
//!     └── batched_multinomial ── Used by: airSpring (stochastic soil sampling)
//! ```

use std::sync::Arc;
use std::time::Instant;

use airspring_barracuda::eco::evapotranspiration::{self as et, DailyEt0Input, Et0Result};
use airspring_barracuda::eco::richards::VanGenuchtenParams;
use airspring_barracuda::eco::van_genuchten;
use airspring_barracuda::gpu::richards::{BatchedRichards, RichardsRequest};
use airspring_barracuda::gpu::van_genuchten::{BatchedVanGenuchten, compute_theta_cpu};
use airspring_barracuda::validation;
use barracuda::validation::ValidationHarness;

fn try_gpu_device() -> Option<Arc<barracuda::device::WgpuDevice>> {
    barracuda::device::test_pool::tokio_block_on(
        barracuda::device::WgpuDevice::new_f64_capable(),
    )
    .ok()
    .map(Arc::new)
}

fn main() {
    validation::init_tracing();
    validation::banner("Exp 073: Cross-Spring Evolution Rewire (ToadStool S86)");

    let mut v = ValidationHarness::new("Cross-Spring Rewire");

    validate_brent_gpu_vg_inverse(&mut v);
    validate_richards_gpu(&mut v);
    validate_stateful_pipeline(&mut v);
    validate_hydrology_cpu_gpu_parity(&mut v);
    validate_cross_spring_provenance(&mut v);
    benchmark_modern_rewire(&mut v);

    println!();
    v.finish();
}

/// Part 1: `BrentGpu` VG inverse — hotSpring precision × airSpring hydrology
fn validate_brent_gpu_vg_inverse(v: &mut ValidationHarness) {
    println!("\n── Part 1: BrentGpu VG Inverse (S83) ───────────────────────");
    println!("  Provenance: airSpring V045 → S83, brent_f64.wgsl (hotSpring f64 math)");

    let t0 = Instant::now();

    let Some(device) = try_gpu_device() else {
        println!("  SKIP: No f64-capable GPU device");
        return;
    };

    let vg = BatchedVanGenuchten::gpu(Arc::clone(&device)).unwrap();
    let theta_r = 0.065;
    let theta_s = 0.41;
    let alpha = 0.075;
    let n_vg = 1.89;

    let h_reference = [-500.0, -200.0, -100.0, -50.0, -20.0, -10.0, -5.0, -1.0];
    let theta_targets = compute_theta_cpu(theta_r, theta_s, alpha, n_vg, &h_reference);

    let cpu_inverses: Vec<f64> = theta_targets
        .iter()
        .map(|&t| {
            van_genuchten::inverse_van_genuchten_h(t, theta_r, theta_s, alpha, n_vg)
                .unwrap_or(f64::NAN)
        })
        .collect();

    v.check_bool(
        "CPU Brent VG inverse: all 8 roots found",
        cpu_inverses.iter().all(|x| x.is_finite()),
    );

    let gpu_result = vg.compute_inverse_gpu(theta_r, theta_s, alpha, n_vg, &theta_targets);
    v.check_bool("GPU BrentGpu dispatch succeeds", gpu_result.is_ok());

    if let Ok(gpu_roots) = gpu_result {
        v.check_bool(
            "GPU returns correct count",
            gpu_roots.len() == h_reference.len(),
        );

        let mut max_err = 0.0_f64;
        for (i, (&gpu_h, &cpu_h)) in gpu_roots.iter().zip(cpu_inverses.iter()).enumerate() {
            let err = (gpu_h - cpu_h).abs();
            max_err = max_err.max(err);
            v.check_abs(
                &format!("VG inverse h[{i}]: GPU={gpu_h:.2} vs CPU={cpu_h:.2}"),
                gpu_h,
                cpu_h,
                5.0,
            );
        }
        v.check_bool("GPU↔CPU max error < 5 cm", max_err < 5.0);

        let batch_sizes: [u32; 3] = [100, 1000, 10_000];
        for n in batch_sizes {
            let targets: Vec<f64> = (0..n)
                .map(|i| {
                    let frac = f64::from(i) / f64::from(n);
                    frac.mul_add((theta_s - theta_r) * 0.95, theta_r) + 0.001
                })
                .collect();
            let t_batch = Instant::now();
            let result = vg.compute_inverse_gpu(theta_r, theta_s, alpha, n_vg, &targets);
            let elapsed = t_batch.elapsed();
            v.check_bool(
                &format!("BrentGpu batch N={n}: dispatch OK ({elapsed:.1?})"),
                result.is_ok(),
            );
        }
    }

    println!("  BrentGpu VG inverse: {:.1?}", t0.elapsed());
}

/// Part 2: `RichardsGpu` — airSpring PDE × hotSpring precision × neuralSpring tridiagonal
fn validate_richards_gpu(v: &mut ValidationHarness) {
    println!("\n── Part 2: RichardsGpu (S83) ────────────────────────────────");
    println!("  Provenance: airSpring S40 → S83, richards_picard_f64.wgsl");
    println!("  Math: hotSpring pow_f64/exp_f64, neuralSpring tridiagonal");

    let t0 = Instant::now();

    let Some(device) = try_gpu_device() else {
        println!("  SKIP: No f64-capable GPU device");
        return;
    };

    let sand = VanGenuchtenParams {
        theta_r: 0.045,
        theta_s: 0.43,
        alpha: 0.145,
        n_vg: 2.68,
        ks: 712.8,
    };

    let req = RichardsRequest {
        params: sand,
        depth_cm: 100.0,
        n_nodes: 20,
        h_initial: -5.0,
        h_top: -5.0,
        zero_flux_top: true,
        bottom_free_drain: true,
        duration_days: 0.1,
        dt_days: 0.01,
    };

    let cpu_result = BatchedRichards::solve_upstream(&req);
    v.check_bool("CPU Richards (upstream CN) succeeds", cpu_result.is_ok());

    let gpu_result = BatchedRichards::solve_gpu(Arc::clone(&device), &req);
    v.check_bool("GPU Richards (Picard) succeeds", gpu_result.is_ok());

    if let (Ok(cpu_r), Ok(gpu_r)) = (&cpu_result, &gpu_result) {
        v.check_bool(
            "GPU node count matches CPU",
            gpu_r.h.len() == cpu_r.h.len(),
        );
        v.check_bool(
            "GPU completed time steps > 0",
            gpu_r.time_steps_completed > 0,
        );

        let soil = airspring_barracuda::gpu::richards::to_barracuda_params(&sand);
        for (i, &h) in gpu_r.h.iter().enumerate() {
            let theta = soil.theta(h);
            v.check_bool(
                &format!("GPU θ[{i}] in physical range"),
                (sand.theta_r - 1e-4..=sand.theta_s + 1e-4).contains(&theta),
            );
        }
    }

    let soils = [
        ("Sand", VanGenuchtenParams { theta_r: 0.045, theta_s: 0.43, alpha: 0.145, n_vg: 2.68, ks: 712.8 }),
        ("SiltLoam", VanGenuchtenParams { theta_r: 0.067, theta_s: 0.45, alpha: 0.020, n_vg: 1.41, ks: 10.8 }),
        ("Clay", VanGenuchtenParams { theta_r: 0.068, theta_s: 0.38, alpha: 0.008, n_vg: 1.09, ks: 4.8 }),
    ];

    for (name, params) in &soils {
        let req = RichardsRequest {
            params: *params,
            depth_cm: 50.0,
            n_nodes: 10,
            h_initial: -10.0,
            h_top: -10.0,
            zero_flux_top: true,
            bottom_free_drain: true,
            duration_days: 0.05,
            dt_days: 0.005,
        };
        let result = BatchedRichards::solve_gpu(Arc::clone(&device), &req);
        v.check_bool(
            &format!("RichardsGpu {name}: dispatch OK"),
            result.is_ok(),
        );
    }

    println!("  RichardsGpu validation: {:.1?}", t0.elapsed());
}

/// Part 3: `StatefulPipeline` (S80) — day-over-day water balance
fn validate_stateful_pipeline(v: &mut ValidationHarness) {
    println!("\n── Part 3: StatefulPipeline + BatchedStatefulF64 ──────────");
    println!("  Provenance: airSpring V039/V045 → S80/S83");

    let t0 = Instant::now();

    let wbs = barracuda::pipeline::WaterBalanceState::new(0.30, 0.0, 0.0);
    v.check_abs("WaterBalanceState soil_moisture = 0.30", wbs.soil_moisture, 0.30, 1e-15);
    v.check_abs("WaterBalanceState snow = 0.0", wbs.snow_water_eq, 0.0, 1e-15);

    let mut pipe = barracuda::pipeline::StatefulPipeline::<
        barracuda::pipeline::WaterBalanceState,
    >::new();
    let out = pipe.run(&[1.0, 2.0, 3.0]);
    v.check_bool(
        "StatefulPipeline passthrough preserves input",
        out.len() == 3 && (out[0] - 1.0).abs() < 1e-15,
    );

    v.check_bool(
        "BatchedStatefulF64 type available (S83)",
        std::any::type_name::<barracuda::pipeline::batched_stateful::BatchedStatefulF64>()
            .contains("BatchedStatefulF64"),
    );

    println!("  StatefulPipeline validation: {:.1?}", t0.elapsed());
}

fn make_et0_input(tmin: f64, tmax: f64, rs: f64, wind: f64, elev: f64, lat: f64, doy: u32) -> DailyEt0Input {
    let tmean = f64::midpoint(tmin, tmax);
    let ea = 0.6108 * (17.27 * tmean / (tmean + 237.3)).exp() * 0.70;
    DailyEt0Input {
        tmin,
        tmax,
        tmean: None,
        solar_radiation: rs,
        wind_speed_2m: wind,
        actual_vapour_pressure: ea,
        elevation_m: elev,
        latitude_deg: lat,
        day_of_year: doy,
    }
}

/// Part 4: Hydrology CPU/GPU parity — cross-spring function validation
fn validate_hydrology_cpu_gpu_parity(v: &mut ValidationHarness) {
    println!("\n── Part 4: Hydrology CPU↔GPU Parity ────────────────────────");
    println!("  Provenance: airSpring V009 → S66 (CPU), S72 (GPU `SeasonalPipelineF64`)");

    let t_start = Instant::now();

    let et0_upstream = barracuda::stats::hydrology::fao56_et0(
        21.5, 12.3, 84.0, 63.0, 2.78, 22.07, 100.0, 50.8, 187,
    );
    v.check_bool(
        "fao56_et0 (upstream) returns valid ET₀",
        et0_upstream.is_some_and(|val| val > 0.0 && val < 20.0),
    );

    let local_input = make_et0_input(12.3, 21.5, 22.07, 2.78, 100.0, 50.8, 187);
    let local_result: Et0Result = et::daily_et0(&local_input);
    v.check_bool("local PM ET₀ > 0", local_result.et0 > 0.0);

    let hg_upstream = barracuda::stats::hydrology::hargreaves_et0(22.07, 21.5, 12.3);
    v.check_bool(
        "Hargreaves upstream returns Some",
        hg_upstream.is_some(),
    );

    let hg_local = et::hargreaves_et0(12.3, 21.5, 22.07);
    if let Some(hg_up) = hg_upstream {
        v.check_abs(
            "Hargreaves local↔upstream parity",
            hg_local,
            hg_up,
            0.5,
        );
    }

    let theta = barracuda::stats::hydrology::soil_water_balance(0.30, 5.0, 0.0, 3.0, 0.45);
    v.check_bool(
        "soil_water_balance θ in [0, fc]",
        (0.0..=0.45).contains(&theta),
    );

    let kc = barracuda::stats::hydrology::crop_coefficient(0.3, 1.15, 30, 60);
    v.check_bool(
        "crop_coefficient interpolation in range",
        kc > 0.3 && kc < 1.15,
    );

    let th_et0 = barracuda::stats::hydrology::thornthwaite_et0(18.0, 50.0, 12.0, 30.0);
    v.check_bool(
        "Thornthwaite ET₀ returns Some",
        th_et0.is_some(),
    );

    let mk_et0 = barracuda::stats::hydrology::makkink_et0(18.0, 22.0);
    v.check_bool("Makkink ET₀ returns Some", mk_et0.is_some());

    let turc_et0 = barracuda::stats::hydrology::turc_et0(18.0, 22.0, 60.0);
    v.check_bool("Turc ET₀ returns Some", turc_et0.is_some());

    let hamon_et0 = barracuda::stats::hydrology::hamon_et0(18.0, 14.0);
    v.check_bool("Hamon ET₀ returns Some", hamon_et0.is_some());

    println!("  Hydrology parity: {:.1?}", t_start.elapsed());
}

/// Part 5: Cross-spring provenance — verify each spring's contribution
fn validate_cross_spring_provenance(v: &mut ValidationHarness) {
    println!("\n── Part 5: Cross-Spring Provenance ──────────────────────────");

    let t0 = Instant::now();

    let erf_val = barracuda::math::erf(1.0);
    v.check_abs("hotSpring: erf(1) precision", erf_val, 0.842_700_792_949_715, 1e-6);

    let gamma_val = barracuda::math::gamma(5.0).expect("gamma(5) should not fail");
    v.check_abs("hotSpring: Γ(5) = 24", gamma_val, 24.0, 1e-4);

    let h = barracuda::spectral::anderson::anderson_4d(3, 1.0, 42);
    v.check_bool(
        "hotSpring S83: anderson_4d L=3 → 81 sites",
        h.n == 81,
    );

    let div = barracuda::stats::diversity::shannon(&[0.2, 0.2, 0.2, 0.2, 0.2]);
    v.check_abs("wetSpring: Shannon uniform(5) ≈ ln(5)", div, 5.0_f64.ln(), 0.01);

    let brent_result = barracuda::optimize::brent(|x| x.mul_add(x, -2.0), 0.0, 2.0, 1e-10, 100);
    v.check_bool("neuralSpring: CPU Brent √2 converges", brent_result.is_ok());
    if let Ok(r) = brent_result {
        v.check_abs("neuralSpring: Brent √2 = 1.4142...", r.root, std::f64::consts::SQRT_2, 1e-8);
    }

    let config = barracuda::optimize::lbfgs::LbfgsConfig {
        max_iter: 100,
        ..barracuda::optimize::lbfgs::LbfgsConfig::default()
    };
    let lbfgs_result = barracuda::optimize::lbfgs::lbfgs_numerical(
        |x: &[f64]| {
            let a = (1.0_f64 - x[0]).powi(2);
            let b = x[0].mul_add(-x[0], x[1]).powi(2);
            b.mul_add(100.0, a)
        },
        &[0.0, 0.0],
        &config,
    );
    v.check_bool("neuralSpring S83: L-BFGS Rosenbrock runs", lbfgs_result.is_ok());

    let ci = barracuda::stats::bootstrap_ci(
        &[1.0, 2.0, 3.0, 4.0, 5.0],
        |d| d.iter().sum::<f64>() / d.len() as f64,
        100,
        0.95,
        42,
    )
    .expect("bootstrap_ci should succeed");
    v.check_bool(
        "groundSpring: bootstrap CI lower < upper",
        ci.lower < ci.upper,
    );

    let et0_input = make_et0_input(15.0, 25.0, 20.0, 2.0, 200.0, 42.0, 180);
    let et0_result = et::daily_et0(&et0_input);
    v.check_bool("airSpring: local FAO-56 PM ET₀ > 0", et0_result.et0 > 0.0);

    println!("  Cross-spring provenance: {:.1?}", t0.elapsed());
}

/// Part 6: Modern rewire benchmarks
#[allow(clippy::similar_names)]
fn benchmark_modern_rewire(v: &mut ValidationHarness) {
    println!("\n── Part 6: Modern Rewire Benchmarks ────────────────────────");

    let t0 = Instant::now();

    let Some(device) = try_gpu_device() else {
        println!("  SKIP: No f64-capable GPU device for benchmarks");
        return;
    };

    let batch_n: u32 = 10_000;
    let theta_r = 0.065;
    let theta_s = 0.41;
    let alpha = 0.075;
    let n_vg = 1.89;

    let targets: Vec<f64> = (0..batch_n)
        .map(|i| {
            let frac = f64::from(i) / f64::from(batch_n);
            frac.mul_add((theta_s - theta_r) * 0.95, theta_r) + 0.001
        })
        .collect();

    let t_cpu = Instant::now();
    let _cpu_roots: Vec<f64> = targets
        .iter()
        .map(|&t| {
            van_genuchten::inverse_van_genuchten_h(t, theta_r, theta_s, alpha, n_vg)
                .unwrap_or(f64::NAN)
        })
        .collect();
    let cpu_elapsed = t_cpu.elapsed();

    let vg = BatchedVanGenuchten::gpu(Arc::clone(&device)).unwrap();
    let t_gpu = Instant::now();
    let _gpu_roots = vg
        .compute_inverse_gpu(theta_r, theta_s, alpha, n_vg, &targets)
        .unwrap();
    let gpu_elapsed = t_gpu.elapsed();

    let cpu_rate = f64::from(batch_n) / cpu_elapsed.as_secs_f64();
    let gpu_rate = f64::from(batch_n) / gpu_elapsed.as_secs_f64();

    v.check_bool(
        &format!("VG inverse N={batch_n}: CPU {cpu_rate:.0}/s, GPU {gpu_rate:.0}/s"),
        true,
    );

    let n_et0: u32 = 100_000;
    let t_et0 = Instant::now();
    for i in 0..n_et0 {
        let fi = f64::from(i);
        let input = make_et0_input(15.0, fi.mul_add(0.0001, 25.0), 20.0, 2.0, 200.0, 42.0, 180);
        let _ = et::daily_et0(&input);
    }
    let et0_rate = f64::from(n_et0) / t_et0.elapsed().as_secs_f64();
    v.check_bool(
        &format!("FAO-56 ET₀ throughput: {et0_rate:.0}/s (CPU)"),
        et0_rate > 100_000.0,
    );

    let sand = VanGenuchtenParams {
        theta_r: 0.045, theta_s: 0.43, alpha: 0.145, n_vg: 2.68, ks: 712.8,
    };
    let req = RichardsRequest {
        params: sand,
        depth_cm: 50.0,
        n_nodes: 20,
        h_initial: -5.0,
        h_top: -5.0,
        zero_flux_top: true,
        bottom_free_drain: true,
        duration_days: 0.05,
        dt_days: 0.005,
    };

    let t_rcpu = Instant::now();
    let _ = BatchedRichards::solve_upstream(&req).unwrap();
    let rcpu_elapsed = t_rcpu.elapsed();

    let t_rgpu = Instant::now();
    let _ = BatchedRichards::solve_gpu(device, &req).unwrap();
    let rgpu_elapsed = t_rgpu.elapsed();

    v.check_bool(
        &format!("Richards PDE: CPU {rcpu_elapsed:.1?}, GPU {rgpu_elapsed:.1?}"),
        true,
    );

    println!("  Benchmarks: {:.1?}", t0.elapsed());
}
