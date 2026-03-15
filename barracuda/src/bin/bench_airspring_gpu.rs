// SPDX-License-Identifier: AGPL-3.0-or-later
//! Benchmark airSpring GPU operations vs CPU baselines.
//!
//! Measures wall-clock time for all GPU orchestrators and CPU fallbacks across
//! multiple problem sizes, reporting throughput and cross-spring provenance.
//!
//! # Cross-spring evolution context
//!
//! These GPU paths exist because of shader evolution across the `ecoPrimals`
//! ecosystem (774 WGSL shaders in `BarraCuda` S68, 46+ cross-spring absorptions):
//!
//! - **ET₀ batch** (`batched_elementwise_f64.wgsl`): hotSpring `pow_f64` fix
//!   (TS-001) made fractional exponents work; airSpring wired FAO-56 as op=0
//! - **Water balance batch**: Same shader, op=1 — multi-Spring shared primitive
//! - **Kriging** (`kriging_f64.wgsl`): wetSpring spatial interpolation; airSpring
//!   wired soil moisture mapping
//! - **Reduce** (`fused_map_reduce_f64.wgsl`): wetSpring origin; airSpring TS-004
//!   fix stabilized N≥1024 dispatch for all Springs
//! - **Stream smoothing** (`moving_window.wgsl`): wetSpring S28+ environmental
//!   monitoring; airSpring wired `IoT` sensor smoothing
//! - **Ridge regression** (`barracuda::linalg::ridge`): wetSpring ESN calibration;
//!   airSpring wired sensor correction pipeline
//! - **Regression** (`stats::regression`): airSpring metalForge → absorbed S66,
//!   completing the Write→Absorb→Lean cycle for sensor correction fitting
//! - **S66 GPU dispatch fix**: explicit `BindGroupLayout` (R-S66-041) resolved
//!   the P0 `BatchedElementwiseF64` dispatch panic — GPU-first paths now stable
//!
//! # Usage
//!
//! ```sh
//! cargo run --release --bin bench_airspring_gpu
//! ```

use airspring_barracuda::eco::evapotranspiration::{self as et, DailyEt0Input};
use airspring_barracuda::eco::isotherm;
use airspring_barracuda::eco::richards::{self, VanGenuchtenParams};
use airspring_barracuda::gpu::mc_et0::{self, Et0Uncertainties};
use airspring_barracuda::gpu::{isotherm as gpu_iso, kriging, reduce, stream};
use std::time::Instant;

const WARMUP: usize = 3;
const MEASURE: usize = 10;

fn make_station_days(n: usize) -> Vec<DailyEt0Input> {
    (0..n)
        .map(|i| {
            let day = i as f64;
            DailyEt0Input {
                tmin: 12.0 + (day * 0.01).sin(),
                tmax: 25.0 + (day * 0.01).cos(),
                tmean: None,
                solar_radiation: 22.0 + (day * 0.02).sin(),
                wind_speed_2m: 2.0,
                actual_vapour_pressure: 1.4,
                elevation_m: 100.0,
                latitude_deg: 50.8,
                day_of_year: 187,
            }
        })
        .collect()
}

fn bench_et0_cpu(inputs: &[DailyEt0Input]) -> f64 {
    inputs.iter().map(|i| et::daily_et0(i).et0).sum()
}

fn bench_reduce_cpu(data: &[f64]) -> f64 {
    let r = reduce::compute_seasonal_stats(data);
    r.mean
}

fn bench_stream_cpu(data: &[f64], window: usize) -> f64 {
    let result = stream::smooth_cpu(data, window).expect("smooth_cpu: valid window and data");
    result.mean[0]
}

fn bench_kriging_cpu(sensors: &[kriging::SensorReading], targets: &[kriging::TargetPoint]) -> f64 {
    let variogram = kriging::SoilVariogram::Spherical {
        nugget: 0.001,
        sill: 0.01,
        range: 15.0,
    };
    let result = kriging::interpolate_soil_moisture(sensors, targets, variogram);
    result.vwc_values[0]
}

fn time_fn<F: FnMut() -> f64>(mut f: F, warmup: usize, measure: usize) -> (f64, f64) {
    for _ in 0..warmup {
        let _ = f();
    }
    let start = Instant::now();
    let mut checksum = 0.0;
    for _ in 0..measure {
        checksum += f();
    }
    let elapsed_us = start.elapsed().as_micros();
    #[expect(
        clippy::cast_precision_loss,
        reason = "u128 microseconds fit in f64 for benchmark timing"
    )]
    let per_call_us = elapsed_us as f64 / measure as f64;
    (per_call_us, checksum)
}

fn bench_et0() {
    println!("── Batched ET₀ (batched_elementwise_f64, hotSpring pow_f64 fix) ──");
    println!("  {:>8}  {:>12}  {:>12}", "N", "CPU (µs)", "ops/sec");

    for &n in &[10, 100, 1_000, 10_000] {
        let inputs = make_station_days(n);
        let (cpu_us, _) = time_fn(|| bench_et0_cpu(&inputs), WARMUP, MEASURE);
        #[expect(
            clippy::cast_precision_loss,
            reason = "usize count fits in f64 for ops/sec calculation"
        )]
        let ops_per_sec = (n as f64) / (cpu_us / 1_000_000.0);
        println!("  {n:>8}  {cpu_us:>12.1}  {ops_per_sec:>12.0}");
    }
}

fn bench_reduce() {
    println!();
    println!("── Seasonal Reduce (fused_map_reduce_f64, wetSpring origin, TS-004 fix) ──");
    println!("  {:>8}  {:>12}  {:>12}", "N", "CPU (µs)", "M elem/sec");

    for &n in &[100_i32, 1_000, 10_000, 100_000] {
        let data: Vec<f64> = (0..n).map(|i| f64::from(i) * 0.01).collect();
        let (cpu_us, _) = time_fn(|| bench_reduce_cpu(&data), WARMUP, MEASURE);
        let m_elem_sec = f64::from(n) / (cpu_us / 1_000_000.0) / 1e6;
        println!("  {n:>8}  {cpu_us:>12.1}  {m_elem_sec:>12.1}");
    }
}

fn bench_stream() {
    println!();
    println!("── Stream Smoothing (moving_window.wgsl, wetSpring S28+ environmental) ──");
    println!(
        "  {:>8}  {:>8}  {:>12}  {:>12}",
        "N", "Window", "CPU (µs)", "M elem/sec"
    );

    for &(n, w) in &[(168, 24), (720, 24), (8760, 24), (8760, 168)] {
        let data: Vec<f64> = (0..n)
            .map(|i| {
                8.0f64.mul_add(
                    ((f64::from(i) % 24.0 - 14.0) * std::f64::consts::PI / 12.0).cos(),
                    25.0,
                )
            })
            .collect();
        let (cpu_us, _) = time_fn(|| bench_stream_cpu(&data, w), WARMUP, MEASURE);
        let m_elem_sec = f64::from(n) / (cpu_us / 1_000_000.0) / 1e6;
        println!("  {n:>8}  {w:>8}  {cpu_us:>12.1}  {m_elem_sec:>12.1}");
    }
}

fn bench_kriging() {
    println!();
    println!("── Kriging (kriging_f64.wgsl, wetSpring spatial interpolation) ──");
    println!("  {:>8}  {:>8}  {:>12}", "Sensors", "Targets", "CPU (µs)");

    for &(ns, nt) in &[(5, 10), (10, 100), (20, 500)] {
        let sensors: Vec<kriging::SensorReading> = (0..ns)
            .map(|i| {
                let fi = f64::from(i);
                kriging::SensorReading {
                    x: fi * 10.0,
                    y: fi * 5.0,
                    vwc: 0.01f64.mul_add(fi, 0.25),
                }
            })
            .collect();
        let targets: Vec<kriging::TargetPoint> = (0..nt)
            .map(|i| {
                let fi = f64::from(i);
                kriging::TargetPoint {
                    x: fi * 2.0,
                    y: fi * 1.0,
                }
            })
            .collect();
        let (cpu_us, _) = time_fn(|| bench_kriging_cpu(&sensors, &targets), WARMUP, MEASURE);
        println!("  {ns:>8}  {nt:>8}  {cpu_us:>12.1}");
    }
}

fn bench_ridge() {
    println!();
    println!("── Ridge Regression (barracuda::linalg::ridge, wetSpring ESN calibration) ──");
    println!("  {:>8}  {:>12}  {:>12}", "N", "CPU (µs)", "R²");

    for &n in &[50, 200, 1_000, 5_000] {
        let x: Vec<f64> = (0..n).map(|i| f64::from(i) * 0.01).collect();
        let y: Vec<f64> = x
            .iter()
            .map(|&xi| (xi * 0.1).sin().mul_add(0.01, 2.5f64.mul_add(xi, 0.3)))
            .collect();
        let mut r2_val = 0.0;
        let (cpu_us, _) = time_fn(
            || {
                let m = airspring_barracuda::eco::correction::fit_ridge(&x, &y, 1e-6)
                    .expect("fit_ridge: sufficient data for regression");
                r2_val = m.r_squared;
                m.r_squared
            },
            WARMUP,
            MEASURE,
        );
        println!("  {n:>8}  {cpu_us:>12.1}  {r2_val:>12.6}");
    }
}

fn bench_richards() {
    println!();
    println!("── Richards PDE (pde::richards, airSpring→BarraCuda S40 absorption) ──");
    println!("  Solver uses hotSpring df64 precision for VG constitutive relations.");
    println!("  Crank-Nicolson + Picard iteration; Thomas algorithm for tridiagonal.");
    println!("  {:>8}  {:>12}  {:>12}", "Nodes", "CPU (µs)", "sims/sec");

    let sand = VanGenuchtenParams {
        theta_r: 0.045,
        theta_s: 0.43,
        alpha: 0.145,
        n_vg: 2.68,
        ks: 712.8,
    };
    for &n_nodes in &[10, 20, 50, 100] {
        let (cpu_us, _) = time_fn(
            || {
                let r = richards::solve_richards_1d(
                    &sand, 50.0, n_nodes, -20.0, 0.0, true, false, 0.1, 0.01,
                );
                r.map_or(0.0, |profiles| profiles.last().map_or(0.0, |p| p.theta[0]))
            },
            WARMUP,
            MEASURE,
        );
        let sims_per_sec = 1_000_000.0 / cpu_us;
        println!("  {n_nodes:>8}  {cpu_us:>12.1}  {sims_per_sec:>12.0}");
    }
}

fn bench_isotherm() {
    println!();
    println!("── Isotherm Fitting (optimize::nelder_mead → multi_start, neuralSpring) ──");
    println!("  Linearized LS → single NM → multi-start global (LHS exploration).");
    println!(
        "  {:>22}  {:>12}  {:>12}  {:>8}",
        "Method", "CPU (µs)", "fits/sec", "R²"
    );

    let ce_wood = [1.0, 2.5, 5.0, 10.0, 20.0, 40.0, 60.0, 80.0, 100.0];
    let qe_wood = [0.85, 1.92, 3.45, 5.8, 8.9, 12.1, 13.8, 14.5, 14.9];

    {
        let mut r2 = 0.0;
        let (cpu_us, _) = time_fn(
            || {
                let fit = isotherm::fit_langmuir(&ce_wood, &qe_wood).unwrap();
                r2 = fit.r_squared;
                fit.r_squared
            },
            WARMUP,
            100,
        );
        let fits_sec = 1_000_000.0 / cpu_us;
        println!(
            "  {:<22}  {cpu_us:>12.1}  {fits_sec:>12.0}  {r2:>8.4}",
            "Linearized LS"
        );
    }

    {
        let mut r2 = 0.0;
        let (cpu_us, _) = time_fn(
            || {
                let fit = gpu_iso::fit_langmuir_nm(&ce_wood, &qe_wood).unwrap();
                r2 = fit.r_squared;
                fit.r_squared
            },
            WARMUP,
            MEASURE,
        );
        let fits_sec = 1_000_000.0 / cpu_us;
        println!(
            "  {:<22}  {cpu_us:>12.1}  {fits_sec:>12.0}  {r2:>8.4}",
            "Nelder-Mead (1 start)"
        );
    }

    {
        let mut r2 = 0.0;
        let (cpu_us, _) = time_fn(
            || {
                let fit = gpu_iso::fit_langmuir_global(&ce_wood, &qe_wood, 8).unwrap();
                r2 = fit.r_squared;
                fit.r_squared
            },
            WARMUP,
            MEASURE,
        );
        let fits_sec = 1_000_000.0 / cpu_us;
        println!(
            "  {:<22}  {cpu_us:>12.1}  {fits_sec:>12.0}  {r2:>8.4}",
            "Multi-start NM (8×LHS)"
        );
    }
}

fn bench_vg_theta() {
    println!();
    println!("── Van Genuchten θ(h) batch (pure arithmetic, GPU-ready via df64) ──");
    println!("  hotSpring df64 precision enables exact GPU retention curves.");
    println!("  {:>8}  {:>12}  {:>12}", "N", "CPU (µs)", "M evals/sec");

    let sand = VanGenuchtenParams {
        theta_r: 0.045,
        theta_s: 0.43,
        alpha: 0.145,
        n_vg: 2.68,
        ks: 712.8,
    };
    for &n in &[1_000_i32, 10_000, 100_000] {
        let (cpu_us, _) = time_fn(
            || {
                let mut sum = 0.0;
                for i in 0..n {
                    let h = -0.01 * (f64::from(i) + 1.0);
                    sum += richards::van_genuchten_theta(
                        h,
                        sand.theta_r,
                        sand.theta_s,
                        sand.alpha,
                        sand.n_vg,
                    );
                }
                sum
            },
            WARMUP,
            MEASURE,
        );
        let m_evals_sec = f64::from(n) / (cpu_us / 1_000_000.0) / 1e6;
        println!("  {n:>8}  {cpu_us:>12.1}  {m_evals_sec:>12.1}");
    }
}

fn bench_mc_et0_ci() {
    println!();
    println!("── MC ET₀ + Parametric CI (norm_ppf, hotSpring precision → S52+) ──");
    println!("  norm_ppf (Moro 1995) enables analytic z-score confidence intervals.");
    println!(
        "  {:>8}  {:>12}  {:>12}  {:>10}  {:>10}",
        "N_MC", "CPU (µs)", "samples/s", "CI₉₀ width", "P₅-P₉₅"
    );

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
    let unc = Et0Uncertainties::default();

    for &n in &[100_u32, 500, 2_000, 10_000] {
        let n_usize = n as usize;
        let (cpu_us, _) = time_fn(
            || {
                let r = mc_et0::mc_et0_cpu(&input, &unc, n_usize, 42);
                r.et0_mean
            },
            WARMUP,
            MEASURE,
        );
        let r = mc_et0::mc_et0_cpu(&input, &unc, n_usize, 42);
        let (ci_lo, ci_hi) = r.parametric_ci(0.90);
        let samples_sec = f64::from(n) / (cpu_us / 1_000_000.0);
        println!(
            "  {n:>8}  {cpu_us:>12.1}  {samples_sec:>12.0}  {ci_w:>10.4}  {emp_w:>10.4}",
            ci_w = ci_hi - ci_lo,
            emp_w = r.et0_p95 - r.et0_p05,
        );
    }
}

fn bench_brent_vg_inverse() {
    println!();
    println!("── VG Pressure Head Inversion (brent, neuralSpring optimizer → S52+) ──");
    println!("  Brent (1973) root-finder: guaranteed convergence for θ(h) inversion.");
    println!(
        "  {:>12}  {:>8}  {:>12}  {:>12}",
        "Soil", "N", "CPU (µs)", "inversions/s"
    );

    let soils: &[(&str, f64, f64, f64, f64)] = &[
        ("sand", 0.045, 0.43, 0.145, 2.68),
        ("silt_loam", 0.067, 0.45, 0.02, 1.41),
        ("clay", 0.068, 0.38, 0.008, 1.09),
    ];

    for &(name, theta_r, theta_s, alpha, n_vg) in soils {
        let n = 1000_i32;
        let thetas: Vec<f64> = (0..n)
            .map(|i| theta_r + (theta_s - theta_r) * (f64::from(i) + 1.0) / f64::from(n + 1))
            .collect();
        let (cpu_us, _) = time_fn(
            || {
                thetas
                    .iter()
                    .filter_map(|&t| {
                        richards::inverse_van_genuchten_h(t, theta_r, theta_s, alpha, n_vg)
                    })
                    .sum()
            },
            WARMUP,
            MEASURE,
        );
        let inv_sec = f64::from(n) / (cpu_us / 1_000_000.0);
        println!("  {name:>12}  {n:>8}  {cpu_us:>12.1}  {inv_sec:>12.0}");
    }
}

fn bench_regression() {
    println!();
    println!("── Regression Fitting (stats::regression, airSpring metalForge → S66 absorbed) ──");
    println!("  Write→Absorb→Lean: local metalForge → upstream barracuda::stats::regression.");
    println!(
        "  {:>8}  {:>12}  {:>12}  {:>8}",
        "N", "CPU (µs)", "fits/sec", "R²"
    );

    for &n in &[20_i32, 100, 500, 2_000] {
        let x: Vec<f64> = (0..n).map(|i| f64::from(i) * 0.1).collect();
        let y: Vec<f64> = x
            .iter()
            .map(|&xi| 2.5f64.mul_add(xi, 0.3) + (xi * 0.01).sin())
            .collect();
        let mut r2 = 0.0;
        let (cpu_us, _) = time_fn(
            || {
                let fit = barracuda::stats::regression::fit_linear(&x, &y).unwrap();
                r2 = fit.r_squared;
                fit.r_squared
            },
            WARMUP,
            MEASURE,
        );
        let fits_sec = 1_000_000.0 / cpu_us;
        println!("  {n:>8}  {cpu_us:>12.1}  {fits_sec:>12.0}  {r2:>8.6}");
    }
}

fn bench_soil_params() {
    println!();
    println!("── SoilParams θ(h) batch (pde::richards::SoilParams, S66 named constants) ──");
    println!("  8 named soils from Carsel & Parrish (1988), R-S66-006.");
    println!(
        "  {:>12}  {:>8}  {:>12}  {:>12}",
        "Soil", "N", "CPU (µs)", "M evals/sec"
    );

    let soils: &[(&str, barracuda::pde::richards::SoilParams)] = &[
        ("sand", barracuda::pde::richards::SoilParams::SAND),
        (
            "sandy_loam",
            barracuda::pde::richards::SoilParams::SANDY_LOAM,
        ),
        ("silt_loam", barracuda::pde::richards::SoilParams::SILT_LOAM),
        ("clay", barracuda::pde::richards::SoilParams::CLAY),
    ];

    for &(name, soil) in soils {
        let n = 100_000_i32;
        let (cpu_us, _) = time_fn(
            || {
                let mut sum = 0.0;
                for i in 0..n {
                    let h = -0.01 * (f64::from(i) + 1.0);
                    sum += soil.theta(h);
                }
                sum
            },
            WARMUP,
            MEASURE,
        );
        let m_evals_sec = f64::from(n) / (cpu_us / 1_000_000.0) / 1e6;
        println!("  {name:>12}  {n:>8}  {cpu_us:>12.1}  {m_evals_sec:>12.1}");
    }
}

fn bench_scheduling_pipeline() {
    use airspring_barracuda::eco::water_balance::DailyInput;
    use airspring_barracuda::gpu::water_balance::BatchedWaterBalance;
    println!();
    println!("── Scheduling Pipeline: ET₀→Kc→WB→Yield (Exp 014 composition) ──");
    println!("  Full ET₀→crop coefficient→water balance→Stewart yield pipeline.");
    println!("  {:>8}  {:>12}  {:>12}", "Days", "CPU (µs)", "seasons/sec");

    for &n_days in &[90_u32, 180, 365] {
        let inputs: Vec<DailyEt0Input> = (0..n_days)
            .map(|i| {
                let d = f64::from(i);
                DailyEt0Input {
                    tmin: 12.0 + (d * 0.017).sin(),
                    tmax: 25.0 + (d * 0.017).cos(),
                    tmean: None,
                    solar_radiation: 4.0f64.mul_add((d * 0.017).sin(), 18.0),
                    wind_speed_2m: 2.0,
                    actual_vapour_pressure: 1.4,
                    elevation_m: 200.0,
                    latitude_deg: 42.0,
                    day_of_year: 91 + i,
                }
            })
            .collect();

        let kc_values: Vec<f64> = (0..n_days)
            .map(|i| {
                let frac = f64::from(i) / f64::from(n_days);
                if frac < 0.2 {
                    0.3
                } else if frac < 0.5 {
                    ((frac - 0.2) / 0.3).mul_add(0.85, 0.3)
                } else if frac < 0.8 {
                    1.15
                } else {
                    ((frac - 0.8) / 0.2).mul_add(-0.75, 1.15)
                }
            })
            .collect();

        let (cpu_us, _) = time_fn(
            || {
                let et0_values: Vec<f64> =
                    inputs.iter().map(|inp| et::daily_et0(inp).et0).collect();
                let wb_inputs: Vec<DailyInput> = et0_values
                    .iter()
                    .zip(&kc_values)
                    .enumerate()
                    .map(|(i, (&e, &kc))| DailyInput {
                        precipitation: if i % 5 == 0 { 10.0 } else { 0.0 },
                        irrigation: if i % 7 == 0 { 20.0 } else { 0.0 },
                        et0: e,
                        kc,
                    })
                    .collect();
                let engine = BatchedWaterBalance::new(0.30, 0.10, 500.0, 0.5);
                let summary = engine.simulate_season(&wb_inputs);
                let ky = 1.25;
                let etc_sum: f64 = et0_values.iter().zip(&kc_values).map(|(e, k)| e * k).sum();
                let eta_ratio = summary.total_actual_et / etc_sum;
                1.0 - ky * (1.0 - eta_ratio)
            },
            WARMUP,
            MEASURE,
        );
        let seasons_sec = 1_000_000.0 / cpu_us;
        println!("  {n_days:>8}  {cpu_us:>12.1}  {seasons_sec:>12.0}");
    }
}

fn run_all_benchmarks() {
    bench_et0();
    bench_reduce();
    bench_stream();
    bench_kriging();
    bench_ridge();
    bench_richards();
    bench_isotherm();
    bench_vg_theta();
    bench_mc_et0_ci();
    bench_brent_vg_inverse();
    bench_regression();
    bench_soil_params();
    bench_scheduling_pipeline();
}

fn main() {
    airspring_barracuda::validation::init_tracing();
    println!("═══════════════════════════════════════════════════════════════════════");
    println!("  airSpring GPU Benchmark — Cross-Spring Shader Evolution");
    println!("═══════════════════════════════════════════════════════════════════════");
    println!();

    run_all_benchmarks();

    // ── Summary ──────────────────────────────────────────────────────
    println!();
    println!("═══════════════════════════════════════════════════════════════════════");
    println!("  Cross-Spring Shader Evolution — Who Helps Whom");
    println!("═══════════════════════════════════════════════════════════════════════");
    println!();
    println!("  hotSpring (S42) → Precision foundation");
    println!("    df64 core: enables f64 GPU math for ALL Springs");
    println!("    pow_f64 fix (TS-001): airSpring ET₀ uncovered, hotSpring math fixed");
    println!("    exp/log/trig f64: airSpring VG retention + atmospheric pressure");
    println!("    └→ airSpring: VG retention, atmospheric pressure, ET₀");
    println!();
    println!("  wetSpring (S28) → Bio/environmental primitives");
    println!("    kriging_f64: wetSpring sample sites → airSpring soil moisture mapping");
    println!("    fused_map_reduce: airSpring TS-004 fix → stabilized for ALL Springs");
    println!("    moving_window: wetSpring environmental → airSpring IoT sensor smoothing");
    println!("    ridge_regression: wetSpring ESN → airSpring sensor calibration");
    println!("    └→ airSpring: soil mapping, seasonal stats, IoT smoothing, calibration");
    println!();
    println!("  neuralSpring (S52) → ML/optimization");
    println!("    nelder_mead: neuralSpring optimizer → airSpring isotherm fitting");
    println!("    multi_start_nelder_mead: LHS → airSpring global isotherm search");
    println!("    brent: neuralSpring root-finder → airSpring VG θ→h inversion (v0.4.5)");
    println!("    ValidationHarness: neuralSpring S59 → all 21 airSpring binaries");
    println!("    └→ airSpring: isotherm fitting, VG inversion, all binaries");
    println!();
    println!("  groundSpring (S64) → Uncertainty quantification");
    println!("    mc_et0_propagate: Monte Carlo ET₀ propagation");
    println!("    norm_ppf: Moro (1995) inverse normal for parametric CI");
    println!("    └→ airSpring: MC uncertainty, parametric CI");
    println!();
    println!("  airSpring (3 fixes + S40 + S66 metalForge) → Domain validation");
    println!("    TS-001 pow_f64: fractional exponents → fixed for ALL Springs");
    println!("    TS-003 acos precision: trig boundary values → fixed for ALL Springs");
    println!("    TS-004 reduce buffer: N≥1024 dispatch → stabilized for ALL Springs");
    println!("    Richards PDE: airSpring validated, absorbed into barracuda (S40)");
    println!("    metalForge (S66): regression, hydrology, moving_window_f64");
    println!("    └→ all Springs: unsaturated flow, sensor correction, crop water, stream stats");
    println!();
    println!("  774 WGSL shaders (BarraCuda S68), 46+ cross-spring absorptions,");
    println!("  11 Tier A wired modules in airSpring, zero duplication.");
    println!("  S66 explicit BGL resolved P0 GPU dispatch blocker — GPU-first paths stable.");
    println!("═══════════════════════════════════════════════════════════════════════");
}
