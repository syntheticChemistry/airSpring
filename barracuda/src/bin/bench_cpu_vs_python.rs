// SPDX-License-Identifier: AGPL-3.0-or-later
#![warn(clippy::pedantic)]
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
//! Benchmark Rust CPU vs Python baseline for airSpring core computations.
//!
//! Measures wall-clock time for:
//! 1. FAO-56 PM ET₀ computation (N station-days)
//! 2. Dual Kc simulation (N-day growing seasons)
//! 3. Cover crop + mulch simulation
//! 4. Richards PDE (1D infiltration)
//! 5. Van Genuchten retention (batch)
//! 6. Isotherm fitting (Langmuir + Freundlich)
//! 7. Yield response (Stewart 1977 single + multi-stage)
//! 8. Water use efficiency
//! 9. Season yield + water balance integration
//! 10. CW2D Richards (gravel + organic media)
//! 11. Scheduling pipeline: ET₀→Kc→WB→Stewart yield (Exp 014)
//! 12. Lysimeter ET: mass→ET conversion throughput (Exp 016)
//! 13. Sensitivity OAT: 6-variable perturbation batch (Exp 017)
//!
//! Outputs timing data in a format directly comparable to Python controls.
//! Run with `--release` for production-representative numbers:
//!
//! ```sh
//! cargo run --release --bin bench_cpu_vs_python
//! ```

use airspring_barracuda::eco::dual_kc::{self, DualKcInput, EvaporationLayerState};
use airspring_barracuda::eco::evapotranspiration::{self as et, DailyEt0Input};
use airspring_barracuda::eco::isotherm;
use airspring_barracuda::eco::richards::{self, VanGenuchtenParams};
use airspring_barracuda::eco::water_balance;
use airspring_barracuda::eco::yield_response;
use std::time::Instant;

const WARMUP: usize = 5;
const MEASURE: usize = 20;

fn make_station_days(n: usize) -> Vec<DailyEt0Input> {
    (0..n)
        .map(|i| {
            let day = i as f64;
            let tmax = 5.0f64.mul_add((day * 0.017).sin(), 30.0);
            let tmin = 3.0f64.mul_add((day * 0.017).cos(), 15.0);
            DailyEt0Input {
                tmax,
                tmin,
                tmean: None,
                solar_radiation: 4.0f64.mul_add((day * 0.017).sin(), 18.0),
                wind_speed_2m: 0.5f64.mul_add((day * 0.05).sin(), 2.0),
                actual_vapour_pressure: et::saturation_vapour_pressure(tmin) * 0.6,
                elevation_m: 190.0,
                latitude_deg: 42.5,
                day_of_year: u32::try_from((i % 365) + 1).expect("day_of_year 1..365 fits in u32"),
            }
        })
        .collect()
}

fn make_dual_kc_inputs(n: usize) -> Vec<DualKcInput> {
    (0..n)
        .map(|i| {
            let day = i as f64;
            DualKcInput {
                et0: 2.0f64.mul_add((day * 0.017).sin(), 4.0),
                precipitation: if i % 7 == 0 { 12.0 } else { 0.0 },
                irrigation: 0.0,
            }
        })
        .collect()
}

fn bench<F: Fn()>(label: &str, n: usize, f: F) {
    for _ in 0..WARMUP {
        f();
    }

    let start = Instant::now();
    for _ in 0..MEASURE {
        f();
    }
    let elapsed = start.elapsed();
    let per_iter = elapsed / u32::try_from(MEASURE).expect("MEASURE fits in u32");
    let throughput = n as f64 / per_iter.as_secs_f64();

    println!("  {label:<40} {n:>8} items  {per_iter:>10.2?}/iter  {throughput:>12.0} items/s",);
}

fn bench_et0_and_kc() {
    println!("── FAO-56 PM ET₀ computation ──\n");
    for &n in &[100, 1_000, 10_000, 100_000] {
        let data = make_station_days(n);
        bench(&format!("ET₀ ({n} station-days)"), n, || {
            for d in &data {
                std::hint::black_box(et::daily_et0(d));
            }
        });
    }

    println!("\n── Dual Kc simulation ──\n");
    let state = EvaporationLayerState {
        de: 0.0,
        tew: 22.5,
        rew: 9.0,
    };
    for &n in &[30, 180, 365, 3650] {
        let inputs = make_dual_kc_inputs(n);
        bench(&format!("Dual Kc ({n}-day season)"), n, || {
            std::hint::black_box(dual_kc::simulate_dual_kc(&inputs, 1.15, 1.20, 0.05, &state));
        });
    }

    println!("\n── Dual Kc + mulch (no-till) ──\n");
    for &n in &[30, 180, 365, 3650] {
        let inputs = make_dual_kc_inputs(n);
        bench(&format!("Mulched Kc ({n}-day season)"), n, || {
            std::hint::black_box(dual_kc::simulate_dual_kc_mulched(
                &inputs, 0.15, 1.20, 1.0, 0.40, &state,
            ));
        });
    }

    println!("\n── Scaling: ET₀ batch (1M station-days) ──\n");
    let big_data = make_station_days(1_000_000);
    bench("ET₀ (1M station-days)", 1_000_000, || {
        for d in &big_data {
            std::hint::black_box(et::daily_et0(d));
        }
    });

    println!("\n── Scaling: Dual Kc (10-year season) ──\n");
    let big_inputs = make_dual_kc_inputs(3650);
    bench("Dual Kc (3650-day, 10yr)", 3650, || {
        std::hint::black_box(dual_kc::simulate_dual_kc(
            &big_inputs,
            1.15,
            1.20,
            0.05,
            &state,
        ));
    });
}

fn bench_richards_and_isotherms() {
    println!("\n── Richards equation (1D infiltration) ──\n");
    let sand = VanGenuchtenParams {
        theta_r: 0.045,
        theta_s: 0.43,
        alpha: 0.145,
        n_vg: 2.68,
        ks: 712.8,
    };
    for &n_nodes in &[20, 50, 100] {
        bench(
            &format!("Richards 1D ({n_nodes} nodes, 0.1d)"),
            n_nodes,
            || {
                let _ = std::hint::black_box(richards::solve_richards_1d(
                    &sand, 100.0, n_nodes, -20.0, 0.0, false, true, 0.1, 0.01,
                ));
            },
        );
    }

    println!("\n── Van Genuchten retention (batch) ──\n");
    for &n in &[1_000, 10_000, 100_000] {
        bench(&format!("VG theta ({n} evaluations)"), n, || {
            for i in 0..n {
                let h = -0.01 * (i as f64 + 1.0);
                std::hint::black_box(richards::van_genuchten_theta(
                    h,
                    sand.theta_r,
                    sand.theta_s,
                    sand.alpha,
                    sand.n_vg,
                ));
            }
        });
    }

    println!("\n── Isotherm fitting (Langmuir + Freundlich) ──\n");
    let ce = [1.0, 2.5, 5.0, 10.0, 20.0, 40.0, 60.0, 80.0, 100.0];
    let qe = [0.85, 1.92, 3.45, 5.8, 8.9, 12.1, 13.8, 14.5, 14.9];
    bench("Langmuir fit (9 points)", 9, || {
        std::hint::black_box(isotherm::fit_langmuir(&ce, &qe));
    });
    bench("Freundlich fit (9 points)", 9, || {
        std::hint::black_box(isotherm::fit_freundlich(&ce, &qe));
    });
}

#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn bench_yield_response() {
    println!("\n── Yield response (Stewart 1977, single-stage) ──\n");
    for &n in &[1_000, 10_000, 100_000, 1_000_000] {
        bench(&format!("Yield single ({n} evaluations)"), n, || {
            for i in 0..n {
                let eta_etc = (i as f64 + 1.0) / (n as f64 + 1.0);
                std::hint::black_box(yield_response::yield_ratio_single(1.25, eta_etc));
            }
        });
    }

    println!("\n── Yield response (multi-stage, 4-stage corn) ──\n");
    let corn_stages = [(0.40, 0.9), (1.50, 0.85), (0.50, 0.95), (0.20, 0.98)];
    for &n in &[1_000, 10_000, 100_000] {
        bench(&format!("Yield multi-stage ({n} corn seasons)"), n, || {
            for _ in 0..n {
                std::hint::black_box(yield_response::yield_ratio_multistage(&corn_stages).unwrap());
            }
        });
    }

    println!("\n── Water use efficiency ──\n");
    for &n in &[1_000, 10_000, 100_000] {
        bench(&format!("WUE ({n} calculations)"), n, || {
            for i in 0..n {
                let y = (i as f64).mul_add(0.04, 8000.0);
                let eta = (i as f64).mul_add(0.002, 300.0);
                std::hint::black_box(yield_response::water_use_efficiency(y, eta).unwrap());
            }
        });
    }

    println!("\n── Yield + WB integration (140-day season) ──\n");
    for &n in &[100, 1_000, 10_000] {
        bench(&format!("Season yield ({n} scenarios)"), n, || {
            for i in 0..n {
                let taw = water_balance::total_available_water(0.18, 0.08, 900.0);
                let raw = water_balance::readily_available_water(taw, 0.55);
                let mut dr = 0.0_f64;
                let mut actual_et_sum = 0.0_f64;
                let mut potential_et_sum = 0.0_f64;
                for day in 0..140_usize {
                    let ks = water_balance::stress_coefficient(dr, taw, raw);
                    let et0 = 5.0 + (day as f64 * 0.04).sin();
                    let etc = 1.2 * et0;
                    let eta = ks * etc;
                    let precip = if (day + i) % 5 == 0 { 8.0 } else { 0.0 };
                    dr = (dr - precip + eta).clamp(0.0, taw);
                    actual_et_sum += eta;
                    potential_et_sum += etc;
                }
                let ratio = actual_et_sum / potential_et_sum;
                std::hint::black_box(yield_response::yield_ratio_single(1.25, ratio));
            }
        });
    }
}

fn bench_cw2d() {
    println!("\n── CW2D Richards (gravel, organic) ──\n");
    let gravel = VanGenuchtenParams {
        theta_r: 0.025,
        theta_s: 0.40,
        alpha: 0.100,
        n_vg: 3.00,
        ks: 5000.0,
    };
    let organic = VanGenuchtenParams {
        theta_r: 0.100,
        theta_s: 0.60,
        alpha: 0.050,
        n_vg: 1.50,
        ks: 50.0,
    };
    for &(label, params) in &[("gravel", &gravel), ("organic", &organic)] {
        bench(
            &format!("CW2D Richards {label} (20 nodes, 0.04d)"),
            20,
            || {
                let _ = std::hint::black_box(richards::solve_richards_1d(
                    params, 60.0, 20, -20.0, -20.0, true, true, 0.04, 0.001,
                ));
            },
        );
    }

    println!("\n── CW2D VG retention (gravel + organic batch) ──\n");
    for &n in &[10_000, 100_000] {
        bench(&format!("CW2D VG gravel ({n} evaluations)"), n, || {
            for i in 0..n {
                let h = -0.01 * (i as f64 + 1.0);
                std::hint::black_box(richards::van_genuchten_theta(
                    h,
                    gravel.theta_r,
                    gravel.theta_s,
                    gravel.alpha,
                    gravel.n_vg,
                ));
            }
        });
        bench(&format!("CW2D VG organic ({n} evaluations)"), n, || {
            for i in 0..n {
                let h = -0.01 * (i as f64 + 1.0);
                std::hint::black_box(richards::van_genuchten_theta(
                    h,
                    organic.theta_r,
                    organic.theta_s,
                    organic.alpha,
                    organic.n_vg,
                ));
            }
        });
    }
}

fn bench_scheduling_pipeline() {
    use airspring_barracuda::gpu::water_balance::BatchedWaterBalance;
    println!("\n── Scheduling Pipeline: ET₀→Kc→WB→Stewart yield (Exp 014) ──\n");

    for &n_days in &[90_u32, 180, 365] {
        let inputs = make_station_days(n_days as usize);
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

        bench(
            &format!("Scheduling pipeline ({n_days}-day)"),
            n_days as usize,
            || {
                let et0_vals: Vec<f64> = inputs.iter().map(|inp| et::daily_et0(inp).et0).collect();
                let wb_inputs: Vec<water_balance::DailyInput> = et0_vals
                    .iter()
                    .zip(&kc_values)
                    .enumerate()
                    .map(|(i, (&e, &kc))| water_balance::DailyInput {
                        precipitation: if i % 5 == 0 { 10.0 } else { 0.0 },
                        irrigation: if i % 7 == 0 { 20.0 } else { 0.0 },
                        et0: e,
                        kc,
                    })
                    .collect();
                let engine = BatchedWaterBalance::new(0.30, 0.10, 500.0, 0.5);
                let summary = engine.simulate_season(&wb_inputs);
                let etc_sum: f64 = et0_vals.iter().zip(&kc_values).map(|(e, k)| e * k).sum();
                let ratio = summary.total_actual_et / etc_sum;
                std::hint::black_box(yield_response::yield_ratio_single(1.25, ratio));
            },
        );
    }
}

fn bench_lysimeter_et() {
    println!("\n── Lysimeter ET: mass→ET conversion (Exp 016) ──\n");

    for &n_readings in &[24_usize, 168, 720, 8760] {
        let mass_readings: Vec<(f64, f64)> = (0..n_readings)
            .map(|i| {
                let hour = i as f64;
                let mass = 0.005f64.mul_add((hour * 0.26).sin(), 0.01f64.mul_add(-hour, 50.0));
                (hour, mass)
            })
            .collect();

        bench(
            &format!("Lysimeter ET ({n_readings} readings)"),
            n_readings,
            || {
                let area = 1.0_f64;
                let mut total_et = 0.0;
                for pair in mass_readings.windows(2) {
                    let dm = pair[1].1 - pair[0].1;
                    let et_mm = -dm / area;
                    total_et += et_mm.max(0.0);
                }
                std::hint::black_box(total_et);
            },
        );
    }
}

fn bench_sensitivity_oat() {
    println!("\n── Sensitivity OAT: 6-var × 2-dir ET₀ perturbation (Exp 017) ──\n");

    for &n_sites in &[1_usize, 10, 100, 1_000] {
        let base_inputs: Vec<DailyEt0Input> = (0..n_sites)
            .map(|i| {
                let fi = i as f64;
                DailyEt0Input {
                    tmin: fi.mul_add(0.01, 19.1),
                    tmax: fi.mul_add(0.01, 32.6),
                    tmean: None,
                    solar_radiation: 22.07,
                    wind_speed_2m: 2.078,
                    actual_vapour_pressure: 1.409,
                    elevation_m: 100.0 + fi,
                    latitude_deg: 50.80,
                    day_of_year: 187,
                }
            })
            .collect();

        bench(
            &format!("OAT sensitivity ({n_sites} sites × 12)"),
            n_sites * 12,
            || {
                for inp in &base_inputs {
                    let base_et0 = et::daily_et0(inp).et0;
                    for delta in &[-0.1_f64, 0.1] {
                        let mut p = *inp;
                        p.tmax *= 1.0 + delta;
                        std::hint::black_box(et::daily_et0(&p).et0 - base_et0);
                        p = *inp;
                        p.tmin *= 1.0 + delta;
                        std::hint::black_box(et::daily_et0(&p).et0 - base_et0);
                        p = *inp;
                        p.solar_radiation *= 1.0 + delta;
                        std::hint::black_box(et::daily_et0(&p).et0 - base_et0);
                        p = *inp;
                        p.wind_speed_2m *= 1.0 + delta;
                        std::hint::black_box(et::daily_et0(&p).et0 - base_et0);
                        p = *inp;
                        p.actual_vapour_pressure *= 1.0 + delta;
                        std::hint::black_box(et::daily_et0(&p).et0 - base_et0);
                        p = *inp;
                        p.elevation_m *= 1.0 + delta;
                        std::hint::black_box(et::daily_et0(&p).et0 - base_et0);
                    }
                }
            },
        );
    }
}

fn main() {
    airspring_barracuda::validation::init_tracing();
    println!("═══════════════════════════════════════════════════════════");
    println!("  airSpring CPU Benchmark — Rust vs Python baseline");
    println!("═══════════════════════════════════════════════════════════\n");

    bench_et0_and_kc();
    bench_richards_and_isotherms();
    bench_yield_response();
    bench_cw2d();
    bench_scheduling_pipeline();
    bench_lysimeter_et();
    bench_sensitivity_oat();

    println!();
    println!("Done. All throughput numbers are Rust --release (pure f64 math).");
    println!("Compare against Python baselines via: python3 scripts/bench_python_baselines.py");
}
