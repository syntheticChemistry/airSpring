// SPDX-License-Identifier: AGPL-3.0-or-later
//! Cross-spring throughput benchmarks — sanity checks that CPU performance
//! is reasonable for the cross-spring primitive composition.
//!
//! These are not micro-benchmarks (use `criterion` for that). They verify
//! that the cross-spring shader evolution pipeline composes without
//! unexpected overhead, at rough order-of-magnitude timescales.

use std::time::Instant;

// ── §6 — Cross-spring benchmark (CPU timing sanity checks) ────────────

#[test]
fn benchmark_et0_throughput_reasonable() {
    use airspring_barracuda::eco::evapotranspiration::{daily_et0, DailyEt0Input};

    let inputs: Vec<DailyEt0Input> = (0..1000)
        .map(|i| {
            let d = f64::from(i);
            DailyEt0Input {
                tmin: 12.0 + (d * 0.01).sin(),
                tmax: 25.0 + (d * 0.01).cos(),
                tmean: None,
                solar_radiation: 22.0,
                wind_speed_2m: 2.0,
                actual_vapour_pressure: 1.4,
                elevation_m: 100.0,
                latitude_deg: 50.8,
                day_of_year: 187,
            }
        })
        .collect();

    let start = Instant::now();
    let total: f64 = inputs.iter().map(|i| daily_et0(i).et0).sum();
    let elapsed = start.elapsed();

    assert!(
        total > 0.0,
        "1000 ET₀ computations should produce positive total"
    );
    assert!(
        elapsed.as_millis() < 500,
        "1000 ET₀ CPU evaluations should complete in <500ms; \
         hotSpring math_f64 primitives power the Tetens/radiation chain"
    );
}

#[test]
fn benchmark_richards_throughput_reasonable() {
    use airspring_barracuda::eco::richards::{solve_richards_1d, VanGenuchtenParams};

    let sand = VanGenuchtenParams {
        theta_r: 0.045,
        theta_s: 0.43,
        alpha: 0.145,
        n_vg: 2.68,
        ks: 712.8,
    };

    let start = Instant::now();
    let result = solve_richards_1d(&sand, 30.0, 20, -20.0, 0.0, true, false, 0.1, 0.01);
    let elapsed = start.elapsed();

    assert!(result.is_ok(), "Richards must converge");
    assert!(
        elapsed.as_millis() < 2000,
        "Richards 20-node solve should complete in <2s; \
         airSpring contributed this solver upstream (ToadStool S40)"
    );
}

#[test]
fn benchmark_isotherm_nm_throughput_reasonable() {
    use airspring_barracuda::gpu::isotherm::fit_langmuir_nm;

    let ce = [1.0, 2.5, 5.0, 10.0, 20.0, 40.0, 60.0, 80.0, 100.0];
    let qe = [0.85, 1.92, 3.45, 5.8, 8.9, 12.1, 13.8, 14.5, 14.9];

    let start = Instant::now();
    for _ in 0..100 {
        let _ = fit_langmuir_nm(&ce, &qe);
    }
    let elapsed = start.elapsed();

    assert!(
        elapsed.as_millis() < 5000,
        "100 NM isotherm fits should complete in <5s; \
         neuralSpring optimize::nelder_mead powers the simplex search"
    );
}

// ── §10 — Cross-Spring Benchmark: Modern System Throughput ────────────

#[test]
fn benchmark_diversity_throughput() {
    use airspring_barracuda::eco::diversity;

    let counts: Vec<f64> = (1..=100).map(|i| f64::from(i) * 1.5).collect();

    let start = Instant::now();
    for _ in 0..10_000 {
        let _ = diversity::alpha_diversity(&counts);
    }
    let elapsed = start.elapsed();

    assert!(
        elapsed.as_millis() < 3000,
        "10K alpha diversity computations (100 species) should complete in <3s; \
         wetSpring bio/diversity.rs absorbed in S64, took {elapsed:?}"
    );
}

#[test]
fn benchmark_mc_et0_throughput() {
    use airspring_barracuda::eco::evapotranspiration::DailyEt0Input;
    use airspring_barracuda::gpu::mc_et0::{mc_et0_cpu, Et0Uncertainties};

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

    let start = Instant::now();
    let result = mc_et0_cpu(&input, &Et0Uncertainties::default(), 10_000, 42);
    let elapsed = start.elapsed();

    assert!(
        elapsed.as_millis() < 5000,
        "10K MC ET₀ samples should complete in <5s (CPU mirror of \
         groundSpring mc_et0_propagate_f64.wgsl); took {elapsed:?}"
    );
    assert_eq!(result.n_samples, 10_000);
}

#[test]
fn benchmark_stats_reexport_throughput() {
    use airspring_barracuda::testutil;

    let a: Vec<f64> = (0..10_000).map(f64::from).collect();
    let b: Vec<f64> = (0..10_000).map(|i| f64::from(i) + 0.1).collect();

    let start = Instant::now();
    for _ in 0..1_000 {
        let _ = testutil::rmse(&a, &b);
        let _ = testutil::mbe(&a, &b);
        let _ = testutil::dot(&a, &b);
        let _ = testutil::l2_norm(&a);
    }
    let elapsed = start.elapsed();

    assert!(
        elapsed.as_millis() < 5000,
        "4K metric computations (10K-element vectors) should complete in <5s; \
         upstream delegation (S64) should not add overhead; took {elapsed:?}"
    );
}

// ── §12 — S66 Throughput: Regression ──────────────────────────────────

#[test]
fn benchmark_s66_regression_throughput() {
    let x: Vec<f64> = (0..50).map(f64::from).collect();
    let y: Vec<f64> = x
        .iter()
        .map(|&xi| 2.0f64.mul_add(xi, 1.0) + (xi * 0.01).sin())
        .collect();

    let start = Instant::now();
    for _ in 0..10_000 {
        let _ = barracuda::stats::regression::fit_linear(&x, &y);
    }
    let elapsed = start.elapsed();

    assert!(
        elapsed.as_millis() < 5000,
        "10K fit_linear (50 points) should complete in <5s; \
         metalForge regression absorbed upstream (R-S66-001); took {elapsed:?}"
    );
}

// ── §14 — S68 cross-spring throughput benchmarks ──────────────────────
//
// Verify that S68's precision refactor does not introduce performance
// regressions in the primitives airSpring depends on.

#[test]
fn benchmark_s68_atlas_pipeline_composition() {
    use airspring_barracuda::eco::{
        evapotranspiration::{self as et, DailyEt0Input},
        yield_response,
    };

    let crops = ["corn", "soybean", "winter_wheat", "sugar_beet", "dry_bean"];

    let start = Instant::now();
    let mut total_yield = 0.0_f64;

    for station in 0..200 {
        let lat = f64::from(station).mul_add(0.05, 42.0);
        let elev = f64::from(station).mul_add(2.0, 200.0);

        for doy in 120u32..270 {
            let phase = f64::from(doy) / 365.0 * std::f64::consts::TAU;
            let input = DailyEt0Input {
                tmin: 5.0f64.mul_add(phase.sin(), 10.0),
                tmax: 5.0f64.mul_add(phase.sin(), 25.0),
                tmean: None,
                solar_radiation: 4.0f64.mul_add(phase.sin(), 18.0),
                wind_speed_2m: 2.0,
                actual_vapour_pressure: 1.2,
                elevation_m: elev,
                latitude_deg: lat,
                day_of_year: doy,
            };
            let _et0 = et::daily_et0(&input);
        }

        for crop_name in &crops {
            if let Some(ky) = yield_response::ky_table(crop_name) {
                let yr = yield_response::yield_ratio_single(ky.ky_total, 0.85);
                total_yield += yr;
            }
        }
    }
    let elapsed = start.elapsed();

    assert!(
        total_yield > 0.0,
        "S68: Atlas pipeline produced valid yields"
    );
    assert!(
        elapsed.as_millis() < 5000,
        "S68: 200 stations × 150 days ET₀ + 1000 crop yields in <5s; \
         composition of hotSpring precision → wetSpring stats → airSpring eco → \
         groundSpring uncertainty; took {elapsed:?}"
    );
}

#[test]
fn benchmark_s68_richards_pde_cross_spring() {
    use airspring_barracuda::eco::richards::{solve_richards_1d, VanGenuchtenParams};

    let soil = VanGenuchtenParams {
        theta_r: 0.065,
        theta_s: 0.41,
        alpha: 0.075,
        n_vg: 1.89,
        ks: 106.1,
    };

    let start = Instant::now();
    let mut count = 0;
    for _ in 0..10 {
        let profiles = solve_richards_1d(&soil, 100.0, 50, -100.0, -75.0, false, true, 0.1, 0.001);
        assert!(profiles.is_ok(), "Richards must converge");
        count += 1;
    }
    let elapsed = start.elapsed();

    assert_eq!(count, 10);
    assert!(
        elapsed.as_millis() < 10_000,
        "S68: 10× Richards 1D (50 nodes, 0.1d) in <10s (debug build); \
         airSpring PDE → ToadStool S40 → Crank-Nicolson cross-val; took {elapsed:?}"
    );
}
