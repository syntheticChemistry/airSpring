// SPDX-License-Identifier: AGPL-3.0-or-later
//! Domain-specific benchmarks — airSpring rewired systems, groundSpring uncertainty, S71 evolution.

use std::time::Instant;

use airspring_barracuda::eco::correction;
use airspring_barracuda::eco::evapotranspiration::{self as et, DailyEt0Input};
use airspring_barracuda::gpu::mc_et0::{mc_et0_cpu, Et0Uncertainties};
use barracuda::validation::ValidationHarness;

/// Shared FAO-56 Example 18 input — Uccle, Belgium, 6 July.
/// Reused across domain benchmarks to avoid duplicating test fixtures.
pub const UCCLE_INPUT: DailyEt0Input = DailyEt0Input {
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

pub fn bench_airspring_rewired(v: &mut ValidationHarness) {
    println!("\n── airSpring Rewired Systems ────────────────────────────────");

    let t0 = Instant::now();

    let x: Vec<f64> = (0..50).map(|i| f64::from(i) * 0.2).collect();
    let y: Vec<f64> = x.iter().map(|&xi| 2.5f64.mul_add(xi, 1.0)).collect();
    let fit = barracuda::stats::regression::fit_linear(&x, &y).expect("linear fit");
    v.check_abs(
        "regression slope [airSpring → S66 absorption]",
        fit.params[0],
        2.5,
        1e-10,
    );
    v.check_abs("regression intercept", fit.params[1], 1.0, 1e-10);

    let tmin = 18.0_f64;
    let tmax = 32.0;
    let ra_mm = 40.0 / 2.45;
    let hg_upstream = barracuda::stats::hydrology::hargreaves_et0(ra_mm, tmax, tmin)
        .expect("Hargreaves upstream");
    let hg_local = et::hargreaves_et0(tmin, tmax, ra_mm);
    v.check_abs(
        "Hargreaves upstream==local [airSpring → S66]",
        hg_upstream,
        hg_local,
        0.01,
    );

    let ridge_x: Vec<f64> = (0..30).map(f64::from).collect();
    let ridge_y: Vec<f64> = ridge_x.iter().map(|&xi| 1.5f64.mul_add(xi, 2.0)).collect();
    let ridge = correction::fit_ridge(&ridge_x, &ridge_y, 1e-10).expect("ridge fit");
    v.check_abs(
        "ridge regression slope [wetSpring ESN → S58]",
        ridge.params[0],
        1.5,
        0.05,
    );

    println!("  airSpring rewired: {:.1?}", t0.elapsed());
}

pub fn bench_groundspring_uncertainty(v: &mut ValidationHarness) {
    println!("\n── groundSpring Uncertainty Lineage ─────────────────────────");

    let t0 = Instant::now();

    let mc = mc_et0_cpu(&UCCLE_INPUT, &Et0Uncertainties::default(), 5000, 42);
    v.check_lower("MC ET₀ σ > 0.05 [groundSpring → S64]", mc.et0_std, 0.05);
    v.check_upper("MC ET₀ σ < 2.0", mc.et0_std, 2.0);
    v.check_lower("MC P05 < central", mc.et0_central - mc.et0_p05, 0.0);
    v.check_lower("MC P95 > central", mc.et0_p95 - mc.et0_central, 0.0);

    let (ci_lo, ci_hi) = mc.parametric_ci(0.90);
    v.check_lower("parametric CI lower < mean", mc.et0_mean - ci_lo, 0.0);
    v.check_lower("parametric CI upper > mean", ci_hi - mc.et0_mean, 0.0);

    let mc2 = mc_et0_cpu(&UCCLE_INPUT, &Et0Uncertainties::default(), 5000, 42);
    v.check_abs(
        "MC ET₀ deterministic (same seed)",
        mc.et0_mean,
        mc2.et0_mean,
        1e-14,
    );

    let ci = barracuda::stats::bootstrap::bootstrap_mean(
        &[3.2, 3.5, 3.1, 3.8, 3.6, 3.3, 3.7, 3.4, 3.9, 3.0],
        1000,
        0.95,
        42,
    )
    .expect("bootstrap");
    v.check_lower(
        "Bootstrap CI lower < upper [hotSpring+groundSpring]",
        ci.upper - ci.lower,
        0.0,
    );

    println!("  groundSpring uncertainty: {:.1?}", t0.elapsed());
}

pub fn bench_s71_upstream_evolution(v: &mut ValidationHarness) {
    println!("\n── ToadStool S71 Upstream Evolution ─────────────────────────");

    let t0 = Instant::now();

    let et0 = barracuda::stats::fao56_et0(21.5, 12.3, 84.0, 63.0, 2.78, 22.07, 100.0, 50.8, 187)
        .expect("fao56_et0 valid inputs");
    v.check_abs(
        "upstream fao56_et0 FAO-56 Example 18 [groundSpring → S70]",
        et0,
        3.88,
        0.15,
    );

    let ea = et::actual_vapour_pressure_rh(12.3, 21.5, 63.0, 84.0);
    let local_result = et::daily_et0(&DailyEt0Input {
        tmin: 12.3,
        tmax: 21.5,
        tmean: None,
        solar_radiation: 22.07,
        wind_speed_2m: 2.78,
        actual_vapour_pressure: ea,
        elevation_m: 100.0,
        latitude_deg: 50.8,
        day_of_year: 187,
    });
    v.check_abs(
        "upstream fao56_et0 ≈ local PM [cross-validation]",
        et0,
        local_result.et0,
        0.15,
    );

    let fix = barracuda::stats::kimura_fixation_prob(1000, 0.01, 0.5);
    v.check_lower(
        "kimura fixation p > 0.5 (s>0) [wetSpring bio → S71]",
        fix,
        0.5,
    );
    v.check_upper("kimura fixation p < 1.0", fix, 1.0);

    let jk =
        barracuda::stats::jackknife_mean_variance(&[1.0, 2.0, 3.0, 4.0, 5.0]).expect("jackknife");
    v.check_abs(
        "jackknife mean of 1..5 = 3.0 [neuralSpring → S70+]",
        jk.estimate,
        3.0,
        1e-12,
    );
    v.check_lower("jackknife variance > 0", jk.variance, 0.0);

    let ci = barracuda::stats::bootstrap_ci(
        &[2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
        barracuda::stats::mean,
        1000,
        0.95,
        42,
    )
    .expect("bootstrap_ci");
    v.check_lower("bootstrap CI lower < upper [S64]", ci.upper - ci.lower, 0.0);
    v.check_abs("bootstrap mean ≈ 6.5", ci.estimate, 6.5, 0.5);

    let hist_data: Vec<f64> = (0..100).map(|i| f64::from(i) * 0.01).collect();
    let p50 = barracuda::stats::percentile(&hist_data, 50.0);
    v.check_abs("percentile(50) of uniform [0,1) [S64]", p50, 0.5, 0.05);

    println!("  S71 upstream evolution: {:.1?}", t0.elapsed());
}
