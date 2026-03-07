// SPDX-License-Identifier: AGPL-3.0-or-later
#![warn(clippy::pedantic)]
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::similar_names
)]
//! Exp 079: Monte Carlo ET₀ Uncertainty Propagation.
//!
//! Validates CPU Monte Carlo propagation of measurement uncertainties through
//! the FAO-56 Penman-Monteith equation. Tests determinism (same seed → same
//! result), spread monotonicity (higher σ → wider CI), convergence (std
//! stabilises with N), and cross-climate consistency (arid > humid ET₀).
//!
//! Benchmark: `control/mc_et0/benchmark_mc_et0.json` (12/12 Python PASS)
//! Baseline: `control/mc_et0/mc_et0_propagation.py`
//!
//! References:
//! - Allen RG et al. (1998) FAO-56 Crop Evapotranspiration
//! - Gong L et al. (2006) Sensitivity of Penman-Monteith ET₀
//! - groundSpring Exp 003: Humidity dominates ET₀ uncertainty at 66%
//!
//! Provenance: script=`control/mc_et0/mc_et0_propagation.py`, commit=1c11763, date=2026-03-07

use airspring_barracuda::eco::evapotranspiration::DailyEt0Input;
use airspring_barracuda::gpu::mc_et0::{mc_et0_cpu, Et0Uncertainties};
use airspring_barracuda::tolerances;
use airspring_barracuda::validation::{
    self, json_f64_required, parse_benchmark_json, ValidationHarness,
};

const BENCHMARK_JSON: &str = include_str!("../../../control/mc_et0/benchmark_mc_et0.json");

const fn uccle_input() -> DailyEt0Input {
    DailyEt0Input {
        tmin: 12.3,
        tmax: 21.5,
        tmean: Some(16.9),
        solar_radiation: 22.07,
        wind_speed_2m: 2.078,
        actual_vapour_pressure: 1.409,
        elevation_m: 100.0,
        latitude_deg: 50.80,
        day_of_year: 187,
    }
}

const fn arid_input() -> DailyEt0Input {
    DailyEt0Input {
        tmin: 28.0,
        tmax: 42.0,
        tmean: Some(35.0),
        solar_radiation: 28.0,
        wind_speed_2m: 1.5,
        actual_vapour_pressure: 0.8,
        elevation_m: 340.0,
        latitude_deg: 33.45,
        day_of_year: 200,
    }
}

const fn humid_input() -> DailyEt0Input {
    DailyEt0Input {
        tmin: 18.0,
        tmax: 29.0,
        tmean: Some(23.5),
        solar_radiation: 20.0,
        wind_speed_2m: 2.5,
        actual_vapour_pressure: 2.0,
        elevation_m: 256.0,
        latitude_deg: 42.73,
        day_of_year: 182,
    }
}

fn validate_default_uncertainty(v: &mut ValidationHarness, benchmark: &serde_json::Value) {
    validation::section("Default uncertainty (N=2000, seed=42)");

    let result = mc_et0_cpu(&uccle_input(), &Et0Uncertainties::default(), 2000, 42);

    let py_central = json_f64_required(benchmark, &["test_default_n2000", "et0_central"]);
    let py_mean = json_f64_required(benchmark, &["test_default_n2000", "et0_mean"]);
    let py_std = json_f64_required(benchmark, &["test_default_n2000", "et0_std"]);
    let py_p05 = json_f64_required(benchmark, &["test_default_n2000", "et0_p05"]);
    let py_p95 = json_f64_required(benchmark, &["test_default_n2000", "et0_p95"]);

    v.check_abs(
        "central ET₀ matches Python",
        result.et0_central,
        py_central,
        tolerances::ET0_REFERENCE.abs_tol,
    );

    v.check_abs(
        "central ET₀ plausible (2-6 mm)",
        result.et0_central,
        4.0,
        2.0,
    );

    v.check_abs(
        "MC mean near Python mean",
        result.et0_mean,
        py_mean,
        tolerances::MC_ET0_PROPAGATION.abs_tol,
    );

    v.check_abs(
        "MC std near Python std",
        result.et0_std,
        py_std,
        tolerances::MC_ET0_PROPAGATION.abs_tol,
    );

    v.check_abs(
        "p05 near Python p05",
        result.et0_p05,
        py_p05,
        tolerances::MC_ET0_PROPAGATION.abs_tol,
    );

    v.check_abs(
        "p95 near Python p95",
        result.et0_p95,
        py_p95,
        tolerances::MC_ET0_PROPAGATION.abs_tol,
    );

    let ci_width = result.et0_p95 - result.et0_p05;
    v.check_abs("90% CI width > 0.1 mm", ci_width, 0.5, 0.4);

    println!(
        "  Rust: central={:.6}, mean={:.6}, std={:.6}, CI90=[{:.4}, {:.4}]",
        result.et0_central, result.et0_mean, result.et0_std, result.et0_p05, result.et0_p95
    );
}

fn validate_zero_uncertainty(v: &mut ValidationHarness, benchmark: &serde_json::Value) {
    validation::section("Zero uncertainty → zero spread");

    let zero_unc = Et0Uncertainties {
        sigma_tmax: 0.0,
        sigma_tmin: 0.0,
        sigma_rh_max: 0.0,
        sigma_rh_min: 0.0,
        sigma_wind_frac: 0.0,
        sigma_rs_frac: 0.0,
    };
    let result = mc_et0_cpu(&uccle_input(), &zero_unc, 500, 42);

    let py_central = json_f64_required(benchmark, &["test_zero_uncertainty", "et0_central"]);

    v.check_abs(
        "zero unc central matches",
        result.et0_central,
        py_central,
        tolerances::ET0_REFERENCE.abs_tol,
    );

    v.check_abs("zero unc std ≈ 0", result.et0_std, 0.0, 0.01);
}

fn validate_high_uncertainty(v: &mut ValidationHarness, benchmark: &serde_json::Value) {
    validation::section("High uncertainty (wider spread)");

    let high_unc = Et0Uncertainties {
        sigma_tmax: 1.0,
        sigma_tmin: 1.0,
        sigma_rh_max: 10.0,
        sigma_rh_min: 10.0,
        sigma_wind_frac: 0.15,
        sigma_rs_frac: 0.15,
    };
    let result_high = mc_et0_cpu(&uccle_input(), &high_unc, 2000, 42);
    let result_default = mc_et0_cpu(&uccle_input(), &Et0Uncertainties::default(), 2000, 42);

    let py_std_high = json_f64_required(benchmark, &["test_high_uncertainty", "et0_std"]);

    v.check_abs(
        "high unc std near Python",
        result_high.et0_std,
        py_std_high,
        tolerances::MC_ET0_PROPAGATION.abs_tol,
    );

    let wider = if result_high.et0_std > result_default.et0_std {
        1.0
    } else {
        0.0
    };
    v.check_abs("high unc wider than default", wider, 1.0, f64::EPSILON);

    println!(
        "  Rust: high_std={:.6} > default_std={:.6}",
        result_high.et0_std, result_default.et0_std
    );
}

fn validate_climate_gradient(v: &mut ValidationHarness, benchmark: &serde_json::Value) {
    validation::section("Climate gradient (arid vs humid)");

    let unc = Et0Uncertainties::default();
    let arid = mc_et0_cpu(&arid_input(), &unc, 2000, 42);
    let humid = mc_et0_cpu(&humid_input(), &unc, 2000, 42);

    let py_arid_central = json_f64_required(benchmark, &["test_arid_climate", "et0_central"]);
    let py_humid_central = json_f64_required(benchmark, &["test_humid_climate", "et0_central"]);

    v.check_abs(
        "arid central matches Python",
        arid.et0_central,
        py_arid_central,
        tolerances::ET0_REFERENCE.abs_tol,
    );

    v.check_abs(
        "humid central matches Python",
        humid.et0_central,
        py_humid_central,
        tolerances::ET0_REFERENCE.abs_tol,
    );

    let arid_higher = if arid.et0_central > humid.et0_central {
        1.0
    } else {
        0.0
    };
    v.check_abs("arid ET₀ > humid ET₀", arid_higher, 1.0, f64::EPSILON);

    v.check_abs("arid ET₀ plausible (5-12 mm)", arid.et0_central, 8.5, 3.5);

    v.check_abs("humid ET₀ plausible (3-7 mm)", humid.et0_central, 5.0, 2.0);

    println!(
        "  Rust: arid={:.4} mm, humid={:.4} mm",
        arid.et0_central, humid.et0_central
    );
}

fn validate_determinism(v: &mut ValidationHarness) {
    validation::section("Determinism (same seed → same result)");

    let unc = Et0Uncertainties::default();
    let r1 = mc_et0_cpu(&uccle_input(), &unc, 1000, 42);
    let r2 = mc_et0_cpu(&uccle_input(), &unc, 1000, 42);

    let identical = if (r1.et0_mean - r2.et0_mean).abs() < f64::EPSILON
        && (r1.et0_std - r2.et0_std).abs() < f64::EPSILON
    {
        1.0
    } else {
        0.0
    };
    v.check_abs("rerun-identical (mean + std)", identical, 1.0, f64::EPSILON);

    let r3 = mc_et0_cpu(&uccle_input(), &unc, 1000, 99);
    let different = if (r1.et0_mean - r3.et0_mean).abs() > 1e-6 {
        1.0
    } else {
        0.0
    };
    v.check_abs(
        "different seed → different result",
        different,
        1.0,
        f64::EPSILON,
    );
}

fn validate_convergence(v: &mut ValidationHarness) {
    validation::section("Convergence (std stabilises with N)");

    let unc = Et0Uncertainties::default();
    let input = uccle_input();

    let stds: Vec<f64> = [100, 500, 1000, 2000, 5000]
        .iter()
        .map(|&n| mc_et0_cpu(&input, &unc, n, 42).et0_std)
        .collect();

    let last = stds[stds.len() - 1];
    let prev = stds[stds.len() - 2];
    let stability = (last - prev).abs() / prev;

    v.check_abs("last two N within 20%", stability, 0.0, 0.20);

    let monotone_increase = stds.windows(2).all(|w| w[1] >= w[0] * 0.7);
    let mono_val = if monotone_increase { 1.0 } else { 0.0 };
    v.check_abs(
        "std doesn't collapse with more samples",
        mono_val,
        1.0,
        f64::EPSILON,
    );

    println!("  Stds by N: {stds:.6?}");
}

fn validate_parametric_ci(v: &mut ValidationHarness) {
    validation::section("Parametric CI consistency");

    let result = mc_et0_cpu(&uccle_input(), &Et0Uncertainties::default(), 5000, 42);
    let (p_lo, p_hi) = result.parametric_ci(0.90);
    let empirical_width = result.et0_p95 - result.et0_p05;
    let parametric_width = p_hi - p_lo;

    let ratio = parametric_width / empirical_width;
    v.check_abs("parametric/empirical CI ratio near 1.0", ratio, 1.0, 0.4);

    let lo_below = if p_lo < result.et0_mean { 1.0 } else { 0.0 };
    let hi_above = if p_hi > result.et0_mean { 1.0 } else { 0.0 };
    v.check_abs("parametric lower < mean", lo_below, 1.0, f64::EPSILON);
    v.check_abs("parametric upper > mean", hi_above, 1.0, f64::EPSILON);

    let (_, hi_90) = result.parametric_ci(0.90);
    let (_, hi_99) = result.parametric_ci(0.99);
    let wider = if hi_99 > hi_90 { 1.0 } else { 0.0 };
    v.check_abs("99% CI wider than 90%", wider, 1.0, f64::EPSILON);
}

fn validate_sample_count(v: &mut ValidationHarness) {
    validation::section("Sample counts");

    let result = mc_et0_cpu(&uccle_input(), &Et0Uncertainties::default(), 2000, 42);
    #[allow(clippy::cast_precision_loss)]
    let n = result.n_samples as f64;
    v.check_abs("all 2000 samples valid", n, 2000.0, f64::EPSILON);

    let zero = mc_et0_cpu(&uccle_input(), &Et0Uncertainties::default(), 0, 42);
    #[allow(clippy::cast_precision_loss)]
    let n0 = zero.n_samples as f64;
    v.check_abs("zero samples returns 0", n0, 0.0, f64::EPSILON);
}

fn main() {
    validation::init_tracing();
    validation::banner("Exp 079: Monte Carlo ET₀ Uncertainty Propagation");

    let mut v = ValidationHarness::new("MC ET₀");
    let benchmark = parse_benchmark_json(BENCHMARK_JSON).expect("benchmark_mc_et0.json must parse");

    validate_default_uncertainty(&mut v, &benchmark);
    validate_zero_uncertainty(&mut v, &benchmark);
    validate_high_uncertainty(&mut v, &benchmark);
    validate_climate_gradient(&mut v, &benchmark);
    validate_determinism(&mut v);
    validate_convergence(&mut v);
    validate_parametric_ci(&mut v);
    validate_sample_count(&mut v);

    v.finish();
}
