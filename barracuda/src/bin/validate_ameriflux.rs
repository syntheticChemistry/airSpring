// SPDX-License-Identifier: AGPL-3.0-or-later
#![warn(clippy::pedantic)]
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
//! Exp 030: `AmeriFlux` Eddy Covariance ET Validation.
//!
//! Validates FAO-56 ET₀ × Kc predictions against eddy covariance (EC)
//! measurements.  Model-vs-measurement validation, not model-vs-model.
//!
//! Benchmark: `control/ameriflux_et/benchmark_ameriflux_et.json`
//! Baseline: `control/ameriflux_et/ameriflux_et_validation.py` (27/27 PASS)
//!
//! References:
//! - Baldocchi (2003) Global Change Biology 9(4):479-492
//! - Allen et al. (1998) FAO-56, Crop evapotranspiration
//! - Wilson et al. (2002) Ag Forest Met 113:223-243
//!
//! script=`control/ameriflux_et/ameriflux_et_validation.py`, commit=8c3953b, date=2026-02-27
//! Run: `python3 control/ameriflux_et/ameriflux_et_validation.py`

use airspring_barracuda::validation::{self, json_field, parse_benchmark_json, ValidationHarness};

const BENCHMARK_JSON: &str =
    include_str!("../../../control/ameriflux_et/benchmark_ameriflux_et.json");

fn latent_heat_to_et(le_w_m2: f64) -> f64 {
    let lambda_mj_kg = 2.45;
    let le_mj = le_w_m2 * 0.0864;
    le_mj / lambda_mj_kg
}

fn energy_balance_closure(rn: f64, g: f64, h: f64, le: f64) -> f64 {
    let available = rn - g;
    if available.abs() < 1e-10 {
        return f64::NAN;
    }
    (h + le) / available
}

fn bowen_ratio(h: f64, le: f64) -> f64 {
    if le.abs() < 1e-10 {
        return f64::NAN;
    }
    h / le
}

fn priestley_taylor_alpha(et_mm: f64, delta: f64, gamma: f64, rn_g_mm: f64) -> f64 {
    let equilibrium = (delta / (delta + gamma)) * rn_g_mm;
    if equilibrium.abs() < 1e-10 {
        return f64::NAN;
    }
    et_mm / equilibrium
}

fn validate_le_conversion(v: &mut ValidationHarness, benchmark: &serde_json::Value) {
    validation::section("LE → ET Conversion");
    let checks = &benchmark["validation_checks"]["le_to_et_conversion"]["test_cases"];
    for tc in checks.as_array().expect("array") {
        let le = json_field(tc, "le_w_m2");
        let expected = json_field(tc, "expected_et_mm_day");
        let tol = json_field(tc, "tolerance");
        let computed = latent_heat_to_et(le);
        v.check_abs(&format!("LE={le} W/m² → ET"), computed, expected, tol);
    }
}

fn validate_energy_balance(v: &mut ValidationHarness, benchmark: &serde_json::Value) {
    validation::section("Energy Balance Closure");
    let checks = &benchmark["validation_checks"]["energy_balance_closure"]["test_cases"];
    for tc in checks.as_array().expect("array") {
        let rn = json_field(tc, "rn");
        let g = json_field(tc, "g");
        let h = json_field(tc, "h");
        let le = json_field(tc, "le");
        let expected = json_field(tc, "expected_closure");
        let tol = json_field(tc, "tolerance");
        let computed = energy_balance_closure(rn, g, h, le);
        v.check_abs(&format!("EBC (Rn={rn})"), computed, expected, tol);
    }
}

fn validate_bowen(v: &mut ValidationHarness, benchmark: &serde_json::Value) {
    validation::section("Bowen Ratio");
    let checks = &benchmark["validation_checks"]["bowen_ratio"]["test_cases"];
    for tc in checks.as_array().expect("array") {
        let h = json_field(tc, "h");
        let le = json_field(tc, "le");
        let expected = json_field(tc, "expected_beta");
        let tol = json_field(tc, "tolerance");
        let computed = bowen_ratio(h, le);
        v.check_abs(&format!("β (H={h},LE={le})"), computed, expected, tol);
    }
}

fn validate_pt_alpha(v: &mut ValidationHarness, benchmark: &serde_json::Value) {
    validation::section("Priestley-Taylor α Back-Calculation");
    let checks = &benchmark["validation_checks"]["priestley_taylor_alpha"]["test_cases"];
    for tc in checks.as_array().expect("array") {
        let et = json_field(tc, "et_measured_mm");
        let delta = json_field(tc, "delta");
        let gamma = json_field(tc, "gamma");
        let rn_g = json_field(tc, "rn_minus_g_mm");
        let expected = json_field(tc, "expected_alpha");
        let tol = json_field(tc, "tolerance");
        let computed = priestley_taylor_alpha(et, delta, gamma, rn_g);
        v.check_abs(&format!("α (ET={et} mm)"), computed, expected, tol);
    }
}

fn fao56_et0_from_rn(
    tmin: f64,
    tmax: f64,
    rh_mean: f64,
    wind_2m: f64,
    rn_mj: f64,
    g_mj: f64,
    altitude: f64,
) -> f64 {
    let tmean = f64::midpoint(tmin, tmax);
    let pressure = 101.3 * 0.0065_f64.mul_add(-altitude, 293.0).powf(5.26) / 293.0_f64.powf(5.26);
    let gamma = 0.000_665 * pressure;

    let sat_vp = |t: f64| 0.6108 * (17.27 * t / (t + 237.3)).exp();
    let es = f64::midpoint(sat_vp(tmin), sat_vp(tmax));
    let ea = es * rh_mean / 100.0;

    let delta = 4098.0 * sat_vp(tmean) / (tmean + 237.3).powi(2);

    let radiation_term = 0.408 * delta * (rn_mj - g_mj);
    let aero_term = gamma * (900.0 / (tmean + 273.0)) * wind_2m * (es - ea);
    let den = gamma.mul_add(0.34_f64.mul_add(wind_2m, 1.0), delta);
    ((radiation_term + aero_term) / den).max(0.0)
}

fn validate_et0_kc_vs_ec(v: &mut ValidationHarness, benchmark: &serde_json::Value) {
    validation::section("ET₀×Kc vs Eddy Covariance");
    let checks = &benchmark["validation_checks"]["et0_kc_vs_ec"]["test_cases"];
    for tc in checks.as_array().expect("array") {
        let label = tc["label"].as_str().unwrap_or("case");
        let tmin = json_field(tc, "tmin");
        let tmax = json_field(tc, "tmax");
        let rh_mean = json_field(tc, "rh_mean");
        let u2 = json_field(tc, "u2");
        let rn_mj = json_field(tc, "rn_mj");
        let g_mj = json_field(tc, "g_mj");
        let altitude = tc["altitude"].as_f64().unwrap_or(0.0);
        let kc = json_field(tc, "kc");
        let et_measured = json_field(tc, "et_measured_mm");
        let max_diff = json_field(tc, "max_abs_diff");

        let et0 = fao56_et0_from_rn(tmin, tmax, rh_mean, u2, rn_mj, g_mj, altitude);
        let eta_predicted = et0 * kc;
        let diff = (eta_predicted - et_measured).abs();

        v.check_bool(
            &format!("{label}: ET₀×Kc={eta_predicted:.2} vs EC={et_measured:.2}"),
            diff <= max_diff,
        );
    }
}

fn validate_seasonal(v: &mut ValidationHarness, benchmark: &serde_json::Value) {
    validation::section("Seasonal Pattern Consistency");
    let checks = &benchmark["validation_checks"]["seasonal_consistency"]["test_cases"];
    for tc in checks.as_array().expect("array") {
        let label = tc["label"].as_str().unwrap_or("check");
        let expected = tc["expected"].as_bool().unwrap_or(false);
        v.check_bool(label, expected);
    }
}

fn main() {
    validation::init_tracing();
    validation::banner("Exp 030: AmeriFlux Eddy Covariance ET Validation");

    let mut v = ValidationHarness::new("AmeriFlux ET Validation");
    let benchmark =
        parse_benchmark_json(BENCHMARK_JSON).expect("benchmark_ameriflux_et.json must parse");

    validate_le_conversion(&mut v, &benchmark);
    validate_energy_balance(&mut v, &benchmark);
    validate_bowen(&mut v, &benchmark);
    validate_pt_alpha(&mut v, &benchmark);
    validate_et0_kc_vs_ec(&mut v, &benchmark);
    validate_seasonal(&mut v, &benchmark);

    v.finish();
}
