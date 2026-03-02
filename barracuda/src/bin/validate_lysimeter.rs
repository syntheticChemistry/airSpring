// SPDX-License-Identifier: AGPL-3.0-or-later
#![warn(clippy::pedantic)]
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
//! Validate lysimeter ET measurement pipeline (Exp 016).
//!
//! Benchmark: `control/lysimeter/benchmark_lysimeter.json`
//! Paper: Dong & Hansen (2023) Smart Ag Tech 4:100147
//! Python: `control/lysimeter/lysimeter_et.py`
//!
//! Covers:
//!   1. Mass-to-ET conversion (1 kg / 1 m² = 1 mm)
//!   2. Temperature compensation (thermal drift correction)
//!   3. Data quality filtering (resolution, rain rejection)
//!   4. Load cell calibration (known-mass linear fit)
//!   5. Hourly diurnal ET pattern (sinusoidal, night ≈ 0)
//!   6. Synthetic daily lysimeter vs ET₀ comparison
//!
//! script=`control/lysimeter/lysimeter_et.py`, commit=e651409, date=2026-02-26
//! Run: `python3 control/lysimeter/lysimeter_et.py`

use airspring_barracuda::validation::{
    self, json_field, json_str, parse_benchmark_json, ValidationHarness,
};
use barracuda::stats::{pearson_correlation, regression::fit_linear, rmse};

const BENCHMARK_JSON: &str = include_str!("../../../control/lysimeter/benchmark_lysimeter.json");

/// Convert lysimeter mass change to ET depth (mm).
/// Standard simplification: 1 kg water / 1 m² = 1 mm.
fn mass_to_et_mm(mass_change_kg: f64, area_m2: f64) -> f64 {
    -mass_change_kg / area_m2
}

/// Correct load cell reading for thermal drift.
fn compensate_temperature(mass_raw_kg: f64, temp_c: f64, alpha_g_per_c: f64, t_ref_c: f64) -> f64 {
    let correction_kg = alpha_g_per_c * (temp_c - t_ref_c) / 1000.0;
    mass_raw_kg - correction_kg
}

/// Check if a mass-change reading is valid for ET computation.
fn is_valid_reading(delta_g: f64, resolution_g: f64, rain_threshold_g: f64) -> bool {
    if delta_g.abs() < resolution_g {
        return false;
    }
    if delta_g > rain_threshold_g {
        return false;
    }
    true
}

/// Fraction of daily ET at a given hour (sinusoidal diurnal pattern).
fn hourly_et_fraction(hour: u32) -> f64 {
    if !(6..=18).contains(&hour) {
        return 0.0;
    }
    let frac = std::f64::consts::PI * (f64::from(hour) - 6.0) / 12.0;
    frac.sin().max(0.0)
}

/// Deterministic synthetic daily ET₀ + lysimeter ET for comparison.
fn generate_synthetic_comparison(n_days: usize) -> (Vec<f64>, Vec<f64>) {
    let mut et0 = Vec::with_capacity(n_days);
    let mut et_lys = Vec::with_capacity(n_days);

    for d in 0..n_days {
        let t = d as f64 / n_days as f64;
        let base_et0 = 1.5f64.mul_add((std::f64::consts::PI * t).sin(), 4.5);
        et0.push(base_et0);

        let noise = 0.2 * ((d as f64 * 7.3).sin() + (d as f64 * 3.1).cos()) * 0.5;
        et_lys.push((base_et0 + noise).max(0.0));
    }

    (et0, et_lys)
}

#[allow(clippy::too_many_lines)]
fn main() {
    validation::init_tracing();
    validation::banner("Lysimeter ET Direct Measurement (Exp 016)");
    let mut v = ValidationHarness::new("Lysimeter Validation");
    let benchmark =
        parse_benchmark_json(BENCHMARK_JSON).expect("benchmark_lysimeter.json must parse");

    // ── Mass-to-ET conversion ──
    validation::section("Mass-to-ET Conversion");
    let et_cases = &benchmark["validation_checks"]["mass_to_et_conversion"]["test_cases"];
    for tc in et_cases.as_array().expect("array") {
        let label = json_str(tc, "label");
        let mass_kg = json_field(tc, "mass_change_kg");
        let area = json_field(tc, "area_m2");
        let expected = json_field(tc, "expected_et_mm");
        let tol = json_field(tc, "tolerance");
        let computed = mass_to_et_mm(mass_kg, area);
        v.check_abs(label, computed, expected, tol);
    }

    // ── Temperature compensation ──
    validation::section("Temperature Compensation");
    let tc_params = &benchmark["temperature_compensation"];
    let alpha = json_field(tc_params, "alpha_g_per_c");
    let t_ref = json_field(tc_params, "t_ref_c");
    let temp_cases = &benchmark["validation_checks"]["temperature_compensation"]["test_cases"];
    for tc in temp_cases.as_array().expect("array") {
        let label = json_str(tc, "label");
        let mass_raw = json_field(tc, "mass_raw_kg");
        let temp_c = json_field(tc, "temp_c");
        let expected = json_field(tc, "expected_corr_kg");
        let tol = json_field(tc, "tolerance");
        let computed = compensate_temperature(mass_raw, temp_c, alpha, t_ref);
        v.check_abs(label, computed, expected, tol);
    }

    // ── Data quality filtering ──
    validation::section("Data Quality Filtering");
    let dq_cases = &benchmark["validation_checks"]["data_quality_filter"]["test_cases"];
    for tc in dq_cases.as_array().expect("array") {
        let label = json_str(tc, "label");
        let delta_g = json_field(tc, "delta_g");
        let expected = tc["expected_valid"].as_bool().expect("bool");
        let computed = is_valid_reading(delta_g, 10.0, 500.0);
        v.check_bool(
            &format!("{label}: valid={computed}, expected={expected}"),
            computed == expected,
        );
    }

    // ── Load cell calibration ──
    validation::section("Load Cell Calibration");
    let cal = &benchmark["calibration"];
    let known: Vec<f64> = cal["known_masses_kg"]
        .as_array()
        .expect("array")
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();
    let measured: Vec<f64> = cal["measured_readings_kg"]
        .as_array()
        .expect("array")
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();
    let fit = fit_linear(&known, &measured).expect("calibration fit must succeed");
    let slope = fit.params[0];
    let intercept = fit.params[1];
    let expected_r2 = json_field(cal, "expected_r_squared");
    let tol_r2 = json_field(
        &benchmark["validation_checks"]["calibration_linearity"],
        "tolerance",
    );
    v.check_abs("calibration R²", fit.r_squared, expected_r2, tol_r2);
    v.check_bool("slope ∈ [0.995, 1.005]", (0.995..=1.005).contains(&slope));
    v.check_bool(
        "intercept ∈ [-0.02, 0.02]",
        (-0.02..=0.02).contains(&intercept),
    );

    // ── Hourly diurnal pattern ──
    validation::section("Hourly Diurnal ET Pattern");
    let hr_cases = &benchmark["validation_checks"]["hourly_et_pattern"]["test_cases"];
    for tc in hr_cases.as_array().expect("array") {
        let label = json_str(tc, "label");
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let hour = json_field(tc, "hour") as u32;
        let frac = hourly_et_fraction(hour);
        if tc
            .get("expected_low")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false)
        {
            v.check_bool(&format!("{label}: frac({hour})={frac:.4} ≈ 0"), frac < 0.05);
        } else if tc
            .get("expected_peak")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false)
        {
            v.check_bool(
                &format!("{label}: frac({hour})={frac:.4} is peak"),
                frac > 0.95,
            );
        } else {
            v.check_bool(&format!("{label}: frac({hour})={frac:.4} > 0"), frac > 0.1);
        }
    }

    // ── Synthetic daily comparison ──
    validation::section("Synthetic Daily Comparison (deterministic)");
    let (et0, et_lys) = generate_synthetic_comparison(30);
    let r = pearson_correlation(&et0, &et_lys).unwrap_or(0.0);
    let rms = rmse(&et0, &et_lys);
    v.check_bool(&format!("correlation r={r:.4} >= 0.80"), r >= 0.80);
    v.check_bool(&format!("RMSE={rms:.4} <= 1.0"), rms <= 1.0);

    v.finish();
}
