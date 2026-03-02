// SPDX-License-Identifier: AGPL-3.0-or-later
#![warn(clippy::pedantic)]
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
//! Exp 031: Hargreaves-Samani (1985) Temperature-Only ET₀ Validation.
//!
//! Standalone validation of the temperature-only ET₀ method — the fallback
//! when humidity, wind, or radiation data are unavailable.
//!
//! Benchmark: `control/hargreaves/benchmark_hargreaves.json`
//! Baseline: `control/hargreaves/hargreaves_samani.py` (24/24 PASS)
//!
//! References:
//! - Hargreaves & Samani (1985) Applied Eng Agric 1(2):96-99
//! - Allen et al. (1998) FAO-56 Eq. 52
//!
//! script=`control/hargreaves/hargreaves_samani.py`, commit=dbfb53a, date=2026-03-02
//! Run: `python3 control/hargreaves/hargreaves_samani.py`

use airspring_barracuda::eco::evapotranspiration::{extraterrestrial_radiation, hargreaves_et0};
use airspring_barracuda::validation::{self, json_field, parse_benchmark_json, ValidationHarness};

const BENCHMARK_JSON: &str = include_str!("../../../control/hargreaves/benchmark_hargreaves.json");

fn validate_analytical(v: &mut ValidationHarness, benchmark: &serde_json::Value) {
    validation::section("Analytical Benchmarks");
    let checks = &benchmark["validation_checks"]["analytical"]["test_cases"];
    for tc in checks.as_array().expect("array") {
        let tmin = json_field(tc, "tmin");
        let tmax = json_field(tc, "tmax");
        let ra = json_field(tc, "ra_mm_day");
        let expected = json_field(tc, "expected_et0");
        let tol = json_field(tc, "tolerance");
        let computed = hargreaves_et0(tmin, tmax, ra);
        v.check_abs(
            &format!("HG(Tmin={tmin},Tmax={tmax},Ra={ra})"),
            computed,
            expected,
            tol,
        );
    }
}

fn validate_ra(v: &mut ValidationHarness, benchmark: &serde_json::Value) {
    validation::section("Extraterrestrial Radiation Ra");
    let checks = &benchmark["validation_checks"]["ra_computation"]["test_cases"];
    for tc in checks.as_array().expect("array") {
        let latitude = json_field(tc, "latitude");
        #[allow(clippy::cast_sign_loss)]
        let doy = json_field(tc, "doy") as u32;
        let expected = json_field(tc, "expected_ra_mm");
        let tol = json_field(tc, "tolerance");
        let computed = extraterrestrial_radiation(latitude.to_radians(), doy) / 2.45;
        v.check_abs(
            &format!("Ra(lat={latitude}°,DOY={doy})"),
            computed,
            expected,
            tol,
        );
    }
}

fn validate_cross_comparison(v: &mut ValidationHarness, benchmark: &serde_json::Value) {
    validation::section("FAO-56 Cross-Comparison (HG vs PM)");
    let checks = &benchmark["validation_checks"]["fao56_cross_comparison"]["test_cases"];
    for tc in checks.as_array().expect("array") {
        let city = tc["city"].as_str().unwrap_or("city");
        let latitude = json_field(tc, "latitude");
        #[allow(clippy::cast_sign_loss)]
        let doy = json_field(tc, "doy") as u32;
        let tmin = json_field(tc, "tmin");
        let tmax = json_field(tc, "tmax");
        let pm_et0 = json_field(tc, "fao56_pm_et0");
        let max_ratio_diff = json_field(tc, "max_ratio_diff");

        let ra_equiv = extraterrestrial_radiation(latitude.to_radians(), doy) / 2.45;
        let hg_et0 = hargreaves_et0(tmin, tmax, ra_equiv);
        let ratio = if pm_et0 > 0.0 {
            hg_et0 / pm_et0
        } else {
            f64::NAN
        };
        let diff = (ratio - 1.0).abs();

        v.check_bool(
            &format!("{city}: HG/PM={ratio:.3} (diff={diff:.3}, max={max_ratio_diff})"),
            diff <= max_ratio_diff,
        );
    }
}

fn validate_edge_cases(v: &mut ValidationHarness, benchmark: &serde_json::Value) {
    validation::section("Edge Cases");
    let checks = &benchmark["validation_checks"]["edge_cases"]["test_cases"];
    for tc in checks.as_array().expect("array") {
        let label = tc["label"].as_str().unwrap_or("edge");
        let tmin = json_field(tc, "tmin");
        let tmax = json_field(tc, "tmax");
        let ra = json_field(tc, "ra_mm_day");
        let computed = hargreaves_et0(tmin, tmax, ra);
        let check = tc["check"].as_str().unwrap_or("");
        let ok = match check {
            "zero" => computed == 0.0,
            "positive" => computed > 0.0,
            _ => false,
        };
        v.check_bool(&format!("{label} (ET₀={computed:.4})"), ok);
    }
}

fn validate_monotonicity(v: &mut ValidationHarness, benchmark: &serde_json::Value) {
    validation::section("Monotonicity");
    let checks = &benchmark["validation_checks"]["monotonicity"]["test_cases"];
    for tc in checks.as_array().expect("array") {
        let label = tc["label"].as_str().unwrap_or("mono");
        let ra = json_field(tc, "ra_mm_day");
        let base = &tc["base"];
        let inc = &tc["increased"];
        let et0_base = hargreaves_et0(json_field(base, "tmin"), json_field(base, "tmax"), ra);
        let et0_inc = hargreaves_et0(json_field(inc, "tmin"), json_field(inc, "tmax"), ra);
        v.check_bool(
            &format!("{label}: {et0_base:.3} < {et0_inc:.3}"),
            et0_inc > et0_base,
        );
    }
}

fn main() {
    validation::init_tracing();
    validation::banner("Exp 031: Hargreaves-Samani Temperature-Only ET₀");

    let mut v = ValidationHarness::new("Hargreaves-Samani ET₀");
    let benchmark =
        parse_benchmark_json(BENCHMARK_JSON).expect("benchmark_hargreaves.json must parse");

    validate_analytical(&mut v, &benchmark);
    validate_ra(&mut v, &benchmark);
    validate_cross_comparison(&mut v, &benchmark);
    validate_edge_cases(&mut v, &benchmark);
    validate_monotonicity(&mut v, &benchmark);

    v.finish();
}
