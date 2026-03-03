// SPDX-License-Identifier: AGPL-3.0-or-later
#![warn(clippy::pedantic)]
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
//! Exp 035: Hamon (1961) Temperature-Based PET Validation.
//!
//! Validates the minimal-data Hamon PET method against analytical benchmarks
//! and daylight hour computations from FAO-56 solar geometry.
//!
//! Benchmark: `control/hamon/benchmark_hamon.json`
//! Baseline: `control/hamon/hamon_pet.py` (20/20 PASS)
//!
//! References:
//! - Hamon WR (1961) J Hydraulics Div ASCE 87(HY3):107-120
//! - Lu J, et al. (2005) J Am Water Resour Assoc 41(3):621-633
//!
//! Provenance: script=`control/hamon/hamon_pet.py`, commit=d3ecdc8, date=2026-02-27

use airspring_barracuda::eco::evapotranspiration::{daylight_hours, hamon_pet};
use airspring_barracuda::validation::{self, json_field, parse_benchmark_json, ValidationHarness};

const BENCHMARK_JSON: &str = include_str!("../../../control/hamon/benchmark_hamon.json");

fn validate_analytical(v: &mut ValidationHarness, benchmark: &serde_json::Value) {
    validation::section("Analytical Benchmarks");
    let checks = &benchmark["validation_checks"]["analytical"]["test_cases"];
    for tc in checks.as_array().expect("array") {
        let tmean = json_field(tc, "tmean");
        let dl = json_field(tc, "day_length_hours");
        let expected = json_field(tc, "expected_pet");
        let tol = json_field(tc, "tolerance");
        let computed = hamon_pet(tmean, dl);
        v.check_abs(
            &format!("Hamon(T={tmean},N={dl}h)"),
            computed,
            expected,
            tol,
        );
    }
}

fn validate_day_length(v: &mut ValidationHarness, benchmark: &serde_json::Value) {
    validation::section("Day Length Computation");
    let checks = &benchmark["validation_checks"]["day_length_computation"]["test_cases"];
    for tc in checks.as_array().expect("array") {
        let lat = json_field(tc, "latitude");
        #[expect(
            clippy::cast_sign_loss,
            reason = "DOY from JSON f64 is a non-negative integer"
        )]
        let doy = json_field(tc, "doy") as u32;
        let expected = json_field(tc, "expected_hours");
        let tol = json_field(tc, "tolerance");
        let computed = daylight_hours(lat.to_radians(), doy);
        v.check_abs(&format!("N(lat={lat}°,DOY={doy})"), computed, expected, tol);
    }
}

fn validate_edge_cases(v: &mut ValidationHarness, benchmark: &serde_json::Value) {
    validation::section("Edge Cases");
    let checks = &benchmark["validation_checks"]["edge_cases"]["test_cases"];
    for tc in checks.as_array().expect("array") {
        let label = tc["label"].as_str().unwrap_or("edge");
        let tmean = json_field(tc, "tmean");
        let dl = json_field(tc, "day_length_hours");
        let computed = hamon_pet(tmean, dl);
        let check = tc["check"].as_str().unwrap_or("");
        let ok = match check {
            "positive" => computed > 0.0,
            "non_negative" => computed >= 0.0,
            "zero" => computed == 0.0,
            _ => false,
        };
        v.check_bool(&format!("{label} (PET={computed:.4})"), ok);
    }
}

fn validate_monotonicity(v: &mut ValidationHarness, benchmark: &serde_json::Value) {
    validation::section("Monotonicity");
    let checks = &benchmark["validation_checks"]["monotonicity"]["test_cases"];
    for tc in checks.as_array().expect("array") {
        let label = tc["label"].as_str().unwrap_or("mono");
        let (low, high) = if tc.get("base_t").is_some() && tc.get("base_dl").is_some() {
            (
                hamon_pet(json_field(tc, "base_t"), json_field(tc, "base_dl")),
                hamon_pet(json_field(tc, "high_t"), json_field(tc, "high_dl")),
            )
        } else if tc.get("base_t").is_some() {
            let dl = json_field(tc, "day_length_hours");
            (
                hamon_pet(json_field(tc, "base_t"), dl),
                hamon_pet(json_field(tc, "high_t"), dl),
            )
        } else {
            let tmean = json_field(tc, "tmean");
            (
                hamon_pet(tmean, json_field(tc, "base_dl")),
                hamon_pet(tmean, json_field(tc, "high_dl")),
            )
        };
        v.check_bool(&format!("{label}: {low:.3} < {high:.3}"), high > low);
    }
}

fn main() {
    validation::init_tracing();
    validation::banner("Exp 035: Hamon (1961) Temperature-Based PET");

    let mut v = ValidationHarness::new("Hamon PET");
    let benchmark = parse_benchmark_json(BENCHMARK_JSON).expect("benchmark_hamon.json must parse");

    validate_analytical(&mut v, &benchmark);
    validate_day_length(&mut v, &benchmark);
    validate_edge_cases(&mut v, &benchmark);
    validate_monotonicity(&mut v, &benchmark);

    v.finish();
}
