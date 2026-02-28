// SPDX-License-Identifier: AGPL-3.0-or-later
#![warn(clippy::pedantic)]
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
//! Exp 049: Blaney-Criddle (1950) PET Validation.
//!
//! Validates the 8th ET₀ method against analytical benchmarks from USDA-SCS
//! Tech Paper 96 and FAO-24 Table 18 daylight fractions.
//!
//! Benchmark: `control/blaney_criddle/benchmark_blaney_criddle.json`
//! Baseline: `control/blaney_criddle/blaney_criddle_et0.py` (18/18 PASS)
//!
//! Reference: Blaney HF, Criddle WD (1950) USDA-SCS Tech Paper 96.

use airspring_barracuda::eco::evapotranspiration::{
    blaney_criddle_et0, blaney_criddle_from_location, blaney_criddle_p,
};
use airspring_barracuda::tolerances;
use airspring_barracuda::validation::{self, json_field, parse_benchmark_json, ValidationHarness};

const BENCHMARK_JSON: &str =
    include_str!("../../../control/blaney_criddle/benchmark_blaney_criddle.json");

fn validate_analytical(v: &mut ValidationHarness, benchmark: &serde_json::Value) {
    validation::section("Analytical Benchmarks");
    let checks = &benchmark["analytical_benchmarks"];
    for tc in checks.as_array().expect("array") {
        let name = tc["name"].as_str().unwrap_or("?");
        let tmean = json_field(&tc["inputs"], "tmean_c");
        let p = json_field(&tc["inputs"], "p");
        let expected = json_field(tc, "expected_et0_mm_day");
        let tol = json_field(tc, "tolerance");
        let computed = blaney_criddle_et0(tmean, p);
        v.check_abs(&format!("BC({name})"), computed, expected, tol);
    }
}

fn validate_daylight(v: &mut ValidationHarness) {
    validation::section("Daylight Fraction");

    let p_tol = tolerances::BLANEY_CRIDDLE_DAYLIGHT.abs_tol;

    // Summer solstice at 40°N
    let p = blaney_criddle_p(40.0_f64.to_radians(), 172);
    v.check_abs("p_summer_40N", p, 0.333, p_tol);

    // Winter solstice at 40°N
    let p = blaney_criddle_p(40.0_f64.to_radians(), 356);
    v.check_abs("p_winter_40N", p, 0.222, p_tol);

    // Equator year-round (tighter: equator p is well-constrained)
    let p = blaney_criddle_p(0.0, 172);
    v.check_abs("p_equator", p, 0.274, p_tol / 3.0);
}

fn validate_monotonicity(v: &mut ValidationHarness) {
    validation::section("Monotonicity");

    let p = 0.274;
    let temps = [-10.0, 0.0, 10.0, 20.0, 30.0, 40.0];
    let et0s: Vec<f64> = temps.iter().map(|&t| blaney_criddle_et0(t, p)).collect();
    let mono = et0s.windows(2).all(|w| w[0] <= w[1]);
    v.check_bool("temperature_monotonic", mono);

    let ps = [0.199, 0.222, 0.274, 0.333, 0.366];
    let et0s: Vec<f64> = ps.iter().map(|&pp| blaney_criddle_et0(25.0, pp)).collect();
    let mono = et0s.windows(2).all(|w| w[0] <= w[1]);
    v.check_bool("daylight_monotonic", mono);

    // Summer p > winter p
    let p_s = blaney_criddle_p(40.0_f64.to_radians(), 172);
    let p_w = blaney_criddle_p(40.0_f64.to_radians(), 356);
    v.check_bool("summer_gt_winter_p", p_s > p_w);
}

fn validate_cross_method(v: &mut ValidationHarness) {
    validation::section("Cross-Method Comparison");

    let bc = blaney_criddle_from_location(22.0, 42.7_f64.to_radians(), 195);
    v.check_bool("bc_michigan_range_low", bc > 5.0);
    v.check_bool("bc_michigan_range_high", bc < 7.5);
}

fn validate_non_negative(v: &mut ValidationHarness) {
    validation::section("Non-Negative Constraint");
    let et0_tol = tolerances::ET0_REFERENCE.abs_tol;
    v.check_abs(
        "clamp_minus20",
        blaney_criddle_et0(-20.0, 0.199),
        0.0,
        et0_tol,
    );
    v.check_abs(
        "clamp_minus30",
        blaney_criddle_et0(-30.0, 0.199),
        0.0,
        et0_tol,
    );
}

fn main() {
    let benchmark = parse_benchmark_json(BENCHMARK_JSON).expect("valid JSON");
    let mut v = ValidationHarness::new("Exp 049: Blaney-Criddle (1950) PET");
    validate_analytical(&mut v, &benchmark);
    validate_daylight(&mut v);
    validate_monotonicity(&mut v);
    validate_cross_method(&mut v);
    validate_non_negative(&mut v);
    v.finish();
}
