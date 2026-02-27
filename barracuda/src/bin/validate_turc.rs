// SPDX-License-Identifier: AGPL-3.0-or-later
#![warn(clippy::pedantic)]
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
//! Exp 034: Turc (1961) Temperature-Radiation ET₀ Validation.
//!
//! Validates the Turc method with humidity correction against analytical
//! benchmarks derived from the published equation.
//!
//! Benchmark: `control/turc/benchmark_turc.json`
//! Baseline: `control/turc/turc_et0.py` (22/22 PASS)
//!
//! References:
//! - Turc L (1961) Annales Agronomiques 12:13-49
//! - Xu CY, Singh VP (2002) Water Resources Management 16:197-219

use airspring_barracuda::eco::evapotranspiration::turc_et0;
use airspring_barracuda::validation::{self, json_field, parse_benchmark_json, ValidationHarness};

const BENCHMARK_JSON: &str = include_str!("../../../control/turc/benchmark_turc.json");

fn validate_analytical(
    v: &mut ValidationHarness,
    benchmark: &serde_json::Value,
    section_key: &str,
    section_label: &str,
) {
    validation::section(section_label);
    let checks = &benchmark["validation_checks"][section_key]["test_cases"];
    for tc in checks.as_array().expect("array") {
        let tmean = json_field(tc, "tmean");
        let rs = json_field(tc, "rs_mj");
        let rh = json_field(tc, "rh");
        let expected = json_field(tc, "expected_et0");
        let tol = json_field(tc, "tolerance");
        let computed = turc_et0(tmean, rs, rh);
        v.check_abs(
            &format!("Turc(T={tmean},Rs={rs},RH={rh}%)"),
            computed,
            expected,
            tol,
        );
    }
}

fn validate_humidity_boundary(v: &mut ValidationHarness, benchmark: &serde_json::Value) {
    validation::section("Humidity Boundary (RH=50%)");
    let checks = &benchmark["validation_checks"]["humidity_boundary"]["test_cases"];
    for tc in checks.as_array().expect("array") {
        let tmean = json_field(tc, "tmean");
        let rs = json_field(tc, "rs_mj");
        let tol = json_field(tc, "tolerance");
        let at_50 = turc_et0(tmean, rs, 50.0);
        let at_49_99 = turc_et0(tmean, rs, 49.99);
        let diff = (at_50 - at_49_99).abs();
        v.check_bool(
            &format!("RH=50 ({at_50:.4}) ≈ RH=49.99 ({at_49_99:.4}), diff={diff:.6}"),
            diff < tol,
        );
    }
}

fn validate_edge_cases(v: &mut ValidationHarness, benchmark: &serde_json::Value) {
    validation::section("Edge Cases");
    let checks = &benchmark["validation_checks"]["edge_cases"]["test_cases"];
    for tc in checks.as_array().expect("array") {
        let label = tc["label"].as_str().unwrap_or("edge");
        let tmean = json_field(tc, "tmean");
        let rs = json_field(tc, "rs_mj");
        let rh = json_field(tc, "rh");
        let computed = turc_et0(tmean, rs, rh);
        let check = tc["check"].as_str().unwrap_or("");
        let ok = match check {
            "positive" => computed > 0.0,
            "non_negative" => computed >= 0.0,
            "zero" => computed == 0.0,
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
        let (low, high) = if tc.get("base_rs").is_some() {
            let tmean = json_field(tc, "tmean");
            let rh = json_field(tc, "rh");
            (
                turc_et0(tmean, json_field(tc, "base_rs"), rh),
                turc_et0(tmean, json_field(tc, "high_rs"), rh),
            )
        } else if tc.get("base_t").is_some() {
            let rs = json_field(tc, "rs_mj");
            let rh = json_field(tc, "rh");
            (
                turc_et0(json_field(tc, "base_t"), rs, rh),
                turc_et0(json_field(tc, "high_t"), rs, rh),
            )
        } else {
            let tmean = json_field(tc, "tmean");
            let rs = json_field(tc, "rs_mj");
            (
                turc_et0(tmean, rs, json_field(tc, "base_rh")),
                turc_et0(tmean, rs, json_field(tc, "low_rh")),
            )
        };
        v.check_bool(&format!("{label}: {low:.3} < {high:.3}"), high > low);
    }
}

fn main() {
    validation::init_tracing();
    validation::banner("Exp 034: Turc (1961) Temperature-Radiation ET₀");

    let mut v = ValidationHarness::new("Turc ET₀");
    let benchmark = parse_benchmark_json(BENCHMARK_JSON).expect("benchmark_turc.json must parse");

    validate_analytical(
        &mut v,
        &benchmark,
        "analytical_high_rh",
        "Analytical (RH ≥ 50%)",
    );
    validate_analytical(
        &mut v,
        &benchmark,
        "analytical_low_rh",
        "Analytical (RH < 50%)",
    );
    validate_humidity_boundary(&mut v, &benchmark);
    validate_edge_cases(&mut v, &benchmark);
    validate_monotonicity(&mut v, &benchmark);

    v.finish();
}
