// SPDX-License-Identifier: AGPL-3.0-or-later
//! Exp 033: Makkink (1957) Radiation-Based ET₀ Validation.
//!
//! Validates the Makkink method with de Bruin (1987) coefficients
//! against analytical benchmarks derived from the published equation.
//!
//! Benchmark: `control/makkink/benchmark_makkink.json`
//! Baseline: `control/makkink/makkink_et0.py` (21/21 PASS)
//!
//! References:
//! - Makkink GF (1957) J Inst Water Eng 11:277-288
//! - de Bruin HAR (1987) From Penman to Makkink, TNO, pp 5-31
//!
//! Provenance: script=`control/makkink/makkink_et0.py`, commit=d3ecdc8, date=2026-02-27

use airspring_barracuda::eco::evapotranspiration::makkink_et0;
use airspring_barracuda::validation::{self, ValidationHarness, json_field, parse_benchmark_json};

const BENCHMARK_JSON: &str = include_str!("../../../control/makkink/benchmark_makkink.json");

fn validate_analytical(v: &mut ValidationHarness, benchmark: &serde_json::Value) {
    validation::section("Analytical Benchmarks");
    let checks = &benchmark["validation_checks"]["analytical"]["test_cases"];
    for tc in checks.as_array().expect("array") {
        let tmean = json_field(tc, "tmean");
        let rs = json_field(tc, "rs_mj");
        let elev = json_field(tc, "elevation_m");
        let expected = json_field(tc, "expected_et0");
        let tol = json_field(tc, "tolerance");
        let computed = makkink_et0(tmean, rs, elev);
        v.check_abs(
            &format!("Makkink(T={tmean},Rs={rs},z={elev})"),
            computed,
            expected,
            tol,
        );
    }
}

fn validate_pm_cross(v: &mut ValidationHarness, benchmark: &serde_json::Value) {
    validation::section("PM Cross-Comparison");
    let checks = &benchmark["validation_checks"]["pm_cross_comparison"]["test_cases"];
    for tc in checks.as_array().expect("array") {
        let label = tc["label"].as_str().unwrap_or("city");
        let tmean = json_field(tc, "tmean");
        let rs = json_field(tc, "rs_mj");
        let elev = json_field(tc, "elevation_m");
        let pm_et0 = json_field(tc, "approx_pm_et0");
        let min_r = json_field(tc, "min_ratio");
        let max_r = json_field(tc, "max_ratio");
        let computed = makkink_et0(tmean, rs, elev);
        let ratio = if pm_et0 > 0.0 {
            computed / pm_et0
        } else {
            f64::NAN
        };
        v.check_bool(
            &format!("{label}: Mak/PM={ratio:.3} in [{min_r},{max_r}]"),
            ratio >= min_r && ratio <= max_r,
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
        let elev = json_field(tc, "elevation_m");
        let computed = makkink_et0(tmean, rs, elev);
        let check = tc["check"].as_str().unwrap_or("");
        let ok = match check {
            "non_negative" => computed >= 0.0,
            "positive" => computed > 0.0,
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
        let elev = json_field(tc, "elevation_m");
        let (low, high) = if tc.get("base_rs").is_some() {
            let tmean = json_field(tc, "tmean");
            (
                makkink_et0(tmean, json_field(tc, "base_rs"), elev),
                makkink_et0(tmean, json_field(tc, "high_rs"), elev),
            )
        } else {
            let rs = json_field(tc, "rs_mj");
            (
                makkink_et0(json_field(tc, "base_t"), rs, elev),
                makkink_et0(json_field(tc, "high_t"), rs, elev),
            )
        };
        v.check_bool(&format!("{label}: {low:.3} < {high:.3}"), high > low);
    }
}

fn main() {
    validation::init_tracing();
    validation::banner("Exp 033: Makkink (1957) Radiation-Based ET₀");

    let mut v = ValidationHarness::new("Makkink ET₀");
    let benchmark =
        parse_benchmark_json(BENCHMARK_JSON).expect("benchmark_makkink.json must parse");

    validate_analytical(&mut v, &benchmark);
    validate_pm_cross(&mut v, &benchmark);
    validate_edge_cases(&mut v, &benchmark);
    validate_monotonicity(&mut v, &benchmark);

    v.finish();
}
