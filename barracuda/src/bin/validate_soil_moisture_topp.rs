// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2025-2026 ecoPrimals Collective
//! Validate Topp et al. (1980) dielectric–VWC equation against Python baseline
//! (`control/soil_moisture_topp/benchmark_soil_moisture_topp.json`).
//!
//! Forward map [`topp_equation`] and inverse [`inverse_topp`] are checked against
//! embedded benchmarks including round-trip and published ranges.

use airspring_barracuda::cast::usize_f64;
use airspring_barracuda::eco::soil_moisture::{inverse_topp, topp_equation};
use airspring_barracuda::validation::{
    self, ValidationHarness, json_f64, parse_benchmark_json,
};
use serde_json::Value;

const BENCHMARK_JSON: &str =
    include_str!("../../../control/soil_moisture_topp/benchmark_soil_moisture_topp.json");

const DIELECTRIC_TOL: f64 = 1e-6;

fn validate_forward_named_checks(v: &mut ValidationHarness, bench: &Value) {
    validation::section("Topp equation — catalogue checks");

    let checks = bench["checks"].as_array().expect("checks array");

    for c in checks {
        let Some(name) = c["name"].as_str() else {
            continue;
        };
        if !name.starts_with("topp_") {
            continue;
        }
        let eps = json_f64(c, &["dielectric_constant"]).expect("dielectric_constant");
        let expected = json_f64(c, &["value"]).expect("value");
        let obs = topp_equation(eps);
        v.check_abs(name, obs, expected, DIELECTRIC_TOL);
    }
}

fn validate_monotonicity(v: &mut ValidationHarness, bench: &Value) {
    validation::section("Monotonicity (Ka = 3..20)");

    bench["checks"]
        .as_array()
        .expect("checks array")
        .iter()
        .find(|x| x["name"] == "monotonicity_3_to_20")
        .expect("benchmark must include monotonicity_3_to_20");

    for ka in 3..20 {
        let ka_lo = usize_f64(ka);
        let ka_hi = usize_f64(ka + 1);
        let cur_lo = topp_equation(ka_lo);
        let cur_hi = topp_equation(ka_hi);
        let label = format!(
            "monotonicity_3_to_20: Ka {ka_hi:.0}: VWC {cur_hi:.6} vs Ka {ka_lo:.0}: VWC {cur_lo:.6}"
        );
        v.check_bool(&label, cur_hi > cur_lo);
    }
}

fn validate_oven_dry(v: &mut ValidationHarness, bench: &Value) {
    validation::section("Oven dry (Ka → 1)");
    let checks = bench["checks"].as_array().expect("checks array");
    let chk = checks
        .iter()
        .find(|x| x["name"] == "oven_dry_near_zero")
        .expect("oven_dry_near_zero");

    // Raw Topp extrapolation at ε≈ε_air is weakly negative; Python bench records |θv| magnitude.
    let v_obs_raw = topp_equation(1.0);
    let mag = v_obs_raw.abs();
    let expected_mag = chk["value"]
        .as_f64()
        .expect("oven_dry_near_zero baseline value magnitude");
    v.check_abs(
        "oven_dry_near_zero — |θv| vs baseline",
        mag,
        expected_mag.abs(),
        DIELECTRIC_TOL,
    );

    let range = chk["expected_range"].as_array().expect("expected_range array");
    let lo = range[0].as_f64().expect("oven_dry lower");
    let hi = range[1].as_f64().expect("oven_dry upper");
    v.check_bool(
        &format!("oven dry |θv|={mag:.6} in [{lo:.4},{hi:.4}]"),
        (lo..=hi).contains(&mag),
    );
}

fn validate_roundtrips_and_published(v: &mut ValidationHarness, bench: &Value) {
    validation::section("Inverse Topp round-trips");

    let checks = bench["checks"].as_array().expect("checks array");

    for c in checks {
        let Some(name) = c["name"].as_str() else {
            continue;
        };

        if let Some(suffix) = name.strip_prefix("roundtrip_ka") {
            let ka_tail =
                suffix
                    .parse::<f64>()
                    .unwrap_or_else(|_| panic!("{name}: expected Ka suffix (e.g. 5.0)"));
            let recovered = inverse_topp(topp_equation(ka_tail));
            let expected = json_f64(c, &["expected"]).expect("expected");
            let tol = json_f64(c, &["tolerance"]).unwrap_or(DIELECTRIC_TOL);
            v.check_abs(name, recovered, expected, tol);
            continue;
        }

        if name == "published_ka15" {
            let theta = topp_equation(15.0);
            let range = c["expected_range"].as_array().expect("expected_range array");
            let lo = range[0].as_f64().expect("published_ka15 lower");
            let hi = range[1].as_f64().expect("published_ka15 upper");
            v.check_bool(
                &format!("published_ka15: VWC={theta:.6} in [{lo:.4},{hi:.4}]"),
                (lo..=hi).contains(&theta),
            );
        }
    }
}

fn validate_reference_table(v: &mut ValidationHarness, bench: &Value) {
    validation::section("Ka → VWC reference table");

    let table = bench["reference"]["ka_to_vwc_table"]
        .as_object()
        .expect("ka_to_vwc_table");

    for (ka_str, theta_json) in table {
        let ka: f64 = ka_str
            .parse()
            .unwrap_or_else(|_| panic!("Ka key `{ka_str}` must parse as f64"));

        let expected = theta_json
            .as_f64()
            .unwrap_or_else(|| panic!("theta for Ka={ka} must be f64"));
        let obs = topp_equation(ka);
        let label = format!("ka_to_vwc_table[{ka}]");
        v.check_abs(&label, obs, expected, DIELECTRIC_TOL);
    }
}

fn main() {
    validation::init_tracing();
    validation::banner("Soil Moisture — Topp Dielectric Calibration");
    let mut v = ValidationHarness::new("Soil Moisture — Topp");

    let Ok(bench) = parse_benchmark_json(BENCHMARK_JSON) else {
        eprintln!("[FAIL] benchmark JSON parse error");
        std::process::exit(1);
    };

    validate_forward_named_checks(&mut v, &bench);
    validate_monotonicity(&mut v, &bench);
    validate_oven_dry(&mut v, &bench);
    validate_roundtrips_and_published(&mut v, &bench);
    validate_reference_table(&mut v, &bench);

    v.finish();
}
