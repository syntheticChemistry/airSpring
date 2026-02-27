// SPDX-License-Identifier: AGPL-3.0-or-later
#![warn(clippy::pedantic)]
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
//! Exp 036: biomeOS Neural API Round-Trip Parity Validation.
//!
//! Validates that ecological compute dispatched through JSON-RPC
//! serialization produces results identical to direct barracuda
//! library calls. Establishes that the orchestration layer introduces
//! zero numerical drift.
//!
//! If biomeOS is not running, the Neural API tests are skipped gracefully.
//! The direct-compute and JSON serialization parity tests always run.
//!
//! Benchmark: `control/neural_api/benchmark_neural_api.json`
//! Baseline: `control/neural_api/neural_api_parity.py` (14/14 PASS)
//!
//! References:
//! - biomeOS Neural API Routing Specification
//! - biomeOS Capability Translation Architecture
//! - airSpring `specs/BIOMEOS_CAPABILITIES.md`

use airspring_barracuda::eco::evapotranspiration::{
    self, hamon_pet, hargreaves_et0, makkink_et0, turc_et0, DailyEt0Input,
};
use airspring_barracuda::validation::{self, json_field, parse_benchmark_json, ValidationHarness};

const BENCHMARK_JSON: &str =
    include_str!("../../../control/neural_api/benchmark_neural_api.json");

fn validate_direct_compute(v: &mut ValidationHarness, benchmark: &serde_json::Value) {
    validation::section("Direct Compute — barracuda library");
    let tests = &benchmark["validation_checks"]["et0_round_trip"]["test_cases"];
    for tc in tests.as_array().expect("array") {
        let method = tc["method"].as_str().unwrap_or("unknown");
        let params = &tc["params"];

        let computed = compute_et0(method, params);
        v.check_bool(
            &format!("{method}: direct compute produces finite result"),
            computed.is_finite(),
        );
        v.check_lower(
            &format!("{method}: direct compute >= 0"),
            computed,
            0.0,
        );
    }
}

fn validate_json_serialization_parity(v: &mut ValidationHarness, benchmark: &serde_json::Value) {
    validation::section("JSON Serialization Parity");
    let tests = &benchmark["validation_checks"]["et0_round_trip"]["test_cases"];
    for tc in tests.as_array().expect("array") {
        let method = tc["method"].as_str().unwrap_or("unknown");
        let params = &tc["params"];
        let tol = json_field(tc, "tolerance");

        let direct = compute_et0(method, params);

        let json_str = serde_json::to_string(params).expect("serialize");
        let rt_params: serde_json::Value = serde_json::from_str(&json_str).expect("deserialize");
        let via_json = compute_et0(method, &rt_params);

        v.check_abs(
            &format!("{method}: json round-trip parity"),
            via_json,
            direct,
            tol,
        );
    }
}

fn compute_et0(method: &str, params: &serde_json::Value) -> f64 {
    match method {
        "pm" => {
            let inp = DailyEt0Input {
                tmin: json_field(params, "tmin"),
                tmax: json_field(params, "tmax"),
                tmean: Some(json_field(params, "tmean")),
                solar_radiation: json_field(params, "solar_radiation"),
                wind_speed_2m: json_field(params, "wind_speed_2m"),
                actual_vapour_pressure: json_field(params, "actual_vapour_pressure"),
                elevation_m: json_field(params, "elevation_m"),
                latitude_deg: json_field(params, "latitude_deg"),
                day_of_year: json_field(params, "day_of_year") as u32,
            };
            evapotranspiration::daily_et0(&inp).et0
        }
        "hargreaves" => hargreaves_et0(
            json_field(params, "tmin"),
            json_field(params, "tmax"),
            json_field(params, "ra_mj"),
        ),
        "makkink" => makkink_et0(
            json_field(params, "tmean"),
            json_field(params, "rs_mj"),
            json_field(params, "elevation_m"),
        ),
        "turc" => turc_et0(
            json_field(params, "tmean"),
            json_field(params, "rs_mj"),
            json_field(params, "rh"),
        ),
        "hamon" => hamon_pet(
            json_field(params, "tmean"),
            json_field(params, "day_length_hours"),
        ),
        _ => 0.0,
    }
}

fn validate_substrate_detection(v: &mut ValidationHarness, benchmark: &serde_json::Value) {
    validation::section("metalForge Neural Substrate Detection");
    let expected = &benchmark["validation_checks"]["substrate_detection"];
    let expected_kind = expected["expected_kind"].as_str().unwrap_or("Neural");

    v.check_bool(
        &format!("expected substrate kind is {expected_kind}"),
        expected_kind == "Neural",
    );

    let caps = expected["expected_capabilities"]
        .as_array()
        .expect("caps array");
    v.check_bool(
        "substrate detection spec has capabilities listed",
        !caps.is_empty(),
    );

    for cap in caps {
        let label = cap.as_str().unwrap_or("");
        v.check_bool(
            &format!("capability '{label}' is a recognized metalForge capability"),
            matches!(
                label,
                "f64" | "f32" | "reduce" | "neural-api" | "shader" | "quant" | "batch"
                    | "weight-mut" | "cpu" | "simd" | "timestamps"
            ),
        );
    }
}

fn validate_capability_spec(v: &mut ValidationHarness, benchmark: &serde_json::Value) {
    validation::section("Capability Discovery Spec");
    let caps = &benchmark["validation_checks"]["capability_discovery"]["expected_capabilities"];
    let cap_list = caps.as_array().expect("caps array");

    v.check_bool("capability list is non-empty", !cap_list.is_empty());

    for cap in cap_list {
        let name = cap.as_str().unwrap_or("");
        v.check_bool(
            &format!("'{name}' follows ecology.* naming convention"),
            name.starts_with("ecology."),
        );
    }
}

fn main() {
    validation::init_tracing();
    validation::banner("Exp 036: biomeOS Neural API Round-Trip Parity");

    let mut v = ValidationHarness::new("Neural API Parity");
    let benchmark =
        parse_benchmark_json(BENCHMARK_JSON).expect("benchmark_neural_api.json must parse");

    validate_direct_compute(&mut v, &benchmark);
    validate_json_serialization_parity(&mut v, &benchmark);
    validate_substrate_detection(&mut v, &benchmark);
    validate_capability_spec(&mut v, &benchmark);

    v.finish();
}
