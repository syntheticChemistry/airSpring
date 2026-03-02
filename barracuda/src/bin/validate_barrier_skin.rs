// SPDX-License-Identifier: AGPL-3.0-or-later
#![warn(clippy::pedantic)]
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
//! Exp 068: Barrier State Model Validation.
//!
//! Applies van Genuchten θ(h)/K(h) to skin barrier permeability modeling.
//! Cross-validates against Python control (`control/barrier_skin/barrier_skin.py`, 16/16 PASS).
//!
//! Benchmark: `control/barrier_skin/benchmark_barrier_skin.json`
//!
//! References:
//! - Paper 12 §2.3: Dimensional promotion via barrier disruption
//! - van Genuchten (1980) SSSA J 44:892-898
//! - Tagami H (2008) Br J Dermatol 158:431-436
//!
//! script=`control/barrier_skin/barrier_skin.py`, commit=dbfb53a, date=2026-03-02
//! Run: `python3 control/barrier_skin/barrier_skin.py`

use airspring_barracuda::eco::tissue::barrier_disruption_d_eff;
use airspring_barracuda::eco::van_genuchten::{van_genuchten_k, van_genuchten_theta};
use airspring_barracuda::validation::{self, json_field, parse_benchmark_json, ValidationHarness};

const BENCHMARK_JSON: &str =
    include_str!("../../../control/barrier_skin/benchmark_barrier_skin.json");

const SKIN_THETA_R: f64 = 0.05;
const SKIN_THETA_S: f64 = 1.0;
const SKIN_ALPHA: f64 = 0.01;
const SKIN_N_VG: f64 = 1.8;
const SKIN_KS: f64 = 50.0;

fn normalize_barrier(theta: f64) -> f64 {
    (theta - SKIN_THETA_R) / (SKIN_THETA_S - SKIN_THETA_R)
}

fn validate_retention(v: &mut ValidationHarness, benchmark: &serde_json::Value) {
    validation::section("Barrier VG Retention");
    let checks = &benchmark["validation_checks"]["barrier_vg_retention"]["test_cases"];
    for tc in checks.as_array().expect("array") {
        let label = tc["label"].as_str().unwrap_or("case");
        let h = json_field(tc, "h");
        let theta = van_genuchten_theta(h, SKIN_THETA_R, SKIN_THETA_S, SKIN_ALPHA, SKIN_N_VG);
        let barrier = normalize_barrier(theta);

        if let Some(expected) = tc
            .get("expected_barrier")
            .and_then(serde_json::Value::as_f64)
        {
            let tol = json_field(tc, "tolerance");
            v.check_abs(&format!("barrier {label}"), barrier, expected, tol);
        } else if let Some(range) = tc
            .get("expected_barrier_range")
            .and_then(serde_json::Value::as_array)
        {
            let lo = range[0].as_f64().unwrap_or(0.0);
            let hi = range[1].as_f64().unwrap_or(1.0);
            v.check_bool(
                &format!("barrier {label} in [{lo},{hi}]"),
                (lo..=hi).contains(&barrier),
            );
        }
    }
}

fn validate_conductivity(v: &mut ValidationHarness, benchmark: &serde_json::Value) {
    validation::section("Barrier Conductivity");
    let checks = &benchmark["validation_checks"]["barrier_conductivity"]["test_cases"];
    for tc in checks.as_array().expect("array") {
        let label = tc["label"].as_str().unwrap_or("case");
        let h = json_field(tc, "h");
        let k = van_genuchten_k(
            h,
            SKIN_KS,
            SKIN_THETA_R,
            SKIN_THETA_S,
            SKIN_ALPHA,
            SKIN_N_VG,
        );

        if let Some(expected) = tc
            .get("expected_k_ratio")
            .and_then(serde_json::Value::as_f64)
        {
            let tol = json_field(tc, "tolerance");
            v.check_abs(&format!("K_ratio {label}"), k / SKIN_KS, expected, tol);
        } else if tc
            .get("expected_k_less_than_max")
            .and_then(serde_json::Value::as_bool)
            == Some(true)
        {
            v.check_bool(&format!("K<Ks {label}"), k < SKIN_KS);
        } else if tc
            .get("expected_k_near_zero")
            .and_then(serde_json::Value::as_bool)
            == Some(true)
        {
            v.check_bool(&format!("K≈0 {label}"), k < 0.001 * SKIN_KS);
        }
    }
}

fn validate_d_eff_mapping(v: &mut ValidationHarness, benchmark: &serde_json::Value) {
    validation::section("Barrier → d_eff Mapping");
    let checks = &benchmark["validation_checks"]["barrier_to_d_eff_mapping"]["test_cases"];
    for tc in checks.as_array().expect("array") {
        let label = tc["label"].as_str().unwrap_or("case");
        let bi = json_field(tc, "barrier_integrity");
        let expected_d = json_field(tc, "expected_d_eff");
        let tol = json_field(tc, "tolerance");
        let breach = (1.0 - bi).clamp(0.0, 1.0);
        let d = barrier_disruption_d_eff(breach);
        v.check_abs(&format!("d_eff {label}"), d, expected_d, tol);
    }
}

fn validate_skin_params(v: &mut ValidationHarness, benchmark: &serde_json::Value) {
    validation::section("Skin VG Parameters");
    let checks = &benchmark["validation_checks"]["skin_vg_params"]["test_cases"];
    for tc in checks.as_array().expect("array") {
        let label = tc["label"].as_str().unwrap_or("case");
        let tr = json_field(tc, "theta_r");
        let ts = json_field(tc, "theta_s");
        let alpha = json_field(tc, "alpha");
        let n = json_field(tc, "n_vg");
        let theta_intact = van_genuchten_theta(0.0, tr, ts, alpha, n);
        let theta_stressed = van_genuchten_theta(-100.0, tr, ts, alpha, n);
        v.check_bool(
            &format!("θ(0)>θ(-100) {label}"),
            theta_intact > theta_stressed,
        );
    }
}

fn validate_duality(v: &mut ValidationHarness, benchmark: &serde_json::Value) {
    validation::section("Dimensional Duality");
    let checks = &benchmark["validation_checks"]["duality_check"]["test_cases"];
    for tc in checks.as_array().expect("array") {
        let label = tc["label"].as_str().unwrap_or("case");
        let d_before = json_field(tc, "d_before");
        let d_after = json_field(tc, "d_after");
        let direction = tc["direction"].as_str().unwrap_or("?");
        let ok = match direction {
            "collapse" => d_after < d_before,
            "promotion" => d_after > d_before,
            _ => false,
        };
        v.check_bool(&format!("duality {label}"), ok);
    }
}

fn main() {
    validation::init_tracing();
    validation::banner("Exp 068: Barrier State Model (Paper 12)");

    let mut v = ValidationHarness::new("Barrier Skin");
    let benchmark =
        parse_benchmark_json(BENCHMARK_JSON).expect("benchmark_barrier_skin.json must parse");

    validate_retention(&mut v, &benchmark);
    validate_conductivity(&mut v, &benchmark);
    validate_d_eff_mapping(&mut v, &benchmark);
    validate_skin_params(&mut v, &benchmark);
    validate_duality(&mut v, &benchmark);

    v.finish();
}
