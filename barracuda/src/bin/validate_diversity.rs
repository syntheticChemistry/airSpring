// SPDX-License-Identifier: AGPL-3.0-or-later
#![warn(clippy::pedantic)]
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
//! Exp 032: Ecological Diversity Indices Validation.
//!
//! Validates Shannon entropy, Simpson index, Chao1 richness estimator,
//! Pielou evenness, and Bray-Curtis dissimilarity against analytically
//! derived expected values for agroecosystem assessment.
//!
//! Benchmark: `control/diversity/benchmark_diversity.json`
//! Baseline: `control/diversity/diversity_indices.py` (22/22 PASS)
//!
//! References:
//! - Shannon (1948) Bell Sys Tech J 27(3):379-423
//! - Simpson (1949) Nature 163:688
//! - Chao (1984) Scand J Statistics 11(4):265-270
//! - Pielou (1966) J Theor Biology 13:131-144
//! - Bray & Curtis (1957) Ecological Monographs 27(4):325-349

use airspring_barracuda::eco::diversity;
use airspring_barracuda::validation::{self, json_field, parse_benchmark_json, ValidationHarness};

const BENCHMARK_JSON: &str = include_str!("../../../control/diversity/benchmark_diversity.json");

fn parse_f64_array(val: &serde_json::Value) -> Vec<f64> {
    val.as_array()
        .expect("array")
        .iter()
        .map(|v| v.as_f64().expect("f64"))
        .collect()
}

fn validate_shannon(v: &mut ValidationHarness, benchmark: &serde_json::Value) {
    validation::section("Shannon Entropy");
    let checks = &benchmark["validation_checks"]["shannon"]["test_cases"];
    for tc in checks.as_array().expect("array") {
        let label = tc["label"].as_str().unwrap_or("case");
        let counts = parse_f64_array(&tc["counts"]);
        let expected = json_field(tc, "expected");
        let tol = json_field(tc, "tolerance");
        let computed = diversity::shannon(&counts);
        v.check_abs(&format!("H' {label}"), computed, expected, tol);
    }
}

fn validate_simpson(v: &mut ValidationHarness, benchmark: &serde_json::Value) {
    validation::section("Simpson Diversity");
    let checks = &benchmark["validation_checks"]["simpson"]["test_cases"];
    for tc in checks.as_array().expect("array") {
        let label = tc["label"].as_str().unwrap_or("case");
        let counts = parse_f64_array(&tc["counts"]);
        let expected = json_field(tc, "expected");
        let tol = json_field(tc, "tolerance");
        let computed = diversity::simpson(&counts);
        v.check_abs(&format!("D {label}"), computed, expected, tol);
    }
}

fn validate_chao1(v: &mut ValidationHarness, benchmark: &serde_json::Value) {
    validation::section("Chao1 Richness");
    let checks = &benchmark["validation_checks"]["chao1"]["test_cases"];
    for tc in checks.as_array().expect("array") {
        let label = tc["label"].as_str().unwrap_or("case");
        let counts = parse_f64_array(&tc["counts"]);
        let expected = json_field(tc, "expected");
        let tol = json_field(tc, "tolerance");
        let computed = diversity::chao1(&counts);
        v.check_abs(&format!("Chao1 {label}"), computed, expected, tol);
    }
}

fn validate_pielou(v: &mut ValidationHarness, benchmark: &serde_json::Value) {
    validation::section("Pielou Evenness");
    let checks = &benchmark["validation_checks"]["pielou"]["test_cases"];
    for tc in checks.as_array().expect("array") {
        let label = tc["label"].as_str().unwrap_or("case");
        let counts = parse_f64_array(&tc["counts"]);
        let expected = json_field(tc, "expected");
        let tol = json_field(tc, "tolerance");
        let computed = diversity::pielou_evenness(&counts);
        v.check_abs(&format!("J' {label}"), computed, expected, tol);
    }
}

fn validate_bray_curtis(v: &mut ValidationHarness, benchmark: &serde_json::Value) {
    validation::section("Bray-Curtis Dissimilarity");
    let checks = &benchmark["validation_checks"]["bray_curtis"]["test_cases"];
    for tc in checks.as_array().expect("array") {
        let label = tc["label"].as_str().unwrap_or("case");
        let a = parse_f64_array(&tc["sample_a"]);
        let b = parse_f64_array(&tc["sample_b"]);
        let expected = json_field(tc, "expected");
        let tol = json_field(tc, "tolerance");
        let computed = diversity::bray_curtis(&a, &b);
        v.check_abs(&format!("BC {label}"), computed, expected, tol);
    }
}

fn main() {
    validation::init_tracing();
    validation::banner("Exp 032: Ecological Diversity Indices");

    let mut v = ValidationHarness::new("Diversity Indices");
    let benchmark =
        parse_benchmark_json(BENCHMARK_JSON).expect("benchmark_diversity.json must parse");

    validate_shannon(&mut v, &benchmark);
    validate_simpson(&mut v, &benchmark);
    validate_chao1(&mut v, &benchmark);
    validate_pielou(&mut v, &benchmark);
    validate_bray_curtis(&mut v, &benchmark);

    v.finish();
}
