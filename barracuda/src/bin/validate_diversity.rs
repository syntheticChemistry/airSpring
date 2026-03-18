// SPDX-License-Identifier: AGPL-3.0-or-later
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
//!
//! Provenance: script=`control/diversity/diversity_indices.py`, commit=fad2e1b, date=2026-02-27

use airspring_barracuda::eco::diversity;
use airspring_barracuda::tolerances::{BIO_BRAY_CURTIS, BIO_DIVERSITY_SHANNON, BIO_DIVERSITY_SIMPSON};
use airspring_barracuda::validation::{self, ValidationHarness, json_field, parse_benchmark_json};

const BENCHMARK_JSON: &str = include_str!("../../../control/diversity/benchmark_diversity.json");

fn parse_f64_array(val: &serde_json::Value) -> Vec<f64> {
    let Some(arr) = val.as_array() else {
        eprintln!("[FAIL] expected JSON array for counts/samples");
        std::process::exit(1);
    };
    arr.iter().filter_map(serde_json::Value::as_f64).collect()
}

fn tol_or_fallback(tc: &serde_json::Value, fallback: f64) -> f64 {
    tc.get("tolerance")
        .and_then(serde_json::Value::as_f64)
        .unwrap_or(fallback)
}

fn test_cases_or_exit(benchmark: &serde_json::Value, section: &str) -> Vec<serde_json::Value> {
    let Some(arr) = benchmark
        .get("validation_checks")
        .and_then(|vc| vc.get(section))
        .and_then(|s| s.get("test_cases"))
        .and_then(|tc| tc.as_array())
    else {
        eprintln!("[FAIL] benchmark JSON: validation_checks.{section}.test_cases missing");
        std::process::exit(1);
    };
    arr.clone()
}

fn validate_shannon(v: &mut ValidationHarness, benchmark: &serde_json::Value) {
    validation::section("Shannon Entropy");
    for tc in &test_cases_or_exit(benchmark, "shannon") {
        let label = tc["label"].as_str().unwrap_or("case");
        let counts = parse_f64_array(&tc["counts"]);
        let expected = json_field(tc, "expected");
        let tol = tol_or_fallback(tc, BIO_DIVERSITY_SHANNON.abs_tol);
        let computed = diversity::shannon(&counts);
        v.check_abs(&format!("H' {label}"), computed, expected, tol);
    }
}

fn validate_simpson(v: &mut ValidationHarness, benchmark: &serde_json::Value) {
    validation::section("Simpson Diversity");
    for tc in &test_cases_or_exit(benchmark, "simpson") {
        let label = tc["label"].as_str().unwrap_or("case");
        let counts = parse_f64_array(&tc["counts"]);
        let expected = json_field(tc, "expected");
        let tol = tol_or_fallback(tc, BIO_DIVERSITY_SIMPSON.abs_tol);
        let computed = diversity::simpson(&counts);
        v.check_abs(&format!("D {label}"), computed, expected, tol);
    }
}

fn validate_chao1(v: &mut ValidationHarness, benchmark: &serde_json::Value) {
    validation::section("Chao1 Richness");
    for tc in &test_cases_or_exit(benchmark, "chao1") {
        let label = tc["label"].as_str().unwrap_or("case");
        let counts = parse_f64_array(&tc["counts"]);
        let expected = json_field(tc, "expected");
        let tol = tol_or_fallback(tc, BIO_DIVERSITY_SHANNON.abs_tol);
        let computed = diversity::chao1(&counts);
        v.check_abs(&format!("Chao1 {label}"), computed, expected, tol);
    }
}

fn validate_pielou(v: &mut ValidationHarness, benchmark: &serde_json::Value) {
    validation::section("Pielou Evenness");
    for tc in &test_cases_or_exit(benchmark, "pielou") {
        let label = tc["label"].as_str().unwrap_or("case");
        let counts = parse_f64_array(&tc["counts"]);
        let expected = json_field(tc, "expected");
        let tol = tol_or_fallback(tc, BIO_DIVERSITY_SHANNON.abs_tol);
        let computed = diversity::pielou_evenness(&counts);
        v.check_abs(&format!("J' {label}"), computed, expected, tol);
    }
}

fn validate_bray_curtis(v: &mut ValidationHarness, benchmark: &serde_json::Value) {
    validation::section("Bray-Curtis Dissimilarity");
    for tc in &test_cases_or_exit(benchmark, "bray_curtis") {
        let label = tc["label"].as_str().unwrap_or("case");
        let a = parse_f64_array(&tc["sample_a"]);
        let b = parse_f64_array(&tc["sample_b"]);
        let expected = json_field(tc, "expected");
        let tol = tol_or_fallback(tc, BIO_BRAY_CURTIS.abs_tol);
        let computed = diversity::bray_curtis(&a, &b);
        v.check_abs(&format!("BC {label}"), computed, expected, tol);
    }
}

fn main() {
    validation::init_tracing();
    validation::banner("Exp 032: Ecological Diversity Indices");

    let mut v = ValidationHarness::new("Diversity Indices");
    let Ok(benchmark) = parse_benchmark_json(BENCHMARK_JSON) else {
        eprintln!("[FAIL] benchmark JSON parse error");
        std::process::exit(1);
    };

    validate_shannon(&mut v, &benchmark);
    validate_simpson(&mut v, &benchmark);
    validate_chao1(&mut v, &benchmark);
    validate_pielou(&mut v, &benchmark);
    validate_bray_curtis(&mut v, &benchmark);

    v.finish();
}
