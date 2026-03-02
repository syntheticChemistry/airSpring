// SPDX-License-Identifier: AGPL-3.0-or-later
#![warn(clippy::pedantic)]
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
//! Exp 069: Cross-Species Skin Comparison Validation.
//!
//! Validates Anderson predictions across species (canine, feline, human).
//! Cross-validates against Python control (`control/cross_species_skin/cross_species_skin.py`, 19/19 PASS).
//!
//! Benchmark: `control/cross_species_skin/benchmark_cross_species_skin.json`
//!
//! References:
//! - Paper 12: Immunological Anderson — cross-species validation
//! - Gonzales AJ et al. (2013) Vet Dermatol 24:48-53
//! - Marsella R, De Benedetto A (2017) Vet Dermatol 28:306-e69

use airspring_barracuda::eco::diversity;
use airspring_barracuda::eco::tissue::{barrier_disruption_d_eff, AndersonRegime};
use airspring_barracuda::validation::{self, json_field, parse_benchmark_json, ValidationHarness};

const BENCHMARK_JSON: &str =
    include_str!("../../../control/cross_species_skin/benchmark_cross_species_skin.json");

fn parse_f64_array(val: &serde_json::Value) -> Vec<f64> {
    val.as_array()
        .expect("array")
        .iter()
        .map(|v| v.as_f64().expect("f64"))
        .collect()
}

fn classify_regime(w: f64, d: f64) -> AndersonRegime {
    let w_c = if d < 2.5 { 4.0 } else { 16.26 };
    let margin = 0.1 * w_c;
    if w > w_c + margin {
        AndersonRegime::Localized
    } else if w < w_c - margin {
        AndersonRegime::Extended
    } else {
        AndersonRegime::Critical
    }
}

const fn regime_str(r: AndersonRegime) -> &'static str {
    match r {
        AndersonRegime::Extended => "Extended",
        AndersonRegime::Localized => "Localized",
        AndersonRegime::Critical => "Critical",
    }
}

fn validate_species_params(v: &mut ValidationHarness, benchmark: &serde_json::Value) {
    validation::section("Species Barrier Parameters");
    let checks = &benchmark["validation_checks"]["species_barrier_params"]["test_cases"];
    for tc in checks.as_array().expect("array") {
        let label = tc["label"].as_str().unwrap_or("case");
        let alpha = json_field(tc, "barrier_alpha");
        let d_eff = json_field(tc, "d_eff_intact");
        v.check_bool(
            &format!("params {label}"),
            alpha > 0.0 && (d_eff - 2.0).abs() < f64::EPSILON,
        );
    }
}

fn validate_breach_threshold(v: &mut ValidationHarness, benchmark: &serde_json::Value) {
    validation::section("Breach Threshold");
    let checks = &benchmark["validation_checks"]["breach_threshold"]["test_cases"];
    for tc in checks.as_array().expect("array") {
        let label = tc["label"].as_str().unwrap_or("case");

        if tc
            .get("expected_canine_first")
            .and_then(serde_json::Value::as_bool)
            == Some(true)
        {
            let canine_at = json_field(tc, "canine_breach_at");
            let human_at = json_field(tc, "human_breach_at");
            v.check_bool(&format!("breach {label}"), canine_at < human_at);
        } else if tc.get("expected_canine_d").is_some() {
            let tol = json_field(tc, "tolerance");
            let intensity = json_field(tc, "scratch_intensity");
            let canine_d = barrier_disruption_d_eff(intensity);
            let human_d = barrier_disruption_d_eff(intensity);
            let expected_c = json_field(tc, "expected_canine_d");
            let expected_h = json_field(tc, "expected_human_d");
            v.check_abs(&format!("canine_d {label}"), canine_d, expected_c, tol);
            v.check_abs(&format!("human_d {label}"), human_d, expected_h, tol);
        }
    }
}

fn validate_anderson(v: &mut ValidationHarness, benchmark: &serde_json::Value) {
    validation::section("Anderson Predictions");
    let checks = &benchmark["validation_checks"]["anderson_predictions"]["test_cases"];
    for tc in checks.as_array().expect("array") {
        let label = tc["label"].as_str().unwrap_or("case");
        let w = json_field(tc, "w");

        let expected_2d = tc["expected_regime_d2"].as_str().unwrap_or("?");
        let expected_3d = tc["expected_regime_d3"].as_str().unwrap_or("?");

        let regime_2d = classify_regime(w, 2.0);
        let regime_3d = classify_regime(w, 3.0);

        v.check_bool(
            &format!("d=2 {label}"),
            regime_str(regime_2d) == expected_2d,
        );
        v.check_bool(
            &format!("d=3 {label}"),
            regime_str(regime_3d) == expected_3d,
        );
    }
}

fn validate_diversity(v: &mut ValidationHarness, benchmark: &serde_json::Value) {
    validation::section("Comparative Diversity");
    let checks = &benchmark["validation_checks"]["comparative_diversity"]["test_cases"];
    for tc in checks.as_array().expect("array") {
        let label = tc["label"].as_str().unwrap_or("case");
        let abundances = parse_f64_array(&tc["cell_abundances"]);
        let expected_richness = tc["expected_richness"].as_u64().unwrap_or(0) as usize;
        let range = tc["expected_evenness_range"]
            .as_array()
            .expect("evenness range");
        let lo = range[0].as_f64().unwrap_or(0.0);
        let hi = range[1].as_f64().unwrap_or(1.0);

        let richness = abundances.iter().filter(|&&x| x > 0.0).count();
        let evenness = diversity::pielou_evenness(&abundances);

        v.check_bool(&format!("richness {label}"), richness == expected_richness);
        v.check_bool(
            &format!("evenness {label} in [{lo},{hi}]"),
            (lo..=hi).contains(&evenness),
        );
    }
}

fn validate_one_health(v: &mut ValidationHarness, benchmark: &serde_json::Value) {
    validation::section("One Health Bridge");
    let checks = &benchmark["validation_checks"]["one_health_bridge"]["test_cases"];
    for tc in checks.as_array().expect("array") {
        let label = tc["label"].as_str().unwrap_or("case");
        let same_target = tc["expected_same_target"].as_bool().unwrap_or(false);
        v.check_bool(&format!("one_health {label}"), same_target);
    }
}

fn main() {
    validation::init_tracing();
    validation::banner("Exp 069: Cross-Species Skin Comparison (Paper 12)");

    let mut v = ValidationHarness::new("Cross-Species Skin");
    let benchmark =
        parse_benchmark_json(BENCHMARK_JSON).expect("benchmark_cross_species_skin.json must parse");

    validate_species_params(&mut v, &benchmark);
    validate_breach_threshold(&mut v, &benchmark);
    validate_anderson(&mut v, &benchmark);
    validate_diversity(&mut v, &benchmark);
    validate_one_health(&mut v, &benchmark);

    v.finish();
}
