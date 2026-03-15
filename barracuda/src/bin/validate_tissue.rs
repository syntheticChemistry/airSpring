// SPDX-License-Identifier: AGPL-3.0-or-later
//! Exp 066: Tissue Diversity Profiling Validation.
//!
//! Validates cell-type diversity → Anderson disorder W mapping for
//! immunological tissue (Paper 12). Cross-validates Rust against
//! Python control (`control/tissue_diversity/tissue_diversity.py`, 30/30 PASS).
//!
//! Benchmark: `control/tissue_diversity/benchmark_tissue_diversity.json`
//!
//! References:
//! - Paper 01: Anderson QS — `W_c` thresholds
//! - Paper 06: No-till dimensional collapse
//! - Paper 12: Immunological Anderson — cytokine propagation
//! - Pielou (1966) J Theoretical Biology 13:131-144
//! - `McCandless` et al. (2014) Vet Immunol Immunopathol 157:42-48
//!
//! script=`control/tissue_diversity/tissue_diversity.py`, commit=dbfb53a, date=2026-03-02
//! Run: `python3 control/tissue_diversity/tissue_diversity.py`

use airspring_barracuda::eco::tissue::{
    AndersonRegime, CellTypeAbundance, SkinCompartment, analyze_tissue_disorder,
    barrier_disruption_d_eff, multi_compartment_analysis,
};
use airspring_barracuda::gpu::diversity::GpuDiversity;
use airspring_barracuda::validation::{self, ValidationHarness, json_field, parse_benchmark_json};

const BENCHMARK_JSON: &str =
    include_str!("../../../control/tissue_diversity/benchmark_tissue_diversity.json");

fn parse_f64_array(val: &serde_json::Value) -> Vec<f64> {
    val.as_array()
        .expect("array")
        .iter()
        .map(|v| v.as_f64().expect("f64"))
        .collect()
}

fn make_cell_types(abundances: &[f64]) -> Vec<CellTypeAbundance> {
    abundances
        .iter()
        .enumerate()
        .map(|(i, &a)| CellTypeAbundance {
            cell_type: format!("type_{i}"),
            abundance: a,
        })
        .collect()
}

fn parse_compartment(s: &str) -> SkinCompartment {
    match s {
        "Epidermis" => SkinCompartment::Epidermis,
        "PapillaryDermis" => SkinCompartment::PapillaryDermis,
        "ReticularDermis" => SkinCompartment::ReticularDermis,
        _ => panic!("unknown compartment: {s}"),
    }
}

fn validate_tissue_shannon(v: &mut ValidationHarness, benchmark: &serde_json::Value) {
    validation::section("Tissue Shannon Entropy");
    let engine = GpuDiversity::cpu();
    let checks = &benchmark["validation_checks"]["tissue_shannon"]["test_cases"];
    for tc in checks.as_array().expect("array") {
        let label = tc["label"].as_str().unwrap_or("case");
        let abundances = parse_f64_array(&tc["abundances"]);
        let expected = json_field(tc, "expected_shannon");
        let tol = json_field(tc, "tolerance");

        let cell_types = make_cell_types(&abundances);
        let result = analyze_tissue_disorder(&cell_types, SkinCompartment::Epidermis, &engine)
            .expect("tissue analysis should succeed");
        v.check_abs(
            &format!("H' {label}"),
            result.diversity.shannon,
            expected,
            tol,
        );
    }
}

fn validate_tissue_pielou(v: &mut ValidationHarness, benchmark: &serde_json::Value) {
    validation::section("Tissue Pielou Evenness");
    let engine = GpuDiversity::cpu();
    let checks = &benchmark["validation_checks"]["tissue_pielou"]["test_cases"];
    for tc in checks.as_array().expect("array") {
        let label = tc["label"].as_str().unwrap_or("case");
        let abundances = parse_f64_array(&tc["abundances"]);
        let expected = json_field(tc, "expected_pielou");
        let tol = json_field(tc, "tolerance");

        let cell_types = make_cell_types(&abundances);
        let result = analyze_tissue_disorder(&cell_types, SkinCompartment::Epidermis, &engine)
            .expect("tissue analysis should succeed");
        v.check_abs(
            &format!("J' {label}"),
            result.diversity.evenness,
            expected,
            tol,
        );
    }
}

fn validate_anderson_w(v: &mut ValidationHarness, benchmark: &serde_json::Value) {
    validation::section("Anderson W Effective");
    let engine = GpuDiversity::cpu();
    let checks = &benchmark["validation_checks"]["anderson_w_effective"]["test_cases"];
    for tc in checks.as_array().expect("array") {
        let label = tc["label"].as_str().unwrap_or("case");
        let abundances = parse_f64_array(&tc["abundances"]);
        let expected = json_field(tc, "expected_w");
        let tol = json_field(tc, "tolerance");

        let cell_types = make_cell_types(&abundances);
        let result = analyze_tissue_disorder(&cell_types, SkinCompartment::Epidermis, &engine)
            .expect("tissue analysis should succeed");
        v.check_abs(&format!("W {label}"), result.w_effective, expected, tol);
    }
}

fn validate_regime(v: &mut ValidationHarness, benchmark: &serde_json::Value) {
    validation::section("Anderson Regime Classification");
    let checks = &benchmark["validation_checks"]["anderson_regime"]["test_cases"];
    for tc in checks.as_array().expect("array") {
        let label = tc["label"].as_str().unwrap_or("case");
        let expected_str = tc["expected_regime"].as_str().unwrap_or("?");

        let expected = match expected_str {
            "Extended" => AndersonRegime::Extended,
            "Localized" => AndersonRegime::Localized,
            "Critical" => AndersonRegime::Critical,
            _ => panic!("unknown regime: {expected_str}"),
        };

        let w = json_field(tc, "w_effective");
        let d = json_field(tc, "d_eff");

        let regime = if d < 2.5 {
            let w_c = 4.0;
            let margin = 0.1 * w_c;
            if w > w_c + margin {
                AndersonRegime::Localized
            } else if w < w_c - margin {
                AndersonRegime::Extended
            } else {
                AndersonRegime::Critical
            }
        } else {
            let w_c = 16.26;
            let margin = 0.1 * w_c;
            if w > w_c + margin {
                AndersonRegime::Localized
            } else if w < w_c - margin {
                AndersonRegime::Extended
            } else {
                AndersonRegime::Critical
            }
        };

        v.check_bool(&format!("regime {label}"), regime == expected);
    }
}

fn validate_barrier(v: &mut ValidationHarness, benchmark: &serde_json::Value) {
    validation::section("Barrier Disruption d_eff");
    let checks = &benchmark["validation_checks"]["barrier_disruption"]["test_cases"];
    for tc in checks.as_array().expect("array") {
        let label = tc["label"].as_str().unwrap_or("case");
        let breach_fraction = json_field(tc, "breach_fraction");
        let expected = json_field(tc, "expected_d_eff");
        let tol = json_field(tc, "tolerance");

        let computed = barrier_disruption_d_eff(breach_fraction);
        v.check_abs(&format!("d_eff {label}"), computed, expected, tol);
    }
}

fn validate_compartments(v: &mut ValidationHarness, benchmark: &serde_json::Value) {
    validation::section("Compartment Dimensions");
    let checks = &benchmark["validation_checks"]["compartment_dimensions"]["test_cases"];
    for tc in checks.as_array().expect("array") {
        let label = tc["label"].as_str().unwrap_or("case");
        let compartment_str = tc["compartment"].as_str().unwrap_or("?");
        let expected = json_field(tc, "expected_d");
        let tol = json_field(tc, "tolerance");

        let compartment = parse_compartment(compartment_str);
        let computed = compartment.effective_dimension_intact();
        v.check_abs(&format!("d {label}"), computed, expected, tol);
    }
}

fn validate_multi_compartment(v: &mut ValidationHarness, benchmark: &serde_json::Value) {
    validation::section("Multi-Compartment Analysis");
    let engine = GpuDiversity::cpu();
    let checks = &benchmark["validation_checks"]["multi_compartment"]["test_cases"];
    for tc in checks.as_array().expect("array") {
        let label = tc["label"].as_str().unwrap_or("case");
        let expected_count = tc["expected_count"].as_u64().unwrap_or(0) as usize;

        let compartments_json = tc["compartments"].as_array().expect("compartments array");
        let compartments: Vec<(SkinCompartment, Vec<CellTypeAbundance>)> = compartments_json
            .iter()
            .map(|c| {
                let comp = parse_compartment(c["compartment"].as_str().unwrap_or("Epidermis"));
                let abundances = parse_f64_array(&c["abundances"]);
                let cell_types = make_cell_types(&abundances);
                (comp, cell_types)
            })
            .collect();

        let results = multi_compartment_analysis(&compartments, &engine)
            .expect("multi-compartment analysis should succeed");

        v.check_bool(&format!("count {label}"), results.len() == expected_count);
    }
}

fn main() {
    validation::init_tracing();
    validation::banner("Exp 066: Tissue Diversity Profiling (Paper 12)");

    let mut v = ValidationHarness::new("Tissue Diversity");
    let benchmark =
        parse_benchmark_json(BENCHMARK_JSON).expect("benchmark_tissue_diversity.json must parse");

    validate_tissue_shannon(&mut v, &benchmark);
    validate_tissue_pielou(&mut v, &benchmark);
    validate_anderson_w(&mut v, &benchmark);
    validate_regime(&mut v, &benchmark);
    validate_barrier(&mut v, &benchmark);
    validate_compartments(&mut v, &benchmark);
    validate_multi_compartment(&mut v, &benchmark);

    v.finish();
}
