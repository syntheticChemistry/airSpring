// SPDX-License-Identifier: AGPL-3.0-or-later
#![warn(clippy::pedantic)]
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::too_many_lines
)]
//! Exp 061: Cross-Spring Shannon H' Diversity Gradient.
//!
//! Validates the coupling between soil moisture (θ), Anderson QS regime,
//! and microbial diversity (Shannon H', Simpson, Bray-Curtis) along a
//! synthetic moisture gradient modeled after PRJNA481146 (Ein Harod).
//!
//! Cross-Spring: airSpring (θ, Anderson) × wetSpring (diversity metrics)
//! Data: Synthetic OTU tables (pending NestGate NCBI provider for real SRA data)

use airspring_barracuda::eco::anderson;
use airspring_barracuda::eco::diversity;
use airspring_barracuda::validation::{self, ValidationHarness};

const BENCHMARK: &str =
    include_str!("../../../control/ncbi_diversity/benchmark_ncbi_diversity.json");

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter("info")
        .init();

    let bench: serde_json::Value = serde_json::from_str(BENCHMARK).expect("benchmark JSON");
    let mut v = ValidationHarness::new("Exp 061: Cross-Spring Shannon H' Diversity Gradient");

    let theta_r = 0.095;
    let theta_s = 0.41;

    let samples = bench["synthetic_otu_tables"]
        .as_array()
        .expect("otu tables");

    let mut shannon_values: Vec<(f64, f64)> = Vec::new();
    let mut all_abundances: Vec<Vec<f64>> = Vec::new();

    for sample in samples {
        let theta = sample["theta"].as_f64().unwrap_or(0.0);
        let label = sample["label"].as_str().unwrap_or("");
        let abundances: Vec<f64> = sample["abundances"]
            .as_array()
            .unwrap_or(&Vec::new())
            .iter()
            .map(|v| v.as_f64().unwrap_or(0.0))
            .collect();

        let sh_range = sample["expected_shannon_range"]
            .as_array()
            .expect("shannon range");
        let sh_min = sh_range[0].as_f64().unwrap_or(0.0);
        let sh_max = sh_range[1].as_f64().unwrap_or(5.0);

        let ev_range = sample["expected_evenness_range"]
            .as_array()
            .expect("evenness range");
        let ev_min = ev_range[0].as_f64().unwrap_or(0.0);
        let ev_max = ev_range[1].as_f64().unwrap_or(1.0);

        // Compute diversity metrics
        let alpha = diversity::alpha_diversity(&abundances);
        let h = alpha.shannon;
        let evenness = diversity::pielou_evenness(&abundances);
        let simpson = diversity::simpson(&abundances);

        shannon_values.push((theta, h));
        all_abundances.push(abundances);

        // Shannon in expected range
        v.check_lower(
            &format!("{label}_shannon_min"),
            h,
            sh_min,
        );
        v.check_upper(
            &format!("{label}_shannon_max"),
            h,
            sh_max,
        );

        // Evenness in expected range
        v.check_lower(
            &format!("{label}_evenness_min"),
            evenness,
            ev_min,
        );
        v.check_upper(
            &format!("{label}_evenness_max"),
            evenness,
            ev_max,
        );

        // Simpson should always be in [0, 1]
        v.check_lower(
            &format!("{label}_simpson_positive"),
            simpson,
            0.0,
        );
        v.check_upper(
            &format!("{label}_simpson_max"),
            simpson,
            1.0,
        );

        // Anderson coupling
        let chain = anderson::coupling_chain(theta, theta_r, theta_s);
        v.check_bool(
            &format!("{label}_se_valid"),
            (0.0..=1.0).contains(&chain.se),
        );

        v.check_bool(
            &format!("{label}_d_eff_non_negative"),
            chain.d_eff >= 0.0,
        );
    }

    validation::section("Moisture-Diversity Coupling");

    // Shannon should peak at moderate moisture (hump-shaped curve)
    if shannon_values.len() >= 5 {
        let mid_idx = shannon_values.len() / 2;
        let mid_h = shannon_values[mid_idx].1;
        let first_h = shannon_values[0].1;
        let last_h = shannon_values[shannon_values.len() - 1].1;

        v.check_lower(
            "shannon_mid_gt_dry",
            mid_h - first_h,
            0.0,
        );
        v.check_lower(
            "shannon_mid_gt_saturated",
            mid_h - last_h,
            0.0,
        );
    }

    // Bray-Curtis: dry vs wet should show meaningful dissimilarity
    if all_abundances.len() >= 2 {
        let bc = diversity::bray_curtis(
            &all_abundances[0],
            &all_abundances[all_abundances.len() - 1],
        );
        let bc_threshold = bench["expected_trends"]["bray_curtis_dry_vs_wet_above"]
            .as_f64()
            .unwrap_or(0.3);
        v.check_lower("bray_curtis_dry_wet", bc, bc_threshold);
    }

    // Anderson regime at extremes
    let dry_chain = anderson::coupling_chain(0.10, theta_r, theta_s);
    let wet_chain = anderson::coupling_chain(0.35, theta_r, theta_s);
    v.check_bool(
        "anderson_dry_localized",
        dry_chain.regime == anderson::QsRegime::Localized,
    );
    v.check_bool(
        "anderson_wet_extended",
        wet_chain.regime == anderson::QsRegime::Extended,
    );

    // Bray-Curtis pairwise matrix should be symmetric (M×M flat)
    let n = all_abundances.len();
    if n >= 3 {
        let bc_matrix = diversity::bray_curtis_matrix(&all_abundances);
        v.check_bool(
            "bray_curtis_matrix_size",
            bc_matrix.len() == n * n,
        );

        let all_valid = bc_matrix.iter().all(|&bc| (0.0..=1.0).contains(&bc));
        v.check_bool("bray_curtis_all_valid", all_valid);
    }

    v.finish();
}
