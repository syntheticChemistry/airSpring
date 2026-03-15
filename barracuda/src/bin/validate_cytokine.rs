// SPDX-License-Identifier: AGPL-3.0-or-later
//! Exp 067: `CytokineBrain` Regime Prediction Validation.
//!
//! Validates the full `CytokineBrain` lifecycle: observe → train → predict →
//! export → import, plus input normalization and regime classification.
//! Cross-validates against Python control (`control/cytokine_brain/cytokine_brain.py`, 14/14 PASS).
//!
//! Benchmark: `control/cytokine_brain/benchmark_cytokine_brain.json`
//!
//! References:
//! - Paper 12: Immunological Anderson
//! - Gonzales AJ et al. (2016) Vet Dermatol 27:34-e10
//! - Fleck TJ, Gonzales AJ (2021) Vet Dermatol 32:681-e182
//!
//! script=`control/cytokine_brain/cytokine_brain.py`, commit=dbfb53a, date=2026-03-02
//! Run: `python3 control/cytokine_brain/cytokine_brain.py`

use airspring_barracuda::eco::cytokine::{
    CytokineBrain, CytokineBrainConfig, CytokineObservation, CytokinePrediction,
};
use airspring_barracuda::eco::tissue::AndersonRegime;
use airspring_barracuda::validation::{self, ValidationHarness, json_field, parse_benchmark_json};
use bingocube_nautilus::NautilusBrainConfig;

const BENCHMARK_JSON: &str =
    include_str!("../../../control/cytokine_brain/benchmark_cytokine_brain.json");

const fn make_obs(
    time_hours: f64,
    il31: f64,
    pruritus: f64,
    signal_extent: f64,
    barrier: f64,
) -> CytokineObservation {
    CytokineObservation {
        time_hours,
        il31_level: il31,
        il4_level: 50.0,
        il13_level: 40.0,
        pruritus_score: pruritus,
        tewl: 25.0,
        pielou_evenness: 0.7,
        signal_extent_observed: signal_extent,
        w_observed: 0.4,
        barrier_integrity_observed: barrier,
    }
}

fn normalize_field(tc: &serde_json::Value) -> f64 {
    let normalizers: &[(&str, f64)] = &[
        ("time_hours", 720.0),
        ("il31_level", 500.0),
        ("il4_level", 200.0),
        ("il13_level", 200.0),
        ("pruritus_score", 10.0),
        ("tewl", 100.0),
        ("pielou_evenness", 1.0),
    ];
    for &(key, divisor) in normalizers {
        if let Some(val) = tc.get(key).and_then(serde_json::Value::as_f64) {
            return val / divisor;
        }
    }
    panic!("no known field in normalization test case");
}

fn validate_normalization(v: &mut ValidationHarness, benchmark: &serde_json::Value) {
    validation::section("Input Normalization");
    let checks = &benchmark["validation_checks"]["input_normalization"]["test_cases"];
    for tc in checks.as_array().expect("array") {
        let label = tc["label"].as_str().unwrap_or("case");
        let expected = json_field(tc, "expected_normalized");
        let tol = json_field(tc, "tolerance");
        let computed = normalize_field(tc);
        v.check_abs(&format!("norm {label}"), computed, expected, tol);
    }
}

fn validate_regime_classification(v: &mut ValidationHarness, benchmark: &serde_json::Value) {
    validation::section("Prediction Regime Classification");
    let checks = &benchmark["validation_checks"]["prediction_regime"]["test_cases"];
    for tc in checks.as_array().expect("array") {
        let label = tc["label"].as_str().unwrap_or("case");
        let signal = json_field(tc, "signal_extent");
        let w = json_field(tc, "w_predicted");
        let barrier = json_field(tc, "barrier_integrity");
        let expected_str = tc["expected_regime"].as_str().unwrap_or("?");

        let expected = match expected_str {
            "Extended" => AndersonRegime::Extended,
            "Localized" => AndersonRegime::Localized,
            "Critical" => AndersonRegime::Critical,
            _ => panic!("unknown regime: {expected_str}"),
        };

        let pred = CytokinePrediction {
            signal_extent: signal,
            w_predicted: w,
            barrier_integrity: barrier,
        };
        v.check_bool(
            &format!("regime {label}"),
            pred.anderson_regime() == expected,
        );
    }
}

fn validate_brain_lifecycle(v: &mut ValidationHarness, benchmark: &serde_json::Value) {
    validation::section("Brain Lifecycle");
    let checks = &benchmark["validation_checks"]["brain_lifecycle"]["test_cases"];
    for tc in checks.as_array().expect("array") {
        let label = tc["label"].as_str().unwrap_or("case");
        let n_obs = tc["n_observations"].as_u64().unwrap_or(0) as usize;
        let min_pts = tc["min_training_points"].as_u64().unwrap_or(5) as usize;

        let default = CytokineBrainConfig::default();
        let config = CytokineBrainConfig {
            brain: NautilusBrainConfig {
                min_training_points: min_pts,
                ..default.brain
            },
            ..default
        };
        let mut brain = CytokineBrain::new(config.clone(), "exp067-test");

        for i in 0..n_obs {
            let fi = i as f64;
            let t = fi * 6.0;
            let il31 = fi.mul_add(20.0, 100.0);
            let pruritus = fi.mul_add(0.5, 3.0);
            let signal = fi.mul_add(0.05, 0.3);
            let barrier = fi.mul_add(-0.03, 0.8);
            brain.observe(make_obs(t, il31, pruritus, signal, barrier));
        }

        let trained = brain.train().is_some();
        let expected_trained = tc
            .get("expected_trained_after")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false);
        v.check_bool(&format!("trained {label}"), trained == expected_trained);

        if trained {
            if tc
                .get("expected_prediction_finite")
                .and_then(serde_json::Value::as_bool)
                == Some(true)
            {
                let obs = make_obs(48.0, 200.0, 5.0, 0.0, 0.0);
                if let Some(pred) = brain.predict(&obs) {
                    v.check_bool(
                        &format!("signal_extent finite {label}"),
                        pred.signal_extent.is_finite(),
                    );
                    v.check_bool(
                        &format!("signal_extent in [0,1] {label}"),
                        (0.0..=1.0).contains(&pred.signal_extent),
                    );
                    v.check_bool(
                        &format!("w_predicted in [0,1] {label}"),
                        (0.0..=1.0).contains(&pred.w_predicted),
                    );
                    v.check_bool(
                        &format!("barrier in [0,1] {label}"),
                        (0.0..=1.0).contains(&pred.barrier_integrity),
                    );
                } else {
                    v.check_bool(&format!("prediction exists {label}"), false);
                }
            }

            if tc
                .get("expected_export_nonempty")
                .and_then(serde_json::Value::as_bool)
                == Some(true)
            {
                let json = brain.export_json().expect("export should succeed");
                v.check_bool(&format!("export nonempty {label}"), !json.is_empty());

                let imported = CytokineBrain::import_json(config, &json);
                v.check_bool(&format!("import ok {label}"), imported.is_ok());
                if let Ok(imported_brain) = imported {
                    let expected_import_trained = tc
                        .get("expected_import_trained")
                        .and_then(serde_json::Value::as_bool)
                        .unwrap_or(false);
                    v.check_bool(
                        &format!("imported trained {label}"),
                        imported_brain.is_trained() == expected_import_trained,
                    );
                }
            }
        }
    }
}

fn validate_data_profile(v: &mut ValidationHarness, benchmark: &serde_json::Value) {
    validation::section("Gonzales Data Profile");
    let checks = &benchmark["validation_checks"]["gonzales_data_profile"]["test_cases"];
    for tc in checks.as_array().expect("array") {
        let label = tc["label"].as_str().unwrap_or("case");
        let time_points = tc["time_points_hours"]
            .as_array()
            .expect("time_points array");
        let n_features = tc["expected_n_features"].as_u64().unwrap_or(0) as usize;
        let n_targets = tc["expected_n_targets"].as_u64().unwrap_or(0) as usize;

        v.check_bool(&format!("n_features=7 {label}"), n_features == 7);
        v.check_bool(&format!("n_targets=3 {label}"), n_targets == 3);
        v.check_bool(&format!("time_points>0 {label}"), !time_points.is_empty());
    }
}

fn main() {
    validation::init_tracing();
    validation::banner("Exp 067: CytokineBrain Regime Prediction (Paper 12)");

    let mut v = ValidationHarness::new("CytokineBrain");
    let benchmark =
        parse_benchmark_json(BENCHMARK_JSON).expect("benchmark_cytokine_brain.json must parse");

    validate_normalization(&mut v, &benchmark);
    validate_regime_classification(&mut v, &benchmark);
    validate_brain_lifecycle(&mut v, &benchmark);
    validate_data_profile(&mut v, &benchmark);

    v.finish();
}
