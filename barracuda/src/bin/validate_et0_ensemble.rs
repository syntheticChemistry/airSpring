// SPDX-License-Identifier: AGPL-3.0-or-later
#![warn(clippy::pedantic)]
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
//! Exp 037: ET₀ Ensemble Consensus (6-Method Weighted) Validation.
//!
//! Validates the multi-method ensemble that combines 6 daily ET₀ methods
//! (PM, PT, Hargreaves, Makkink, Turc, Hamon) into a data-adaptive
//! consensus estimate. Thornthwaite excluded (monthly method).
//!
//! Benchmark: `control/et0_ensemble/benchmark_et0_ensemble.json`
//! Baseline: `control/et0_ensemble/et0_ensemble.py` (9/9 PASS)
//!
//! References:
//! - Oudin et al. (2005) J Hydrol 303:290-306
//! - Droogers & Allen (2002) Irrig Drain Syst 16:33-45
//!
//! script=`control/et0_ensemble/et0_ensemble.py`, commit=97e7533, date=2026-02-28
//! Run: `python3 control/et0_ensemble/et0_ensemble.py`

use airspring_barracuda::eco::evapotranspiration::{et0_ensemble, EnsembleInput};
use airspring_barracuda::validation::{self, json_field, parse_benchmark_json, ValidationHarness};

const BENCHMARK_JSON: &str =
    include_str!("../../../control/et0_ensemble/benchmark_et0_ensemble.json");

fn opt_field(v: &serde_json::Value, key: &str) -> Option<f64> {
    v.get(key).and_then(serde_json::Value::as_f64)
}

fn build_input(tc: &serde_json::Value) -> EnsembleInput {
    EnsembleInput {
        tmin: opt_field(tc, "tmin"),
        tmax: opt_field(tc, "tmax"),
        tmean: opt_field(tc, "tmean"),
        rs_mj: opt_field(tc, "rs_mj"),
        wind_speed_2m: opt_field(tc, "wind_speed_2m"),
        actual_vapour_pressure: opt_field(tc, "e_a"),
        rh_pct: opt_field(tc, "rh_pct"),
        elevation_m: opt_field(tc, "elevation_m").unwrap_or(100.0),
        latitude_deg: json_field(tc, "latitude_deg"),
        day_of_year: json_field(tc, "day_of_year") as u32,
        day_length_hours: opt_field(tc, "day_length_hours"),
    }
}

fn validate_full_weather(v: &mut ValidationHarness, benchmark: &serde_json::Value) {
    validation::section("Full Weather Ensemble");
    let tests = &benchmark["validation_checks"]["full_weather_ensemble"]["test_cases"];
    for tc in tests.as_array().expect("array") {
        let label = tc["label"].as_str().unwrap_or("test");
        let input = build_input(tc);
        let result = et0_ensemble(&input);
        let tol = json_field(tc, "tolerance");

        v.check_bool(&format!("{label}: n_methods >= 5"), result.n_methods >= 5);
        v.check_lower(&format!("{label}: consensus > 0"), result.consensus, 0.0);

        if let Some(pm_ref) = opt_field(tc, "pm_reference") {
            v.check_abs(
                &format!("{label}: consensus near PM"),
                result.consensus,
                pm_ref,
                tol,
            );
        }
    }
}

fn validate_temp_only(v: &mut ValidationHarness, benchmark: &serde_json::Value) {
    validation::section("Temperature-Only Ensemble");
    let tests = &benchmark["validation_checks"]["temperature_only_ensemble"]["test_cases"];
    for tc in tests.as_array().expect("array") {
        let label = tc["label"].as_str().unwrap_or("test");
        let input = build_input(tc);
        let result = et0_ensemble(&input);

        v.check_bool(&format!("{label}: n_methods >= 2"), result.n_methods >= 2);
        v.check_bool(&format!("{label}: consensus >= 0"), result.consensus >= 0.0);
        v.check_bool(
            &format!("{label}: PM is NaN (no radiation data)"),
            result.pm.is_nan(),
        );
    }
}

fn validate_method_ranking(v: &mut ValidationHarness, benchmark: &serde_json::Value) {
    validation::section("Method Ranking");
    let tests = &benchmark["validation_checks"]["method_ranking"]["test_cases"];
    for tc in tests.as_array().expect("array") {
        let label = tc["label"].as_str().unwrap_or("test");
        let check = tc["check"].as_str().unwrap_or("");
        let input = build_input(tc);
        let result = et0_ensemble(&input);

        match check {
            "spread_positive" => {
                v.check_lower(&format!("{label}: spread > 0"), result.spread, 0.0);
            }
            "consensus_in_range" => {
                let methods = [
                    result.pm,
                    result.pt,
                    result.hargreaves,
                    result.makkink,
                    result.turc,
                    result.hamon,
                ];
                let valid: Vec<f64> = methods.iter().copied().filter(|x| x.is_finite()).collect();
                let min_v = valid.iter().copied().fold(f64::INFINITY, f64::min);
                let max_v = valid.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                v.check_bool(
                    &format!("{label}: consensus in [{min_v:.3}, {max_v:.3}]"),
                    result.consensus >= min_v && result.consensus <= max_v,
                );
            }
            _ => {}
        }
    }
}

fn validate_monotonicity(v: &mut ValidationHarness, benchmark: &serde_json::Value) {
    validation::section("Monotonicity");
    let tests = &benchmark["validation_checks"]["monotonicity"]["test_cases"];
    for tc in tests.as_array().expect("array") {
        let base_t = json_field(tc, "base_tmean");
        let step_t = json_field(tc, "step_tmean");
        let tmin_off = json_field(tc, "tmin_offset");
        let tmax_off = json_field(tc, "tmax_offset");

        let low = et0_ensemble(&EnsembleInput {
            tmin: Some(base_t + tmin_off),
            tmax: Some(base_t + tmax_off),
            tmean: Some(base_t),
            rs_mj: opt_field(tc, "rs_mj"),
            wind_speed_2m: opt_field(tc, "wind_speed_2m"),
            actual_vapour_pressure: opt_field(tc, "e_a"),
            rh_pct: opt_field(tc, "rh_pct"),
            elevation_m: json_field(tc, "elevation_m"),
            latitude_deg: json_field(tc, "latitude_deg"),
            day_of_year: json_field(tc, "day_of_year") as u32,
            day_length_hours: opt_field(tc, "day_length_hours"),
        });

        let high = et0_ensemble(&EnsembleInput {
            tmin: Some(step_t + tmin_off),
            tmax: Some(step_t + tmax_off),
            tmean: Some(step_t),
            rs_mj: opt_field(tc, "rs_mj"),
            wind_speed_2m: opt_field(tc, "wind_speed_2m"),
            actual_vapour_pressure: opt_field(tc, "e_a"),
            rh_pct: opt_field(tc, "rh_pct"),
            elevation_m: json_field(tc, "elevation_m"),
            latitude_deg: json_field(tc, "latitude_deg"),
            day_of_year: json_field(tc, "day_of_year") as u32,
            day_length_hours: opt_field(tc, "day_length_hours"),
        });

        v.check_lower(
            &format!("T={base_t}→{step_t}: ensemble increases"),
            high.consensus,
            low.consensus,
        );
    }
}

fn main() {
    validation::init_tracing();
    validation::banner("Exp 037: ET₀ Ensemble Consensus (6-Method)");

    let mut v = ValidationHarness::new("ET₀ Ensemble");
    let benchmark =
        parse_benchmark_json(BENCHMARK_JSON).expect("benchmark_et0_ensemble.json must parse");

    validate_full_weather(&mut v, &benchmark);
    validate_temp_only(&mut v, &benchmark);
    validate_method_ranking(&mut v, &benchmark);
    validate_monotonicity(&mut v, &benchmark);

    v.finish();
}
