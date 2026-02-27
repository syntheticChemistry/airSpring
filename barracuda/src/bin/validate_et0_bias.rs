// SPDX-License-Identifier: AGPL-3.0-or-later
#![warn(clippy::pedantic)]
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::similar_names
)]
//! Exp 039: Cross-Method ET₀ Bias Correction.
//!
//! Quantifies systematic bias of simplified ET₀ methods relative to
//! FAO-56 PM and validates linear correction factors.
//!
//! Benchmark: `control/et0_bias_correction/benchmark_et0_bias.json`
//! Baseline: `control/et0_bias_correction/et0_bias_correction.py` (24/24 PASS)

use airspring_barracuda::eco::evapotranspiration::{et0_ensemble, EnsembleInput};
use airspring_barracuda::validation::{self, json_field, parse_benchmark_json, ValidationHarness};

const BENCHMARK_JSON: &str =
    include_str!("../../../control/et0_bias_correction/benchmark_et0_bias.json");

fn opt_f64(v: &serde_json::Value, key: &str) -> Option<f64> {
    v.get(key).and_then(serde_json::Value::as_f64)
}

fn build_ensemble_input(sc: &serde_json::Value) -> EnsembleInput {
    EnsembleInput {
        tmin: opt_f64(sc, "tmin"),
        tmax: opt_f64(sc, "tmax"),
        tmean: opt_f64(sc, "tmean"),
        rs_mj: opt_f64(sc, "rs_mj"),
        wind_speed_2m: opt_f64(sc, "wind_speed_2m"),
        actual_vapour_pressure: opt_f64(sc, "e_a"),
        rh_pct: opt_f64(sc, "rh_pct"),
        elevation_m: json_field(sc, "elevation_m"),
        latitude_deg: json_field(sc, "latitude_deg"),
        day_of_year: json_field(sc, "day_of_year") as u32,
        day_length_hours: opt_f64(sc, "day_length_hours"),
    }
}

struct BiasRow {
    label: String,
    pm: f64,
    hargreaves: f64,
    makkink: f64,
    turc: f64,
    hamon: f64,
}

fn compute_bias_table(scenarios: &[serde_json::Value]) -> Vec<BiasRow> {
    scenarios
        .iter()
        .map(|sc| {
            let input = build_ensemble_input(sc);
            let r = et0_ensemble(&input);
            BiasRow {
                label: sc["label"].as_str().unwrap_or("").to_string(),
                pm: r.pm,
                hargreaves: r.hargreaves,
                makkink: r.makkink,
                turc: r.turc,
                hamon: r.hamon,
            }
        })
        .collect()
}

fn correction_factor(table: &[BiasRow], method: fn(&BiasRow) -> f64) -> f64 {
    let pm_sum: f64 = table.iter().filter(|r| r.pm > 0.0).map(|r| r.pm).sum();
    let method_sum: f64 = table
        .iter()
        .filter(|r| method(r) > 0.0)
        .map(method)
        .sum();
    if method_sum > 0.0 {
        pm_sum / method_sum
    } else {
        1.0
    }
}

fn validate_bias_quantification(v: &mut ValidationHarness, benchmark: &serde_json::Value) {
    validation::section("Bias Quantification");
    let scenarios = benchmark["validation_checks"]["bias_quantification"]["climate_scenarios"]
        .as_array()
        .expect("array");
    let table = compute_bias_table(scenarios);

    for row in &table {
        v.check_lower(&format!("{}: PM > 0", row.label), row.pm, 0.0);
    }
}

fn validate_correction_factors(v: &mut ValidationHarness, benchmark: &serde_json::Value) {
    validation::section("Correction Factor Validation");
    let scenarios = benchmark["validation_checks"]["bias_quantification"]["climate_scenarios"]
        .as_array()
        .expect("array");
    let tol_pct = json_field(
        &benchmark["validation_checks"]["correction_factor_validation"],
        "tolerance_pct",
    );
    let table = compute_bias_table(scenarios);

    let f_hg = correction_factor(&table, |r| r.hargreaves);
    let f_mk = correction_factor(&table, |r| r.makkink);
    let f_tc = correction_factor(&table, |r| r.turc);
    let f_hm = correction_factor(&table, |r| r.hamon);

    for row in &table {
        if row.pm <= 0.0 {
            continue;
        }
        let methods: [(&str, f64, f64); 4] = [
            ("hargreaves", row.hargreaves, f_hg),
            ("makkink", row.makkink, f_mk),
            ("turc", row.turc, f_tc),
            ("hamon", row.hamon, f_hm),
        ];
        for (name, val, factor) in &methods {
            let corrected = val * factor;
            let err_pct = ((corrected - row.pm) / row.pm).abs() * 100.0;
            v.check_bool(
                &format!("{}: {} corrected err={err_pct:.1}% < {tol_pct}%", row.label, name),
                err_pct < tol_pct,
            );
        }
    }
}

fn validate_structural(v: &mut ValidationHarness, benchmark: &serde_json::Value) {
    validation::section("Structural Checks");
    let scenarios = benchmark["validation_checks"]["bias_quantification"]["climate_scenarios"]
        .as_array()
        .expect("array");
    let table = compute_bias_table(scenarios);

    let f_hg = correction_factor(&table, |r| r.hargreaves);
    let f_mk = correction_factor(&table, |r| r.makkink);
    let f_tc = correction_factor(&table, |r| r.turc);
    let f_hm = correction_factor(&table, |r| r.hamon);

    v.check_bool(
        "All correction factors positive",
        f_hg > 0.0 && f_mk > 0.0 && f_tc > 0.0 && f_hm > 0.0,
    );

    let humid_overestimates = table
        .iter()
        .filter(|r| r.label.to_lowercase().contains("humid"))
        .all(|r| r.hargreaves > r.pm);
    v.check_bool("Hargreaves overestimates in humid", humid_overestimates);

    let hamon_underest_count = table
        .iter()
        .filter(|r| r.pm > 0.0 && r.hamon.is_finite())
        .filter(|r| r.hamon < r.pm)
        .count();
    let hamon_total = table
        .iter()
        .filter(|r| r.pm > 0.0 && r.hamon.is_finite())
        .count();
    // Hamon underestimates in most conditions (Xu & Singh 2002),
    // but can overestimate PM in cool conditions where wind is significant
    v.check_bool(
        &format!("Hamon underestimates in majority ({hamon_underest_count}/{hamon_total})"),
        hamon_underest_count * 2 >= hamon_total,
    );

    let avg_mak_bias: f64 = table
        .iter()
        .map(|r| ((r.makkink - r.pm) / r.pm * 100.0).abs())
        .sum::<f64>()
        / table.len() as f64;
    v.check_bool(
        &format!("Makkink avg |bias| = {avg_mak_bias:.1}% < 50%"),
        avg_mak_bias < 50.0,
    );
}

fn main() {
    validation::init_tracing();
    validation::banner("Exp 039: Cross-Method ET₀ Bias Correction");

    let mut v = ValidationHarness::new("ET₀ Bias Correction");
    let benchmark =
        parse_benchmark_json(BENCHMARK_JSON).expect("benchmark_et0_bias.json must parse");

    validate_bias_quantification(&mut v, &benchmark);
    validate_correction_factors(&mut v, &benchmark);
    validate_structural(&mut v, &benchmark);

    v.finish();
}
