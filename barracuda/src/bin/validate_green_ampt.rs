// SPDX-License-Identifier: AGPL-3.0-or-later
#![warn(clippy::pedantic)]
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
//! Exp 051: Green-Ampt (1911) Infiltration Validation.
//!
//! Validates the Green-Ampt infiltration model against analytical benchmarks,
//! soil parameter tables (Rawls et al. 1983), and physical constraints.
//!
//! Benchmark: `control/green_ampt/benchmark_green_ampt.json`
//! Baseline: `control/green_ampt/green_ampt_infiltration.py` (37/37 PASS)
//!
//! Reference: Green WH, Ampt GA (1911) J Agr Sci 4(1):1-24.

use airspring_barracuda::eco::infiltration::{
    cumulative_infiltration, infiltration_rate, ponding_time, GreenAmptParams,
};
use airspring_barracuda::tolerances;
use airspring_barracuda::validation::{self, json_field, parse_benchmark_json, ValidationHarness};

const BENCHMARK_JSON: &str = include_str!("../../../control/green_ampt/benchmark_green_ampt.json");

fn validate_analytical(v: &mut ValidationHarness, benchmark: &serde_json::Value) {
    validation::section("Analytical Benchmarks");
    let checks = &benchmark["analytical_benchmarks"];
    for tc in checks.as_array().expect("array") {
        let name = tc["name"].as_str().unwrap_or("?");
        let inputs = &tc["inputs"];
        let ks = json_field(inputs, "Ks_cm_hr");
        let psi = json_field(inputs, "psi_cm");
        let dt = json_field(inputs, "delta_theta");
        let params = GreenAmptParams {
            ks_cm_hr: ks,
            psi_cm: psi,
            delta_theta: dt,
        };

        if let Some(t) = inputs.get("t_hr").and_then(serde_json::Value::as_f64) {
            // Cumulative infiltration check
            if let Some(expected_f) = tc.get("expected_F_cm").and_then(serde_json::Value::as_f64) {
                if !tc
                    .get("expected_f_infinite")
                    .and_then(serde_json::Value::as_bool)
                    .unwrap_or(false)
                {
                    let tol = json_field(tc, "tolerance");
                    let computed = cumulative_infiltration(&params, t);
                    v.check_abs(&format!("F({name})"), computed, expected_f, tol);
                }
            }

            // Rate check
            if let Some(expected_rate) = tc
                .get("expected_f_cm_hr")
                .and_then(serde_json::Value::as_f64)
            {
                let tol = json_field(tc, "tolerance");
                let f_cum = cumulative_infiltration(&params, t);
                if f_cum > 0.0 {
                    let rate = infiltration_rate(&params, f_cum);
                    v.check_abs(&format!("f({name})"), rate, expected_rate, tol);
                }
            }

            // Asymptotic check
            if tc
                .get("expected_f_approaches_Ks")
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(false)
            {
                let f_cum = cumulative_infiltration(&params, t);
                if f_cum > 0.0 {
                    let rate = infiltration_rate(&params, f_cum);
                    let tol_r = tc
                        .get("tolerance_ratio")
                        .and_then(serde_json::Value::as_f64)
                        .unwrap_or(0.05);
                    v.check_abs(&format!("ratio({name})"), rate / ks, 1.0, tol_r);
                }
            }

            // Zero time check
            if t == 0.0
                && tc
                    .get("expected_f_infinite")
                    .and_then(serde_json::Value::as_bool)
                    .unwrap_or(false)
            {
                v.check_abs(
                    &format!("F_zero({name})"),
                    cumulative_infiltration(&params, 0.0),
                    0.0,
                    tolerances::GREEN_AMPT_ANALYTICAL.abs_tol,
                );
            }
        }

        // Ponding time check
        if let Some(i_rain) = inputs
            .get("rain_intensity_cm_hr")
            .and_then(serde_json::Value::as_f64)
        {
            let expected_tp = json_field(tc, "expected_tp_hr");
            let tol = json_field(tc, "tolerance");
            let computed = ponding_time(&params, i_rain);
            v.check_abs(&format!("tp({name})"), computed, expected_tp, tol);
        }
    }
}

fn validate_soil_params(v: &mut ValidationHarness) {
    validation::section("Soil Parameter Table");
    let soils = [
        ("Sand", GreenAmptParams::SAND),
        ("Loamy Sand", GreenAmptParams::LOAMY_SAND),
        ("Sandy Loam", GreenAmptParams::SANDY_LOAM),
        ("Loam", GreenAmptParams::LOAM),
        ("Silt Loam", GreenAmptParams::SILT_LOAM),
        ("Clay Loam", GreenAmptParams::CLAY_LOAM),
        ("Clay", GreenAmptParams::CLAY),
    ];
    for (name, p) in soils {
        v.check_bool(&format!("{name}_Ks_pos"), p.ks_cm_hr > 0.0);
        v.check_bool(&format!("{name}_psi_pos"), p.psi_cm > 0.0);
        v.check_bool(
            &format!("{name}_dt_physical"),
            p.delta_theta > 0.0 && p.delta_theta < 1.0,
        );
    }
}

fn validate_monotonicity(v: &mut ValidationHarness) {
    validation::section("Monotonicity");
    let p = GreenAmptParams {
        delta_theta: 0.312,
        ..GreenAmptParams::SANDY_LOAM
    };

    // F monotonically increasing
    let times = [0.1, 0.5, 1.0, 2.0, 4.0, 8.0, 24.0];
    let f_vals: Vec<f64> = times
        .iter()
        .map(|&t| cumulative_infiltration(&p, t))
        .collect();
    v.check_bool(
        "cumulative_monotonic",
        f_vals.windows(2).all(|w| w[0] <= w[1]),
    );

    // Rate decreasing
    let rates: Vec<f64> = f_vals
        .iter()
        .filter(|&&f| f > 0.0)
        .map(|&f| infiltration_rate(&p, f))
        .collect();
    v.check_bool("rate_decreasing", rates.windows(2).all(|w| w[0] >= w[1]));

    // Rate bounded below by Ks
    v.check_bool(
        "rate_geq_Ks",
        rates.iter().all(|&r| r >= p.ks_cm_hr - 1e-10),
    );

    // Sand > Clay at t=1hr
    let p_sand = GreenAmptParams {
        delta_theta: 0.367,
        ..GreenAmptParams::SAND
    };
    let p_clay = GreenAmptParams {
        delta_theta: 0.285,
        ..GreenAmptParams::CLAY
    };
    let f_sand = cumulative_infiltration(&p_sand, 1.0);
    let f_clay = cumulative_infiltration(&p_clay, 1.0);
    v.check_bool("sand_gt_clay", f_sand > f_clay);
}

fn main() {
    let benchmark = parse_benchmark_json(BENCHMARK_JSON).expect("valid JSON");
    let mut v = ValidationHarness::new("Exp 051: Green-Ampt (1911) Infiltration");
    validate_analytical(&mut v, &benchmark);
    validate_soil_params(&mut v);
    validate_monotonicity(&mut v);
    v.finish();
}
