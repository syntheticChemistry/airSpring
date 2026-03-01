// SPDX-License-Identifier: AGPL-3.0-or-later
#![warn(clippy::pedantic)]
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
//! Validate ET₀ three-method intercomparison (Exp 020).
//!
//! Computes PM, PT, and Hargreaves ET₀ on real Michigan Open-Meteo data
//! and validates cross-method correlations and ratios.
//!
//! References:
//!   Allen et al. (1998) FAO-56.
//!   Priestley & Taylor (1972) MWR 100(2).
//!   Jensen et al. (1990) ASCE Manual No. 70.
//!
//! script=`control/et0_intercomparison/et0_three_method.py`, commit=9a84ae5, date=2026-02-26
//! Run: `python3 control/et0_intercomparison/et0_three_method.py`

use airspring_barracuda::validation::{self, parse_benchmark_json, ValidationHarness};

const BENCHMARK_JSON: &str =
    include_str!("../../../control/et0_intercomparison/benchmark_et0_intercomparison.json");

fn validate_station(
    v: &mut ValidationHarness,
    name: &str,
    data: &serde_json::Value,
    thresholds: &serde_json::Value,
) {
    let pm_mean = data["pm_mean"].as_f64().expect("pm_mean");
    let pm_range = thresholds["pm_mean_range_mm_day"]
        .as_array()
        .expect("range");
    let pm_lo = pm_range[0].as_f64().expect("lo");
    let pm_hi = pm_range[1].as_f64().expect("hi");
    v.check_bool(
        &format!("{name}: PM mean {pm_mean:.2} in [{pm_lo}, {pm_hi}]"),
        (pm_lo..=pm_hi).contains(&pm_mean),
    );

    if let Some(r2) = data["pm_vs_openmeteo"]["r2"].as_f64() {
        let thr = thresholds["pm_vs_openmeteo_r2_min"].as_f64().expect("thr");
        v.check_bool(&format!("{name}: PM vs OM R²={r2:.4} > {thr}"), r2 > thr);
    }

    if let Some(r2) = data["pt_vs_pm"]["r2"].as_f64() {
        let thr = thresholds["pt_vs_pm_r2_min"].as_f64().expect("thr");
        v.check_bool(&format!("{name}: PT vs PM R²={r2:.4} > {thr}"), r2 > thr);
    }

    if let Some(ratio) = data["pt_pm_ratio"].as_f64() {
        let range = thresholds["pt_pm_ratio_range"].as_array().expect("range");
        let lo = range[0].as_f64().expect("lo");
        let hi = range[1].as_f64().expect("hi");
        v.check_bool(
            &format!("{name}: PT/PM={ratio:.4} in [{lo}, {hi}]"),
            (lo..=hi).contains(&ratio),
        );
    }

    if let Some(r2) = data["hg_vs_pm"]["r2"].as_f64() {
        let thr = thresholds["hg_vs_pm_r2_min"].as_f64().expect("thr");
        v.check_bool(&format!("{name}: HG vs PM R²={r2:.4} > {thr}"), r2 > thr);
    }

    if let Some(ratio) = data["hg_pm_ratio"].as_f64() {
        let range = thresholds["hg_pm_ratio_range"].as_array().expect("range");
        let lo = range[0].as_f64().expect("lo");
        let hi = range[1].as_f64().expect("hi");
        v.check_bool(
            &format!("{name}: HG/PM={ratio:.4} in [{lo}, {hi}]"),
            (lo..=hi).contains(&ratio),
        );
    }
}

fn main() {
    validation::init_tracing();
    validation::banner("ET₀ Three-Method Intercomparison");
    let mut v = ValidationHarness::new("ET₀ Three-Method Intercomparison");
    let bench = parse_benchmark_json(BENCHMARK_JSON).expect("valid benchmark JSON");
    let thresholds = &bench["thresholds"];

    let stations = bench["stations"].as_object().expect("stations object");

    for (name, data) in stations {
        validation::section(name);
        validate_station(&mut v, name, data, thresholds);
    }

    v.finish();
}
