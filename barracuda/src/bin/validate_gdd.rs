// SPDX-License-Identifier: AGPL-3.0-or-later
#![deny(clippy::unwrap_used)]
#![warn(clippy::pedantic)]
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
//! Validate Growing Degree Days (GDD) against Python baseline (Exp 022).
//!
//! `McMaster` GS, Wilhelm WW (1997) "Growing degree-days: one equation,
//! two interpretations." Agricultural and Forest Meteorology, 87:291-300.
//!
//! Provenance: script=`control/gdd/growing_degree_days.py`, commit=fad2e1b, date=2026-02-27

use airspring_barracuda::eco::crop::{
    accumulated_gdd_avg, accumulated_gdd_clamp, gdd_avg, gdd_clamp, kc_from_gdd, CropType,
};
use airspring_barracuda::tolerances::GDD_EXACT;
use airspring_barracuda::validation::{
    self, json_array, json_f64_required, json_field, json_u64_required, parse_benchmark_json,
    ValidationHarness,
};

const BENCHMARK_JSON: &str = include_str!("../../../control/gdd/benchmark_gdd.json");

fn generate_season_data(bench: &serde_json::Value) -> (Vec<f64>, Vec<f64>) {
    let months = ["apr", "may", "jun", "jul", "aug", "sep", "oct"];
    let mut tmax = Vec::new();
    let mut tmin = Vec::new();
    for m in &months {
        let days = json_u64_required(
            bench,
            &["east_lansing_season", "monthly_profiles", m, "days"],
        ) as usize;
        let tx = json_f64_required(
            bench,
            &["east_lansing_season", "monthly_profiles", m, "tmax"],
        );
        let tn = json_f64_required(
            bench,
            &["east_lansing_season", "monthly_profiles", m, "tmin"],
        );
        for _ in 0..days {
            tmax.push(tx);
            tmin.push(tn);
        }
    }
    (tmax, tmin)
}

fn validate_analytical(v: &mut ValidationHarness, bench: &serde_json::Value) {
    validation::section("Analytical: GDD average method");
    let cases = json_array(bench, &["analytical", "gdd_avg"]);
    for tc in cases {
        let tmax = json_field(tc, "tmax");
        let tmin = json_field(tc, "tmin");
        let tbase = json_field(tc, "tbase");
        let expected = json_field(tc, "expected");
        v.check_abs(
            &format!("avg: Tmax={tmax}, Tmin={tmin}, Tbase={tbase}"),
            gdd_avg(tmax, tmin, tbase),
            expected,
            GDD_EXACT.abs_tol,
        );
    }

    validation::section("Analytical: GDD clamp method");
    let cases = json_array(bench, &["analytical", "gdd_clamp"]);
    for tc in cases {
        let tmax = json_field(tc, "tmax");
        let tmin = json_field(tc, "tmin");
        let tbase = json_field(tc, "tbase");
        let tceil = json_field(tc, "tceil");
        let expected = json_field(tc, "expected");
        v.check_abs(
            &format!("clamp: Tmax={tmax}, Tmin={tmin}, Tbase={tbase}, Tceil={tceil}"),
            gdd_clamp(tmax, tmin, tbase, tceil),
            expected,
            GDD_EXACT.abs_tol,
        );
    }
}

fn last_or_exit(v: &[f64], msg: &str) -> f64 {
    v.last().copied().unwrap_or_else(|| {
        eprintln!("FATAL: {msg}");
        std::process::exit(1);
    })
}

fn validate_accumulation(v: &mut ValidationHarness, bench: &serde_json::Value) {
    validation::section("Season accumulation");
    let (tmax, tmin) = generate_season_data(bench);
    let tol = json_f64_required(bench, &["accumulation", "tol"]);

    let cum_avg = match accumulated_gdd_avg(&tmax, &tmin, 10.0) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("FATAL: accumulated_gdd_avg failed: {e}");
            std::process::exit(1);
        }
    };
    v.check_abs(
        "corn avg total GDD",
        last_or_exit(&cum_avg, "accumulation produced empty"),
        json_f64_required(bench, &["accumulation", "corn_avg_total_gdd"]),
        tol,
    );

    let cum_clamp = match accumulated_gdd_clamp(&tmax, &tmin, 10.0, 30.0) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("FATAL: accumulated_gdd_clamp failed: {e}");
            std::process::exit(1);
        }
    };
    v.check_abs(
        "corn clamp total GDD",
        last_or_exit(&cum_clamp, "accumulation produced empty"),
        json_f64_required(bench, &["accumulation", "corn_clamp_total_gdd"]),
        tol,
    );

    let cum_alfalfa = match accumulated_gdd_avg(&tmax, &tmin, 5.0) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("FATAL: accumulated_gdd_avg failed: {e}");
            std::process::exit(1);
        }
    };
    v.check_abs(
        "alfalfa avg total GDD",
        last_or_exit(&cum_alfalfa, "accumulation produced empty"),
        json_f64_required(bench, &["accumulation", "alfalfa_avg_total_gdd"]),
        tol,
    );

    // Monotonicity
    for i in 0..cum_avg.len() - 1 {
        if cum_avg[i] > cum_avg[i + 1] {
            v.check_bool(&format!("monotonic at day {i}"), false);
            break;
        }
    }
    v.check_bool("accumulation monotonic", true);

    // Alfalfa (Tbase=5) > corn (Tbase=10) on same data
    let alfalfa_last = last_or_exit(&cum_alfalfa, "alfalfa accumulation empty");
    let corn_last = last_or_exit(&cum_avg, "corn accumulation empty");
    v.check_bool(
        &format!("alfalfa GDD ({alfalfa_last:.0}) > corn GDD ({corn_last:.0})"),
        alfalfa_last > corn_last,
    );
}

fn validate_phenology(v: &mut ValidationHarness, bench: &serde_json::Value) {
    validation::section("Phenological stage (Kc from GDD)");
    let tol = json_f64_required(bench, &["phenology", "tol"]);
    let corn = CropType::Corn.gdd_params();

    let cases = json_array(bench, &["phenology", "corn_kc_at_gdd"]);
    for tc in cases {
        let gdd = json_field(tc, "gdd");
        let expected = json_field(tc, "expected_kc");
        let kc = match kc_from_gdd(gdd, &corn.kc_stages_gdd, &corn.kc_values) {
            Ok(k) => k,
            Err(e) => {
                eprintln!("FATAL: kc_from_gdd failed: {e}");
                std::process::exit(1);
            }
        };
        v.check_abs(&format!("corn Kc at GDD={gdd:.0}"), kc, expected, tol);
    }
}

fn validate_pattern(v: &mut ValidationHarness, bench: &serde_json::Value) {
    validation::section("Seasonal pattern checks");
    let (tmax, tmin) = generate_season_data(bench);
    let cum = match accumulated_gdd_avg(&tmax, &tmin, 10.0) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("FATAL: accumulated_gdd_avg failed: {e}");
            std::process::exit(1);
        }
    };
    let total = last_or_exit(&cum, "accumulation produced empty");

    let range = json_array(bench, &["thresholds", "corn_gdd_range"]);
    let lo = json_f64_required(&range[0], &[]);
    let hi = json_f64_required(&range[1], &[]);
    v.check_bool(
        &format!("corn total GDD {total:.0} in [{lo:.0}, {hi:.0}]"),
        (lo..=hi).contains(&total),
    );

    // July contributes more than April
    let apr_end = 30_usize;
    let jul_start = 30 + 31 + 30; // Apr+May+Jun
    let jul_end = jul_start + 31;
    let jul_gdd = cum[jul_end - 1] - cum[jul_start - 1];
    let apr_gdd = cum[apr_end - 1];
    v.check_bool(
        &format!("July GDD ({jul_gdd:.0}) > April GDD ({apr_gdd:.0})"),
        jul_gdd > apr_gdd,
    );

    // Method comparison
    let cum_clamp = match accumulated_gdd_clamp(&tmax, &tmin, 10.0, 30.0) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("FATAL: accumulated_gdd_clamp failed: {e}");
            std::process::exit(1);
        }
    };
    let diff =
        (last_or_exit(&cum, "cum empty") - last_or_exit(&cum_clamp, "cum_clamp empty")).abs();
    let max_diff = json_f64_required(bench, &["thresholds", "method_diff_max"]);
    v.check_bool(
        &format!("avg-clamp diff {diff:.0} < {max_diff:.0}"),
        diff < max_diff,
    );
}

fn main() {
    validation::init_tracing();
    validation::banner("Growing Degree Days Validation");
    let mut v = ValidationHarness::new("Growing Degree Days");
    let bench = match parse_benchmark_json(BENCHMARK_JSON) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("FATAL: invalid benchmark JSON: {e}");
            std::process::exit(1);
        }
    };

    validate_analytical(&mut v, &bench);
    validate_accumulation(&mut v, &bench);
    validate_phenology(&mut v, &bench);
    validate_pattern(&mut v, &bench);

    v.finish();
}
