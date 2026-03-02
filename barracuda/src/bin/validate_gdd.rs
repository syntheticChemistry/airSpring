// SPDX-License-Identifier: AGPL-3.0-or-later
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
//! script=`control/gdd/growing_degree_days.py`, commit=8c3953b, date=2026-02-27
//! Run: `python3 control/gdd/growing_degree_days.py`

use airspring_barracuda::eco::crop::{
    accumulated_gdd_avg, accumulated_gdd_clamp, gdd_avg, gdd_clamp, kc_from_gdd, CropType,
};
use airspring_barracuda::tolerances::GDD_EXACT;
use airspring_barracuda::validation::{self, parse_benchmark_json, ValidationHarness};

const BENCHMARK_JSON: &str = include_str!("../../../control/gdd/benchmark_gdd.json");

fn generate_season_data(bench: &serde_json::Value) -> (Vec<f64>, Vec<f64>) {
    let profiles = &bench["east_lansing_season"]["monthly_profiles"];
    let months = ["apr", "may", "jun", "jul", "aug", "sep", "oct"];
    let mut tmax = Vec::new();
    let mut tmin = Vec::new();
    for m in &months {
        let mp = &profiles[*m];
        let days = mp["days"].as_u64().unwrap() as usize;
        let tx = mp["tmax"].as_f64().unwrap();
        let tn = mp["tmin"].as_f64().unwrap();
        for _ in 0..days {
            tmax.push(tx);
            tmin.push(tn);
        }
    }
    (tmax, tmin)
}

fn validate_analytical(v: &mut ValidationHarness, bench: &serde_json::Value) {
    validation::section("Analytical: GDD average method");
    let cases = bench["analytical"]["gdd_avg"]
        .as_array()
        .expect("gdd_avg cases");
    for tc in cases {
        let tmax = tc["tmax"].as_f64().unwrap();
        let tmin = tc["tmin"].as_f64().unwrap();
        let tbase = tc["tbase"].as_f64().unwrap();
        let expected = tc["expected"].as_f64().unwrap();
        v.check_abs(
            &format!("avg: Tmax={tmax}, Tmin={tmin}, Tbase={tbase}"),
            gdd_avg(tmax, tmin, tbase),
            expected,
            GDD_EXACT.abs_tol,
        );
    }

    validation::section("Analytical: GDD clamp method");
    let cases = bench["analytical"]["gdd_clamp"]
        .as_array()
        .expect("gdd_clamp cases");
    for tc in cases {
        let tmax = tc["tmax"].as_f64().unwrap();
        let tmin = tc["tmin"].as_f64().unwrap();
        let tbase = tc["tbase"].as_f64().unwrap();
        let tceil = tc["tceil"].as_f64().unwrap();
        let expected = tc["expected"].as_f64().unwrap();
        v.check_abs(
            &format!("clamp: Tmax={tmax}, Tmin={tmin}, Tbase={tbase}, Tceil={tceil}"),
            gdd_clamp(tmax, tmin, tbase, tceil),
            expected,
            GDD_EXACT.abs_tol,
        );
    }
}

fn validate_accumulation(v: &mut ValidationHarness, bench: &serde_json::Value) {
    validation::section("Season accumulation");
    let (tmax, tmin) = generate_season_data(bench);
    let acc = &bench["accumulation"];
    let tol = acc["tol"].as_f64().unwrap();

    let cum_avg = accumulated_gdd_avg(&tmax, &tmin, 10.0).expect("matched tmax/tmin");
    v.check_abs(
        "corn avg total GDD",
        *cum_avg.last().unwrap(),
        acc["corn_avg_total_gdd"].as_f64().unwrap(),
        tol,
    );

    let cum_clamp = accumulated_gdd_clamp(&tmax, &tmin, 10.0, 30.0).expect("matched tmax/tmin");
    v.check_abs(
        "corn clamp total GDD",
        *cum_clamp.last().unwrap(),
        acc["corn_clamp_total_gdd"].as_f64().unwrap(),
        tol,
    );

    let cum_alfalfa = accumulated_gdd_avg(&tmax, &tmin, 5.0).expect("matched tmax/tmin");
    v.check_abs(
        "alfalfa avg total GDD",
        *cum_alfalfa.last().unwrap(),
        acc["alfalfa_avg_total_gdd"].as_f64().unwrap(),
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
    v.check_bool(
        &format!(
            "alfalfa GDD ({:.0}) > corn GDD ({:.0})",
            cum_alfalfa.last().unwrap(),
            cum_avg.last().unwrap()
        ),
        cum_alfalfa.last().unwrap() > cum_avg.last().unwrap(),
    );
}

fn validate_phenology(v: &mut ValidationHarness, bench: &serde_json::Value) {
    validation::section("Phenological stage (Kc from GDD)");
    let pheno = &bench["phenology"];
    let tol = pheno["tol"].as_f64().unwrap();
    let corn = CropType::Corn.gdd_params();

    let cases = pheno["corn_kc_at_gdd"].as_array().unwrap();
    for tc in cases {
        let gdd = tc["gdd"].as_f64().unwrap();
        let expected = tc["expected_kc"].as_f64().unwrap();
        let kc = kc_from_gdd(gdd, &corn.kc_stages_gdd, &corn.kc_values).expect("matched stages/kc");
        v.check_abs(&format!("corn Kc at GDD={gdd:.0}"), kc, expected, tol);
    }
}

fn validate_pattern(v: &mut ValidationHarness, bench: &serde_json::Value) {
    validation::section("Seasonal pattern checks");
    let (tmax, tmin) = generate_season_data(bench);
    let cum = accumulated_gdd_avg(&tmax, &tmin, 10.0).expect("matched tmax/tmin");
    let total = *cum.last().unwrap();

    let range = &bench["thresholds"]["corn_gdd_range"];
    let lo = range[0].as_f64().unwrap();
    let hi = range[1].as_f64().unwrap();
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
    let cum_clamp = accumulated_gdd_clamp(&tmax, &tmin, 10.0, 30.0).expect("matched tmax/tmin");
    let diff = (cum.last().unwrap() - cum_clamp.last().unwrap()).abs();
    let max_diff = bench["thresholds"]["method_diff_max"].as_f64().unwrap();
    v.check_bool(
        &format!("avg-clamp diff {diff:.0} < {max_diff:.0}"),
        diff < max_diff,
    );
}

fn main() {
    validation::init_tracing();
    validation::banner("Growing Degree Days Validation");
    let mut v = ValidationHarness::new("Growing Degree Days");
    let bench = parse_benchmark_json(BENCHMARK_JSON).expect("valid benchmark JSON");

    validate_analytical(&mut v, &bench);
    validate_accumulation(&mut v, &bench);
    validate_phenology(&mut v, &bench);
    validate_pattern(&mut v, &bench);

    v.finish();
}
