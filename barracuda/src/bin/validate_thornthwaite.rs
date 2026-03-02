// SPDX-License-Identifier: AGPL-3.0-or-later
#![warn(clippy::pedantic)]
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
//! Validate Thornthwaite (1948) monthly ET₀ against Python baseline (Exp 021).
//!
//! Thornthwaite C.W. (1948) "An approach toward a rational classification
//! of climate." Geographical Review, 38(1), 55-94.

use airspring_barracuda::eco::evapotranspiration as et;
use airspring_barracuda::tolerances::THORNTHWAITE_ANALYTICAL;
use airspring_barracuda::validation::{self, parse_benchmark_json, ValidationHarness};

const BENCHMARK_JSON: &str =
    include_str!("../../../control/thornthwaite/benchmark_thornthwaite.json");

fn parse_monthly_temps(arr: &serde_json::Value) -> [f64; 12] {
    let v = arr.as_array().expect("monthly_tmean_c array");
    assert_eq!(v.len(), 12, "need 12 monthly temperatures");
    let mut temps = [0.0; 12];
    for (i, val) in v.iter().enumerate() {
        temps[i] = val.as_f64().expect("temperature");
    }
    temps
}

fn validate_analytical(v: &mut ValidationHarness, bench: &serde_json::Value) {
    validation::section("Analytical properties");
    let an = &bench["analytical"];

    let hi_25 = et::monthly_heat_index_term(25.0);
    v.check_abs(
        "heat_index_term(25°C)",
        hi_25,
        an["heat_index_25C"]
            .as_f64()
            .expect("benchmark heat_index_25C must be f64"),
        THORNTHWAITE_ANALYTICAL.abs_tol,
    );

    let hi_annual = et::annual_heat_index(&[25.0; 12]);
    v.check_abs(
        "annual_heat_index(uniform 25°C)",
        hi_annual,
        an["heat_index_annual_uniform_25C"]
            .as_f64()
            .expect("benchmark heat_index_annual_uniform_25C must be f64"),
        THORNTHWAITE_ANALYTICAL.abs_tol * 10.0,
    );

    let a = et::thornthwaite_exponent(hi_annual);
    v.check_abs(
        "exponent(uniform 25°C)",
        a,
        an["exponent_uniform_25C"]
            .as_f64()
            .expect("benchmark exponent_uniform_25C must be f64"),
        THORNTHWAITE_ANALYTICAL.abs_tol,
    );

    let pet_unadj = et::thornthwaite_unadjusted_et0(25.0, hi_annual, a);
    v.check_abs(
        "unadjusted_et0(25°C)",
        pet_unadj,
        an["unadjusted_et0_25C"]
            .as_f64()
            .expect("benchmark unadjusted_et0_25C must be f64"),
        0.01,
    );

    let freezing = et::thornthwaite_unadjusted_et0(-5.0, 50.0, 1.5);
    v.check_abs("freezing → 0", freezing, 0.0, f64::EPSILON);
}

fn validate_station(v: &mut ValidationHarness, bench: &serde_json::Value, key: &str) {
    validation::section(&format!("Station: {key}"));
    let station = &bench[key];
    let temps = parse_monthly_temps(&station["monthly_tmean_c"]);
    let lat = station["latitude"]
        .as_f64()
        .expect("benchmark station latitude must be f64");
    let tol = station["tol"]
        .as_f64()
        .expect("benchmark station tol must be f64");

    let hi = et::annual_heat_index(&temps);
    v.check_abs(
        &format!("{key}: heat_index"),
        hi,
        station["heat_index"]
            .as_f64()
            .expect("benchmark station heat_index must be f64"),
        THORNTHWAITE_ANALYTICAL.abs_tol * 100.0,
    );

    let a = et::thornthwaite_exponent(hi);
    v.check_abs(
        &format!("{key}: exponent_a"),
        a,
        station["exponent_a"]
            .as_f64()
            .expect("benchmark station exponent_a must be f64"),
        THORNTHWAITE_ANALYTICAL.abs_tol,
    );

    let et0 = et::thornthwaite_monthly_et0(&temps, lat);
    let expected: Vec<f64> = station["monthly_et0_mm"]
        .as_array()
        .expect("benchmark monthly_et0_mm must be array")
        .iter()
        .map(|x| {
            x.as_f64()
                .expect("benchmark monthly_et0_mm element must be f64")
        })
        .collect();

    for (m, (computed, want)) in et0.iter().zip(expected.iter()).enumerate() {
        v.check_abs(&format!("{key}: month {m} ET₀"), *computed, *want, tol);
    }

    let annual: f64 = et0.iter().sum();
    // Annual sum accumulates daylight-hour rounding across 12 months
    let annual_tol = tol * 4.0;
    v.check_abs(
        &format!("{key}: annual ET₀"),
        annual,
        station["annual_et0_mm"]
            .as_f64()
            .expect("benchmark station annual_et0_mm must be f64"),
        annual_tol,
    );
}

fn validate_monotonicity(v: &mut ValidationHarness, bench: &serde_json::Value) {
    validation::section("Monotonicity");
    let temps: Vec<f64> = bench["monotonicity_temps"]
        .as_array()
        .expect("benchmark monotonicity_temps must be array")
        .iter()
        .map(|x| {
            x.as_f64()
                .expect("benchmark monotonicity_temps element must be f64")
        })
        .collect();
    let lat = bench["monotonicity_latitude"]
        .as_f64()
        .expect("benchmark monotonicity_latitude must be f64");

    let mut prev_annual = 0.0_f64;
    for &t in &temps {
        let monthly = et::thornthwaite_monthly_et0(&[t; 12], lat);
        let annual: f64 = monthly.iter().sum();
        v.check_bool(
            &format!("T={t:.0}°C → annual {annual:.0} > prev {prev_annual:.0}"),
            annual > prev_annual,
        );
        prev_annual = annual;
    }
}

fn validate_edge_cases(v: &mut ValidationHarness, bench: &serde_json::Value) {
    validation::section("Edge cases");
    let edges = &bench["edge_cases"];

    let frozen_temps = parse_monthly_temps(&edges["all_frozen"]["monthly_tmean_c"]);
    let et0_frozen = et::thornthwaite_monthly_et0(&frozen_temps, 45.0);
    let annual_frozen: f64 = et0_frozen.iter().sum();
    v.check_abs("all_frozen → 0", annual_frozen, 0.0, f64::EPSILON);

    let tropical_temps = parse_monthly_temps(&edges["tropical_uniform"]["monthly_tmean_c"]);
    let tropical_lat = edges["tropical_uniform"]["latitude"]
        .as_f64()
        .expect("benchmark tropical_uniform latitude must be f64");
    let range = edges["tropical_uniform"]["annual_range"]
        .as_array()
        .expect("benchmark tropical_uniform annual_range must be array");
    let lo = range[0]
        .as_f64()
        .expect("benchmark annual_range[0] must be f64");
    let hi = range[1]
        .as_f64()
        .expect("benchmark annual_range[1] must be f64");
    let et0_tropical = et::thornthwaite_monthly_et0(&tropical_temps, tropical_lat);
    let annual_tropical: f64 = et0_tropical.iter().sum();
    v.check_bool(
        &format!("tropical annual {annual_tropical:.0} in [{lo:.0}, {hi:.0}]"),
        (lo..=hi).contains(&annual_tropical),
    );
}

fn validate_pattern(v: &mut ValidationHarness, bench: &serde_json::Value) {
    validation::section("Seasonal pattern checks");
    let thresholds = &bench["thresholds"];

    for key in &["east_lansing", "wooster"] {
        let station = &bench[key];
        let temps = parse_monthly_temps(&station["monthly_tmean_c"]);
        let lat = station["latitude"]
            .as_f64()
            .expect("benchmark station latitude must be f64");
        let et0 = et::thornthwaite_monthly_et0(&temps, lat);
        let annual: f64 = et0.iter().sum();

        let range = thresholds["annual_et0_range_mm"]
            .as_array()
            .expect("benchmark annual_et0_range_mm must be array");
        let lo = range[0]
            .as_f64()
            .expect("benchmark annual_et0_range_mm[0] must be f64");
        let hi = range[1]
            .as_f64()
            .expect("benchmark annual_et0_range_mm[1] must be f64");
        v.check_bool(
            &format!("{key}: annual {annual:.0} in [{lo:.0}, {hi:.0}]"),
            (lo..=hi).contains(&annual),
        );

        let summer: f64 = et0[5..8].iter().sum();
        let winter = et0[0] + et0[1] + et0[11];
        v.check_bool(
            &format!("{key}: summer {summer:.0} > winter {winter:.0}"),
            summer > winter,
        );

        let growing: f64 = et0[4..9].iter().sum();
        let min_frac = thresholds["growing_season_fraction_min"]
            .as_f64()
            .expect("benchmark growing_season_fraction_min must be f64");
        let frac = growing / annual;
        v.check_bool(
            &format!("{key}: growing fraction {frac:.2} ≥ {min_frac}"),
            frac >= min_frac,
        );

        let peak = et0
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).expect("non-NaN ET₀ comparison"))
            .expect("non-empty monthly ET₀ for peak month")
            .0;
        v.check_bool(
            &format!("{key}: peak month {peak} in [5,7]"),
            (5..=7).contains(&peak),
        );
    }
}

fn main() {
    validation::init_tracing();
    validation::banner("Thornthwaite Monthly ET₀ Validation");
    let mut v = ValidationHarness::new("Thornthwaite Monthly ET₀");
    let bench = parse_benchmark_json(BENCHMARK_JSON).expect("valid benchmark JSON");

    validate_analytical(&mut v, &bench);
    validate_station(&mut v, &bench, "east_lansing");
    validate_station(&mut v, &bench, "wooster");
    validate_monotonicity(&mut v, &bench);
    validate_edge_cases(&mut v, &bench);
    validate_pattern(&mut v, &bench);

    v.finish();
}
