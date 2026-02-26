// SPDX-License-Identifier: AGPL-3.0-or-later
//! Validate Priestley-Taylor ET₀ against Python baseline (Exp 019).
//!
//! Priestley CHB, Taylor RJ (1972) "On the assessment of surface heat flux
//! and evaporation using large-scale parameters." Monthly Weather Review
//! 100(2): 81-92.
//!
//! Digitized: 2026-02-26, commit: 9a84ae5.

use airspring_barracuda::eco::evapotranspiration::{self as et, DailyEt0Input};
use airspring_barracuda::validation::{self, parse_benchmark_json, ValidationHarness};

const BENCHMARK_JSON: &str =
    include_str!("../../../control/priestley_taylor/benchmark_priestley_taylor.json");

fn parse_daily_input(obj: &serde_json::Value) -> DailyEt0Input {
    DailyEt0Input {
        tmin: obj["tmin"].as_f64().expect("tmin"),
        tmax: obj["tmax"].as_f64().expect("tmax"),
        tmean: obj["tmean"].as_f64(),
        solar_radiation: obj["solar_rad"].as_f64().expect("solar_rad"),
        wind_speed_2m: obj["wind_2m"].as_f64().expect("wind_2m"),
        actual_vapour_pressure: obj["ea"].as_f64().expect("ea"),
        elevation_m: obj["elev"].as_f64().expect("elev"),
        latitude_deg: obj["lat_deg"].as_f64().expect("lat_deg"),
        day_of_year: u32::try_from(obj["doy"].as_u64().expect("doy")).expect("doy fits u32"),
    }
}

fn validate_analytical(v: &mut ValidationHarness, bench: &serde_json::Value) {
    validation::section("Analytical properties");
    let tests = bench["analytical_tests"]
        .as_array()
        .expect("analytical_tests array");
    for tc in tests {
        let name = tc["name"].as_str().expect("name");
        let rn = tc["rn"].as_f64().expect("rn");
        let ghf = tc["g"].as_f64().expect("g");
        let tmean = tc["tmean_c"].as_f64().expect("tmean_c");
        let elev = tc["elevation_m"].as_f64().expect("elevation_m");
        let expected = tc["expected_pt_et0"].as_f64().expect("expected_pt_et0");
        let tol = tc["tolerance"].as_f64().expect("tolerance");

        let computed = et::priestley_taylor_et0(rn, ghf, tmean, elev);
        v.check_abs(&format!("analytical: {name}"), computed, expected, tol);
    }
}

fn validate_uccle(v: &mut ValidationHarness, bench: &serde_json::Value) {
    validation::section("FAO-56 Example 18 cross-validation");
    let uccle = &bench["fao56_example_18"];
    let input = parse_daily_input(&uccle["inputs"]);
    let (priestley, penman) = et::daily_et0_pt_and_pm(&input);

    let want_priestley = uccle["expected"]["pt_et0"].as_f64().expect("pt_et0");
    let want_penman = uccle["expected"]["pm_et0"].as_f64().expect("pm_et0");
    v.check_abs(
        "Uccle PT ET₀",
        priestley,
        want_priestley,
        uccle["tolerance_pt"].as_f64().expect("tolerance_pt"),
    );
    v.check_abs(
        "Uccle PM ET₀",
        penman.et0,
        want_penman,
        uccle["tolerance_pm"].as_f64().expect("tolerance_pm"),
    );

    let range = uccle["expected"]["pt_pm_ratio_range"]
        .as_array()
        .expect("ratio range");
    let lo = range[0].as_f64().expect("lo");
    let hi = range[1].as_f64().expect("hi");
    let ratio = priestley / penman.et0;
    v.check_bool(
        &format!("Uccle PT/PM ratio {ratio:.4} in [{lo}, {hi}]"),
        (lo..=hi).contains(&ratio),
    );
}

fn validate_climate_gradient(v: &mut ValidationHarness, bench: &serde_json::Value) {
    validation::section("Climate gradient (humid → arid)");
    let cases = bench["climate_gradient"]["cases"]
        .as_array()
        .expect("gradient cases");
    let mut prev_ratio: Option<f64> = None;
    for case in cases {
        let name = case["name"].as_str().expect("name");
        let input = parse_daily_input(&case["inputs"]);
        let (priestley, penman) = et::daily_et0_pt_and_pm(&input);
        let expected = case["expected_pt_et0"].as_f64().expect("expected");
        let tol = case["tolerance"].as_f64().expect("tolerance");
        v.check_abs(
            &format!("gradient: {name} PT ET₀"),
            priestley,
            expected,
            tol,
        );

        let current_ratio = priestley / penman.et0;
        if let Some(prev) = prev_ratio {
            v.check_bool(
                &format!("gradient: PT/PM decreasing {prev:.4} → {current_ratio:.4}"),
                current_ratio < prev,
            );
        }
        prev_ratio = Some(current_ratio);
    }
}

fn validate_monotonicity(
    v: &mut ValidationHarness,
    tests: &[serde_json::Value],
    prefix: &str,
    label_fn: &dyn Fn(&serde_json::Value) -> String,
) {
    let mut prev_val: Option<f64> = None;
    for tc in tests {
        let rn = tc["rn"].as_f64().expect("rn");
        let tmean = tc["tmean_c"].as_f64().expect("tmean_c");
        let elev = tc["elevation_m"].as_f64().expect("elevation_m");
        let expected = tc["expected_pt_et0"].as_f64().expect("expected");
        let tol = tc["tolerance"].as_f64().expect("tolerance");

        let computed = et::priestley_taylor_et0(rn, 0.0, tmean, elev);
        v.check_abs(&label_fn(tc), computed, expected, tol);

        if let Some(prev) = prev_val {
            v.check_bool(
                &format!("{prefix}: increasing {prev:.4} → {computed:.4}"),
                computed > prev,
            );
        }
        prev_val = Some(computed);
    }
}

fn main() {
    validation::init_tracing();
    validation::banner("Priestley-Taylor ET₀ Validation");
    let mut v = ValidationHarness::new("Priestley-Taylor ET₀ Validation");
    let bench = parse_benchmark_json(BENCHMARK_JSON).expect("valid benchmark JSON");

    validate_analytical(&mut v, &bench);
    validate_uccle(&mut v, &bench);
    validate_climate_gradient(&mut v, &bench);

    validation::section("Monotonicity");
    let mono = bench["monotonicity_tests"]["increasing_rn"]
        .as_array()
        .expect("monotonicity");
    validate_monotonicity(&mut v, mono, "mono", &|tc| {
        format!("mono: Rn={:.1}", tc["rn"].as_f64().unwrap_or(0.0))
    });

    validation::section("Temperature sensitivity");
    let temps = bench["temperature_sensitivity"]["increasing_temp"]
        .as_array()
        .expect("temp sensitivity");
    validate_monotonicity(&mut v, temps, "temp", &|tc| {
        format!("temp: T={:.0}°C", tc["tmean_c"].as_f64().unwrap_or(0.0))
    });

    v.finish();
}
