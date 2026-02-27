// SPDX-License-Identifier: AGPL-3.0-or-later
#![warn(clippy::pedantic)]
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
//! Exp 026: USDA SCAN Soil Moisture Validation — Rust parity binary.
//!
//! Validates:
//! 1. VG retention θ(h) matches Carsel & Parrish (1988) values
//! 2. Mualem-VG K(h)/Ks matches analytical values
//! 3. Richards solver produces bounded θ profiles
//! 4. Ks ordering: sand > `silt_loam` > clay
//! 5. Analytical θ at seasonal pressure heads falls within SCAN ranges
//! 6. Depth-dependent infiltration response
//!
//! Benchmark: `control/scan_moisture/benchmark_scan_moisture.json`
//! Python baseline: `control/scan_moisture/scan_moisture_validation.py`

use airspring_barracuda::eco::richards::{
    solve_richards_1d, van_genuchten_k, van_genuchten_theta, VanGenuchtenParams,
};
use airspring_barracuda::validation::{self, parse_benchmark_json, ValidationHarness};

const BENCHMARK_JSON: &str =
    include_str!("../../../control/scan_moisture/benchmark_scan_moisture.json");

struct SoilDef {
    name: &'static str,
    params: VanGenuchtenParams,
}

const SOILS: &[SoilDef] = &[
    SoilDef {
        name: "sand",
        params: VanGenuchtenParams {
            theta_r: 0.045,
            theta_s: 0.430,
            alpha: 0.145,
            n_vg: 2.68,
            ks: 712.8,
        },
    },
    SoilDef {
        name: "silt_loam",
        params: VanGenuchtenParams {
            theta_r: 0.067,
            theta_s: 0.450,
            alpha: 0.020,
            n_vg: 1.41,
            ks: 10.8,
        },
    },
    SoilDef {
        name: "clay",
        params: VanGenuchtenParams {
            theta_r: 0.068,
            theta_s: 0.380,
            alpha: 0.008,
            n_vg: 1.09,
            ks: 4.8,
        },
    },
];

struct SeasonalProfile {
    spring_head: f64,
    summer_head: f64,
    spring_theta_lo: f64,
    spring_theta_hi: f64,
    summer_theta_lo: f64,
    summer_theta_hi: f64,
}

fn seasonal_profile(name: &str) -> Option<SeasonalProfile> {
    match name {
        "sand" => Some(SeasonalProfile {
            spring_head: -20.0,
            summer_head: -150.0,
            spring_theta_lo: 0.10,
            spring_theta_hi: 0.35,
            summer_theta_lo: 0.045,
            summer_theta_hi: 0.25,
        }),
        "silt_loam" => Some(SeasonalProfile {
            spring_head: -50.0,
            summer_head: -200.0,
            spring_theta_lo: 0.25,
            spring_theta_hi: 0.45,
            summer_theta_lo: 0.15,
            summer_theta_hi: 0.40,
        }),
        "clay" => Some(SeasonalProfile {
            spring_head: -30.0,
            summer_head: -100.0,
            spring_theta_lo: 0.28,
            spring_theta_hi: 0.38,
            summer_theta_lo: 0.20,
            summer_theta_hi: 0.38,
        }),
        _ => None,
    }
}

fn validate_retention(harness: &mut ValidationHarness, benchmark: &serde_json::Value) {
    validation::section("Retention Curves (van Genuchten 1980 Eq. 1)");

    let checks = validation::json_array(benchmark, &["retention_checks"]);
    for tc in checks {
        let soil_name = validation::json_str(tc, "soil");
        let h_cm = validation::json_field(tc, "h_cm");
        let expected = validation::json_field(tc, "expected_theta");
        let tol = validation::json_field(tc, "tolerance");

        let soil = SOILS
            .iter()
            .find(|s| s.name == soil_name)
            .unwrap_or_else(|| panic!("unknown soil: {soil_name}"));
        let p = &soil.params;
        let theta = van_genuchten_theta(h_cm, p.theta_r, p.theta_s, p.alpha, p.n_vg);
        harness.check_abs(&format!("{soil_name} h={h_cm}cm"), theta, expected, tol);
    }
}

fn validate_conductivity(harness: &mut ValidationHarness, benchmark: &serde_json::Value) {
    validation::section("Hydraulic Conductivity (Mualem-van Genuchten)");

    let checks = validation::json_array(benchmark, &["conductivity_checks"]);
    for tc in checks {
        let soil_name = validation::json_str(tc, "soil");
        let h_cm = validation::json_field(tc, "h_cm");
        let expected_ratio = validation::json_field(tc, "expected_K_ratio");
        let tol = validation::json_field(tc, "tolerance");

        let soil = SOILS
            .iter()
            .find(|s| s.name == soil_name)
            .unwrap_or_else(|| panic!("unknown soil: {soil_name}"));
        let p = &soil.params;
        let k = van_genuchten_k(h_cm, p.ks, p.theta_r, p.theta_s, p.alpha, p.n_vg);
        let ratio = k / p.ks;
        harness.check_abs(
            &format!("{soil_name} h={h_cm}cm K/Ks"),
            ratio,
            expected_ratio,
            tol,
        );
    }
}

fn validate_infiltration(harness: &mut ValidationHarness) {
    validation::section("Infiltration Dynamics (Richards solver)");

    for soil in SOILS {
        let p = &soil.params;
        let result = solve_richards_1d(
            p, 50.0,   // depth_cm
            25,     // n_nodes
            -200.0, // h_initial
            -190.0, // h_top (slightly wetter = small infiltration)
            false,  // not zero flux
            true,   // free drain bottom
            10.0,   // 10 days
            0.5,    // dt
        );

        match result {
            Ok(profiles) => {
                let last = profiles.last().expect("no profiles");
                let theta_surface = last.theta[0];
                let theta_deep = *last.theta.last().unwrap_or(&0.0);

                harness.check_bool(
                    &format!("{} surface θ in [θr, θs] ({theta_surface:.4})", soil.name),
                    theta_surface >= p.theta_r && theta_surface <= p.theta_s,
                );
                harness.check_bool(
                    &format!("{} deep θ in [θr, θs] ({theta_deep:.4})", soil.name),
                    theta_deep >= p.theta_r && theta_deep <= p.theta_s,
                );
                harness.check_bool(
                    &format!("{} solver produced {} profiles", soil.name, profiles.len()),
                    !profiles.is_empty(),
                );
            }
            Err(e) => {
                harness.check_bool(&format!("{} solver failed: {e}", soil.name), false);
            }
        }
    }
}

fn validate_drainage_ordering(harness: &mut ValidationHarness) {
    validation::section("Drainage Ordering (Ks hierarchy)");

    let ks_sand = SOILS[0].params.ks;
    let ks_silt = SOILS[1].params.ks;
    let ks_clay = SOILS[2].params.ks;

    harness.check_bool(
        &format!("Ks sand ({ks_sand:.1}) > Ks silt_loam ({ks_silt:.1})"),
        ks_sand > ks_silt,
    );
    harness.check_bool(
        &format!("Ks silt_loam ({ks_silt:.1}) > Ks clay ({ks_clay:.1})"),
        ks_silt > ks_clay,
    );

    let ps = &SOILS[0].params;
    let pl = &SOILS[1].params;
    let k_sand_wet = van_genuchten_k(-10.0, ps.ks, ps.theta_r, ps.theta_s, ps.alpha, ps.n_vg);
    let k_loam_wet = van_genuchten_k(-10.0, pl.ks, pl.theta_r, pl.theta_s, pl.alpha, pl.n_vg);

    harness.check_bool(
        &format!("K(h=-10) sand ({k_sand_wet:.2}) > silt_loam ({k_loam_wet:.2})"),
        k_sand_wet > k_loam_wet,
    );
}

fn validate_seasonal_theta(harness: &mut ValidationHarness) {
    validation::section("Seasonal θ Ranges (VG analytical at SCAN-typical pressures)");

    for soil in SOILS {
        let p = &soil.params;
        let profile = seasonal_profile(soil.name)
            .unwrap_or_else(|| panic!("no seasonal profile for soil: {}", soil.name));

        let theta_spring =
            van_genuchten_theta(profile.spring_head, p.theta_r, p.theta_s, p.alpha, p.n_vg);
        harness.check_bool(
            &format!(
                "{} spring θ(h={}) = {theta_spring:.4} in [{}, {}]",
                soil.name, profile.spring_head, profile.spring_theta_lo, profile.spring_theta_hi
            ),
            theta_spring >= profile.spring_theta_lo && theta_spring <= profile.spring_theta_hi,
        );

        let theta_summer =
            van_genuchten_theta(profile.summer_head, p.theta_r, p.theta_s, p.alpha, p.n_vg);
        harness.check_bool(
            &format!(
                "{} summer θ(h={}) = {theta_summer:.4} in [{}, {}]",
                soil.name, profile.summer_head, profile.summer_theta_lo, profile.summer_theta_hi
            ),
            theta_summer >= profile.summer_theta_lo && theta_summer <= profile.summer_theta_hi,
        );
    }
}

fn validate_depth_response(harness: &mut ValidationHarness) {
    validation::section("Depth Response (surface changes more than deep)");

    let p = &SOILS[1].params; // silt_loam
    let theta_init = van_genuchten_theta(-200.0, p.theta_r, p.theta_s, p.alpha, p.n_vg);

    let result = solve_richards_1d(
        p, 100.0,  // depth_cm
        20,     // n_nodes
        -200.0, // h_initial
        -100.0, // h_top (wetter)
        false,  // not zero flux
        true,   // free drain
        5.0,    // 5 days
        0.5,    // dt
    );

    match result {
        Ok(profiles) => {
            let last = profiles.last().expect("no profiles");
            let theta_shallow = last.theta[1];
            let theta_deep = *last.theta.last().unwrap_or(&0.0);

            let change_shallow = (theta_shallow - theta_init).abs();
            let change_deep = (theta_deep - theta_init).abs();

            harness.check_bool(
                &format!("surface Δθ ({change_shallow:.4}) >= deep Δθ ({change_deep:.4})"),
                change_shallow >= change_deep - 0.001,
            );
            harness.check_bool(
                &format!("solver produced {} profiles", profiles.len()),
                !profiles.is_empty(),
            );
        }
        Err(e) => {
            harness.check_bool(&format!("depth response solver failed: {e}"), false);
        }
    }
}

fn main() {
    validation::init_tracing();
    validation::banner("Exp 026: USDA SCAN Soil Moisture Validation (Rust)");

    let benchmark = parse_benchmark_json(BENCHMARK_JSON).expect("invalid benchmark JSON");
    let mut harness = ValidationHarness::new("SCAN Soil Moisture");

    validate_retention(&mut harness, &benchmark);
    validate_conductivity(&mut harness, &benchmark);
    validate_infiltration(&mut harness);
    validate_drainage_ordering(&mut harness);
    validate_seasonal_theta(&mut harness);
    validate_depth_response(&mut harness);

    harness.finish();
}
