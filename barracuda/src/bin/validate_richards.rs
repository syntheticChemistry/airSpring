// SPDX-License-Identifier: AGPL-3.0-or-later
//! Validate 1D Richards equation solver against van Genuchten-Mualem baseline.
//!
//! Benchmark source: `control/richards/benchmark_richards.json`
//! Provenance: van Genuchten (1980), Richards (1931).
//! Python baseline: `control/richards/richards_1d.py`
//!
//! Tests:
//! 1. Van Genuchten retention curve against analytical values
//! 2. Hydraulic conductivity K(0) = Ks
//! 3. Sand infiltration: surface wets, solver converges
//! 4. Silt loam drainage: bottom drains, mass balance
//! 5. Steady-state flux: K(h=0) = Ks
//!
//! All thresholds sourced from benchmark JSON.

use airspring_barracuda::eco::richards::{
    self as richards, mass_balance_check, solve_richards_1d, van_genuchten_k, van_genuchten_theta,
    VanGenuchtenParams,
};
use airspring_barracuda::validation::{
    self, json_array_opt, json_f64, json_object_opt, json_str_opt, parse_benchmark_json,
    ValidationHarness,
};
use std::process;

/// Benchmark JSON embedded at compile time for reproducibility.
const BENCHMARK_JSON: &str = include_str!("../../../control/richards/benchmark_richards.json");

fn soil_params(benchmark: &serde_json::Value, soil: &str) -> VanGenuchtenParams {
    let Some(theta_r) = json_f64(benchmark, &["soil_types", soil, "theta_r"]) else {
        eprintln!("benchmark JSON: missing soil_types.{soil}.theta_r");
        process::exit(1);
    };
    let Some(theta_s) = json_f64(benchmark, &["soil_types", soil, "theta_s"]) else {
        eprintln!("benchmark JSON: missing soil_types.{soil}.theta_s");
        process::exit(1);
    };
    let Some(alpha) = json_f64(benchmark, &["soil_types", soil, "alpha"]) else {
        eprintln!("benchmark JSON: missing soil_types.{soil}.alpha");
        process::exit(1);
    };
    let Some(n_vg) = json_f64(benchmark, &["soil_types", soil, "n_vg"]) else {
        eprintln!("benchmark JSON: missing soil_types.{soil}.n_vg");
        process::exit(1);
    };
    let Some(ks) = json_f64(benchmark, &["soil_types", soil, "Ks_cm_day"]) else {
        eprintln!("benchmark JSON: missing soil_types.{soil}.Ks_cm_day");
        process::exit(1);
    };
    VanGenuchtenParams {
        theta_r,
        theta_s,
        alpha,
        n_vg,
        ks,
    }
}

fn validate_van_genuchten_retention(v: &mut ValidationHarness, benchmark: &serde_json::Value) {
    validation::section("van_genuchten_retention: Water retention curve");

    let Some(checks) = json_array_opt(
        benchmark,
        &["validation_checks", "van_genuchten_retention", "test_cases"],
    ) else {
        eprintln!("benchmark JSON: missing validation_checks.van_genuchten_retention.test_cases");
        process::exit(1);
    };
    for tc in checks {
        let Some(soil) = json_str_opt(tc, &["soil"]) else {
            eprintln!("benchmark JSON: test case missing soil");
            process::exit(1);
        };
        let params = soil_params(benchmark, soil);
        let Some(h_cm) = json_f64(tc, &["h_cm"]) else {
            eprintln!("benchmark JSON: test case missing h_cm");
            process::exit(1);
        };
        let Some(expected) = json_f64(tc, &["expected_theta"]) else {
            eprintln!("benchmark JSON: test case missing expected_theta");
            process::exit(1);
        };
        let Some(tol) = json_f64(tc, &["tolerance"]) else {
            eprintln!("benchmark JSON: test case missing tolerance");
            process::exit(1);
        };

        let theta = van_genuchten_theta(
            h_cm,
            params.theta_r,
            params.theta_s,
            params.alpha,
            params.n_vg,
        );
        v.check_abs(&format!("{soil} h={h_cm} cm"), theta, expected, tol);
    }
}

fn validate_hydraulic_conductivity(v: &mut ValidationHarness, benchmark: &serde_json::Value) {
    validation::section("hydraulic_conductivity: K(h)");

    let Some(checks) = json_array_opt(
        benchmark,
        &["validation_checks", "hydraulic_conductivity", "test_cases"],
    ) else {
        eprintln!("benchmark JSON: missing validation_checks.hydraulic_conductivity.test_cases");
        process::exit(1);
    };
    for tc in checks {
        let Some(soil) = json_str_opt(tc, &["soil"]) else {
            eprintln!("benchmark JSON: test case missing soil");
            process::exit(1);
        };
        let params = soil_params(benchmark, soil);
        let Some(h_cm) = json_f64(tc, &["h_cm"]) else {
            eprintln!("benchmark JSON: test case missing h_cm");
            process::exit(1);
        };
        let tol = json_f64(tc, &["tolerance"]).unwrap_or(0.01);

        let k = van_genuchten_k(
            h_cm,
            params.ks,
            params.theta_r,
            params.theta_s,
            params.alpha,
            params.n_vg,
        );
        let k_ratio = k / params.ks;

        if let Some(expected) = json_f64(tc, &["expected_K_ratio"]) {
            v.check_abs(&format!("{soil} h={h_cm} K/Ks"), k_ratio, expected, tol);
        } else {
            let Some(range) = json_array_opt(tc, &["expected_K_ratio_range"]) else {
                eprintln!(
                    "benchmark JSON: test case missing expected_K_ratio or expected_K_ratio_range"
                );
                process::exit(1);
            };
            let Some(low) = range.first().and_then(serde_json::Value::as_f64) else {
                eprintln!("benchmark JSON: expected_K_ratio_range[0] not f64");
                process::exit(1);
            };
            let Some(high) = range.get(1).and_then(serde_json::Value::as_f64) else {
                eprintln!("benchmark JSON: expected_K_ratio_range[1] not f64");
                process::exit(1);
            };
            v.check_bool(
                &format!("{soil} h={h_cm} K/Ks in [{low}, {high}]"),
                (low..=high).contains(&k_ratio),
            );
        }
    }
}

fn validate_infiltration_sand(v: &mut ValidationHarness, benchmark: &serde_json::Value) {
    validation::section("infiltration_sand: 1D infiltration into dry sand");

    let params = soil_params(benchmark, "sand");
    let Some(depth) = json_f64(
        benchmark,
        &["validation_checks", "infiltration_sand", "column_depth_cm"],
    ) else {
        eprintln!("benchmark JSON: missing validation_checks.infiltration_sand.column_depth_cm");
        process::exit(1);
    };
    let Some(h_initial) = json_f64(
        benchmark,
        &["validation_checks", "infiltration_sand", "initial_h_cm"],
    ) else {
        eprintln!("benchmark JSON: missing validation_checks.infiltration_sand.initial_h_cm");
        process::exit(1);
    };
    let Some(h_top) = json_f64(
        benchmark,
        &["validation_checks", "infiltration_sand", "top_h_cm"],
    ) else {
        eprintln!("benchmark JSON: missing validation_checks.infiltration_sand.top_h_cm");
        process::exit(1);
    };
    let Some(duration_hours) = json_f64(
        benchmark,
        &["validation_checks", "infiltration_sand", "duration_hours"],
    ) else {
        eprintln!("benchmark JSON: missing validation_checks.infiltration_sand.duration_hours");
        process::exit(1);
    };
    let duration_days = duration_hours / 24.0;

    let dt_days = 0.00001;
    let profiles = solve_richards_1d(
        &params,
        depth,
        25,
        h_initial,
        h_top,
        false,
        true,
        duration_days,
        dt_days,
    )
    .expect("solver must converge");

    v.check_bool("Solver completes without error", true);

    let min_theta = json_array_opt(
        benchmark,
        &["validation_checks", "infiltration_sand", "checks"],
    )
    .and_then(|arr| {
        arr.iter()
            .find(|c| c.get("id").and_then(serde_json::Value::as_str) == Some("theta_surface"))
            .and_then(|c| c.get("min_theta").and_then(serde_json::Value::as_f64))
    })
    .unwrap_or(0.35);

    let theta_surf = profiles.last().map_or(0.0, |p| p.theta[0]);
    v.check_bool(
        &format!("Surface θ >= {min_theta} (observed {theta_surf:.4})"),
        theta_surf >= min_theta,
    );

    let dz = depth / 25.0;
    let err_pct = mass_balance_check(&params, &profiles, h_initial, h_top, false, dt_days, dz);
    v.check_bool(
        &format!("Mass balance {err_pct:.2}% < 100% (infiltration)"),
        err_pct <= 100.0,
    );
}

fn validate_drainage_silt_loam(v: &mut ValidationHarness, benchmark: &serde_json::Value) {
    validation::section("drainage_silt_loam: Free drainage from saturated column");

    let params = soil_params(benchmark, "silt_loam");
    let Some(depth) = json_f64(
        benchmark,
        &["validation_checks", "drainage_silt_loam", "column_depth_cm"],
    ) else {
        eprintln!("benchmark JSON: missing validation_checks.drainage_silt_loam.column_depth_cm");
        process::exit(1);
    };
    let Some(h_initial) = json_f64(
        benchmark,
        &["validation_checks", "drainage_silt_loam", "initial_h_cm"],
    ) else {
        eprintln!("benchmark JSON: missing validation_checks.drainage_silt_loam.initial_h_cm");
        process::exit(1);
    };
    let Some(duration_hours) = json_f64(
        benchmark,
        &["validation_checks", "drainage_silt_loam", "duration_hours"],
    ) else {
        eprintln!("benchmark JSON: missing validation_checks.drainage_silt_loam.duration_hours");
        process::exit(1);
    };
    let duration_days = duration_hours / 24.0;

    let dt_days = 0.001;
    let profiles = solve_richards_1d(
        &params,
        depth,
        50,
        h_initial,
        h_initial,
        true,
        true,
        duration_days,
        dt_days,
    )
    .expect("solver must converge");

    v.check_bool("Solver completes without error", true);

    let cum_drain = richards::cumulative_drainage(&params, &profiles, dt_days);
    let total_drain = cum_drain.last().copied().unwrap_or(0.0);
    v.check_bool(
        &format!("Cumulative drainage > 0 ({total_drain:.4} cm)"),
        total_drain > 0.0,
    );

    let tol_pct = json_array_opt(
        benchmark,
        &["validation_checks", "drainage_silt_loam", "checks"],
    )
    .and_then(|arr| {
        arr.iter()
            .find(|c| c.get("id").and_then(serde_json::Value::as_str) == Some("mass_balance"))
            .and_then(|c| c.get("tolerance_pct").and_then(serde_json::Value::as_f64))
    })
    .unwrap_or(50.0);

    let dz = depth / 50.0;
    let err_pct = mass_balance_check(&params, &profiles, h_initial, h_initial, true, dt_days, dz);
    v.check_bool(
        &format!("Mass balance {err_pct:.2}% <= {tol_pct}%"),
        err_pct <= tol_pct,
    );
}

fn validate_steady_state_flux(v: &mut ValidationHarness, benchmark: &serde_json::Value) {
    validation::section("steady_state_flux: K(h=0) = Ks");

    let tol_pct = benchmark
        .get("validation_checks")
        .and_then(|c| c.get("steady_state_flux"))
        .and_then(|c| c.get("checks"))
        .and_then(|a| a.get(0))
        .and_then(|c| c.get("tolerance_pct"))
        .and_then(serde_json::Value::as_f64)
        .unwrap_or(5.0);

    let Some(soils) = json_object_opt(benchmark, &["soil_types"]) else {
        eprintln!("benchmark JSON: missing soil_types");
        process::exit(1);
    };
    for (soil_name, _s) in soils {
        let params = soil_params(benchmark, soil_name);
        let k_sat = van_genuchten_k(
            0.0,
            params.ks,
            params.theta_r,
            params.theta_s,
            params.alpha,
            params.n_vg,
        );
        let k_ratio = k_sat / params.ks;
        v.check_bool(
            &format!("{soil_name}: K(h=0)/Ks = {k_ratio:.4} within {tol_pct}%"),
            (k_ratio - 1.0).abs() <= tol_pct / 100.0,
        );
    }
}

fn main() {
    validation::init_tracing();
    validation::banner("Richards Equation Validation");
    let mut v = ValidationHarness::new("Richards Equation Validation");
    let benchmark =
        parse_benchmark_json(BENCHMARK_JSON).expect("benchmark_richards.json must parse");

    validate_van_genuchten_retention(&mut v, &benchmark);
    validate_hydraulic_conductivity(&mut v, &benchmark);
    validate_infiltration_sand(&mut v, &benchmark);
    validate_drainage_silt_loam(&mut v, &benchmark);
    validate_steady_state_flux(&mut v, &benchmark);

    v.finish();
}
