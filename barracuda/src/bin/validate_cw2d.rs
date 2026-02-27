// SPDX-License-Identifier: AGPL-3.0-or-later
#![warn(clippy::pedantic)]
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
//! Validate Richards equation on constructed wetland media (HYDRUS CW2D parameters).
//!
//! Benchmark source: `control/cw2d/benchmark_cw2d.json`
//! Provenance: Dong et al. (2019), Šimůnek et al. (2012), van Genuchten (1980).
//! Python baseline: `control/cw2d/cw2d_richards.py`
//!
//! Tests:
//! 1. VG retention curves for gravel, coarse sand, organic, fine gravel
//! 2. Mualem-VG conductivity at key pressure heads
//! 3. Gravel infiltration: fast drainage, solver converges
//! 4. Organic drainage: high retention, positive drainage
//! 5. Mass balance on CW2D media simulations

use airspring_barracuda::eco::richards::{
    self as richards, mass_balance_check, solve_richards_1d, van_genuchten_k, van_genuchten_theta,
    VanGenuchtenParams,
};
use airspring_barracuda::validation::{
    self, json_array_opt, json_f64, json_str_opt, parse_benchmark_json, ValidationHarness,
};
use std::process;

const BENCHMARK_JSON: &str = include_str!("../../../control/cw2d/benchmark_cw2d.json");

fn media_params(benchmark: &serde_json::Value, media: &str) -> VanGenuchtenParams {
    let path = |field: &str| -> f64 {
        json_f64(benchmark, &["cw2d_media", media, field]).unwrap_or_else(|| {
            eprintln!("benchmark JSON: missing cw2d_media.{media}.{field}");
            process::exit(1);
        })
    };
    VanGenuchtenParams {
        theta_r: path("theta_r"),
        theta_s: path("theta_s"),
        alpha: path("alpha"),
        n_vg: path("n_vg"),
        ks: path("Ks_cm_day"),
    }
}

fn validate_retention(v: &mut ValidationHarness, benchmark: &serde_json::Value) {
    validation::section("CW2D Retention Curves");

    let checks = json_array_opt(
        benchmark,
        &["validation_checks", "cw2d_retention_curves", "test_cases"],
    )
    .expect("missing cw2d_retention_curves.test_cases");

    for tc in checks {
        let media = json_str_opt(tc, &["media"]).expect("missing media");
        let h_cm = json_f64(tc, &["h_cm"]).expect("missing h_cm");
        let expected = json_f64(tc, &["expected_theta"]).expect("missing expected_theta");
        let tol = json_f64(tc, &["tolerance"]).expect("missing tolerance");

        let p = media_params(benchmark, media);
        let theta = van_genuchten_theta(h_cm, p.theta_r, p.theta_s, p.alpha, p.n_vg);
        v.check_abs(&format!("θ({media}, h={h_cm})"), theta, expected, tol);
    }
}

fn validate_conductivity(v: &mut ValidationHarness, benchmark: &serde_json::Value) {
    validation::section("CW2D Hydraulic Conductivity");

    let checks = json_array_opt(
        benchmark,
        &["validation_checks", "cw2d_conductivity", "test_cases"],
    )
    .expect("missing cw2d_conductivity.test_cases");

    for tc in checks {
        let media = json_str_opt(tc, &["media"]).expect("missing media");
        let h_cm = json_f64(tc, &["h_cm"]).expect("missing h_cm");
        let p = media_params(benchmark, media);

        let k = van_genuchten_k(h_cm, p.ks, p.theta_r, p.theta_s, p.alpha, p.n_vg);
        let k_ratio = k / p.ks;

        if let Some(expected) = json_f64(tc, &["expected_K_ratio"]) {
            let tol = json_f64(tc, &["tolerance"]).unwrap_or(0.001);
            v.check_abs(
                &format!("K_ratio({media}, h={h_cm})"),
                k_ratio,
                expected,
                tol,
            );
        } else if let Some(range) = json_array_opt(tc, &["K_ratio_range"]) {
            let lo = range[0].as_f64().expect("range lo");
            let hi = range[1].as_f64().expect("range hi");
            v.check_bool(
                &format!("K_ratio({media}, h={h_cm}) in [{lo}, {hi}]"),
                k_ratio >= lo && k_ratio <= hi,
            );
        }
    }
}

fn validate_gravel_infiltration(v: &mut ValidationHarness, benchmark: &serde_json::Value) {
    validation::section("CW2D Gravel Infiltration");

    let spec = &benchmark["validation_checks"]["cw2d_gravel_infiltration"];
    let p = media_params(benchmark, "gravel");
    let depth = json_f64(spec, &["column_depth_cm"]).expect("depth");
    let n_nodes = json_f64(spec, &["n_nodes"]).expect("n_nodes") as usize;
    let h_init = json_f64(spec, &["initial_h_cm"]).expect("h_init");
    let h_top = json_f64(spec, &["top_h_cm"]).expect("h_top");
    let dur_h = json_f64(spec, &["duration_hours"]).expect("duration");
    let dur_days = dur_h / 24.0;
    let dt_days = 0.0001;

    let result = solve_richards_1d(
        &p, depth, n_nodes, h_init, h_top, false, true, dur_days, dt_days,
    );

    match result {
        Ok(profiles) => {
            v.check_bool("gravel solver converges", true);

            let min_theta = json_array_opt(spec, &["checks"])
                .and_then(|arr| {
                    arr.iter()
                        .find(|c| {
                            c.get("id").and_then(serde_json::Value::as_str) == Some("surface_wets")
                        })
                        .and_then(|c| c.get("min_theta").and_then(serde_json::Value::as_f64))
                })
                .unwrap_or(0.15);

            let theta_surf = profiles.last().map_or(0.0, |pr| pr.theta[0]);
            v.check_bool(
                &format!("gravel surface θ={theta_surf:.4} >= {min_theta}"),
                theta_surf >= min_theta,
            );

            let drainage = richards::cumulative_drainage(&p, &profiles, dt_days);
            let total = drainage.last().copied().unwrap_or(0.0);
            v.check_bool(&format!("gravel drainage > 0 ({total:.6} cm)"), total > 0.0);
        }
        Err(e) => {
            v.check_bool(&format!("gravel solver converges (err: {e})"), false);
            v.check_bool("gravel surface_wets (solver failed)", false);
            v.check_bool("gravel drainage_starts (solver failed)", false);
        }
    }
}

fn validate_organic_drainage(v: &mut ValidationHarness, benchmark: &serde_json::Value) {
    validation::section("CW2D Organic Layer Drainage");

    let spec = &benchmark["validation_checks"]["cw2d_organic_drainage"];
    let p = media_params(benchmark, "organic_substrate");
    let depth = json_f64(spec, &["column_depth_cm"]).expect("depth");
    let n_nodes = json_f64(spec, &["n_nodes"]).expect("n_nodes") as usize;
    let h_init = json_f64(spec, &["initial_h_cm"]).expect("h_init");
    let dur_h = json_f64(spec, &["duration_hours"]).expect("duration");
    let dur_days = dur_h / 24.0;
    let dt_days = 0.005;

    let result = solve_richards_1d(
        &p, depth, n_nodes, h_init, h_init, true, true, dur_days, dt_days,
    );

    match result {
        Ok(profiles) => {
            v.check_bool("organic solver converges", true);

            let drainage = richards::cumulative_drainage(&p, &profiles, dt_days);
            let total = drainage.last().copied().unwrap_or(0.0);
            v.check_bool(
                &format!("organic drainage > 0 ({total:.4} cm)"),
                total > 0.0,
            );

            let avg_theta = profiles.last().map_or(0.0, |pr| {
                pr.theta.iter().sum::<f64>() / pr.theta.len() as f64
            });
            let threshold = p.theta_r + 0.1;
            v.check_bool(
                &format!("organic avg θ={avg_theta:.4} >= {threshold:.4}"),
                avg_theta >= threshold,
            );
        }
        Err(e) => {
            v.check_bool(&format!("organic solver converges (err: {e})"), false);
            v.check_bool("organic drainage (solver failed)", false);
            v.check_bool("organic retention (solver failed)", false);
        }
    }
}

fn validate_mass_balance(v: &mut ValidationHarness, benchmark: &serde_json::Value) {
    validation::section("CW2D Mass Balance");

    let checks = json_array_opt(
        benchmark,
        &["validation_checks", "cw2d_mass_balance", "test_cases"],
    )
    .expect("missing cw2d_mass_balance.test_cases");

    for tc in checks {
        let media = json_str_opt(tc, &["media"]).expect("missing media");
        let p = media_params(benchmark, media);
        let depth = json_f64(tc, &["column_depth_cm"]).expect("depth");
        let n_nodes = json_f64(tc, &["n_nodes"]).expect("n_nodes") as usize;
        let h_init = json_f64(tc, &["initial_h_cm"]).expect("h_init");
        let h_top = json_f64(tc, &["top_h_cm"]).expect("h_top");
        let zero_flux = tc
            .get("zero_flux_top")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(true);
        let dur_h = json_f64(tc, &["duration_hours"]).expect("duration");
        let dur_days = dur_h / 24.0;
        let max_err = json_f64(tc, &["max_balance_error_pct"]).expect("max_err");
        let dt_days = 0.001;

        let result = solve_richards_1d(
            &p, depth, n_nodes, h_init, h_top, zero_flux, true, dur_days, dt_days,
        );

        match result {
            Ok(profiles) => {
                let dz = depth / n_nodes as f64;
                let err_pct =
                    mass_balance_check(&p, &profiles, h_init, h_top, zero_flux, dt_days, dz);
                v.check_bool(
                    &format!("{media} mass balance {err_pct:.1}% <= {max_err}%"),
                    err_pct <= max_err,
                );
            }
            Err(e) => {
                v.check_bool(&format!("{media} mass balance (err: {e})"), false);
            }
        }
    }
}

fn main() {
    validation::init_tracing();
    validation::banner("CW2D Richards Extension (Dong et al. 2019)");
    let mut v = ValidationHarness::new("CW2D Richards Validation");
    let benchmark = parse_benchmark_json(BENCHMARK_JSON).expect("benchmark_cw2d.json must parse");

    validate_retention(&mut v, &benchmark);
    validate_conductivity(&mut v, &benchmark);
    validate_gravel_infiltration(&mut v, &benchmark);
    validate_organic_drainage(&mut v, &benchmark);
    validate_mass_balance(&mut v, &benchmark);

    v.finish();
}
