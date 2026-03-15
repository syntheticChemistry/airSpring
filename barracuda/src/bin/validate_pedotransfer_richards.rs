// SPDX-License-Identifier: AGPL-3.0-or-later
//! Exp 038: Pedotransfer → Richards Coupled Simulation.
//!
//! Validates the full pipeline: Saxton-Rawls soil texture → Van Genuchten
//! parameters → Richards 1D PDE solver. The Python control validates the
//! pedotransfer mapping; this binary adds the implicit Richards solver
//! for wetting/drainage dynamics.
//!
//! Benchmark: `control/pedotransfer_richards/benchmark_pedotransfer_richards.json`
//! Baseline: `control/pedotransfer_richards/pedotransfer_richards.py` (29/29 PASS)
//!
//! script=`control/pedotransfer_richards/pedotransfer_richards.py`, commit=97e7533, date=2026-02-28
//! Run: `python3 control/pedotransfer_richards/pedotransfer_richards.py`

use airspring_barracuda::eco::richards::{
    VanGenuchtenParams, cumulative_drainage, solve_richards_1d,
};
use airspring_barracuda::eco::soil_moisture::{SaxtonRawlsInput, saxton_rawls};
use airspring_barracuda::eco::van_genuchten::{van_genuchten_k, van_genuchten_theta};
use airspring_barracuda::tolerances;
use airspring_barracuda::validation::{self, ValidationHarness, json_field, parse_benchmark_json};

const BENCHMARK_JSON: &str =
    include_str!("../../../control/pedotransfer_richards/benchmark_pedotransfer_richards.json");

fn sr_to_vg(sand: f64, clay: f64, om_pct: f64) -> VanGenuchtenParams {
    let sr = saxton_rawls(&SaxtonRawlsInput { sand, clay, om_pct });

    let n_vg = (sr.lambda + 1.0).max(1.05);
    let m = 1.0 - 1.0 / n_vg;

    let se_33 = if sr.theta_s > sr.theta_wp {
        (sr.theta_fc - sr.theta_wp) / (sr.theta_s - sr.theta_wp)
    } else {
        0.5
    };

    let alpha = if se_33 > 0.0 && se_33 < 1.0 && m > 0.0 {
        let pb = 33.0 * (sr.theta_s - sr.theta_wp) / (sr.theta_fc - sr.theta_wp);
        1.0 / pb.max(5.0)
    } else {
        0.02
    };

    let ks_cm_day = sr.ksat_mm_hr * 2.4;

    VanGenuchtenParams {
        theta_r: sr.theta_wp,
        theta_s: sr.theta_s,
        alpha,
        n_vg,
        ks: ks_cm_day,
    }
}

fn validate_pedotransfer(v: &mut ValidationHarness, benchmark: &serde_json::Value) {
    validation::section("Pedotransfer → VG Parameters");
    let tests = &benchmark["validation_checks"]["pedotransfer_to_vg"]["test_cases"];
    for tc in tests.as_array().expect("array") {
        let label = tc["label"].as_str().unwrap_or("test");
        let sand = json_field(tc, "sand");
        let clay = json_field(tc, "clay");
        let om = json_field(tc, "om_pct");
        let vg = sr_to_vg(sand, clay, om);
        let ref_data = &tc["vg_reference"];

        let range = |key: &str| -> (f64, f64) {
            let arr = ref_data[key].as_array().expect("range");
            (
                arr[0]
                    .as_f64()
                    .expect("benchmark range element must be f64"),
                arr[1]
                    .as_f64()
                    .expect("benchmark range element must be f64"),
            )
        };

        let (lo, hi) = range("theta_r_range");
        v.check_bool(
            &format!("{label}: θ_r={:.4} in [{lo},{hi}]", vg.theta_r),
            vg.theta_r >= lo && vg.theta_r <= hi,
        );
        let (lo, hi) = range("theta_s_range");
        v.check_bool(
            &format!("{label}: θ_s={:.4} in [{lo},{hi}]", vg.theta_s),
            vg.theta_s >= lo && vg.theta_s <= hi,
        );
        let (lo, hi) = range("alpha_range");
        v.check_bool(
            &format!("{label}: α={:.4} in [{lo},{hi}]", vg.alpha),
            vg.alpha >= lo && vg.alpha <= hi,
        );
        let (lo, hi) = range("n_range");
        v.check_bool(
            &format!("{label}: n={:.4} in [{lo},{hi}]", vg.n_vg),
            vg.n_vg >= lo && vg.n_vg <= hi,
        );
        let (lo, hi) = range("ks_range");
        v.check_bool(
            &format!("{label}: Ks={:.2} in [{lo},{hi}]", vg.ks),
            vg.ks >= lo && vg.ks <= hi,
        );
    }
}

fn validate_vg_retention(v: &mut ValidationHarness, benchmark: &serde_json::Value) {
    validation::section("VG Retention Curves");
    let tests = &benchmark["validation_checks"]["pedotransfer_to_vg"]["test_cases"];
    for tc in tests.as_array().expect("array") {
        let label = tc["label"].as_str().unwrap_or("test");
        let vg = sr_to_vg(
            json_field(tc, "sand"),
            json_field(tc, "clay"),
            json_field(tc, "om_pct"),
        );

        let theta_sat = van_genuchten_theta(0.0, vg.theta_r, vg.theta_s, vg.alpha, vg.n_vg);
        v.check_abs(
            &format!("{label}: θ(0)=θ_s"),
            theta_sat,
            vg.theta_s,
            tolerances::PEDOTRANSFER_MOISTURE.abs_tol,
        );

        let k_sat = van_genuchten_k(0.0, vg.ks, vg.theta_r, vg.theta_s, vg.alpha, vg.n_vg);
        v.check_abs(
            &format!("{label}: K(0)=Ks"),
            k_sat,
            vg.ks,
            tolerances::PEDOTRANSFER_MOISTURE.abs_tol,
        );

        let theta_wet = van_genuchten_theta(-50.0, vg.theta_r, vg.theta_s, vg.alpha, vg.n_vg);
        let theta_dry = van_genuchten_theta(-500.0, vg.theta_r, vg.theta_s, vg.alpha, vg.n_vg);
        v.check_lower(&format!("{label}: θ(-50) > θ(-500)"), theta_wet, theta_dry);
    }
}

fn validate_richards_wetting(v: &mut ValidationHarness, benchmark: &serde_json::Value) {
    validation::section("Richards Wetting (implicit solver)");
    let tests = &benchmark["validation_checks"]["richards_wetting"]["test_cases"];
    for tc in tests.as_array().expect("array") {
        let label = tc["label"].as_str().unwrap_or("test");
        let vg = sr_to_vg(
            json_field(tc, "sand"),
            json_field(tc, "clay"),
            json_field(tc, "om_pct"),
        );

        let h_init = json_field(tc, "h_initial");
        let h_top = json_field(tc, "h_top");
        let depth = json_field(tc, "depth_cm");
        let n_nodes = json_field(tc, "n_nodes") as usize;
        let duration = json_field(tc, "duration_days");
        let dt = json_field(tc, "dt_days");

        match solve_richards_1d(
            &vg, depth, n_nodes, h_init, h_top, false, false, duration, dt,
        ) {
            Ok(profiles) => {
                let initial_theta_top =
                    van_genuchten_theta(h_init, vg.theta_r, vg.theta_s, vg.alpha, vg.n_vg);
                let final_profile = &profiles[profiles.len() - 1];
                let final_theta_top = final_profile.theta[0];

                v.check_lower(
                    &format!("{label}: top wets (θ_final > θ_initial)"),
                    final_theta_top,
                    initial_theta_top,
                );

                let n_profiles = profiles.len();
                v.check_lower(
                    &format!("{label}: solver produced {n_profiles} profiles"),
                    n_profiles as f64,
                    1.0,
                );
            }
            Err(e) => {
                v.check_bool(&format!("{label}: solver converged ({e})"), false);
            }
        }
    }
}

fn validate_richards_drainage(v: &mut ValidationHarness, benchmark: &serde_json::Value) {
    validation::section("Richards Free Drainage (implicit solver)");
    let tests = &benchmark["validation_checks"]["richards_drainage"]["test_cases"];
    for tc in tests.as_array().expect("array") {
        let label = tc["label"].as_str().unwrap_or("test");
        let vg = sr_to_vg(
            json_field(tc, "sand"),
            json_field(tc, "clay"),
            json_field(tc, "om_pct"),
        );

        let h_init = json_field(tc, "h_initial");
        let depth = json_field(tc, "depth_cm");
        let n_nodes = json_field(tc, "n_nodes") as usize;
        let duration = json_field(tc, "duration_days");
        let dt = json_field(tc, "dt_days");

        match solve_richards_1d(
            &vg, depth, n_nodes, h_init, h_init, true, true, duration, dt,
        ) {
            Ok(profiles) => {
                let drain = cumulative_drainage(&vg, &profiles, dt);
                let total_drain = drain.last().copied().unwrap_or(0.0);
                v.check_lower(
                    &format!("{label}: cumulative drainage > 0 (drain={total_drain:.4})"),
                    total_drain,
                    0.0,
                );

                let initial_avg: f64 =
                    profiles[0].theta.iter().sum::<f64>() / profiles[0].theta.len() as f64;
                let final_profile = profiles.last().expect("non-empty profile result");
                let final_avg: f64 =
                    final_profile.theta.iter().sum::<f64>() / final_profile.theta.len() as f64;
                v.check_lower(
                    &format!("{label}: avg θ decreases ({initial_avg:.4} → {final_avg:.4})"),
                    initial_avg,
                    final_avg,
                );
            }
            Err(e) => {
                v.check_bool(&format!("{label}: solver converged ({e})"), false);
            }
        }
    }
}

fn validate_texture_sensitivity(v: &mut ValidationHarness, benchmark: &serde_json::Value) {
    validation::section("Texture Sensitivity");
    let tests = &benchmark["validation_checks"]["texture_sensitivity"]["test_cases"];
    for tc in tests.as_array().expect("array") {
        let label = tc["label"].as_str().unwrap_or("test");
        let sandy = &tc["sandy"];
        let clayey = &tc["clayey"];

        let vg_sand = sr_to_vg(
            json_field(sandy, "sand"),
            json_field(sandy, "clay"),
            json_field(sandy, "om_pct"),
        );
        let vg_clay = sr_to_vg(
            json_field(clayey, "sand"),
            json_field(clayey, "clay"),
            json_field(clayey, "om_pct"),
        );

        v.check_lower(
            &format!("{label}: Ks_sand > Ks_clay"),
            vg_sand.ks,
            vg_clay.ks,
        );

        let theta_s = van_genuchten_theta(
            -100.0,
            vg_sand.theta_r,
            vg_sand.theta_s,
            vg_sand.alpha,
            vg_sand.n_vg,
        );
        let theta_c = van_genuchten_theta(
            -100.0,
            vg_clay.theta_r,
            vg_clay.theta_s,
            vg_clay.alpha,
            vg_clay.n_vg,
        );
        v.check_lower(
            &format!("{label}: θ_clay(-100) > θ_sand(-100)"),
            theta_c,
            theta_s,
        );
    }
}

fn main() {
    validation::init_tracing();
    validation::banner("Exp 038: Pedotransfer → Richards Coupled Simulation");

    let mut v = ValidationHarness::new("Pedotransfer-Richards");
    let benchmark = parse_benchmark_json(BENCHMARK_JSON)
        .expect("benchmark_pedotransfer_richards.json must parse");

    validate_pedotransfer(&mut v, &benchmark);
    validate_vg_retention(&mut v, &benchmark);
    validate_richards_wetting(&mut v, &benchmark);
    validate_richards_drainage(&mut v, &benchmark);
    validate_texture_sensitivity(&mut v, &benchmark);

    v.finish();
}
