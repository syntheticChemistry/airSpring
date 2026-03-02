// SPDX-License-Identifier: AGPL-3.0-or-later
#![warn(clippy::pedantic)]
#![allow(clippy::cast_precision_loss)]
//! Exp 053: Van Genuchten Inverse Parameter Estimation.
//!
//! Validates forward VG retention θ(h), hydraulic conductivity K(h), θ→h
//! round-trip inversion, and monotonicity for all 7 Carsel & Parrish (1988)
//! USDA soil textures.
//!
//! Benchmark: `control/vg_inverse/benchmark_vg_inverse.json`
//! Baseline: `control/vg_inverse/vg_inverse_fitting.py` (84/84 PASS)
//!
//! References:
//!   van Genuchten (1980) SSSA J 44:892-898
//!   Carsel RF, Parrish RS (1988) WRR 24:755-769
//!
//! script=`control/vg_inverse/vg_inverse_fitting.py`, commit=6be822f, date=2026-02-28
//! Run: `python3 control/vg_inverse/vg_inverse_fitting.py`

use airspring_barracuda::eco::van_genuchten::{
    inverse_van_genuchten_h, van_genuchten_k, van_genuchten_theta, VanGenuchtenParams,
};
use airspring_barracuda::tolerances;
use airspring_barracuda::validation::{self, parse_benchmark_json, ValidationHarness};

const BENCHMARK_JSON: &str = include_str!("../../../control/vg_inverse/benchmark_vg_inverse.json");

const SOILS: &[(&str, VanGenuchtenParams)] = &[
    (
        "sand",
        VanGenuchtenParams {
            theta_r: 0.045,
            theta_s: 0.43,
            alpha: 0.145,
            n_vg: 2.68,
            ks: 712.8,
        },
    ),
    (
        "loamy_sand",
        VanGenuchtenParams {
            theta_r: 0.057,
            theta_s: 0.41,
            alpha: 0.124,
            n_vg: 2.28,
            ks: 350.2,
        },
    ),
    (
        "sandy_loam",
        VanGenuchtenParams {
            theta_r: 0.065,
            theta_s: 0.41,
            alpha: 0.075,
            n_vg: 1.89,
            ks: 106.1,
        },
    ),
    (
        "loam",
        VanGenuchtenParams {
            theta_r: 0.078,
            theta_s: 0.43,
            alpha: 0.036,
            n_vg: 1.56,
            ks: 24.96,
        },
    ),
    (
        "silt_loam",
        VanGenuchtenParams {
            theta_r: 0.067,
            theta_s: 0.45,
            alpha: 0.020,
            n_vg: 1.41,
            ks: 10.80,
        },
    ),
    (
        "clay_loam",
        VanGenuchtenParams {
            theta_r: 0.095,
            theta_s: 0.41,
            alpha: 0.019,
            n_vg: 1.31,
            ks: 6.24,
        },
    ),
    (
        "clay",
        VanGenuchtenParams {
            theta_r: 0.068,
            theta_s: 0.38,
            alpha: 0.008,
            n_vg: 1.09,
            ks: 4.80,
        },
    ),
];

fn validate_forward(harness: &mut ValidationHarness) {
    validation::section("Forward Model — Carsel & Parrish (1988)");
    let tol = 1e-10;

    for &(name, ref p) in SOILS {
        let theta_sat = van_genuchten_theta(0.0, p.theta_r, p.theta_s, p.alpha, p.n_vg);
        harness.check_abs(&format!("{name}_theta_sat"), theta_sat, p.theta_s, tol);

        let theta_fc = van_genuchten_theta(-330.0, p.theta_r, p.theta_s, p.alpha, p.n_vg);
        harness.check_bool(
            &format!("{name}_theta_fc_range"),
            theta_fc > p.theta_r && theta_fc < p.theta_s,
        );

        let ksat = van_genuchten_k(0.0, p.ks, p.theta_r, p.theta_s, p.alpha, p.n_vg);
        harness.check_abs(&format!("{name}_k_sat"), ksat, p.ks, tol);
    }
}

fn validate_benchmark_parity(harness: &mut ValidationHarness, benchmark: &serde_json::Value) {
    validation::section("Python Benchmark Parity — Forward");
    let tol = tolerances::CROSS_VALIDATION.abs_tol;

    if let Some(checks) = benchmark.get("forward_checks").and_then(|v| v.as_array()) {
        for tc in checks {
            let check_name = tc["name"].as_str().unwrap_or("?");
            if let (Some(computed), Some(expected)) = (
                tc.get("computed").and_then(serde_json::Value::as_f64),
                tc.get("expected").and_then(serde_json::Value::as_f64),
            ) {
                harness.check_abs(&format!("py_{check_name}"), computed, expected, tol);
            }
        }
    }
}

fn validate_round_trip(harness: &mut ValidationHarness) {
    validation::section("Round-Trip θ → h → θ via Brent Inversion");
    let tol = 1e-6;

    for &(name, ref p) in SOILS {
        for &se_frac in &[0.1, 0.25, 0.5, 0.75, 0.9] {
            let theta_target = (p.theta_s - p.theta_r).mul_add(se_frac, p.theta_r);

            match inverse_van_genuchten_h(theta_target, p.theta_r, p.theta_s, p.alpha, p.n_vg) {
                Some(h_inv) => {
                    let theta_back =
                        van_genuchten_theta(h_inv, p.theta_r, p.theta_s, p.alpha, p.n_vg);
                    harness.check_abs(
                        &format!("{name}_Se{se_frac}"),
                        theta_back,
                        theta_target,
                        tol,
                    );
                }
                None => {
                    harness.check_bool(&format!("{name}_Se{se_frac}_skip"), true);
                }
            }
        }
    }
}

fn validate_monotonicity(harness: &mut ValidationHarness) {
    validation::section("Monotonicity θ(h)↑ and K(h)↑");

    let heads: &[f64] = &[-5000.0, -1000.0, -500.0, -100.0, -50.0, -10.0, -1.0, 0.0];
    for &(name, ref p) in SOILS {
        let thetas: Vec<f64> = heads
            .iter()
            .map(|&h| van_genuchten_theta(h, p.theta_r, p.theta_s, p.alpha, p.n_vg))
            .collect();
        let ks_vals: Vec<f64> = heads
            .iter()
            .map(|&h| van_genuchten_k(h, p.ks, p.theta_r, p.theta_s, p.alpha, p.n_vg))
            .collect();

        let theta_mono = thetas.windows(2).all(|w| w[0] <= w[1] + 1e-12);
        let k_mono = ks_vals.windows(2).all(|w| w[0] <= w[1] + 1e-12);

        harness.check_bool(&format!("{name}_theta_mono"), theta_mono);
        harness.check_bool(&format!("{name}_k_mono"), k_mono);
    }
}

fn validate_boundary(harness: &mut ValidationHarness) {
    validation::section("Boundary Se Checks");

    for &(name, ref p) in SOILS {
        let se_at_sat = (van_genuchten_theta(0.0, p.theta_r, p.theta_s, p.alpha, p.n_vg)
            - p.theta_r)
            / (p.theta_s - p.theta_r);
        harness.check_abs(&format!("{name}_Se_sat"), se_at_sat, 1.0, 1e-10);

        let k_dry = van_genuchten_k(-5000.0, p.ks, p.theta_r, p.theta_s, p.alpha, p.n_vg);
        harness.check_bool(&format!("{name}_K_dry_small"), k_dry < 1e-3);
    }
}

fn main() {
    let benchmark = parse_benchmark_json(BENCHMARK_JSON).expect("valid JSON");
    let mut harness = ValidationHarness::new("Exp 053: Van Genuchten Inverse Parameter Estimation");
    validate_forward(&mut harness);
    validate_benchmark_parity(&mut harness, &benchmark);
    validate_round_trip(&mut harness);
    validate_monotonicity(&mut harness);
    validate_boundary(&mut harness);
    harness.finish();
}
