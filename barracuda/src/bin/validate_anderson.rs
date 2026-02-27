// SPDX-License-Identifier: AGPL-3.0-or-later
#![warn(clippy::pedantic)]
#![allow(clippy::cast_precision_loss)]
//! Anderson soil-moisture coupling — Rust CPU validation for Experiment 045.
//!
//! Cross-validates the θ → d_eff coupling chain against the Python control
//! (`control/anderson_coupling/anderson_coupling.py`).
//!
//! Tests: point coupling, monotonicity, boundaries, disorder, seasonal regime,
//! tillage effects, and numeric reference parity at 1e-10.

use airspring_barracuda::eco::anderson::{
    self, CouplingResult, D_EFF_CRITICAL, QsRegime,
};
use airspring_barracuda::validation::{self, ValidationHarness};

struct SoilParams {
    key: &'static str,
    theta_r: f64,
    theta_s: f64,
}

const SOILS: &[SoilParams] = &[
    SoilParams {
        key: "sand",
        theta_r: 0.045,
        theta_s: 0.43,
    },
    SoilParams {
        key: "loam",
        theta_r: 0.078,
        theta_s: 0.43,
    },
    SoilParams {
        key: "silt_loam",
        theta_r: 0.067,
        theta_s: 0.45,
    },
    SoilParams {
        key: "clay",
        theta_r: 0.068,
        theta_s: 0.38,
    },
];

fn regime_str(r: QsRegime) -> &'static str {
    r.as_str()
}

fn check_point_coupling(v: &mut ValidationHarness) {
    validation::section("Point coupling");
    let cases: &[(&str, f64, &str)] = &[
        ("sand", 0.35, "extended"),
        ("sand", 0.10, "localized"),
        ("loam", 0.30, "marginal"),
        ("loam", 0.12, "localized"),
        ("silt_loam", 0.35, "extended"),
        ("silt_loam", 0.10, "localized"),
        ("clay", 0.30, "extended"),
        ("clay", 0.10, "localized"),
        ("loam", 0.20, "localized"),
    ];
    for &(soil_key, theta, expected) in cases {
        let soil = SOILS.iter().find(|s| s.key == soil_key).unwrap();
        let r = anderson::coupling_chain(theta, soil.theta_r, soil.theta_s);
        let actual = regime_str(r.regime);
        v.check_bool(
            &format!(
                "point({soil_key}, θ={theta:.2}) → {actual} (d_eff={:.3})",
                r.d_eff
            ),
            actual == expected,
        );
    }
}

fn check_monotonicity(v: &mut ValidationHarness) {
    println!();
    validation::section("Monotonicity");
    for soil in SOILS {
        let mut prev_d = -1.0;
        let mut ok = true;
        for i in 0..=10 {
            let theta =
                soil.theta_r + (i as f64) * (soil.theta_s - soil.theta_r) / 10.0;
            let r = anderson::coupling_chain(theta, soil.theta_r, soil.theta_s);
            if r.d_eff < prev_d - 1e-12 {
                ok = false;
            }
            prev_d = r.d_eff;
        }
        v.check_bool(
            &format!("monotonicity({}): d_eff ↑ with θ", soil.key),
            ok,
        );
    }
}

fn check_boundaries(v: &mut ValidationHarness) {
    println!();
    validation::section("Boundary conditions");
    for soil in SOILS {
        let sat = anderson::coupling_chain(soil.theta_s, soil.theta_r, soil.theta_s);
        v.check_abs(
            &format!("boundary({}, θ=θ_s) → d_eff=3.0", soil.key),
            sat.d_eff,
            3.0,
            1e-10,
        );

        let dry = anderson::coupling_chain(soil.theta_r, soil.theta_r, soil.theta_s);
        v.check_abs(
            &format!("boundary({}, θ=θ_r) → d_eff=0.0", soil.key),
            dry.d_eff,
            0.0,
            1e-10,
        );
    }
}

fn check_disorder_monotonicity(v: &mut ValidationHarness) {
    println!();
    validation::section("Disorder monotonicity");
    for soil in SOILS {
        let mut prev_w = f64::INFINITY;
        let mut ok = true;
        for i in 0..=10 {
            let theta =
                soil.theta_r + (i as f64) * (soil.theta_s - soil.theta_r) / 10.0;
            let r = anderson::coupling_chain(theta, soil.theta_r, soil.theta_s);
            if r.disorder > prev_w + 1e-12 {
                ok = false;
            }
            prev_w = r.disorder;
        }
        v.check_bool(
            &format!("disorder_monotonicity({}): W ↓ with θ", soil.key),
            ok,
        );
    }
}

fn seasonal_theta_profile(soil: &SoilParams, conventional: bool) -> Vec<f64> {
    let mut theta_fc = soil.theta_r + 0.65 * (soil.theta_s - soil.theta_r);
    let mut theta_wp = soil.theta_r + 0.15 * (soil.theta_s - soil.theta_r);
    if conventional {
        theta_fc *= 0.85;
        theta_wp *= 0.90;
    }
    let days = 153;
    (0..days)
        .map(|d| {
            let frac = d as f64 / (days - 1) as f64;
            if frac < 0.15 {
                theta_fc
            } else if frac < 0.55 {
                let t = (frac - 0.15) / 0.40;
                theta_fc - t * (theta_fc - theta_wp)
            } else if frac < 0.75 {
                theta_wp
            } else {
                let t = (frac - 0.75) / 0.25;
                theta_wp + t * (theta_fc - theta_wp) * 0.8
            }
        })
        .collect()
}

fn check_seasonal(v: &mut ValidationHarness) {
    println!();
    validation::section("Seasonal regime transitions");
    for soil_key in &["loam", "silt_loam"] {
        let soil = SOILS.iter().find(|s| s.key == *soil_key).unwrap();
        let mut notill_mean = 0.0;
        let mut conv_mean = 0.0;
        for conventional in [false, true] {
            let thetas = seasonal_theta_profile(soil, conventional);
            let chain: Vec<CouplingResult> =
                anderson::coupling_series(&thetas, soil.theta_r, soil.theta_s);
            let d_effs: Vec<f64> = chain.iter().map(|c| c.d_eff).collect();

            let spring_d: f64 = d_effs[..23].iter().sum::<f64>() / 23.0;
            let midsummer_d: f64 = d_effs[84..115].iter().sum::<f64>() / 31.0;
            let tillage = if conventional { "conventional" } else { "notill" };

            v.check_bool(
                &format!(
                    "seasonal({soil_key}, {tillage}): spring d_eff={spring_d:.2} > {D_EFF_CRITICAL}"
                ),
                spring_d > D_EFF_CRITICAL,
            );
            v.check_bool(
                &format!(
                    "seasonal({soil_key}, {tillage}): spring > summer ({spring_d:.2} > {midsummer_d:.2})"
                ),
                spring_d > midsummer_d,
            );

            let mean_d: f64 = d_effs.iter().sum::<f64>() / d_effs.len() as f64;
            if conventional {
                conv_mean = mean_d;
            } else {
                notill_mean = mean_d;
            }
        }
        v.check_bool(
            &format!(
                "tillage_effect({soil_key}): notill d̄={notill_mean:.3} > conv d̄={conv_mean:.3}"
            ),
            notill_mean > conv_mean,
        );
    }
}

fn check_reference_values(v: &mut ValidationHarness) {
    println!();
    validation::section("Numeric reference parity (Python cross-validation)");
    for soil in SOILS {
        for &frac in &[0.0, 0.25, 0.50, 0.75, 1.0] {
            let theta = soil.theta_r + frac * (soil.theta_s - soil.theta_r);
            let r = anderson::coupling_chain(theta, soil.theta_r, soil.theta_s);

            let expected_se = frac;
            let expected_pc = if frac <= 0.0 { 0.0 } else { frac.sqrt() };
            let expected_z = anderson::Z_MAX * expected_pc;
            let expected_d = expected_z / 2.0;
            let expected_w = anderson::W_0 * (1.0 - frac);

            v.check_abs(
                &format!("ref({}, S_e={frac:.2}): se", soil.key),
                r.se,
                expected_se,
                1e-12,
            );
            v.check_abs(
                &format!("ref({}, S_e={frac:.2}): d_eff", soil.key),
                r.d_eff,
                expected_d,
                1e-10,
            );
            v.check_abs(
                &format!("ref({}, S_e={frac:.2}): disorder", soil.key),
                r.disorder,
                expected_w,
                1e-10,
            );
        }
    }
}

fn main() {
    validation::init_tracing();
    validation::banner("Anderson Soil-Moisture Coupling (Exp 045)");

    let mut v = ValidationHarness::new("Anderson Coupling Validation");

    check_point_coupling(&mut v);
    check_monotonicity(&mut v);
    check_boundaries(&mut v);
    check_disorder_monotonicity(&mut v);
    check_seasonal(&mut v);
    check_reference_values(&mut v);

    v.finish();
}
