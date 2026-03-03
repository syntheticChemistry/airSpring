// SPDX-License-Identifier: AGPL-3.0-or-later
#![warn(clippy::pedantic)]
#![allow(clippy::cast_precision_loss)]
//! Anderson soil-moisture coupling — Rust CPU validation for Experiment 045.
//!
//! Cross-validates the θ → `d_eff` coupling chain against the Python control
//! (`control/anderson_coupling/anderson_coupling.py`).
//!
//! Benchmark: `control/anderson_coupling/benchmark_anderson_coupling.json`
//!
//! Provenance: script=`control/anderson_coupling/anderson_coupling.py`, commit=0500398, date=2026-02-27
//!
//! Tests: point coupling, monotonicity, boundaries, disorder, seasonal regime,
//! tillage effects, and numeric reference parity at 1e-10.

use airspring_barracuda::eco::anderson::{self, CouplingResult, D_EFF_CRITICAL};
use airspring_barracuda::tolerances;
use airspring_barracuda::validation::{self, ValidationHarness};

const BENCHMARK_JSON: &str =
    include_str!("../../../control/anderson_coupling/benchmark_anderson_coupling.json");

struct SoilParams {
    key: String,
    theta_r: f64,
    theta_s: f64,
}

fn load_soils(benchmark: &serde_json::Value) -> Vec<SoilParams> {
    let soils_obj = benchmark["soil_types"]
        .as_object()
        .expect("benchmark must contain soil_types");
    soils_obj
        .iter()
        .map(|(key, v)| SoilParams {
            key: key.clone(),
            theta_r: v["theta_r"].as_f64().expect("theta_r"),
            theta_s: v["theta_s"].as_f64().expect("theta_s"),
        })
        .collect()
}

fn check_point_coupling(
    v: &mut ValidationHarness,
    benchmark: &serde_json::Value,
    soils: &[SoilParams],
) {
    validation::section("Point coupling (from benchmark JSON)");
    let checks = benchmark["checks"]
        .as_array()
        .expect("benchmark must contain checks");

    for check in checks {
        let label = check["label"].as_str().unwrap_or("?");
        if !label.starts_with("point(") {
            continue;
        }
        let Some(soil_key) = check["soil"].as_str() else {
            continue;
        };
        let Some(soil) = soils.iter().find(|s| s.key == soil_key) else {
            continue;
        };
        let theta = check["theta"].as_f64().expect("theta in check");
        let expected_regime = check["expected_regime"].as_str().expect("expected_regime");
        let expected_d_eff = check["d_eff"].as_f64().expect("d_eff in check");

        let r = anderson::coupling_chain(theta, soil.theta_r, soil.theta_s);
        let actual = r.regime.as_str();
        v.check_bool(
            &format!(
                "point({soil_key}, θ={theta:.2}) → {actual} (d_eff={:.3})",
                r.d_eff
            ),
            actual == expected_regime,
        );
        v.check_abs(
            &format!("point({soil_key}, θ={theta:.2}): d_eff vs Python"),
            r.d_eff,
            expected_d_eff,
            tolerances::SENSOR_EXACT.abs_tol,
        );
    }
}

fn check_monotonicity(v: &mut ValidationHarness, soils: &[SoilParams]) {
    println!();
    validation::section("Monotonicity");
    for soil in soils {
        let mut prev_d = -1.0;
        let mut ok = true;
        for i in 0..=10 {
            let theta = f64::from(i).mul_add((soil.theta_s - soil.theta_r) / 10.0, soil.theta_r);
            let r = anderson::coupling_chain(theta, soil.theta_r, soil.theta_s);
            if r.d_eff < prev_d - 1e-12 {
                ok = false;
            }
            prev_d = r.d_eff;
        }
        v.check_bool(&format!("monotonicity({}): d_eff ↑ with θ", soil.key), ok);
    }
}

fn check_boundaries(v: &mut ValidationHarness, soils: &[SoilParams]) {
    println!();
    validation::section("Boundary conditions");
    for soil in soils {
        let sat = anderson::coupling_chain(soil.theta_s, soil.theta_r, soil.theta_s);
        v.check_abs(
            &format!("boundary({}, θ=θ_s) → d_eff=3.0", soil.key),
            sat.d_eff,
            3.0,
            tolerances::SENSOR_EXACT.abs_tol,
        );
        let dry = anderson::coupling_chain(soil.theta_r, soil.theta_r, soil.theta_s);
        v.check_abs(
            &format!("boundary({}, θ=θ_r) → d_eff=0.0", soil.key),
            dry.d_eff,
            0.0,
            tolerances::SENSOR_EXACT.abs_tol,
        );
    }
}

fn check_disorder_monotonicity(v: &mut ValidationHarness, soils: &[SoilParams]) {
    println!();
    validation::section("Disorder monotonicity");
    for soil in soils {
        let mut prev_w = f64::INFINITY;
        let mut ok = true;
        for i in 0..=10 {
            let theta = f64::from(i).mul_add((soil.theta_s - soil.theta_r) / 10.0, soil.theta_r);
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
    let mut theta_fc = 0.65f64.mul_add(soil.theta_s - soil.theta_r, soil.theta_r);
    let mut theta_wp = 0.15f64.mul_add(soil.theta_s - soil.theta_r, soil.theta_r);
    if conventional {
        theta_fc *= 0.85;
        theta_wp *= 0.90;
    }
    let days: i32 = 153;
    (0..days)
        .map(|d| {
            let frac = f64::from(d) / f64::from(days - 1);
            if frac < 0.15 {
                theta_fc
            } else if frac < 0.55 {
                let t = (frac - 0.15) / 0.40;
                t.mul_add(-(theta_fc - theta_wp), theta_fc)
            } else if frac < 0.75 {
                theta_wp
            } else {
                let t = (frac - 0.75) / 0.25;
                t.mul_add((theta_fc - theta_wp) * 0.8, theta_wp)
            }
        })
        .collect()
}

fn check_seasonal(v: &mut ValidationHarness, soils: &[SoilParams]) {
    println!();
    validation::section("Seasonal regime transitions");
    for soil_key in &["loam", "silt_loam"] {
        let Some(soil) = soils.iter().find(|s| s.key == *soil_key) else {
            continue;
        };
        let mut notill_mean = 0.0;
        let mut conv_mean = 0.0;
        for conventional in [false, true] {
            let theta_series = seasonal_theta_profile(soil, conventional);
            let chain: Vec<CouplingResult> =
                anderson::coupling_series(&theta_series, soil.theta_r, soil.theta_s);
            let d_effs: Vec<f64> = chain.iter().map(|c| c.d_eff).collect();

            let spring_d: f64 = d_effs[..23].iter().sum::<f64>() / 23.0;
            let midsummer_d: f64 = d_effs[84..115].iter().sum::<f64>() / 31.0;
            let tillage = if conventional {
                "conventional"
            } else {
                "notill"
            };

            v.check_bool(
                &format!("seasonal({soil_key}, {tillage}): spring d_eff={spring_d:.2} > {D_EFF_CRITICAL}"),
                spring_d > D_EFF_CRITICAL,
            );
            v.check_bool(
                &format!("seasonal({soil_key}, {tillage}): spring > summer ({spring_d:.2} > {midsummer_d:.2})"),
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

fn check_reference_values(
    v: &mut ValidationHarness,
    benchmark: &serde_json::Value,
    soils: &[SoilParams],
) {
    println!();
    validation::section("Numeric reference parity (from benchmark JSON)");
    let refs = benchmark["reference_values"]
        .as_array()
        .expect("benchmark must contain reference_values");

    for rv in refs {
        let soil_key = rv["soil"].as_str().expect("soil in reference_values");
        let Some(soil) = soils.iter().find(|s| s.key == soil_key) else {
            continue;
        };
        let theta = rv["theta"].as_f64().expect("theta");
        let expected_d_eff = rv["d_eff"].as_f64().expect("d_eff");
        let expected_se = rv["se"].as_f64().expect("se");
        let expected_disorder = rv["disorder"].as_f64().expect("disorder");

        let r = anderson::coupling_chain(theta, soil.theta_r, soil.theta_s);

        v.check_abs(
            &format!("ref({soil_key}, θ={theta:.4}): se"),
            r.se,
            expected_se,
            tolerances::SENSOR_EXACT.abs_tol,
        );
        v.check_abs(
            &format!("ref({soil_key}, θ={theta:.4}): d_eff"),
            r.d_eff,
            expected_d_eff,
            tolerances::SENSOR_EXACT.abs_tol,
        );
        v.check_abs(
            &format!("ref({soil_key}, θ={theta:.4}): disorder"),
            r.disorder,
            expected_disorder,
            tolerances::SENSOR_EXACT.abs_tol,
        );
    }
}

fn main() {
    validation::init_tracing();
    validation::banner("Anderson Soil-Moisture Coupling (Exp 045)");

    let benchmark =
        validation::parse_benchmark_json(BENCHMARK_JSON).expect("benchmark JSON must parse");
    let soils = load_soils(&benchmark);
    let mut v = ValidationHarness::new("Anderson Coupling Validation");

    check_point_coupling(&mut v, &benchmark, &soils);
    check_monotonicity(&mut v, &soils);
    check_boundaries(&mut v, &soils);
    check_disorder_monotonicity(&mut v, &soils);
    check_seasonal(&mut v, &soils);
    check_reference_values(&mut v, &benchmark, &soils);

    v.finish();
}
