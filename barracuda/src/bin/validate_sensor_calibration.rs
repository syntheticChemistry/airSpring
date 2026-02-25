// SPDX-License-Identifier: AGPL-3.0-or-later
//! Validate `SoilWatch` 10 sensor calibration and irrigation model
//! against published values from Dong et al. (2024).
//!
//! Benchmark source: `control/iot_irrigation/benchmark_dong2024.json`
//! Provenance: Dong, Werling, Cao, Li (2024) *Frontiers in Water* 6, 1353597.
//! Digitized: 2026-02-16, commit: initial airSpring.

use airspring_barracuda::eco::sensor_calibration as sc;
use airspring_barracuda::validation::{self, ValidationHarness};

/// Exact-match tolerance for polynomial/analytical computations.
const EXACT_TOL: f64 = 1e-10;

/// Irrigation recommendation tolerance (cm). Justified by depth precision
/// of typical VWC sensors (±0.01 m³/m³ at 30 cm depth → ±0.3 cm IR).
const IR_TOL: f64 = 0.01;

/// Index of Agreement criterion from Dong et al. (2020) Table 3.
const IA_CRITERION: f64 = 0.80;

/// Statistical significance threshold (standard two-tailed, α=0.05).
const P_SIGNIFICANT: f64 = 0.05;

/// Water savings tolerance (percentage points).
const SAVINGS_TOL: f64 = 0.1;

/// Helper: extract `f64` from a JSON value by key (single level).
fn jf(val: &serde_json::Value, key: &str) -> f64 {
    validation::json_f64(val, &[key]).expect("benchmark JSON missing required key")
}

/// Validate `SoilWatch` 10 polynomial against benchmark coefficients and range.
fn validate_soilwatch10(v: &mut ValidationHarness, bm: &serde_json::Value) {
    validation::section("SoilWatch 10 calibration (Eq. 5)");

    let coeffs = &bm["soilwatch10_calibration"]["equation_coefficients"];
    let a3 = jf(coeffs, "a3");
    let a2 = jf(coeffs, "a2");
    let a1 = jf(coeffs, "a1");
    let a0 = jf(coeffs, "a0");

    for &rc in &[10_000.0_f64, 25_000.0] {
        let expected = a3.mul_add(rc, a2).mul_add(rc, a1).mul_add(rc, a0);
        let computed = sc::soilwatch10_vwc(rc);
        v.check_abs(
            &format!("VWC(RC={rc:.0}) vs JSON coefficients"),
            computed,
            expected,
            EXACT_TOL,
        );
    }

    validation::section("Calibration range behaviour");

    let vwc_min = jf(
        &bm["soilwatch10_calibration"]["vwc_calibration_range"],
        "min_cm3_cm3",
    );
    let vwc_max = jf(
        &bm["soilwatch10_calibration"]["vwc_calibration_range"],
        "max_cm3_cm3",
    );

    let raw_counts: Vec<f64> = (1000..=50000).step_by(100).map(f64::from).collect();
    let vwc_values = sc::soilwatch10_vwc_vec(&raw_counts);
    let valid_count = vwc_values
        .iter()
        .filter(|&&val| val >= vwc_min && val <= vwc_max)
        .count();
    v.check_bool(
        &format!("Produces VWC in [{vwc_min}, {vwc_max}]: {valid_count} valid points"),
        valid_count > 0,
    );

    let valid_vwc: Vec<f64> = vwc_values
        .iter()
        .copied()
        .filter(|&val| val >= vwc_min && val <= vwc_max)
        .collect();
    if valid_vwc.len() > 1 {
        let monotonic = valid_vwc.windows(2).all(|w| w[1] >= w[0]);
        v.check_bool("Monotonically increasing in valid range", monotonic);
    }

    v.check_bool(
        "VWC(RC=0) < 0 (below calibration range)",
        sc::soilwatch10_vwc(0.0) < 0.0,
    );
}

/// Validate irrigation recommendation equations (Eq. 1, multi-layer).
fn validate_irrigation(v: &mut ValidationHarness, bm: &serde_json::Value) {
    println!();
    validation::section("Irrigation recommendation (Eq. 1)");

    let ir_example = &bm["irrigation_recommendation"]["example_sandy_soil"];
    let fc = jf(ir_example, "field_capacity_cm_cm");
    let vwc = jf(ir_example, "current_vwc_cm_cm");
    let depth = jf(ir_example, "depth_cm");
    let expected_ir = jf(ir_example, "expected_ir_cm");

    let computed_ir = sc::irrigation_recommendation(fc, vwc, depth);
    v.check_abs("IR (sandy soil example)", computed_ir, expected_ir, IR_TOL);

    v.check_abs(
        "IR at field capacity",
        sc::irrigation_recommendation(0.12, 0.12, 30.0),
        0.0,
        EXACT_TOL,
    );
    v.check_abs(
        "IR above field capacity",
        sc::irrigation_recommendation(0.12, 0.15, 30.0),
        0.0,
        EXACT_TOL,
    );

    let layers = [
        sc::SoilLayer {
            field_capacity: 0.12,
            current_vwc: 0.08,
            depth_cm: 30.0,
        },
        sc::SoilLayer {
            field_capacity: 0.15,
            current_vwc: 0.10,
            depth_cm: 30.0,
        },
        sc::SoilLayer {
            field_capacity: 0.18,
            current_vwc: 0.12,
            depth_cm: 30.0,
        },
    ];
    let total_ir = sc::multi_layer_irrigation(&layers);
    v.check_abs("Multi-layer IR (3 depths)", total_ir, 4.5, IR_TOL);
}

/// Validate sensor performance criteria (Table 2) and field demonstrations.
fn validate_performance_and_demos(v: &mut ValidationHarness, bm: &serde_json::Value) {
    println!();
    validation::section("Sensor performance criteria (Table 2)");

    let mbe_thresh = jf(&bm["criteria"], "mbe_threshold_cm3_cm3");
    let rmse_thresh = jf(&bm["criteria"], "rmse_threshold_cm3_cm3");

    let perf = &bm["sensor_performance_table2"];
    for soil in ["sand", "loamy_sand"] {
        if let Some(stats) = perf.get(soil) {
            let rmse = jf(stats, "rmse_cm3_cm3");
            let ia = jf(stats, "ia");
            let mbe = jf(stats, "mbe_cm3_cm3");

            v.check_bool(
                &format!("{soil}: RMSE={rmse:.3} < {rmse_thresh} (criteria)"),
                rmse < rmse_thresh,
            );
            v.check_bool(
                &format!("{soil}: |MBE|={:.3} ≤ {mbe_thresh} (criteria)", mbe.abs()),
                mbe.abs() <= mbe_thresh,
            );
            v.check_bool(
                &format!("{soil}: IA={ia:.2} > {IA_CRITERION} (criteria)"),
                ia > IA_CRITERION,
            );
        }
    }

    println!();
    validation::section("Field demonstrations (published results)");

    let bb = &bm["blueberry_demonstration"];
    let bb_rec_yield = jf(&bb["treatment_recommended"], "yield_per_plant_g");
    let bb_far_yield = jf(&bb["treatment_farmer"], "yield_per_plant_g");
    v.check_bool(
        &format!("Blueberry: recommended yield ({bb_rec_yield}g) > farmer ({bb_far_yield}g)"),
        bb_rec_yield > bb_far_yield,
    );

    let yield_p = jf(&bb["anova_results"], "yield_p_value");
    v.check_bool(
        &format!("Blueberry yield p={yield_p} < {P_SIGNIFICANT} (significant)"),
        yield_p < P_SIGNIFICANT,
    );

    let bw_p = jf(&bb["anova_results"], "berry_weight_p_value");
    v.check_bool(
        &format!("Blueberry berry weight p={bw_p} < {P_SIGNIFICANT} (significant)"),
        bw_p < P_SIGNIFICANT,
    );

    let tom = &bm["tomato_demonstration"];
    let count_p = jf(&tom["anova_results"], "marketable_count_p_value");
    v.check_bool(
        &format!("Tomato count p={count_p} > {P_SIGNIFICANT} (not significant — same yield)"),
        count_p > P_SIGNIFICANT,
    );

    let weight_p = jf(&tom["anova_results"], "weight_p_value");
    v.check_bool(
        &format!("Tomato weight p={weight_p} > {P_SIGNIFICANT} (not significant — same quality)"),
        weight_p > P_SIGNIFICANT,
    );

    let water_savings = jf(tom, "water_savings_pct");
    v.check_abs("Tomato water savings (%)", water_savings, 30.0, SAVINGS_TOL);
}

fn main() {
    validation::banner("Sensor Calibration Validation (Dong et al. 2024)");
    let mut v = ValidationHarness::new("Sensor Calibration Validation (Dong et al. 2024)");

    let json_str = include_str!("../../../control/iot_irrigation/benchmark_dong2024.json");
    let bm = validation::parse_benchmark_json(json_str).expect("benchmark JSON");

    validate_soilwatch10(&mut v, &bm);
    validate_irrigation(&mut v, &bm);
    validate_performance_and_demos(&mut v, &bm);

    v.finish();
}
