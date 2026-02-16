//! Validate `SoilWatch` 10 sensor calibration and irrigation model
//! against published values from Dong et al. (2024).
//!
//! Benchmark source: `control/iot_irrigation/benchmark_dong2024.json`
//! Provenance: Dong, Werling, Cao, Li (2024) *Frontiers in Water* 6, 1353597.
//! Digitized: 2026-02-16, commit: initial airSpring.

use airspring_barracuda::eco::sensor_calibration as sc;
use airspring_barracuda::validation::{self, ValidationRunner};

/// Helper: extract `f64` from a JSON value by key (single level).
fn jf(val: &serde_json::Value, key: &str) -> f64 {
    validation::json_f64(val, &[key]).unwrap_or_else(|| panic!("Missing JSON key: {key}"))
}

#[allow(clippy::too_many_lines)]
fn main() {
    let mut v = ValidationRunner::new("Sensor Calibration Validation (Dong et al. 2024)");

    let json_str = include_str!("../../../control/iot_irrigation/benchmark_dong2024.json");
    let bm = validation::parse_benchmark_json(json_str).expect("benchmark JSON");

    // ── SoilWatch 10 equation coefficients ────────────────────────
    v.section("SoilWatch 10 calibration (Eq. 5)");

    let coeffs = &bm["soilwatch10_calibration"]["equation_coefficients"];
    let a3 = jf(coeffs, "a3");
    let a2 = jf(coeffs, "a2");
    let a1 = jf(coeffs, "a1");
    let a0 = jf(coeffs, "a0");

    // Verify our Rust implementation matches the published coefficients
    // by computing VWC at reference raw counts and comparing against
    // the polynomial evaluated with the JSON coefficients directly.
    let test_rc: f64 = 10_000.0;
    // Horner's form: ((a3×RC + a2)×RC + a1)×RC + a0
    let expected = a3
        .mul_add(test_rc, a2)
        .mul_add(test_rc, a1)
        .mul_add(test_rc, a0);
    let computed = sc::soilwatch10_vwc(test_rc);
    v.check(
        "VWC(RC=10000) vs JSON coefficients",
        computed,
        expected,
        1e-10,
    );

    let test_rc2: f64 = 25_000.0;
    let expected2 = a3
        .mul_add(test_rc2, a2)
        .mul_add(test_rc2, a1)
        .mul_add(test_rc2, a0);
    let computed2 = sc::soilwatch10_vwc(test_rc2);
    v.check(
        "VWC(RC=25000) vs JSON coefficients",
        computed2,
        expected2,
        1e-10,
    );

    // ── Calibration range validity ────────────────────────────────
    v.section("Calibration range behaviour");

    let vwc_min = jf(
        &bm["soilwatch10_calibration"]["vwc_calibration_range"],
        "min_cm3_cm3",
    );
    let vwc_max = jf(
        &bm["soilwatch10_calibration"]["vwc_calibration_range"],
        "max_cm3_cm3",
    );

    // Equation should produce VWC in calibrated range for some span of raw counts
    let raw_counts: Vec<f64> = (1000..=50000).step_by(100).map(f64::from).collect();
    let vwc_values = sc::soilwatch10_vwc_vec(&raw_counts);
    let valid_count = vwc_values
        .iter()
        .filter(|&&val| val >= vwc_min && val <= vwc_max)
        .count();
    v.check_bool(
        &format!("Produces VWC in [{vwc_min}, {vwc_max}]: {valid_count} valid points"),
        valid_count > 0,
        true,
    );

    // Monotonicity in valid range
    let valid_vwc: Vec<f64> = vwc_values
        .iter()
        .copied()
        .filter(|&val| val >= vwc_min && val <= vwc_max)
        .collect();
    if valid_vwc.len() > 1 {
        let monotonic = valid_vwc.windows(2).all(|w| w[1] >= w[0]);
        v.check_bool("Monotonically increasing in valid range", monotonic, true);
    }

    // Below calibration range at RC=0
    v.check_bool(
        "VWC(RC=0) < 0 (below calibration range)",
        sc::soilwatch10_vwc(0.0) < 0.0,
        true,
    );

    // ── Irrigation recommendation (Eq. 1) ─────────────────────────
    println!();
    v.section("Irrigation recommendation (Eq. 1)");

    let ir_example = &bm["irrigation_recommendation"]["example_sandy_soil"];
    let fc = jf(ir_example, "field_capacity_cm_cm");
    let vwc = jf(ir_example, "current_vwc_cm_cm");
    let depth = jf(ir_example, "depth_cm");
    let expected_ir = jf(ir_example, "expected_ir_cm");

    let computed_ir = sc::irrigation_recommendation(fc, vwc, depth);
    v.check("IR (sandy soil example)", computed_ir, expected_ir, 0.01);

    // At field capacity: IR = 0
    v.check(
        "IR at field capacity",
        sc::irrigation_recommendation(0.12, 0.12, 30.0),
        0.0,
        1e-10,
    );

    // Above field capacity: IR = 0 (clamped)
    v.check(
        "IR above field capacity",
        sc::irrigation_recommendation(0.12, 0.15, 30.0),
        0.0,
        1e-10,
    );

    // Multi-layer test (corn: 3 depths at 30 cm each)
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
    // (0.04 × 30) + (0.05 × 30) + (0.06 × 30) = 4.5 cm
    v.check("Multi-layer IR (3 depths)", total_ir, 4.5, 0.01);

    // ── Sensor performance criteria (Table 2) ──────────────────────
    println!();
    v.section("Sensor performance criteria (Table 2)");

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
                true,
            );
            v.check_bool(
                &format!("{soil}: |MBE|={:.3} ≤ {mbe_thresh} (criteria)", mbe.abs()),
                mbe.abs() <= mbe_thresh,
                true,
            );
            v.check_bool(
                &format!("{soil}: IA={ia:.2} > 0.80 (criteria)"),
                ia > 0.80,
                true,
            );
        }
    }

    // ── Blueberry demonstration ────────────────────────────────────
    println!();
    v.section("Field demonstrations (published results)");

    let bb = &bm["blueberry_demonstration"];
    let bb_rec_yield = jf(&bb["treatment_recommended"], "yield_per_plant_g");
    let bb_far_yield = jf(&bb["treatment_farmer"], "yield_per_plant_g");
    v.check_bool(
        &format!("Blueberry: recommended yield ({bb_rec_yield}g) > farmer ({bb_far_yield}g)"),
        bb_rec_yield > bb_far_yield,
        true,
    );

    let yield_p = jf(&bb["anova_results"], "yield_p_value");
    v.check_bool(
        &format!("Blueberry yield p={yield_p} < 0.05 (significant)"),
        yield_p < 0.05,
        true,
    );

    let bw_p = jf(&bb["anova_results"], "berry_weight_p_value");
    v.check_bool(
        &format!("Blueberry berry weight p={bw_p} < 0.05 (significant)"),
        bw_p < 0.05,
        true,
    );

    // ── Tomato demonstration ───────────────────────────────────────
    let tom = &bm["tomato_demonstration"];
    let count_p = jf(&tom["anova_results"], "marketable_count_p_value");
    v.check_bool(
        &format!("Tomato count p={count_p} > 0.05 (not significant — same yield)"),
        count_p > 0.05,
        true,
    );

    let weight_p = jf(&tom["anova_results"], "weight_p_value");
    v.check_bool(
        &format!("Tomato weight p={weight_p} > 0.05 (not significant — same quality)"),
        weight_p > 0.05,
        true,
    );

    let water_savings = jf(tom, "water_savings_pct");
    v.check("Tomato water savings (%)", water_savings, 30.0, 0.1);

    v.finish();
}
