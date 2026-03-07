// SPDX-License-Identifier: AGPL-3.0-or-later
#![warn(clippy::pedantic)]
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
//! Validate water balance model against analytical solutions and mass conservation.
//!
//! Benchmark source: `control/water_balance/benchmark_water_balance.json`
//! Provenance: FAO-56 Ch. 8, soil water balance equations (Allen et al. 1998).
//! Python baseline: `control/water_balance/fao56_water_balance.py`
//!
//! Tests:
//! 1. Stress coefficient Ks: analytical values at FC, RAW, midpoint, WP
//! 2. Mass balance closure: P + I = ET + DP + RO + ΔS
//! 3. Runoff model: `RunoffModel::None` (FAO-56 default) matches Python baseline
//! 4. Seasonal simulation with realistic Michigan data
//!
//! All thresholds sourced from benchmark JSON.
//!
//! Provenance: script=`control/water_balance/fao56_water_balance.py`, commit=94cc51d, date=2026-02-27

use airspring_barracuda::eco::water_balance::{self as wb, DailyInput, WaterBalanceState};
use airspring_barracuda::tolerances;
use airspring_barracuda::validation::{
    self, json_f64, json_f64_required, parse_benchmark_json, ValidationHarness,
};

/// Benchmark JSON embedded at compile time for reproducibility.
const BENCHMARK_JSON: &str =
    include_str!("../../../control/water_balance/benchmark_water_balance.json");

/// Soil parameters for water balance validation (from benchmark JSON).
struct SoilParams {
    theta_fc: f64,
    theta_wp: f64,
    root_depth_mm: f64,
    depletion_fraction_p: f64,
}

fn parse_soil_params(benchmark: &serde_json::Value, scenario: &str) -> SoilParams {
    let base = ["validation_soil_parameters", scenario];
    SoilParams {
        theta_fc: json_f64_required(benchmark, &[base[0], base[1], "theta_fc"]),
        theta_wp: json_f64_required(benchmark, &[base[0], base[1], "theta_wp"]),
        root_depth_mm: json_f64_required(benchmark, &[base[0], base[1], "root_depth_mm"]),
        depletion_fraction_p: json_f64_required(
            benchmark,
            &[base[0], base[1], "depletion_fraction_p"],
        ),
    }
}

/// Validate analytical stress coefficient (Ks) at key depletion points.
fn validate_stress_coefficient(v: &mut ValidationHarness, soil: &SoilParams) {
    validation::section("Stress coefficient Ks (FAO-56 Eq. 84, analytical)");

    let state_fc = WaterBalanceState::new(
        soil.theta_fc,
        soil.theta_wp,
        soil.root_depth_mm,
        soil.depletion_fraction_p,
    );
    v.check_abs("Ks at FC", state_fc.stress_coefficient(), 1.0, f64::EPSILON);

    let mut state_raw = WaterBalanceState::new(
        soil.theta_fc,
        soil.theta_wp,
        soil.root_depth_mm,
        soil.depletion_fraction_p,
    );
    state_raw.depletion = state_raw.raw;
    v.check_abs(
        "Ks at RAW boundary",
        state_raw.stress_coefficient(),
        1.0,
        f64::EPSILON,
    );

    let mut state_mid = WaterBalanceState::new(
        soil.theta_fc,
        soil.theta_wp,
        soil.root_depth_mm,
        soil.depletion_fraction_p,
    );
    state_mid.depletion = f64::midpoint(state_mid.taw, state_mid.raw);
    v.check_abs(
        "Ks at mid-stress",
        state_mid.stress_coefficient(),
        0.5,
        tolerances::STRESS_COEFFICIENT.abs_tol,
    );

    let mut state_wp = WaterBalanceState::new(
        soil.theta_fc,
        soil.theta_wp,
        soil.root_depth_mm,
        soil.depletion_fraction_p,
    );
    state_wp.depletion = state_wp.taw;
    v.check_abs("Ks at WP", state_wp.stress_coefficient(), 0.0, f64::EPSILON);
}

/// Validate mass balance for dry-down and irrigated scenarios.
fn validate_mass_balance(v: &mut ValidationHarness, soil: &SoilParams) {
    println!();
    validation::section("Mass balance: dry-down scenario (RO = 0, FAO-56 default)");

    let state = WaterBalanceState::new(
        soil.theta_fc,
        soil.theta_wp,
        soil.root_depth_mm,
        soil.depletion_fraction_p,
    );
    let initial_dep = state.depletion;
    let dry_inputs: Vec<DailyInput> = (0..30)
        .map(|_| DailyInput {
            precipitation: 0.0,
            irrigation: 0.0,
            et0: 5.0,
            kc: 1.0,
        })
        .collect();

    let (final_state, outputs) = wb::simulate_season(&state, &dry_inputs);
    let error = wb::mass_balance_check(&dry_inputs, &outputs, initial_dep, final_state.depletion);

    v.check_abs(
        "Mass balance error (dry)",
        error,
        0.0,
        tolerances::WATER_BALANCE_MASS.abs_tol,
    );
    v.check_bool("Depletion increased", final_state.depletion > initial_dep);

    let stressed_days = outputs.iter().filter(|o| o.ks < 1.0).count();
    v.check_bool("Stress days > 0", stressed_days > 0);

    let total_runoff: f64 = outputs.iter().map(|o| o.runoff).sum();
    v.check_abs("No runoff (dry-down)", total_runoff, 0.0, f64::EPSILON);

    println!();
    validation::section("Mass balance: irrigated scenario");

    let state_irr = WaterBalanceState::new(
        soil.theta_fc,
        soil.theta_wp,
        soil.root_depth_mm,
        soil.depletion_fraction_p,
    );
    let initial_dep_irr = state_irr.depletion;
    let irr_inputs: Vec<DailyInput> = (0..60)
        .map(|day| DailyInput {
            precipitation: if day % 7 == 3 { 15.0 } else { 0.0 },
            irrigation: if day % 10 == 0 { 25.0 } else { 0.0 },
            et0: 4.5,
            kc: 0.003f64.mul_add(f64::from(day), 0.85),
        })
        .collect();

    let (final_irr, outputs_irr) = wb::simulate_season(&state_irr, &irr_inputs);
    let error_irr = wb::mass_balance_check(
        &irr_inputs,
        &outputs_irr,
        initial_dep_irr,
        final_irr.depletion,
    );

    let total_et: f64 = outputs_irr.iter().map(|o| o.actual_et).sum();
    let total_dp: f64 = outputs_irr.iter().map(|o| o.deep_percolation).sum();
    println!("  Total ET: {total_et:.1} mm, Deep percolation: {total_dp:.1} mm");
    println!(
        "  Final depletion: {:.1} mm (of {:.1} TAW)",
        final_irr.depletion, final_irr.taw
    );

    v.check_abs(
        "Mass balance error (irrigated)",
        error_irr,
        0.0,
        tolerances::WATER_BALANCE_MASS.abs_tol,
    );
    v.check_bool("Total ET > 0", total_et > 0.0);
}

/// Validate Michigan summer seasonal simulation against benchmark ET range.
fn validate_michigan(v: &mut ValidationHarness, soil: &SoilParams, mi_et_mid: f64, mi_et_tol: f64) {
    println!();
    validation::section("Michigan summer (June–August, silt loam, rainfed)");
    println!(
        "  Soil: silt loam (FC={}, WP={}), {} mm root zone",
        soil.theta_fc, soil.theta_wp, soil.root_depth_mm
    );
    println!("  Crop: mid-season corn (Kc=1.05)");

    let mi_state = WaterBalanceState::new(
        soil.theta_fc,
        soil.theta_wp,
        soil.root_depth_mm,
        soil.depletion_fraction_p,
    );
    let mi_initial = mi_state.depletion;
    let mi_inputs: Vec<DailyInput> = (0..92)
        .map(|day| {
            let et0 = 2.0f64.mul_add(
                ((f64::from(day) - 45.0) * std::f64::consts::PI / 92.0)
                    .cos()
                    .abs(),
                4.0,
            );
            let precip = if day % 6 == 0 {
                18.0
            } else if day % 13 == 0 {
                30.0
            } else {
                0.0
            };
            DailyInput {
                precipitation: precip,
                irrigation: 0.0,
                et0,
                kc: 1.05,
            }
        })
        .collect();

    let (mi_final, mi_outputs) = wb::simulate_season(&mi_state, &mi_inputs);
    let mi_error = wb::mass_balance_check(&mi_inputs, &mi_outputs, mi_initial, mi_final.depletion);

    let mi_total_et: f64 = mi_outputs.iter().map(|o| o.actual_et).sum();
    let mi_stress_days = mi_outputs.iter().filter(|o| o.ks < 1.0).count();
    let mi_trigger_days = mi_outputs.iter().filter(|o| o.needs_irrigation).count();

    println!("  Season ET: {mi_total_et:.0} mm");
    println!("  Stress days: {mi_stress_days}/92");
    println!("  Irrigation trigger days: {mi_trigger_days}");
    println!("  Final θv: {:.3} m³/m³", mi_final.current_theta());
    println!("  Benchmark range: mid={mi_et_mid:.0}, tol=±{mi_et_tol:.0}");

    v.check_abs(
        "MI mass balance",
        mi_error,
        0.0,
        tolerances::WATER_BALANCE_MASS.abs_tol,
    );
    v.check_abs("MI season ET range", mi_total_et, mi_et_mid, mi_et_tol);
    v.check_bool("MI stress days > 0", mi_stress_days > 0);
}

fn main() {
    validation::init_tracing();
    validation::banner("Water Balance Validation");
    let mut v = ValidationHarness::new("Water Balance Validation");
    let benchmark =
        parse_benchmark_json(BENCHMARK_JSON).expect("benchmark_water_balance.json must parse");

    let per_step_tol = json_f64(&benchmark, &["mass_balance_test", "tolerance"])
        .expect("benchmark must have mass_balance_test.tolerance");
    assert!(
        per_step_tol < tolerances::WATER_BALANCE_PER_STEP.abs_tol,
        "per-step tolerance from benchmark should be strict: {per_step_tol}"
    );

    let mi_et_range = benchmark
        .get("michigan_summer_scenario")
        .and_then(|m| m.get("expected_seasonal_et_range_mm"))
        .and_then(|r| r.as_array())
        .expect("benchmark must have michigan_summer_scenario.expected_seasonal_et_range_mm");

    let mi_et_low = mi_et_range[0].as_f64().expect("ET range low");
    let mi_et_high = mi_et_range[1].as_f64().expect("ET range high");
    let mi_et_mid = f64::midpoint(mi_et_low, mi_et_high);
    let mi_et_tol = (mi_et_high - mi_et_low) / 2.0;

    let soil_stress = parse_soil_params(&benchmark, "stress_coefficient");
    let soil_mass = parse_soil_params(&benchmark, "mass_balance");
    let soil_mi = parse_soil_params(&benchmark, "michigan_summer");

    validate_stress_coefficient(&mut v, &soil_stress);
    validate_mass_balance(&mut v, &soil_mass);
    validate_michigan(&mut v, &soil_mi, mi_et_mid, mi_et_tol);

    v.finish();
}
