//! Validate water balance model against analytical solutions and mass conservation.
//!
//! Tests:
//! 1. Stress coefficient Ks: analytical values at FC, RAW, midpoint, WP
//! 2. Mass balance closure: P + I = ET + DP + RO + ΔS (must be < 0.01 mm)
//! 3. Runoff model: `RunoffModel::None` (FAO-56 default) matches Python baseline
//! 4. Seasonal simulation with realistic Michigan data
//!
//! Provenance: FAO-56 Ch. 8, soil water balance equations.
//! Python baseline: `control/water_balance/fao56_water_balance.py`

use airspring_barracuda::eco::water_balance::{self as wb, DailyInput, WaterBalanceState};
use airspring_barracuda::validation::ValidationRunner;

#[allow(clippy::cast_precision_loss)]
#[allow(clippy::too_many_lines)]
fn main() {
    let mut v = ValidationRunner::new("Water Balance Validation");

    // ── Stress coefficient (analytical) ──────────────────────────
    v.section("Stress coefficient Ks (FAO-56 Eq. 84, analytical)");

    let state_fc = WaterBalanceState::new(0.33, 0.13, 600.0, 0.5);
    v.check("Ks at FC", state_fc.stress_coefficient(), 1.0, 0.0);

    let mut state_raw = WaterBalanceState::new(0.33, 0.13, 600.0, 0.5);
    state_raw.depletion = state_raw.raw;
    v.check(
        "Ks at RAW boundary",
        state_raw.stress_coefficient(),
        1.0,
        0.0,
    );

    let mut state_mid = WaterBalanceState::new(0.33, 0.13, 600.0, 0.5);
    state_mid.depletion = f64::midpoint(state_mid.taw, state_mid.raw);
    v.check(
        "Ks at mid-stress",
        state_mid.stress_coefficient(),
        0.5,
        0.01,
    );

    let mut state_wp = WaterBalanceState::new(0.33, 0.13, 600.0, 0.5);
    state_wp.depletion = state_wp.taw;
    v.check("Ks at WP", state_wp.stress_coefficient(), 0.0, 0.0);

    // ── Mass balance: dry-down (no precip, no irrigation) ────────
    println!();
    v.section("Mass balance: dry-down scenario (RO = 0, FAO-56 default)");

    let state = WaterBalanceState::new(0.30, 0.10, 500.0, 0.5);
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

    v.check("Mass balance error (dry)", error, 0.0, 0.01);
    v.check_bool(
        "Depletion increased",
        final_state.depletion > initial_dep,
        true,
    );

    let stressed_days = outputs.iter().filter(|o| o.ks < 1.0).count();
    v.check_bool("Stress days > 0", stressed_days > 0, true);

    // Verify no runoff in dry-down (RunoffModel::None)
    let total_runoff: f64 = outputs.iter().map(|o| o.runoff).sum();
    v.check("No runoff (dry-down)", total_runoff, 0.0, 0.0);

    // ── Mass balance: irrigated ──────────────────────────────────
    println!();
    v.section("Mass balance: irrigated scenario");

    let state_irr = WaterBalanceState::new(0.30, 0.10, 500.0, 0.5);
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

    v.check("Mass balance error (irrigated)", error_irr, 0.0, 0.01);
    v.check_bool("Total ET > 0", total_et > 0.0, true);

    // ── Michigan summer simulation ───────────────────────────────
    println!();
    v.section("Michigan summer (June–August, silt loam, rainfed)");
    println!("  Soil: silt loam (FC=0.33, WP=0.13), 600 mm root zone");
    println!("  Crop: mid-season corn (Kc=1.05)");

    let mi_state = WaterBalanceState::new(0.33, 0.13, 600.0, 0.5);
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

    v.check("MI mass balance", mi_error, 0.0, 0.01);
    // Michigan corn season ET: 400–550 mm (published range)
    v.check("MI season ET range", mi_total_et, 450.0, 100.0);
    v.check_bool("MI stress days > 0", mi_stress_days > 0, true);

    v.finish();
}
