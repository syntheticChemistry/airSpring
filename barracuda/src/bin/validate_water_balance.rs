//! Validate water balance model against analytical solutions and mass conservation.
//!
//! Tests:
//! 1. Mass balance closure (P + I = ET + DP + RO + ΔS)
//! 2. Stress coefficient behavior (Ks)
//! 3. Irrigation triggering
//! 4. Seasonal simulation with realistic Michigan data

use airspring_barracuda::eco::water_balance::{self as wb, DailyInput, WaterBalanceState};

fn check(label: &str, actual: f64, expected: f64, tolerance: f64) -> bool {
    let pass = (actual - expected).abs() <= tolerance;
    let tag = if pass { "OK" } else { "FAIL" };
    println!(
        "  [{}]  {}: {:.4} (expected {:.4}, tol {:.4})",
        tag, label, actual, expected, tolerance
    );
    pass
}

fn main() {
    println!("═══════════════════════════════════════════════════════════");
    println!("  airSpring Water Balance Validation");
    println!("  Reference: FAO-56 Ch. 8 soil water balance");
    println!("═══════════════════════════════════════════════════════════\n");

    let mut total = 0u32;
    let mut passed = 0u32;

    // ── Stress coefficient tests ────────────────────────────────
    println!("── Stress coefficient (Ks) tests ──");

    let state_fc = WaterBalanceState::new(0.33, 0.13, 600.0, 0.5);
    total += 1;
    if check("Ks at FC", state_fc.stress_coefficient(), 1.0, 0.0) { passed += 1; }

    let mut state_mid = WaterBalanceState::new(0.33, 0.13, 600.0, 0.5);
    state_mid.depletion = state_mid.raw; // At RAW boundary
    total += 1;
    if check("Ks at RAW boundary", state_mid.stress_coefficient(), 1.0, 0.0) { passed += 1; }

    let mut state_stressed = WaterBalanceState::new(0.33, 0.13, 600.0, 0.5);
    state_stressed.depletion = (state_stressed.taw + state_stressed.raw) / 2.0;
    total += 1;
    if check("Ks at mid-stress", state_stressed.stress_coefficient(), 0.5, 0.01) { passed += 1; }

    let mut state_wp = WaterBalanceState::new(0.33, 0.13, 600.0, 0.5);
    state_wp.depletion = state_wp.taw;
    total += 1;
    if check("Ks at WP", state_wp.stress_coefficient(), 0.0, 0.0) { passed += 1; }

    // ── Mass balance (no irrigation, no precip) ─────────────────
    println!("\n── Mass balance: dry-down scenario ──");

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

    total += 1;
    if check("Mass balance error (dry)", error, 0.0, 0.01) { passed += 1; }

    // Depletion should increase
    total += 1;
    if check(
        "Depletion increased",
        (final_state.depletion > initial_dep) as u32 as f64,
        1.0,
        0.0,
    ) {
        passed += 1;
    }

    // Should hit stress at some point
    let stressed_days = outputs.iter().filter(|o| o.ks < 1.0).count();
    total += 1;
    if check("Stress days > 0", (stressed_days > 0) as u32 as f64, 1.0, 0.0) {
        passed += 1;
    }

    // ── Mass balance (with irrigation) ──────────────────────────
    println!("\n── Mass balance: irrigated scenario ──");

    let state_irr = WaterBalanceState::new(0.30, 0.10, 500.0, 0.5);
    let initial_dep_irr = state_irr.depletion;
    let irr_inputs: Vec<DailyInput> = (0..60)
        .map(|day| DailyInput {
            precipitation: if day % 7 == 3 { 15.0 } else { 0.0 },
            irrigation: if day % 10 == 0 { 25.0 } else { 0.0 },
            et0: 4.5,
            kc: 0.85 + 0.003 * day as f64, // growing Kc
        })
        .collect();

    let (final_irr, outputs_irr) = wb::simulate_season(&state_irr, &irr_inputs);
    let error_irr = wb::mass_balance_check(
        &irr_inputs,
        &outputs_irr,
        initial_dep_irr,
        final_irr.depletion,
    );

    total += 1;
    if check("Mass balance error (irrigated)", error_irr, 0.0, 0.01) { passed += 1; }

    let total_et: f64 = outputs_irr.iter().map(|o| o.actual_et).sum();
    let total_dp: f64 = outputs_irr.iter().map(|o| o.deep_percolation).sum();
    println!("  Total ET: {:.1} mm, Deep percolation: {:.1} mm", total_et, total_dp);
    println!("  Final depletion: {:.1} mm (of {:.1} TAW)", final_irr.depletion, final_irr.taw);

    total += 1;
    if check("Total ET > 0", (total_et > 0.0) as u32 as f64, 1.0, 0.0) { passed += 1; }

    // ── Michigan summer simulation ──────────────────────────────
    println!("\n── Michigan summer (June-August, silt loam) ──");

    // Typical Michigan summer: ET₀ ~4-6 mm/day, rainfall every 5-7 days
    let mi_state = WaterBalanceState::new(0.33, 0.13, 600.0, 0.5);
    let mi_initial = mi_state.depletion;
    let mi_inputs: Vec<DailyInput> = (0..92)
        .map(|day| {
            let et0 = 4.0 + 2.0 * ((day as f64 - 45.0) * std::f64::consts::PI / 92.0).cos().abs();
            let precip = if day % 6 == 0 { 18.0 } else if day % 13 == 0 { 30.0 } else { 0.0 };
            DailyInput {
                precipitation: precip,
                irrigation: 0.0, // rainfed
                et0,
                kc: 1.05, // mid-season corn
            }
        })
        .collect();

    let (mi_final, mi_outputs) = wb::simulate_season(&mi_state, &mi_inputs);
    let mi_error = wb::mass_balance_check(&mi_inputs, &mi_outputs, mi_initial, mi_final.depletion);

    let mi_total_et: f64 = mi_outputs.iter().map(|o| o.actual_et).sum();
    let mi_stress_days = mi_outputs.iter().filter(|o| o.ks < 1.0).count();
    let mi_trigger_days = mi_outputs.iter().filter(|o| o.needs_irrigation).count();

    println!("  Season ET: {:.0} mm", mi_total_et);
    println!("  Stress days: {}/{}", mi_stress_days, 92);
    println!("  Irrigation trigger days: {}", mi_trigger_days);
    println!("  Final θv: {:.3} m³/m³", mi_final.current_theta());

    total += 1;
    if check("MI mass balance", mi_error, 0.0, 0.01) { passed += 1; }

    // Season ET for Michigan corn: 400-550 mm
    total += 1;
    if check("MI season ET range", mi_total_et, 450.0, 100.0) { passed += 1; }

    // Should have some stress in rainfed Michigan
    total += 1;
    if check("MI stress days > 0", (mi_stress_days > 0) as u32 as f64, 1.0, 0.0) { passed += 1; }

    // ── Summary ─────────────────────────────────────────────────
    println!("\n═══════════════════════════════════════════════════════════");
    println!(
        "  Water Balance Validation: {}/{} checks passed",
        passed, total
    );
    if passed == total {
        println!("  RESULT: PASS");
    } else {
        println!("  RESULT: FAIL ({} checks failed)", total - passed);
    }
    println!("═══════════════════════════════════════════════════════════");

    std::process::exit(if passed == total { 0 } else { 1 });
}
