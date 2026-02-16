//! Validate soil moisture sensor calibration against published values.
//!
//! Reference: Topp et al. (1980) Water Resources Research 16(3), 574-582
//! Also: Dong et al. (2020) Agriculture 10(12), 598

use airspring_barracuda::eco::soil_moisture::{self as sm, SoilTexture};

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
    println!("  airSpring Soil Moisture Calibration Validation");
    println!("  Reference: Topp et al. (1980), Dong et al. (2020)");
    println!("═══════════════════════════════════════════════════════════\n");

    let mut total = 0u32;
    let mut passed = 0u32;

    // ── Topp equation validation ────────────────────────────────
    println!("── Topp equation (published calibration points) ──");

    // Topp (1980) Table 1: dielectric → θv for mineral soils
    let topp_data = [
        (3.0, 0.031),  // dry soil
        (5.0, 0.083),  // slightly moist
        (10.0, 0.187), // moist
        (15.0, 0.271), // wet
        (20.0, 0.347), // very wet
        (25.0, 0.405), // saturated
        (30.0, 0.440), // near saturation
    ];

    for &(eps, expected_theta) in &topp_data {
        let theta = sm::topp_equation(eps);
        total += 1;
        if check(
            &format!("Topp(ε={:.0})", eps),
            theta,
            expected_theta,
            0.02,
        ) {
            passed += 1;
        }
    }

    // ── Inverse Topp (round-trip) ───────────────────────────────
    println!("\n── Inverse Topp equation (round-trip) ──");
    for &theta in &[0.10, 0.20, 0.30, 0.40] {
        let eps = sm::inverse_topp(theta);
        let recovered = sm::topp_equation(eps);
        total += 1;
        if check(
            &format!("Round-trip(θ={:.2})", theta),
            recovered,
            theta,
            0.001,
        ) {
            passed += 1;
        }
    }

    // ── Soil hydraulic properties ───────────────────────────────
    println!("\n── Soil texture hydraulic properties (USDA) ──");

    let textures = [
        (SoilTexture::Sand, "Sand", 0.10, 0.05),
        (SoilTexture::SandyLoam, "SandyLoam", 0.18, 0.08),
        (SoilTexture::Loam, "Loam", 0.27, 0.12),
        (SoilTexture::SiltLoam, "SiltLoam", 0.33, 0.13),
        (SoilTexture::Clay, "Clay", 0.36, 0.25),
    ];

    for &(ref texture, name, exp_fc, exp_wp) in &textures {
        let props = texture.hydraulic_properties();
        total += 1;
        if check(&format!("{} FC", name), props.field_capacity, exp_fc, 0.01) {
            passed += 1;
        }
        total += 1;
        if check(&format!("{} WP", name), props.wilting_point, exp_wp, 0.01) {
            passed += 1;
        }
    }

    // ── Plant available water ───────────────────────────────────
    println!("\n── Plant available water calculations ──");

    // Silt loam, 600mm root zone: PAW = (0.33-0.13)*600 = 120mm
    let paw = sm::plant_available_water(0.33, 0.13, 600.0);
    total += 1;
    if check("SiltLoam PAW (600mm)", paw, 120.0, 0.1) {
        passed += 1;
    }

    // Soil water deficit
    let swd = sm::soil_water_deficit(0.33, 0.25, 600.0);
    total += 1;
    if check("SiltLoam SWD (θ=0.25)", swd, 48.0, 0.1) {
        passed += 1;
    }

    // ── Irrigation trigger ──────────────────────────────────────
    println!("\n── Irrigation trigger tests ──");

    // At FC: no trigger
    total += 1;
    let trigger_fc = sm::irrigation_trigger(0.33, 0.13, 0.33, 0.50);
    if check("At FC → no trigger", trigger_fc as u32 as f64, 0.0, 0.0) {
        passed += 1;
    }

    // Below MAD: trigger
    total += 1;
    let trigger_dry = sm::irrigation_trigger(0.33, 0.13, 0.20, 0.50);
    if check("Below MAD → trigger", trigger_dry as u32 as f64, 1.0, 0.0) {
        passed += 1;
    }

    // ── Summary ─────────────────────────────────────────────────
    println!("\n═══════════════════════════════════════════════════════════");
    println!(
        "  Soil Moisture Validation: {}/{} checks passed",
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
