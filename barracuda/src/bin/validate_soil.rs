//! Validate soil moisture sensor calibration against published values.
//!
//! Benchmark source: `control/soil_sensors/benchmark_dong2020.json`
//! Provenance: Topp et al. (1980) WRR 16(3), Dong et al. (2020) Agriculture 10(12).
//! Digitized: 2026-02-16, commit: initial airSpring.

use airspring_barracuda::eco::soil_moisture::{self as sm, SoilTexture};
use airspring_barracuda::validation::ValidationRunner;

fn main() {
    let mut v = ValidationRunner::new("Soil Moisture Calibration Validation");

    // ── Topp equation (published calibration points) ─────────────
    v.section("Topp equation (Topp 1980, Table 1)");

    // Topp (1980) Table 1: dielectric → θv for mineral soils
    // Source: WRR 16(3), pp. 574–582, Table 1
    let topp_data = [
        (3.0, 0.031),
        (5.0, 0.083),
        (10.0, 0.187),
        (15.0, 0.271),
        (20.0, 0.347),
        (25.0, 0.405),
        (30.0, 0.440),
    ];

    for (eps, expected_theta) in topp_data {
        let theta = sm::topp_equation(eps);
        v.check(&format!("Topp(ε={eps:.0})"), theta, expected_theta, 0.02);
    }

    // ── Inverse Topp (round-trip) ────────────────────────────────
    println!();
    v.section("Inverse Topp equation (Newton–Raphson round-trip)");

    for &theta in &[0.10, 0.20, 0.30, 0.40] {
        let eps = sm::inverse_topp(theta);
        let recovered = sm::topp_equation(eps);
        v.check(
            &format!("Round-trip(θ={theta:.2})"),
            recovered,
            theta,
            0.001,
        );
    }

    // ── Soil hydraulic properties ────────────────────────────────
    println!();
    v.section("Soil texture hydraulic properties (USDA/Saxton & Rawls 2006)");

    let textures: &[(SoilTexture, &str, f64, f64)] = &[
        (SoilTexture::Sand, "Sand", 0.10, 0.05),
        (SoilTexture::SandyLoam, "SandyLoam", 0.18, 0.08),
        (SoilTexture::Loam, "Loam", 0.27, 0.12),
        (SoilTexture::SiltLoam, "SiltLoam", 0.33, 0.13),
        (SoilTexture::Clay, "Clay", 0.36, 0.25),
    ];

    for &(texture, name, exp_fc, exp_wp) in textures {
        let props = texture.hydraulic_properties();
        v.check(&format!("{name} FC"), props.field_capacity, exp_fc, 0.01);
        v.check(&format!("{name} WP"), props.wilting_point, exp_wp, 0.01);
    }

    // ── Plant available water ────────────────────────────────────
    println!();
    v.section("Plant available water calculations");

    // Silt loam, 600 mm root zone: PAW = (0.33 − 0.13) × 600 = 120 mm
    let paw = sm::plant_available_water(0.33, 0.13, 600.0);
    v.check("SiltLoam PAW (600 mm)", paw, 120.0, 0.1);

    // Soil water deficit: θ = 0.25, FC = 0.33, depth = 600 mm → SWD = 48 mm
    let swd = sm::soil_water_deficit(0.33, 0.25, 600.0);
    v.check("SiltLoam SWD (θ=0.25)", swd, 48.0, 0.1);

    // ── Irrigation trigger ───────────────────────────────────────
    println!();
    v.section("Irrigation trigger tests");

    let trigger_fc = sm::irrigation_trigger(0.33, 0.13, 0.33, 0.50);
    v.check_bool("At FC → no trigger", trigger_fc, false);

    let trigger_dry = sm::irrigation_trigger(0.33, 0.13, 0.20, 0.50);
    v.check_bool("Below MAD → trigger", trigger_dry, true);

    v.finish();
}
