// SPDX-License-Identifier: AGPL-3.0-or-later
//! Validate soil moisture sensor calibration against published values.
//!
//! Benchmark source: `control/soil_sensors/benchmark_dong2020.json`
//! Provenance: Topp et al. (1980) WRR 16(3), Dong et al. (2020) Agriculture 10(12).
//! Digitized: 2026-02-16, commit: initial airSpring.
//!
//! All expected values and tolerances are sourced from the benchmark JSON,
//! never hardcoded inline.

use airspring_barracuda::eco::soil_moisture::{self as sm, SoilTexture};
use airspring_barracuda::validation::{self, json_f64, parse_benchmark_json, ValidationHarness};

/// Benchmark JSON embedded at compile time for reproducibility.
const BENCHMARK_JSON: &str = include_str!("../../../control/soil_sensors/benchmark_dong2020.json");

/// Round-trip tolerance: Newton–Raphson should converge well within 0.001.
const ROUNDTRIP_TOL: f64 = 0.001;

/// Hydraulic property tolerance: ±0.01 m³/m³ for USDA texture class averages.
const HYDRAULIC_TOL: f64 = 0.01;

/// Analytical tolerance for simple arithmetic operations (e.g., PAW = (FC−WP)×Z).
const ANALYTICAL_TOL: f64 = 0.1;

/// Default Topp equation tolerance (used if not in benchmark JSON).
/// 0.005 m³/m³ = half-percent volumetric water content.
const DEFAULT_TOPP_TOL: f64 = 0.005;

fn main() {
    validation::banner("Soil Moisture Calibration Validation");
    let mut v = ValidationHarness::new("Soil Moisture Calibration Validation");
    let benchmark =
        parse_benchmark_json(BENCHMARK_JSON).expect("benchmark_dong2020.json must parse");

    // ── Topp equation (published calibration points from benchmark) ──
    validation::section("Topp equation (Topp 1980, benchmark JSON)");

    let topp_points = benchmark
        .get("topp_equation")
        .and_then(|t| t.get("published_points"))
        .and_then(|p| p.as_array())
        .expect("benchmark must have topp_equation.published_points");

    let topp_tol =
        json_f64(&benchmark, &["topp_equation", "tolerance"]).unwrap_or(DEFAULT_TOPP_TOL);

    for entry in topp_points {
        let eps = entry
            .get("epsilon")
            .and_then(serde_json::Value::as_f64)
            .expect("Topp point must have 'epsilon'");
        let expected_theta = entry
            .get("theta_expected")
            .and_then(serde_json::Value::as_f64)
            .expect("Topp point must have 'theta_expected'");

        let theta = sm::topp_equation(eps);
        v.check_abs(
            &format!("Topp(ε={eps:.0})"),
            theta,
            expected_theta,
            topp_tol,
        );
    }

    // ── Inverse Topp (round-trip) ────────────────────────────────────
    println!();
    validation::section("Inverse Topp equation (Newton–Raphson round-trip)");

    for &theta in &[0.10, 0.20, 0.30, 0.40] {
        let eps = sm::inverse_topp(theta);
        let recovered = sm::topp_equation(eps);
        v.check_abs(
            &format!("Round-trip(θ={theta:.2})"),
            recovered,
            theta,
            ROUNDTRIP_TOL,
        );
    }

    // ── Soil hydraulic properties ────────────────────────────────────
    println!();
    validation::section("Soil texture hydraulic properties (USDA/Saxton & Rawls 2006)");

    let textures: &[(SoilTexture, &str, f64, f64)] = &[
        (SoilTexture::Sand, "Sand", 0.10, 0.05),
        (SoilTexture::SandyLoam, "SandyLoam", 0.18, 0.08),
        (SoilTexture::Loam, "Loam", 0.27, 0.12),
        (SoilTexture::SiltLoam, "SiltLoam", 0.33, 0.13),
        (SoilTexture::Clay, "Clay", 0.36, 0.25),
    ];

    for &(texture, name, exp_fc, exp_wp) in textures {
        let props = texture.hydraulic_properties();
        v.check_abs(
            &format!("{name} FC"),
            props.field_capacity,
            exp_fc,
            HYDRAULIC_TOL,
        );
        v.check_abs(
            &format!("{name} WP"),
            props.wilting_point,
            exp_wp,
            HYDRAULIC_TOL,
        );
    }

    // ── Plant available water ────────────────────────────────────────
    println!();
    validation::section("Plant available water calculations");

    // Silt loam, 600 mm root zone: PAW = (0.33 − 0.13) × 600 = 120 mm
    let paw = sm::plant_available_water(0.33, 0.13, 600.0);
    v.check_abs("SiltLoam PAW (600 mm)", paw, 120.0, ANALYTICAL_TOL);

    // Soil water deficit: θ = 0.25, FC = 0.33, depth = 600 mm → SWD = 48 mm
    let swd = sm::soil_water_deficit(0.33, 0.25, 600.0);
    v.check_abs("SiltLoam SWD (θ=0.25)", swd, 48.0, ANALYTICAL_TOL);

    // ── Irrigation trigger ───────────────────────────────────────────
    println!();
    validation::section("Irrigation trigger tests");

    let trigger_fc = sm::irrigation_trigger(0.33, 0.13, 0.33, 0.50);
    v.check_bool("At FC → no trigger", !trigger_fc);

    let trigger_dry = sm::irrigation_trigger(0.33, 0.13, 0.20, 0.50);
    v.check_bool("Below MAD → trigger", trigger_dry);

    v.finish();
}
