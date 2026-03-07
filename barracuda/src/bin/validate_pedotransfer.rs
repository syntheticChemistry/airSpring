// SPDX-License-Identifier: AGPL-3.0-or-later
#![deny(clippy::unwrap_used)]
#![warn(clippy::pedantic)]
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
//! Validate Saxton & Rawls (2006) pedotransfer functions against Python baseline (Exp 023).
//!
//! Saxton KE, Rawls WJ (2006) "Soil water characteristic estimates by texture
//! and organic matter for hydrologic solutions." SSSAJ 70(5):1569-1578.
//!
//! Provenance: script=`control/pedotransfer/saxton_rawls.py`, commit=fad2e1b, date=2026-02-27

use airspring_barracuda::eco::soil_moisture::{saxton_rawls, SaxtonRawlsInput};
use airspring_barracuda::tolerances::{PEDOTRANSFER_KSAT, PEDOTRANSFER_MOISTURE};
use airspring_barracuda::validation::{
    self, json_array, json_f64_required, json_field, json_object_required, parse_benchmark_json,
    ValidationHarness,
};

const BENCHMARK_JSON: &str =
    include_str!("../../../control/pedotransfer/benchmark_pedotransfer.json");

fn validate_loam_intermediates(v: &mut ValidationHarness, bench: &serde_json::Value) {
    validation::section("Loam analytical intermediates");
    let s = json_f64_required(bench, &["loam_intermediates", "S"]);
    let c = json_f64_required(bench, &["loam_intermediates", "C"]);
    let om = json_f64_required(bench, &["loam_intermediates", "OM"]);
    let input = SaxtonRawlsInput {
        sand: s,
        clay: c,
        om_pct: om,
    };
    let r = saxton_rawls(&input);

    v.check_abs(
        "loam: θ_wp",
        r.theta_wp,
        json_f64_required(bench, &["loam_intermediates", "theta_1500"]),
        PEDOTRANSFER_MOISTURE.abs_tol,
    );
    v.check_abs(
        "loam: θ_fc",
        r.theta_fc,
        json_f64_required(bench, &["loam_intermediates", "theta_33"]),
        PEDOTRANSFER_MOISTURE.abs_tol,
    );
    v.check_abs(
        "loam: θ_s",
        r.theta_s,
        json_f64_required(bench, &["loam_intermediates", "theta_s"]),
        PEDOTRANSFER_MOISTURE.abs_tol,
    );
    v.check_abs(
        "loam: λ",
        r.lambda,
        json_f64_required(bench, &["loam_intermediates", "lambda"]),
        PEDOTRANSFER_MOISTURE.abs_tol,
    );
    v.check_abs(
        "loam: Ksat",
        r.ksat_mm_hr,
        json_f64_required(bench, &["loam_intermediates", "ksat_mm_hr"]),
        PEDOTRANSFER_KSAT.abs_tol,
    );
}

fn validate_texture_classes(v: &mut ValidationHarness, bench: &serde_json::Value) {
    validation::section("USDA texture classes");
    let tol_m = json_f64_required(bench, &["tol_moisture"]);
    let tol_k = json_f64_required(bench, &["tol_ksat"]);

    let textures = json_object_required(bench, &["texture_classes"]);
    for (name, data) in textures {
        let input = SaxtonRawlsInput {
            sand: json_field(data, "S"),
            clay: json_field(data, "C"),
            om_pct: json_field(data, "OM"),
        };
        let r = saxton_rawls(&input);

        v.check_abs(
            &format!("{name}: θ_wp"),
            r.theta_wp,
            json_field(data, "theta_wp"),
            tol_m,
        );
        v.check_abs(
            &format!("{name}: θ_fc"),
            r.theta_fc,
            json_field(data, "theta_fc"),
            tol_m,
        );
        v.check_abs(
            &format!("{name}: θ_s"),
            r.theta_s,
            json_field(data, "theta_s"),
            tol_m,
        );
        v.check_abs(
            &format!("{name}: Ksat"),
            r.ksat_mm_hr,
            json_field(data, "ksat_mm_hr"),
            tol_k,
        );

        // Physical ordering
        v.check_bool(
            &format!("{name}: wp < fc < θs"),
            r.theta_wp < r.theta_fc && r.theta_fc < r.theta_s,
        );

        // AWC
        let awc = r.theta_fc - r.theta_wp;
        let awc_range = json_array(bench, &["thresholds", "awc_range"]);
        let lo = json_f64_required(&awc_range[0], &[]);
        let hi = json_f64_required(&awc_range[1], &[]);
        v.check_bool(
            &format!("{name}: AWC {awc:.3} in [{lo}, {hi}]"),
            (lo..=hi).contains(&awc),
        );
    }
}

fn validate_physical_ordering(v: &mut ValidationHarness, bench: &serde_json::Value) {
    validation::section("Physical ordering across textures");
    let textures = json_object_required(bench, &["texture_classes"]);

    let sand_data = &textures["sand"];
    let clay_data = &textures["clay"];

    let sand = saxton_rawls(&SaxtonRawlsInput {
        sand: json_field(sand_data, "S"),
        clay: json_field(sand_data, "C"),
        om_pct: json_field(sand_data, "OM"),
    });
    let clay = saxton_rawls(&SaxtonRawlsInput {
        sand: json_field(clay_data, "S"),
        clay: json_field(clay_data, "C"),
        om_pct: json_field(clay_data, "OM"),
    });

    v.check_bool(
        &format!(
            "sand Ksat ({:.1}) > clay Ksat ({:.1})",
            sand.ksat_mm_hr, clay.ksat_mm_hr
        ),
        sand.ksat_mm_hr > clay.ksat_mm_hr,
    );
    v.check_bool(
        &format!(
            "clay WP ({:.3}) > sand WP ({:.3})",
            clay.theta_wp, sand.theta_wp
        ),
        clay.theta_wp > sand.theta_wp,
    );
    v.check_bool(
        &format!(
            "clay FC ({:.3}) > sand FC ({:.3})",
            clay.theta_fc, sand.theta_fc
        ),
        clay.theta_fc > sand.theta_fc,
    );
}

fn validate_om_sensitivity(v: &mut ValidationHarness) {
    validation::section("Organic matter sensitivity");
    let lo = saxton_rawls(&SaxtonRawlsInput {
        sand: 0.40,
        clay: 0.20,
        om_pct: 0.5,
    });
    let hi = saxton_rawls(&SaxtonRawlsInput {
        sand: 0.40,
        clay: 0.20,
        om_pct: 5.0,
    });
    v.check_bool(
        &format!(
            "OM 5% WP ({:.4}) > OM 0.5% WP ({:.4})",
            hi.theta_wp, lo.theta_wp
        ),
        hi.theta_wp > lo.theta_wp,
    );
    v.check_bool(
        &format!(
            "OM 5% FC ({:.4}) > OM 0.5% FC ({:.4})",
            hi.theta_fc, lo.theta_fc
        ),
        hi.theta_fc > lo.theta_fc,
    );
}

fn main() {
    validation::init_tracing();
    validation::banner("Pedotransfer (Saxton-Rawls) Validation");
    let mut v = ValidationHarness::new("Pedotransfer Saxton-Rawls");
    let bench = match parse_benchmark_json(BENCHMARK_JSON) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("FATAL: invalid benchmark JSON: {e}");
            std::process::exit(1);
        }
    };

    validate_loam_intermediates(&mut v, &bench);
    validate_texture_classes(&mut v, &bench);
    validate_physical_ordering(&mut v, &bench);
    validate_om_sensitivity(&mut v);

    v.finish();
}
