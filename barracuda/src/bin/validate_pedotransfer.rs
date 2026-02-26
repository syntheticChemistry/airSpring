// SPDX-License-Identifier: AGPL-3.0-or-later
//! Validate Saxton & Rawls (2006) pedotransfer functions against Python baseline (Exp 023).
//!
//! Saxton KE, Rawls WJ (2006) "Soil water characteristic estimates by texture
//! and organic matter for hydrologic solutions." SSSAJ 70(5):1569-1578.

use airspring_barracuda::eco::soil_moisture::{saxton_rawls, SaxtonRawlsInput};
use airspring_barracuda::validation::{self, parse_benchmark_json, ValidationHarness};

const BENCHMARK_JSON: &str =
    include_str!("../../../control/pedotransfer/benchmark_pedotransfer.json");

fn validate_loam_intermediates(v: &mut ValidationHarness, bench: &serde_json::Value) {
    validation::section("Loam analytical intermediates");
    let li = &bench["loam_intermediates"];
    let s = li["S"].as_f64().unwrap();
    let c = li["C"].as_f64().unwrap();
    let om = li["OM"].as_f64().unwrap();
    let input = SaxtonRawlsInput {
        sand: s,
        clay: c,
        om_pct: om,
    };
    let r = saxton_rawls(&input);

    v.check_abs(
        "loam: θ_wp",
        r.theta_wp,
        li["theta_1500"].as_f64().unwrap(),
        1e-4,
    );
    v.check_abs(
        "loam: θ_fc",
        r.theta_fc,
        li["theta_33"].as_f64().unwrap(),
        1e-4,
    );
    v.check_abs(
        "loam: θ_s",
        r.theta_s,
        li["theta_s"].as_f64().unwrap(),
        1e-4,
    );
    v.check_abs("loam: λ", r.lambda, li["lambda"].as_f64().unwrap(), 1e-4);
    v.check_abs(
        "loam: Ksat",
        r.ksat_mm_hr,
        li["ksat_mm_hr"].as_f64().unwrap(),
        0.5,
    );
}

fn validate_texture_classes(v: &mut ValidationHarness, bench: &serde_json::Value) {
    validation::section("USDA texture classes");
    let tol_m = bench["tol_moisture"].as_f64().unwrap();
    let tol_k = bench["tol_ksat"].as_f64().unwrap();

    let textures = bench["texture_classes"].as_object().unwrap();
    for (name, data) in textures {
        let input = SaxtonRawlsInput {
            sand: data["S"].as_f64().unwrap(),
            clay: data["C"].as_f64().unwrap(),
            om_pct: data["OM"].as_f64().unwrap(),
        };
        let r = saxton_rawls(&input);

        v.check_abs(
            &format!("{name}: θ_wp"),
            r.theta_wp,
            data["theta_wp"].as_f64().unwrap(),
            tol_m,
        );
        v.check_abs(
            &format!("{name}: θ_fc"),
            r.theta_fc,
            data["theta_fc"].as_f64().unwrap(),
            tol_m,
        );
        v.check_abs(
            &format!("{name}: θ_s"),
            r.theta_s,
            data["theta_s"].as_f64().unwrap(),
            tol_m,
        );
        v.check_abs(
            &format!("{name}: Ksat"),
            r.ksat_mm_hr,
            data["ksat_mm_hr"].as_f64().unwrap(),
            tol_k,
        );

        // Physical ordering
        v.check_bool(
            &format!("{name}: wp < fc < θs"),
            r.theta_wp < r.theta_fc && r.theta_fc < r.theta_s,
        );

        // AWC
        let awc = r.theta_fc - r.theta_wp;
        let awc_range = &bench["thresholds"]["awc_range"];
        let lo = awc_range[0].as_f64().unwrap();
        let hi = awc_range[1].as_f64().unwrap();
        v.check_bool(
            &format!("{name}: AWC {awc:.3} in [{lo}, {hi}]"),
            (lo..=hi).contains(&awc),
        );
    }
}

fn validate_physical_ordering(v: &mut ValidationHarness, bench: &serde_json::Value) {
    validation::section("Physical ordering across textures");
    let textures = bench["texture_classes"].as_object().unwrap();

    let sand_data = &textures["sand"];
    let clay_data = &textures["clay"];

    let sand = saxton_rawls(&SaxtonRawlsInput {
        sand: sand_data["S"].as_f64().unwrap(),
        clay: sand_data["C"].as_f64().unwrap(),
        om_pct: sand_data["OM"].as_f64().unwrap(),
    });
    let clay = saxton_rawls(&SaxtonRawlsInput {
        sand: clay_data["S"].as_f64().unwrap(),
        clay: clay_data["C"].as_f64().unwrap(),
        om_pct: clay_data["OM"].as_f64().unwrap(),
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
    let bench = parse_benchmark_json(BENCHMARK_JSON).expect("valid benchmark JSON");

    validate_loam_intermediates(&mut v, &bench);
    validate_texture_classes(&mut v, &bench);
    validate_physical_ordering(&mut v, &bench);
    validate_om_sensitivity(&mut v);

    v.finish();
}
