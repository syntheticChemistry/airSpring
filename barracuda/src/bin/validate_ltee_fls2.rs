// SPDX-License-Identifier: AGPL-3.0-or-later
//! LTEE E3 — FLS2 Plant Immunity Sentinel (Dolgikh et al. 2025).
//!
//! Rust validation binary for the FLS2 receptor-ligand binding analysis.
//! Reproduces binding models (Langmuir/Hill/two-site), glycosylation Kd
//! shift, and soil-immune coupling from the Python baseline, then
//! cross-validates against `benchmark_ltee_fls2.json`.
//!
//! # Provenance
//! Paper: Dolgikh VV et al. (2025) "Tuning Yeast Glycosylation for
//! Improved FLS2 Receptor Production" — bioRxiv.
//! Python: `control/ltee_fls2_plant_immunity/ltee_fls2_plant_immunity.py`

#![forbid(unsafe_code)]

use airspring_barracuda::validation::{self, ValidationHarness};

const BENCHMARK_JSON: &str =
    include_str!("../../../control/ltee_fls2_plant_immunity/benchmark_ltee_fls2.json");

// ── Binding models (pure Rust, matching Python) ─────────────────────

fn langmuir(ligand_nm: f64, bmax: f64, kd: f64) -> f64 {
    bmax * ligand_nm / (kd + ligand_nm)
}

fn hill(ligand_nm: f64, bmax: f64, kd: f64, n: f64) -> f64 {
    let ln = ligand_nm.powf(n);
    bmax * ln / (kd.powf(n) + ln)
}

fn two_site(ligand_nm: f64, b1: f64, k1: f64, b2: f64, k2: f64) -> f64 {
    b1 * ligand_nm / (k1 + ligand_nm) + b2 * ligand_nm / (k2 + ligand_nm)
}

// ── Soil-immune coupling (deterministic, matching Python) ───────────

struct SoilCoupling {
    moisture_factor: f64,
    temp_factor: f64,
    microbial_activity: f64,
    flagellin_relative: f64,
}

fn soil_immune_coupling(
    soil_moisture_vwc: f64,
    soil_temp_c: f64,
    microbial_density_cfu_g: f64,
) -> SoilCoupling {
    const THETA_WP: f64 = 0.10;
    const THETA_FC: f64 = 0.33;
    const T_REF: f64 = 25.0;
    const Q10: f64 = 2.0;

    let moisture_factor =
        ((soil_moisture_vwc - THETA_WP) / (THETA_FC - THETA_WP)).clamp(0.0, 1.0);
    #[expect(clippy::suboptimal_flops, reason = "Q10 is a domain constant, not literal 2")]
    let temp_factor = Q10.powf((soil_temp_c - T_REF) / 10.0);
    let activity = moisture_factor * temp_factor;
    let flagellin_relative = activity * microbial_density_cfu_g / 1e7;

    SoilCoupling {
        moisture_factor,
        temp_factor,
        microbial_activity: activity,
        flagellin_relative,
    }
}

// ── Validation logic ────────────────────────────────────────────────

fn validate_benchmark_structure(v: &mut ValidationHarness, bm: &serde_json::Value) {
    v.check_bool(
        "benchmark has checks array",
        bm.get("checks").and_then(|c| c.as_array()).is_some(),
    );
    v.check_bool(
        "benchmark has model_fits",
        bm.get("model_fits").is_some(),
    );
    v.check_bool(
        "benchmark pass_count == 12",
        bm.get("pass_count").and_then(serde_json::Value::as_u64) == Some(12),
    );
    v.check_bool(
        "benchmark fail_count == 0",
        bm.get("fail_count").and_then(serde_json::Value::as_u64) == Some(0),
    );
}

fn validate_model_fits(v: &mut ValidationHarness, bm: &serde_json::Value) {
    let fits = &bm["model_fits"];

    for model in &["langmuir", "hill", "two_site"] {
        let r2 = fits[model]["r_squared"].as_f64().unwrap_or(0.0);
        v.check_lower(&format!("{model} R² > 0.95"), r2, 0.95);
    }

    let lang_aic = fits["langmuir"]["aic"].as_f64().unwrap_or(f64::MAX);
    let hill_aic = fits["hill"]["aic"].as_f64().unwrap_or(f64::MAX);
    let two_site_aic = fits["two_site"]["aic"].as_f64().unwrap_or(f64::MAX);
    v.check_bool(
        "Langmuir AIC <= Hill AIC + 2",
        lang_aic <= hill_aic + 2.0,
    );
    v.check_bool(
        "Langmuir AIC <= two_site AIC + 2",
        lang_aic <= two_site_aic + 2.0,
    );

    let kd_fit = fits["langmuir"]["params"]["kd"].as_f64().unwrap_or(0.0);
    v.check_abs("Kd recovery (Langmuir)", kd_fit, 28.0, 5.0);
}

fn validate_binding_models_rust(v: &mut ValidationHarness, bm: &serde_json::Value) {
    let fits = &bm["model_fits"];

    let lang_bmax = fits["langmuir"]["params"]["bmax"].as_f64().unwrap_or(0.0);
    let lang_kd = fits["langmuir"]["params"]["kd"].as_f64().unwrap_or(0.0);
    let test_conc = 50.0_f64;
    let rust_langmuir = langmuir(test_conc, lang_bmax, lang_kd);
    let expected_langmuir = lang_bmax * test_conc / (lang_kd + test_conc);
    v.check_abs(
        "Rust Langmuir(50nM) matches formula",
        rust_langmuir,
        expected_langmuir,
        1e-12,
    );

    let hill_bmax = fits["hill"]["params"]["bmax"].as_f64().unwrap_or(0.0);
    let hill_kd = fits["hill"]["params"]["kd"].as_f64().unwrap_or(0.0);
    let hill_n = fits["hill"]["params"]["n"].as_f64().unwrap_or(1.0);
    let rust_hill = hill(test_conc, hill_bmax, hill_kd, hill_n);
    v.check_bool("Rust Hill(50nM) > 0", rust_hill > 0.0);
    v.check_bool("Rust Hill(50nM) < Bmax", rust_hill < hill_bmax * 1.01);

    let twosite_b1 = fits["two_site"]["params"]["b1"].as_f64().unwrap_or(0.0);
    let twosite_kd1 = fits["two_site"]["params"]["k1"].as_f64().unwrap_or(0.0);
    let twosite_b2 = fits["two_site"]["params"]["b2"].as_f64().unwrap_or(0.0);
    let twosite_kd2 = fits["two_site"]["params"]["k2"].as_f64().unwrap_or(0.0);
    let rust_two_site = two_site(test_conc, twosite_b1, twosite_kd1, twosite_b2, twosite_kd2);
    v.check_bool("Rust two_site(50nM) > 0", rust_two_site > 0.0);

    let conc_low = 1.0;
    let conc_high = 500.0;
    v.check_bool(
        "Langmuir monotonically increasing",
        langmuir(conc_high, lang_bmax, lang_kd) > langmuir(conc_low, lang_bmax, lang_kd),
    );
    v.check_bool(
        "Langmuir saturates toward Bmax",
        (langmuir(1000.0, lang_bmax, lang_kd) - lang_bmax).abs() < 0.05,
    );
}

fn validate_glycosylation_shift(v: &mut ValidationHarness, bm: &serde_json::Value) {
    let gs = &bm["glycosylation_shift"];
    let ratio = gs["sensitivity_ratio"].as_f64().unwrap_or(0.0);

    let rust_ratio = 28.0_f64 / 15.0;
    v.check_abs("Rust glycosylation ratio", rust_ratio, ratio, 1e-12);
    v.check_bool("sensitivity ratio in (1.5, 2.5)", ratio > 1.5 && ratio < 2.5);
    v.check_abs(
        "activation improvement %",
        gs["activation_improvement_pct"].as_f64().unwrap_or(0.0),
        (rust_ratio - 1.0) * 100.0,
        1e-10,
    );
}

fn validate_soil_coupling_rust(v: &mut ValidationHarness, bm: &serde_json::Value) {
    let checks = bm["checks"].as_array().unwrap();

    let scenarios: &[(&str, f64, f64)] = &[
        ("dry_cool", 0.12, 15.0),
        ("optimal", 0.25, 25.0),
        ("wet_warm", 0.35, 30.0),
        ("saturated_hot", 0.45, 35.0),
    ];

    for (name, theta, temp) in scenarios {
        let coupling = soil_immune_coupling(*theta, *temp, 1e7);

        let benchmark_val = checks
            .iter()
            .find(|c| {
                c.get("name")
                    .and_then(|n| n.as_str())
                    .is_some_and(|n| n == format!("soil_coupling_{name}"))
            })
            .and_then(|c| c.get("value").and_then(serde_json::Value::as_f64))
            .unwrap_or(f64::NAN);

        v.check_abs(
            &format!("Rust soil coupling {name}"),
            coupling.flagellin_relative,
            benchmark_val,
            1e-10,
        );
    }

    let f_dry = soil_immune_coupling(0.12, 25.0, 1e7).flagellin_relative;
    let f_wet = soil_immune_coupling(0.30, 25.0, 1e7).flagellin_relative;
    v.check_bool("flagellin increases with moisture", f_wet > f_dry);

    let f_cool = soil_immune_coupling(0.25, 15.0, 1e7).flagellin_relative;
    let f_warm = soil_immune_coupling(0.25, 30.0, 1e7).flagellin_relative;
    v.check_bool("flagellin increases with temperature", f_warm > f_cool);

    let extreme = soil_immune_coupling(0.0, -10.0, 1e7);
    v.check_abs("frozen soil → zero moisture factor", extreme.moisture_factor, 0.0, 1e-15);

    let fc = soil_immune_coupling(0.33, 25.0, 1e7);
    v.check_abs("field capacity → moisture factor = 1.0", fc.moisture_factor, 1.0, 1e-15);
    v.check_abs(
        "field capacity @ 25°C → activity = moisture × temp",
        fc.microbial_activity,
        fc.moisture_factor * fc.temp_factor,
        1e-15,
    );

    let ref_temp = soil_immune_coupling(0.25, 25.0, 1e7);
    v.check_abs("reference temp → Q10 factor = 1.0", ref_temp.temp_factor, 1.0, 1e-15);
}

fn main() {
    validation::init_tracing();
    let mut v = ValidationHarness::new("LTEE E3 — FLS2 Plant Immunity");
    validation::banner("LTEE E3 — FLS2 Plant Immunity (Dolgikh et al. 2025)");

    let bm = validation::parse_benchmark(BENCHMARK_JSON);

    validation::section("Benchmark structure");
    validate_benchmark_structure(&mut v, &bm);

    validation::section("Model fits (from Python benchmark)");
    validate_model_fits(&mut v, &bm);

    validation::section("Binding models — Rust parity");
    validate_binding_models_rust(&mut v, &bm);

    validation::section("Glycosylation Kd shift");
    validate_glycosylation_shift(&mut v, &bm);

    validation::section("Soil-immune coupling — Rust parity");
    validate_soil_coupling_rust(&mut v, &bm);

    v.finish();
}
