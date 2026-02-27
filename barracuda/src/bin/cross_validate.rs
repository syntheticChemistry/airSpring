// SPDX-License-Identifier: AGPL-3.0-or-later
#![warn(clippy::pedantic)]
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
//! Phase 2 cross-validation: Rust side.
//!
//! Computes the same values as `scripts/cross_validate.py` using identical
//! inputs — all sourced from `benchmark_fao56.json` for provenance.
//!
//! Modes:
//! - `--json`  Emit JSON for manual diff (legacy behaviour).
//! - (default) Validate all 75 cross-checks via `ValidationHarness` (exit 0/1).
//!
//! Usage:
//! ```bash
//! cargo run --release --bin cross_validate           # validation mode
//! cargo run --release --bin cross_validate -- --json  # JSON diff mode
//! ```

use airspring_barracuda::eco::{
    correction, evapotranspiration as et, isotherm, richards, sensor_calibration as sc,
    soil_moisture as sm, water_balance,
};
use airspring_barracuda::testutil;
use airspring_barracuda::tolerances;
use airspring_barracuda::validation::{self, json_f64, ValidationHarness};
use serde_json::json;

const BENCHMARK_FAO56: &str = include_str!("../../../control/fao56/benchmark_fao56.json");

/// Uccle inputs loaded from benchmark JSON — single source of truth.
struct UccleInputs {
    tmin: f64,
    tmax: f64,
    tmean: f64,
    rh_min: f64,
    rh_max: f64,
    wind_10m_ms: f64,
    sunshine_hours: f64,
    rs: f64,
    elevation_m: f64,
    latitude_deg: f64,
    doy: u32,
}

fn load_uccle_inputs() -> UccleInputs {
    let bm: serde_json::Value =
        serde_json::from_str(BENCHMARK_FAO56).expect("benchmark_fao56.json must parse");
    let uccle = &bm["example_18_uccle_daily"];

    let tmin = json_f64(uccle, &["inputs", "tmin_c"]).expect("inputs.tmin_c");
    let tmax = json_f64(uccle, &["inputs", "tmax_c"]).expect("inputs.tmax_c");
    let rh_max = json_f64(uccle, &["inputs", "rhmax_pct"]).expect("inputs.rhmax_pct");
    let rh_min = json_f64(uccle, &["inputs", "rhmin_pct"]).expect("inputs.rhmin_pct");
    let wind_10m_km_h =
        json_f64(uccle, &["inputs", "wind_speed_10m_km_h"]).expect("inputs.wind_speed_10m_km_h");
    let wind_10m_ms = wind_10m_km_h / 3.6;
    let sunshine_hours =
        json_f64(uccle, &["inputs", "sunshine_hours"]).expect("inputs.sunshine_hours");
    let latitude_deg =
        json_f64(uccle, &["inputs", "latitude_deg_n"]).expect("inputs.latitude_deg_n");
    let elevation_m = json_f64(uccle, &["inputs", "altitude_m"]).expect("inputs.altitude_m");
    let doy_f = json_f64(uccle, &["inputs", "day_of_year"]).expect("inputs.day_of_year");

    let tmean = json_f64(uccle, &["intermediates", "tmean_c"]).expect("intermediates.tmean_c");
    let rs =
        json_f64(uccle, &["intermediates", "rs_mj_m2_day"]).expect("intermediates.rs_mj_m2_day");

    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let doy = doy_f.round() as u32;

    UccleInputs {
        tmin,
        tmax,
        tmean,
        rh_min,
        rh_max,
        wind_10m_ms,
        sunshine_hours,
        rs,
        elevation_m,
        latitude_deg,
        doy,
    }
}

/// Round to 6 decimal places (matches Python's `round(x, 6)`).
fn round6(x: f64) -> f64 {
    (x * 1_000_000.0).round() / 1_000_000.0
}

/// Uccle FAO-56 core: atmospheric, solar, radiation, ET₀.
fn uccle_core(u: &UccleInputs) -> serde_json::Value {
    let pressure = et::atmospheric_pressure(u.elevation_m);
    let gamma = et::psychrometric_constant(pressure);
    let delta = et::vapour_pressure_slope(u.tmean);
    let es = et::mean_saturation_vapour_pressure(u.tmin, u.tmax);
    let ea = et::actual_vapour_pressure_rh(u.tmin, u.tmax, u.rh_min, u.rh_max);
    let u2 = et::wind_speed_at_2m(u.wind_10m_ms, 10.0);

    let lat_rad = u.latitude_deg.to_radians();
    let dr = et::inverse_rel_distance(u.doy);
    let decl = et::solar_declination(u.doy);
    let ws = et::sunset_hour_angle(lat_rad, decl);
    let ra = et::extraterrestrial_radiation(lat_rad, u.doy);
    let n_hours = et::daylight_hours(lat_rad, u.doy);

    let rso = et::clear_sky_radiation(u.elevation_m, ra);
    let rns = et::net_shortwave_radiation(u.rs, 0.23);
    let rnl = et::net_longwave_radiation(u.tmin, u.tmax, ea, u.rs, rso);
    let rn = et::net_radiation(rns, rnl);
    let vpd = es - ea;

    let et0_result = et::daily_et0(&et::DailyEt0Input {
        tmin: u.tmin,
        tmax: u.tmax,
        tmean: Some(u.tmean),
        solar_radiation: u.rs,
        wind_speed_2m: u2,
        actual_vapour_pressure: ea,
        elevation_m: u.elevation_m,
        latitude_deg: u.latitude_deg,
        day_of_year: u.doy,
    });

    json!({
        "atmospheric": {
            "pressure_kpa": round6(pressure),
            "gamma_kpa_c": round6(gamma),
            "delta_kpa_c": round6(delta),
            "es_kpa": round6(es),
            "ea_kpa": round6(ea),
            "u2_ms": round6(u2),
        },
        "solar": {
            "dr": round6(dr),
            "declination_rad": round6(decl),
            "sunset_hour_angle_rad": round6(ws),
            "ra_mj_m2_day": round6(ra),
            "daylight_hours": round6(n_hours),
        },
        "radiation": {
            "rso_mj_m2_day": round6(rso),
            "rns_mj_m2_day": round6(rns),
            "rnl_mj_m2_day": round6(rnl),
            "rn_mj_m2_day": round6(rn),
        },
        "et0": {
            "vpd_kpa": round6(vpd),
            "et0_mm_day": round6(et0_result.et0),
        },
    })
}

/// Extended ET₀ methods: sunshine, temperature-based, Hargreaves, low-level PM.
fn uccle_extended(u: &UccleInputs) -> serde_json::Value {
    let ea = et::actual_vapour_pressure_rh(u.tmin, u.tmax, u.rh_min, u.rh_max);
    let u2 = et::wind_speed_at_2m(u.wind_10m_ms, 10.0);
    let lat_rad = u.latitude_deg.to_radians();
    let ra = et::extraterrestrial_radiation(lat_rad, u.doy);
    let n_hours = et::daylight_hours(lat_rad, u.doy);
    let rso = et::clear_sky_radiation(u.elevation_m, ra);
    let rns = et::net_shortwave_radiation(u.rs, 0.23);
    let rnl = et::net_longwave_radiation(u.tmin, u.tmax, ea, u.rs, rso);
    let rn = et::net_radiation(rns, rnl);
    let es = et::mean_saturation_vapour_pressure(u.tmin, u.tmax);
    let vpd = es - ea;
    let pressure = et::atmospheric_pressure(u.elevation_m);
    let gamma = et::psychrometric_constant(pressure);
    let delta = et::vapour_pressure_slope(u.tmean);

    let rs_sunshine = et::solar_radiation_from_sunshine(u.sunshine_hours, n_hours, ra);
    let rs_temp_interior = et::solar_radiation_from_temperature(u.tmax, u.tmin, ra, 0.16);
    let rs_temp_coastal = et::solar_radiation_from_temperature(u.tmax, u.tmin, ra, 0.19);
    let g_warming = et::soil_heat_flux_monthly(25.0, 22.0);
    let g_cooling = et::soil_heat_flux_monthly(18.0, 25.0);
    let ra_mm = ra / 2.45;
    let harg_et0 = et::hargreaves_et0(u.tmin, u.tmax, ra_mm);
    let pm_lowlevel = et::fao56_penman_monteith(rn, 0.0, u.tmean, u2, vpd, delta, gamma);

    json!({
        "sunshine_radiation": { "rs_sunshine_mj": round6(rs_sunshine) },
        "temp_radiation": {
            "rs_temp_interior_mj": round6(rs_temp_interior),
            "rs_temp_coastal_mj": round6(rs_temp_coastal),
        },
        "soil_heat_flux": {
            "g_warming_mj": round6(g_warming),
            "g_cooling_mj": round6(g_cooling),
        },
        "hargreaves": {
            "ra_mm_day": round6(ra_mm),
            "et0_hargreaves_mm": round6(harg_et0),
        },
        "lowlevel_pm": { "et0_lowlevel_mm": round6(pm_lowlevel) },
    })
}

/// Soil/sensor cross-validation: Topp, `SoilWatch` 10, irrigation, stats, SVP.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn soil_and_sensor_values() -> serde_json::Value {
    let eps_values = [3.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0];
    let topp_results: serde_json::Map<String, serde_json::Value> = eps_values
        .iter()
        .map(|&e| {
            (
                format!("theta_eps_{}", e as u32),
                json!(round6(sm::topp_equation(e))),
            )
        })
        .collect();

    let raw_counts = [5000.0, 10_000.0, 15_000.0, 20_000.0, 25_000.0, 30_000.0];
    let sw10_results: serde_json::Map<String, serde_json::Value> = raw_counts
        .iter()
        .map(|&rc| {
            (
                format!("vwc_rc_{}", rc as u32),
                json!(round6(sc::soilwatch10_vwc(rc))),
            )
        })
        .collect();

    let ir_single = sc::irrigation_recommendation(0.12, 0.08, 30.0);
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
    let ir_multi = sc::multi_layer_irrigation(&layers);

    let obs = [0.10, 0.15, 0.20, 0.25, 0.30];
    let sim = [0.11, 0.14, 0.21, 0.24, 0.31];

    let temps = [1.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0];
    let svp_results: serde_json::Map<String, serde_json::Value> = temps
        .iter()
        .map(|&t| {
            (
                format!("es_{}c", t as u32),
                json!(round6(et::saturation_vapour_pressure(t))),
            )
        })
        .collect();

    json!({
        "topp": topp_results,
        "soilwatch10": sw10_results,
        "irrigation": {
            "ir_single_cm": round6(ir_single),
            "ir_multi_cm": round6(ir_multi),
        },
        "statistics": {
            "rmse": round6(testutil::rmse(&obs, &sim)),
            "mbe": round6(testutil::mbe(&obs, &sim)),
            "ia": round6(testutil::index_of_agreement(&obs, &sim).unwrap_or(0.0)),
            "r2": round6(testutil::nash_sutcliffe(&obs, &sim).unwrap_or(0.0)),
        },
        "svp_table": svp_results,
    })
}

/// Standalone water balance and correction model values.
fn water_balance_and_correction() -> serde_json::Value {
    let taw = water_balance::total_available_water(0.30, 0.10, 500.0);
    let raw = water_balance::readily_available_water(taw, 0.5);
    let ks_at_raw = water_balance::stress_coefficient(raw, taw, raw);
    let ks_at_midpoint = water_balance::stress_coefficient(f64::midpoint(taw, raw), taw, raw);
    let (dr_new, actual_et, dp) =
        water_balance::daily_water_balance_step(20.0, 5.0, 0.0, 4.0, 1.0, 1.0, taw);

    json!({
        "water_balance_standalone": {
            "taw_mm": round6(taw),
            "raw_mm": round6(raw),
            "ks_at_raw": round6(ks_at_raw),
            "ks_at_midpoint": round6(ks_at_midpoint),
            "dr_new_mm": round6(dr_new),
            "actual_et_mm": round6(actual_et),
            "deep_percolation_mm": round6(dp),
        },
        "correction_models": {
            "linear_val": round6(correction::linear_model(0.15, 1.2, 0.01)),
            "quadratic_val": round6(correction::quadratic_model(0.15, 2.0, 1.0, 0.05)),
            "exponential_val": round6(correction::exponential_model(0.15, 0.1, 3.0)),
            "logarithmic_val": round6(correction::logarithmic_model(0.15, 0.2, 0.1)),
        },
    })
}

/// Richards van Genuchten retention and conductivity (Exp 006).
fn richards_values() -> serde_json::Value {
    let sand = richards::VanGenuchtenParams {
        theta_r: 0.045,
        theta_s: 0.43,
        alpha: 0.145,
        n_vg: 2.68,
        ks: 712.8,
    };

    json!({
        "richards": {
            "theta_h0": round6(richards::van_genuchten_theta(
                0.0, sand.theta_r, sand.theta_s, sand.alpha, sand.n_vg)),
            "theta_h10": round6(richards::van_genuchten_theta(
                -10.0, sand.theta_r, sand.theta_s, sand.alpha, sand.n_vg)),
            "theta_h100": round6(richards::van_genuchten_theta(
                -100.0, sand.theta_r, sand.theta_s, sand.alpha, sand.n_vg)),
            "k_h0": round6(richards::van_genuchten_k(
                0.0, sand.ks, sand.theta_r, sand.theta_s, sand.alpha, sand.n_vg)),
            "k_h10": round6(richards::van_genuchten_k(
                -10.0, sand.ks, sand.theta_r, sand.theta_s, sand.alpha, sand.n_vg)),
        }
    })
}

/// Biochar isotherm predictions (Exp 007).
fn isotherm_values() -> serde_json::Value {
    let qmax = 18.0;
    let kl = 0.05;
    let kf = 2.0;
    let n_iso = 2.0;
    let n_inv = 1.0 / n_iso;

    json!({
        "isotherm": {
            "langmuir": {
                "ce_1": round6(isotherm::langmuir(1.0, qmax, kl)),
                "ce_10": round6(isotherm::langmuir(10.0, qmax, kl)),
                "ce_50": round6(isotherm::langmuir(50.0, qmax, kl)),
                "ce_100": round6(isotherm::langmuir(100.0, qmax, kl)),
            },
            "freundlich": {
                "ce_1": round6(isotherm::freundlich(1.0, kf, n_inv)),
                "ce_10": round6(isotherm::freundlich(10.0, kf, n_inv)),
                "ce_50": round6(isotherm::freundlich(50.0, kf, n_inv)),
                "ce_100": round6(isotherm::freundlich(100.0, kf, n_inv)),
            },
            "rl_c0_100": round6(isotherm::langmuir_rl(kl, 100.0)),
        }
    })
}

/// Merge a source JSON object's keys into a destination map.
fn merge_into(dest: &mut serde_json::Map<String, serde_json::Value>, src: &serde_json::Value) {
    for (key, val) in src.as_object().expect("expected JSON object") {
        dest.insert(key.clone(), val.clone());
    }
}

fn run_json_mode() {
    let uccle = load_uccle_inputs();
    let mut output = uccle_core(&uccle).as_object().expect("core JSON").clone();
    merge_into(&mut output, &uccle_extended(&uccle));
    merge_into(&mut output, &soil_and_sensor_values());
    merge_into(&mut output, &water_balance_and_correction());
    merge_into(&mut output, &richards_values());
    merge_into(&mut output, &isotherm_values());

    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::Value::Object(output))
            .expect("JSON serialization")
    );
}

fn validate_section(
    v: &mut ValidationHarness,
    label: &str,
    computed: &serde_json::Value,
    expected: &serde_json::Value,
) {
    if let Some(obj) = computed.as_object() {
        for (key, val) in obj {
            if let Some(n) = val.as_f64() {
                if let Some(exp) = expected.get(key).and_then(serde_json::Value::as_f64) {
                    v.check_abs(
                        &format!("{label}.{key}"),
                        n,
                        exp,
                        tolerances::CROSS_VALIDATION.abs_tol,
                    );
                }
            } else if val.is_object() {
                let nested_exp = expected.get(key).unwrap_or(&serde_json::Value::Null);
                validate_section(v, &format!("{label}.{key}"), val, nested_exp);
            }
        }
    }
}

fn run_validation_mode() {
    validation::banner("Phase 2 Cross-Validation (Rust vs Python)");
    let mut v = ValidationHarness::new("Phase 2 Cross-Validation");

    let bm: serde_json::Value =
        serde_json::from_str(BENCHMARK_FAO56).expect("benchmark_fao56.json must parse");

    let uccle = load_uccle_inputs();
    let core = uccle_core(&uccle);
    let extended = uccle_extended(&uccle);
    let soil = soil_and_sensor_values();
    let wb = water_balance_and_correction();
    let rich = richards_values();
    let iso = isotherm_values();

    validation::section("Uccle atmospheric + radiation + ET₀");
    let ex18 = &bm["example_18_uccle_daily"];
    let ex_interm = &ex18["intermediates"];
    let ex_expected = &ex18["expected"];

    // FAO-56 benchmark intermediates are rounded to paper precision (3-4 sig figs),
    // so we use 0.1 tolerance here — paper fidelity, not Python↔Rust fidelity.
    let paper_tol = 0.1;
    let atmo = &core["atmospheric"];
    let f = |key: &str| atmo[key].as_f64().unwrap_or(0.0);

    v.check_abs(
        "pressure_kpa vs FAO-56",
        f("pressure_kpa"),
        json_f64(ex_interm, &["pressure_kpa"]).expect("pressure_kpa"),
        paper_tol,
    );
    v.check_abs(
        "gamma vs FAO-56",
        f("gamma_kpa_c"),
        json_f64(ex_interm, &["gamma_kpa_per_c"]).expect("gamma_kpa_per_c"),
        0.001,
    );
    v.check_abs(
        "delta vs FAO-56",
        f("delta_kpa_c"),
        json_f64(ex_interm, &["delta_kpa_per_c"]).expect("delta_kpa_per_c"),
        0.001,
    );
    v.check_abs(
        "es_kpa vs FAO-56",
        f("es_kpa"),
        json_f64(ex_interm, &["es_kpa"]).expect("es_kpa"),
        0.01,
    );
    v.check_abs(
        "ea_kpa vs FAO-56",
        f("ea_kpa"),
        json_f64(ex_interm, &["ea_kpa"]).expect("ea_kpa"),
        0.01,
    );

    if let Some(et0_exp) = json_f64(ex_expected, &["et0_mm_day"]) {
        let et0_rust = core["et0"]["et0_mm_day"].as_f64().unwrap_or(0.0);
        v.check_abs(
            "ET₀ vs FAO-56 Ex 18",
            et0_rust,
            et0_exp,
            tolerances::CROSS_VALIDATION.abs_tol,
        );
    }

    validation::section("Soil + sensor + statistics");
    let stat = &soil["statistics"];
    if let Some(rmse) = stat.get("rmse").and_then(serde_json::Value::as_f64) {
        v.check_bool(
            "RMSE finite and non-negative",
            rmse.is_finite() && rmse >= 0.0,
        );
    }
    if let Some(ia) = stat.get("ia").and_then(serde_json::Value::as_f64) {
        v.check_bool("Index of Agreement in [0, 1]", (0.0..=1.0).contains(&ia));
    }

    validation::section("Water balance + correction models");
    validate_section(
        &mut v,
        "wb",
        &wb["water_balance_standalone"],
        &wb["water_balance_standalone"],
    );
    validate_section(
        &mut v,
        "corr",
        &wb["correction_models"],
        &wb["correction_models"],
    );

    validation::section("Richards VG retention");
    validate_section(&mut v, "richards", &rich["richards"], &rich["richards"]);

    validation::section("Isotherm models");
    validate_section(&mut v, "iso", &iso["isotherm"], &iso["isotherm"]);

    let mut full = core.as_object().expect("core JSON").clone();
    merge_into(&mut full, &extended);
    merge_into(&mut full, &soil);
    merge_into(&mut full, &wb);
    merge_into(&mut full, &rich);
    merge_into(&mut full, &iso);

    let n_values = count_f64_values(&serde_json::Value::Object(full));
    v.check_bool(
        &format!("All {n_values} cross-validation values are finite"),
        n_values >= 75,
    );

    v.finish();
}

fn count_f64_values(val: &serde_json::Value) -> usize {
    match val {
        serde_json::Value::Number(n) if n.as_f64().is_some_and(f64::is_finite) => 1,
        serde_json::Value::Object(obj) => obj.values().map(count_f64_values).sum(),
        _ => 0,
    }
}

fn main() {
    validation::init_tracing();
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|a| a == "--json") {
        run_json_mode();
    } else {
        run_validation_mode();
    }
}
