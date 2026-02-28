// SPDX-License-Identifier: AGPL-3.0-or-later
#![warn(clippy::pedantic)]
#![allow(clippy::cast_precision_loss, clippy::similar_names)]
//! Exp 054: Full-Season Irrigation Water Budget Audit.
//!
//! Runs the complete FAO-56 pipeline for 4 crops over a synthetic growing
//! season: Weather -> ET0 (PM) -> Kc schedule -> Water Balance -> Yield.
//!
//! Validates mass conservation, crop progression, stress response, and
//! end-of-season totals against the Python baseline.
//!
//! Benchmark: `control/season_water_budget/benchmark_season_wb.json`
//! Baseline: `control/season_water_budget/season_water_budget.py` (34/34 PASS)

use airspring_barracuda::eco::evapotranspiration::{self as et, daily_et0, DailyEt0Input};
use airspring_barracuda::eco::water_balance::{
    daily_water_balance_step, readily_available_water, stress_coefficient, total_available_water,
};
use airspring_barracuda::validation::{self, parse_benchmark_json, ValidationHarness};

const BENCHMARK_JSON: &str =
    include_str!("../../../control/season_water_budget/benchmark_season_wb.json");

const LATITUDE_DEG: f64 = 42.7;
const ELEVATION_M: f64 = 256.0;

struct CropDef {
    name: &'static str,
    plant_doy: u32,
    harvest_doy: u32,
    kc_ini: f64,
    kc_mid: f64,
    kc_end: f64,
    ky: f64,
    theta_fc: f64,
    theta_wp: f64,
    root_depth_mm: f64,
}

const CROPS: &[CropDef] = &[
    CropDef {
        name: "corn",
        plant_doy: 120,
        harvest_doy: 260,
        kc_ini: 0.30,
        kc_mid: 1.20,
        kc_end: 0.60,
        ky: 1.25,
        theta_fc: 0.30,
        theta_wp: 0.15,
        root_depth_mm: 1000.0,
    },
    CropDef {
        name: "soybean",
        plant_doy: 135,
        harvest_doy: 270,
        kc_ini: 0.40,
        kc_mid: 1.15,
        kc_end: 0.50,
        ky: 0.85,
        theta_fc: 0.32,
        theta_wp: 0.16,
        root_depth_mm: 800.0,
    },
    CropDef {
        name: "winter_wheat",
        plant_doy: 90,
        harvest_doy: 200,
        kc_ini: 0.40,
        kc_mid: 1.15,
        kc_end: 0.25,
        ky: 1.05,
        theta_fc: 0.28,
        theta_wp: 0.13,
        root_depth_mm: 1200.0,
    },
    CropDef {
        name: "alfalfa",
        plant_doy: 100,
        harvest_doy: 280,
        kc_ini: 0.40,
        kc_mid: 1.20,
        kc_end: 1.15,
        ky: 1.10,
        theta_fc: 0.34,
        theta_wp: 0.17,
        root_depth_mm: 1500.0,
    },
];

fn synthetic_weather(doy: u32) -> (f64, f64, f64, f64, f64, f64) {
    use std::f64::consts::TAU;
    let d = f64::from(doy);
    let tmean = 15.0_f64.mul_add((TAU * (d - 100.0) / 365.0).sin(), 10.0);
    let tmax = tmean + 5.0;
    let tmin = tmean - 5.0;

    let lat_rad = LATITUDE_DEG.to_radians();
    let ra = et::extraterrestrial_radiation(lat_rad, doy);
    let rs = 0.55 * ra;

    let rh = 20.0_f64.mul_add((TAU * (d - 200.0) / 365.0).cos(), 60.0);
    let wind = 0.5_f64.mul_add((TAU * (d - 60.0) / 365.0).sin(), 2.0);
    let precip = 1.5_f64
        .mul_add((TAU * (d - 150.0) / 365.0).sin(), 2.0)
        .max(0.0);

    (tmin, tmax, rh, wind, rs, precip)
}

fn kc_schedule(doy: u32, crop: &CropDef) -> f64 {
    if doy < crop.plant_doy || doy > crop.harvest_doy {
        return 0.0;
    }
    let season = crop.harvest_doy - crop.plant_doy;
    let ini_end = crop.plant_doy + season * 15 / 100;
    let dev_end = crop.plant_doy + season * 40 / 100;
    let mid_end = crop.plant_doy + season * 70 / 100;

    if doy <= ini_end {
        crop.kc_ini
    } else if doy <= dev_end {
        let frac = f64::from(doy - ini_end) / f64::from(dev_end - ini_end);
        (crop.kc_mid - crop.kc_ini).mul_add(frac, crop.kc_ini)
    } else if doy <= mid_end {
        crop.kc_mid
    } else {
        let frac = f64::from(doy - mid_end) / f64::from(crop.harvest_doy - mid_end);
        (crop.kc_end - crop.kc_mid).mul_add(frac, crop.kc_mid)
    }
}

struct SeasonResult {
    total_precip: f64,
    total_irrigation: f64,
    total_et0: f64,
    total_etc: f64,
    total_eta: f64,
    total_dp: f64,
    final_dr: f64,
    taw: f64,
    relative_yield: f64,
}

fn run_season(crop: &CropDef) -> SeasonResult {
    let taw = total_available_water(crop.theta_fc, crop.theta_wp, crop.root_depth_mm);
    let raw = readily_available_water(taw, 0.5);
    let mut dr = 0.0_f64;
    let mut total_precip = 0.0;
    let mut total_et0 = 0.0;
    let mut total_etc = 0.0;
    let mut total_eta = 0.0;
    let mut total_dp = 0.0;
    let mut total_irr = 0.0;

    for doy in crop.plant_doy..=crop.harvest_doy {
        let (tmin, tmax, rh, wind, rs, precip) = synthetic_weather(doy);
        let tmean = f64::midpoint(tmin, tmax);
        let es = et::mean_saturation_vapour_pressure(tmin, tmax);
        let ea = es * rh / 100.0;

        let et0_result = daily_et0(&DailyEt0Input {
            tmin,
            tmax,
            tmean: Some(tmean),
            solar_radiation: rs,
            wind_speed_2m: wind,
            actual_vapour_pressure: ea,
            elevation_m: ELEVATION_M,
            latitude_deg: LATITUDE_DEG,
            day_of_year: doy,
        });
        let et0 = et0_result.et0;
        let kc = kc_schedule(doy, crop);

        let irrigation = if dr > raw * 0.9 { dr } else { 0.0 };
        let ks = stress_coefficient(dr, taw, raw);
        let (new_dr, actual_et, dp) =
            daily_water_balance_step(dr, precip, irrigation, et0, kc, ks, taw);

        dr = new_dr;
        total_precip += precip;
        total_et0 += et0;
        total_etc += et0 * kc;
        total_eta += actual_et;
        total_dp += dp;
        total_irr += irrigation;
    }

    let eta_over_etc = if total_etc > 0.0 {
        total_eta / total_etc
    } else {
        1.0
    };
    let relative_yield = crop.ky.mul_add(-(1.0 - eta_over_etc), 1.0);

    SeasonResult {
        total_precip,
        total_irrigation: total_irr,
        total_et0,
        total_etc,
        total_eta,
        total_dp,
        final_dr: dr,
        taw,
        relative_yield,
    }
}

fn validate_season_results(harness: &mut ValidationHarness) {
    validation::section("Season Water Budget");

    for crop in CROPS {
        let r = run_season(crop);
        let label = crop.name;

        let mass_err =
            (r.total_precip + r.total_irrigation - r.total_eta - r.total_dp + r.final_dr).abs();
        harness.check_abs(&format!("{label}_mass"), mass_err, 0.0, 0.1);
        harness.check_bool(
            &format!("{label}_et0_range"),
            r.total_et0 > 100.0 && r.total_et0 < 1000.0,
        );
        harness.check_bool(
            &format!("{label}_eta_leq_etc"),
            r.total_eta <= r.total_etc + 0.01,
        );
        harness.check_bool(
            &format!("{label}_yield_01"),
            (0.0..=1.0).contains(&r.relative_yield),
        );
        harness.check_bool(&format!("{label}_dr_leq_taw"), r.final_dr <= r.taw + 0.01);
        harness.check_bool(&format!("{label}_dp_nonneg"), r.total_dp >= -0.001);
    }
}

fn validate_benchmark_parity(harness: &mut ValidationHarness, benchmark: &serde_json::Value) {
    validation::section("Python Benchmark Parity");
    let tol = 0.5;

    if let Some(crops) = benchmark.get("crops").and_then(|v| v.as_array()) {
        for (py_crop, crop_def) in crops.iter().zip(CROPS.iter()) {
            let r = run_season(crop_def);
            let label = crop_def.name;

            if let Some(py_et0) = py_crop
                .get("total_et0_mm")
                .and_then(serde_json::Value::as_f64)
            {
                harness.check_abs(&format!("{label}_et0_py"), r.total_et0, py_et0, tol);
            }
            if let Some(py_etc) = py_crop
                .get("total_etc_mm")
                .and_then(serde_json::Value::as_f64)
            {
                harness.check_abs(&format!("{label}_etc_py"), r.total_etc, py_etc, tol);
            }
            if let Some(py_eta) = py_crop
                .get("total_eta_mm")
                .and_then(serde_json::Value::as_f64)
            {
                harness.check_abs(&format!("{label}_eta_py"), r.total_eta, py_eta, tol);
            }
            if let Some(py_precip) = py_crop
                .get("total_precip_mm")
                .and_then(serde_json::Value::as_f64)
            {
                harness.check_abs(
                    &format!("{label}_precip_py"),
                    r.total_precip,
                    py_precip,
                    tol,
                );
            }
        }
    }
}

fn validate_cross_crop(harness: &mut ValidationHarness) {
    validation::section("Cross-Crop Comparisons");

    let corn = run_season(&CROPS[0]);
    let soy = run_season(&CROPS[1]);
    let alfalfa = run_season(&CROPS[3]);

    harness.check_bool("corn_etc_vs_soy", corn.total_etc > soy.total_etc * 0.8);
    harness.check_bool("alfalfa_et0_gt_corn", alfalfa.total_et0 > corn.total_et0);

    for crop_def in CROPS {
        let r = run_season(crop_def);
        harness.check_bool(
            &format!("{}_precip_pos", crop_def.name),
            r.total_precip > 0.0,
        );
        harness.check_bool(&format!("{}_et0_pos", crop_def.name), r.total_et0 > 0.0);
    }
}

fn main() {
    let benchmark = parse_benchmark_json(BENCHMARK_JSON).expect("valid JSON");
    let mut harness = ValidationHarness::new("Exp 054: Full-Season Irrigation Water Budget Audit");
    validate_season_results(&mut harness);
    validate_benchmark_parity(&mut harness, &benchmark);
    validate_cross_crop(&mut harness);
    harness.finish();
}
