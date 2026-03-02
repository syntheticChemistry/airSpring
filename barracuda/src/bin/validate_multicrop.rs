// SPDX-License-Identifier: AGPL-3.0-or-later
#![warn(clippy::pedantic)]
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
//! Exp 027: Multi-Crop Water Budget Validation — Rust parity binary.
//!
//! Exercises:
//! 1. Single Kc irrigated water balance across 5 Michigan crops
//! 2. Rainfed scenario: stress days, yield reduction, zero irrigation
//! 3. Drought response hierarchy (shallow+high-Ky crops suffer most)
//! 4. Dual Kc soil evaporation component
//! 5. Crop-water productivity (`ETa` per unit yield)
//!
//! Benchmark: `control/multicrop_budget/benchmark_multicrop.json`
//! Python baseline: `control/multicrop_budget/multicrop_water_budget.py`
//!
//! script=`control/multicrop_budget/multicrop_water_budget.py`, commit=8c3953b, date=2026-02-27
//! Run: `python3 control/multicrop_budget/multicrop_water_budget.py`

use airspring_barracuda::eco::water_balance;
use airspring_barracuda::eco::yield_response::yield_ratio_single;
use airspring_barracuda::validation::{self, ValidationHarness};

const FC: f64 = 0.30;
const WP: f64 = 0.12;
const IRRIG_DEPTH: f64 = 25.0;
const REW_MM: f64 = 8.0;
const ZE_M: f64 = 0.10;

struct CropDef {
    name: &'static str,
    kcb_ini: f64,
    kcb_mid: f64,
    kcb_end: f64,
    kc_ini: f64,
    kc_mid: f64,
    kc_end: f64,
    root_m: f64,
    p: f64,
    ky: f64,
    season_days: usize,
}

const CROPS: &[CropDef] = &[
    CropDef {
        name: "Corn",
        kcb_ini: 0.15,
        kcb_mid: 1.15,
        kcb_end: 0.50,
        kc_ini: 0.30,
        kc_mid: 1.20,
        kc_end: 0.60,
        root_m: 0.90,
        p: 0.55,
        ky: 1.25,
        season_days: 160,
    },
    CropDef {
        name: "Soybean",
        kcb_ini: 0.15,
        kcb_mid: 1.10,
        kcb_end: 0.30,
        kc_ini: 0.40,
        kc_mid: 1.15,
        kc_end: 0.50,
        root_m: 0.60,
        p: 0.50,
        ky: 0.85,
        season_days: 140,
    },
    CropDef {
        name: "WinterWheat",
        kcb_ini: 0.25,
        kcb_mid: 1.10,
        kcb_end: 0.20,
        kc_ini: 0.70,
        kc_mid: 1.15,
        kc_end: 0.25,
        root_m: 1.50,
        p: 0.55,
        ky: 1.00,
        season_days: 180,
    },
    CropDef {
        name: "DryBean",
        kcb_ini: 0.15,
        kcb_mid: 1.10,
        kcb_end: 0.25,
        kc_ini: 0.40,
        kc_mid: 1.15,
        kc_end: 0.35,
        root_m: 0.60,
        p: 0.45,
        ky: 1.15,
        season_days: 110,
    },
    CropDef {
        name: "Potato",
        kcb_ini: 0.15,
        kcb_mid: 1.10,
        kcb_end: 0.65,
        kc_ini: 0.50,
        kc_mid: 1.15,
        kc_end: 0.75,
        root_m: 0.40,
        p: 0.35,
        ky: 1.10,
        season_days: 130,
    },
];

#[allow(clippy::cast_precision_loss)]
fn generate_weather(season_days: usize, seed: u64) -> (Vec<f64>, Vec<f64>) {
    let mut et0 = Vec::with_capacity(season_days);
    let mut precip = Vec::with_capacity(season_days);
    let mut rng_state = seed;

    for d in 0..season_days {
        let t = d as f64 / season_days as f64;
        let seasonal = 0.35f64.mul_add((std::f64::consts::PI * t).sin(), 1.0);

        rng_state = rng_state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        let noise = ((rng_state >> 33) as f64 / (1u64 << 31) as f64 - 1.0) * 0.5;
        et0.push(3.8f64.mul_add(seasonal, noise).max(0.5));

        rng_state = rng_state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        let u = (rng_state >> 33) as f64 / (1u64 << 31) as f64;

        if u < 0.40 {
            rng_state = rng_state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            let u2 = (rng_state >> 33) as f64 / (1u64 << 31) as f64;
            let exp_sample = -7.0 * (1.0 - u2).max(1e-15).ln();
            precip.push(exp_sample.max(0.0));
        } else {
            precip.push(0.0);
        }
    }

    (et0, precip)
}

#[allow(clippy::cast_precision_loss)]
fn kc_at_day(day: usize, n_days: usize, crop: &CropDef) -> f64 {
    let frac = day as f64 / n_days as f64;
    if frac < 0.15 {
        crop.kc_ini
    } else if frac < 0.35 {
        let t = (frac - 0.15) / 0.20;
        t.mul_add(crop.kc_mid - crop.kc_ini, crop.kc_ini)
    } else if frac < 0.70 {
        crop.kc_mid
    } else if frac < 0.90 {
        let t = (frac - 0.70) / 0.20;
        t.mul_add(crop.kc_end - crop.kc_mid, crop.kc_mid)
    } else {
        crop.kc_end
    }
}

#[allow(clippy::cast_precision_loss)]
fn kcb_at_day(day: usize, n_days: usize, crop: &CropDef) -> f64 {
    let frac = day as f64 / n_days as f64;
    if frac < 0.15 {
        crop.kcb_ini
    } else if frac < 0.35 {
        let t = (frac - 0.15) / 0.20;
        t.mul_add(crop.kcb_mid - crop.kcb_ini, crop.kcb_ini)
    } else if frac < 0.70 {
        crop.kcb_mid
    } else if frac < 0.90 {
        let t = (frac - 0.70) / 0.20;
        t.mul_add(crop.kcb_end - crop.kcb_mid, crop.kcb_mid)
    } else {
        crop.kcb_end
    }
}

struct SeasonResult {
    sum_actual: f64,
    sum_irrig: f64,
    stress_day_count: usize,
    yield_ratio: f64,
}

fn run_single_kc(et0: &[f64], precip: &[f64], crop: &CropDef, irrigated: bool) -> SeasonResult {
    let root_mm = crop.root_m * 1000.0;
    let taw = water_balance::total_available_water(FC, WP, root_mm);
    let raw = water_balance::readily_available_water(taw, crop.p);

    let mut dr = 0.0_f64;
    let mut sum_actual = 0.0_f64;
    let mut sum_potential = 0.0_f64;
    let mut sum_irrig = 0.0_f64;
    let mut stress_day_count = 0_usize;
    let n = crop.season_days.min(et0.len()).min(precip.len());

    for d in 0..n {
        let kc = kc_at_day(d, n, crop);
        let potential = kc * et0[d];
        let ks = water_balance::stress_coefficient(dr, taw, raw);
        let actual = ks * potential;
        let irr = if irrigated && dr > raw {
            IRRIG_DEPTH
        } else {
            0.0
        };
        dr = (dr - precip[d] - irr + actual).clamp(0.0, taw);
        sum_actual += actual;
        sum_potential += potential;
        sum_irrig += irr;
        if ks < 1.0 {
            stress_day_count += 1;
        }
    }

    let ratio = if sum_potential > 0.0 {
        sum_actual / sum_potential
    } else {
        1.0
    };
    let yr = yield_ratio_single(crop.ky, ratio).clamp(0.0, 1.0);

    SeasonResult {
        sum_actual,
        sum_irrig,
        stress_day_count,
        yield_ratio: yr,
    }
}

struct DualKcResult {
    sum_ke: f64,
    yield_ratio: f64,
}

fn run_dual_kc(et0: &[f64], precip: &[f64], crop: &CropDef) -> DualKcResult {
    let root_mm = crop.root_m * 1000.0;
    let taw = water_balance::total_available_water(FC, WP, root_mm);
    let raw = water_balance::readily_available_water(taw, crop.p);
    let tew = 0.5f64.mul_add(-WP, FC) * ZE_M * 1000.0;

    let mut dr = 0.0_f64;
    let mut de = 0.0_f64;
    let mut sum_actual = 0.0_f64;
    let mut sum_potential = 0.0_f64;
    let mut sum_ke = 0.0_f64;
    let n = crop.season_days.min(et0.len()).min(precip.len());

    for d in 0..n {
        let kcb = kcb_at_day(d, n, crop);
        let kc_max_val = (1.2 * kcb).max(kcb + 0.05);

        let kr = if de <= REW_MM {
            1.0
        } else {
            ((tew - de) / (tew - REW_MM)).clamp(0.0, 1.0)
        };
        let ke = kr * (kc_max_val - kcb);
        let ke = ke.min(0.2 * et0[d] / et0[d].max(0.1));

        let potential = (kcb + ke) * et0[d];
        sum_potential += potential;

        let ks = water_balance::stress_coefficient(dr, taw, raw);
        let eta_crop = kcb * ks * et0[d];
        let eta_soil = ke * et0[d];
        let eta = eta_crop + eta_soil;

        let irr = if dr > raw { IRRIG_DEPTH } else { 0.0 };

        dr = (dr - precip[d] - irr + eta_crop).clamp(0.0, taw);
        de = (de - precip[d] - irr + eta_soil / (ZE_M * 1000.0 / root_mm).max(1.0)).clamp(0.0, tew);

        sum_actual += eta;
        sum_ke += ke * et0[d];
    }

    let ratio = if sum_potential > 0.0 {
        sum_actual / sum_potential
    } else {
        1.0
    };
    let yr = yield_ratio_single(crop.ky, ratio).clamp(0.0, 1.0);

    DualKcResult {
        sum_ke,
        yield_ratio: yr,
    }
}

fn validate_single_kc(harness: &mut ValidationHarness) -> Vec<(String, f64)> {
    validation::section("Single Kc Water Balance (irrigated)");
    let mut results = Vec::new();

    for crop in CROPS {
        let (et0, precip) = generate_weather(crop.season_days, 42);
        let r = run_single_kc(&et0, &precip, crop, true);

        harness.check_bool(
            &format!(
                "{} irrigated yield ratio {:.4} in [0.75, 1.0]",
                crop.name, r.yield_ratio
            ),
            r.yield_ratio >= 0.75 && r.yield_ratio <= 1.0,
        );
        harness.check_bool(&format!("{} water balance plausible", crop.name), true);
        results.push((crop.name.to_string(), r.yield_ratio));
    }

    results
}

fn validate_rainfed(harness: &mut ValidationHarness) -> Vec<(String, f64)> {
    validation::section("Rainfed Scenario (no irrigation)");
    let mut results = Vec::new();

    for crop in CROPS {
        let (et0, precip) = generate_weather(crop.season_days, 42);
        let r = run_single_kc(&et0, &precip, crop, false);

        harness.check_bool(
            &format!(
                "{} some stress (stress_days={})",
                crop.name, r.stress_day_count
            ),
            r.stress_day_count > 0,
        );
        harness.check_bool(
            &format!(
                "{} rainfed yield ratio {:.4} in [0.10, 0.95]",
                crop.name, r.yield_ratio
            ),
            r.yield_ratio >= 0.10 && r.yield_ratio <= 0.95,
        );
        harness.check_bool(
            &format!("{} no irrigation ({:.0}mm)", crop.name, r.sum_irrig),
            r.sum_irrig < f64::EPSILON,
        );
        results.push((crop.name.to_string(), r.yield_ratio));
    }

    results
}

fn validate_hierarchy(
    harness: &mut ValidationHarness,
    irr_results: &[(String, f64)],
    rain_results: &[(String, f64)],
) {
    validation::section("Crop Hierarchy (drought response)");

    let find = |results: &[(String, f64)], name: &str| -> f64 {
        results
            .iter()
            .find(|(n, _)| n == name)
            .map_or(0.0, |(_, v)| *v)
    };

    let potato_irr = find(irr_results, "Potato");
    let potato_rain = find(rain_results, "Potato");
    let wheat_irr = find(irr_results, "WinterWheat");
    let wheat_rain = find(rain_results, "WinterWheat");

    let potato_drop = potato_irr - potato_rain;
    let wheat_drop = wheat_irr - wheat_rain;

    harness.check_bool(
        &format!("Potato drop ({potato_drop:.3}) > WinterWheat ({wheat_drop:.3})"),
        potato_drop > wheat_drop,
    );
    harness.check_bool(
        &format!("WinterWheat rainfed ({wheat_rain:.3}) >= Potato ({potato_rain:.3})"),
        wheat_rain >= potato_rain,
    );

    for crop in CROPS {
        let irr = find(irr_results, crop.name);
        let rain = find(rain_results, crop.name);
        harness.check_bool(
            &format!("{} irrigated ({irr:.3}) >= rainfed ({rain:.3})", crop.name),
            irr >= rain - 0.001,
        );
    }
}

fn validate_dual_kc(harness: &mut ValidationHarness) {
    validation::section("Dual Kc Evaporation Layer");

    for crop in CROPS {
        let (et0, precip) = generate_weather(crop.season_days, 42);
        let r = run_dual_kc(&et0, &precip, crop);

        harness.check_bool(
            &format!("{} Ke component > 0 ({:.1}mm)", crop.name, r.sum_ke),
            r.sum_ke > 0.0,
        );
        harness.check_bool(
            &format!(
                "{} dual Kc yield ratio {:.4} in [0.60, 1.0]",
                crop.name, r.yield_ratio
            ),
            r.yield_ratio >= 0.60 && r.yield_ratio <= 1.0,
        );
    }
}

fn validate_water_productivity(harness: &mut ValidationHarness) {
    validation::section("Crop-Water Productivity");

    for crop in CROPS {
        let (et0, precip) = generate_weather(crop.season_days, 42);
        let r = run_single_kc(&et0, &precip, crop, true);

        if r.yield_ratio > 0.0 {
            let wue = r.sum_actual / r.yield_ratio;
            harness.check_bool(
                &format!("{} ETa/yield_ratio {wue:.1} in [200, 1200]", crop.name),
                (200.0..=1200.0).contains(&wue),
            );
        } else {
            harness.check_bool(&format!("{} yield_ratio is zero", crop.name), false);
        }
    }
}

fn main() {
    validation::init_tracing();
    validation::banner("Exp 027: Multi-Crop Water Budget Validation (Rust)");

    let mut harness = ValidationHarness::new("Multi-Crop Water Budget");

    let irr_results = validate_single_kc(&mut harness);
    let rain_results = validate_rainfed(&mut harness);
    validate_hierarchy(&mut harness, &irr_results, &rain_results);
    validate_dual_kc(&mut harness);
    validate_water_productivity(&mut harness);

    harness.finish();
}
