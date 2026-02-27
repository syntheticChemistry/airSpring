// SPDX-License-Identifier: AGPL-3.0-or-later
#![warn(clippy::pedantic)]
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
//! Exp 024: NASS Yield Validation — Stewart (1977) pipeline consistency.
//!
//! Validates:
//! 1. Ky table values match FAO-56 Table 24
//! 2. Drought response is monotonically decreasing with severity
//! 3. Soil type sensitivity (sandy < loam < clay under drought)
//! 4. Multi-year variability produces realistic distributions
//! 5. Crop ranking under drought (soybean tolerant, corn sensitive)
//! 6. Mass balance conservation (`ETa` ≤ `ETc`)
//!
//! Benchmark: `control/nass_yield/benchmark_nass_yield.json`
//! Python baseline: `control/nass_yield/nass_yield_validation.py`

use airspring_barracuda::eco::water_balance;
use airspring_barracuda::eco::yield_response::{clamp_yield_ratio, ky_table, yield_ratio_single};
use airspring_barracuda::validation::{self, parse_benchmark_json, ValidationHarness};

const BENCHMARK_JSON: &str = include_str!("../../../control/nass_yield/benchmark_nass_yield.json");

struct CropParams {
    name: &'static str,
    ky_total: f64,
    kc_mid: f64,
    season_days: usize,
    root_m: f64,
    p: f64,
}

const CROPS: &[CropParams] = &[
    CropParams {
        name: "corn",
        ky_total: 1.25,
        kc_mid: 1.20,
        season_days: 150,
        root_m: 1.0,
        p: 0.55,
    },
    CropParams {
        name: "soybean",
        ky_total: 0.85,
        kc_mid: 1.15,
        season_days: 135,
        root_m: 0.8,
        p: 0.50,
    },
    CropParams {
        name: "winter_wheat",
        ky_total: 1.00,
        kc_mid: 1.15,
        season_days: 180,
        root_m: 1.2,
        p: 0.55,
    },
    CropParams {
        name: "alfalfa",
        ky_total: 1.10,
        kc_mid: 1.20,
        season_days: 200,
        root_m: 1.5,
        p: 0.55,
    },
    CropParams {
        name: "dry_bean",
        ky_total: 1.15,
        kc_mid: 1.15,
        season_days: 100,
        root_m: 0.6,
        p: 0.45,
    },
];

struct SoilParams {
    theta_fc: f64,
    theta_wp: f64,
}

const SANDY_LOAM: SoilParams = SoilParams {
    theta_fc: 0.18,
    theta_wp: 0.08,
};
const LOAM: SoilParams = SoilParams {
    theta_fc: 0.28,
    theta_wp: 0.14,
};
const CLAY_LOAM: SoilParams = SoilParams {
    theta_fc: 0.36,
    theta_wp: 0.22,
};

/// Deterministic Michigan-like growing season weather.
///
/// Uses the same algorithm as the Python baseline but in Rust-native math.
/// Reproduces the `NumPy` `default_rng` via a simplified linear-feedback approach
/// that matches the *physical character* of the season (ET₀ pattern, rain
/// frequency) without requiring exact RNG parity. Validation is on physical
/// consistency (monotonicity, ranking, conservation) not exact values.
fn michigan_season(seed: u64, season_days: usize, dry_fraction: f64) -> (Vec<f64>, Vec<f64>) {
    let mut et0 = Vec::with_capacity(season_days);
    let mut precip = Vec::with_capacity(season_days);

    let mean_et0: f64 = 3.8;
    let mean_precip: f64 = 3.0;
    let drought_factor = 0.7f64.mul_add(-dry_fraction, 1.0);
    let rain_prob = 0.42 * drought_factor;

    let mut rng_state = seed;

    for d in 0..season_days {
        let t = d as f64 / season_days as f64;
        let seasonal = 0.4f64.mul_add((std::f64::consts::PI * t).sin(), 1.0);

        rng_state = rng_state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        let noise = ((rng_state >> 33) as f64 / (1u64 << 31) as f64 - 1.0) * 0.6;

        et0.push(mean_et0.mul_add(seasonal, noise).max(0.5));

        rng_state = rng_state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        let u = (rng_state >> 33) as f64 / (1u64 << 31) as f64;

        if u < rain_prob {
            rng_state = rng_state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            let u2 = (rng_state >> 33) as f64 / (1u64 << 31) as f64;
            let exp_sample = -(mean_precip / 0.42) * (1.0 - u2).max(1e-15).ln();
            precip.push(exp_sample.max(0.0));
        } else {
            precip.push(0.0);
        }
    }

    (et0, precip)
}

struct WbConfig<'a> {
    et0: &'a [f64],
    precip: &'a [f64],
    kc: f64,
    theta_fc: f64,
    theta_wp: f64,
    root_m: f64,
    p: f64,
    season_days: usize,
}

/// Returns `(eta_etc_ratio, cumulative_eta, cumulative_etc, stress_day_count)`.
fn water_balance_season(cfg: &WbConfig<'_>) -> (f64, f64, f64, usize) {
    let root_mm = cfg.root_m * 1000.0;
    let taw = water_balance::total_available_water(cfg.theta_fc, cfg.theta_wp, root_mm);
    let raw = water_balance::readily_available_water(taw, cfg.p);

    let mut dr = 0.0_f64;
    let mut sum_actual = 0.0_f64;
    let mut sum_potential = 0.0_f64;
    let mut stress_day_count = 0_usize;

    let n = cfg.season_days.min(cfg.et0.len()).min(cfg.precip.len());
    for d in 0..n {
        let ks = water_balance::stress_coefficient(dr, taw, raw);
        if ks < 1.0 {
            stress_day_count += 1;
        }
        let potential = cfg.kc * cfg.et0[d];
        let actual = ks * potential;
        dr = (dr - cfg.precip[d] + actual).clamp(0.0, taw);
        sum_actual += actual;
        sum_potential += potential;
    }

    let ratio = if sum_potential > 0.0 {
        sum_actual / sum_potential
    } else {
        1.0
    };
    (ratio, sum_actual, sum_potential, stress_day_count)
}

fn validate_ky_consistency(v: &mut ValidationHarness, benchmark: &serde_json::Value) {
    validation::section("Ky Table Consistency (FAO-56 Table 24)");

    for crop in CROPS {
        let expected = benchmark["ky_table"][crop.name]["ky_total"]
            .as_f64()
            .unwrap_or_else(|| panic!("missing ky_total for {}", crop.name));

        let ky = ky_table(crop.name).unwrap_or_else(|| panic!("crop not in table: {}", crop.name));
        v.check_abs(
            &format!("{} Ky total", crop.name),
            ky.ky_total,
            expected,
            0.01,
        );

        if let Some(ref stages) = ky.stages {
            let (ky_veg, _) = stages[0];
            let (ky_flower, _) = stages[1];
            v.check_bool(
                &format!("{} Ky_flower >= Ky_veg", crop.name),
                ky_flower >= ky_veg,
            );
        }
    }
}

fn validate_drought_response(v: &mut ValidationHarness) {
    validation::section("Drought Response Monotonicity");

    for crop in &CROPS[..3] {
        let mut prev_yr = 2.0_f64;
        for (label, dry_frac) in &[
            ("normal", 0.0),
            ("mild", 0.3),
            ("moderate", 0.6),
            ("severe", 0.9),
        ] {
            let (et0, precip) = michigan_season(42, crop.season_days, *dry_frac);
            let (ratio, _, _, _) = water_balance_season(&WbConfig {
                et0: &et0,
                precip: &precip,
                kc: crop.kc_mid,
                theta_fc: LOAM.theta_fc,
                theta_wp: LOAM.theta_wp,
                root_m: crop.root_m,
                p: crop.p,
                season_days: crop.season_days,
            });
            let yr = clamp_yield_ratio(yield_ratio_single(crop.ky_total, ratio));

            v.check_bool(
                &format!("{} {} yield {yr:.3} <= prev {prev_yr:.3}", crop.name, label),
                yr <= prev_yr + 0.001,
            );
            prev_yr = yr;
        }
    }
}

fn validate_soil_sensitivity(v: &mut ValidationHarness) {
    validation::section("Soil Type Sensitivity");

    for crop in &CROPS[..2] {
        let (et0, precip) = michigan_season(100, crop.season_days, 0.5);

        let mut yields = Vec::new();
        for (soil_name, soil) in &[
            ("sandy_loam", &SANDY_LOAM),
            ("loam", &LOAM),
            ("clay_loam", &CLAY_LOAM),
        ] {
            let (ratio, _, _, _) = water_balance_season(&WbConfig {
                et0: &et0,
                precip: &precip,
                kc: crop.kc_mid,
                theta_fc: soil.theta_fc,
                theta_wp: soil.theta_wp,
                root_m: crop.root_m,
                p: crop.p,
                season_days: crop.season_days,
            });
            yields.push((
                *soil_name,
                clamp_yield_ratio(yield_ratio_single(crop.ky_total, ratio)),
            ));
        }

        v.check_bool(
            &format!(
                "{}: loam ({:.3}) >= sandy ({:.3})",
                crop.name, yields[1].1, yields[0].1
            ),
            yields[1].1 >= yields[0].1 - 0.001,
        );
        v.check_bool(
            &format!(
                "{}: clay_loam ({:.3}) >= loam ({:.3})",
                crop.name, yields[2].1, yields[1].1
            ),
            yields[2].1 >= yields[1].1 - 0.001,
        );
    }
}

fn validate_multi_year_variability(v: &mut ValidationHarness, benchmark: &serde_json::Value) {
    validation::section("Multi-Year Variability (20 years, corn on loam)");

    let corn = &CROPS[0];
    let mut yield_ratios = Vec::new();
    let mut stress_list = Vec::new();

    for yr_seed in 2000..2020_u64 {
        let dry_frac = 0.15 * (yr_seed as f64 * 0.7).sin().abs();
        let (et0, precip) = michigan_season(yr_seed, corn.season_days, dry_frac);
        let (ratio, _, _, sd) = water_balance_season(&WbConfig {
            et0: &et0,
            precip: &precip,
            kc: corn.kc_mid,
            theta_fc: LOAM.theta_fc,
            theta_wp: LOAM.theta_wp,
            root_m: corn.root_m,
            p: corn.p,
            season_days: corn.season_days,
        });
        yield_ratios.push(clamp_yield_ratio(yield_ratio_single(corn.ky_total, ratio)));
        stress_list.push(sd as f64);
    }

    let mean_yr: f64 = yield_ratios.iter().sum::<f64>() / yield_ratios.len() as f64;
    let mean_sq: f64 = yield_ratios
        .iter()
        .map(|y| (y - mean_yr).powi(2))
        .sum::<f64>()
        / yield_ratios.len() as f64;
    let std_yr = mean_sq.sqrt();
    let cv = if mean_yr > 0.0 { std_yr / mean_yr } else { 0.0 };
    let mean_stress: f64 = stress_list.iter().sum::<f64>() / stress_list.len() as f64;

    let my = &benchmark["multi_year"];
    let yr_range = my["mean_yr_range"].as_array().expect("array");
    let cv_range = my["cv_range"].as_array().expect("array");
    let stress_range = my["mean_stress_range"].as_array().expect("array");

    v.check_bool(
        &format!(
            "mean yield ratio {mean_yr:.4} in [{}, {}]",
            yr_range[0].as_f64().unwrap(),
            yr_range[1].as_f64().unwrap()
        ),
        mean_yr >= yr_range[0].as_f64().unwrap() && mean_yr <= yr_range[1].as_f64().unwrap(),
    );
    v.check_bool(
        &format!(
            "yield CV {cv:.4} in [{}, {}]",
            cv_range[0].as_f64().unwrap(),
            cv_range[1].as_f64().unwrap()
        ),
        cv >= cv_range[0].as_f64().unwrap() && cv <= cv_range[1].as_f64().unwrap(),
    );
    v.check_bool(
        &format!(
            "mean stress days {mean_stress:.1} in [{}, {}]",
            stress_range[0].as_f64().unwrap(),
            stress_range[1].as_f64().unwrap()
        ),
        mean_stress >= stress_range[0].as_f64().unwrap()
            && mean_stress <= stress_range[1].as_f64().unwrap(),
    );
    v.check_bool(
        "some years > 0.55 (better years exist)",
        yield_ratios.iter().any(|&y| y > 0.55),
    );
    v.check_bool(
        "some years < 0.45 (stress years exist)",
        yield_ratios.iter().any(|&y| y < 0.45),
    );
}

fn validate_crop_ranking(v: &mut ValidationHarness) {
    validation::section("Crop Ranking Under Drought");

    let (et0, precip) = michigan_season(77, 180, 0.6);
    let mut yields = Vec::new();

    for crop in CROPS {
        let n = crop.season_days.min(180);
        let (ratio, _, _, _) = water_balance_season(&WbConfig {
            et0: &et0[..n],
            precip: &precip[..n],
            kc: crop.kc_mid,
            theta_fc: LOAM.theta_fc,
            theta_wp: LOAM.theta_wp,
            root_m: crop.root_m,
            p: crop.p,
            season_days: n,
        });
        let yr = clamp_yield_ratio(yield_ratio_single(crop.ky_total, ratio));
        yields.push((crop.name, yr));
    }

    let soy_yr = yields.iter().find(|y| y.0 == "soybean").unwrap().1;
    let corn_yr = yields.iter().find(|y| y.0 == "corn").unwrap().1;
    v.check_bool(
        &format!("soybean ({soy_yr:.3}) > corn ({corn_yr:.3}) under drought"),
        soy_yr > corn_yr,
    );

    for (name, yr) in &yields {
        v.check_bool(
            &format!("{name} yield ratio {yr:.4} in [0, 1]"),
            *yr >= 0.0 && *yr <= 1.0,
        );
    }
}

fn validate_mass_balance(v: &mut ValidationHarness) {
    validation::section("Mass Balance Conservation");

    for crop in &CROPS[..2] {
        let (et0, precip) = michigan_season(42, crop.season_days, 0.0);
        let (ratio, eta, etc, _) = water_balance_season(&WbConfig {
            et0: &et0,
            precip: &precip,
            kc: crop.kc_mid,
            theta_fc: LOAM.theta_fc,
            theta_wp: LOAM.theta_wp,
            root_m: crop.root_m,
            p: crop.p,
            season_days: crop.season_days,
        });

        v.check_bool(
            &format!("{}: ETa ({eta:.1}) <= ETc ({etc:.1})", crop.name),
            eta <= etc + 0.01,
        );
        v.check_bool(
            &format!("{}: ETa/ETc ratio ({ratio:.4}) in [0, 1]", crop.name),
            (0.0..=1.001).contains(&ratio),
        );
    }
}

fn main() {
    validation::init_tracing();
    validation::banner("Exp 024: NASS Yield Validation (Stewart 1977 + Michigan Pipeline)");
    let mut v = ValidationHarness::new("NASS Yield Validation");
    let benchmark =
        parse_benchmark_json(BENCHMARK_JSON).expect("benchmark_nass_yield.json must parse");

    validate_ky_consistency(&mut v, &benchmark);
    validate_drought_response(&mut v);
    validate_soil_sensitivity(&mut v);
    validate_multi_year_variability(&mut v, &benchmark);
    validate_crop_ranking(&mut v);
    validate_mass_balance(&mut v);

    v.finish();
}
