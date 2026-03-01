// SPDX-License-Identifier: AGPL-3.0-or-later
#![allow(clippy::too_many_lines)]
#![warn(clippy::pedantic)]
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::similar_names,
    clippy::option_if_let_else
)]
//! Exp 058: Climate Scenario Water Demand Analysis.
//!
//! Assesses how Michigan crop water demand changes under warming scenarios.
//! Uses the validated FAO-56 chain with synthetic CMIP6-like temperature
//! offsets applied to a baseline Michigan growing season.
//!
//! Scenarios: Baseline (0°C), SSP2-4.5 (+1.5°C), SSP3-7.0 (+2.5°C), SSP5-8.5 (+4.0°C)
//! Crops: Corn, Soybean, Winter Wheat
//! Soil: Loam (FC=0.28, WP=0.14)
//! Season: 153-day Michigan growing season (May 1 – Sep 30)
//!
//! Reference: CMIP6 AR6 WG1 Ch4

use std::f64::consts::PI;

use airspring_barracuda::eco::crop::CropType;
use airspring_barracuda::gpu::seasonal_pipeline::{CropConfig, SeasonalPipeline, WeatherDay};
use airspring_barracuda::validation::{self, parse_benchmark_json, ValidationHarness};

const BENCHMARK_JSON: &str =
    include_str!("../../../control/climate_scenario/benchmark_climate_scenario.json");

// Michigan growing season: May 1 (doy 121) to Sep 30 (doy 273) = 153 days
const SEASON_START_DOY: u32 = 121;
const SEASON_DAYS: usize = 153;
const LATITUDE_DEG: f64 = 42.5;
const ELEVATION_M: f64 = 250.0;

/// Precip pattern from Python: `np.random.default_rng(42)`; (rng.random(153) < 0.4) * 3.0
const PRECIP_MM: &[f64; 153] = &[
    0., 0., 0., 0., 3., 0., 0., 0., 3., 0., 3., 0., 0., 0., 0., 3., 0., 3., 0., 0., 0., 3., 0., 0.,
    0., 3., 0., 3., 3., 0., 0., 0., 3., 3., 0., 3., 3., 0., 3., 0., 0., 0., 0., 3., 0., 0., 3., 3.,
    0., 3., 3., 3., 0., 0., 0., 0., 0., 0., 3., 3., 0., 0., 0., 0., 0., 0., 0., 3., 3., 0., 3., 0.,
    0., 3., 3., 3., 3., 0., 0., 0., 0., 0., 0., 3., 3., 3., 0., 0., 3., 0., 3., 0., 0., 3., 3., 0.,
    3., 3., 3., 0., 0., 0., 3., 0., 0., 0., 0., 3., 3., 0., 0., 3., 3., 0., 3., 0., 0., 0., 0., 0.,
    0., 0., 3., 0., 3., 0., 3., 3., 3., 0., 3., 0., 0., 3., 0., 3., 0., 0., 0., 3., 0., 0., 0., 0.,
    0., 3., 3., 0., 0., 3., 0., 0., 3.,
];

#[allow(clippy::too_many_lines)]
fn main() {
    validation::init_tracing();
    validation::banner("Exp 058: Climate Scenario Water Demand Analysis");
    println!(
        "Assesses Michigan crop water demand under CMIP6-like warming.\n\
         FAO-56 + Stewart + SeasonalPipeline::cpu()\n"
    );

    let benchmark = parse_benchmark_json(BENCHMARK_JSON).expect("parse benchmark");
    let theta_fc = benchmark["soil"]["field_capacity"].as_f64().unwrap_or(0.28);
    let theta_wp = benchmark["soil"]["wilting_point"].as_f64().unwrap_or(0.14);
    let scenarios = benchmark["scenarios"].as_array().expect("scenarios array");
    let et0_lo = benchmark["et0_pct_increase_per_degC"]["lo"]
        .as_f64()
        .unwrap_or(2.0);
    let et0_hi = benchmark["et0_pct_increase_per_degC"]["hi"]
        .as_f64()
        .unwrap_or(8.0);
    let mb_tol = benchmark["tolerance_mass_balance_mm"]
        .as_f64()
        .unwrap_or(1.0);

    let mut v = ValidationHarness::new("Climate Scenario");

    let pipeline = SeasonalPipeline::cpu();
    let crop_types = [CropType::Corn, CropType::Soybean, CropType::WinterWheat];
    let crop_names = ["corn", "soybean", "winter_wheat"];

    let mut results: std::collections::HashMap<String, std::collections::HashMap<String, _>> =
        std::collections::HashMap::new();

    for scen in scenarios {
        let name = scen["name"].as_str().unwrap_or("?");
        let delta_t = scen["delta_t"].as_f64().unwrap_or(0.0);

        let weather = generate_michigan_weather(delta_t);
        results.insert(name.to_string(), std::collections::HashMap::new());

        for (ct, &crop_name) in crop_types.iter().zip(crop_names.iter()) {
            let mut config = CropConfig::standard(*ct);
            config.field_capacity = theta_fc;
            config.wilting_point = theta_wp;
            config.irrigation_depth_mm = 0.0; // rainfed

            let result = pipeline.run_season(&weather, &config);
            results.get_mut(name).unwrap().insert(
                crop_name.to_string(),
                (
                    result.total_et0,
                    result.total_actual_et,
                    result.stress_days,
                    result.yield_ratio,
                    result.mass_balance_error,
                ),
            );
        }
    }

    // 1. ET₀ increases monotonically with temperature offset
    validation::section("ET₀ Monotonicity with Warming");
    for &crop_name in &crop_names {
        let et0_vals: Vec<f64> = scenarios
            .iter()
            .map(|s| results[s["name"].as_str().unwrap()][crop_name].0)
            .collect();
        let monotonic = et0_vals.windows(2).all(|w| w[0] <= w[1]);
        v.check_bool(
            &format!("{crop_name}: ET₀ increases with delta_T"),
            monotonic,
        );
    }

    // 2. Water demand (ETc) increases with warming — ET0 increases → ETc increases
    validation::section("Water Demand (ETc) Increases with Warming");
    for &crop_name in &crop_names {
        let etc_vals: Vec<f64> = scenarios
            .iter()
            .map(|s| results[s["name"].as_str().unwrap()][crop_name].1)
            .collect();
        let monotonic = etc_vals.windows(2).all(|w| w[0] <= w[1]);
        v.check_bool(
            &format!("{crop_name}: ETc increases with delta_T"),
            monotonic,
        );
    }

    // 3. Stress days increase with warming
    validation::section("Stress Days Increase with Warming");
    for &crop_name in &crop_names {
        let sd_vals: Vec<usize> = scenarios
            .iter()
            .map(|s| results[s["name"].as_str().unwrap()][crop_name].2)
            .collect();
        let monotonic = sd_vals.windows(2).all(|w| w[0] <= w[1]);
        v.check_bool(
            &format!("{crop_name}: stress_days increase with delta_T"),
            monotonic,
        );
    }

    // 4. Yield ratio decreases with warming
    validation::section("Yield Ratio Decreases with Warming");
    for &crop_name in &crop_names {
        let yr_vals: Vec<f64> = scenarios
            .iter()
            .map(|s| results[s["name"].as_str().unwrap()][crop_name].3)
            .collect();
        let monotonic = yr_vals.windows(2).all(|w| w[0] >= w[1]);
        v.check_bool(
            &format!("{crop_name}: yield_ratio decreases with delta_T"),
            monotonic,
        );
    }

    // 5. Corn more sensitive than soybean
    validation::section("Corn More Sensitive Than Soybean");
    let baseline_corn = results["baseline"]["corn"].3;
    let baseline_soy = results["baseline"]["soybean"].3;
    let ssp585_corn = results["ssp585_2050"]["corn"].3;
    let ssp585_soy = results["ssp585_2050"]["soybean"].3;
    let corn_drop = baseline_corn - ssp585_corn;
    let soy_drop = baseline_soy - ssp585_soy;
    v.check_bool(
        "corn yield drop > soybean under SSP5-8.5",
        corn_drop > soy_drop,
    );

    // 6. All yield ratios in [0, 1]
    validation::section("Yield Ratios in [0, 1]");
    for scen in scenarios {
        let name = scen["name"].as_str().unwrap();
        for &crop_name in &crop_names {
            let yr = results[name][crop_name].3;
            v.check_bool(
                &format!("{name} {crop_name} yield_ratio in [0,1]"),
                (0.0..=1.0).contains(&yr),
            );
        }
    }

    // 7. Mass balance conservation
    validation::section("Mass Balance Conservation");
    for scen in scenarios {
        let name = scen["name"].as_str().unwrap();
        for &crop_name in &crop_names {
            let mb = results[name][crop_name].4;
            v.check_abs(&format!("{name} {crop_name} mass balance"), mb, 0.0, mb_tol);
        }
    }

    // 8. ET₀ per-degree increase in plausible range
    validation::section("ET₀ Per-Degree Increase (FAO-56 Literature)");
    let baseline_et0: f64 = crop_names
        .iter()
        .map(|c| results["baseline"][*c].0)
        .sum::<f64>()
        / 3.0;
    let ssp245_et0: f64 = crop_names
        .iter()
        .map(|c| results["ssp245_2050"][*c].0)
        .sum::<f64>()
        / 3.0;
    let pct_per_deg = (ssp245_et0 / baseline_et0 - 1.0) / 1.5 * 100.0;
    v.check_bool(
        &format!("ET₀ % increase per °C in [{et0_lo},{et0_hi}]"),
        pct_per_deg >= et0_lo && pct_per_deg <= et0_hi,
    );

    // 9. Cross-crop ET₀ identical (same weather)
    validation::section("Cross-Crop ET₀ Identical (Same Weather)");
    for scen in scenarios {
        let name = scen["name"].as_str().unwrap();
        let et0_corn = results[name]["corn"].0;
        let et0_soy = results[name]["soybean"].0;
        let et0_wheat = results[name]["winter_wheat"].0;
        v.check_bool(
            &format!("{name}: corn ET₀ ≈ soybean"),
            (et0_corn - et0_soy).abs() < 0.1,
        );
        v.check_bool(
            &format!("{name}: corn ET₀ ≈ wheat"),
            (et0_corn - et0_wheat).abs() < 0.1,
        );
    }

    v.finish();
}

fn generate_michigan_weather(delta_t: f64) -> Vec<WeatherDay> {
    let mut weather = Vec::with_capacity(SEASON_DAYS);
    for (i, &precip) in PRECIP_MM.iter().enumerate() {
        let doy = SEASON_START_DOY + i as u32;
        let doy_frac = i as f64 / (SEASON_DAYS - 1) as f64;

        let tmax_base = 7.0f64.mul_add((PI * doy_frac).sin(), 25.0);
        let tmin_base = tmax_base - 10.0;
        let tmax = tmax_base + delta_t;
        let tmin = tmin_base + delta_t;

        let solar_base = 18.0;
        let solar = solar_base * 0.15f64.mul_add((PI * doy_frac).sin(), 0.85);

        weather.push(WeatherDay {
            tmax,
            tmin,
            rh_max: 65.0,
            rh_min: 65.0,
            wind_2m: 2.0,
            solar_rad: solar,
            precipitation: precip,
            elevation: ELEVATION_M,
            latitude_deg: LATITUDE_DEG,
            day_of_year: doy,
        });
    }
    weather
}
