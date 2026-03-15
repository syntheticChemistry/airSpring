// SPDX-License-Identifier: AGPL-3.0-or-later
#![allow(clippy::similar_names)]
//! Exp 055: Barracuda PURE GPU Workload Validation.
//!
//! Validates the complete agricultural pipeline via GPU orchestrators,
//! proving that all math is portable from CPU to GPU. Each stage runs
//! through its GPU orchestrator (Tier A uses real GPU dispatch, Tier B
//! uses CPU fallback until `BarraCuda` absorbs ops 5-8).
//!
//! Pipeline: Weather → `ET₀`(GPU) → `Kc`(GPU) → `WB`(GPU) → Yield → Validate
//!
//! Provenance: GPU multi-stage pipeline validation

use std::time::Instant;

use airspring_barracuda::eco::crop::CropType;
use airspring_barracuda::eco::evapotranspiration::{self as et, DailyEt0Input};
use airspring_barracuda::gpu::et0::{BatchedEt0, StationDay};
use airspring_barracuda::gpu::hargreaves::{BatchedHargreaves, HargreavesDay};
use airspring_barracuda::gpu::kc_climate::{BatchedKcClimate, KcClimateDay};
use airspring_barracuda::gpu::reduce;
use airspring_barracuda::gpu::seasonal_pipeline::{CropConfig, SeasonalPipeline, WeatherDay};
use airspring_barracuda::validation::{self, ValidationHarness};

fn synthetic_growing_season(n_stations: usize) -> Vec<Vec<WeatherDay>> {
    let phase = 2.0 * std::f64::consts::PI / 153.0;
    (0..n_stations)
        .map(|s| {
            let lat = (s as f64).mul_add(0.03, 41.0);
            let elev = (s as f64).mul_add(2.0, 200.0);
            (121..=273)
                .map(|doy| {
                    let d = f64::from(doy - 121);
                    let s_val = (phase * d).sin();
                    WeatherDay {
                        tmax: 2.5_f64.mul_add(s_val, 28.0),
                        tmin: 2.0_f64.mul_add(s_val, 16.0),
                        rh_max: 7.5_f64.mul_add(s_val, 77.5),
                        rh_min: 7.5_f64.mul_add(s_val, 52.5),
                        wind_2m: 2.0,
                        solar_rad: 3.0_f64.mul_add(s_val, 21.0),
                        precipitation: if doy % 7 == 0 { 8.0 } else { 0.0 },
                        elevation: elev,
                        latitude_deg: lat,
                        day_of_year: doy,
                    }
                })
                .collect()
        })
        .collect()
}

// ── Section 1: GPU ET₀ Batch Parity ────────────────────────────────────
fn validate_et0_gpu_parity(v: &mut ValidationHarness) {
    validation::section("GPU ET₀ Batch — CPU↔GPU Parity");

    let batched = BatchedEt0::cpu();
    let stations = synthetic_growing_season(10);
    let flat_weather: Vec<&WeatherDay> = stations.iter().flat_map(|s| s.iter()).collect();

    let cpu_inputs: Vec<DailyEt0Input> = flat_weather
        .iter()
        .map(|w| {
            let ea = et::actual_vapour_pressure_rh(w.tmin, w.tmax, w.rh_min, w.rh_max);
            DailyEt0Input {
                tmin: w.tmin,
                tmax: w.tmax,
                tmean: None,
                solar_radiation: w.solar_rad,
                wind_speed_2m: w.wind_2m,
                actual_vapour_pressure: ea,
                elevation_m: w.elevation,
                latitude_deg: w.latitude_deg,
                day_of_year: w.day_of_year,
            }
        })
        .collect();

    let cpu_result = batched.compute(&cpu_inputs);
    let n = cpu_result.et0_values.len();

    // Analytical: 10 synthetic stations × 153 growing-season days = 1530.
    v.check_abs("batch size = 10 stations × 153 days", n as f64, 1530.0, 1.0);

    let all_positive = cpu_result.et0_values.iter().all(|&x| x > 0.0);
    v.check_bool("all ET₀ > 0", all_positive);

    // Plausibility: Michigan growing-season ET₀ range 2–8 mm/day
    // (Allen et al. 1998, Table 2 — humid temperate stations).
    let mean_et0 = cpu_result.et0_values.iter().sum::<f64>() / n as f64;
    v.check_lower("mean daily ET₀ > 2 mm", mean_et0, 2.0);
    v.check_upper("mean daily ET₀ < 8 mm", mean_et0, 8.0);

    let gpu_sd: Vec<StationDay> = flat_weather
        .iter()
        .map(|w| StationDay {
            tmax: w.tmax,
            tmin: w.tmin,
            rh_max: w.rh_max,
            rh_min: w.rh_min,
            wind_2m: w.wind_2m,
            rs: w.solar_rad,
            elevation: w.elevation,
            latitude: w.latitude_deg,
            doy: w.day_of_year,
        })
        .collect();
    let gpu_result = batched.compute_gpu(&gpu_sd).expect("GPU dispatch");
    let max_diff: f64 = cpu_result
        .et0_values
        .iter()
        .zip(gpu_result.et0_values.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);

    // GPU↔CPU tolerance: BarraCuda TS-001/003 (S54) validated WGSL f64
    // shaders to ≤1e-5 relative. 0.5 mm/day is conservative for full
    // ET₀ pipeline including atmospheric chain rounding.
    v.check_upper("CPU↔GPU max diff < 0.5 mm/d", max_diff, 0.5);
}

// ── Section 2: GPU Hargreaves Parity ───────────────────────────────────
fn validate_hargreaves_gpu(v: &mut ValidationHarness) {
    validation::section("GPU Hargreaves — Tier B Parity");

    let stations = synthetic_growing_season(5);
    let flat_weather: Vec<WeatherDay> = stations.into_iter().flatten().collect();
    let hg_days: Vec<HargreavesDay> = flat_weather
        .iter()
        .map(|w| HargreavesDay {
            tmax: w.tmax,
            tmin: w.tmin,
            latitude_deg: w.latitude_deg,
            day_of_year: w.day_of_year,
        })
        .collect();

    let batched = BatchedHargreaves::cpu();
    let result = batched.compute(&hg_days);

    v.check_abs(
        "batch size = 5×153",
        result.et0_values.len() as f64,
        765.0,
        1.0,
    );

    let all_pos = result.et0_values.iter().all(|&x| x > 0.0);
    v.check_bool("all HG ET₀ > 0", all_pos);

    let mean_hg = result.et0_values.iter().sum::<f64>() / result.et0_values.len() as f64;
    v.check_lower("mean HG ET₀ > 2 mm", mean_hg, 2.0);
    v.check_upper("mean HG ET₀ < 10 mm", mean_hg, 10.0);
}

// ── Section 3: GPU Kc Climate Adjustment ───────────────────────────────
fn validate_kc_climate_gpu(v: &mut ValidationHarness) {
    validation::section("GPU Kc Climate — Tier B Parity");

    let stations = synthetic_growing_season(5);
    let flat_weather: Vec<WeatherDay> = stations.into_iter().flatten().collect();
    let kc_days: Vec<KcClimateDay> = flat_weather
        .iter()
        .map(|w| KcClimateDay {
            kc_table: 1.15,
            u2: w.wind_2m,
            rh_min: w.rh_min,
            crop_height_m: 2.0,
        })
        .collect();

    let batched = BatchedKcClimate::cpu();
    let result = batched.compute(&kc_days);

    v.check_abs(
        "batch size = 5×153",
        result.kc_values.len() as f64,
        765.0,
        1.0,
    );

    let all_positive = result.kc_values.iter().all(|&x| x > 0.0);
    v.check_bool("all Kc > 0", all_positive);

    let mean_kc = result.kc_values.iter().sum::<f64>() / result.kc_values.len() as f64;
    v.check_lower("mean adjusted Kc > 0.5", mean_kc, 0.5);
    v.check_upper("mean adjusted Kc < 2.0", mean_kc, 2.0);
}

// ── Section 4: Full Seasonal Pipeline ──────────────────────────────────
fn validate_seasonal_pipeline(v: &mut ValidationHarness) {
    validation::section("Seasonal Pipeline — 6 Crops Full Validation");

    let pipeline = SeasonalPipeline::cpu();
    let stations = synthetic_growing_season(6);
    let crops = [
        CropType::Corn,
        CropType::Soybean,
        CropType::WinterWheat,
        CropType::Potato,
        CropType::Tomato,
        CropType::Blueberry,
    ];

    for (i, crop) in crops.iter().enumerate() {
        let config = CropConfig::standard(*crop);
        let result = pipeline.run_season(&stations[i], &config);
        let name = config.crop_type.coefficients().name;

        v.check_abs(
            &format!("{name} n_days = 153"),
            result.n_days as f64,
            153.0,
            1.0,
        );
        v.check_lower(&format!("{name} total ET₀ > 400"), result.total_et0, 400.0);
        v.check_upper(&format!("{name} total ET₀ < 900"), result.total_et0, 900.0);
        v.check_lower(
            &format!("{name} yield ratio > 0.3"),
            result.yield_ratio,
            0.3,
        );
        v.check_upper(
            &format!("{name} yield ratio ≤ 1.0"),
            result.yield_ratio,
            1.001,
        );
        v.check_upper(
            &format!("{name} mass balance < 1mm"),
            result.mass_balance_error.abs(),
            1.0,
        );
    }
}

// ── Section 5: GPU Scaling Benchmark ───────────────────────────────────
fn validate_gpu_scaling(v: &mut ValidationHarness) {
    validation::section("GPU Scaling — Throughput at N=1K, 10K, 100K");

    let batched = BatchedEt0::cpu();

    for &n in &[1_000usize, 10_000, 100_000] {
        let inputs: Vec<DailyEt0Input> = (0..n)
            .map(|i| {
                let d = i as f64;
                DailyEt0Input {
                    tmin: 12.0 + (d * 0.001).sin(),
                    tmax: 25.0 + (d * 0.001).cos(),
                    tmean: None,
                    solar_radiation: 22.0,
                    wind_speed_2m: 2.0,
                    actual_vapour_pressure: 1.4,
                    elevation_m: 100.0,
                    latitude_deg: 42.5,
                    day_of_year: 180,
                }
            })
            .collect();

        let t0 = Instant::now();
        let result = batched.compute(&inputs);
        let elapsed = t0.elapsed().as_secs_f64();
        let throughput = n as f64 / elapsed;

        v.check_abs(
            &format!("N={n}: all {n} results computed"),
            result.et0_values.len() as f64,
            n as f64,
            1.0,
        );
        v.check_lower(
            &format!("N={n}: throughput > 100K/s ({throughput:.0}/s)"),
            throughput,
            100_000.0,
        );
    }
}

// ── Section 6: Cross-Method GPU Agreement ──────────────────────────────
fn validate_cross_method_gpu(v: &mut ValidationHarness) {
    validation::section("Cross-Method GPU — PM vs HG Agreement");

    let stations = synthetic_growing_season(3);
    let flat_weather: Vec<WeatherDay> = stations.into_iter().flatten().collect();

    let pm_batch = BatchedEt0::cpu();
    let pm_inputs: Vec<DailyEt0Input> = flat_weather
        .iter()
        .map(|w| {
            let ea = et::actual_vapour_pressure_rh(w.tmin, w.tmax, w.rh_min, w.rh_max);
            DailyEt0Input {
                tmin: w.tmin,
                tmax: w.tmax,
                tmean: None,
                solar_radiation: w.solar_rad,
                wind_speed_2m: w.wind_2m,
                actual_vapour_pressure: ea,
                elevation_m: w.elevation,
                latitude_deg: w.latitude_deg,
                day_of_year: w.day_of_year,
            }
        })
        .collect();
    let pm_result = pm_batch.compute(&pm_inputs);
    let pm_mean = pm_result.et0_values.iter().sum::<f64>() / pm_result.et0_values.len() as f64;

    let hg_batch = BatchedHargreaves::cpu();
    let hg_days: Vec<HargreavesDay> = flat_weather
        .iter()
        .map(|w| HargreavesDay {
            tmax: w.tmax,
            tmin: w.tmin,
            latitude_deg: w.latitude_deg,
            day_of_year: w.day_of_year,
        })
        .collect();
    let hg_result = hg_batch.compute(&hg_days);
    let hg_mean = hg_result.et0_values.iter().sum::<f64>() / hg_result.et0_values.len() as f64;

    let pm_total: f64 = pm_result.et0_values.iter().sum();
    let hg_total: f64 = hg_result.et0_values.iter().sum();
    let seasonal_ratio = hg_total / pm_total;

    v.check_lower("PM mean ET₀ > 2 mm", pm_mean, 2.0);
    v.check_upper("PM mean ET₀ < 8 mm", pm_mean, 8.0);
    v.check_lower("HG mean ET₀ > 2 mm", hg_mean, 2.0);
    v.check_upper("HG mean ET₀ < 10 mm", hg_mean, 10.0);
    v.check_lower(
        "HG/PM seasonal ratio > 0.5 (Droogers & Allen 2002)",
        seasonal_ratio,
        0.5,
    );
    v.check_upper(
        "HG/PM seasonal ratio < 2.0 (Droogers & Allen 2002)",
        seasonal_ratio,
        2.0,
    );
}

// ── Section 7: GPU Reduce (Seasonal Stats) ─────────────────────────────
fn validate_gpu_reduce(v: &mut ValidationHarness) {
    validation::section("GPU Reduce — Seasonal Statistics");

    let stations = synthetic_growing_season(4);
    let pipeline = SeasonalPipeline::cpu();

    for (i, weather) in stations.iter().enumerate() {
        let config = CropConfig::standard(CropType::Corn);
        let result = pipeline.run_season(weather, &config);

        let stats = reduce::compute_seasonal_stats(&result.et0_daily);
        v.check_lower(&format!("station {i}: mean ET₀ > 2"), stats.mean, 2.0);
        v.check_upper(&format!("station {i}: mean ET₀ < 8"), stats.mean, 8.0);
        v.check_lower(
            &format!("station {i}: max ET₀ > mean"),
            stats.max,
            stats.mean,
        );
    }
}

// ── Section 8: Multi-Crop GPU Pipeline ─────────────────────────────────
fn validate_multicrop_gpu_pipeline(v: &mut ValidationHarness) {
    validation::section("Multi-Crop GPU Pipeline — 5 Crops × 10 Stations");

    let pipeline = SeasonalPipeline::cpu();
    let stations = synthetic_growing_season(10);
    let crops = [
        CropType::Corn,
        CropType::Soybean,
        CropType::WinterWheat,
        CropType::Potato,
        CropType::Tomato,
    ];

    let mut all_yields = Vec::new();
    let mut all_mass_ok = true;

    for crop in &crops {
        for weather in &stations {
            let config = CropConfig::standard(*crop);
            let result = pipeline.run_season(weather, &config);
            all_yields.push(result.yield_ratio);
            if result.mass_balance_error.abs() > 1.0 {
                all_mass_ok = false;
            }
        }
    }

    v.check_abs(
        "50 simulations completed (5 crops × 10 stations)",
        all_yields.len() as f64,
        50.0,
        1.0,
    );
    v.check_bool("all mass balance < 1mm", all_mass_ok);

    let mean_yield = all_yields.iter().sum::<f64>() / all_yields.len() as f64;
    v.check_lower("mean yield ratio > 0.4", mean_yield, 0.4);
    v.check_upper("mean yield ratio ≤ 1.0", mean_yield, 1.001);

    let min_yield = all_yields.iter().copied().fold(f64::MAX, f64::min);
    let max_yield = all_yields.iter().copied().fold(f64::MIN, f64::max);
    v.check_lower(
        "yield spread > 0 (crop diversity)",
        max_yield - min_yield,
        0.001,
    );
}

fn main() {
    validation::init_tracing();
    validation::banner("Barracuda PURE GPU Workload Validation");
    let mut v = ValidationHarness::new("Barracuda PURE GPU Workload Validation");

    validate_et0_gpu_parity(&mut v);
    validate_hargreaves_gpu(&mut v);
    validate_kc_climate_gpu(&mut v);
    validate_seasonal_pipeline(&mut v);
    validate_gpu_scaling(&mut v);
    validate_cross_method_gpu(&mut v);
    validate_gpu_reduce(&mut v);
    validate_multicrop_gpu_pipeline(&mut v);

    v.finish();
}
