// SPDX-License-Identifier: AGPL-3.0-or-later
#![allow(clippy::doc_markdown)] // paths in doc comments
//! Experiment 015: 60-year water balance validation.
//!
//! Validates that the Rust water balance (Exp 004) runs at multi-decade scale
//! against the Python baseline (`control/long_term_wb/long_term_water_balance.py`).
//!
//! Loads cached weather from `control/long_term_wb/data/wooster_era5_1960_2023.json`
//! and benchmark from `control/long_term_wb/benchmark_long_term_wb.json`.
//!
//! If the weather cache does not exist, skips with a message and exits 0.
//!
//! script=`control/long_term_wb/long_term_water_balance.py`, commit=5684b1e, date=2026-02-26
//! Run: `python3 control/long_term_wb/long_term_water_balance.py`

use airspring_barracuda::eco::{
    crop::CropType,
    evapotranspiration::{extraterrestrial_radiation, hargreaves_et0},
    water_balance::{self as wb, DailyInput, WaterBalanceState},
};
use airspring_barracuda::tolerances;
use airspring_barracuda::validation::{self, ValidationHarness, json_f64, parse_benchmark_json};
use std::path::Path;

const BENCHMARK_JSON: &str =
    include_str!("../../../control/long_term_wb/benchmark_long_term_wb.json");

/// Growing season: May 1 - Sep 30 (inclusive).
const SEASON_START_MONTH: u32 = 5;
const SEASON_START_DAY: u32 = 1;
const SEASON_END_MONTH: u32 = 9;
const SEASON_END_DAY: u32 = 30;

/// Minimum days in a full season (skip truncated).
const MIN_SEASON_DAYS: usize = 100;

/// One day of weather from the cached JSON.
struct WeatherDay {
    year: u32,
    month: u32,
    day: u32,
    tmax_c: f64,
    tmin_c: f64,
    precip_mm: f64,
    et0_openmeteo: f64,
}

fn date_to_doy(year: u32, month: u32, day: u32) -> u32 {
    let is_leap = year.is_multiple_of(4) && (!year.is_multiple_of(100) || year.is_multiple_of(400));
    let days_before: [u32; 12] = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334];
    let mut doy = days_before[(month - 1).min(11) as usize] + day;
    if is_leap && month > 2 {
        doy += 1;
    }
    doy
}

const fn in_season(month: u32, day: u32) -> bool {
    (month > SEASON_START_MONTH || (month == SEASON_START_MONTH && day >= SEASON_START_DAY))
        && (month < SEASON_END_MONTH || (month == SEASON_END_MONTH && day <= SEASON_END_DAY))
}

/// Parse cached weather JSON into seasons (year -> `Vec<WeatherDay>`).
fn parse_weather_cache(
    json: &serde_json::Value,
) -> Option<std::collections::BTreeMap<u32, Vec<WeatherDay>>> {
    let times = json.get("time")?.as_array()?;
    let tmax = json.get("temperature_2m_max")?.as_array()?;
    let tmin = json.get("temperature_2m_min")?.as_array()?;
    let precip = json.get("precipitation_sum")?.as_array()?;
    let et0_om = json
        .get("et0_fao_evapotranspiration")
        .and_then(|a| a.as_array());

    let n = times.len();
    if tmax.len() != n || tmin.len() != n || precip.len() != n {
        return None;
    }

    let get_f64 = |arr: &[serde_json::Value], idx: usize| -> f64 {
        arr.get(idx)
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(f64::NAN)
    };

    let mut seasons: std::collections::BTreeMap<u32, Vec<WeatherDay>> =
        std::collections::BTreeMap::new();

    for (i, time_val) in times.iter().enumerate().take(n) {
        let date_str = time_val.as_str()?;
        let parts: Vec<&str> = date_str.split('-').collect();
        if parts.len() != 3 {
            continue;
        }
        let year: u32 = parts[0].parse().ok()?;
        let month: u32 = parts[1].parse().ok()?;
        let day: u32 = parts[2].parse().ok()?;

        if !in_season(month, day) {
            continue;
        }
        if !(1960..=2023).contains(&year) {
            continue;
        }

        let tmax_c = get_f64(tmax, i);
        let tmin_c = get_f64(tmin, i);
        let precip_mm = get_f64(precip, i);
        let et0_openmeteo = et0_om.map_or(f64::NAN, |a| get_f64(a, i));

        seasons.entry(year).or_default().push(WeatherDay {
            year,
            month,
            day,
            tmax_c: tmax_c.clamp(-50.0, 60.0),
            tmin_c: tmin_c.clamp(-50.0, 60.0),
            precip_mm: precip_mm.max(0.0),
            et0_openmeteo,
        });
    }

    // Sort each season by date
    for days in seasons.values_mut() {
        days.sort_by(|a, b| (a.month, a.day).cmp(&(b.month, b.day)));
    }

    Some(seasons)
}

/// Site parameters for water balance (from benchmark).
struct SiteParams {
    lat_deg: f64,
    fc: f64,
    wp: f64,
    root_depth_mm: f64,
    p: f64,
    kc: f64,
    irrig_depth_mm: f64,
}

/// Result for one growing season.
struct SeasonResult {
    year: u32,
    total_et0_rust: f64,
    total_et0_om: f64,
    total_precip: f64,
    total_et: f64,
    /// Total deep percolation (mm); used to derive has_dp.
    _total_dp: f64,
    /// Total irrigation applied (mm); used for mass balance / audit.
    _total_irrig: f64,
    irrig_events: usize,
    mb_error: f64,
    /// Days with water stress; used to derive has_stress.
    _stress_days: usize,
    has_stress: bool,
    has_dp: bool,
}

fn run_season(days: &[WeatherDay], params: &SiteParams) -> SeasonResult {
    let lat_rad = params.lat_deg.to_radians();

    let et0_rust: Vec<f64> = days
        .iter()
        .map(|d| {
            let doy = date_to_doy(d.year, d.month, d.day);
            let ra_mj = extraterrestrial_radiation(lat_rad, doy);
            let ra_mm_day = ra_mj / 2.45;
            hargreaves_et0(d.tmin_c, d.tmax_c, ra_mm_day)
        })
        .collect();

    let precip: Vec<f64> = days.iter().map(|d| d.precip_mm).collect();
    let et0_om: Vec<f64> = days
        .iter()
        .map(|d| {
            if d.et0_openmeteo.is_finite() {
                d.et0_openmeteo
            } else {
                0.0
            }
        })
        .collect();

    let initial = WaterBalanceState::new(params.fc, params.wp, params.root_depth_mm, params.p);
    let mut state = initial.clone();
    let mut inputs = Vec::with_capacity(days.len());
    let mut outputs = Vec::with_capacity(days.len());

    for (&et0, &precip_day) in et0_rust.iter().zip(precip.iter()) {
        let irr = if state.depletion > state.raw {
            state.depletion.min(params.irrig_depth_mm)
        } else {
            0.0
        };

        let input = DailyInput {
            precipitation: precip_day,
            irrigation: irr,
            et0,
            kc: params.kc,
        };
        let output = state.step(&input);
        inputs.push(input);
        outputs.push(output);
    }

    let total_et0_rust: f64 = et0_rust.iter().sum();
    let total_et0_om: f64 = et0_om.iter().sum();
    let total_precip: f64 = precip.iter().sum();
    let total_et: f64 = outputs.iter().map(|o| o.actual_et).sum();
    let total_dp: f64 = outputs.iter().map(|o| o.deep_percolation).sum();
    let total_irrig: f64 = inputs.iter().map(|i| i.irrigation).sum();
    let irrig_events = inputs.iter().filter(|i| i.irrigation > 0.0).count();
    let stress_days = outputs.iter().filter(|o| o.ks < 1.0).count();
    let has_stress = stress_days > 0;
    let has_dp = total_dp > 0.0;

    let mb_error = wb::mass_balance_check(&inputs, &outputs, initial.depletion, state.depletion);

    SeasonResult {
        year: days[0].year,
        total_et0_rust,
        total_et0_om,
        total_precip,
        total_et,
        _total_dp: total_dp,
        _total_irrig: total_irrig,
        irrig_events,
        mb_error,
        _stress_days: stress_days,
        has_stress,
        has_dp,
    }
}

fn validate_physical_checks(
    v: &mut ValidationHarness,
    results: &[SeasonResult],
    benchmark: &serde_json::Value,
) {
    validation::section("Physical reasonableness");

    let n = results.len();
    if n == 0 {
        v.check_bool("No season data", false);
        return;
    }

    let mean_et0: f64 = results.iter().map(|r| r.total_et0_rust).sum::<f64>() / n as f64;
    let max_mb: f64 = results.iter().map(|r| r.mb_error).fold(0.0_f64, f64::max);
    let stress_seasons = results.iter().filter(|r| r.has_stress).count();
    let stress_pct = 100.0 * stress_seasons as f64 / n as f64;
    let seasons_with_dp = results.iter().filter(|r| r.has_dp).count();

    let pr = benchmark
        .get("validation_checks")
        .and_then(|c| c.get("physical_reasonableness"))
        .and_then(|p| p.get("checks"))
        .and_then(|a| a.as_array());

    if let Some(checks) = pr {
        for c in checks {
            let cid = c.get("id").and_then(|v| v.as_str()).unwrap_or("");
            match cid {
                "annual_et0_range" => {
                    let min_v = c
                        .get("min")
                        .and_then(serde_json::Value::as_f64)
                        .unwrap_or(0.0);
                    let max_v = c
                        .get("max")
                        .and_then(serde_json::Value::as_f64)
                        .unwrap_or(1000.0);
                    let ok = (min_v..=max_v).contains(&mean_et0);
                    v.check_bool(&format!("Annual ET₀ 400-800 mm: {mean_et0:.0} mm"), ok);
                }
                "seasonal_et0_mean" => {
                    let min_v = c
                        .get("min")
                        .and_then(serde_json::Value::as_f64)
                        .unwrap_or(0.0);
                    let max_v = c
                        .get("max")
                        .and_then(serde_json::Value::as_f64)
                        .unwrap_or(1000.0);
                    let ok = (min_v..=max_v).contains(&mean_et0);
                    v.check_bool(
                        &format!("Growing season ET₀ 500-700 mm: {mean_et0:.0} mm"),
                        ok,
                    );
                }
                "mass_balance" => {
                    let tol = c
                        .get("tolerance")
                        .and_then(serde_json::Value::as_f64)
                        .unwrap_or(0.01);
                    v.check_bool(
                        &format!("Mass balance < {tol} mm: max error {max_mb:.6} mm"),
                        max_mb <= tol,
                    );
                }
                "stress_fraction" => {
                    let min_pct = c
                        .get("min_pct")
                        .and_then(serde_json::Value::as_f64)
                        .unwrap_or(0.0);
                    let max_pct = c
                        .get("max_pct")
                        .and_then(serde_json::Value::as_f64)
                        .unwrap_or(100.0);
                    let ok = (min_pct..=max_pct).contains(&stress_pct);
                    v.check_bool(
                        &format!("Drought stress in 50-100% of seasons: {stress_pct:.1}%"),
                        ok,
                    );
                }
                "deep_percolation" => {
                    let min_seasons = c
                        .get("min_seasons_with_dp")
                        .and_then(serde_json::Value::as_i64)
                        .and_then(|v| usize::try_from(v.max(0)).ok())
                        .unwrap_or(0);
                    v.check_bool(
                        &format!("Deep percolation in wet years: {seasons_with_dp} seasons"),
                        seasons_with_dp >= min_seasons,
                    );
                }
                _ => {}
            }
        }
    }
}

fn cv_percent(data: &[f64]) -> f64 {
    let m = barracuda::stats::mean(data);
    if m <= 0.0 || data.len() < 2 {
        return 0.0;
    }
    let sd = barracuda::stats::correlation::std_dev(data).unwrap_or(0.0);
    100.0 * sd / m
}

fn compute_trend_stats(results: &[SeasonResult]) -> (f64, f64, f64) {
    let et0_arr: Vec<f64> = results.iter().map(|r| r.total_et0_rust).collect();
    let precip_arr: Vec<f64> = results.iter().map(|r| r.total_precip).collect();
    let years: Vec<f64> = results.iter().map(|r| f64::from(r.year)).collect();

    let mean_y = barracuda::stats::mean(&et0_arr);
    let mean_x = barracuda::stats::mean(&years);
    let mut num = 0.0_f64;
    let mut den = 0.0_f64;
    for (i, &y) in years.iter().enumerate() {
        let dx = y - mean_x;
        num += dx * (et0_arr[i] - mean_y);
        den += dx * dx;
    }
    let slope = if den.abs() > 1e-10 { num / den } else { 0.0 };

    (slope, cv_percent(&precip_arr), compute_decade_cv(results))
}

fn compute_decade_cv(results: &[SeasonResult]) -> f64 {
    let mut decade_means: Vec<f64> = Vec::new();
    for d in (1960..2024).step_by(10) {
        let vals: Vec<f64> = results
            .iter()
            .filter(|r| r.year >= d && r.year < d + 10)
            .map(|r| r.total_et0_rust)
            .collect();
        if !vals.is_empty() {
            decade_means.push(barracuda::stats::mean(&vals));
        }
    }
    if decade_means.is_empty() {
        return 0.0;
    }
    cv_percent(&decade_means)
}

fn validate_climate_trends(
    v: &mut ValidationHarness,
    results: &[SeasonResult],
    benchmark: &serde_json::Value,
) {
    validation::section("Climate trends");

    if results.len() < 2 {
        return;
    }

    let (slope, cv_precip, decade_cv) = compute_trend_stats(results);

    let ct = benchmark
        .get("validation_checks")
        .and_then(|c| c.get("climate_trends"))
        .and_then(|p| p.get("checks"))
        .and_then(|a| a.as_array());

    if let Some(checks) = ct {
        for c in checks {
            let cid = c.get("id").and_then(|v| v.as_str()).unwrap_or("");
            match cid {
                "et0_trend" => {
                    v.check_bool(
                        &format!("ET₀ trend ≥ -0.5 mm/yr: slope {slope:.4}"),
                        slope >= -0.5,
                    );
                }
                "precip_variability" => {
                    let min_cv = c
                        .get("min_cv")
                        .and_then(serde_json::Value::as_f64)
                        .unwrap_or(0.0);
                    let max_cv = c
                        .get("max_cv")
                        .and_then(serde_json::Value::as_f64)
                        .unwrap_or(100.0);
                    v.check_bool(
                        &format!("Precip CV 15-40%: {cv_precip:.1}%"),
                        (min_cv..=max_cv).contains(&cv_precip),
                    );
                }
                "decade_means_stable" => {
                    let max_cv = c
                        .get("max_decade_cv")
                        .and_then(serde_json::Value::as_f64)
                        .unwrap_or(100.0);
                    v.check_bool(
                        &format!("Decade ET₀ CV < 15%: {decade_cv:.1}%"),
                        decade_cv <= max_cv,
                    );
                }
                _ => {}
            }
        }
    }
}

fn validate_cross_checks(
    v: &mut ValidationHarness,
    results: &[SeasonResult],
    benchmark: &serde_json::Value,
) {
    validation::section("Cross-validation");

    let n = results.len();
    if n == 0 {
        return;
    }

    let mean_et: f64 = results.iter().map(|r| r.total_et).sum::<f64>() / n as f64;
    let mean_precip: f64 = results.iter().map(|r| r.total_precip).sum::<f64>() / n as f64;
    let et_precip_ratio = if mean_precip > 0.0 {
        mean_et / mean_precip
    } else {
        0.0
    };

    let irrig_needed = results.iter().filter(|r| r.irrig_events > 0).count();
    let irrig_pct = 100.0 * irrig_needed as f64 / n as f64;

    let cv = benchmark
        .get("validation_checks")
        .and_then(|c| c.get("cross_validation"))
        .and_then(|p| p.get("checks"))
        .and_then(|a| a.as_array());

    if let Some(checks) = cv {
        for c in checks {
            let cid = c.get("id").and_then(|v| v.as_str()).unwrap_or("");
            match cid {
                "et_precip_ratio" => {
                    let min_v = c
                        .get("min")
                        .and_then(serde_json::Value::as_f64)
                        .unwrap_or(0.0);
                    let max_v = c
                        .get("max")
                        .and_then(serde_json::Value::as_f64)
                        .unwrap_or(2.0);
                    v.check_bool(
                        &format!("ET/Precip ratio 0.6-1.8: {et_precip_ratio:.2}"),
                        (min_v..=max_v).contains(&et_precip_ratio),
                    );
                }
                "irrigation_need" => {
                    let min_pct = c
                        .get("min_pct")
                        .and_then(serde_json::Value::as_f64)
                        .unwrap_or(0.0);
                    let max_pct = c
                        .get("max_pct")
                        .and_then(serde_json::Value::as_f64)
                        .unwrap_or(100.0);
                    v.check_bool(
                        &format!("Irrigation needed in 60-100%: {irrig_pct:.1}%"),
                        (min_pct..=max_pct).contains(&irrig_pct),
                    );
                }
                _ => {}
            }
        }
    }
}

fn validate_et0_cross(v: &mut ValidationHarness, results: &[SeasonResult]) {
    let with_om: Vec<&SeasonResult> = results.iter().filter(|r| r.total_et0_om > 0.0).collect();

    if with_om.len() < 10 {
        return;
    }

    validation::section("Rust Hargreaves vs Open-Meteo ET₀");

    let mean_rust: f64 =
        with_om.iter().map(|r| r.total_et0_rust).sum::<f64>() / with_om.len() as f64;
    let mean_om: f64 = with_om.iter().map(|r| r.total_et0_om).sum::<f64>() / with_om.len() as f64;
    let pct_diff = if mean_om > 0.0 {
        100.0 * (mean_rust - mean_om).abs() / mean_om
    } else {
        0.0
    };

    v.check_bool(
        &format!(
            "Rust Hargreaves vs Open-Meteo ET₀ within {}%: {pct_diff:.1}% diff",
            tolerances::ET0_CROSS_METHOD_PCT.abs_tol
        ),
        pct_diff <= tolerances::ET0_CROSS_METHOD_PCT.abs_tol,
    );
}

fn main() {
    validation::init_tracing();
    validation::banner("Long-Term Water Balance Validation (Exp 015)");

    let benchmark =
        parse_benchmark_json(BENCHMARK_JSON).expect("benchmark_long_term_wb.json must parse");

    let site = benchmark.get("site").expect("benchmark must have site");
    let lat = json_f64(site, &["latitude"]).expect("site.latitude");
    let fc = json_f64(site, &["field_capacity"]).expect("site.field_capacity");
    let wp = json_f64(site, &["wilting_point"]).expect("site.wilting_point");
    let root_depth_m = json_f64(site, &["root_depth_m"]).expect("site.root_depth_m");
    let p = json_f64(site, &["depletion_fraction"]).expect("site.depletion_fraction");

    let root_depth_mm = root_depth_m * 1000.0;
    let coeffs = CropType::Corn.coefficients();
    let kc = coeffs.kc_mid;
    let irrig_depth_mm = 25.0;

    let cache_path = std::env::var("LONG_TERM_WB_CACHE").map_or_else(
        |_| {
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .parent()
                .expect("manifest parent")
                .join("control")
                .join("long_term_wb")
                .join("data")
                .join("wooster_era5_1960_2023.json")
        },
        std::path::PathBuf::from,
    );

    if !cache_path.exists() {
        println!("═══════════════════════════════════════════════════════════");
        println!("  [SKIP] Weather cache not found: {}", cache_path.display());
        println!("  Generate it by running:");
        println!("    python control/long_term_wb/long_term_water_balance.py");
        println!("  (Downloads from Open-Meteo ERA5 archive)");
        println!("═══════════════════════════════════════════════════════════");
        std::process::exit(0);
    }

    let cache: serde_json::Value = {
        let file = std::fs::File::open(&cache_path).expect("open weather cache");
        let reader = std::io::BufReader::new(file);
        serde_json::from_reader(reader).expect("parse weather cache")
    };

    let seasons = parse_weather_cache(&cache).expect("parse seasons from cache");

    let params = SiteParams {
        lat_deg: lat,
        fc,
        wp,
        root_depth_mm,
        p,
        kc,
        irrig_depth_mm,
    };

    let mut results: Vec<SeasonResult> = Vec::new();
    for days in seasons.values() {
        if days.len() < MIN_SEASON_DAYS {
            continue;
        }
        results.push(run_season(days, &params));
    }

    results.sort_by_key(|r| r.year);

    println!("  Site: Wooster, OH | Corn on silt loam");
    println!("  Seasons: {} (May 1 - Sep 30)", results.len());
    println!();

    let mut v = ValidationHarness::new("Long-Term Water Balance Validation");

    validate_physical_checks(&mut v, &results, &benchmark);
    validate_climate_trends(&mut v, &results, &benchmark);
    validate_cross_checks(&mut v, &results, &benchmark);
    validate_et0_cross(&mut v, &results);

    v.finish();
}
