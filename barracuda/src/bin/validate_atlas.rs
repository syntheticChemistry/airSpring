// SPDX-License-Identifier: AGPL-3.0-or-later
#![allow(clippy::doc_markdown)]
//! Experiment 018: Michigan Crop Water Atlas.
//!
//! Runs the validated ET₀ + water balance + yield response pipeline across
//! all available stations and crops, producing a comprehensive crop water
//! budget dataset.
//!
//! This is NOT new science — it is the validated stack (Exp 001, 004, 008)
//! applied at scale. The value is the dataset itself.
//!
//! Data: Open-Meteo ERA5 daily CSVs (from `scripts/download_open_meteo.py`)
//! Crops: 10 Michigan crops (from `eco::crop::CropType`)
//! Output: Per-station, per-crop seasonal water budgets
//!
//! Runtime config via env vars (all optional):
//!   ATLAS_DATA_DIR   — path to Open-Meteo CSVs (default: data/open_meteo)
//!   ATLAS_YEAR_START — first year to process (default: 1945)
//!   ATLAS_YEAR_END   — last year to process (default: 2024)
//!   ATLAS_OUT_DIR    — output directory for results (default: data/atlas_results)

use airspring_barracuda::eco::{
    crop::CropType,
    evapotranspiration::{self as et, DailyEt0Input},
    water_balance::{self as wb, DailyInput, WaterBalanceState},
    yield_response,
};
use airspring_barracuda::testutil;
use airspring_barracuda::validation::{self, ValidationHarness};
use std::io::BufRead;
use std::path::Path;

const ALL_CROPS: &[(CropType, &str, f64)] = &[
    (CropType::Corn, "corn", 1.25),
    (CropType::Soybean, "soybean", 0.85),
    (CropType::WinterWheat, "wheat", 1.00),
    (CropType::SugarBeet, "sugarbeet", 1.10),
    (CropType::DryBean, "drybean", 1.15),
    (CropType::Potato, "potato", 1.10),
    (CropType::Tomato, "tomato", 1.05),
    (CropType::Blueberry, "blueberry", 0.80),
    (CropType::Alfalfa, "alfalfa", 1.10),
    (CropType::Turfgrass, "turfgrass", 0.80),
];

const DEFAULT_YEAR_START: u32 = 1945;
const DEFAULT_YEAR_END: u32 = 2024;
const MASS_BALANCE_TOL: f64 = 0.01;
const SEASON_START_DOY: u32 = 121; // May 1
const SEASON_END_DOY: u32 = 273; // Sep 30

struct AtlasConfig {
    data_dir: std::path::PathBuf,
    out_dir: std::path::PathBuf,
    year_start: u32,
    year_end: u32,
}

impl AtlasConfig {
    fn discover() -> Self {
        let data_dir = std::env::var("ATLAS_DATA_DIR").map_or_else(
            |_| {
                Path::new(env!("CARGO_MANIFEST_DIR"))
                    .parent()
                    .expect("CARGO_MANIFEST_DIR parent")
                    .join("data")
                    .join("open_meteo")
            },
            std::path::PathBuf::from,
        );
        let out_dir = std::env::var("ATLAS_OUT_DIR").map_or_else(
            |_| {
                Path::new(env!("CARGO_MANIFEST_DIR"))
                    .parent()
                    .expect("CARGO_MANIFEST_DIR parent")
                    .join("data")
                    .join("atlas_results")
            },
            std::path::PathBuf::from,
        );
        let year_start = std::env::var("ATLAS_YEAR_START")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(DEFAULT_YEAR_START);
        let year_end = std::env::var("ATLAS_YEAR_END")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(DEFAULT_YEAR_END);
        Self {
            data_dir,
            out_dir,
            year_start,
            year_end,
        }
    }
}

struct WeatherDay {
    year: u32,
    doy: u32,
    tmax_c: f64,
    tmin_c: f64,
    tmean_c: f64,
    rh_max_pct: f64,
    rh_min_pct: f64,
    wind_2m_m_s: f64,
    solar_rad_mj_m2: f64,
    precip_mm: f64,
    et0_openmeteo_mm: f64,
    lat: f64,
    elevation_m: f64,
}

struct StationResult {
    station: String,
    n_days: usize,
    n_years: usize,
    mean_annual_et0: f64,
    et0_r2_vs_openmeteo: f64,
    crop_results: Vec<CropSeasonSummary>,
    mb_max_error: f64,
}

struct CropSeasonSummary {
    crop_name: String,
    n_seasons: usize,
    mean_seasonal_et: f64,
    mean_seasonal_precip: f64,
    mean_stress_days: f64,
    mean_yield_ratio: f64,
    mean_irrig_mm: f64,
}

fn date_to_doy(date_str: &str) -> (u32, u32) {
    let parts: Vec<&str> = date_str.split('-').collect();
    if parts.len() != 3 {
        return (2023, 1);
    }
    let year: u32 = parts[0].parse().unwrap_or(2023);
    let month: u32 = parts[1].parse().unwrap_or(1);
    let day: u32 = parts[2].parse().unwrap_or(1);

    let is_leap = year.is_multiple_of(4) && (!year.is_multiple_of(100) || year.is_multiple_of(400));
    let days_before: [u32; 12] = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334];
    let mut doy = days_before[(month - 1).min(11) as usize] + day;
    if is_leap && month > 2 {
        doy += 1;
    }
    (year, doy)
}

fn parse_csv(path: &Path) -> Vec<WeatherDay> {
    let Ok(file) = std::fs::File::open(path) else {
        return Vec::new();
    };
    let reader = std::io::BufReader::new(file);
    let mut header_map: Vec<String> = Vec::new();
    let mut rows = Vec::new();

    for (i, line) in reader.lines().enumerate() {
        let Ok(line) = line else { continue };
        if i == 0 {
            header_map = line.split(',').map(String::from).collect();
            continue;
        }
        let fields: Vec<&str> = line.split(',').collect();
        if fields.len() < header_map.len() {
            continue;
        }

        let col = |name: &str| -> f64 {
            header_map
                .iter()
                .position(|h| h == name)
                .and_then(|idx| fields[idx].parse::<f64>().ok())
                .unwrap_or(f64::NAN)
        };

        let date_idx = header_map.iter().position(|h| h == "date").unwrap_or(0);
        let (year, doy) = date_to_doy(fields[date_idx]);

        rows.push(WeatherDay {
            year,
            doy,
            tmax_c: col("tmax_c"),
            tmin_c: col("tmin_c"),
            tmean_c: col("tmean_c"),
            rh_max_pct: col("rh_max_pct"),
            rh_min_pct: col("rh_min_pct"),
            wind_2m_m_s: col("wind_2m_m_s"),
            solar_rad_mj_m2: col("solar_rad_mj_m2"),
            precip_mm: col("precip_mm"),
            et0_openmeteo_mm: col("et0_openmeteo_mm"),
            lat: col("lat"),
            elevation_m: col("elevation_m"),
        });
    }
    rows
}

fn compute_et0(day: &WeatherDay) -> f64 {
    if day.tmax_c.is_nan() || day.solar_rad_mj_m2.is_nan() {
        return 0.0;
    }
    let ea = et::actual_vapour_pressure_rh(day.tmin_c, day.tmax_c, day.rh_min_pct, day.rh_max_pct);
    let input = DailyEt0Input {
        tmin: day.tmin_c,
        tmax: day.tmax_c,
        tmean: if day.tmean_c.is_finite() {
            Some(day.tmean_c)
        } else {
            None
        },
        solar_radiation: day.solar_rad_mj_m2,
        wind_speed_2m: if day.wind_2m_m_s.is_finite() {
            day.wind_2m_m_s
        } else {
            2.0
        },
        actual_vapour_pressure: ea,
        elevation_m: day.elevation_m,
        latitude_deg: day.lat,
        day_of_year: day.doy,
    };
    et::daily_et0(&input).et0.max(0.0)
}

const fn default_soil() -> (f64, f64) {
    (0.30, 0.12)
}

fn run_crop_seasons(
    seasons: &std::collections::BTreeMap<u32, Vec<(f64, f64)>>,
    crop_type: CropType,
    ky_total: f64,
) -> (CropSeasonSummary, f64) {
    let kc = crop_type.coefficients();
    let (fc, wp) = default_soil();
    let root_mm = kc.root_depth_m * 1000.0;
    let p = kc.depletion_fraction;
    let irrig_depth = 25.0;

    let mut total_et = 0.0_f64;
    let mut total_precip = 0.0_f64;
    let mut total_stress = 0.0_f64;
    let mut total_yield = 0.0_f64;
    let mut total_irrig = 0.0_f64;
    let mut max_mb = 0.0_f64;
    let mut n_seasons = 0_usize;

    for days in seasons.values() {
        if days.len() < 60 {
            continue;
        }
        n_seasons += 1;

        let initial = WaterBalanceState::new(fc, wp, root_mm, p);
        let mut state = initial.clone();
        let mut inputs = Vec::with_capacity(days.len());
        let mut outputs = Vec::with_capacity(days.len());
        let mut season_irrig = 0.0_f64;

        let n_days = days.len();
        for (i, &(et0, precip)) in days.iter().enumerate() {
            let frac = i as f64 / n_days as f64;
            let kc_daily = if frac < 0.2 {
                kc.kc_ini
            } else if frac < 0.7 {
                kc.kc_mid
            } else {
                kc.kc_end
            };

            let irr = if state.depletion > state.raw {
                irrig_depth
            } else {
                0.0
            };
            season_irrig += irr;

            let input = DailyInput {
                precipitation: precip,
                irrigation: irr,
                et0,
                kc: kc_daily,
            };
            let output = state.step(&input);
            inputs.push(input);
            outputs.push(output);
        }

        let mb = wb::mass_balance_check(&inputs, &outputs, 0.0, state.depletion);
        max_mb = max_mb.max(mb);

        let season_et: f64 = outputs.iter().map(|o| o.actual_et).sum();
        let season_etc: f64 = outputs.iter().map(|o| o.etc).sum();
        let season_precip: f64 = days.iter().map(|(_, p)| p).sum();
        let stress_days = outputs.iter().filter(|o| o.ks < 1.0).count();

        let eta_etc = if season_etc > 0.0 {
            season_et / season_etc
        } else {
            1.0
        };
        let yr = yield_response::clamp_yield_ratio(yield_response::yield_ratio_single(
            ky_total, eta_etc,
        ));

        total_et += season_et;
        total_precip += season_precip;
        total_stress += stress_days as f64;
        total_yield += yr;
        total_irrig += season_irrig;
    }

    let n = n_seasons.max(1) as f64;
    let summary = CropSeasonSummary {
        crop_name: kc.name.to_string(),
        n_seasons,
        mean_seasonal_et: total_et / n,
        mean_seasonal_precip: total_precip / n,
        mean_stress_days: total_stress / n,
        mean_yield_ratio: total_yield / n,
        mean_irrig_mm: total_irrig / n,
    };

    (summary, max_mb)
}

fn discover_station_csvs(data_dir: &Path) -> Vec<(String, std::path::PathBuf)> {
    let mut stations = Vec::new();
    let Ok(entries) = std::fs::read_dir(data_dir) else {
        return stations;
    };

    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().to_string();
        if name.ends_with("_daily.csv") && !name.starts_with("all_stations") {
            let station_id = name
                .split('_')
                .take_while(|p| p.parse::<u32>().is_err())
                .collect::<Vec<&str>>()
                .join("_");
            if !station_id.is_empty() {
                stations.push((station_id, entry.path()));
            }
        }
    }

    stations.sort_by(|a, b| a.0.cmp(&b.0));
    stations.dedup_by(|a, b| a.0 == b.0);
    stations
}

fn process_station(
    station_id: &str,
    csv_path: &Path,
    config: &AtlasConfig,
) -> Option<StationResult> {
    let all_days = parse_csv(csv_path);
    if all_days.is_empty() {
        return None;
    }

    let days: Vec<&WeatherDay> = all_days
        .iter()
        .filter(|d| {
            d.year >= config.year_start
                && d.year <= config.year_end
                && !d.tmax_c.is_nan()
                && !d.solar_rad_mj_m2.is_nan()
        })
        .collect();

    if days.is_empty() {
        return None;
    }

    let et0_series: Vec<f64> = days.iter().map(|d| compute_et0(d)).collect();
    let om_series: Vec<f64> = days.iter().map(|d| d.et0_openmeteo_mm).collect();

    let valid_pairs: Vec<(f64, f64)> = et0_series
        .iter()
        .zip(&om_series)
        .filter(|(r, o)| o.is_finite() && **o > 0.0 && r.is_finite())
        .map(|(&r, &o)| (r, o))
        .collect();

    let r2 = if valid_pairs.len() > 10 {
        let (rust_v, om_v): (Vec<f64>, Vec<f64>) = valid_pairs.into_iter().unzip();
        testutil::r_squared(&om_v, &rust_v).unwrap_or(0.0)
    } else {
        f64::NAN
    };

    let mut seasons: std::collections::BTreeMap<u32, Vec<(f64, f64)>> =
        std::collections::BTreeMap::new();
    for (day, &et0) in days.iter().zip(&et0_series) {
        if day.doy >= SEASON_START_DOY && day.doy <= SEASON_END_DOY {
            seasons
                .entry(day.year)
                .or_default()
                .push((et0, day.precip_mm.max(0.0)));
        }
    }

    let n_years = seasons.len();
    let total_annual_et0: f64 = {
        let mut yearly: std::collections::BTreeMap<u32, f64> = std::collections::BTreeMap::new();
        for (day, &et0) in days.iter().zip(&et0_series) {
            *yearly.entry(day.year).or_default() += et0;
        }
        yearly.values().sum::<f64>() / yearly.len().max(1) as f64
    };

    let mut crop_results = Vec::new();
    let mut mb_max = 0.0_f64;

    for &(crop_type, _, ky) in ALL_CROPS {
        let (summary, mb) = run_crop_seasons(&seasons, crop_type, ky);
        mb_max = mb_max.max(mb);
        crop_results.push(summary);
    }

    Some(StationResult {
        station: station_id.to_string(),
        n_days: days.len(),
        n_years,
        mean_annual_et0: total_annual_et0,
        et0_r2_vs_openmeteo: r2,
        crop_results,
        mb_max_error: mb_max,
    })
}

fn write_summary_csv(results: &[StationResult], out_dir: &Path) {
    if results.is_empty() {
        return;
    }
    std::fs::create_dir_all(out_dir).ok();

    let path = out_dir.join("atlas_station_summary.csv");
    let mut lines = vec![
        "station,n_days,n_years,mean_annual_et0_mm,et0_r2_vs_openmeteo,mb_max_error_mm".to_string(),
    ];
    for r in results {
        lines.push(format!(
            "{},{},{},{:.1},{:.4},{:.6}",
            r.station,
            r.n_days,
            r.n_years,
            r.mean_annual_et0,
            r.et0_r2_vs_openmeteo,
            r.mb_max_error
        ));
    }
    std::fs::write(&path, lines.join("\n")).ok();
    println!("  Station summary: {}", path.display());

    let crop_path = out_dir.join("atlas_crop_summary.csv");
    let mut crop_lines = vec![
        "station,crop,n_seasons,mean_et_mm,mean_precip_mm,mean_stress_days,mean_yield_ratio,mean_irrig_mm".to_string(),
    ];
    for r in results {
        for c in &r.crop_results {
            crop_lines.push(format!(
                "{},{},{},{:.1},{:.1},{:.1},{:.3},{:.1}",
                r.station,
                c.crop_name,
                c.n_seasons,
                c.mean_seasonal_et,
                c.mean_seasonal_precip,
                c.mean_stress_days,
                c.mean_yield_ratio,
                c.mean_irrig_mm,
            ));
        }
    }
    std::fs::write(&crop_path, crop_lines.join("\n")).ok();
    println!("  Crop summary: {}", crop_path.display());
}

fn validate_station(station_id: &str, result: &StationResult, v: &mut ValidationHarness) {
    v.check_bool(
        &format!(
            "{station_id} mass balance < {MASS_BALANCE_TOL} (err={:.6})",
            result.mb_max_error
        ),
        result.mb_max_error < MASS_BALANCE_TOL,
    );

    if result.et0_r2_vs_openmeteo.is_finite() {
        v.check_bool(
            &format!(
                "{station_id} ET₀ R² > 0.85 (R²={:.3})",
                result.et0_r2_vs_openmeteo
            ),
            result.et0_r2_vs_openmeteo > 0.85,
        );
    }

    v.check_bool(
        &format!(
            "{station_id} annual ET₀ 400-1200 mm ({:.0})",
            result.mean_annual_et0
        ),
        (400.0..=1200.0).contains(&result.mean_annual_et0),
    );

    for c in &result.crop_results {
        if c.n_seasons > 0 {
            v.check_bool(
                &format!(
                    "{station_id}/{} yield ratio 0.3-1.0 ({:.3})",
                    c.crop_name, c.mean_yield_ratio
                ),
                (0.3..=1.0).contains(&c.mean_yield_ratio),
            );
        }
    }
}

fn validate_atlas_summary(results: &[StationResult], v: &mut ValidationHarness) {
    validation::section("Atlas Summary");

    if results.is_empty() {
        return;
    }

    let n = results.len();
    let mean_et0 = results.iter().map(|r| r.mean_annual_et0).sum::<f64>() / n as f64;
    let total_station_days: usize = results.iter().map(|r| r.n_days).sum();
    let total_station_years: usize = results.iter().map(|r| r.n_years).sum();

    println!("  Stations processed: {n}");
    println!("  Total station-days: {total_station_days}");
    println!("  Total station-years: {total_station_years}");
    println!("  Mean annual ET₀: {mean_et0:.0} mm");
    println!(
        "  Total crop-station-seasons: {}",
        results
            .iter()
            .flat_map(|r| &r.crop_results)
            .map(|c| c.n_seasons)
            .sum::<usize>()
    );

    v.check_bool(&format!("At least 1 station processed ({n})"), n > 0);
    v.check_bool(
        &format!("Michigan mean ET₀ 500-1200 mm ({mean_et0:.0})"),
        (500.0..=1200.0).contains(&mean_et0),
    );
}

fn main() {
    validation::init_tracing();
    let config = AtlasConfig::discover();

    if !config.data_dir.exists() {
        println!("═══════════════════════════════════════════════════════════");
        println!("  [SKIP] No weather data at {}", config.data_dir.display());
        println!("  Download atlas data first:");
        println!("    python scripts/download_open_meteo.py --atlas --year-range 1945-2024");
        println!("  Or for a quick test with 6 stations:");
        println!("    python scripts/download_open_meteo.py --all-stations --growing-season 2023");
        println!("═══════════════════════════════════════════════════════════");
        std::process::exit(0);
    }

    validation::banner("Michigan Crop Water Atlas (Exp 018)");
    println!(
        "  Years: {}-{} | Crops: {} | Data: {}",
        config.year_start,
        config.year_end,
        ALL_CROPS.len(),
        config.data_dir.display()
    );

    let station_csvs = discover_station_csvs(&config.data_dir);
    if station_csvs.is_empty() {
        println!(
            "  [SKIP] No station CSVs found in {}",
            config.data_dir.display()
        );
        std::process::exit(0);
    }

    println!("  Stations discovered: {}\n", station_csvs.len());

    let mut v = ValidationHarness::new("Michigan Crop Water Atlas");
    let mut results: Vec<StationResult> = Vec::new();

    for (station_id, csv_path) in &station_csvs {
        print!("  {station_id}...");
        match process_station(station_id, csv_path, &config) {
            Some(result) => {
                let r2_str = if result.et0_r2_vs_openmeteo.is_finite() {
                    format!("{:.3}", result.et0_r2_vs_openmeteo)
                } else {
                    "N/A".to_string()
                };
                println!(
                    " {} days, {} yrs, ET₀={:.0} mm/yr, R²={}, MB<{:.6}",
                    result.n_days,
                    result.n_years,
                    result.mean_annual_et0,
                    r2_str,
                    result.mb_max_error
                );
                validate_station(station_id, &result, &mut v);
                results.push(result);
            }
            None => {
                println!(" skipped (no valid data)");
            }
        }
    }

    println!();
    validate_atlas_summary(&results, &mut v);
    write_summary_csv(&results, &config.out_dir);
    println!();

    v.finish();
}
