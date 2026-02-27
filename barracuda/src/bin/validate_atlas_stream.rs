// SPDX-License-Identifier: AGPL-3.0-or-later
#![warn(clippy::pedantic)]
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
//! Experiment 046: Atlas Stream — real 80yr data through `SeasonalPipeline` + `AtlasStream`.
//!
//! Wires the new GPU-ready orchestrators (`gpu::seasonal_pipeline`, `gpu::atlas_stream`)
//! to real Open-Meteo ERA5 80-year daily CSVs downloaded by `scripts/download_atlas_80yr.py`.
//!
//! This validates:
//! 1. CSV → `WeatherDay` → `StationBatch` parsing on real data
//! 2. `SeasonalPipeline` ET₀→Kc→WB→Yield chain produces physically plausible results
//! 3. `AtlasStream` multi-station, multi-crop orchestration
//! 4. Mass balance closure, yield ratio bounds, ET₀ ranges on real climate
//!
//! Data: 80yr Open-Meteo ERA5 daily CSVs (11+ stations, ~29,220 days each)
//! Crops: 5 Michigan crops (corn, soybean, wheat, potato, alfalfa)
//!
//! Runtime config via env vars (all optional):
//!   `ATLAS_STREAM_DIR` — path to 80yr CSVs (default: `data/open_meteo`)

use airspring_barracuda::eco::crop::CropType;
use airspring_barracuda::gpu::atlas_stream::{AtlasStream, AtlasStreamConfig, StationBatch};
use airspring_barracuda::gpu::seasonal_pipeline::{CropConfig, WeatherDay};
use airspring_barracuda::validation::ValidationHarness;
use std::io::BufRead;
use std::path::Path;

const SEASON_START_DOY: u32 = 121;
const SEASON_END_DOY: u32 = 273;

const TEST_CROPS: &[CropType] = &[
    CropType::Corn,
    CropType::Soybean,
    CropType::WinterWheat,
    CropType::Potato,
    CropType::Alfalfa,
];

struct CsvRow {
    year: u32,
    doy: u32,
    tmax_c: f64,
    tmin_c: f64,
    rh_max_pct: f64,
    rh_min_pct: f64,
    wind_2m_m_s: f64,
    solar_rad_mj_m2: f64,
    precip_mm: f64,
    lat: f64,
    elevation_m: f64,
}

fn date_to_doy(date_str: &str) -> (u32, u32) {
    let parts: Vec<&str> = date_str.split('-').collect();
    if parts.len() != 3 {
        return (2023, 1);
    }
    let year: u32 = parts[0].parse().unwrap_or(2023);
    let month: u32 = parts[1].parse().unwrap_or(1);
    let day: u32 = parts[2].parse().unwrap_or(1);

    let is_leap =
        year.is_multiple_of(4) && (!year.is_multiple_of(100) || year.is_multiple_of(400));
    let days_before: [u32; 12] = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334];
    let mut doy = days_before[(month - 1).min(11) as usize] + day;
    if is_leap && month > 2 {
        doy += 1;
    }
    (year, doy)
}

fn parse_80yr_csv(path: &Path) -> Vec<CsvRow> {
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
                .and_then(|idx| fields.get(idx).and_then(|v| v.parse::<f64>().ok()))
                .unwrap_or(f64::NAN)
        };

        let date_idx = header_map.iter().position(|h| h == "date").unwrap_or(0);
        let (year, doy) = date_to_doy(fields[date_idx]);

        rows.push(CsvRow {
            year,
            doy,
            tmax_c: col("tmax_c"),
            tmin_c: col("tmin_c"),
            rh_max_pct: col("rh_max_pct"),
            rh_min_pct: col("rh_min_pct"),
            wind_2m_m_s: col("wind_2m_m_s"),
            solar_rad_mj_m2: col("solar_rad_mj_m2"),
            precip_mm: col("precip_mm"),
            lat: col("lat"),
            elevation_m: col("elevation_m"),
        });
    }
    rows
}

const fn csv_row_to_weather_day(row: &CsvRow) -> WeatherDay {
    WeatherDay {
        tmax: row.tmax_c,
        tmin: row.tmin_c,
        rh_max: if row.rh_max_pct.is_finite() {
            row.rh_max_pct
        } else {
            80.0
        },
        rh_min: if row.rh_min_pct.is_finite() {
            row.rh_min_pct
        } else {
            40.0
        },
        wind_2m: if row.wind_2m_m_s.is_finite() {
            row.wind_2m_m_s
        } else {
            2.0
        },
        solar_rad: if row.solar_rad_mj_m2.is_finite() {
            row.solar_rad_mj_m2
        } else {
            15.0
        },
        precipitation: if row.precip_mm.is_finite() {
            row.precip_mm
        } else {
            0.0
        },
        elevation: if row.elevation_m.is_finite() {
            row.elevation_m
        } else {
            250.0
        },
        latitude_deg: if row.lat.is_finite() { row.lat } else { 42.5 },
        day_of_year: row.doy,
    }
}

fn discover_80yr_csvs(dir: &Path) -> Vec<(String, std::path::PathBuf)> {
    let suffix = "_1945-01-01_2024-12-31_daily.csv";
    let mut stations = Vec::new();
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            if name.ends_with(suffix) {
                let station_id = name.replace(suffix, "");
                stations.push((station_id, entry.path()));
            }
        }
    }
    stations.sort_by(|a, b| a.0.cmp(&b.0));
    stations
}

fn build_station_batches(
    station_id: &str,
    rows: &[CsvRow],
    year_start: u32,
    year_end: u32,
) -> Vec<StationBatch> {
    let mut batches = Vec::new();
    for year in year_start..=year_end {
        let weather: Vec<WeatherDay> = rows
            .iter()
            .filter(|r| r.year == year && r.doy >= SEASON_START_DOY && r.doy <= SEASON_END_DOY)
            .filter(|r| r.tmax_c.is_finite() && r.tmin_c.is_finite())
            .map(csv_row_to_weather_day)
            .collect();
        if weather.len() >= 100 {
            batches.push(StationBatch {
                station_id: station_id.to_string(),
                year,
                weather,
            });
        }
    }
    batches
}

fn validate_station_results(
    h: &mut ValidationHarness,
    station_id: &str,
    results: &[airspring_barracuda::gpu::atlas_stream::StationSeasonResult],
) {
    let n = results.len();
    h.check_lower(&format!("{station_id} produced results"), n as f64, 1.0);

    let mut total_mb_err = 0.0;
    let mut n_mb = 0_usize;
    let mut total_yield = 0.0;
    let mut n_yield = 0_usize;
    let mut total_et0 = 0.0;
    let mut n_et0 = 0_usize;

    for r in results {
        let sr = &r.result;
        let mb = sr.mass_balance_error.abs();
        total_mb_err += mb;
        n_mb += 1;

        if sr.yield_ratio > 0.0 && sr.yield_ratio <= 1.0 {
            total_yield += sr.yield_ratio;
            n_yield += 1;
        }

        if sr.n_days > 0 {
            let daily_et0 = sr.total_et0 / sr.n_days as f64;
            total_et0 += daily_et0;
            n_et0 += 1;
        }
    }

    if n_mb > 0 {
        let mean_mb = total_mb_err / n_mb as f64;
        h.check_upper(&format!("{station_id} mass balance < 1 mm"), mean_mb, 1.0);
    }

    if n_yield > 0 {
        let mean_yield = total_yield / n_yield as f64;
        h.check_bool(
            &format!("{station_id} mean yield ratio in [0.3, 1.0]"),
            (0.3..=1.0).contains(&mean_yield),
        );
    }

    if n_et0 > 0 {
        let mean_daily_et0 = total_et0 / n_et0 as f64;
        h.check_bool(
            &format!("{station_id} mean daily ET₀ in [2, 8] mm"),
            (2.0..=8.0).contains(&mean_daily_et0),
        );
    }
}

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter("info")
        .with_target(false)
        .init();

    let data_dir = std::env::var("ATLAS_STREAM_DIR").map_or_else(
        |_| {
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .parent()
                .expect("parent")
                .join("data")
                .join("open_meteo")
        },
        std::path::PathBuf::from,
    );

    let stations = discover_80yr_csvs(&data_dir);
    println!("Atlas Stream: found {} 80yr station files", stations.len());

    let mut h = ValidationHarness::new("Atlas Stream — Real 80yr Data → SeasonalPipeline");

    h.check_lower("≥1 station available", stations.len() as f64, 1.0);

    if stations.is_empty() {
        println!("No 80yr station files found — download in progress.");
        h.finish();
        #[allow(unreachable_code)]
        return;
    }

    let stream = AtlasStream::new();
    let crop_configs: Vec<CropConfig> =
        TEST_CROPS.iter().map(|ct| CropConfig::standard(*ct)).collect();

    let config = AtlasStreamConfig {
        crop_configs: crop_configs.clone(),
        year_range: 1945..2025,
    };

    let mut total_batches = 0_usize;
    let mut total_results = 0_usize;
    let mut station_years: Vec<(String, usize)> = Vec::new();

    for (station_id, path) in &stations {
        let rows = parse_80yr_csv(path);
        let valid_rows = rows.iter().filter(|r| r.tmax_c.is_finite()).count();
        println!(
            "  {station_id}: {} total rows, {valid_rows} valid",
            rows.len()
        );

        let batches = build_station_batches(station_id, &rows, 1945, 2024);
        let n_batches = batches.len();
        station_years.push((station_id.clone(), n_batches));
        total_batches += n_batches;

        h.check_lower(
            &format!("{station_id} has ≥50 valid seasons"),
            n_batches as f64,
            50.0,
        );

        let results = stream.process_batch(&batches, &config);
        total_results += results.len();

        let expected = n_batches * crop_configs.len();
        h.check_bool(
            &format!("{station_id} result count = seasons × crops ({expected})"),
            results.len() == expected,
        );

        validate_station_results(&mut h, station_id, &results);
    }

    println!(
        "\n=== Atlas Stream Summary ===\n  Stations: {}\n  Total season-batches: {total_batches}\n  Total results (station × crop × year): {total_results}\n  Crops: {}",
        stations.len(),
        TEST_CROPS
            .iter()
            .map(|c| c.coefficients().name)
            .collect::<Vec<_>>()
            .join(", ")
    );

    for (sid, years) in &station_years {
        println!("  {sid}: {years} valid seasons (1945–2024)");
    }

    h.finish();
}
