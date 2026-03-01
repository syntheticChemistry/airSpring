// SPDX-License-Identifier: AGPL-3.0-or-later
#![warn(clippy::pedantic)]
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::too_many_lines,
    clippy::option_if_let_else
)]
//! Experiment 059: Atlas 80-Year Decade Analysis.
//!
//! Validates decade-aggregated ET₀ trends across Michigan stations against
//! the Python control (`control/atlas_decade/atlas_decade_analysis.py`).
//!
//! Uses Open-Meteo ERA5 daily CSVs and the validated FAO-56 pipeline to
//! compute seasonal ET₀ by decade, then compares against Python benchmarks.

use airspring_barracuda::eco::evapotranspiration::{self as et, DailyEt0Input};
use airspring_barracuda::validation::ValidationHarness;
use std::collections::BTreeMap;
use std::io::BufRead;
use std::path::Path;

const BENCHMARK_JSON: &str =
    include_str!("../../../control/atlas_decade/benchmark_atlas_decade.json");

const SEASON_START_DOY: u32 = 121;
const SEASON_END_DOY: u32 = 273;

struct StationMeta {
    id: &'static str,
    lat: f64,
    elevation_m: f64,
}

const STATIONS: &[StationMeta] = &[
    StationMeta {
        id: "east_lansing",
        lat: 42.727,
        elevation_m: 256.0,
    },
    StationMeta {
        id: "grand_junction",
        lat: 42.375,
        elevation_m: 197.0,
    },
    StationMeta {
        id: "sparta",
        lat: 43.160,
        elevation_m: 262.0,
    },
    StationMeta {
        id: "hart",
        lat: 43.698,
        elevation_m: 244.0,
    },
    StationMeta {
        id: "west_olive",
        lat: 42.917,
        elevation_m: 192.0,
    },
    StationMeta {
        id: "manchester",
        lat: 42.153,
        elevation_m: 290.0,
    },
];

const DECADES: &[(u32, u32)] = &[
    (1950, 1959),
    (1960, 1969),
    (1970, 1979),
    (1980, 1989),
    (1990, 1999),
    (2000, 2009),
    (2010, 2019),
    (2020, 2024),
];

struct DayRow {
    year: u32,
    doy: u32,
    tmax_c: f64,
    tmin_c: f64,
    rh_max_pct: f64,
    rh_min_pct: f64,
    wind_2m_m_s: f64,
    solar_rad_mj_m2: f64,
    precip_mm: f64,
}

fn load_station_csv(station_id: &str) -> Option<Vec<DayRow>> {
    let data_dir =
        std::env::var("ATLAS_DATA_DIR").unwrap_or_else(|_| "data/open_meteo".to_string());
    let path = format!("{data_dir}/{station_id}_1945-01-01_2024-12-31_daily.csv");
    let p = Path::new(&path);
    if !p.exists() {
        return None;
    }

    let file = std::fs::File::open(p).ok()?;
    let reader = std::io::BufReader::new(file);

    let mut rows = Vec::new();
    let mut header_map: BTreeMap<String, usize> = BTreeMap::new();

    for (i, line) in reader.lines().enumerate() {
        let line = line.ok()?;
        if i == 0 {
            for (col_idx, col_name) in line.split(',').enumerate() {
                header_map.insert(col_name.trim().to_string(), col_idx);
            }
            continue;
        }

        let fields: Vec<&str> = line.split(',').collect();
        let get_f64 = |name: &str| -> f64 {
            header_map
                .get(name)
                .and_then(|&idx| fields.get(idx))
                .and_then(|s| s.trim().parse::<f64>().ok())
                .unwrap_or(f64::NAN)
        };

        let date_str = header_map
            .get("date")
            .and_then(|&idx| fields.get(idx))
            .unwrap_or(&"");

        let parts: Vec<&str> = date_str.split('-').collect();
        if parts.len() < 3 {
            continue;
        }
        let year: u32 = parts[0].parse().unwrap_or(0);
        let month: u32 = parts[1].parse().unwrap_or(0);
        let day: u32 = parts[2].parse().unwrap_or(0);
        if year == 0 {
            continue;
        }

        let doy = day_of_year(year, month, day);

        let rh_max = get_f64("rh_max_pct");
        let rh_min = get_f64("rh_min_pct");

        rows.push(DayRow {
            year,
            doy,
            tmax_c: get_f64("tmax_c"),
            tmin_c: get_f64("tmin_c"),
            rh_max_pct: if rh_max.is_nan() {
                70.0
            } else {
                rh_max.clamp(20.0, 100.0)
            },
            rh_min_pct: if rh_min.is_nan() {
                40.0
            } else {
                rh_min.clamp(10.0, 100.0)
            },
            wind_2m_m_s: {
                let w = get_f64("wind_2m_m_s");
                if w.is_nan() {
                    2.0
                } else {
                    w.max(0.5)
                }
            },
            solar_rad_mj_m2: {
                let s = get_f64("solar_rad_mj_m2");
                if s.is_nan() {
                    15.0
                } else {
                    s.max(0.1)
                }
            },
            precip_mm: {
                let p = get_f64("precip_mm");
                if p.is_nan() {
                    0.0
                } else {
                    p
                }
            },
        });
    }

    Some(rows)
}

fn day_of_year(year: u32, month: u32, day: u32) -> u32 {
    let is_leap = (year.is_multiple_of(4) && !year.is_multiple_of(100)) || year.is_multiple_of(400);
    let month_days: [u32; 12] = if is_leap {
        [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    } else {
        [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    };
    day + month_days
        .iter()
        .take((month as usize - 1).min(11))
        .sum::<u32>()
}

fn compute_et0(row: &DayRow, lat_deg: f64, elevation_m: f64) -> f64 {
    if row.tmax_c.is_nan() || row.tmin_c.is_nan() {
        return f64::NAN;
    }

    let ea = et::actual_vapour_pressure_rh(row.tmin_c, row.tmax_c, row.rh_min_pct, row.rh_max_pct);

    let input = DailyEt0Input {
        tmax: row.tmax_c,
        tmin: row.tmin_c,
        tmean: None,
        solar_radiation: row.solar_rad_mj_m2,
        wind_speed_2m: row.wind_2m_m_s,
        actual_vapour_pressure: ea,
        day_of_year: row.doy,
        latitude_deg: lat_deg,
        elevation_m,
    };

    et::daily_et0(&input).et0.max(0.0)
}

fn main() {
    tracing_subscriber::fmt().with_env_filter("info").init();

    let benchmark: serde_json::Value =
        serde_json::from_str(BENCHMARK_JSON).expect("benchmark JSON");

    let mut v = ValidationHarness::new("Exp 059: Atlas 80yr Decade Analysis");

    let bench_stations = benchmark["stations"].as_array().expect("stations array");
    let mut total_checks = 0;

    for station_meta in STATIONS {
        let Some(rows) = load_station_csv(station_meta.id) else {
            println!("  SKIP {}: no CSV", station_meta.id);
            continue;
        };

        let bench_station = bench_stations
            .iter()
            .find(|s| s["station_id"].as_str() == Some(station_meta.id));

        let Some(bench_station) = bench_station else {
            println!("  SKIP {}: no benchmark", station_meta.id);
            continue;
        };

        let bench_decades = bench_station["decades"].as_array().expect("decades");

        for &(dec_start, dec_end) in DECADES {
            let season_rows: Vec<&DayRow> = rows
                .iter()
                .filter(|r| {
                    r.year >= dec_start
                        && r.year <= dec_end
                        && r.doy >= SEASON_START_DOY
                        && r.doy <= SEASON_END_DOY
                })
                .collect();

            if season_rows.is_empty() {
                continue;
            }

            let mut yearly_et0: BTreeMap<u32, f64> = BTreeMap::new();
            let mut yearly_precip: BTreeMap<u32, f64> = BTreeMap::new();

            for row in &season_rows {
                let et0 = compute_et0(row, station_meta.lat, station_meta.elevation_m);
                if !et0.is_nan() {
                    *yearly_et0.entry(row.year).or_insert(0.0) += et0;
                }
                *yearly_precip.entry(row.year).or_insert(0.0) += row.precip_mm;
            }

            if yearly_et0.len() < 2 {
                continue;
            }

            let mean_et0: f64 = yearly_et0.values().sum::<f64>() / yearly_et0.len() as f64;
            let mean_precip: f64 = yearly_precip.values().sum::<f64>() / yearly_precip.len() as f64;

            let bench_decade = bench_decades
                .iter()
                .find(|d| d["decade_start"].as_u64() == Some(u64::from(dec_start)));

            if let Some(bd) = bench_decade {
                let expected_et0 = bd["mean_seasonal_et0_mm"].as_f64().unwrap_or(0.0);
                let expected_precip = bd["mean_seasonal_precip_mm"].as_f64().unwrap_or(0.0);

                let label_et0 = format!("{}_{dec_start}s_mean_et0", station_meta.id);
                v.check_abs(&label_et0, mean_et0, expected_et0, 2.0);
                total_checks += 1;

                let label_precip = format!("{}_{dec_start}s_mean_precip", station_meta.id);
                v.check_abs(&label_precip, mean_precip, expected_precip, 0.01);
                total_checks += 1;
            }
        }

        let bench_trend = bench_station["trend_mm_per_decade"].as_f64().unwrap_or(0.0);

        let mut decade_means: Vec<(f64, f64)> = Vec::new();
        for &(dec_start, dec_end) in DECADES {
            let season_rows: Vec<&DayRow> = rows
                .iter()
                .filter(|r| {
                    r.year >= dec_start
                        && r.year <= dec_end
                        && r.doy >= SEASON_START_DOY
                        && r.doy <= SEASON_END_DOY
                })
                .collect();

            if season_rows.is_empty() {
                continue;
            }

            let mut yearly_et0: BTreeMap<u32, f64> = BTreeMap::new();
            for row in &season_rows {
                let et0 = compute_et0(row, station_meta.lat, station_meta.elevation_m);
                if !et0.is_nan() {
                    *yearly_et0.entry(row.year).or_insert(0.0) += et0;
                }
            }

            if yearly_et0.len() >= 2 {
                let mean = yearly_et0.values().sum::<f64>() / yearly_et0.len() as f64;
                decade_means.push((f64::from(dec_start) + 5.0, mean));
            }
        }

        if decade_means.len() >= 3 {
            #[allow(clippy::suspicious_operation_groupings)]
            let slope = {
                let n = decade_means.len() as f64;
                let sum_x: f64 = decade_means.iter().map(|(x, _)| x).sum();
                let sum_y: f64 = decade_means.iter().map(|(_, y)| y).sum();
                let sum_x_times_y: f64 = decade_means.iter().map(|(x, y)| x * y).sum();
                let sum_xx: f64 = decade_means.iter().map(|(x, _)| x * x).sum();
                // OLS slope: (n·Σxy - Σx·Σy) / (n·Σx² - (Σx)²)
                n.mul_add(sum_x_times_y, -(sum_x * sum_y)) / n.mul_add(sum_xx, -(sum_x * sum_x))
            };
            let trend_mm_per_decade = slope * 10.0;

            let label = format!("{}_trend_mm_decade", station_meta.id);
            v.check_abs(&label, trend_mm_per_decade, bench_trend, 0.5);
            total_checks += 1;
        }
    }

    println!("\n  Total decade checks: {total_checks}");
    v.finish();
}
