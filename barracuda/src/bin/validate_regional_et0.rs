//! Regional ET₀ intercomparison — Rust CPU validation for Experiment 010.
//!
//! Cross-station statistical analysis for Michigan microclimates.
//! Discovers stations from filesystem, computes FAO-56 PM ET₀,
//! validates physical reasonableness, spatial variability, and
//! cross-station temporal correlation.
//!
//! Python control: `control/regional_et0/regional_et0_intercomparison.py`
//! (61/61 PASS)

use std::path::{Path, PathBuf};

use airspring_barracuda::eco::evapotranspiration::{self as et, DailyEt0Input};
use airspring_barracuda::testutil;
use airspring_barracuda::validation::{self, ValidationHarness};

struct StationResult {
    id: String,
    lat: f64,
    n_days: usize,
    seasonal_total: f64,
    daily_mean: f64,
    daily_max: f64,
    daily_min: f64,
    r2_vs_om: f64,
    rmse_vs_om: f64,
    et0_series: Vec<f64>,
}

/// Station metadata discovered from data directory.
struct StationMeta {
    id: String,
    lat: f64,
    elevation_m: f64,
}

fn discover_stations(data_dir: &Path, start: &str, end: &str) -> Vec<StationMeta> {
    let suffix = format!("_{start}_{end}_daily.csv");
    let lats: &[(&str, f64, f64)] = &[
        ("east_lansing", 42.73, 256.0),
        ("grand_junction", 42.38, 213.0),
        ("hart", 43.70, 253.0),
        ("manchester", 42.15, 280.0),
        ("sparta", 43.16, 241.0),
        ("west_olive", 42.92, 190.0),
    ];

    let mut stations = Vec::new();
    if let Ok(entries) = std::fs::read_dir(data_dir) {
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            if let Some(station_id) = name.strip_suffix(&suffix) {
                if station_id == "all_stations" {
                    continue;
                }
                let (lat, elev) = lats
                    .iter()
                    .find(|(id, _, _)| *id == station_id)
                    .map(|(_, lat, elev)| (*lat, *elev))
                    .unwrap_or((42.5, 200.0));

                stations.push(StationMeta {
                    id: station_id.to_string(),
                    lat,
                    elevation_m: elev,
                });
            }
        }
    }
    stations.sort_by(|a, b| a.id.cmp(&b.id));
    stations
}

fn date_to_doy(date_str: &str) -> u32 {
    let parts: Vec<&str> = date_str.split('-').collect();
    if parts.len() != 3 {
        return 1;
    }
    let year: i32 = parts[0].parse().unwrap_or(2023);
    let month: u32 = parts[1].parse().unwrap_or(1);
    let day: u32 = parts[2].parse().unwrap_or(1);

    let month_days: &[u32] = &[0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334];
    let mut doy = month_days.get(month.wrapping_sub(1) as usize).copied().unwrap_or(0) + day;
    if month > 2 && (year % 4 == 0 && (year % 100 != 0 || year % 400 == 0)) {
        doy += 1;
    }
    doy
}

fn load_and_compute(
    data_dir: &Path,
    meta: &StationMeta,
    start: &str,
    end: &str,
) -> Option<StationResult> {
    let csv_path = data_dir.join(format!("{}_{start}_{end}_daily.csv", meta.id));
    let content = std::fs::read_to_string(&csv_path).ok()?;

    let mut lines = content.lines();
    let header_line = lines.next()?;
    let headers: Vec<&str> = header_line.split(',').collect();

    let col_idx = |name: &str| headers.iter().position(|h| *h == name);

    let i_date = col_idx("date")?;
    let i_tmax = col_idx("tmax_c")?;
    let i_tmin = col_idx("tmin_c")?;
    let i_tmean = col_idx("tmean_c")?;
    let i_rh_max = col_idx("rh_max_pct")?;
    let i_rh_min = col_idx("rh_min_pct")?;
    let i_wind = col_idx("wind_2m_m_s")?;
    let i_solar = col_idx("solar_rad_mj_m2")?;
    let i_et0_om = col_idx("et0_openmeteo_mm")?;

    let mut et0_rust = Vec::new();
    let mut et0_om = Vec::new();

    for line in lines {
        let fields: Vec<&str> = line.split(',').collect();
        if fields.len() < headers.len() {
            continue;
        }

        let parse = |idx: usize| fields[idx].parse::<f64>().unwrap_or(f64::NAN);

        let tmax = parse(i_tmax);
        let tmin = parse(i_tmin);
        let tmean = parse(i_tmean);
        let rh_max = parse(i_rh_max);
        let rh_min = parse(i_rh_min);
        let wind = parse(i_wind);
        let solar = parse(i_solar);
        let om_val = parse(i_et0_om);
        let doy = date_to_doy(fields[i_date]);

        if tmax.is_nan() || tmin.is_nan() || solar.is_nan() {
            continue;
        }

        let ea = et::actual_vapour_pressure_rh(tmin, tmax, rh_min, rh_max);
        let input = DailyEt0Input {
            tmin,
            tmax,
            tmean: Some(tmean),
            solar_radiation: solar,
            wind_speed_2m: wind,
            actual_vapour_pressure: ea,
            elevation_m: meta.elevation_m,
            latitude_deg: meta.lat,
            day_of_year: doy,
        };

        let result = et::daily_et0(&input);
        et0_rust.push(result.et0);
        et0_om.push(om_val);
    }

    if et0_rust.is_empty() {
        return None;
    }

    let r2 = testutil::r_squared(&et0_om, &et0_rust).unwrap_or(0.0);
    let rmse = testutil::rmse(&et0_om, &et0_rust);
    let total: f64 = et0_rust.iter().sum();
    let mean = total / et0_rust.len() as f64;
    let max_val = et0_rust.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let min_val = et0_rust.iter().copied().fold(f64::INFINITY, f64::min);

    Some(StationResult {
        id: meta.id.clone(),
        lat: meta.lat,
        n_days: et0_rust.len(),
        seasonal_total: total,
        daily_mean: mean,
        daily_max: max_val,
        daily_min: min_val,
        r2_vs_om: r2,
        rmse_vs_om: rmse,
        et0_series: et0_rust,
    })
}

fn main() {
    let start = std::env::var("AIRSPRING_SEASON_START").unwrap_or_else(|_| "2023-05-01".into());
    let end = std::env::var("AIRSPRING_SEASON_END").unwrap_or_else(|_| "2023-09-30".into());
    let data_dir = std::env::var("AIRSPRING_DATA_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .parent()
                .expect("CARGO_MANIFEST_DIR parent")
                .join("data")
                .join("open_meteo")
        });

    if !data_dir.exists() {
        println!("═══════════════════════════════════════════════════════════");
        println!("  [SKIP] No data at {}", data_dir.display());
        println!("═══════════════════════════════════════════════════════════");
        std::process::exit(0);
    }

    validation::banner("Regional ET₀ Intercomparison (Exp 010)");

    let stations = discover_stations(&data_dir, &start, &end);
    println!(
        "  Discovered {} stations in {}\n",
        stations.len(),
        data_dir.display()
    );

    let mut v = ValidationHarness::new("Regional ET₀ Validation");
    let mut results: Vec<StationResult> = Vec::new();

    for meta in &stations {
        if let Some(sr) = load_and_compute(&data_dir, meta, &start, &end) {
            println!(
                "  {} ({} days): total={:.0} mm, mean={:.2} mm/day, R²={:.4}",
                sr.id, sr.n_days, sr.seasonal_total, sr.daily_mean, sr.r2_vs_om
            );
            results.push(sr);
        }
    }

    if results.is_empty() {
        println!("\n  No stations processed — check data directory");
        v.check_bool("At least one station processed", false);
        v.finish();
    }

    // ── Physical reasonableness ─────────────────────────────────
    println!();
    validation::section("Physical reasonableness");
    for sr in &results {
        v.check_bool(
            &format!("{} daily mean {:.2} in [2, 6] mm/day", sr.id, sr.daily_mean),
            (2.0..=6.0).contains(&sr.daily_mean),
        );
        v.check_bool(
            &format!("{} daily max {:.2} in [3, 10] mm/day", sr.id, sr.daily_max),
            (3.0..=10.0).contains(&sr.daily_max),
        );
        v.check_bool(
            &format!("{} daily min {:.2} >= 0", sr.id, sr.daily_min),
            sr.daily_min >= 0.0,
        );
    }

    // ── Seasonal totals ─────────────────────────────────────────
    println!();
    validation::section("Seasonal totals");
    for sr in &results {
        v.check_bool(
            &format!("{} season total {:.0} in [350, 750] mm", sr.id, sr.seasonal_total),
            (350.0..=750.0).contains(&sr.seasonal_total),
        );
    }

    // ── FAO-56 PM vs Open-Meteo ERA5 ───────────────────────────
    println!();
    validation::section("FAO-56 PM vs Open-Meteo ERA5 correlation");
    for sr in &results {
        v.check_bool(
            &format!("{} R² {:.4} > 0.85", sr.id, sr.r2_vs_om),
            sr.r2_vs_om > 0.85,
        );
        v.check_bool(
            &format!("{} RMSE {:.3} < 2.0 mm/day", sr.id, sr.rmse_vs_om),
            sr.rmse_vs_om < 2.0,
        );
    }

    // ── Spatial variability ─────────────────────────────────────
    println!();
    validation::section("Spatial variability");
    let all_means: Vec<f64> = results.iter().map(|sr| sr.daily_mean).collect();
    let all_totals: Vec<f64> = results.iter().map(|sr| sr.seasonal_total).collect();
    let grand_mean = all_means.iter().sum::<f64>() / all_means.len() as f64;
    let std_dev = (all_means
        .iter()
        .map(|x| (x - grand_mean).powi(2))
        .sum::<f64>()
        / all_means.len() as f64)
        .sqrt();
    let cv_pct = 100.0 * std_dev / grand_mean;
    let spread = all_totals.iter().copied().fold(f64::NEG_INFINITY, f64::max)
        - all_totals.iter().copied().fold(f64::INFINITY, f64::min);

    println!("  Grand mean: {grand_mean:.2} mm/day, CV: {cv_pct:.1}%, Spread: {spread:.0} mm");

    v.check_bool(
        &format!("CV of daily means {cv_pct:.1}% in [1, 30]"),
        (1.0..=30.0).contains(&cv_pct),
    );
    v.check_bool(
        &format!("Season total spread {spread:.0} mm in [20, 250]"),
        (20.0..=250.0).contains(&spread),
    );
    v.check_bool(
        &format!("Processed {} stations", results.len()),
        results.len() >= 6,
    );

    // ── Geographic consistency ───────────────────────────────────
    println!();
    validation::section("Geographic consistency");
    let lat_min = results.iter().map(|sr| sr.lat).fold(f64::INFINITY, f64::min);
    let lat_max = results
        .iter()
        .map(|sr| sr.lat)
        .fold(f64::NEG_INFINITY, f64::max);
    let lat_range = lat_max - lat_min;

    v.check_bool(
        &format!("Latitude span {lat_range:.2}° in [0.5, 5.0]"),
        (0.5..=5.0).contains(&lat_range),
    );

    for sr in &results {
        v.check_bool(
            &format!("{} lat {:.2} in Michigan [41, 47]", sr.id, sr.lat),
            (41.0..=47.0).contains(&sr.lat),
        );
    }

    // ── Cross-station temporal correlation ───────────────────────
    println!();
    validation::section("Cross-station temporal correlation");
    let min_days = results.iter().map(|sr| sr.n_days).min().unwrap_or(0);
    for i in 0..results.len() {
        for j in (i + 1)..results.len() {
            let s1 = &results[i].et0_series[..min_days];
            let s2 = &results[j].et0_series[..min_days];
            let r = testutil::pearson_r(s1, s2);
            v.check_bool(
                &format!(
                    "r({}, {}) = {r:.3} > 0.70",
                    results[i].id, results[j].id
                ),
                r > 0.70,
            );
        }
    }

    v.finish();
}
