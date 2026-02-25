//! Real data validation: compute ET₀ on Open-Meteo Michigan weather.
//!
//! Replaces `control/fao56/compute_et0_real_data.py` + `simulate_real_data.py`.
//! Reads actual CSV data, computes ET₀ using Rust, compares with Open-Meteo ET₀,
//! and runs water balance simulations on real weather.
//!
//! This is the key proof that the Rust pipeline works on real, non-synthetic data.
//! Uses `ValidationRunner` for hotSpring-pattern pass/fail and exit codes.

use airspring_barracuda::eco::{
    evapotranspiration::{self as et, DailyEt0Input},
    water_balance::{DailyInput, WaterBalanceState},
};
use airspring_barracuda::testutil;
use airspring_barracuda::validation::{self, ValidationHarness};
use std::io::BufRead;
use std::path::Path;

// ── Scenarios loaded from benchmark JSON (single source of truth) ────

const BENCHMARK_WB: &str =
    include_str!("../../../control/water_balance/benchmark_water_balance.json");

/// A water balance scenario loaded from `benchmark_water_balance.json`.
struct Scenario {
    name: String,
    station: String,
    crop: String,
    kc: f64,
    theta_fc: f64,
    theta_wp: f64,
    root_depth_mm: f64,
    p: f64,
    irrig_depth_mm: f64,
}

fn load_scenarios() -> Vec<Scenario> {
    let bm: serde_json::Value =
        serde_json::from_str(BENCHMARK_WB).expect("benchmark_water_balance.json must parse");
    let arr = bm["real_data_scenarios"]["scenarios"]
        .as_array()
        .expect("real_data_scenarios.scenarios must be an array");

    arr.iter()
        .map(|s| {
            let f = |key| validation::json_f64(s, &[key]).expect(key);
            Scenario {
                name: s["name"].as_str().expect("name").to_string(),
                station: s["station"].as_str().expect("station").to_string(),
                crop: s["crop"].as_str().expect("crop").to_string(),
                kc: f("kc"),
                theta_fc: f("theta_fc"),
                theta_wp: f("theta_wp"),
                root_depth_mm: f("root_depth_mm"),
                p: f("p"),
                irrig_depth_mm: f("irrig_depth_mm"),
            }
        })
        .collect()
}

/// Stations whose data we validate.
const STATIONS: &[&str] = &[
    "east_lansing",
    "grand_junction",
    "hart",
    "manchester",
    "sparta",
    "west_olive",
];

/// Mass balance tolerance (mm) — FAO-56 Chapter 8 conservation law.
const MASS_BALANCE_TOLERANCE: f64 = 0.01;

/// Minimum acceptable R² for ET₀ vs Open-Meteo.
/// Justified: Open-Meteo uses ERA5 reanalysis which differs systematically from
/// point-based FAO-56 PM (different radiation, wind models). R²>0.90 confirms
/// our implementation is sound while acknowledging methodological differences.
const MIN_R2: f64 = 0.90;

/// Maximum acceptable RMSE (mm/day) for ET₀ vs Open-Meteo.
/// Justified: ~1 mm/day is typical for FAO-56 PM vs reanalysis comparisons
/// (Allen et al., 2005). We allow 1.5 to account for Michigan's variable climate.
const MAX_RMSE: f64 = 1.5;

/// One row of Open-Meteo daily weather data.
struct WeatherRow {
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
    day_of_year: u32,
}

fn parse_open_meteo_csv(path: &Path) -> Vec<WeatherRow> {
    let file = std::fs::File::open(path).expect("Cannot open CSV — verify data/ exists");
    let reader = std::io::BufReader::new(file);
    let mut rows = Vec::new();
    let mut header_map: Vec<String> = Vec::new();

    for (i, line) in reader.lines().enumerate() {
        let line = line.expect("I/O error reading CSV line");
        if i == 0 {
            header_map = line.split(',').map(String::from).collect();
            continue;
        }
        let fields: Vec<&str> = line.split(',').collect();
        if fields.len() < header_map.len() {
            continue;
        }

        let col = |name: &str| -> f64 {
            let idx = header_map
                .iter()
                .position(|h| h == name)
                .expect("Open-Meteo CSV missing required column");
            fields[idx].parse::<f64>().unwrap_or(f64::NAN)
        };

        let date_idx = header_map
            .iter()
            .position(|h| h == "date")
            .expect("Open-Meteo CSV must have a 'date' column");
        let doy = date_to_doy(fields[date_idx]);

        rows.push(WeatherRow {
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
            day_of_year: doy,
        });
    }
    rows
}

fn date_to_doy(date: &str) -> u32 {
    let parts: Vec<&str> = date.split('-').collect();
    if parts.len() != 3 {
        return 1;
    }
    let year: u32 = parts[0].parse().unwrap_or(2023);
    let month: u32 = parts[1].parse().unwrap_or(1);
    let day: u32 = parts[2].parse().unwrap_or(1);

    let is_leap = year.is_multiple_of(4) && !year.is_multiple_of(100) || year.is_multiple_of(400);
    let days_before_month = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334];
    let mut doy = days_before_month[(month - 1).min(11) as usize] + day;
    if is_leap && month > 2 {
        doy += 1;
    }
    doy
}

fn compute_et0_for_row(row: &WeatherRow) -> f64 {
    let ea = et::actual_vapour_pressure_rh(row.tmin_c, row.tmax_c, row.rh_min_pct, row.rh_max_pct);
    let input = DailyEt0Input {
        tmin: row.tmin_c,
        tmax: row.tmax_c,
        tmean: Some(row.tmean_c),
        solar_radiation: row.solar_rad_mj_m2,
        wind_speed_2m: row.wind_2m_m_s,
        actual_vapour_pressure: ea,
        elevation_m: row.elevation_m,
        latitude_deg: row.lat,
        day_of_year: row.day_of_year,
    };
    et::daily_et0(&input).et0
}

/// Validate ET₀ for a single station; accumulate Rust/Open-Meteo series.
fn validate_station_et0(
    station: &str,
    data_dir: &Path,
    v: &mut ValidationHarness,
    all_rust: &mut Vec<f64>,
    all_om: &mut Vec<f64>,
) {
    let csv_path = data_dir.join(format!("{station}_2023-05-01_2023-09-30_daily.csv"));
    if !csv_path.exists() {
        println!("  [SKIP] {station}: CSV not found");
        return;
    }

    let rows = parse_open_meteo_csv(&csv_path);
    let valid_rows: Vec<&WeatherRow> = rows
        .iter()
        .filter(|r| !r.et0_openmeteo_mm.is_nan() && r.et0_openmeteo_mm > 0.0)
        .collect();

    if valid_rows.is_empty() {
        println!("  [SKIP] {station}: no valid rows");
        return;
    }

    let rust_et0: Vec<f64> = valid_rows.iter().map(|r| compute_et0_for_row(r)).collect();
    let om_et0: Vec<f64> = valid_rows.iter().map(|r| r.et0_openmeteo_mm).collect();

    let rmse = testutil::rmse(&om_et0, &rust_et0);
    let r2 = testutil::r_squared(&om_et0, &rust_et0).unwrap_or(0.0);

    v.check_bool(
        &format!(
            "{station} R²>{MIN_R2} ({} days, R²={r2:.3})",
            valid_rows.len()
        ),
        r2 > MIN_R2,
    );
    v.check_bool(
        &format!("{station} RMSE<{MAX_RMSE} (RMSE={rmse:.3})"),
        rmse < MAX_RMSE,
    );

    all_rust.extend_from_slice(&rust_et0);
    all_om.extend_from_slice(&om_et0);
}

/// Validate water balance for a single crop scenario (rainfed + irrigated).
fn validate_scenario(scenario: &Scenario, data_dir: &Path, v: &mut ValidationHarness) {
    use airspring_barracuda::eco::water_balance;

    let csv_path = data_dir.join(format!(
        "{}_2023-05-01_2023-09-30_daily.csv",
        scenario.station
    ));
    if !csv_path.exists() {
        println!(
            "  [SKIP] {} @ {}: CSV not found",
            scenario.crop, scenario.station
        );
        return;
    }

    let rows = parse_open_meteo_csv(&csv_path);
    let valid: Vec<&WeatherRow> = rows
        .iter()
        .filter(|r| !r.solar_rad_mj_m2.is_nan())
        .collect();

    if valid.is_empty() {
        println!(
            "  [SKIP] {} @ {}: no valid rows",
            scenario.crop, scenario.station
        );
        return;
    }

    let et0_series: Vec<f64> = valid.iter().map(|r| compute_et0_for_row(r)).collect();
    let precip_series: Vec<f64> = valid.iter().map(|r| r.precip_mm).collect();

    println!(
        "\n  ── {} ({}) @ {} ({} days) ──",
        scenario.name,
        scenario.crop,
        scenario.station,
        valid.len()
    );

    let initial = WaterBalanceState::new(
        scenario.theta_fc,
        scenario.theta_wp,
        scenario.root_depth_mm,
        scenario.p,
    );

    // Rainfed (no irrigation)
    let rainfed_inputs: Vec<DailyInput> = et0_series
        .iter()
        .zip(&precip_series)
        .map(|(&et0, &precip)| DailyInput {
            precipitation: precip,
            irrigation: 0.0,
            et0,
            kc: scenario.kc,
        })
        .collect();

    let (rf_final, rf_out) = water_balance::simulate_season(&initial, &rainfed_inputs);
    let rf_mb = water_balance::mass_balance_check(
        &rainfed_inputs,
        &rf_out,
        initial.depletion,
        rf_final.depletion,
    );

    v.check_bool(
        &format!(
            "{} rainfed mass balance < {MASS_BALANCE_TOLERANCE} (err={rf_mb:.6})",
            scenario.name
        ),
        rf_mb < MASS_BALANCE_TOLERANCE,
    );

    // Irrigated (smart trigger)
    let (irr_mb, irr_et, total_irrig, irr_stress) =
        run_irrigated(scenario, &initial, &et0_series, &precip_series);

    v.check_bool(
        &format!(
            "{} irrigated mass balance < {MASS_BALANCE_TOLERANCE} (err={irr_mb:.6})",
            scenario.name
        ),
        irr_mb < MASS_BALANCE_TOLERANCE,
    );

    // Summary
    let rf_et: f64 = rf_out.iter().map(|o| o.actual_et).sum();
    let rf_precip: f64 = precip_series.iter().sum();
    let rf_stress = rf_out.iter().filter(|o| o.ks < 1.0).count();

    println!("    Rainfed:   ET={rf_et:.0}mm, P={rf_precip:.0}mm, stress={rf_stress}d");
    println!("    Irrigated: ET={irr_et:.0}mm, irrig={total_irrig:.0}mm, stress={irr_stress}d");
}

/// Run irrigated simulation: returns `(mb_error, total_et, total_irrig, stress_days)`.
fn run_irrigated(
    scenario: &Scenario,
    initial: &WaterBalanceState,
    et0_series: &[f64],
    precip_series: &[f64],
) -> (f64, f64, f64, usize) {
    use airspring_barracuda::eco::water_balance;

    let mut irr_state = initial.clone();
    let mut irr_inputs = Vec::with_capacity(et0_series.len());
    let mut irr_outputs = Vec::with_capacity(et0_series.len());
    let mut total_irrig = 0.0_f64;

    for (&et0, &precip) in et0_series.iter().zip(precip_series) {
        let irr_today = if irr_state.depletion > irr_state.raw {
            scenario.irrig_depth_mm
        } else {
            0.0
        };
        total_irrig += irr_today;

        let input = DailyInput {
            precipitation: precip,
            irrigation: irr_today,
            et0,
            kc: scenario.kc,
        };
        let output = irr_state.step(&input);
        irr_inputs.push(input);
        irr_outputs.push(output);
    }

    let mb = water_balance::mass_balance_check(
        &irr_inputs,
        &irr_outputs,
        initial.depletion,
        irr_state.depletion,
    );

    let total_et: f64 = irr_outputs.iter().map(|o| o.actual_et).sum();
    let stress = irr_outputs.iter().filter(|o| o.ks < 1.0).count();

    (mb, total_et, total_irrig, stress)
}

fn main() {
    let data_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("CARGO_MANIFEST_DIR must have a parent directory")
        .join("data")
        .join("open_meteo");

    if !data_dir.exists() {
        println!("═══════════════════════════════════════════════════════════");
        println!("  [SKIP] No real data at {}", data_dir.display());
        println!(
            "  Run: python scripts/download_open_meteo.py --all-stations --growing-season 2023"
        );
        println!("═══════════════════════════════════════════════════════════");
        std::process::exit(0);
    }

    validation::banner("Real Data Validation");
    let mut v = ValidationHarness::new("Real Data Validation");

    let mut all_rust_et0 = Vec::new();
    let mut all_om_et0 = Vec::new();

    // Per-station ET₀ validation
    validation::section("Per-Station ET₀ (Rust vs Open-Meteo)");
    for station in STATIONS {
        validate_station_et0(
            station,
            &data_dir,
            &mut v,
            &mut all_rust_et0,
            &mut all_om_et0,
        );
    }

    // Overall ET₀
    if !all_rust_et0.is_empty() {
        validation::section("Overall ET₀");
        let overall_r2 = testutil::r_squared(&all_om_et0, &all_rust_et0).unwrap_or(0.0);
        let overall_rmse = testutil::rmse(&all_om_et0, &all_rust_et0);
        let overall_mbe = testutil::mbe(&all_om_et0, &all_rust_et0);
        let overall_ia = testutil::index_of_agreement(&all_om_et0, &all_rust_et0);

        println!(
            "  {} station-days: R²={overall_r2:.4}, RMSE={overall_rmse:.4}, MBE={overall_mbe:.4}, IA={overall_ia:.4}",
            all_rust_et0.len()
        );

        v.check_bool(
            &format!("Overall R²>{MIN_R2} (R²={overall_r2:.4})"),
            overall_r2 > MIN_R2,
        );
    }

    // Water balance: crop scenarios loaded from benchmark JSON
    let scenarios = load_scenarios();
    validation::section("Water Balance Scenarios");
    for scenario in &scenarios {
        validate_scenario(scenario, &data_dir, &mut v);
    }

    v.finish();
}
