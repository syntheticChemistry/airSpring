//! Real data validation: compute ET₀ on Open-Meteo Michigan weather.
//!
//! Replaces `control/fao56/compute_et0_real_data.py` + `simulate_real_data.py`.
//! Reads actual CSV data, computes ET₀ using Rust, compares with Open-Meteo ET₀,
//! and runs water balance simulations on real weather.
//!
//! This is the key proof that the Rust pipeline works on real, non-synthetic data.

use airspring_barracuda::eco::{
    evapotranspiration::{self as et, DailyEt0Input},
    water_balance::{DailyInput, WaterBalanceState},
};
use airspring_barracuda::testutil;
use std::io::BufRead;
use std::path::Path;

// ── Crop/soil scenarios from Dong et al. (2024) & FAO-56 ────────────

/// A water balance scenario matching Python `simulate_real_data.py` SCENARIOS.
struct Scenario {
    name: &'static str,
    station: &'static str,
    crop: &'static str,
    kc: f64,
    theta_fc: f64,
    theta_wp: f64,
    root_depth_mm: f64,
    p: f64,
    irrig_depth_mm: f64,
}

const SCENARIOS: &[Scenario] = &[
    Scenario {
        name: "blueberry_west_olive",
        station: "west_olive",
        crop: "Blueberry",
        kc: 0.85,
        theta_fc: 0.30,
        theta_wp: 0.12,
        root_depth_mm: 400.0,
        p: 0.50,
        irrig_depth_mm: 15.0,
    },
    Scenario {
        name: "tomato_hart",
        station: "hart",
        crop: "Tomato",
        kc: 1.05,
        theta_fc: 0.36,
        theta_wp: 0.15,
        root_depth_mm: 600.0,
        p: 0.40,
        irrig_depth_mm: 25.0,
    },
    Scenario {
        name: "corn_manchester",
        station: "manchester",
        crop: "Corn",
        kc: 1.15,
        theta_fc: 0.33,
        theta_wp: 0.13,
        root_depth_mm: 800.0,
        p: 0.55,
        irrig_depth_mm: 30.0,
    },
    Scenario {
        name: "reference_east_lansing",
        station: "east_lansing",
        crop: "Reference grass",
        kc: 1.00,
        theta_fc: 0.32,
        theta_wp: 0.14,
        root_depth_mm: 500.0,
        p: 0.50,
        irrig_depth_mm: 25.0,
    },
];

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
    let file = std::fs::File::open(path).expect("Cannot open CSV");
    let reader = std::io::BufReader::new(file);
    let mut rows = Vec::new();

    let mut header_map: Vec<String> = Vec::new();

    for (i, line) in reader.lines().enumerate() {
        let line = line.expect("Cannot read line");
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
                .unwrap_or_else(|| panic!("Missing column: {name}"));
            fields[idx].parse::<f64>().unwrap_or(f64::NAN)
        };

        // Parse date for DOY
        let date_str = fields[header_map.iter().position(|h| h == "date").unwrap()];
        let doy = date_to_doy(date_str);

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
    // "2023-05-01" → DOY
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
    // Estimate actual vapour pressure from RH (FAO-56 Eq. 17)
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

#[allow(clippy::cast_precision_loss, clippy::too_many_lines)]
fn main() {
    println!("═══════════════════════════════════════════════════════════");
    println!("  airSpring Real Data Validation (Rust)");
    println!("═══════════════════════════════════════════════════════════\n");

    let data_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("data")
        .join("open_meteo");

    if !data_dir.exists() {
        println!("  [SKIP] No real data at {}", data_dir.display());
        println!(
            "  Run: python scripts/download_open_meteo.py --all-stations --growing-season 2023"
        );
        println!("\n═══════════════════════════════════════════════════════════");
        return;
    }

    let stations = [
        "east_lansing",
        "grand_junction",
        "hart",
        "manchester",
        "sparta",
        "west_olive",
    ];

    let mut total_days = 0usize;
    let mut all_rust_et0 = Vec::new();
    let mut all_om_et0 = Vec::new();
    let mut passed = 0u32;
    let mut failed = 0u32;

    for station in &stations {
        let csv_path = data_dir.join(format!("{station}_2023-05-01_2023-09-30_daily.csv"));
        if !csv_path.exists() {
            println!("  [SKIP] {station}: CSV not found");
            continue;
        }

        let rows = parse_open_meteo_csv(&csv_path);
        let valid_rows: Vec<&WeatherRow> = rows
            .iter()
            .filter(|r| !r.et0_openmeteo_mm.is_nan() && r.et0_openmeteo_mm > 0.0)
            .collect();

        if valid_rows.is_empty() {
            println!("  [SKIP] {station}: no valid rows");
            continue;
        }

        let rust_et0: Vec<f64> = valid_rows.iter().map(|r| compute_et0_for_row(r)).collect();
        let om_et0: Vec<f64> = valid_rows.iter().map(|r| r.et0_openmeteo_mm).collect();

        let rmse = testutil::rmse(&om_et0, &rust_et0);
        let r2 = testutil::r_squared(&om_et0, &rust_et0).unwrap_or(0.0);
        let mbe = testutil::mbe(&om_et0, &rust_et0);
        let ia = testutil::index_of_agreement(&om_et0, &rust_et0);

        let r2_pass = r2 > 0.90;
        let rmse_pass = rmse < 1.5;
        let status = if r2_pass && rmse_pass { "PASS" } else { "FAIL" };
        if r2_pass && rmse_pass {
            passed += 1;
        } else {
            failed += 1;
        }

        println!(
            "  [{status}] {station}: {n} days, R²={r2:.3}, RMSE={rmse:.3}, MBE={mbe:.3}, IA={ia:.3}",
            n = valid_rows.len()
        );

        total_days += valid_rows.len();
        all_rust_et0.extend_from_slice(&rust_et0);
        all_om_et0.extend_from_slice(&om_et0);
    }

    // Overall stats
    if !all_rust_et0.is_empty() {
        println!("\n  --- Overall ({total_days} station-days) ---");
        let overall_r2 = testutil::r_squared(&all_om_et0, &all_rust_et0).unwrap_or(0.0);
        let overall_rmse = testutil::rmse(&all_om_et0, &all_rust_et0);
        let overall_mbe = testutil::mbe(&all_om_et0, &all_rust_et0);
        let overall_ia = testutil::index_of_agreement(&all_om_et0, &all_rust_et0);

        println!("  R²    = {overall_r2:.4}");
        println!("  RMSE  = {overall_rmse:.4} mm/day");
        println!("  MBE   = {overall_mbe:.4} mm/day");
        println!("  IA    = {overall_ia:.4}");

        if overall_r2 > 0.90 {
            passed += 1;
            println!("  [PASS] Overall R² > 0.90");
        } else {
            failed += 1;
            println!("  [FAIL] Overall R² = {overall_r2:.4} (need > 0.90)");
        }

        // ── Water balance: 4 crop scenarios (rainfed + irrigated) ────
        println!("\n  ╌╌╌ Water Balance Scenarios ╌╌╌");
        for scenario in SCENARIOS {
            let csv_path = data_dir.join(format!(
                "{}_2023-05-01_2023-09-30_daily.csv",
                scenario.station
            ));
            if !csv_path.exists() {
                println!(
                    "  [SKIP] {} @ {}: CSV not found",
                    scenario.crop, scenario.station
                );
                continue;
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
                continue;
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
            let taw = airspring_barracuda::eco::water_balance::total_available_water(
                scenario.theta_fc,
                scenario.theta_wp,
                scenario.root_depth_mm,
            );
            let raw =
                airspring_barracuda::eco::water_balance::readily_available_water(taw, scenario.p);
            println!("  TAW: {taw:.1} mm, RAW: {raw:.1} mm");

            let initial = WaterBalanceState::new(
                scenario.theta_fc,
                scenario.theta_wp,
                scenario.root_depth_mm,
                scenario.p,
            );

            // ── Rainfed (no irrigation) ──────────────────────────────
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

            let (rf_final, rf_out) =
                airspring_barracuda::eco::water_balance::simulate_season(&initial, &rainfed_inputs);
            let rf_mb = airspring_barracuda::eco::water_balance::mass_balance_check(
                &rainfed_inputs,
                &rf_out,
                initial.depletion,
                rf_final.depletion,
            );

            let rf_et: f64 = rf_out.iter().map(|o| o.actual_et).sum();
            let rf_precip: f64 = precip_series.iter().sum();
            let rf_stress = rf_out.iter().filter(|o| o.ks < 1.0).count();
            let rf_min_ks = rf_out.iter().map(|o| o.ks).fold(f64::INFINITY, f64::min);

            println!("  Rainfed:");
            println!("    Total ET:      {rf_et:.1} mm");
            println!("    Total precip:  {rf_precip:.1} mm");
            println!("    Days stressed: {rf_stress}/{}", valid.len());
            println!("    Min Ks:        {rf_min_ks:.3}");
            println!("    Mass balance:  {rf_mb:.6} mm");

            if rf_mb < 0.01 {
                passed += 1;
                println!("    [PASS] rainfed mass balance < 0.01");
            } else {
                failed += 1;
                println!("    [FAIL] rainfed mass balance = {rf_mb:.6}");
            }

            // ── Irrigated (smart trigger) ────────────────────────────
            // Run day-by-day so we can inject irrigation when triggered.
            let mut irr_state = initial.clone();
            let mut irr_out = Vec::with_capacity(valid.len());
            let mut total_irrig = 0.0_f64;
            let mut irrig_events = 0u32;

            for (&et0, &precip) in et0_series.iter().zip(&precip_series) {
                // Apply irrigation *before* step if the trigger is set
                let irr_today = if irr_state.depletion > irr_state.raw {
                    irrig_events += 1;
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
                irr_out.push((input, output));
            }

            let irr_inputs: Vec<DailyInput> = irr_out.iter().map(|(inp, _)| inp.clone()).collect();
            let irr_outputs: Vec<airspring_barracuda::eco::water_balance::DailyOutput> =
                irr_out.iter().map(|(_, out)| out.clone()).collect();
            let irr_mb = airspring_barracuda::eco::water_balance::mass_balance_check(
                &irr_inputs,
                &irr_outputs,
                initial.depletion,
                irr_state.depletion,
            );

            let irr_et: f64 = irr_outputs.iter().map(|o| o.actual_et).sum();
            let irr_stress = irr_outputs.iter().filter(|o| o.ks < 1.0).count();

            println!("  Irrigated:");
            println!("    Total ET:      {irr_et:.1} mm");
            println!("    Total irrig:   {total_irrig:.1} mm ({irrig_events} events)");
            println!("    Total precip:  {rf_precip:.1} mm");
            println!("    Days stressed: {irr_stress}/{}", valid.len());
            println!("    Mass balance:  {irr_mb:.6} mm");

            if irr_mb < 0.01 {
                passed += 1;
                println!("    [PASS] irrigated mass balance < 0.01");
            } else {
                failed += 1;
                println!("    [FAIL] irrigated mass balance = {irr_mb:.6}");
            }

            // ── Water savings vs naive (25mm every 5 days) ───────────
            #[allow(clippy::cast_precision_loss)]
            let naive_irrig = (valid.len() / 5) as f64 * 25.0;
            if naive_irrig > 0.0 {
                let savings = (naive_irrig - total_irrig) / naive_irrig * 100.0;
                println!("  Water savings vs naive (25mm/5d):");
                println!(
                    "    Naive: {naive_irrig:.0} mm, Smart: {total_irrig:.0} mm → {savings:.1}%"
                );
            }
        }
    }

    println!("\n═══════════════════════════════════════════════════════════");
    println!(
        "  Real Data Validation: {passed}/{total} checks passed",
        total = passed + failed
    );
    println!("  RESULT: {}", if failed == 0 { "PASS" } else { "FAIL" });
    println!("═══════════════════════════════════════════════════════════");
}
