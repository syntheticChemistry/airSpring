// SPDX-License-Identifier: AGPL-3.0-or-later
#![warn(clippy::pedantic)]
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
//! Validate `IoT` time series parser and statistics.
//!
//! Uses deterministic synthetic agricultural sensor data to validate:
//! - CSV streaming parser (columnar storage)
//! - Column statistics (mean, std, min, max)
//! - Round-trip fidelity (generate → write → parse → compare)
//!
//! Provenance: Synthetic data from `testutil::generate_synthetic_iot_data(168)`
//! with analytically known properties (deterministic, seed-free — same output
//! on every run). No Python baseline: this validates the Rust CSV parser and
//! statistics engine against known mathematical properties of the generator.
//!
//! Generator properties (from `testutil/generators.rs`):
//! - Temperature: 25 ± 8 °C sinusoidal diurnal → mean ≈ 25.0, min ≈ 17.0, max ≈ 33.0
//! - PAR: bell curve peaking at ~1800 µmol/m²/s during midday, 0 at night
//! - Soil moisture: base 0.25 m³/m³ with ±0.05 diurnal variation

use airspring_barracuda::io::csv_ts;
use airspring_barracuda::testutil::generate_synthetic_iot_data;
use airspring_barracuda::tolerances;
use airspring_barracuda::validation::{self, ValidationHarness};
use std::io::Write;

/// Valid range for synthetic soil moisture sensor 1 (m³/m³).
const SM1_VALID_MIN: f64 = 0.09;
const SM1_VALID_MAX: f64 = 0.41;

/// Validate sensor column statistics against analytically known properties.
fn validate_sensor_stats(
    v: &mut ValidationHarness,
    data: &csv_ts::TimeseriesData,
) -> csv_ts::ColumnStats {
    validation::section("Temperature statistics (analytical: 25 ± 8 °C diurnal)");

    let temp_stats = data
        .column_stats("temperature")
        .expect("synthetic data must have 'temperature' column");
    println!(
        "  Mean: {:.1}°C, StdDev: {:.1}, Range: {:.1}–{:.1}",
        temp_stats.mean, temp_stats.std_dev, temp_stats.min, temp_stats.max
    );

    v.check_abs(
        "Temp mean ≈ 25°C",
        temp_stats.mean,
        25.0,
        tolerances::IOT_TEMPERATURE_MEAN.abs_tol,
    );
    v.check_abs(
        "Temp min ≈ 17°C",
        temp_stats.min,
        17.0,
        tolerances::IOT_TEMPERATURE_EXTREMES.abs_tol,
    );
    v.check_abs(
        "Temp max ≈ 33°C",
        temp_stats.max,
        33.0,
        tolerances::IOT_TEMPERATURE_EXTREMES.abs_tol,
    );

    println!();
    validation::section("Soil moisture statistics");

    let sm_stats = data
        .column_stats("soil_moisture_1")
        .expect("synthetic data must have 'soil_moisture_1' column");
    println!(
        "  Mean: {:.3} m³/m³, Range: {:.3}–{:.3}",
        sm_stats.mean, sm_stats.min, sm_stats.max
    );

    v.check_bool(
        &format!("SM1 in valid range [{SM1_VALID_MIN}, {SM1_VALID_MAX}]"),
        sm_stats.min >= SM1_VALID_MIN && sm_stats.max <= SM1_VALID_MAX,
    );

    println!();
    validation::section("PAR statistics (bell curve, max ≈ 1800 µmol/m²/s)");

    let par_stats = data
        .column_stats("par")
        .expect("synthetic data must have 'par' column");
    println!(
        "  Mean: {:.0} µmol/m²/s, Max: {:.0}",
        par_stats.mean, par_stats.max
    );

    v.check_abs(
        "PAR max ≈ 1800",
        par_stats.max,
        1800.0,
        tolerances::IOT_PAR_MAX.abs_tol,
    );
    v.check_bool("PAR has zero (nighttime)", par_stats.min < 1.0);

    temp_stats
}

/// Write synthetic data to CSV and parse it back, checking round-trip fidelity.
fn validate_csv_round_trip(
    v: &mut ValidationHarness,
    data: &csv_ts::TimeseriesData,
    expected_temp_mean: f64,
) {
    println!();
    validation::section("CSV round-trip (generate → write → stream-parse → compare)");

    let tmp_path = std::env::temp_dir().join("iot_csv_roundtrip_test.csv");
    {
        let mut f =
            std::fs::File::create(&tmp_path).expect("failed to create temp CSV for round-trip");
        writeln!(
            f,
            "timestamp,soil_moisture_1,soil_moisture_2,temperature,humidity,par"
        )
        .expect("failed to write CSV header");
        let sm1 = data.column("soil_moisture_1").expect("sm1 column");
        let sm2 = data.column("soil_moisture_2").expect("sm2 column");
        let temp = data.column("temperature").expect("temperature column");
        let hum = data.column("humidity").expect("humidity column");
        let par = data.column("par").expect("par column");
        for i in 0..data.len() {
            writeln!(
                f,
                "{},{:.4},{:.4},{:.2},{:.2},{:.1}",
                data.timestamps()[i],
                sm1[i],
                sm2[i],
                temp[i],
                hum[i],
                par[i],
            )
            .expect("failed to write CSV row");
        }
    }

    match csv_ts::parse_csv(&tmp_path, Some("timestamp")) {
        Ok(parsed) => {
            let parsed_n = parsed.len() as f64;
            let parsed_cols = parsed.num_columns() as f64;
            v.check_abs("Parsed record count", parsed_n, 168.0, f64::EPSILON);
            v.check_abs("Parsed column count", parsed_cols, 5.0, f64::EPSILON);
            let parsed_temp = parsed
                .column_stats("temperature")
                .expect("parsed data must have 'temperature' column");
            v.check_abs(
                "Round-trip temp mean",
                parsed_temp.mean,
                expected_temp_mean,
                tolerances::IOT_CSV_ROUNDTRIP.abs_tol,
            );
        }
        Err(e) => {
            println!("  FAILED: {e}");
            v.check_abs("Parsed record count", 0.0, 168.0, f64::EPSILON);
            v.check_abs("Parsed column count", 0.0, 5.0, f64::EPSILON);
            v.check_abs("Round-trip temp mean", 0.0, 25.0, f64::EPSILON);
        }
    }

    let _ = std::fs::remove_file(&tmp_path);
}

fn main() {
    validation::init_tracing();
    validation::banner("IoT Time Series Validation");
    let mut v = ValidationHarness::new("IoT Time Series Validation");

    validation::section("Synthetic sensor data (deterministic, known properties)");
    let data = generate_synthetic_iot_data(168); // 7 days hourly

    let record_count = data.len() as f64;
    let column_count = data.num_columns() as f64;
    println!("  Generated {record_count:.0} records, {column_count:.0} columns");

    v.check_abs("Record count", record_count, 168.0, f64::EPSILON);
    v.check_abs("Column count", column_count, 5.0, f64::EPSILON);

    println!();
    let temp_stats = validate_sensor_stats(&mut v, &data);
    validate_csv_round_trip(&mut v, &data, temp_stats.mean);

    v.finish();
}
