//! Validate `IoT` time series parser and statistics.
//!
//! Uses deterministic synthetic agricultural sensor data to validate:
//! - CSV streaming parser (columnar storage)
//! - Column statistics (mean, std, min, max)
//! - Round-trip fidelity (generate → write → parse → compare)
//!
//! Provenance: Synthetic data with analytically known properties.
//! Temperature: 25 ± 8 °C diurnal cycle → mean = 25.0, min = 17.0, max = 33.0.

use airspring_barracuda::io::csv_ts;
use airspring_barracuda::testutil::generate_synthetic_iot_data;
use airspring_barracuda::validation::ValidationRunner;
use std::io::Write;

#[allow(clippy::cast_precision_loss)]
fn main() {
    let mut v = ValidationRunner::new("IoT Time Series Validation");

    // ── Synthetic data generation ────────────────────────────────
    v.section("Synthetic sensor data (deterministic, known properties)");

    let data = generate_synthetic_iot_data(168); // 7 days hourly
    println!(
        "  Generated {} records, {} columns",
        data.len(),
        data.num_columns()
    );

    v.check("Record count", data.len() as f64, 168.0, 0.0);
    v.check("Column count", data.num_columns() as f64, 5.0, 0.0);

    // ── Column statistics ────────────────────────────────────────
    println!();
    v.section("Temperature statistics (analytical: 25 ± 8 °C diurnal)");

    let temp_stats = data.column_stats("temperature").unwrap();
    println!(
        "  Mean: {:.1}°C, StdDev: {:.1}, Range: {:.1}–{:.1}",
        temp_stats.mean, temp_stats.std_dev, temp_stats.min, temp_stats.max
    );

    v.check("Temp mean ≈ 25°C", temp_stats.mean, 25.0, 2.0);
    v.check("Temp min ≈ 17°C", temp_stats.min, 17.0, 3.0);
    v.check("Temp max ≈ 33°C", temp_stats.max, 33.0, 3.0);

    println!();
    v.section("Soil moisture statistics");

    let sm_stats = data.column_stats("soil_moisture_1").unwrap();
    println!(
        "  Mean: {:.3} m³/m³, Range: {:.3}–{:.3}",
        sm_stats.mean, sm_stats.min, sm_stats.max
    );

    v.check_bool(
        "SM1 in valid range [0.09, 0.41]",
        sm_stats.min >= 0.09 && sm_stats.max <= 0.41,
        true,
    );

    println!();
    v.section("PAR statistics (bell curve, max ≈ 1800 µmol/m²/s)");

    let par_stats = data.column_stats("par").unwrap();
    println!(
        "  Mean: {:.0} µmol/m²/s, Max: {:.0}",
        par_stats.mean, par_stats.max
    );

    v.check("PAR max ≈ 1800", par_stats.max, 1800.0, 200.0);
    v.check_bool("PAR has zero (nighttime)", par_stats.min < 1.0, true);

    // ── CSV round-trip (write → parse) ───────────────────────────
    println!();
    v.section("CSV round-trip (generate → write → stream-parse → compare)");

    let tmp_path = std::env::temp_dir().join("airspring_test_iot.csv");
    {
        let mut f = std::fs::File::create(&tmp_path).unwrap();
        writeln!(
            f,
            "timestamp,soil_moisture_1,soil_moisture_2,temperature,humidity,par"
        )
        .unwrap();
        let sm1 = data.column("soil_moisture_1").unwrap();
        let sm2 = data.column("soil_moisture_2").unwrap();
        let temp = data.column("temperature").unwrap();
        let hum = data.column("humidity").unwrap();
        let par = data.column("par").unwrap();
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
            .unwrap();
        }
    }

    match csv_ts::parse_csv(&tmp_path, Some("timestamp")) {
        Ok(parsed) => {
            v.check("Parsed record count", parsed.len() as f64, 168.0, 0.0);
            v.check("Parsed column count", parsed.num_columns() as f64, 5.0, 0.0);
            let parsed_temp = parsed.column_stats("temperature").unwrap();
            v.check(
                "Round-trip temp mean",
                parsed_temp.mean,
                temp_stats.mean,
                0.1,
            );
        }
        Err(e) => {
            println!("  FAILED: {e}");
            // Register 3 failures
            v.check("Parsed record count", 0.0, 168.0, 0.0);
            v.check("Parsed column count", 0.0, 5.0, 0.0);
            v.check("Round-trip temp mean", 0.0, 25.0, 0.0);
        }
    }

    let _ = std::fs::remove_file(&tmp_path);

    v.finish();
}
