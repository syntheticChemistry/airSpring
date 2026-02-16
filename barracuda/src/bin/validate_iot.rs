//! Validate IoT time series parser and statistics.
//!
//! Uses synthetic agricultural sensor data (soil moisture, temperature,
//! PAR, humidity) to validate CSV parsing and statistical computations.

use airspring_barracuda::io::csv_ts;
use std::io::Write;

fn check(label: &str, actual: f64, expected: f64, tolerance: f64) -> bool {
    let pass = (actual - expected).abs() <= tolerance;
    let tag = if pass { "OK" } else { "FAIL" };
    println!(
        "  [{}]  {}: {:.4} (expected {:.4}, tol {:.4})",
        tag, label, actual, expected, tolerance
    );
    pass
}

fn main() {
    println!("═══════════════════════════════════════════════════════════");
    println!("  airSpring IoT Time Series Validation");
    println!("  Reference: Synthetic agricultural sensor data");
    println!("═══════════════════════════════════════════════════════════\n");

    let mut total = 0u32;
    let mut passed = 0u32;

    // ── Synthetic data generation ───────────────────────────────
    println!("── Synthetic sensor data ──");
    let data = csv_ts::generate_synthetic_data(168); // 7 days hourly
    println!("  Generated {} records, {} columns", data.len(), data.columns.len());

    total += 1;
    if check("Record count", data.len() as f64, 168.0, 0.0) { passed += 1; }

    total += 1;
    if check("Column count", data.columns.len() as f64, 5.0, 0.0) { passed += 1; }

    // ── Column statistics ───────────────────────────────────────
    println!("\n── Temperature statistics ──");
    let temp_stats = data.column_stats("temperature").unwrap();
    println!("  Mean: {:.1}°C, StdDev: {:.1}, Range: {:.1}-{:.1}",
             temp_stats.mean, temp_stats.std_dev, temp_stats.min, temp_stats.max);

    total += 1;
    if check("Temp mean ~25°C", temp_stats.mean, 25.0, 2.0) { passed += 1; }
    total += 1;
    if check("Temp min > 15°C", temp_stats.min, 17.0, 3.0) { passed += 1; }
    total += 1;
    if check("Temp max < 35°C", temp_stats.max, 33.0, 3.0) { passed += 1; }

    println!("\n── Soil moisture statistics ──");
    let sm_stats = data.column_stats("soil_moisture_1").unwrap();
    println!("  Mean: {:.3} m³/m³, Range: {:.3}-{:.3}",
             sm_stats.mean, sm_stats.min, sm_stats.max);

    total += 1;
    if check("SM1 in valid range [0.1, 0.4]",
             (sm_stats.min >= 0.09 && sm_stats.max <= 0.41) as u32 as f64,
             1.0, 0.0) { passed += 1; }

    println!("\n── PAR statistics ──");
    let par_stats = data.column_stats("par").unwrap();
    println!("  Mean: {:.0} µmol/m²/s, Max: {:.0}",
             par_stats.mean, par_stats.max);

    total += 1;
    if check("PAR max < 2000", par_stats.max, 1800.0, 200.0) { passed += 1; }
    total += 1;
    if check("PAR has zero (nighttime)", (par_stats.min < 1.0) as u32 as f64, 1.0, 0.0) {
        passed += 1;
    }

    // ── CSV round-trip (write → read) ───────────────────────────
    println!("\n── CSV round-trip (write → parse) ──");

    // Write synthetic data to temp CSV
    let tmp_path = std::env::temp_dir().join("airspring_test_iot.csv");
    {
        let mut f = std::fs::File::create(&tmp_path).unwrap();
        writeln!(f, "timestamp,soil_moisture_1,soil_moisture_2,temperature,humidity,par").unwrap();
        for rec in &data.records {
            writeln!(
                f,
                "{},{:.4},{:.4},{:.2},{:.2},{:.1}",
                rec.timestamp,
                rec.fields.get("soil_moisture_1").unwrap_or(&0.0),
                rec.fields.get("soil_moisture_2").unwrap_or(&0.0),
                rec.fields.get("temperature").unwrap_or(&0.0),
                rec.fields.get("humidity").unwrap_or(&0.0),
                rec.fields.get("par").unwrap_or(&0.0),
            )
            .unwrap();
        }
    }

    // Read it back
    match csv_ts::parse_csv(&tmp_path, Some("timestamp")) {
        Ok(parsed) => {
            total += 1;
            if check("Parsed record count", parsed.len() as f64, 168.0, 0.0) {
                passed += 1;
            }

            total += 1;
            if check("Parsed column count", parsed.columns.len() as f64, 5.0, 0.0) {
                passed += 1;
            }

            // Verify values survived round-trip
            let parsed_temp = parsed.column_stats("temperature").unwrap();
            total += 1;
            if check("Round-trip temp mean", parsed_temp.mean, temp_stats.mean, 0.1) {
                passed += 1;
            }
        }
        Err(e) => {
            println!("  FAILED: {}", e);
            total += 3;
        }
    }

    // Clean up
    let _ = std::fs::remove_file(&tmp_path);

    // ── Summary ─────────────────────────────────────────────────
    println!("\n═══════════════════════════════════════════════════════════");
    println!(
        "  IoT Validation: {}/{} checks passed",
        passed, total
    );
    if passed == total {
        println!("  RESULT: PASS");
    } else {
        println!("  RESULT: FAIL ({} checks failed)", total - passed);
    }
    println!("═══════════════════════════════════════════════════════════");

    std::process::exit(if passed == total { 0 } else { 1 });
}
