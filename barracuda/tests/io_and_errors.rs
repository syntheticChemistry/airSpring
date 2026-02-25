// SPDX-License-Identifier: AGPL-3.0-or-later
//! I/O parsing and error handling integration tests for airSpring `BarraCuda`.
//!
//! Tests CSV streaming parser, round-trip fidelity, and the unified error type.

use airspring_barracuda::io::csv_ts;
use airspring_barracuda::testutil;
use std::io::Write;

// ── CSV parsing integration ──────────────────────────────────────────

#[test]
fn test_csv_parse_from_memory_buffer() {
    let csv_data = b"time,temp_c,rh_pct,wind_ms\n\
        2024-07-01T06:00,18.5,82.0,1.2\n\
        2024-07-01T12:00,28.3,55.0,2.5\n\
        2024-07-01T18:00,25.1,65.0,1.8\n";

    let cursor = std::io::Cursor::new(csv_data as &[u8]);
    let data = csv_ts::parse_csv_reader(cursor, Some("time")).unwrap();

    assert_eq!(data.len(), 3);
    assert_eq!(data.num_columns(), 3);

    let temps = data.column("temp_c").unwrap();
    assert!((temps[0] - 18.5).abs() < 0.01);
    assert!((temps[1] - 28.3).abs() < 0.01);
    assert!((temps[2] - 25.1).abs() < 0.01);

    let stats = data.column_stats("temp_c").unwrap();
    assert!(stats.mean > 20.0 && stats.mean < 30.0);
}

#[test]
fn test_csv_round_trip_to_disk() {
    let data = testutil::generate_synthetic_iot_data(72);

    let tmp = std::env::temp_dir().join("airspring_integration_test.csv");
    {
        let mut f = std::fs::File::create(&tmp).unwrap();
        writeln!(
            f,
            "timestamp,soil_moisture_1,soil_moisture_2,temperature,humidity,par"
        )
        .unwrap();
        let sm1 = data.column("soil_moisture_1").unwrap();
        let sm2 = data.column("soil_moisture_2").unwrap();
        let temperature = data.column("temperature").unwrap();
        let hum = data.column("humidity").unwrap();
        let par = data.column("par").unwrap();
        for i in 0..data.len() {
            writeln!(
                f,
                "{},{:.6},{:.6},{:.4},{:.4},{:.2}",
                data.timestamps()[i],
                sm1[i],
                sm2[i],
                temperature[i],
                hum[i],
                par[i],
            )
            .unwrap();
        }
    }

    let parsed = csv_ts::parse_csv(&tmp, Some("timestamp")).unwrap();
    assert_eq!(parsed.len(), data.len());
    assert_eq!(parsed.num_columns(), data.num_columns());

    let orig_stats = data.column_stats("temperature").unwrap();
    let parsed_stats = parsed.column_stats("temperature").unwrap();
    assert!(
        (orig_stats.mean - parsed_stats.mean).abs() < 0.01,
        "Temperature mean drifted: {} → {}",
        orig_stats.mean,
        parsed_stats.mean
    );

    let _ = std::fs::remove_file(&tmp);
}

// ── Error path tests ─────────────────────────────────────────────────

#[test]
fn test_csv_parse_empty_input() {
    let cursor = std::io::Cursor::new(b"" as &[u8]);
    assert!(csv_ts::parse_csv_reader(cursor, None).is_err());
}

#[test]
fn test_csv_parse_missing_timestamp_col() {
    let csv = b"col_a,col_b\n1.0,2.0\n";
    let cursor = std::io::Cursor::new(csv as &[u8]);
    assert!(csv_ts::parse_csv_reader(cursor, Some("nonexistent")).is_err());
}

#[test]
fn test_csv_nonexistent_file() {
    let path = std::env::temp_dir().join("airspring_no_such_file_12345.csv");
    assert!(csv_ts::parse_csv(&path, None).is_err());
}

// ── Error type integration ──────────────────────────────────────────

#[test]
fn test_error_type_io_variant() {
    let path = std::env::temp_dir().join("airspring_no_such_file_99999.csv");
    let err = csv_ts::parse_csv(&path, None).unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("I/O error"),
        "Expected I/O error variant, got: {msg}"
    );
}

#[test]
fn test_error_type_csv_parse_variant() {
    let cursor = std::io::Cursor::new(b"" as &[u8]);
    let err = csv_ts::parse_csv_reader(cursor, None).unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("CSV parse error"),
        "Expected CSV parse error variant, got: {msg}"
    );
}

#[test]
fn test_error_type_display_and_source() {
    use airspring_barracuda::error::AirSpringError;
    use std::error::Error;

    let path = std::env::temp_dir().join("airspring_no_such_file_99999.csv");
    let err = csv_ts::parse_csv(&path, None).unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("I/O error"), "Io display: {msg}");
    assert!(err.source().is_some(), "Io should have source");

    let csv_err = AirSpringError::CsvParse("test error".to_string());
    let msg = format!("{csv_err}");
    assert!(msg.contains("CSV parse error"), "CsvParse display: {msg}");
    assert!(csv_err.source().is_none(), "CsvParse has no source");

    let bad_json = serde_json::from_str::<serde_json::Value>("{{invalid}").unwrap_err();
    let json_err = AirSpringError::JsonParse(bad_json);
    let msg = format!("{json_err}");
    assert!(msg.contains("JSON parse error"), "JsonParse display: {msg}");
    assert!(json_err.source().is_some(), "JsonParse should have source");

    let input_err = AirSpringError::InvalidInput("out of range".to_string());
    let msg = format!("{input_err}");
    assert!(msg.contains("Invalid input"), "InvalidInput display: {msg}");
    assert!(input_err.source().is_none(), "InvalidInput has no source");

    let bc_err = AirSpringError::Barracuda("primitive failed".to_string());
    let msg = format!("{bc_err}");
    assert!(msg.contains("barracuda error"), "Barracuda display: {msg}");
    assert!(bc_err.source().is_none(), "Barracuda has no source");

    let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "gone");
    let converted: AirSpringError = io_err.into();
    assert!(format!("{converted}").contains("I/O error"));

    let json_err2 = serde_json::from_str::<serde_json::Value>("not json").unwrap_err();
    let converted: AirSpringError = json_err2.into();
    assert!(format!("{converted}").contains("JSON parse error"));

    let debug_err = AirSpringError::InvalidInput("test".to_string());
    let debug_msg = format!("{debug_err:?}");
    assert!(debug_msg.contains("InvalidInput"), "Debug: {debug_msg}");
}

// ── CSV parser edge cases ────────────────────────────────────────────

#[test]
fn test_csv_malformed_floats_stored_as_nan() {
    // CSV with unparseable values should store NaN, not crash
    let csv = "timestamp,temp,rh\n2024-01-01,18.5,notanumber\n2024-01-02,20.0,65.0\n";
    let cursor = std::io::Cursor::new(csv);
    let data = csv_ts::parse_csv_reader(cursor, Some("timestamp")).unwrap();
    assert_eq!(data.len(), 2);
    assert!(data.column("rh").unwrap()[0].is_nan());
    assert!((data.column("rh").unwrap()[1] - 65.0).abs() < f64::EPSILON);
}

#[test]
fn test_csv_wrong_column_count_skipped() {
    // Rows with wrong column count should be skipped
    let csv = "timestamp,temp,rh\n2024-01-01,18.5,70.0\n2024-01-02,20.0\n2024-01-03,22.0,75.0\n";
    let cursor = std::io::Cursor::new(csv);
    let data = csv_ts::parse_csv_reader(cursor, Some("timestamp")).unwrap();
    assert_eq!(data.len(), 2);
    assert_eq!(data.skipped_rows(), 1);
}

#[test]
fn test_csv_single_row() {
    let csv = "timestamp,temp\n2024-01-01,18.5\n";
    let cursor = std::io::Cursor::new(csv);
    let data = csv_ts::parse_csv_reader(cursor, Some("timestamp")).unwrap();
    assert_eq!(data.len(), 1);
}
