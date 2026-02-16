//! Streaming CSV time series parser for `IoT` sensor data.
//!
//! Parses timestamped sensor data (soil moisture, temperature, PAR, weather)
//! from standard CSV files as produced by agricultural `IoT` systems.
//!
//! # Design
//!
//! - **Streaming**: Uses `BufReader` — never buffers entire file in memory.
//! - **Columnar**: Data stored as one `Vec<f64>` per column, not per-record
//!   `HashMap`. This is cache-friendly for column-wise statistical operations.
//! - **Zero-copy column access**: `column()` returns `&[f64]` slice, no allocation.

use crate::error::{AirSpringError, Result};
use std::collections::HashMap;
use std::io::{self, BufRead};
use std::path::Path;

/// Columnar time series dataset.
///
/// Stores data in column-major order for efficient statistical operations.
#[derive(Debug, Clone)]
pub struct TimeseriesData {
    /// Column names (excluding timestamp)
    column_names: Vec<String>,
    /// Column name → index mapping for O(1) lookup
    column_index: HashMap<String, usize>,
    /// Timestamp strings in chronological order
    timestamps: Vec<String>,
    /// Column data: `columns[col_idx][row_idx]` = value.
    /// `NaN` for missing values.
    columns: Vec<Vec<f64>>,
    /// Name of the timestamp column.
    pub timestamp_column: String,
}

impl TimeseriesData {
    /// Construct a `TimeseriesData` from pre-built columns.
    ///
    /// Used by [`crate::testutil`] for synthetic data generation.
    /// For parsing CSV files, use [`parse_csv`] or [`parse_csv_reader`].
    #[must_use]
    pub fn new(column_names: Vec<String>, timestamps: Vec<String>, columns: Vec<Vec<f64>>) -> Self {
        let column_index: HashMap<String, usize> = column_names
            .iter()
            .enumerate()
            .map(|(i, name)| (name.clone(), i))
            .collect();
        Self {
            column_names,
            column_index,
            timestamps,
            columns,
            timestamp_column: "timestamp".to_string(),
        }
    }

    /// Number of records (rows).
    #[must_use]
    pub const fn len(&self) -> usize {
        self.timestamps.len()
    }

    /// Whether the dataset is empty.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.timestamps.is_empty()
    }

    /// Column names (excluding timestamp).
    #[must_use]
    pub fn column_names(&self) -> &[String] {
        &self.column_names
    }

    /// Number of data columns (excluding timestamp).
    #[must_use]
    pub const fn num_columns(&self) -> usize {
        self.column_names.len()
    }

    /// Zero-copy column access by name. Returns `None` if column not found.
    #[must_use]
    pub fn column(&self, name: &str) -> Option<&[f64]> {
        self.column_index
            .get(name)
            .map(|&idx| self.columns[idx].as_slice())
    }

    /// All timestamps.
    #[must_use]
    pub fn timestamps(&self) -> &[String] {
        &self.timestamps
    }

    /// Compute basic statistics for a column.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn column_stats(&self, name: &str) -> Option<ColumnStats> {
        let values = self.column(name)?;
        let valid: Vec<f64> = values.iter().copied().filter(|v| !v.is_nan()).collect();
        if valid.is_empty() {
            return None;
        }

        let n = valid.len();
        let sum: f64 = valid.iter().sum();
        let mean = sum / n as f64;
        let variance = valid.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n as f64;
        let min = valid.iter().copied().fold(f64::MAX, f64::min);
        let max = valid.iter().copied().fold(f64::MIN, f64::max);

        Some(ColumnStats {
            count: n,
            mean,
            std_dev: variance.sqrt(),
            min,
            max,
            missing: values.len() - n,
        })
    }
}

/// Basic statistics for a data column.
#[derive(Debug, Clone)]
pub struct ColumnStats {
    /// Number of non-NaN values
    pub count: usize,
    /// Arithmetic mean
    pub mean: f64,
    /// Population standard deviation
    pub std_dev: f64,
    /// Minimum value
    pub min: f64,
    /// Maximum value
    pub max: f64,
    /// Number of missing (NaN) values
    pub missing: usize,
}

/// Parse a CSV file with streaming `BufReader`.
///
/// First column is assumed to be the timestamp unless `timestamp_col` is specified.
/// Data is stored in columnar format for efficient statistical access.
///
/// # Errors
///
/// Returns [`AirSpringError::Io`] if the file cannot be opened,
/// [`AirSpringError::CsvParse`] if the input is empty or has no columns.
#[must_use = "parsed timeseries should be used"]
pub fn parse_csv(path: &Path, timestamp_col: Option<&str>) -> Result<TimeseriesData> {
    let file = std::fs::File::open(path)?;
    let reader = io::BufReader::new(file);
    parse_csv_reader(reader, timestamp_col)
}

/// Parse CSV from any `BufRead` source (file, stdin, network stream, bytes).
///
/// # Errors
///
/// Returns [`AirSpringError::CsvParse`] if the input is empty or has no columns.
#[must_use = "parsed timeseries should be used"]
pub fn parse_csv_reader<R: BufRead>(
    reader: R,
    timestamp_col: Option<&str>,
) -> Result<TimeseriesData> {
    let mut lines = reader.lines();

    // Parse header
    let header_line = lines
        .next()
        .ok_or_else(|| AirSpringError::CsvParse("Empty CSV input".to_string()))?
        .map_err(AirSpringError::Io)?;
    let headers: Vec<String> = header_line
        .split(',')
        .map(|s| s.trim().to_string())
        .collect();

    if headers.is_empty() {
        return Err(AirSpringError::CsvParse("No columns in CSV".to_string()));
    }

    let ts_col = timestamp_col.unwrap_or(&headers[0]);
    let ts_idx = headers.iter().position(|h| h == ts_col).ok_or_else(|| {
        AirSpringError::CsvParse(format!("Timestamp column '{ts_col}' not found"))
    })?;

    // Build column name list and index map (excluding timestamp)
    let column_names: Vec<String> = headers
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != ts_idx)
        .map(|(_, h)| h.clone())
        .collect();

    let column_index: HashMap<String, usize> = column_names
        .iter()
        .enumerate()
        .map(|(i, name)| (name.clone(), i))
        .collect();

    let num_cols = column_names.len();
    let mut columns: Vec<Vec<f64>> = vec![Vec::new(); num_cols];
    let mut timestamps = Vec::new();

    // Stream rows — never buffer entire file
    for line_result in lines {
        let line = line_result.map_err(AirSpringError::Io)?;
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() != headers.len() {
            continue; // Skip malformed rows
        }

        timestamps.push(parts[ts_idx].trim().to_string());

        let mut col_idx = 0;
        for (i, _) in headers.iter().enumerate() {
            if i == ts_idx {
                continue;
            }
            let value = parts[i].trim().parse::<f64>().unwrap_or(f64::NAN);
            columns[col_idx].push(value);
            col_idx += 1;
        }
    }

    Ok(TimeseriesData {
        column_names,
        column_index,
        timestamps,
        columns,
        timestamp_column: ts_col.to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::generate_synthetic_iot_data;

    #[test]
    fn test_synthetic_data_structure() {
        let data = generate_synthetic_iot_data(48);
        assert_eq!(data.len(), 48);
        assert_eq!(data.num_columns(), 5);
        assert!(!data.is_empty());
    }

    #[test]
    fn test_synthetic_temperature_stats() {
        let data = generate_synthetic_iot_data(48);
        let stats = data.column_stats("temperature").unwrap();
        assert!(stats.mean > 20.0 && stats.mean < 30.0);
        assert!(stats.min > 15.0);
        assert!(stats.max < 35.0);
    }

    #[test]
    fn test_column_access_zero_copy() {
        let data = generate_synthetic_iot_data(24);
        let temps = data.column("temperature").unwrap();
        assert_eq!(temps.len(), 24);
        assert!(temps.iter().all(|&t| t > 10.0 && t < 40.0));
    }

    #[test]
    fn test_column_not_found() {
        let data = generate_synthetic_iot_data(24);
        assert!(data.column("nonexistent").is_none());
    }

    #[test]
    fn test_csv_round_trip() {
        use std::io::Write;

        let data = generate_synthetic_iot_data(48);
        let mut buf = Vec::new();
        writeln!(
            buf,
            "timestamp,soil_moisture_1,soil_moisture_2,temperature,humidity,par"
        )
        .unwrap();
        for i in 0..data.len() {
            writeln!(
                buf,
                "{},{:.4},{:.4},{:.2},{:.2},{:.1}",
                data.timestamps()[i],
                data.column("soil_moisture_1").unwrap()[i],
                data.column("soil_moisture_2").unwrap()[i],
                data.column("temperature").unwrap()[i],
                data.column("humidity").unwrap()[i],
                data.column("par").unwrap()[i],
            )
            .unwrap();
        }

        let cursor = io::Cursor::new(buf);
        let parsed = parse_csv_reader(cursor, Some("timestamp")).unwrap();
        assert_eq!(parsed.len(), 48);
        assert_eq!(parsed.num_columns(), 5);

        // Values survive round-trip within formatting precision
        let orig_stats = data.column_stats("temperature").unwrap();
        let parsed_stats = parsed.column_stats("temperature").unwrap();
        assert!((orig_stats.mean - parsed_stats.mean).abs() < 0.1);
    }

    #[test]
    fn test_parse_empty_input() {
        let cursor = io::Cursor::new(b"" as &[u8]);
        assert!(parse_csv_reader(cursor, None).is_err());
    }

    #[test]
    fn test_parse_header_only() {
        let cursor = io::Cursor::new(b"time,temp,rh\n" as &[u8]);
        let data = parse_csv_reader(cursor, Some("time")).unwrap();
        assert_eq!(data.len(), 0);
        assert_eq!(data.num_columns(), 2);
    }

    #[test]
    fn test_parse_skips_malformed_rows() {
        let input = b"time,temp\n2024-01-01,20.0\n2024-01-02\n2024-01-03,22.0\n";
        let cursor = io::Cursor::new(input as &[u8]);
        let data = parse_csv_reader(cursor, Some("time")).unwrap();
        assert_eq!(data.len(), 2); // malformed row skipped
    }

    #[test]
    fn test_parse_handles_comments() {
        let input = b"time,temp\n# comment\n2024-01-01,20.0\n";
        let cursor = io::Cursor::new(input as &[u8]);
        let data = parse_csv_reader(cursor, Some("time")).unwrap();
        assert_eq!(data.len(), 1);
    }
}
