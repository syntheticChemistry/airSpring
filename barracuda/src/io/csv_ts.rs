//! CSV time series parser for IoT sensor data.
//!
//! Parses timestamped sensor data (soil moisture, temperature, PAR, weather)
//! from standard CSV files as produced by agricultural IoT systems.

use std::collections::HashMap;
use std::path::Path;

/// A single timestamped record from a sensor CSV.
#[derive(Debug, Clone)]
pub struct TimeseriesRecord {
    /// Timestamp as string (ISO 8601 or custom)
    pub timestamp: String,
    /// Named fields: column_name -> value
    pub fields: HashMap<String, f64>,
}

/// Parsed time series dataset.
#[derive(Debug, Clone)]
pub struct TimeseriesData {
    /// Column headers (excluding timestamp)
    pub columns: Vec<String>,
    /// All records in chronological order
    pub records: Vec<TimeseriesRecord>,
    /// Name of the timestamp column
    pub timestamp_column: String,
}

impl TimeseriesData {
    /// Number of records.
    pub fn len(&self) -> usize {
        self.records.len()
    }

    /// Whether the dataset is empty.
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// Extract a single column as a Vec<f64>.
    pub fn column(&self, name: &str) -> Option<Vec<f64>> {
        if !self.columns.contains(&name.to_string()) {
            return None;
        }
        Some(
            self.records
                .iter()
                .map(|r| *r.fields.get(name).unwrap_or(&f64::NAN))
                .collect(),
        )
    }

    /// Compute basic statistics for a column.
    pub fn column_stats(&self, name: &str) -> Option<ColumnStats> {
        let values = self.column(name)?;
        let valid: Vec<f64> = values.iter().filter(|v| !v.is_nan()).copied().collect();
        if valid.is_empty() {
            return None;
        }

        let n = valid.len();
        let sum: f64 = valid.iter().sum();
        let mean = sum / n as f64;
        let variance = valid.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n as f64;
        let min = valid.iter().cloned().fold(f64::MAX, f64::min);
        let max = valid.iter().cloned().fold(f64::MIN, f64::max);

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
    pub count: usize,
    pub mean: f64,
    pub std_dev: f64,
    pub min: f64,
    pub max: f64,
    pub missing: usize,
}

/// Parse a CSV file with a header row.
/// First column is assumed to be the timestamp unless `timestamp_col` is specified.
pub fn parse_csv(path: &Path, timestamp_col: Option<&str>) -> Result<TimeseriesData, String> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| format!("Read error {}: {}", path.display(), e))?;

    let mut lines = content.lines();

    // Parse header
    let header_line = lines
        .next()
        .ok_or_else(|| "Empty CSV file".to_string())?;
    let headers: Vec<String> = header_line.split(',').map(|s| s.trim().to_string()).collect();

    if headers.is_empty() {
        return Err("No columns in CSV".to_string());
    }

    let ts_col = timestamp_col.unwrap_or(&headers[0]);
    let ts_idx = headers
        .iter()
        .position(|h| h == ts_col)
        .ok_or_else(|| format!("Timestamp column '{}' not found", ts_col))?;

    let data_columns: Vec<String> = headers
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != ts_idx)
        .map(|(_, h)| h.clone())
        .collect();

    let mut records = Vec::new();
    for line in lines {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() != headers.len() {
            continue; // Skip malformed rows
        }

        let timestamp = parts[ts_idx].trim().to_string();
        let mut fields = HashMap::new();

        for (i, col_name) in headers.iter().enumerate() {
            if i == ts_idx {
                continue;
            }
            if let Ok(val) = parts[i].trim().parse::<f64>() {
                fields.insert(col_name.clone(), val);
            }
        }

        records.push(TimeseriesRecord { timestamp, fields });
    }

    Ok(TimeseriesData {
        columns: data_columns,
        records,
        timestamp_column: ts_col.to_string(),
    })
}

/// Generate a synthetic IoT sensor CSV dataset for testing.
pub fn generate_synthetic_data(n_records: usize) -> TimeseriesData {
    let columns = vec![
        "soil_moisture_1".to_string(),
        "soil_moisture_2".to_string(),
        "temperature".to_string(),
        "humidity".to_string(),
        "par".to_string(),
    ];

    let mut records = Vec::with_capacity(n_records);
    for i in 0..n_records {
        let hour = i % 24;
        let day = i / 24;
        let timestamp = format!("2024-07-{:02}T{:02}:00:00", day + 1, hour);

        let mut fields = HashMap::new();
        // Soil moisture: slowly decreasing with daily ET, recharge events
        let base_sm = 0.28 - 0.002 * (i as f64);
        let sm1 = (base_sm + 0.02 * ((i as f64 * 0.1).sin())).clamp(0.10, 0.40);
        let sm2 = (base_sm - 0.01 + 0.015 * ((i as f64 * 0.1 + 1.0).sin())).clamp(0.10, 0.40);
        fields.insert("soil_moisture_1".to_string(), sm1);
        fields.insert("soil_moisture_2".to_string(), sm2);

        // Temperature: diurnal cycle
        let temp = 25.0 + 8.0 * ((hour as f64 - 14.0) * PI / 12.0).cos();
        fields.insert("temperature".to_string(), temp);

        // Humidity: inverse of temperature
        let rh = 70.0 - 15.0 * ((hour as f64 - 14.0) * PI / 12.0).cos();
        fields.insert("humidity".to_string(), rh);

        // PAR: bell curve centered at noon
        let par = if hour >= 6 && hour <= 20 {
            1800.0 * (-(((hour as f64 - 13.0) / 3.5).powi(2))).exp()
        } else {
            0.0
        };
        fields.insert("par".to_string(), par);

        records.push(TimeseriesRecord { timestamp, fields });
    }

    TimeseriesData {
        columns,
        records,
        timestamp_column: "timestamp".to_string(),
    }
}

use std::f64::consts::PI;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_synthetic_data() {
        let data = generate_synthetic_data(48); // 2 days
        assert_eq!(data.len(), 48);
        assert_eq!(data.columns.len(), 5);

        let stats = data.column_stats("temperature").unwrap();
        assert!(stats.mean > 20.0 && stats.mean < 30.0);
    }

    #[test]
    fn test_column_extraction() {
        let data = generate_synthetic_data(24);
        let temps = data.column("temperature").unwrap();
        assert_eq!(temps.len(), 24);
        assert!(temps.iter().all(|t| *t > 10.0 && *t < 40.0));
    }
}
