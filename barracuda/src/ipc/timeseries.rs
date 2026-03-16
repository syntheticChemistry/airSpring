// SPDX-License-Identifier: AGPL-3.0-or-later
//! Cross-spring time series exchange format.
//!
//! Implements the `ecoPrimals/time-series/v1` schema from wateringHole
//! `CROSS_SPRING_DATA_FLOW_STANDARD.md` for structured data exchange
//! between springs via `capability.call`.
//!
//! # Pattern
//!
//! Adopted from wetSpring's `ipc/timeseries.rs` for cross-spring
//! interoperability (e.g., wetSpring sends rainfall time series →
//! airSpring runs water balance / SCS-CN / Richards analysis).

use crate::error::AirSpringError;

/// Schema identifier for the cross-spring time series format.
pub const SCHEMA: &str = "ecoPrimals/time-series/v1";

/// Parsed cross-spring time series data.
#[derive(Debug, Clone)]
pub struct TimeSeriesData {
    /// Variable name (`snake_case`).
    pub variable: String,
    /// SI or documented unit.
    pub unit: String,
    /// ISO 8601 UTC timestamps.
    pub timestamps: Vec<String>,
    /// Numeric values (same length as timestamps).
    pub values: Vec<f64>,
    /// Source spring identifier.
    pub source_spring: String,
}

/// Build a time series payload conforming to the cross-spring standard.
///
/// Returns a JSON object with `schema`, `variable`, `unit`, `source`,
/// `timestamps`, and `values` fields.
#[must_use]
pub fn build_time_series(
    variable: &str,
    unit: &str,
    timestamps: &[String],
    values: &[f64],
    experiment: Option<&str>,
) -> serde_json::Value {
    serde_json::json!({
        "schema": SCHEMA,
        "variable": variable,
        "unit": unit,
        "source": {
            "spring": crate::niche::NICHE_NAME,
            "experiment": experiment.unwrap_or(""),
            "capability": format!("science.{variable}"),
        },
        "timestamps": timestamps,
        "values": values,
    })
}

/// Parse an incoming time series payload, validating the schema version.
///
/// # Errors
///
/// Returns `AirSpringError::Ipc` if schema is missing or wrong version,
/// or if required fields are absent.
pub fn parse_time_series(params: &serde_json::Value) -> Result<TimeSeriesData, AirSpringError> {
    let schema = params
        .get("schema")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("");

    if schema != SCHEMA {
        return Err(AirSpringError::Ipc(format!(
            "expected schema '{SCHEMA}', got '{schema}'"
        )));
    }

    let variable = params
        .get("variable")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("unknown")
        .to_string();

    let unit = params
        .get("unit")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("dimensionless")
        .to_string();

    let timestamps: Vec<String> = params
        .get("timestamps")
        .and_then(serde_json::Value::as_array)
        .map(|arr| {
            arr.iter()
                .filter_map(serde_json::Value::as_str)
                .map(String::from)
                .collect()
        })
        .unwrap_or_default();

    let values: Vec<f64> = params
        .get("values")
        .and_then(serde_json::Value::as_array)
        .map(|arr| arr.iter().filter_map(serde_json::Value::as_f64).collect())
        .unwrap_or_default();

    let source_spring = params
        .get("source")
        .and_then(|s| s.get("spring"))
        .and_then(serde_json::Value::as_str)
        .unwrap_or("unknown")
        .to_string();

    Ok(TimeSeriesData {
        variable,
        unit,
        timestamps,
        values,
        source_spring,
    })
}

/// Handle `science.timeseries` — analyze incoming cross-spring time series.
///
/// Computes basic statistics (mean, variance, trend) on the values.
///
/// # Errors
///
/// Returns `AirSpringError::InvalidInput` if `time_series` is missing or empty.
#[expect(clippy::cast_precision_loss)]
pub fn handle_timeseries(params: &serde_json::Value) -> Result<serde_json::Value, AirSpringError> {
    let data = extract_ts_data(params)?;
    let values = &data.values;

    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;

    let trend = if values.len() >= 2 {
        let last = values[values.len() - 1];
        let first = values[0];
        if first.abs() > f64::EPSILON {
            (last - first) / first
        } else {
            0.0
        }
    } else {
        0.0
    };

    Ok(serde_json::json!({
        "variable": data.variable,
        "unit": data.unit,
        "source_spring": data.source_spring,
        "n_points": values.len(),
        "mean": mean,
        "variance": variance,
        "trend_pct": trend * 100.0,
        "min": values.iter().copied().fold(f64::INFINITY, f64::min),
        "max": values.iter().copied().fold(f64::NEG_INFINITY, f64::max),
    }))
}

/// Build an ET₀ time series for outbound exchange (airSpring → other springs).
#[must_use]
pub fn build_et0_series(
    timestamps: &[String],
    et0_values: &[f64],
    experiment: Option<&str>,
) -> serde_json::Value {
    build_time_series("et0_fao56", "mm/day", timestamps, et0_values, experiment)
}

/// Build a soil moisture time series for outbound exchange.
#[must_use]
pub fn build_soil_moisture_series(
    timestamps: &[String],
    vwc_values: &[f64],
    experiment: Option<&str>,
) -> serde_json::Value {
    build_time_series("soil_moisture_vwc", "m³/m³", timestamps, vwc_values, experiment)
}

fn extract_ts_data(params: &serde_json::Value) -> Result<TimeSeriesData, AirSpringError> {
    let data = params
        .get("time_series")
        .ok_or_else(|| AirSpringError::InvalidInput("missing time_series object".into()))
        .and_then(parse_time_series)?;

    if data.values.is_empty() {
        return Err(AirSpringError::InvalidInput(
            "time_series values array is empty".into(),
        ));
    }

    Ok(data)
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test code uses unwrap for clarity")]
mod tests {
    use super::*;

    #[test]
    fn build_time_series_has_correct_schema() {
        let ts = build_time_series(
            "et0_fao56",
            "mm/day",
            &["2024-01-01T00:00:00Z".into()],
            &[5.2],
            Some("test-exp"),
        );
        assert_eq!(ts["schema"], SCHEMA);
        assert_eq!(ts["variable"], "et0_fao56");
        assert_eq!(ts["unit"], "mm/day");
        assert_eq!(ts["source"]["spring"], crate::niche::NICHE_NAME);
        assert_eq!(ts["source"]["experiment"], "test-exp");
    }

    #[test]
    fn parse_time_series_roundtrip() {
        let ts = build_time_series(
            "rainfall",
            "mm",
            &["2024-01-01T00:00:00Z".into(), "2024-01-02T00:00:00Z".into()],
            &[10.0, 15.5],
            None,
        );
        let parsed = parse_time_series(&ts).unwrap();
        assert_eq!(parsed.variable, "rainfall");
        assert_eq!(parsed.unit, "mm");
        assert_eq!(parsed.values.len(), 2);
        assert_eq!(parsed.source_spring, crate::niche::NICHE_NAME);
    }

    #[test]
    fn parse_rejects_wrong_schema() {
        let bad = serde_json::json!({ "schema": "wrong/v99" });
        assert!(parse_time_series(&bad).is_err());
    }

    #[test]
    fn handle_timeseries_computes_stats() {
        let ts = build_time_series(
            "temperature",
            "°C",
            &["2024-01-01".into(), "2024-01-02".into(), "2024-01-03".into()],
            &[10.0, 20.0, 30.0],
            None,
        );
        let params = serde_json::json!({ "time_series": ts });
        let result = handle_timeseries(&params).unwrap();
        assert_eq!(result["n_points"], 3);
        assert!((result["mean"].as_f64().unwrap() - 20.0).abs() < 1e-10);
        assert_eq!(result["min"], 10.0);
        assert_eq!(result["max"], 30.0);
    }

    #[test]
    fn handle_timeseries_rejects_empty() {
        let ts = build_time_series("temp", "°C", &[], &[], None);
        let params = serde_json::json!({ "time_series": ts });
        assert!(handle_timeseries(&params).is_err());
    }

    #[test]
    fn handle_timeseries_rejects_missing_ts() {
        let params = serde_json::json!({});
        assert!(handle_timeseries(&params).is_err());
    }

    #[test]
    fn build_et0_series_uses_correct_variable() {
        let ts = build_et0_series(
            &["2024-01-01".into()],
            &[5.0],
            None,
        );
        assert_eq!(ts["variable"], "et0_fao56");
        assert_eq!(ts["unit"], "mm/day");
    }

    #[test]
    fn build_soil_moisture_series_uses_correct_variable() {
        let ts = build_soil_moisture_series(
            &["2024-01-01".into()],
            &[0.35],
            None,
        );
        assert_eq!(ts["variable"], "soil_moisture_vwc");
        assert_eq!(ts["unit"], "m³/m³");
    }

    #[test]
    fn trend_calculation_positive() {
        let ts = build_time_series(
            "temp",
            "°C",
            &["t1".into(), "t2".into()],
            &[10.0, 15.0],
            None,
        );
        let params = serde_json::json!({ "time_series": ts });
        let result = handle_timeseries(&params).unwrap();
        let trend = result["trend_pct"].as_f64().unwrap();
        assert!((trend - 50.0).abs() < 1e-10);
    }
}
