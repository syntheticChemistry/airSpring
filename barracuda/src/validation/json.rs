// SPDX-License-Identifier: AGPL-3.0-or-later
//! JSON benchmark extraction DSL for validation binaries.
//!
//! Functions to parse and extract values from compile-time embedded benchmark JSON.

use tracing::error;

/// Load a benchmark JSON file embedded at compile time.
///
/// Returns the parsed `serde_json::Value` tree.
///
/// # Errors
///
/// Returns [`crate::error::AirSpringError::JsonParse`] if the JSON is malformed.
#[must_use = "parsed benchmark should be used"]
pub fn parse_benchmark_json(json_str: &str) -> crate::error::Result<serde_json::Value> {
    Ok(serde_json::from_str(json_str)?)
}

/// Extract a float from a nested JSON path.
///
/// # Examples
///
/// ```
/// use airspring_barracuda::validation::json_f64;
///
/// let json: serde_json::Value = serde_json::from_str(
///     r#"{"example_18": {"expected_et0_mm_day": 3.88}}"#,
/// ).unwrap();
/// let val = json_f64(&json, &["example_18", "expected_et0_mm_day"]);
/// assert!((val.unwrap() - 3.88).abs() < f64::EPSILON);
/// ```
#[must_use]
pub fn json_f64(value: &serde_json::Value, path: &[&str]) -> Option<f64> {
    let mut current = value;
    for &key in path {
        current = current.get(key)?;
    }
    current.as_f64()
}

/// Extract an f64 from a nested JSON path, or fail the benchmark with a
/// descriptive message and `exit(1)`.
///
/// Use this in validation binaries where a missing benchmark field means the
/// test infrastructure is broken — not a validation failure.
#[must_use]
pub fn json_f64_required(value: &serde_json::Value, path: &[&str]) -> f64 {
    json_f64(value, path).unwrap_or_else(|| {
        let path_str = path.join(".");
        error!(path = %path_str, "benchmark JSON missing required f64");
        std::process::exit(1)
    })
}

/// Extract a u64 from a nested JSON path, or fail the benchmark with `exit(1)`.
#[must_use]
pub fn json_u64_required(value: &serde_json::Value, path: &[&str]) -> u64 {
    let mut current = value;
    for &key in path {
        current = current.get(key).unwrap_or_else(|| {
            let path_str = path.join(".");
            error!(path = %path_str, "benchmark JSON missing required u64 path");
            std::process::exit(1);
        });
    }
    current.as_u64().unwrap_or_else(|| {
        let path_str = path.join(".");
        error!(path = %path_str, "benchmark JSON value is not u64");
        std::process::exit(1);
    })
}

/// Extract a string from a JSON value with path context for error messages.
///
/// # Errors
///
/// Returns `BenchmarkParse` if `key` is missing or the value is not a string.
pub fn json_str_checked<'a>(tc: &'a serde_json::Value, key: &str) -> crate::error::Result<&'a str> {
    tc.get(key).and_then(|v| v.as_str()).ok_or_else(|| {
        crate::error::AirSpringError::BenchmarkParse(format!(
            "benchmark JSON missing string key '{key}'"
        ))
    })
}

/// Extract a string from a JSON value, or exit with a structured error.
///
/// Intended for compile-time embedded benchmark JSON where a missing field
/// means broken test infrastructure. Calls `exit(1)` instead of panicking
/// to produce clean diagnostics in validation binary output.
#[must_use]
pub fn json_str<'a>(tc: &'a serde_json::Value, key: &str) -> &'a str {
    json_str_checked(tc, key).unwrap_or_else(|e| {
        error!(error = %e, "fatal");
        std::process::exit(1)
    })
}

/// Extract an f64 from a JSON test case.
///
/// # Errors
///
/// Returns `BenchmarkParse` if `key` is missing or the value is not an f64.
pub fn json_field_checked(tc: &serde_json::Value, key: &str) -> crate::error::Result<f64> {
    tc.get(key)
        .and_then(serde_json::Value::as_f64)
        .ok_or_else(|| {
            crate::error::AirSpringError::BenchmarkParse(format!(
                "benchmark JSON missing f64 key '{key}'"
            ))
        })
}

/// Extract an f64 from a JSON test case, or exit with a structured error.
///
/// Intended for compile-time embedded benchmark JSON where a missing field
/// means broken test infrastructure.
#[must_use]
pub fn json_field(tc: &serde_json::Value, key: &str) -> f64 {
    json_field_checked(tc, key).unwrap_or_else(|e| {
        error!(error = %e, "fatal");
        std::process::exit(1)
    })
}

/// Extract a JSON array from a nested path.
///
/// # Errors
///
/// Returns `BenchmarkParse` if any key in `path` is missing or the final value is not an array.
pub fn json_array_checked<'a>(
    value: &'a serde_json::Value,
    path: &[&str],
) -> crate::error::Result<&'a Vec<serde_json::Value>> {
    let mut current = value;
    for &key in path {
        current = current.get(key).ok_or_else(|| {
            crate::error::AirSpringError::BenchmarkParse(format!(
                "benchmark JSON missing key '{key}'"
            ))
        })?;
    }
    current.as_array().ok_or_else(|| {
        crate::error::AirSpringError::BenchmarkParse(format!(
            "benchmark JSON: expected array at {path:?}"
        ))
    })
}

/// Extract a JSON array, or exit with a structured error.
///
/// Intended for compile-time embedded benchmark JSON where a missing field
/// means broken test infrastructure.
#[must_use]
pub fn json_array<'a>(value: &'a serde_json::Value, path: &[&str]) -> &'a Vec<serde_json::Value> {
    json_array_checked(value, path).unwrap_or_else(|e| {
        error!(error = %e, "fatal");
        std::process::exit(1)
    })
}

/// Extract a string from a nested JSON path; returns `None` if missing or not a string.
#[must_use]
pub fn json_str_opt<'a>(value: &'a serde_json::Value, path: &[&str]) -> Option<&'a str> {
    let mut current = value;
    for &key in path {
        current = current.get(key)?;
    }
    current.as_str()
}

/// Extract a JSON array from a nested path; returns `None` if missing or not an array.
#[must_use]
pub fn json_array_opt<'a>(
    value: &'a serde_json::Value,
    path: &[&str],
) -> Option<&'a Vec<serde_json::Value>> {
    let mut current = value;
    for &key in path {
        current = current.get(key)?;
    }
    current.as_array()
}

/// Extract a JSON object from a nested path; returns `None` if missing or not an object.
#[must_use]
pub fn json_object_opt<'a>(
    value: &'a serde_json::Value,
    path: &[&str],
) -> Option<&'a serde_json::Map<String, serde_json::Value>> {
    let mut current = value;
    for &key in path {
        current = current.get(key)?;
    }
    current.as_object()
}

/// Extract a JSON object from a nested path, or fail with `exit(1)`.
///
/// Use in validation binaries where a missing benchmark field means broken infrastructure.
#[must_use]
pub fn json_object_required<'a>(
    value: &'a serde_json::Value,
    path: &[&str],
) -> &'a serde_json::Map<String, serde_json::Value> {
    json_object_opt(value, path).unwrap_or_else(|| {
        let path_str = path.join(".");
        error!(path = %path_str, "benchmark JSON missing required object");
        std::process::exit(1);
    })
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test code uses unwrap for clarity")]
mod tests {
    use super::*;

    fn sample_json() -> serde_json::Value {
        serde_json::json!({
            "example_18": {
                "expected_et0_mm_day": 3.88,
                "station": "East Lansing",
                "count": 365,
                "days": [1.0, 2.0, 3.0],
                "meta": { "source": "FAO-56" }
            }
        })
    }

    #[test]
    fn parse_benchmark_json_valid() {
        let result = parse_benchmark_json(r#"{"a": 1}"#);
        assert!(result.is_ok());
    }

    #[test]
    fn parse_benchmark_json_invalid() {
        let result = parse_benchmark_json("not json");
        assert!(result.is_err());
    }

    #[test]
    fn json_f64_nested_path() {
        let j = sample_json();
        let val = json_f64(&j, &["example_18", "expected_et0_mm_day"]);
        assert!((val.unwrap() - 3.88).abs() < f64::EPSILON);
    }

    #[test]
    fn json_f64_missing_key() {
        let j = sample_json();
        assert!(json_f64(&j, &["example_18", "nonexistent"]).is_none());
    }

    #[test]
    fn json_f64_wrong_type() {
        let j = sample_json();
        assert!(json_f64(&j, &["example_18", "station"]).is_none());
    }

    #[test]
    fn json_str_checked_success() {
        let j = sample_json();
        let tc = j.get("example_18").unwrap();
        assert_eq!(json_str_checked(tc, "station").unwrap(), "East Lansing");
    }

    #[test]
    fn json_str_checked_missing() {
        let j = sample_json();
        let tc = j.get("example_18").unwrap();
        assert!(json_str_checked(tc, "nonexistent").is_err());
    }

    #[test]
    fn json_str_checked_wrong_type() {
        let j = sample_json();
        let tc = j.get("example_18").unwrap();
        assert!(json_str_checked(tc, "count").is_err());
    }

    #[test]
    fn json_field_checked_success() {
        let j = sample_json();
        let tc = j.get("example_18").unwrap();
        let val = json_field_checked(tc, "expected_et0_mm_day").unwrap();
        assert!((val - 3.88).abs() < f64::EPSILON);
    }

    #[test]
    fn json_field_checked_missing() {
        let j = sample_json();
        let tc = j.get("example_18").unwrap();
        assert!(json_field_checked(tc, "nonexistent").is_err());
    }

    #[test]
    fn json_array_checked_success() {
        let j = sample_json();
        let arr = json_array_checked(&j, &["example_18", "days"]).unwrap();
        assert_eq!(arr.len(), 3);
    }

    #[test]
    fn json_array_checked_missing_key() {
        let j = sample_json();
        assert!(json_array_checked(&j, &["example_18", "nonexistent"]).is_err());
    }

    #[test]
    fn json_array_checked_not_array() {
        let j = sample_json();
        assert!(json_array_checked(&j, &["example_18", "station"]).is_err());
    }

    #[test]
    fn json_str_opt_success() {
        let j = sample_json();
        assert_eq!(
            json_str_opt(&j, &["example_18", "station"]),
            Some("East Lansing")
        );
    }

    #[test]
    fn json_str_opt_missing() {
        let j = sample_json();
        assert!(json_str_opt(&j, &["example_18", "missing"]).is_none());
    }

    #[test]
    fn json_array_opt_success() {
        let j = sample_json();
        let arr = json_array_opt(&j, &["example_18", "days"]).unwrap();
        assert_eq!(arr.len(), 3);
    }

    #[test]
    fn json_array_opt_missing() {
        let j = sample_json();
        assert!(json_array_opt(&j, &["example_18", "missing"]).is_none());
    }

    #[test]
    fn json_object_opt_success() {
        let j = sample_json();
        let obj = json_object_opt(&j, &["example_18", "meta"]).unwrap();
        assert_eq!(obj.get("source").unwrap().as_str(), Some("FAO-56"));
    }

    #[test]
    fn json_object_opt_missing() {
        let j = sample_json();
        assert!(json_object_opt(&j, &["example_18", "missing"]).is_none());
    }

    #[test]
    fn json_object_opt_not_object() {
        let j = sample_json();
        assert!(json_object_opt(&j, &["example_18", "station"]).is_none());
    }
}
