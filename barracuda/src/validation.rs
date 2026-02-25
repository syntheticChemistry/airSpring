// SPDX-License-Identifier: AGPL-3.0-or-later
//! Shared validation infrastructure for hotSpring-pattern binaries.
//!
//! Leans on [`barracuda::validation::ValidationHarness`] (absorbed upstream
//! from `neuralSpring` → `ToadStool` Feb 2026) for structured pass/fail checks
//! with exit codes. airSpring adds JSON benchmark loading utilities on top.
//!
//! # Usage
//!
//! ```
//! use airspring_barracuda::validation::ValidationHarness;
//!
//! let mut v = ValidationHarness::new("ET₀ Validation");
//! v.check_abs("es(20°C)", 2.338, 2.338, 0.001);
//! assert_eq!(v.passed_count(), 1);
//! assert_eq!(v.total_count(), 1);
//! // v.finish() exits the process — call only in validation binaries
//! ```

pub use barracuda::validation::ValidationHarness;

/// Print a section header for visual grouping in validation output.
pub fn section(name: &str) {
    println!("── {name} ──");
}

/// Print an airSpring validation banner.
pub fn banner(name: &str) {
    println!("═══════════════════════════════════════════════════════════");
    println!("  airSpring {name}");
    println!("═══════════════════════════════════════════════════════════\n");
}

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

/// Extract a string from a JSON value with path context for error messages.
///
/// Intended for compile-time embedded benchmark JSON where the schema is known.
///
/// # Panics
///
/// Panics if `key` is missing from `tc` or if the value at `key` is not a string.
#[must_use]
pub fn json_str<'a>(tc: &'a serde_json::Value, key: &str) -> &'a str {
    tc[key]
        .as_str()
        .unwrap_or_else(|| panic!("benchmark JSON missing string key '{key}'"))
}

/// Extract an f64 from a JSON test case with descriptive panic.
///
/// Intended for compile-time embedded benchmark JSON.
///
/// # Panics
///
/// Panics if `key` is missing from `tc` or if the value at `key` is not an f64.
#[must_use]
pub fn json_field(tc: &serde_json::Value, key: &str) -> f64 {
    tc[key]
        .as_f64()
        .unwrap_or_else(|| panic!("benchmark JSON missing f64 key '{key}'"))
}

/// Extract a JSON array with descriptive panic.
///
/// # Panics
///
/// Panics if any key in `path` is missing, or if the value at the final path is not an array.
#[must_use]
pub fn json_array<'a>(value: &'a serde_json::Value, path: &[&str]) -> &'a Vec<serde_json::Value> {
    let mut current = value;
    for &key in path {
        current = current
            .get(key)
            .unwrap_or_else(|| panic!("benchmark JSON missing key '{key}'"));
    }
    current
        .as_array()
        .unwrap_or_else(|| panic!("benchmark JSON: expected array at {path:?}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_harness_check_pass() {
        let mut v = ValidationHarness::new("Unit Test Suite");
        v.check_abs("pass check", 1.0, 1.0, 0.01);
        assert_eq!(v.passed_count(), 1);
        assert_eq!(v.total_count(), 1);
    }

    #[test]
    fn test_harness_check_fail() {
        let mut v = ValidationHarness::new("Unit Test Suite");
        v.check_abs("fail check", 1.0, 2.0, 0.01);
        assert_eq!(v.passed_count(), 0);
        assert_eq!(v.total_count(), 1);
    }

    #[test]
    fn test_harness_check_bool() {
        let mut v = ValidationHarness::new("Unit Test Suite");
        v.check_bool("true check", true);
        v.check_bool("false check", false);
        assert_eq!(v.passed_count(), 1);
        assert_eq!(v.total_count(), 2);
    }

    #[test]
    fn test_parse_benchmark_json_valid() {
        let json = r#"{"key": 42.0}"#;
        let result = parse_benchmark_json(json);
        assert!(result.is_ok());
    }

    #[test]
    fn test_parse_benchmark_json_invalid() {
        let result = parse_benchmark_json("{{not json");
        assert!(result.is_err());
    }

    #[test]
    fn test_json_f64_nested() {
        let json: serde_json::Value = serde_json::from_str(r#"{"a": {"b": {"c": 42.5}}}"#).unwrap();
        assert!((json_f64(&json, &["a", "b", "c"]).unwrap() - 42.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_json_f64_missing_path() {
        let json: serde_json::Value = serde_json::from_str(r#"{"a": 1}"#).unwrap();
        assert!(json_f64(&json, &["a", "b"]).is_none());
    }

    #[test]
    fn test_json_f64_root_level() {
        let json: serde_json::Value = serde_json::from_str(r#"{"val": 98.76543}"#).unwrap();
        let v = json_f64(&json, &["val"]).unwrap();
        assert!((v - 98.765_43).abs() < 1e-10);
    }

    #[test]
    fn test_json_f64_empty_path() {
        let json: serde_json::Value = serde_json::from_str("42.0").unwrap();
        let v = json_f64(&json, &[]);
        assert!(v.is_some());
        assert!((v.unwrap() - 42.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_check_tolerance_boundary() {
        let mut v = ValidationHarness::new("Boundary Test");
        v.check_abs("within", 1.005, 1.0, 0.01);
        v.check_abs("beyond", 1.02, 1.0, 0.01);
        assert_eq!(v.passed_count(), 1);
        assert_eq!(v.total_count(), 2);
    }
}
