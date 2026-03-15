// SPDX-License-Identifier: AGPL-3.0-or-later
// Data strategy: validation binaries embed benchmark JSON at compile time via
// include_str!. Library code never accesses the filesystem directly — callers
// provide data. Runtime file access is isolated to io::csv_ts (accepts paths
// from callers) and validation binaries.
//! Shared validation infrastructure for structured validation binaries.
//!
//! Leans on [`barracuda::validation::ValidationHarness`] for structured pass/fail checks
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

pub use barracuda::validation::{ValidationHarness, exit_no_gpu, gpu_required};

/// Initialise tracing so that `ValidationHarness::finish()` output is visible.
///
/// Call once at the top of every validation binary's `main()`.
/// Uses the `RUST_LOG` env-var if set, otherwise defaults to `info`.
pub fn init_tracing() {
    use tracing_subscriber::EnvFilter;
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .without_time()
        .with_target(false)
        .init();
}

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

/// Extract an f64 from a nested JSON path, or fail the benchmark with a
/// descriptive message and `exit(1)`.
///
/// Use this in validation binaries where a missing benchmark field means the
/// test infrastructure is broken — not a validation failure.
#[must_use]
pub fn json_f64_required(value: &serde_json::Value, path: &[&str]) -> f64 {
    json_f64(value, path).unwrap_or_else(|| {
        let path_str = path.join(".");
        eprintln!("FATAL: benchmark JSON missing required f64 at: {path_str}");
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
            eprintln!("FATAL: benchmark JSON missing required u64 path: {path_str}");
            std::process::exit(1);
        });
    }
    current.as_u64().unwrap_or_else(|| {
        let path_str = path.join(".");
        eprintln!("FATAL: benchmark JSON value at {path_str} is not u64");
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
        eprintln!("FATAL: {e}");
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
        eprintln!("FATAL: {e}");
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
        eprintln!("FATAL: {e}");
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
        eprintln!("FATAL: benchmark JSON missing required object at: {path_str}");
        std::process::exit(1);
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
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

    // ── json_str ────────────────────────────────────────────────────────────
    #[test]
    fn test_json_str_valid() {
        let json: serde_json::Value = serde_json::from_str(r#"{"label": "hello world"}"#).unwrap();
        assert_eq!(json_str(&json, "label"), "hello world");
    }

    #[test]
    fn test_json_str_missing_key() {
        let json: serde_json::Value = serde_json::from_str(r#"{"other": "x"}"#).unwrap();
        let err = json_str_checked(&json, "missing").unwrap_err();
        assert!(format!("{err}").contains("missing string key"));
    }

    #[test]
    fn test_json_str_non_string_value() {
        let json: serde_json::Value = serde_json::from_str(r#"{"num": 42}"#).unwrap();
        let err = json_str_checked(&json, "num").unwrap_err();
        assert!(format!("{err}").contains("missing string key"));
    }

    #[test]
    fn test_json_str_checked_ok() {
        let json: serde_json::Value = serde_json::from_str(r#"{"label": "ok"}"#).unwrap();
        assert_eq!(json_str_checked(&json, "label").unwrap(), "ok");
    }

    #[test]
    fn test_json_str_checked_missing() {
        let json: serde_json::Value = serde_json::from_str(r#"{"other": "x"}"#).unwrap();
        let err = json_str_checked(&json, "missing").unwrap_err();
        assert!(matches!(
            err,
            crate::error::AirSpringError::BenchmarkParse(_)
        ));
        assert!(format!("{err}").contains("missing string key"));
    }

    // ── json_field ───────────────────────────────────────────────────────────
    #[test]
    fn test_json_field_valid() {
        let json: serde_json::Value = serde_json::from_str(r#"{"val": 42.5}"#).unwrap();
        assert!((json_field(&json, "val") - 42.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_json_field_missing_key() {
        let json: serde_json::Value = serde_json::from_str(r#"{"other": 1.0}"#).unwrap();
        let err = json_field_checked(&json, "missing").unwrap_err();
        assert!(format!("{err}").contains("missing f64 key"));
    }

    #[test]
    fn test_json_field_non_number_value() {
        let json: serde_json::Value = serde_json::from_str(r#"{"str": "hello"}"#).unwrap();
        let err = json_field_checked(&json, "str").unwrap_err();
        assert!(format!("{err}").contains("missing f64 key"));
    }

    #[test]
    fn test_json_field_checked_ok() {
        let json: serde_json::Value = serde_json::from_str(r#"{"val": 42.5}"#).unwrap();
        assert!((json_field_checked(&json, "val").unwrap() - 42.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_json_field_checked_missing() {
        let json: serde_json::Value = serde_json::from_str(r#"{"other": 1.0}"#).unwrap();
        let err = json_field_checked(&json, "missing").unwrap_err();
        assert!(matches!(
            err,
            crate::error::AirSpringError::BenchmarkParse(_)
        ));
    }

    // ── json_array ──────────────────────────────────────────────────────────
    #[test]
    fn test_json_array_valid() {
        let json: serde_json::Value = serde_json::from_str(r#"{"data": [1.0, 2.0, 3.0]}"#).unwrap();
        let arr = json_array(&json, &["data"]);
        assert_eq!(arr.len(), 3);
        assert!((arr[0].as_f64().unwrap() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_json_array_empty() {
        let json: serde_json::Value = serde_json::from_str(r#"{"data": []}"#).unwrap();
        let arr = json_array(&json, &["data"]);
        assert!(arr.is_empty());
    }

    #[test]
    fn test_json_array_nested_path() {
        let json: serde_json::Value =
            serde_json::from_str(r#"{"outer": {"inner": [1.0, 2.0]}}"#).unwrap();
        let arr = json_array(&json, &["outer", "inner"]);
        assert_eq!(arr.len(), 2);
    }

    #[test]
    fn test_json_array_missing_key() {
        let json: serde_json::Value = serde_json::from_str(r#"{"other": []}"#).unwrap();
        let err = json_array_checked(&json, &["missing"]).unwrap_err();
        assert!(format!("{err}").contains("missing key"));
    }

    #[test]
    fn test_json_array_non_array_value() {
        let json: serde_json::Value = serde_json::from_str(r#"{"data": 42}"#).unwrap();
        let err = json_array_checked(&json, &["data"]).unwrap_err();
        assert!(format!("{err}").contains("expected array"));
    }

    #[test]
    fn test_json_array_object_not_array() {
        let json: serde_json::Value = serde_json::from_str(r#"{"data": {"nested": 1}}"#).unwrap();
        let err = json_array_checked(&json, &["data"]).unwrap_err();
        assert!(format!("{err}").contains("expected array"));
    }

    // ── json_f64 extended ───────────────────────────────────────────────────
    #[test]
    fn test_json_f64_string_value_returns_none() {
        let json: serde_json::Value = serde_json::from_str(r#"{"a": "not a number"}"#).unwrap();
        assert!(json_f64(&json, &["a"]).is_none());
    }

    #[test]
    fn test_json_f64_null_returns_none() {
        let json: serde_json::Value = serde_json::from_str(r#"{"a": null}"#).unwrap();
        assert!(json_f64(&json, &["a"]).is_none());
    }

    #[test]
    fn test_json_f64_integer_value() {
        let json: serde_json::Value = serde_json::from_str(r#"{"a": 42}"#).unwrap();
        let v = json_f64(&json, &["a"]).unwrap();
        assert!((v - 42.0).abs() < f64::EPSILON);
    }

    // ── check_abs pass, fail, exact boundary ────────────────────────────────
    #[test]
    fn test_check_abs_pass() {
        let mut v = ValidationHarness::new("Pass");
        v.check_abs("exact", 7.25, 7.25, 0.001);
        assert_eq!(v.passed_count(), 1);
    }

    #[test]
    fn test_check_abs_fail() {
        let mut v = ValidationHarness::new("Fail");
        v.check_abs("way off", 10.0, 1.0, 0.01);
        assert_eq!(v.passed_count(), 0);
    }

    #[test]
    fn test_check_abs_exact_boundary() {
        let mut v = ValidationHarness::new("Boundary");
        v.check_abs("within tol", 1.009, 1.0, 0.01);
        v.check_abs("just over", 1.02, 1.0, 0.01);
        assert_eq!(v.passed_count(), 1);
        assert_eq!(v.total_count(), 2);
    }

    // ── check_bool true and false ───────────────────────────────────────────
    #[test]
    fn test_check_bool_true_only() {
        let mut v = ValidationHarness::new("Bool");
        v.check_bool("ok", true);
        assert_eq!(v.passed_count(), 1);
    }

    #[test]
    fn test_check_bool_false_only() {
        let mut v = ValidationHarness::new("Bool");
        v.check_bool("fail", false);
        assert_eq!(v.passed_count(), 0);
    }

    // ── passed_count / total_count mixed ────────────────────────────────────
    #[test]
    fn test_harness_mixed_pass_fail_counts() {
        let mut v = ValidationHarness::new("Mixed");
        v.check_abs("p1", 1.0, 1.0, 0.01);
        v.check_abs("f1", 1.0, 5.0, 0.01);
        v.check_bool("p2", true);
        v.check_bool("f2", false);
        v.check_abs("p3", 2.0, 2.0, 0.01);
        assert_eq!(v.passed_count(), 3);
        assert_eq!(v.total_count(), 5);
    }

    // ── banner and section (exercise print paths) ───────────────────────────
    #[test]
    fn test_banner_and_section() {
        banner("Test Banner");
        section("Test Section");
    }

    // ── parse_benchmark_json (valid already covered, invalid already covered) ─
    #[test]
    fn test_parse_benchmark_json_valid_complex() {
        let json = r#"{"nested": {"arr": [1, 2, 3], "val": 7.25}}"#;
        let result = parse_benchmark_json(json);
        assert!(result.is_ok());
        let val = result.unwrap();
        assert!(val.get("nested").is_some());
    }

    // ── json_str_opt ─────────────────────────────────────────────────────
    #[test]
    fn test_json_str_opt_present() {
        let json: serde_json::Value = serde_json::from_str(r#"{"a": {"label": "hello"}}"#).unwrap();
        assert_eq!(json_str_opt(&json, &["a", "label"]), Some("hello"));
    }

    #[test]
    fn test_json_str_opt_missing_path() {
        let json: serde_json::Value = serde_json::from_str(r#"{"a": 1}"#).unwrap();
        assert!(json_str_opt(&json, &["a", "b"]).is_none());
    }

    #[test]
    fn test_json_str_opt_not_a_string() {
        let json: serde_json::Value = serde_json::from_str(r#"{"a": 42}"#).unwrap();
        assert!(json_str_opt(&json, &["a"]).is_none());
    }

    #[test]
    fn test_json_str_opt_empty_path() {
        let json: serde_json::Value = serde_json::from_str(r#""bare string""#).unwrap();
        assert_eq!(json_str_opt(&json, &[]), Some("bare string"));
    }

    // ── json_array_opt ───────────────────────────────────────────────────
    #[test]
    fn test_json_array_opt_present() {
        let json: serde_json::Value =
            serde_json::from_str(r#"{"data": {"items": [1, 2, 3]}}"#).unwrap();
        let arr = json_array_opt(&json, &["data", "items"]);
        assert!(arr.is_some());
        assert_eq!(arr.unwrap().len(), 3);
    }

    #[test]
    fn test_json_array_opt_missing_path() {
        let json: serde_json::Value = serde_json::from_str(r#"{"a": 1}"#).unwrap();
        assert!(json_array_opt(&json, &["a", "b"]).is_none());
    }

    #[test]
    fn test_json_array_opt_not_an_array() {
        let json: serde_json::Value = serde_json::from_str(r#"{"a": "text"}"#).unwrap();
        assert!(json_array_opt(&json, &["a"]).is_none());
    }

    #[test]
    fn test_json_array_opt_empty_array() {
        let json: serde_json::Value = serde_json::from_str(r#"{"a": []}"#).unwrap();
        let arr = json_array_opt(&json, &["a"]);
        assert!(arr.is_some());
        assert!(arr.unwrap().is_empty());
    }

    // ── json_object_opt ──────────────────────────────────────────────────
    #[test]
    fn test_json_object_opt_present() {
        let json: serde_json::Value =
            serde_json::from_str(r#"{"meta": {"k": "v", "n": 1}}"#).unwrap();
        let obj = json_object_opt(&json, &["meta"]);
        assert!(obj.is_some());
        assert_eq!(obj.unwrap().len(), 2);
    }

    #[test]
    fn test_json_object_opt_missing_path() {
        let json: serde_json::Value = serde_json::from_str(r#"{"a": 1}"#).unwrap();
        assert!(json_object_opt(&json, &["x", "y"]).is_none());
    }

    #[test]
    fn test_json_object_opt_not_an_object() {
        let json: serde_json::Value = serde_json::from_str(r#"{"a": [1]}"#).unwrap();
        assert!(json_object_opt(&json, &["a"]).is_none());
    }

    #[test]
    fn test_json_object_opt_nested() {
        let json: serde_json::Value =
            serde_json::from_str(r#"{"a": {"b": {"c": "deep"}}}"#).unwrap();
        let obj = json_object_opt(&json, &["a", "b"]);
        assert!(obj.is_some());
        assert!(obj.unwrap().contains_key("c"));
    }
}
