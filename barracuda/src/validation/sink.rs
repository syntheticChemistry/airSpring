// SPDX-License-Identifier: AGPL-3.0-or-later
//! Pluggable validation output — wetSpring V132 `ValidationSink` pattern.
//!
//! The upstream [`barracuda::validation::ValidationHarness`] always writes to
//! `tracing::info!` and calls `process::exit`. This module provides a
//! *consumer-side* trait so airSpring binaries can additionally emit
//! machine-readable reports (JSON for CI, structured log events, etc.) from
//! the accumulated [`barracuda::validation::Check`] data *before* the harness
//! exits.
//!
//! # Usage
//!
//! ```no_run
//! use airspring_barracuda::validation::{ValidationHarness, ValidationSink, JsonSink};
//!
//! let mut harness = ValidationHarness::new("example");
//! harness.check_abs("demo", 1.0, 1.0, 0.01);
//!
//! // Emit JSON report before the harness exits.
//! let sink = JsonSink::default();
//! sink.emit(&harness);
//!
//! harness.finish();
//! ```

use super::{Check, ValidationHarness};

/// Receives completed validation results for alternative output formats.
pub trait ValidationSink {
    /// Consume the harness results. Called once before the harness exits.
    fn emit(&self, harness: &ValidationHarness);
}

/// Emits a single JSON object to stdout with all check results.
///
/// The JSON structure is stable for CI parsers:
///
/// ```json
/// {
///   "suite": "ET₀ Validation",
///   "passed": 10,
///   "total": 10,
///   "checks": [
///     {
///       "label": "es(20°C)",
///       "passed": true,
///       "observed": 2.338,
///       "expected": 2.338,
///       "tolerance": 0.001,
///       "mode": "abs"
///     }
///   ]
/// }
/// ```
#[derive(Debug, Default)]
pub struct JsonSink {
    /// When true, output is pretty-printed. Default: compact.
    pub pretty: bool,
}

impl JsonSink {
    /// Create a pretty-printing JSON sink.
    #[must_use]
    pub const fn pretty() -> Self {
        Self { pretty: true }
    }

    fn check_to_json(check: &Check) -> serde_json::Value {
        let mut obj = serde_json::json!({
            "label": check.label,
            "passed": check.passed,
            "observed": check.observed,
            "expected": check.expected,
            "tolerance": check.tolerance,
            "mode": check.mode.to_string(),
        });
        if let Some(rel_tol) = check.rel_tolerance {
            obj["rel_tolerance"] = serde_json::json!(rel_tol);
        }
        obj
    }
}

impl ValidationSink for JsonSink {
    fn emit(&self, harness: &ValidationHarness) {
        let checks: Vec<serde_json::Value> =
            harness.checks.iter().map(Self::check_to_json).collect();
        let report = serde_json::json!({
            "suite": harness.name,
            "passed": harness.passed_count(),
            "total": harness.total_count(),
            "all_passed": harness.all_passed(),
            "checks": checks,
        });
        let output = if self.pretty {
            serde_json::to_string_pretty(&report)
        } else {
            serde_json::to_string(&report)
        };
        if let Ok(s) = output {
            println!("{s}");
        }
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test code uses unwrap for clarity")]
mod tests {
    use super::*;

    #[test]
    fn json_sink_emits_valid_json() {
        let mut h = ValidationHarness::new("sink test");
        h.check_abs("pass", 1.0, 1.0, 0.01);
        h.check_abs("fail", 5.0, 1.0, 0.01);
        h.check_abs_or_rel("dual", 100.001, 100.0, 0.01, 1e-6);

        let checks: Vec<serde_json::Value> = h.checks.iter().map(JsonSink::check_to_json).collect();
        let report = serde_json::json!({
            "suite": h.name,
            "passed": h.passed_count(),
            "total": h.total_count(),
            "all_passed": h.all_passed(),
            "checks": checks,
        });
        let s = serde_json::to_string(&report).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&s).unwrap();

        assert_eq!(parsed["suite"], "sink test");
        assert_eq!(parsed["passed"], 2);
        assert_eq!(parsed["total"], 3);
        assert!(!parsed["all_passed"].as_bool().unwrap());
        assert_eq!(parsed["checks"].as_array().unwrap().len(), 3);
        assert!(parsed["checks"][0]["passed"].as_bool().unwrap());
        assert!(!parsed["checks"][1]["passed"].as_bool().unwrap());
        assert!(parsed["checks"][2]["rel_tolerance"].is_number());
    }

    #[test]
    fn json_sink_empty_harness() {
        let h = ValidationHarness::new("empty");
        let checks: Vec<serde_json::Value> = h.checks.iter().map(JsonSink::check_to_json).collect();
        let report = serde_json::json!({
            "suite": h.name,
            "passed": h.passed_count(),
            "total": h.total_count(),
            "all_passed": h.all_passed(),
            "checks": checks,
        });
        let s = serde_json::to_string(&report).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&s).unwrap();

        assert_eq!(parsed["passed"], 0);
        assert_eq!(parsed["total"], 0);
        assert!(parsed["all_passed"].as_bool().unwrap());
        assert!(parsed["checks"].as_array().unwrap().is_empty());
    }

    #[test]
    fn json_sink_pretty_produces_multiline() {
        let mut h = ValidationHarness::new("pretty test");
        h.check_bool("ok", true);

        let checks: Vec<serde_json::Value> = h.checks.iter().map(JsonSink::check_to_json).collect();
        let report = serde_json::json!({
            "suite": h.name,
            "passed": h.passed_count(),
            "total": h.total_count(),
            "all_passed": h.all_passed(),
            "checks": checks,
        });
        let s = serde_json::to_string_pretty(&report).unwrap();
        assert!(s.contains('\n'));
    }

    #[test]
    fn json_sink_mode_strings() {
        let mut h = ValidationHarness::new("modes");
        h.check_abs("abs", 1.0, 1.0, 0.01);
        h.check_relative("rel", 1.0, 1.0, 0.01);
        h.check_upper("upper", 0.5, 1.0);
        h.check_lower("lower", 1.5, 1.0);
        h.check_abs_or_rel("dual", 1.0, 1.0, 0.01, 0.01);

        let checks: Vec<serde_json::Value> = h.checks.iter().map(JsonSink::check_to_json).collect();

        assert_eq!(checks[0]["mode"], "abs");
        assert_eq!(checks[1]["mode"], "rel");
        assert_eq!(checks[2]["mode"], "<");
        assert_eq!(checks[3]["mode"], ">");
        assert_eq!(checks[4]["mode"], "abs|rel");
    }

    #[test]
    fn json_sink_pretty_ctor() {
        let s = JsonSink::pretty();
        assert!(s.pretty);
    }

    #[test]
    fn json_sink_default_compact() {
        let s = JsonSink::default();
        assert!(!s.pretty);
    }

    #[test]
    fn check_to_json_failing_check() {
        let mut h = ValidationHarness::new("fail");
        h.check_abs("divergent", 10.0, 1.0, 0.001);
        let json = JsonSink::check_to_json(&h.checks[0]);
        assert!(!json["passed"].as_bool().unwrap());
        assert_eq!(json["observed"].as_f64().unwrap(), 10.0);
        assert_eq!(json["expected"].as_f64().unwrap(), 1.0);
    }

    #[test]
    fn json_sink_emit_to_stdout() {
        let mut h = ValidationHarness::new("emit_test");
        h.check_bool("flag", true);
        let sink = JsonSink::default();
        sink.emit(&h);
        let pretty_sink = JsonSink::pretty();
        pretty_sink.emit(&h);
    }
}
