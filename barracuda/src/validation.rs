//! Shared validation infrastructure for hotSpring-pattern binaries.
//!
//! Implements the validation pattern established by hotSpring:
//! - Hardcoded expected values with published provenance
//! - Explicit pass/fail per check
//! - Exit code 0 (all pass) or 1 (any fail)
//! - Benchmark JSON loading for validation fidelity
//!
//! # Usage
//!
//! ```rust,ignore
//! use airspring_barracuda::validation::ValidationRunner;
//!
//! let mut v = ValidationRunner::new("ET₀ Validation");
//! v.section("Component functions");
//! v.check("es(20°C)", computed, 2.338, 0.001);
//! v.finish();  // exits with code 0 or 1
//! ```

/// Accumulates validation checks and reports pass/fail summary.
pub struct ValidationRunner {
    name: String,
    total: u32,
    passed: u32,
}

impl ValidationRunner {
    /// Create a new validation runner.
    #[must_use]
    pub fn new(name: &str) -> Self {
        println!("═══════════════════════════════════════════════════════════");
        println!("  airSpring {name}");
        println!("═══════════════════════════════════════════════════════════\n");
        Self {
            name: name.to_string(),
            total: 0,
            passed: 0,
        }
    }

    /// Print a section header.
    pub fn section(&self, name: &str) {
        println!("── {name} ──");
    }

    /// Check a floating-point value against an expected value with tolerance.
    ///
    /// Returns `true` if the check passes.
    pub fn check(&mut self, label: &str, actual: f64, expected: f64, tolerance: f64) -> bool {
        let pass = (actual - expected).abs() <= tolerance;
        let tag = if pass { "OK" } else { "FAIL" };
        println!("  [{tag}]  {label}: {actual:.4} (expected {expected:.4}, tol {tolerance:.4})");
        self.total += 1;
        if pass {
            self.passed += 1;
        }
        pass
    }

    /// Check a boolean condition.
    pub fn check_bool(&mut self, label: &str, actual: bool, expected: bool) -> bool {
        let pass = actual == expected;
        let tag = if pass { "OK" } else { "FAIL" };
        println!("  [{tag}]  {label}: {actual} (expected {expected})");
        self.total += 1;
        if pass {
            self.passed += 1;
        }
        pass
    }

    /// Print summary and exit with appropriate code.
    ///
    /// Exit code 0 if all checks passed, 1 otherwise.
    pub fn finish(&self) -> ! {
        println!("\n═══════════════════════════════════════════════════════════");
        println!(
            "  {}: {}/{} checks passed",
            self.name, self.passed, self.total
        );
        if self.passed == self.total {
            println!("  RESULT: PASS");
        } else {
            println!(
                "  RESULT: FAIL ({} checks failed)",
                self.total - self.passed
            );
        }
        println!("═══════════════════════════════════════════════════════════");
        std::process::exit(i32::from(self.passed != self.total));
    }

    /// Current pass count.
    #[must_use]
    pub const fn passed(&self) -> u32 {
        self.passed
    }

    /// Current total count.
    #[must_use]
    pub const fn total(&self) -> u32 {
        self.total
    }
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
/// ```rust,ignore
/// let val = json_f64(&benchmark, &["example_18", "expected_et0_mm_day"]);
/// ```
#[must_use]
pub fn json_f64(value: &serde_json::Value, path: &[&str]) -> Option<f64> {
    let mut current = value;
    for &key in path {
        current = current.get(key)?;
    }
    current.as_f64()
}
