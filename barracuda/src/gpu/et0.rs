//! Batched ET₀ GPU orchestrator with CPU fallback.
//!
//! Dispatches N station-day ET₀ computations. Currently uses the validated
//! CPU path because the `ToadStool` `batched_elementwise_f64.wgsl` shader's
//! `pow_f64` function returns 0.0 for non-integer exponents (the atmospheric
//! pressure calculation requires exponent 5.26).
//!
//! # `ToadStool` Issue: `pow_f64` Non-Integer Exponents
//!
//! **Shader**: `batched_elementwise_f64.wgsl`, lines 113–139
//! **Bug**: `pow_f64(base, 5.26)` returns 0.0 (hits the placeholder branch)
//! **Impact**: Atmospheric pressure P = 101.3 × ((293 − 0.0065z)/293)^5.26
//!            silently computes P = 0.0, cascading γ = 0.0 and wrong ET₀
//! **Fix**: Replace the placeholder at line 138 with `exp_f64(exp * log_f64(base))`
//!          when `base > 0`. The shader already has working `exp_f64` and `log_f64`.
//!
//! Once `ToadStool` ships the fix, flip [`Backend::Gpu`] to default.

use crate::eco::evapotranspiration::{self as et, DailyEt0Input};

/// Backend selection for batched ET₀.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Backend {
    /// Validated CPU path — current default.
    #[default]
    Cpu,
    /// GPU path via `ToadStool` (blocked on `pow_f64` fix).
    Gpu,
}

/// Result from a batched ET₀ computation.
#[derive(Debug, Clone)]
pub struct BatchedEt0Result {
    /// ET₀ values in mm/day, one per input row.
    pub et0_values: Vec<f64>,
    /// Which backend was actually used.
    pub backend_used: Backend,
}

/// Batched ET₀ orchestrator.
///
/// Computes FAO-56 Penman-Monteith ET₀ for N station-days in a single call.
/// Designed for the GPU hot path once `ToadStool` fixes `pow_f64`.
#[derive(Debug)]
pub struct BatchedEt0 {
    backend: Backend,
}

impl BatchedEt0 {
    /// Create a new batched ET₀ orchestrator.
    #[must_use]
    pub const fn new(backend: Backend) -> Self {
        Self { backend }
    }

    /// Create with CPU fallback (always safe).
    #[must_use]
    pub const fn cpu() -> Self {
        Self {
            backend: Backend::Cpu,
        }
    }

    /// Compute ET₀ for a batch of station-days.
    ///
    /// Each input is a `DailyEt0Input` from the validated CPU module.
    /// Returns one ET₀ value per input.
    #[must_use]
    pub fn compute(&self, inputs: &[DailyEt0Input]) -> BatchedEt0Result {
        match self.backend {
            Backend::Cpu => Self::compute_cpu(inputs),
            Backend::Gpu => {
                // GPU path blocked — fall back to CPU with a note
                Self::compute_cpu(inputs)
            }
        }
    }

    fn compute_cpu(inputs: &[DailyEt0Input]) -> BatchedEt0Result {
        let et0_values: Vec<f64> = inputs
            .iter()
            .map(|input| et::daily_et0(input).et0)
            .collect();
        BatchedEt0Result {
            et0_values,
            backend_used: Backend::Cpu,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_input() -> DailyEt0Input {
        DailyEt0Input {
            tmin: 12.3,
            tmax: 21.5,
            tmean: Some(16.9),
            solar_radiation: 22.07,
            wind_speed_2m: 2.078,
            actual_vapour_pressure: 1.409,
            elevation_m: 100.0,
            latitude_deg: 50.80,
            day_of_year: 187,
        }
    }

    #[test]
    fn test_batched_et0_single() {
        let engine = BatchedEt0::cpu();
        let result = engine.compute(&[sample_input()]);
        assert_eq!(result.et0_values.len(), 1);
        assert!(result.et0_values[0] > 2.0 && result.et0_values[0] < 6.0);
        assert_eq!(result.backend_used, Backend::Cpu);
    }

    #[test]
    fn test_batched_et0_matches_scalar() {
        let engine = BatchedEt0::cpu();
        let input = sample_input();
        let scalar = et::daily_et0(&input).et0;
        let batched = engine.compute(std::slice::from_ref(&input));
        assert!(
            (batched.et0_values[0] - scalar).abs() < f64::EPSILON,
            "Batched {} != scalar {scalar}",
            batched.et0_values[0]
        );
    }

    #[test]
    fn test_batched_et0_multiple() {
        let engine = BatchedEt0::cpu();
        let inputs: Vec<DailyEt0Input> = (0..100)
            .map(|i| DailyEt0Input {
                day_of_year: 150 + i,
                ..sample_input()
            })
            .collect();
        let result = engine.compute(&inputs);
        assert_eq!(result.et0_values.len(), 100);
        for &val in &result.et0_values {
            assert!(val > 0.0, "ET₀ should be positive: {val}");
        }
    }

    #[test]
    fn test_batched_et0_empty() {
        let engine = BatchedEt0::cpu();
        let result = engine.compute(&[]);
        assert!(result.et0_values.is_empty());
    }

    #[test]
    fn test_batched_et0_deterministic() {
        let engine = BatchedEt0::cpu();
        let inputs = vec![sample_input(); 50];
        let r1 = engine.compute(&inputs);
        let r2 = engine.compute(&inputs);
        for (a, b) in r1.et0_values.iter().zip(&r2.et0_values) {
            assert!((a - b).abs() < f64::EPSILON);
        }
    }
}
