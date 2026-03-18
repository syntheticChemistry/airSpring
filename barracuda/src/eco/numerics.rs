// SPDX-License-Identifier: AGPL-3.0-or-later
//! Numerical utilities delegating to `barraCuda` precision primitives.
//!
//! Thin wrappers that keep ecology code readable while using upstream
//! high-precision implementations.

/// Kahan compensated summation for improved accuracy over large arrays.
///
/// Delegates to [`barracuda::shaders::precision::cpu::kahan_sum`] — the
/// canonical implementation. Standard `Iterator::sum()` accumulates O(n)
/// floating-point error; Kahan summation reduces this to O(1).
#[inline]
#[must_use]
pub fn kahan_sum(values: &[f64]) -> f64 {
    barracuda::shaders::precision::cpu::kahan_sum(values)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, reason = "test code uses unwrap for clarity")]
mod tests {
    use super::*;

    #[test]
    fn kahan_sum_empty() {
        assert!(kahan_sum(&[]).abs() < f64::EPSILON);
    }

    #[test]
    fn kahan_sum_single() {
        assert!((kahan_sum(&[std::f64::consts::PI]) - std::f64::consts::PI).abs() < f64::EPSILON);
    }

    #[test]
    fn kahan_sum_parity_with_naive() {
        let values = [1.0, 2.0, 3.0, 4.0, 5.0];
        let naive: f64 = values.iter().sum();
        let compensated = kahan_sum(&values);
        assert!((naive - compensated).abs() < f64::EPSILON);
    }

    #[test]
    fn kahan_sum_precision_advantage() {
        let n = 10_000_usize;
        let val = 0.1_f64;
        let values: Vec<f64> = vec![val; n];
        let compensated = kahan_sum(&values);
        let expected = val * crate::cast::usize_f64(n);
        assert!(
            (compensated - expected).abs() < 1e-10,
            "kahan_sum should be within 1e-10 of exact: got {compensated}, expected {expected}"
        );
    }
}
