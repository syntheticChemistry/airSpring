// SPDX-License-Identifier: AGPL-3.0-or-later
//! Numerical utilities delegating to `barraCuda` precision primitives.
//!
//! Thin wrappers that keep ecology code readable while using upstream
//! high-precision implementations.

/// Kahan compensated summation for improved accuracy over large arrays.
///
/// Delegates to [`barracuda::shaders::precision::cpu::kahan_sum`] when `local`
/// is enabled; otherwise uses the same algorithm inlined for IPC-only builds.
#[inline]
#[must_use]
pub fn kahan_sum(values: &[f64]) -> f64 {
    #[cfg(feature = "local")]
    {
        barracuda::shaders::precision::cpu::kahan_sum(values)
    }
    #[cfg(not(feature = "local"))]
    {
        let mut sum = 0.0_f64;
        let mut c = 0.0_f64;
        for &x in values {
            let y = x - c;
            let t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        sum
    }
}

#[cfg(test)]
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
