// SPDX-License-Identifier: AGPL-3.0-or-later
//! Bootstrap confidence intervals for validation metrics.
//!
//! Wraps `barracuda::stats::bootstrap_ci` for non-parametric uncertainty
//! quantification around error metrics like RMSE.

use crate::len_f64;

/// Deterministic seed for bootstrap sampling — ensures reproducible CI bounds
/// across runs. Value is arbitrary but fixed for validation fidelity.
const BOOTSTRAP_SEED: u64 = 42;

/// Bootstrap confidence interval for RMSE.
///
/// Uses [`barracuda::stats::bootstrap::bootstrap_ci`] to compute a
/// non-parametric confidence interval around the RMSE estimate. This
/// quantifies the uncertainty in our error metric.
///
/// Returns `(lower, upper)` bounds at the specified confidence level.
///
/// # Panics
///
/// Panics if `observed` and `simulated` have different lengths.
///
/// # Errors
///
/// Returns [`crate::error::AirSpringError::Barracuda`] on failure.
#[must_use = "bootstrap CI should be checked"]
pub fn bootstrap_rmse(
    observed: &[f64],
    simulated: &[f64],
    n_bootstrap: usize,
    confidence: f64,
) -> crate::error::Result<(f64, f64)> {
    assert_eq!(observed.len(), simulated.len());
    let residuals: Vec<f64> = observed
        .iter()
        .zip(simulated.iter())
        .map(|(o, s)| (o - s).powi(2))
        .collect();

    let ci = barracuda::stats::bootstrap_ci(
        &residuals,
        |data| {
            let n = len_f64(data);
            if n == 0.0 {
                return 0.0;
            }
            (data.iter().sum::<f64>() / n).sqrt()
        },
        n_bootstrap,
        confidence,
        BOOTSTRAP_SEED,
    )
    .map_err(|e| crate::error::AirSpringError::Barracuda(format!("{e}")))?;

    Ok((ci.lower, ci.upper))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::stats::rmse;

    #[test]
    fn bootstrap_rmse_reproducible_with_seed() {
        let obs: Vec<f64> = (0..50).map(|i| f64::from(i) * 0.2).collect();
        let sim: Vec<f64> = obs.iter().map(|v| v + 0.3).collect();

        let (l1, u1) = bootstrap_rmse(&obs, &sim, 200, 0.95).unwrap();
        let (l2, u2) = bootstrap_rmse(&obs, &sim, 200, 0.95).unwrap();

        assert!(
            (l1 - l2).abs() < 1e-10,
            "lower bound reproducible: {l1} vs {l2}"
        );
        assert!(
            (u1 - u2).abs() < 1e-10,
            "upper bound reproducible: {u1} vs {u2}"
        );
    }

    #[test]
    fn bootstrap_rmse_contains_point_estimate() {
        let obs = [0.0, 1.0, 2.0, 3.0, 4.0];
        let sim = [0.5, 1.5, 2.5, 3.5, 4.5];
        let point = rmse(&obs, &sim);
        let (lower, upper) = bootstrap_rmse(&obs, &sim, 500, 0.95).unwrap();
        assert!(
            lower <= point && point <= upper,
            "CI [{lower},{upper}] contains {point}"
        );
    }

    #[test]
    fn bootstrap_rmse_identical_vectors_narrow_ci() {
        let obs = [1.0, 2.0, 3.0, 4.0, 5.0];
        let (lower, upper) = bootstrap_rmse(&obs, &obs, 300, 0.95).unwrap();
        assert!(lower >= 0.0 && upper >= 0.0);
        assert!(
            (upper - lower) < 0.01,
            "CI should be narrow for perfect match"
        );
    }
}
