//! Statistical agreement metrics for model validation.
//!
//! Pure-Rust implementations of standard metrics used across hydrology,
//! agricultural science, and environmental modeling. These are absorption
//! targets for `barracuda::stats::metrics`.
//!
//! # Metrics
//!
//! | Metric | Range | Perfect | Reference |
//! |--------|-------|---------|-----------|
//! | [`rmse`] | \[0, ∞) | 0.0 | Standard |
//! | [`mbe`] | (−∞, ∞) | 0.0 | Standard |
//! | [`nash_sutcliffe`] | (−∞, 1\] | 1.0 | Nash & Sutcliffe (1970) |
//! | [`index_of_agreement`] | \[0, 1\] | 1.0 | Willmott (1981) |
//! | [`coefficient_of_determination`] | (−∞, 1\] | 1.0 | Standard regression R² |
//!
//! # Provenance
//!
//! Validated against Dong et al. (2020) *Agriculture* 10(12):598 — soil sensor
//! calibration study using RMSE, IA, and MBE (Paper Eqs. 1-3).

use crate::len_f64;

/// Root Mean Square Error.
///
/// RMSE = √(Σ(obsᵢ − simᵢ)² / n)
///
/// # Panics
///
/// Panics if slices have different lengths.
#[must_use]
pub fn rmse(observed: &[f64], simulated: &[f64]) -> f64 {
    assert_eq!(observed.len(), simulated.len(), "length mismatch");
    let n = len_f64(observed);
    let ss: f64 = observed
        .iter()
        .zip(simulated)
        .map(|(o, s)| (o - s).powi(2))
        .sum();
    (ss / n).sqrt()
}

/// Mean Bias Error.
///
/// MBE = Σ(simᵢ − obsᵢ) / n
///
/// Positive MBE indicates over-prediction.
///
/// # Panics
///
/// Panics if slices have different lengths.
#[must_use]
pub fn mbe(observed: &[f64], simulated: &[f64]) -> f64 {
    assert_eq!(observed.len(), simulated.len(), "length mismatch");
    let n = len_f64(observed);
    let bias: f64 = observed.iter().zip(simulated).map(|(o, s)| s - o).sum();
    bias / n
}

/// Nash-Sutcliffe Efficiency (Nash & Sutcliffe, 1970).
///
/// NSE = 1 − Σ(obsᵢ − simᵢ)² / Σ(obsᵢ − obs̄)²
///
/// NSE = 1.0 is perfect; NSE = 0.0 means the model is no better than the
/// mean; NSE < 0 means the model is worse than the mean.
///
/// # Panics
///
/// Panics if slices have different lengths or are empty.
#[must_use]
pub fn nash_sutcliffe(observed: &[f64], simulated: &[f64]) -> f64 {
    assert_eq!(observed.len(), simulated.len(), "length mismatch");
    assert!(!observed.is_empty(), "empty input");

    let n = len_f64(observed);
    let mean_obs: f64 = observed.iter().sum::<f64>() / n;

    let ss_res: f64 = observed
        .iter()
        .zip(simulated)
        .map(|(o, s)| (o - s).powi(2))
        .sum();
    let ss_tot: f64 = observed.iter().map(|o| (o - mean_obs).powi(2)).sum();

    if ss_tot == 0.0 {
        return 1.0;
    }
    1.0 - ss_res / ss_tot
}

/// Index of Agreement (Willmott, 1981).
///
/// IA = 1 − Σ(obsᵢ − simᵢ)² / Σ(|simᵢ − obs̄| + |obsᵢ − obs̄|)²
///
/// Values range from 0.0 (no agreement) to 1.0 (perfect).
///
/// # Panics
///
/// Panics if slices have different lengths or are empty.
#[must_use]
pub fn index_of_agreement(observed: &[f64], simulated: &[f64]) -> f64 {
    assert_eq!(observed.len(), simulated.len(), "length mismatch");
    assert!(!observed.is_empty(), "empty input");

    let n = len_f64(observed);
    let mean_obs: f64 = observed.iter().sum::<f64>() / n;

    let numerator: f64 = observed
        .iter()
        .zip(simulated)
        .map(|(o, s)| (o - s).powi(2))
        .sum();
    let denominator: f64 = observed
        .iter()
        .zip(simulated)
        .map(|(o, s)| ((s - mean_obs).abs() + (o - mean_obs).abs()).powi(2))
        .sum();

    if denominator == 0.0 {
        return 1.0;
    }
    1.0 - numerator / denominator
}

/// Coefficient of determination (R²) via sum-of-squares.
///
/// R² = 1 − `SS_res` / `SS_tot`
///
/// Equivalent to [`nash_sutcliffe`] — provided as a named alias for
/// domains where "R²" is the conventional term.
#[must_use]
pub fn coefficient_of_determination(observed: &[f64], simulated: &[f64]) -> f64 {
    nash_sutcliffe(observed, simulated)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rmse_perfect() {
        let a = [1.0, 2.0, 3.0];
        assert!(rmse(&a, &a) < f64::EPSILON);
    }

    #[test]
    fn test_rmse_known() {
        let obs = [1.0, 2.0, 3.0, 4.0];
        let sim = [1.1, 2.1, 2.9, 3.9];
        assert!((rmse(&obs, &sim) - 0.1).abs() < 1e-10);
    }

    #[test]
    fn test_mbe_positive_bias() {
        let obs = [1.0, 2.0, 3.0];
        let sim = [1.5, 2.5, 3.5];
        assert!((mbe(&obs, &sim) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_mbe_zero() {
        let obs = [1.0, 2.0, 3.0];
        let sim = [0.9, 2.1, 3.0];
        assert!(mbe(&obs, &sim).abs() < 1e-10);
    }

    #[test]
    fn test_nse_perfect() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0];
        assert!((nash_sutcliffe(&a, &a) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_nse_mean_predictor() {
        let obs = [1.0, 2.0, 3.0, 4.0, 5.0];
        let mean = [3.0; 5];
        assert!(nash_sutcliffe(&obs, &mean).abs() < 1e-10);
    }

    #[test]
    fn test_ia_perfect() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0];
        assert!((index_of_agreement(&a, &a) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_ia_constant_bias() {
        let obs = [1.0, 2.0, 3.0, 4.0];
        let sim = [1.5, 2.5, 3.5, 4.5];
        let ia = index_of_agreement(&obs, &sim);
        assert!(ia > 0.9 && ia < 1.0, "IA={ia}");
    }

    #[test]
    fn test_r2_equals_nse() {
        let obs = [1.0, 2.5, 3.1, 4.7, 5.3];
        let sim = [1.1, 2.3, 3.4, 4.5, 5.5];
        assert!(
            (coefficient_of_determination(&obs, &sim) - nash_sutcliffe(&obs, &sim)).abs()
                < f64::EPSILON
        );
    }
}
