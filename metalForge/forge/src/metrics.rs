// SPDX-License-Identifier: AGPL-3.0-or-later
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

/// Errors from forge metric computations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ForgeError {
    /// Input slices have different lengths.
    LengthMismatch { expected: usize, got: usize },
    /// Input slices are empty.
    EmptyInput,
}

impl std::fmt::Display for ForgeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::LengthMismatch { expected, got } => {
                write!(f, "length mismatch: expected {expected}, got {got}")
            }
            Self::EmptyInput => write!(f, "empty input"),
        }
    }
}

impl std::error::Error for ForgeError {}

const fn validate_slices(observed: &[f64], simulated: &[f64]) -> Result<(), ForgeError> {
    if observed.len() != simulated.len() {
        return Err(ForgeError::LengthMismatch {
            expected: observed.len(),
            got: simulated.len(),
        });
    }
    if observed.is_empty() {
        return Err(ForgeError::EmptyInput);
    }
    Ok(())
}

/// Root Mean Square Error.
///
/// RMSE = √(Σ(obsᵢ − simᵢ)² / n)
///
/// # Examples
///
/// ```
/// use airspring_forge::metrics::rmse;
///
/// let obs = [1.0, 2.0, 3.0];
/// let sim = [1.1, 2.1, 3.1];
/// let r = rmse(&obs, &sim).unwrap();
/// assert!((r - 0.1).abs() < 1e-10);
/// ```
///
/// # Errors
///
/// Returns [`ForgeError::LengthMismatch`] if slices have different lengths,
/// or [`ForgeError::EmptyInput`] if either slice is empty.
pub fn rmse(observed: &[f64], simulated: &[f64]) -> Result<f64, ForgeError> {
    validate_slices(observed, simulated)?;
    let n = len_f64(observed);
    let ss: f64 = observed
        .iter()
        .zip(simulated)
        .map(|(o, s)| (o - s).powi(2))
        .sum();
    Ok((ss / n).sqrt())
}

/// Mean Bias Error.
///
/// MBE = Σ(simᵢ − obsᵢ) / n
///
/// Positive MBE indicates over-prediction.
///
/// # Examples
///
/// ```
/// use airspring_forge::metrics::mbe;
///
/// let obs = [1.0, 2.0, 3.0];
/// let sim = [1.5, 2.5, 3.5];
/// let b = mbe(&obs, &sim).unwrap();
/// assert!((b - 0.5).abs() < 1e-10);
/// ```
///
/// # Errors
///
/// Returns [`ForgeError::LengthMismatch`] if slices have different lengths,
/// or [`ForgeError::EmptyInput`] if either slice is empty.
pub fn mbe(observed: &[f64], simulated: &[f64]) -> Result<f64, ForgeError> {
    validate_slices(observed, simulated)?;
    let n = len_f64(observed);
    let bias: f64 = observed.iter().zip(simulated).map(|(o, s)| s - o).sum();
    Ok(bias / n)
}

/// Nash-Sutcliffe Efficiency (Nash & Sutcliffe, 1970).
///
/// NSE = 1 − Σ(obsᵢ − simᵢ)² / Σ(obsᵢ − obs̄)²
///
/// NSE = 1.0 is perfect; NSE = 0.0 means the model is no better than the
/// mean; NSE < 0 means the model is worse than the mean.
///
/// # Examples
///
/// ```
/// use airspring_forge::metrics::nash_sutcliffe;
///
/// let obs = [1.0, 2.0, 3.0, 4.0, 5.0];
/// let sim = [1.0, 2.0, 3.0, 4.0, 5.0];
/// let nse = nash_sutcliffe(&obs, &sim).unwrap();
/// assert!((nse - 1.0).abs() < 1e-10);
/// ```
///
/// # Errors
///
/// Returns [`ForgeError::LengthMismatch`] if slices have different lengths,
/// or [`ForgeError::EmptyInput`] if either slice is empty.
pub fn nash_sutcliffe(observed: &[f64], simulated: &[f64]) -> Result<f64, ForgeError> {
    validate_slices(observed, simulated)?;

    let n = len_f64(observed);
    let mean_obs: f64 = observed.iter().sum::<f64>() / n;

    let ss_res: f64 = observed
        .iter()
        .zip(simulated)
        .map(|(o, s)| (o - s).powi(2))
        .sum();
    let ss_tot: f64 = observed.iter().map(|o| (o - mean_obs).powi(2)).sum();

    if ss_tot == 0.0 {
        return Ok(1.0);
    }
    Ok(1.0 - ss_res / ss_tot)
}

/// Index of Agreement (Willmott, 1981).
///
/// IA = 1 − Σ(obsᵢ − simᵢ)² / Σ(|simᵢ − obs̄| + |obsᵢ − obs̄|)²
///
/// Values range from 0.0 (no agreement) to 1.0 (perfect).
///
/// # Examples
///
/// ```
/// use airspring_forge::metrics::index_of_agreement;
///
/// let obs = [1.0, 2.0, 3.0, 4.0];
/// let sim = [1.0, 2.0, 3.0, 4.0];
/// let ia = index_of_agreement(&obs, &sim).unwrap();
/// assert!((ia - 1.0).abs() < 1e-10);
/// ```
///
/// # Errors
///
/// Returns [`ForgeError::LengthMismatch`] if slices have different lengths,
/// or [`ForgeError::EmptyInput`] if either slice is empty.
pub fn index_of_agreement(observed: &[f64], simulated: &[f64]) -> Result<f64, ForgeError> {
    validate_slices(observed, simulated)?;

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
        return Ok(1.0);
    }
    Ok(1.0 - numerator / denominator)
}

/// Coefficient of determination (R²) via sum-of-squares.
///
/// R² = 1 − `SS_res` / `SS_tot`
///
/// Equivalent to [`nash_sutcliffe`] — provided as a named alias for
/// domains where "R²" is the conventional term.
///
/// # Examples
///
/// ```
/// use airspring_forge::metrics::coefficient_of_determination;
///
/// let obs = [1.0, 2.0, 3.0];
/// let sim = [1.1, 2.1, 2.9];
/// let r2 = coefficient_of_determination(&obs, &sim).unwrap();
/// assert!(r2 > 0.9 && r2 <= 1.0);
/// ```
///
/// # Errors
///
/// Returns [`ForgeError::LengthMismatch`] if slices have different lengths,
/// or [`ForgeError::EmptyInput`] if either slice is empty.
pub fn coefficient_of_determination(
    observed: &[f64],
    simulated: &[f64],
) -> Result<f64, ForgeError> {
    nash_sutcliffe(observed, simulated)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rmse_perfect() {
        let a = [1.0, 2.0, 3.0];
        assert!(rmse(&a, &a).unwrap() < f64::EPSILON);
    }

    #[test]
    fn test_rmse_known() {
        let obs = [1.0, 2.0, 3.0, 4.0];
        let sim = [1.1, 2.1, 2.9, 3.9];
        assert!((rmse(&obs, &sim).unwrap() - 0.1).abs() < 1e-10);
    }

    #[test]
    fn test_mbe_positive_bias() {
        let obs = [1.0, 2.0, 3.0];
        let sim = [1.5, 2.5, 3.5];
        assert!((mbe(&obs, &sim).unwrap() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_mbe_zero() {
        let obs = [1.0, 2.0, 3.0];
        let sim = [0.9, 2.1, 3.0];
        assert!(mbe(&obs, &sim).unwrap().abs() < 1e-10);
    }

    #[test]
    fn test_nse_perfect() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0];
        assert!((nash_sutcliffe(&a, &a).unwrap() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_nse_mean_predictor() {
        let obs = [1.0, 2.0, 3.0, 4.0, 5.0];
        let mean = [3.0; 5];
        assert!(nash_sutcliffe(&obs, &mean).unwrap().abs() < 1e-10);
    }

    #[test]
    fn test_ia_perfect() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0];
        assert!((index_of_agreement(&a, &a).unwrap() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_ia_constant_bias() {
        let obs = [1.0, 2.0, 3.0, 4.0];
        let sim = [1.5, 2.5, 3.5, 4.5];
        let ia = index_of_agreement(&obs, &sim).unwrap();
        assert!(ia > 0.9 && ia < 1.0, "IA={ia}");
    }

    #[test]
    fn test_r2_equals_nse() {
        let obs = [1.0, 2.5, 3.1, 4.7, 5.3];
        let sim = [1.1, 2.3, 3.4, 4.5, 5.5];
        assert!(
            (coefficient_of_determination(&obs, &sim).unwrap()
                - nash_sutcliffe(&obs, &sim).unwrap())
            .abs()
                < f64::EPSILON
        );
    }

    #[test]
    fn test_length_mismatch() {
        let obs = [1.0, 2.0, 3.0];
        let sim = [1.0, 2.0];
        assert_eq!(
            rmse(&obs, &sim).unwrap_err(),
            ForgeError::LengthMismatch {
                expected: 3,
                got: 2
            }
        );
    }

    #[test]
    fn test_empty_input() {
        let obs: [f64; 0] = [];
        let sim: [f64; 0] = [];
        assert_eq!(rmse(&obs, &sim).unwrap_err(), ForgeError::EmptyInput);
    }
}
