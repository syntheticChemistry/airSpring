// SPDX-License-Identifier: AGPL-3.0-or-later
//! Validation metrics: RMSE, MBE, R², IA, NSE, Pearson/Spearman correlation,
//! variance, and standard deviation.
//!
//! These wrap `barracuda::stats` primitives where possible and provide
//! additional metrics (IA, NSE, MBE) ported from the Python baselines.
//!
//! Staged for upstream absorption — see `metalForge/forge/src/metrics.rs`.

use crate::len_f64;

/// Compute Pearson correlation coefficient r.
///
/// Uses barracuda's `pearson_correlation` primitive. Returns 0.0
/// on degenerate data (constant series, length mismatch) rather than
/// propagating errors — suitable for validation binaries.
#[must_use]
pub fn pearson_r(x: &[f64], y: &[f64]) -> f64 {
    barracuda::stats::pearson_correlation(x, y).unwrap_or(0.0)
}

/// Compute Pearson R² between observed and simulated data.
///
/// Uses barracuda's `pearson_correlation` primitive for cross-validation.
///
/// # Errors
///
/// Returns [`crate::error::AirSpringError::Barracuda`] if the barracuda
/// primitive fails (e.g. length mismatch, degenerate data).
#[must_use = "R² value should be checked"]
pub fn r_squared(observed: &[f64], simulated: &[f64]) -> crate::error::Result<f64> {
    let r = barracuda::stats::pearson_correlation(observed, simulated)
        .map_err(|e| crate::error::AirSpringError::Barracuda(format!("{e}")))?;
    Ok(r * r)
}

/// Compute Root Mean Square Error (RMSE).
///
/// RMSE = sqrt(Σ(obs - sim)² / n)
///
/// # Panics
///
/// Panics if `observed` and `simulated` have different lengths.
#[must_use]
pub fn rmse(observed: &[f64], simulated: &[f64]) -> f64 {
    assert_eq!(
        observed.len(),
        simulated.len(),
        "Vectors must be same length"
    );
    let n = len_f64(observed);
    let sum_sq: f64 = observed
        .iter()
        .zip(simulated.iter())
        .map(|(o, s)| (o - s).powi(2))
        .sum();
    (sum_sq / n).sqrt()
}

/// Compute Mean Bias Error (MBE).
///
/// MBE = Σ(sim - obs) / n
///
/// # Panics
///
/// Panics if `observed` and `simulated` have different lengths.
#[must_use]
pub fn mbe(observed: &[f64], simulated: &[f64]) -> f64 {
    assert_eq!(
        observed.len(),
        simulated.len(),
        "Vectors must be same length"
    );
    let n = len_f64(observed);
    let sum_bias: f64 = observed
        .iter()
        .zip(simulated.iter())
        .map(|(o, s)| s - o)
        .sum();
    sum_bias / n
}

/// Index of Agreement (Willmott, 1981).
///
/// IA = 1 − Σ(Mᵢ − Pᵢ)² / Σ(|Pᵢ − M̄| + |Mᵢ − M̄|)²
///
/// Ported from the Python baseline (`control/soil_sensors/calibration_dong2020.py`
/// `compute_ia`).  Values range from 0.0 (no agreement) to 1.0 (perfect).
///
/// # Panics
///
/// Panics if `observed` and `simulated` have different lengths or are empty.
#[must_use]
pub fn index_of_agreement(observed: &[f64], simulated: &[f64]) -> f64 {
    assert_eq!(
        observed.len(),
        simulated.len(),
        "Vectors must be same length"
    );
    assert!(!observed.is_empty(), "Vectors must not be empty");

    let n = len_f64(observed);
    let mean_obs: f64 = observed.iter().sum::<f64>() / n;

    let numerator: f64 = observed
        .iter()
        .zip(simulated.iter())
        .map(|(o, s)| (o - s).powi(2))
        .sum();

    let denominator: f64 = observed
        .iter()
        .zip(simulated.iter())
        .map(|(o, s)| ((s - mean_obs).abs() + (o - mean_obs).abs()).powi(2))
        .sum();

    if denominator == 0.0 {
        return 1.0;
    }
    1.0 - numerator / denominator
}

/// Nash-Sutcliffe Efficiency (NSE).
///
/// NSE = 1 − Σ(Obsᵢ − Simᵢ)² / Σ(Obsᵢ − Obs̄)²
///
/// Widely used in hydrology (Nash & Sutcliffe, 1970). NSE = 1.0 is perfect
/// agreement; NSE < 0 means the model is worse than using the mean.
///
/// # Panics
///
/// Panics if `observed` and `simulated` have different lengths or are empty.
#[must_use]
pub fn nash_sutcliffe(observed: &[f64], simulated: &[f64]) -> f64 {
    assert_eq!(
        observed.len(),
        simulated.len(),
        "Vectors must be same length"
    );
    assert!(!observed.is_empty(), "Vectors must not be empty");

    let n = len_f64(observed);
    let mean_obs: f64 = observed.iter().sum::<f64>() / n;

    let ss_res: f64 = observed
        .iter()
        .zip(simulated.iter())
        .map(|(o, s)| (o - s).powi(2))
        .sum();

    let ss_tot: f64 = observed.iter().map(|o| (o - mean_obs).powi(2)).sum();

    if ss_tot == 0.0 {
        return 1.0;
    }
    1.0 - ss_res / ss_tot
}

/// Coefficient of determination (R²) using sum-of-squares method.
///
/// R² = 1 − `SS_res` / `SS_tot`
///
/// Unlike [`r_squared`], which wraps barracuda's Pearson R, this uses the
/// standard regression definition: it can be negative if the model is poor.
///
/// # Panics
///
/// Panics if `observed` and `simulated` have different lengths or are empty.
#[must_use]
pub fn coefficient_of_determination(observed: &[f64], simulated: &[f64]) -> f64 {
    nash_sutcliffe(observed, simulated)
}

/// Spearman rank correlation coefficient.
///
/// Wraps [`barracuda::stats::correlation::spearman_correlation`] for
/// nonparametric validation — useful when the relationship between
/// observed and simulated data may not be strictly linear.
///
/// Returns a value in \[-1, 1\].
///
/// # Errors
///
/// Returns [`crate::error::AirSpringError::Barracuda`] on failure.
#[must_use = "Spearman ρ value should be checked"]
pub fn spearman_r(observed: &[f64], simulated: &[f64]) -> crate::error::Result<f64> {
    barracuda::stats::correlation::spearman_correlation(observed, simulated)
        .map_err(|e| crate::error::AirSpringError::Barracuda(format!("{e}")))
}

/// Compute sample variance using barracuda's implementation.
///
/// Wraps [`barracuda::stats::correlation::variance`] for consistency
/// with the barracuda ecosystem.
///
/// # Errors
///
/// Returns [`crate::error::AirSpringError::Barracuda`] on failure.
#[must_use = "variance value should be checked"]
pub fn variance(data: &[f64]) -> crate::error::Result<f64> {
    barracuda::stats::correlation::variance(data)
        .map_err(|e| crate::error::AirSpringError::Barracuda(format!("{e}")))
}

/// Compute sample standard deviation using barracuda's implementation.
///
/// Wraps [`barracuda::stats::correlation::std_dev`].
///
/// # Errors
///
/// Returns [`crate::error::AirSpringError::Barracuda`] on failure.
#[must_use = "standard deviation value should be checked"]
pub fn std_deviation(data: &[f64]) -> crate::error::Result<f64> {
    barracuda::stats::correlation::std_dev(data)
        .map_err(|e| crate::error::AirSpringError::Barracuda(format!("{e}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── RMSE ─────────────────────────────────────────────────────────────

    #[test]
    fn rmse_identical_vectors_zero() {
        let obs = [1.0, 2.0, 3.0, 4.0, 5.0];
        let rmse_val = rmse(&obs, &obs);
        assert!(rmse_val < f64::EPSILON, "identical vectors RMSE={rmse_val}");
    }

    #[test]
    fn rmse_known_hand_calculated() {
        let obs = [1.0, 2.0, 3.0];
        let sim = [1.0, 3.0, 3.0];
        let rmse_val = rmse(&obs, &sim);
        let expected = (1.0_f64 / 3.0).sqrt();
        assert!(
            (rmse_val - expected).abs() < 1e-10,
            "RMSE={rmse_val} expected {expected}"
        );
    }

    #[test]
    fn rmse_single_element() {
        let obs = [7.0];
        let sim = [9.0];
        let rmse_val = rmse(&obs, &sim);
        assert!((rmse_val - 2.0).abs() < 1e-10, "RMSE={rmse_val}");
    }

    #[test]
    fn rmse_two_elements() {
        let obs = [0.0, 4.0];
        let sim = [2.0, 2.0];
        let rmse_val = rmse(&obs, &sim);
        assert!((rmse_val - 2.0).abs() < 1e-10, "RMSE={rmse_val}");
    }

    #[test]
    fn rmse_negative_values() {
        let obs = [-1.0, -2.0, -3.0];
        let sim = [-1.0, -4.0, -3.0];
        let rmse_val = rmse(&obs, &sim);
        let expected = (4.0_f64 / 3.0).sqrt();
        assert!((rmse_val - expected).abs() < 1e-10, "RMSE={rmse_val}");
    }

    // ── MBE ───────────────────────────────────────────────────────────────

    #[test]
    fn mbe_identical_vectors_zero() {
        let obs = [1.0, 2.0, 3.0];
        let mbe_val = mbe(&obs, &obs);
        assert!(mbe_val.abs() < f64::EPSILON, "MBE={mbe_val}");
    }

    #[test]
    fn mbe_known_positive_bias() {
        let obs = [1.0, 2.0, 3.0];
        let sim = [2.0, 3.0, 4.0];
        let mbe_val = mbe(&obs, &sim);
        assert!((mbe_val - 1.0).abs() < 1e-10, "MBE={mbe_val}");
    }

    #[test]
    fn mbe_negative_bias() {
        let obs = [5.0, 6.0, 7.0];
        let sim = [4.0, 5.0, 6.0];
        let mbe_val = mbe(&obs, &sim);
        assert!((mbe_val + 1.0).abs() < 1e-10, "MBE={mbe_val}");
    }

    #[test]
    fn mbe_single_element() {
        let obs = [10.0];
        let sim = [12.0];
        let mbe_val = mbe(&obs, &sim);
        assert!((mbe_val - 2.0).abs() < 1e-10, "MBE={mbe_val}");
    }

    // ── Pearson r ──────────────────────────────────────────────────────────

    #[test]
    fn pearson_r_perfect_correlation() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let y = [2.0, 4.0, 6.0, 8.0, 10.0];
        let r = pearson_r(&x, &y);
        assert!((r - 1.0).abs() < 1e-10, "perfect linear r={r}");
    }

    #[test]
    fn pearson_r_anti_correlation() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let y = [5.0, 4.0, 3.0, 2.0, 1.0];
        let r = pearson_r(&x, &y);
        assert!((r + 1.0).abs() < 1e-10, "perfect anti-correlation r={r}");
    }

    #[test]
    fn pearson_r_identical_vectors() {
        let x = [1.0, 2.0, 3.0];
        let r = pearson_r(&x, &x);
        assert!((r - 1.0).abs() < 1e-10, "identical r={r}");
    }

    #[test]
    fn pearson_r_degenerate_constant_returns_zero_or_nan() {
        let x = [1.0, 1.0, 1.0];
        let y = [2.0, 3.0, 4.0];
        let r = pearson_r(&x, &y);
        assert!(
            r.abs() < f64::EPSILON || r.is_nan(),
            "constant series returns 0 or NaN, got r={r}"
        );
    }

    #[test]
    fn pearson_r_zero_correlation() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let y = [1.0, 1.0, 1.0, 1.0, 1.0];
        let r = pearson_r(&x, &y);
        assert!(
            r.abs() < f64::EPSILON || r.is_nan(),
            "orthogonal/constant y returns 0 or NaN, got r={r}"
        );
    }

    // ── R² (Result-returning) ────────────────────────────────────────────

    #[test]
    fn r_squared_perfect_match() {
        let obs = [1.0, 2.0, 3.0, 4.0, 5.0];
        let r2 = r_squared(&obs, &obs).unwrap();
        assert!((r2 - 1.0).abs() < 1e-10, "R²={r2}");
    }

    #[test]
    fn r_squared_known_linear() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let y = [2.0, 4.0, 6.0, 8.0, 10.0];
        let r2 = r_squared(&x, &y).unwrap();
        assert!((r2 - 1.0).abs() < 1e-10, "R²={r2}");
    }

    #[test]
    fn r_squared_length_mismatch_errors() {
        let a = [1.0, 2.0];
        let b = [1.0, 2.0, 3.0];
        assert!(r_squared(&a, &b).is_err());
    }

    // ── Index of Agreement ────────────────────────────────────────────────

    #[test]
    fn index_of_agreement_perfect_is_one() {
        let obs = [1.0, 2.0, 3.0, 4.0, 5.0];
        let ia = index_of_agreement(&obs, &obs);
        assert!((ia - 1.0).abs() < f64::EPSILON, "IA={ia}");
    }

    #[test]
    fn index_of_agreement_constant_obs_denominator_zero_returns_one() {
        let obs = [5.0, 5.0, 5.0];
        let sim = [5.0, 5.0, 5.0];
        let ia = index_of_agreement(&obs, &sim);
        assert!((ia - 1.0).abs() < f64::EPSILON, "IA={ia}");
    }

    #[test]
    fn index_of_agreement_negative_values() {
        let obs = [-2.0, -1.0, 0.0, 1.0, 2.0];
        let sim = [-2.0, -1.0, 0.0, 1.0, 2.0];
        let ia = index_of_agreement(&obs, &sim);
        assert!((ia - 1.0).abs() < 1e-10, "IA={ia}");
    }

    #[test]
    fn index_of_agreement_poor_model() {
        let obs = [1.0, 2.0, 3.0, 4.0, 5.0];
        let sim = [5.0, 4.0, 3.0, 2.0, 1.0];
        let ia = index_of_agreement(&obs, &sim);
        assert!((0.0..1.0).contains(&ia), "IA={ia}");
    }

    // ── Nash-Sutcliffe ───────────────────────────────────────────────────

    #[test]
    fn nash_sutcliffe_perfect_is_one() {
        let obs = [1.0, 2.0, 3.0, 4.0, 5.0];
        let nse = nash_sutcliffe(&obs, &obs);
        assert!((nse - 1.0).abs() < f64::EPSILON, "NSE={nse}");
    }

    #[test]
    fn nash_sutcliffe_constant_obs_ss_tot_zero_returns_one() {
        let obs = [3.0, 3.0, 3.0];
        let sim = [3.0, 3.0, 3.0];
        let nse = nash_sutcliffe(&obs, &sim);
        assert!((nse - 1.0).abs() < f64::EPSILON, "NSE={nse}");
    }

    #[test]
    fn nash_sutcliffe_worse_than_mean_negative() {
        let obs = [1.0, 2.0, 3.0, 4.0, 5.0];
        let sim = [5.0, 4.0, 3.0, 2.0, 1.0];
        let nse = nash_sutcliffe(&obs, &sim);
        assert!(
            nse < 0.0,
            "NSE should be negative for inverted model, got {nse}"
        );
    }

    #[test]
    fn nash_sutcliffe_mean_predictor_zero() {
        let obs = [1.0, 2.0, 3.0, 4.0, 5.0];
        let mean = 3.0;
        let pred = vec![mean; 5];
        let nse = nash_sutcliffe(&obs, &pred);
        assert!(nse.abs() < 1e-10, "NSE (mean predictor)={nse}");
    }

    // ── Spearman r ───────────────────────────────────────────────────────

    #[test]
    fn spearman_r_perfect_monotonic() {
        let x: Vec<f64> = (1..=10).map(f64::from).collect();
        let y: Vec<f64> = x.iter().map(|v| v * v).collect();
        let rho = spearman_r(&x, &y).unwrap();
        assert!((rho - 1.0).abs() < 1e-10, "ρ={rho}");
    }

    #[test]
    fn spearman_r_anti_correlation() {
        let x: Vec<f64> = (1..=10).map(f64::from).collect();
        let y: Vec<f64> = x.iter().rev().copied().collect();
        let rho = spearman_r(&x, &y).unwrap();
        assert!((rho + 1.0).abs() < 1e-10, "ρ={rho}");
    }

    #[test]
    fn spearman_r_identical_vectors() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let rho = spearman_r(&x, &x).unwrap();
        assert!((rho - 1.0).abs() < 1e-10, "ρ={rho}");
    }

    // ── coefficient_of_determination ─────────────────────────────────────

    #[test]
    fn coefficient_of_determination_equals_nse() {
        let obs = [1.0, 3.0, 5.0, 7.0, 9.0];
        let sim = [1.1, 2.9, 5.2, 6.8, 9.1];
        let r2 = coefficient_of_determination(&obs, &sim);
        let nse = nash_sutcliffe(&obs, &sim);
        assert!((r2 - nse).abs() < f64::EPSILON, "R²={r2} NSE={nse}");
    }

    #[test]
    fn coefficient_of_determination_can_be_negative() {
        let obs = [1.0, 2.0, 3.0, 4.0, 5.0];
        let sim = [5.0, 4.0, 3.0, 2.0, 1.0];
        let r2 = coefficient_of_determination(&obs, &sim);
        assert!(r2 < 0.0, "poor model R²={r2}");
    }

    // ── variance & std_deviation ─────────────────────────────────────────

    #[test]
    fn variance_known_values() {
        let data = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let var = variance(&data).unwrap();
        let expected = 4.571_428_571_428_571;
        assert!((var - expected).abs() < 1e-6, "var={var}");
    }

    #[test]
    fn std_deviation_equals_sqrt_variance() {
        let data = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let sd = std_deviation(&data).unwrap();
        let var = variance(&data).unwrap();
        assert!(
            (sd - var.sqrt()).abs() < 1e-12,
            "sd={sd} sqrt(var)={}",
            var.sqrt()
        );
    }

    #[test]
    fn variance_single_element_errors() {
        let data = [42.0];
        assert!(variance(&data).is_err(), "single element should return Err");
    }

    #[test]
    fn std_deviation_single_element_errors() {
        let data = [42.0];
        assert!(
            std_deviation(&data).is_err(),
            "single element should return Err"
        );
    }
}
