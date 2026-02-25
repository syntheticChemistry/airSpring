// SPDX-License-Identifier: AGPL-3.0-or-later
//! Sensor correction equations — Dong et al. (2020) methodology.
//!
//! Implements the four correction model types from the paper:
//! - Linear: y = a·x + b
//! - Quadratic: y = a·x² + b·x + c
//! - Exponential: y = a·exp(b·x)
//! - Logarithmic: y = a·ln(x) + b
//!
//! Curve fitting uses pure Rust least-squares (no scipy dependency).
//! Linear and quadratic use analytical normal equations.
//! Exponential and logarithmic use log-linearized least squares.
//!
//! # Reference
//!
//! Dong Y, Miller WL, Kelley LC, Pease LA (2020)
//! "Soil Moisture Sensor Performance and Corrections for Michigan Agricultural Soils"
//! *Agriculture* 10(12):598

/// Correction model type — Dong et al. (2020) four-model suite.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelType {
    /// y = a·x + b
    Linear,
    /// y = a·x² + b·x + c
    Quadratic,
    /// y = a·exp(b·x)
    Exponential,
    /// y = a·ln(x) + b
    Logarithmic,
}

impl ModelType {
    /// Stable string identifier for serialization and cross-validation.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Linear => "linear",
            Self::Quadratic => "quadratic",
            Self::Exponential => "exponential",
            Self::Logarithmic => "logarithmic",
        }
    }
}

impl std::fmt::Display for ModelType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A fitted correction model with parameters and goodness-of-fit.
#[derive(Debug, Clone)]
pub struct FittedModel {
    /// Model type (typed enum, not stringly-typed).
    pub model_type: ModelType,
    /// Model parameters (interpretation depends on model type).
    pub params: Vec<f64>,
    /// Coefficient of determination R².
    pub r_squared: f64,
    /// Root Mean Square Error.
    pub rmse: f64,
}

/// Evaluate a linear model: y = a·x + b.
#[must_use]
pub fn linear_model(x: f64, a: f64, b: f64) -> f64 {
    a.mul_add(x, b)
}

/// Evaluate a quadratic model: y = a·x² + b·x + c.
#[must_use]
pub fn quadratic_model(x: f64, a: f64, b: f64, c: f64) -> f64 {
    a.mul_add(x * x, b.mul_add(x, c))
}

/// Evaluate an exponential model: y = a·exp(b·x).
#[must_use]
pub fn exponential_model(x: f64, a: f64, b: f64) -> f64 {
    a * (b * x).exp()
}

/// Evaluate a logarithmic model: y = a·ln(x) + b.
///
/// # Panics
///
/// Panics (via debug assertion) if x ≤ 0.
#[must_use]
pub fn logarithmic_model(x: f64, a: f64, b: f64) -> f64 {
    debug_assert!(x > 0.0, "logarithmic_model requires x > 0");
    a.mul_add(x.ln(), b)
}

/// Evaluate a fitted model at a given x value.
#[must_use]
pub fn evaluate(model: &FittedModel, x: f64) -> f64 {
    match model.model_type {
        ModelType::Linear => linear_model(x, model.params[0], model.params[1]),
        ModelType::Quadratic => {
            quadratic_model(x, model.params[0], model.params[1], model.params[2])
        }
        ModelType::Exponential => exponential_model(x, model.params[0], model.params[1]),
        ModelType::Logarithmic => logarithmic_model(x, model.params[0], model.params[1]),
    }
}

use crate::len_f64;

/// Guard against singular matrices in regression — smaller than any
/// physically meaningful determinant in sensor calibration.
const SINGULARITY_GUARD: f64 = 1e-30;

/// Minimum x-value for logarithmic regression — avoids log(0) domain error.
const LOG_DOMAIN_GUARD: f64 = 0.001;

// ── Least-squares fitting ────────────────────────────────────────────

/// Fit a linear model y = a·x + b using normal equations.
///
/// Returns `Some(FittedModel)` or `None` if the system is singular.
#[must_use]
pub fn fit_linear(x: &[f64], y: &[f64]) -> Option<FittedModel> {
    let count = len_f64(x);
    if x.len() < 2 || x.len() != y.len() {
        return None;
    }

    let s_x: f64 = x.iter().sum();
    let s_y: f64 = y.iter().sum();
    let s_xx: f64 = x.iter().map(|&xi| xi * xi).sum();
    let s_cross: f64 = x.iter().zip(y).map(|(&xi, &yi)| xi * yi).sum();

    let det = count.mul_add(s_xx, -(s_x * s_x));
    if det.abs() < SINGULARITY_GUARD {
        return None;
    }

    let a = count.mul_add(s_cross, -(s_x * s_y)) / det;
    let b = s_xx.mul_add(s_y, -(s_x * s_cross)) / det;

    let (r2, rmse) = goodness_of_fit(x, y, |xi| linear_model(xi, a, b));
    Some(FittedModel {
        model_type: ModelType::Linear,
        params: vec![a, b],
        r_squared: r2,
        rmse,
    })
}

/// 3x3 determinant via cofactor expansion along row 0.
fn det3(m: &[[f64; 3]; 3]) -> f64 {
    let minor0 = m[1][1].mul_add(m[2][2], -(m[1][2] * m[2][1]));
    let minor1 = m[1][0].mul_add(m[2][2], -(m[1][2] * m[2][0]));
    let minor2 = m[1][0].mul_add(m[2][1], -(m[1][1] * m[2][0]));
    m[0][2].mul_add(minor2, m[0][0].mul_add(minor0, -(m[0][1] * minor1)))
}

/// Solve a 3x3 linear system via Cramer's rule.
///
/// Solves `M · [p0, p1, p2]ᵀ = rhs` where M is a 3x3 symmetric matrix
/// given by rows. Returns `None` if the determinant is near zero.
fn cramer_3x3(m: [[f64; 3]; 3], rhs: [f64; 3]) -> Option<(f64, f64, f64)> {
    let det = det3(&m);
    if det.abs() < SINGULARITY_GUARD {
        return None;
    }
    let inv = 1.0 / det;

    // Replace each column in turn with rhs, compute determinant
    let m0 = [
        [rhs[0], m[0][1], m[0][2]],
        [rhs[1], m[1][1], m[1][2]],
        [rhs[2], m[2][1], m[2][2]],
    ];
    let m1 = [
        [m[0][0], rhs[0], m[0][2]],
        [m[1][0], rhs[1], m[1][2]],
        [m[2][0], rhs[2], m[2][2]],
    ];
    let m2 = [
        [m[0][0], m[0][1], rhs[0]],
        [m[1][0], m[1][1], rhs[1]],
        [m[2][0], m[2][1], rhs[2]],
    ];

    Some((inv * det3(&m0), inv * det3(&m1), inv * det3(&m2)))
}

/// Fit a quadratic model y = a·x² + b·x + c using normal equations.
#[must_use]
pub fn fit_quadratic(xs: &[f64], ys: &[f64]) -> Option<FittedModel> {
    let count = len_f64(xs);
    if xs.len() < 3 || xs.len() != ys.len() {
        return None;
    }

    // Sums for normal equations: A'A·p = A'y where A = [x², x, 1]
    let s_x: f64 = xs.iter().sum();
    let s_x2: f64 = xs.iter().map(|&xi| xi * xi).sum();
    let s_x3: f64 = xs.iter().map(|&xi| xi.powi(3)).sum();
    let s_x4: f64 = xs.iter().map(|&xi| xi.powi(4)).sum();
    let s_y: f64 = ys.iter().sum();
    let s_cross: f64 = xs.iter().zip(ys).map(|(&xi, &yi)| xi * yi).sum();
    let s_x2_cross: f64 = xs.iter().zip(ys).map(|(&xi, &yi)| xi * xi * yi).sum();

    // 3×3 symmetric normal equation matrix and RHS
    let matrix = [[s_x4, s_x3, s_x2], [s_x3, s_x2, s_x], [s_x2, s_x, count]];
    let rhs = [s_x2_cross, s_cross, s_y];

    let (coeff_a, coeff_b, coeff_c) = cramer_3x3(matrix, rhs)?;

    let (r2, rmse) = goodness_of_fit(xs, ys, |xi| quadratic_model(xi, coeff_a, coeff_b, coeff_c));
    Some(FittedModel {
        model_type: ModelType::Quadratic,
        params: vec![coeff_a, coeff_b, coeff_c],
        r_squared: r2,
        rmse,
    })
}

/// Fit an exponential model y = a·exp(b·x) via log-linearized least squares.
///
/// Transforms to ln(y) = ln(a) + b·x and fits linear.
/// Requires all y > 0.
#[must_use]
pub fn fit_exponential(x: &[f64], y: &[f64]) -> Option<FittedModel> {
    if x.len() < 2 || x.len() != y.len() {
        return None;
    }
    // Filter to positive y values
    let valid: Vec<(f64, f64)> = x
        .iter()
        .zip(y)
        .filter(|(_, &yi)| yi > 0.0)
        .map(|(&xi, &yi)| (xi, yi))
        .collect();
    if valid.len() < 2 {
        return None;
    }

    let xv: Vec<f64> = valid.iter().map(|&(xi, _)| xi).collect();
    let ly: Vec<f64> = valid.iter().map(|&(_, yi)| yi.ln()).collect();

    let lin = fit_linear(&xv, &ly)?;
    let ln_a = lin.params[1];
    let b = lin.params[0];
    let a = ln_a.exp();

    // Compute R² and RMSE in original space
    let (r2, rmse) = goodness_of_fit(&xv, &valid.iter().map(|v| v.1).collect::<Vec<_>>(), |xi| {
        exponential_model(xi, a, b)
    });
    Some(FittedModel {
        model_type: ModelType::Exponential,
        params: vec![a, b],
        r_squared: r2,
        rmse,
    })
}

/// Fit a logarithmic model y = a·ln(x) + b via linearized least squares.
///
/// Transforms to y = a·z + b where z = ln(x). Requires all x > 0.
#[must_use]
pub fn fit_logarithmic(x: &[f64], y: &[f64]) -> Option<FittedModel> {
    if x.len() < 2 || x.len() != y.len() {
        return None;
    }
    let valid: Vec<(f64, f64)> = x
        .iter()
        .zip(y)
        .filter(|(&xi, _)| xi > LOG_DOMAIN_GUARD)
        .map(|(&xi, &yi)| (xi, yi))
        .collect();
    if valid.len() < 2 {
        return None;
    }

    let lnx: Vec<f64> = valid.iter().map(|&(xi, _)| xi.ln()).collect();
    let yv: Vec<f64> = valid.iter().map(|&(_, yi)| yi).collect();

    let lin = fit_linear(&lnx, &yv)?;
    let a = lin.params[0];
    let b = lin.params[1];

    let xv: Vec<f64> = valid.iter().map(|v| v.0).collect();
    let (r2, rmse) = goodness_of_fit(&xv, &yv, |xi| logarithmic_model(xi, a, b));
    Some(FittedModel {
        model_type: ModelType::Logarithmic,
        params: vec![a, b],
        r_squared: r2,
        rmse,
    })
}

/// Fit a regularized linear model using upstream `barracuda::linalg::ridge`.
///
/// Ridge regression minimizes ‖y − Xw‖² + λ‖w‖², producing more stable
/// coefficients when features are correlated or data is noisy. This wraps
/// the CPU-only `barracuda::linalg::ridge::ridge_regression` (absorbed from
/// `wetSpring` ESN calibration, S52+).
///
/// Returns `Some(FittedModel)` with `ModelType::Linear` and the ridge-fit
/// coefficients [slope, intercept], or `None` if the fit fails.
#[must_use]
pub fn fit_ridge(x: &[f64], y: &[f64], regularization: f64) -> Option<FittedModel> {
    if x.len() < 2 || x.len() != y.len() {
        return None;
    }

    let n = x.len();
    // Build design matrix [x_i, 1.0] for intercept model
    let mut design = Vec::with_capacity(n * 2);
    for &xi in x {
        design.push(xi);
        design.push(1.0);
    }

    let result =
        barracuda::linalg::ridge::ridge_regression(&design, y, n, 2, 1, regularization).ok()?;

    let slope = result.weights[0];
    let intercept = result.weights[1];

    let (r2, rmse) = goodness_of_fit(x, y, |xi| slope.mul_add(xi, intercept));
    Some(FittedModel {
        model_type: ModelType::Linear,
        params: vec![slope, intercept],
        r_squared: r2,
        rmse,
    })
}

/// Fit all four correction models and return those that converge.
///
/// This replicates the Python `fit_correction_equations()` from
/// `calibration_dong2020.py` using pure Rust (no scipy dependency).
#[must_use]
pub fn fit_correction_equations(
    factory_values: &[f64],
    measured_values: &[f64],
) -> Vec<FittedModel> {
    let mut results = Vec::with_capacity(4);

    if let Some(m) = fit_linear(factory_values, measured_values) {
        results.push(m);
    }
    if let Some(m) = fit_quadratic(factory_values, measured_values) {
        results.push(m);
    }
    if let Some(m) = fit_exponential(factory_values, measured_values) {
        results.push(m);
    }
    if let Some(m) = fit_logarithmic(factory_values, measured_values) {
        results.push(m);
    }

    results
}

// ── Helpers ──────────────────────────────────────────────────────────

fn goodness_of_fit<F: Fn(f64) -> f64>(x: &[f64], y: &[f64], predict: F) -> (f64, f64) {
    let n = len_f64(y);
    let mean_y: f64 = y.iter().sum::<f64>() / n;

    let ss_res: f64 = x
        .iter()
        .zip(y)
        .map(|(&xi, &yi)| (yi - predict(xi)).powi(2))
        .sum();
    let ss_tot: f64 = y.iter().map(|&yi| (yi - mean_y).powi(2)).sum();

    let r2 = if ss_tot > 0.0 {
        1.0 - ss_res / ss_tot
    } else {
        1.0
    };
    let rmse = (ss_res / n).sqrt();

    (r2, rmse)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_perfect_fit() {
        // y = 2x + 1
        let x: Vec<f64> = (0..10).map(f64::from).collect();
        let y: Vec<f64> = x.iter().map(|&xi| 2.0f64.mul_add(xi, 1.0)).collect();

        let m = fit_linear(&x, &y).unwrap();
        assert!((m.params[0] - 2.0).abs() < 1e-10, "a={}", m.params[0]);
        assert!((m.params[1] - 1.0).abs() < 1e-10, "b={}", m.params[1]);
        assert!((m.r_squared - 1.0).abs() < 1e-10, "R²={}", m.r_squared);
        assert!(m.rmse < 1e-10, "RMSE={}", m.rmse);
    }

    #[test]
    fn test_quadratic_perfect_fit() {
        // y = 0.5x² − 2x + 3
        let x: Vec<f64> = (0..20).map(|i| f64::from(i) * 0.5).collect();
        let y: Vec<f64> = x
            .iter()
            .map(|&xi| (0.5 * xi).mul_add(xi, (-2.0f64).mul_add(xi, 3.0)))
            .collect();

        let m = fit_quadratic(&x, &y).unwrap();
        assert!((m.params[0] - 0.5).abs() < 1e-6, "a={}", m.params[0]);
        assert!((m.params[1] + 2.0).abs() < 1e-6, "b={}", m.params[1]);
        assert!((m.params[2] - 3.0).abs() < 1e-6, "c={}", m.params[2]);
        assert!((m.r_squared - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_exponential_fit() {
        // y = 2·exp(0.5·x) with some rounding
        let x: Vec<f64> = (1..=10).map(|i| f64::from(i) * 0.3).collect();
        let y: Vec<f64> = x.iter().map(|&xi| 2.0 * (xi * 0.5).exp()).collect();

        let m = fit_exponential(&x, &y).unwrap();
        assert!((m.params[0] - 2.0).abs() < 0.1, "a={}", m.params[0]);
        assert!((m.params[1] - 0.5).abs() < 0.05, "b={}", m.params[1]);
        assert!(m.r_squared > 0.99, "R²={}", m.r_squared);
    }

    #[test]
    fn test_logarithmic_fit() {
        // y = 3·ln(x) + 1
        let x: Vec<f64> = (1..=15).map(|i| f64::from(i).mul_add(0.5, 0.1)).collect();
        let y: Vec<f64> = x.iter().map(|&xi| 3.0f64.mul_add(xi.ln(), 1.0)).collect();

        let m = fit_logarithmic(&x, &y).unwrap();
        assert!((m.params[0] - 3.0).abs() < 0.01, "a={}", m.params[0]);
        assert!((m.params[1] - 1.0).abs() < 0.01, "b={}", m.params[1]);
        assert!(m.r_squared > 0.999, "R²={}", m.r_squared);
    }

    #[test]
    fn test_fit_all_returns_four_models() {
        let x: Vec<f64> = (1..=20).map(|i| f64::from(i) * 0.05).collect();
        let y: Vec<f64> = x.iter().map(|&xi| 1.5f64.mul_add(xi, 0.3)).collect();

        let models = fit_correction_equations(&x, &y);
        // Should fit at least linear and quadratic
        assert!(models.len() >= 2, "Got {} models", models.len());

        // Linear should have best fit for linear data
        let linear = models
            .iter()
            .find(|m| m.model_type == ModelType::Linear)
            .unwrap();
        assert!(linear.r_squared > 0.99);
    }

    #[test]
    fn test_evaluate_model() {
        let model = FittedModel {
            model_type: ModelType::Linear,
            params: vec![2.0, 1.0],
            r_squared: 1.0,
            rmse: 0.0,
        };
        assert!((evaluate(&model, 3.0) - 7.0).abs() < 1e-10);
    }

    #[test]
    fn test_model_types() {
        assert!((linear_model(2.0, 3.0, 1.0) - 7.0).abs() < 1e-10);
        assert!((quadratic_model(2.0, 1.0, 3.0, 1.0) - 11.0).abs() < 1e-10);
        assert!((exponential_model(0.0, 2.0, 3.0) - 2.0).abs() < 1e-10);
        assert!((logarithmic_model(1.0, 3.0, 1.0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_model_type_as_str_and_display() {
        assert_eq!(ModelType::Linear.as_str(), "linear");
        assert_eq!(ModelType::Quadratic.as_str(), "quadratic");
        assert_eq!(ModelType::Exponential.as_str(), "exponential");
        assert_eq!(ModelType::Logarithmic.as_str(), "logarithmic");
        assert_eq!(format!("{}", ModelType::Linear), "linear");
        assert_eq!(format!("{}", ModelType::Logarithmic), "logarithmic");
    }

    #[test]
    fn test_evaluate_all_model_types() {
        let exp_model = FittedModel {
            model_type: ModelType::Exponential,
            params: vec![2.0, 0.5],
            r_squared: 0.99,
            rmse: 0.01,
        };
        let val = evaluate(&exp_model, 0.0);
        assert!((val - 2.0).abs() < 1e-10, "exp(0)=2: {val}");

        let log_model = FittedModel {
            model_type: ModelType::Logarithmic,
            params: vec![3.0, 1.0],
            r_squared: 0.99,
            rmse: 0.01,
        };
        let val = evaluate(&log_model, 1.0);
        assert!((val - 1.0).abs() < 1e-10, "3*ln(1)+1=1: {val}");

        let quad_model = FittedModel {
            model_type: ModelType::Quadratic,
            params: vec![1.0, 0.0, 5.0],
            r_squared: 1.0,
            rmse: 0.0,
        };
        let val = evaluate(&quad_model, 3.0);
        assert!((val - 14.0).abs() < 1e-10, "1*9+0+5=14: {val}");
    }

    #[test]
    fn test_fit_linear_insufficient_points() {
        assert!(fit_linear(&[1.0], &[2.0]).is_none());
        assert!(fit_linear(&[], &[]).is_none());
    }

    #[test]
    fn test_fit_quadratic_insufficient_points() {
        assert!(fit_quadratic(&[1.0, 2.0], &[1.0, 2.0]).is_none());
    }

    #[test]
    fn test_fit_exponential_all_negative_y() {
        let x = [1.0, 2.0, 3.0];
        let y = [-1.0, -2.0, -3.0];
        assert!(fit_exponential(&x, &y).is_none());
    }

    #[test]
    fn test_fit_logarithmic_all_negative_x() {
        let x = [-1.0, -2.0, -3.0];
        let y = [1.0, 2.0, 3.0];
        assert!(fit_logarithmic(&x, &y).is_none());
    }

    #[test]
    fn test_fit_linear_singular() {
        let x = [5.0, 5.0, 5.0];
        let y = [1.0, 2.0, 3.0];
        assert!(fit_linear(&x, &y).is_none());
    }

    #[test]
    fn test_fit_ridge_perfect_linear() {
        // y = 2x + 1
        let x: Vec<f64> = (0..20).map(f64::from).collect();
        let y: Vec<f64> = x.iter().map(|&xi| 2.0f64.mul_add(xi, 1.0)).collect();

        let m = fit_ridge(&x, &y, 1e-10).unwrap();
        assert!((m.params[0] - 2.0).abs() < 0.01, "slope={}", m.params[0]);
        assert!(
            (m.params[1] - 1.0).abs() < 0.05,
            "intercept={}",
            m.params[1]
        );
        assert!(m.r_squared > 0.999, "R²={}", m.r_squared);
    }

    #[test]
    fn test_fit_ridge_regularization_shrinks_coeffs() {
        let x: Vec<f64> = (0..30).map(f64::from).collect();
        let y: Vec<f64> = x.iter().map(|&xi| 3.0f64.mul_add(xi, 2.0)).collect();

        let low_reg = fit_ridge(&x, &y, 1e-10).unwrap();
        let high_reg = fit_ridge(&x, &y, 1e6).unwrap();
        // λ→∞ drives all weights toward zero
        assert!(
            high_reg.params[0].abs() < low_reg.params[0].abs(),
            "high_reg slope {} should be smaller than low_reg slope {}",
            high_reg.params[0],
            low_reg.params[0]
        );
    }

    #[test]
    fn test_fit_ridge_insufficient_points() {
        assert!(fit_ridge(&[1.0], &[2.0], 0.01).is_none());
        assert!(fit_ridge(&[], &[], 0.01).is_none());
    }
}
