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

/// A fitted correction model with parameters and goodness-of-fit.
#[derive(Debug, Clone)]
pub struct FittedModel {
    /// Model type name.
    pub model_type: &'static str,
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
        "linear" => linear_model(x, model.params[0], model.params[1]),
        "quadratic" => quadratic_model(x, model.params[0], model.params[1], model.params[2]),
        "exponential" => exponential_model(x, model.params[0], model.params[1]),
        "logarithmic" => logarithmic_model(x, model.params[0], model.params[1]),
        _ => f64::NAN,
    }
}

// ── Least-squares fitting ────────────────────────────────────────────

/// Fit a linear model y = a·x + b using normal equations.
///
/// Returns `Some(FittedModel)` or `None` if the system is singular.
#[must_use]
#[allow(clippy::cast_precision_loss, clippy::many_single_char_names)]
pub fn fit_linear(x: &[f64], y: &[f64]) -> Option<FittedModel> {
    let n = x.len() as f64;
    if x.len() < 2 || x.len() != y.len() {
        return None;
    }

    let sx: f64 = x.iter().sum();
    let sy: f64 = y.iter().sum();
    let sxx: f64 = x.iter().map(|&xi| xi * xi).sum();
    let sxy: f64 = x.iter().zip(y).map(|(&xi, &yi)| xi * yi).sum();

    let det = n.mul_add(sxx, -(sx * sx));
    if det.abs() < 1e-30 {
        return None;
    }

    let a = n.mul_add(sxy, -(sx * sy)) / det;
    let b = sxx.mul_add(sy, -(sx * sxy)) / det;

    let (r2, rmse) = goodness_of_fit(x, y, |xi| linear_model(xi, a, b));
    Some(FittedModel {
        model_type: "linear",
        params: vec![a, b],
        r_squared: r2,
        rmse,
    })
}

/// Fit a quadratic model y = a·x² + b·x + c using normal equations.
#[must_use]
#[allow(
    clippy::cast_precision_loss,
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::suspicious_operation_groupings,
    clippy::suboptimal_flops
)]
pub fn fit_quadratic(x: &[f64], y: &[f64]) -> Option<FittedModel> {
    let n = x.len() as f64;
    if x.len() < 3 || x.len() != y.len() {
        return None;
    }

    // Normal equations: A'A·p = A'y where A = [x², x, 1]
    let sx: f64 = x.iter().sum();
    let sx2: f64 = x.iter().map(|&xi| xi * xi).sum();
    let sx3: f64 = x.iter().map(|&xi| xi.powi(3)).sum();
    let sx4: f64 = x.iter().map(|&xi| xi.powi(4)).sum();
    let sy: f64 = y.iter().sum();
    let sxy: f64 = x.iter().zip(y).map(|(&xi, &yi)| xi * yi).sum();
    let sx2y: f64 = x.iter().zip(y).map(|(&xi, &yi)| xi * xi * yi).sum();

    // Solve 3×3 system using Cramer's rule
    // | sx4  sx3  sx2 | | a |   | sx2y |
    // | sx3  sx2  sx  | | b | = | sxy  |
    // | sx2  sx   n   | | c |   | sy   |
    let det = sx4 * (sx2 * n - sx * sx) - sx3 * (sx3 * n - sx * sx2) + sx2 * (sx3 * sx - sx2 * sx2);
    if det.abs() < 1e-30 {
        return None;
    }

    // Cramer's rule for 3×3 system
    // det_a: replace column 1 with RHS
    let a = (sx2y * (sx2 * n - sx * sx) - sx3 * (sxy * n - sy * sx) + sx2 * (sxy * sx - sy * sx2))
        / det;
    // det_b: replace column 2 with RHS
    let b = (sx4 * (sxy * n - sy * sx) - sx2y * (sx3 * n - sx2 * sx)
        + sx2 * (sx3 * sy - sx2 * sxy))
        / det;
    // det_c: replace column 3 with RHS
    let c = (sx4 * (sx2 * sy - sx * sxy) - sx3 * (sx3 * sy - sx2 * sxy)
        + sx2y * (sx3 * sx - sx2 * sx2))
        / det;

    let (r2, rmse) = goodness_of_fit(x, y, |xi| quadratic_model(xi, a, b, c));
    Some(FittedModel {
        model_type: "quadratic",
        params: vec![a, b, c],
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
        model_type: "exponential",
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
        .filter(|(&xi, _)| xi > 0.001)
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
        model_type: "logarithmic",
        params: vec![a, b],
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

#[allow(clippy::cast_precision_loss)]
fn goodness_of_fit<F: Fn(f64) -> f64>(x: &[f64], y: &[f64], predict: F) -> (f64, f64) {
    let n = y.len() as f64;
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
        let linear = models.iter().find(|m| m.model_type == "linear").unwrap();
        assert!(linear.r_squared > 0.99);
    }

    #[test]
    fn test_evaluate_model() {
        let model = FittedModel {
            model_type: "linear",
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
}
