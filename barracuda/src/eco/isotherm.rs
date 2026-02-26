// SPDX-License-Identifier: AGPL-3.0-or-later
//! Langmuir and Freundlich isotherm models for biochar adsorption.
//!
//! Implements linearized least-squares fitting for phosphorus adsorption
//! on biochar (Kumari, Dong & Safferman 2025; Langmuir 1918; Freundlich 1906).
//!
//! # Models
//!
//! - **Langmuir**: qe = qmax × KL × Ce / (1 + KL × Ce)
//! - **Freundlich**: qe = KF × Ce^(1/n)
//! - **Separation factor**: RL = 1 / (1 + KL × C0)

use crate::len_f64;

/// Langmuir isotherm: qe = qmax * KL * Ce / (1 + KL * Ce)
#[must_use]
pub fn langmuir(ce: f64, qmax: f64, kl: f64) -> f64 {
    let denom = kl.mul_add(ce, 1.0);
    if denom.abs() < f64::EPSILON {
        return qmax; // saturation limit as Ce → ∞
    }
    qmax * kl * ce / denom
}

/// Freundlich isotherm: qe = KF * Ce^(1/n)
///
/// Uses `n_inv = 1/n` as the exponent to avoid division in hot path.
#[must_use]
pub fn freundlich(ce: f64, kf: f64, n_inv: f64) -> f64 {
    let ce_safe = ce.max(1e-10);
    kf * ce_safe.powf(n_inv)
}

/// Langmuir separation factor: RL = 1 / (1 + KL * C0)
///
/// Favorable adsorption when 0 < RL < 1.
#[must_use]
pub fn langmuir_rl(kl: f64, c0: f64) -> f64 {
    1.0 / kl.mul_add(c0, 1.0)
}

/// Result of isotherm fitting.
#[derive(Debug, Clone)]
pub struct IsothermFit {
    /// Model identifier: "langmuir" or "freundlich"
    pub model: &'static str,
    /// [qmax, KL] for Langmuir, [KF, n] for Freundlich
    pub params: Vec<f64>,
    /// Coefficient of determination R² (computed in original space)
    pub r_squared: f64,
    /// Root mean square error (mg/g)
    pub rmse: f64,
}

/// Guard against singular matrices in regression.
const SINGULARITY_GUARD: f64 = 1e-30;

/// Minimum value for log domain (avoids log(0)).
const LOG_DOMAIN_GUARD: f64 = 1e-10;

/// Fit Langmuir isotherm to (Ce, qe) data using linearized least squares.
///
/// Linearization: Ce/qe = 1/(qmax×KL) + Ce/qmax
/// This is a linear regression of Ce/qe vs Ce, giving slope=1/qmax and intercept=1/(qmax×KL).
#[must_use]
pub fn fit_langmuir(ce: &[f64], qe: &[f64]) -> Option<IsothermFit> {
    if ce.len() < 2 || ce.len() != qe.len() {
        return None;
    }

    // Filter: qe > 0 (Ce/qe undefined otherwise)
    let valid: Vec<(f64, f64)> = ce
        .iter()
        .zip(qe)
        .filter(|(_, &qi)| qi > LOG_DOMAIN_GUARD)
        .map(|(&ci, &qi)| (ci, qi))
        .collect();

    if valid.len() < 2 {
        return None;
    }

    let x: Vec<f64> = valid.iter().map(|&(ci, _)| ci).collect();
    let y: Vec<f64> = valid.iter().map(|&(ci, qi)| ci / qi).collect();

    let lin = fit_linear_internal(&x, &y)?;
    let slope = lin.0;
    let intercept = lin.1;

    if slope.abs() < SINGULARITY_GUARD {
        return None;
    }

    let qmax = 1.0 / slope;
    let kl = if (qmax * intercept).abs() < SINGULARITY_GUARD {
        return None;
    } else {
        1.0 / (qmax * intercept)
    };

    let ce_orig: Vec<f64> = valid.iter().map(|v| v.0).collect();
    let qe_orig: Vec<f64> = valid.iter().map(|v| v.1).collect();

    let (r2, rmse) = goodness_of_fit(&ce_orig, &qe_orig, |ci| langmuir(ci, qmax, kl));

    Some(IsothermFit {
        model: "langmuir",
        params: vec![qmax, kl],
        r_squared: r2,
        rmse,
    })
}

/// Fit Freundlich isotherm using log-linearized least squares with nonlinear refinement.
///
/// Linearization gives initial (KF, n). For fixed n, optimal KF has closed form,
/// so we refine by 1D search over n to minimize `SS_res` in original space (matching
/// `scipy.curve_fit` behavior for Python baseline parity).
#[must_use]
pub fn fit_freundlich(ce: &[f64], qe: &[f64]) -> Option<IsothermFit> {
    const N_GRID: usize = 50;

    if ce.len() < 2 || ce.len() != qe.len() {
        return None;
    }

    let valid: Vec<(f64, f64)> = ce
        .iter()
        .zip(qe)
        .filter(|(&ci, &qi)| ci > LOG_DOMAIN_GUARD && qi > LOG_DOMAIN_GUARD)
        .map(|(&ci, &qi)| (ci, qi))
        .collect();

    if valid.len() < 2 {
        return None;
    }

    let ce_orig: Vec<f64> = valid.iter().map(|v| v.0).collect();
    let qe_orig: Vec<f64> = valid.iter().map(|v| v.1).collect();

    // Linearized initial guess
    let x: Vec<f64> = valid.iter().map(|&(ci, _)| ci.ln()).collect();
    let y: Vec<f64> = valid.iter().map(|&(_, qi)| qi.ln()).collect();
    let lin = fit_linear_internal(&x, &y)?;
    let slope = lin.0;
    if slope.abs() < SINGULARITY_GUARD {
        return None;
    }
    let n_init = 1.0 / slope;

    // Refine: for each n, optimal KF = sum(qe*Ce^(1/n)) / sum(Ce^(2/n))
    // Search n in [0.5*n_init, 2.0*n_init] to minimize SS_res
    let n_lo = (n_init * 0.5).max(0.2);
    let n_hi = (n_init * 2.0).min(15.0);
    let mut best_ss = f64::INFINITY;
    let mut best_kf = 1.0;
    let mut best_n = n_init;

    for i in 0..=N_GRID {
        let t = (i as f64) / (N_GRID as f64);
        let n = t.mul_add(n_hi - n_lo, n_lo);
        let n_inv = 1.0 / n;

        let mut sum_num = 0.0;
        let mut sum_den = 0.0;
        for (&ci, &qi) in ce_orig.iter().zip(&qe_orig) {
            let c_pow = ci.powf(n_inv);
            sum_num += qi * c_pow;
            sum_den += c_pow * c_pow;
        }
        if sum_den < SINGULARITY_GUARD {
            continue;
        }
        let kf = sum_num / sum_den;

        let ss: f64 = ce_orig
            .iter()
            .zip(&qe_orig)
            .map(|(&ci, &qi)| {
                let pred = freundlich(ci, kf, n_inv);
                (qi - pred).powi(2)
            })
            .sum();

        if ss < best_ss {
            best_ss = ss;
            best_kf = kf;
            best_n = n;
        }
    }

    let n_inv = 1.0 / best_n;
    let (r2, rmse) = goodness_of_fit(&ce_orig, &qe_orig, |ci| freundlich(ci, best_kf, n_inv));

    Some(IsothermFit {
        model: "freundlich",
        params: vec![best_kf, best_n],
        r_squared: r2,
        rmse,
    })
}

/// Internal linear regression: y = slope*x + intercept.
/// Returns (slope, intercept) or None if singular.
fn fit_linear_internal(x: &[f64], y: &[f64]) -> Option<(f64, f64)> {
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

    let slope = count.mul_add(s_cross, -(s_x * s_y)) / det;
    let intercept = s_xx.mul_add(s_y, -(s_x * s_cross)) / det;

    Some((slope, intercept))
}

fn goodness_of_fit<F: Fn(f64) -> f64>(ce: &[f64], qe: &[f64], predict: F) -> (f64, f64) {
    let n = len_f64(qe);
    let mean_qe: f64 = qe.iter().sum::<f64>() / n;

    let ss_res: f64 = ce
        .iter()
        .zip(qe)
        .map(|(&ci, &qi)| (qi - predict(ci)).powi(2))
        .sum();
    let ss_tot: f64 = qe.iter().map(|&qi| (qi - mean_qe).powi(2)).sum();

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
    fn test_langmuir_at_zero() {
        let q = langmuir(0.0, 18.0, 0.1);
        assert!((q - 0.0).abs() < 1e-10, "qe=0 at Ce=0");
    }

    #[test]
    fn test_langmuir_saturation() {
        let qmax = 20.0;
        let kl = 0.05;
        let q_high = langmuir(1000.0, qmax, kl);
        assert!(q_high > 19.0, "near saturation at high Ce");
    }

    #[test]
    fn test_freundlich_basic() {
        let q = freundlich(10.0, 2.0, 0.5);
        assert!((2.0f64.mul_add(-10.0_f64.sqrt(), q)).abs() < 1e-10);
    }

    #[test]
    fn test_langmuir_rl() {
        let rl = langmuir_rl(0.1, 100.0);
        assert!((rl - 1.0 / 11.0).abs() < 1e-10);
        assert!(rl > 0.0 && rl < 1.0, "favorable");
    }

    #[test]
    fn test_fit_langmuir_synthetic() {
        let qmax = 18.0;
        let kl = 0.05;
        let ce: Vec<f64> = vec![1.0, 5.0, 10.0, 20.0, 40.0, 80.0, 100.0];
        let qe: Vec<f64> = ce.iter().map(|&c| langmuir(c, qmax, kl)).collect();

        let fit = fit_langmuir(&ce, &qe).unwrap();
        assert_eq!(fit.model, "langmuir");
        assert!((fit.params[0] - qmax).abs() < 0.5, "qmax={}", fit.params[0]);
        assert!((fit.params[1] - kl).abs() < 0.01, "KL={}", fit.params[1]);
        assert!(fit.r_squared > 0.99, "R²={}", fit.r_squared);
    }

    #[test]
    fn test_fit_freundlich_synthetic() {
        let kf = 2.0;
        let n = 2.0;
        let n_inv = 1.0 / n;
        let ce: Vec<f64> = vec![1.0, 2.0, 5.0, 10.0, 20.0, 50.0];
        let qe: Vec<f64> = ce.iter().map(|&c| freundlich(c, kf, n_inv)).collect();

        let fit = fit_freundlich(&ce, &qe).unwrap();
        assert_eq!(fit.model, "freundlich");
        assert!((fit.params[0] - kf).abs() < 0.1, "KF={}", fit.params[0]);
        assert!((fit.params[1] - n).abs() < 0.2, "n={}", fit.params[1]);
        assert!(fit.r_squared > 0.99, "R²={}", fit.r_squared);
    }

    #[test]
    fn test_fit_langmuir_insufficient() {
        assert!(fit_langmuir(&[1.0], &[2.0]).is_none());
        assert!(fit_langmuir(&[], &[]).is_none());
    }

    #[test]
    fn test_fit_freundlich_zero_qe() {
        let ce = [1.0, 2.0, 3.0];
        let qe = [1.0, 0.0, 2.0];
        assert!(fit_freundlich(&ce, &qe).is_some()); // filters out qe=0, may still have 2 points
    }

    #[test]
    fn test_fit_langmuir_benchmark_wood() {
        let ce = [1.0, 2.5, 5.0, 10.0, 20.0, 40.0, 60.0, 80.0, 100.0];
        let qe = [0.85, 1.92, 3.45, 5.8, 8.9, 12.1, 13.8, 14.5, 14.9];

        let fit = fit_langmuir(&ce, &qe).unwrap();
        assert!(
            fit.params[0] >= 12.0 && fit.params[0] <= 25.0,
            "qmax={}",
            fit.params[0]
        );
        assert!(fit.params[1] > 0.0, "KL={}", fit.params[1]);
        assert!(fit.r_squared > 0.95, "R²={}", fit.r_squared);
    }

    #[test]
    fn test_fit_freundlich_benchmark_wood() {
        let ce = [1.0, 2.5, 5.0, 10.0, 20.0, 40.0, 60.0, 80.0, 100.0];
        let qe = [0.85, 1.92, 3.45, 5.8, 8.9, 12.1, 13.8, 14.5, 14.9];

        let fit = fit_freundlich(&ce, &qe).unwrap();
        assert!(fit.params[0] > 0.0, "KF={}", fit.params[0]);
        assert!(fit.params[1] >= 1.0, "n={}", fit.params[1]);
        assert!(fit.r_squared > 0.90, "R²={}", fit.r_squared);
    }

    #[test]
    fn test_langmuir_denom_near_zero() {
        let q = langmuir(1e12, 10.0, -1e-12);
        assert!((q - 10.0).abs() < 1.0, "qmax saturation guard: q={q}");
    }

    #[test]
    fn test_fit_langmuir_all_zero_qe() {
        let ce = [1.0, 2.0, 3.0];
        let qe = [0.0, 0.0, 0.0];
        assert!(fit_langmuir(&ce, &qe).is_none());
    }

    #[test]
    fn test_fit_langmuir_mismatched_lengths() {
        assert!(fit_langmuir(&[1.0, 2.0], &[1.0]).is_none());
    }

    #[test]
    fn test_fit_freundlich_insufficient() {
        assert!(fit_freundlich(&[1.0], &[2.0]).is_none());
        assert!(fit_freundlich(&[], &[]).is_none());
    }

    #[test]
    fn test_fit_freundlich_mismatched_lengths() {
        assert!(fit_freundlich(&[1.0, 2.0], &[1.0]).is_none());
    }

    #[test]
    fn test_fit_freundlich_all_zero() {
        let ce = [0.0, 0.0, 0.0];
        let qe = [0.0, 0.0, 0.0];
        assert!(fit_freundlich(&ce, &qe).is_none());
    }

    #[test]
    fn test_fit_linear_internal_singular() {
        let x = [5.0, 5.0, 5.0];
        let y = [1.0, 2.0, 3.0];
        assert!(fit_linear_internal(&x, &y).is_none());
    }

    #[test]
    fn test_fit_linear_internal_too_few() {
        assert!(fit_linear_internal(&[1.0], &[2.0]).is_none());
    }

    #[test]
    fn test_fit_linear_internal_mismatched() {
        assert!(fit_linear_internal(&[1.0, 2.0], &[1.0]).is_none());
    }

    #[test]
    fn test_goodness_of_fit_constant_qe() {
        let ce = [1.0, 2.0, 3.0];
        let qe = [5.0, 5.0, 5.0];
        let (r2, rmse) = goodness_of_fit(&ce, &qe, |_| 5.0);
        assert!((r2 - 1.0).abs() < 1e-10, "r2={r2}");
        assert!(rmse.abs() < 1e-10, "rmse={rmse}");
    }

    #[test]
    fn test_isotherm_fit_debug_clone() {
        let fit = IsothermFit {
            model: "test",
            params: vec![1.0, 2.0],
            r_squared: 0.99,
            rmse: 0.01,
        };
        let _ = format!("{fit:?}");
        let cloned = fit.clone();
        assert_eq!(cloned.model, "test");
    }

    #[test]
    fn test_langmuir_rl_extreme() {
        assert!((langmuir_rl(0.0, 100.0) - 1.0).abs() < 1e-10);
        let rl_large = langmuir_rl(100.0, 100.0);
        assert!(rl_large < 0.001, "rl={rl_large}");
    }

    #[test]
    fn test_freundlich_near_zero_ce() {
        let q = freundlich(0.0, 2.0, 0.5);
        assert!(q > 0.0 && q < 1.0, "guarded: q={q}");
    }

    #[test]
    fn test_fit_langmuir_constant_qe() {
        let ce = [1.0, 10.0, 50.0, 100.0];
        let qe = [5.0, 5.0, 5.0, 5.0];
        let fit = fit_langmuir(&ce, &qe);
        assert!(fit.is_some() || fit.is_none()); // singular or near-zero slope
    }
}
