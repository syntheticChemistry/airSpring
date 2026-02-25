// SPDX-License-Identifier: AGPL-3.0-or-later
//! GPU-accelerated isotherm fitting — bridges `eco::isotherm` ↔ `barracuda::optimize`.
//!
//! Provides three API levels:
//!
//! | API | Backend | Best for |
//! |-----|---------|----------|
//! | [`fit_batch_cpu`] | `eco::isotherm` (linearized LS + 1D grid) | Single-point validation |
//! | [`fit_langmuir_nm`] / [`fit_freundlich_nm`] | `barracuda::optimize::nelder_mead` | Matching scipy |
//! | [`fit_langmuir_global`] / [`fit_freundlich_global`] | `barracuda::optimize::multi_start_nelder_mead` | Robust global search |
//!
//! GPU path via `NelderMeadGpu` available for large batch fitting (5+ parameters)
//! when `ToadStool` integrates batch dispatch. For 2-parameter isotherms, CPU NM
//! is already faster than GPU round-trip overhead.
//!
//! # Strategy
//!
//! `eco::isotherm` uses linearized least squares (fast, validated against Python).
//! `barracuda::optimize::nelder_mead` provides true nonlinear optimization that
//! can match `scipy.optimize.curve_fit` more closely for ill-conditioned data.
//! `barracuda::optimize::multi_start_nelder_mead` adds Latin Hypercube Sampling
//! for global exploration — catches cases where linearized initial guess lands
//! in a local minimum.

use barracuda::optimize;

use crate::eco::isotherm::{self, IsothermFit};

/// Fit Langmuir model using `barracuda::optimize::nelder_mead`.
///
/// Minimizes sum of squared residuals: `Σ(qe_i - qmax·KL·Ce_i/(1+KL·Ce_i))²`
/// Starting from linearized initial guess (`eco::isotherm`).
///
/// # Errors
///
/// Returns `None` if optimization fails or data is insufficient.
pub fn fit_langmuir_nm(ce: &[f64], qe: &[f64]) -> Option<IsothermFit> {
    if ce.len() < 2 || ce.len() != qe.len() {
        return None;
    }

    let lin_fit = isotherm::fit_langmuir(ce, qe)?;
    let qmax_init = lin_fit.params[0];
    let kl_init = lin_fit.params[1];

    let ce_owned: Vec<f64> = ce.to_vec();
    let qe_owned: Vec<f64> = qe.to_vec();

    let objective = |x: &[f64]| -> f64 {
        let qmax = x[0];
        let kl = x[1];
        ce_owned
            .iter()
            .zip(&qe_owned)
            .map(|(&c, &q)| {
                let pred = isotherm::langmuir(c, qmax, kl);
                (q - pred).powi(2)
            })
            .sum()
    };

    let x0 = vec![qmax_init, kl_init];
    let bounds = vec![(0.1, 1000.0), (1e-6, 100.0)];

    let (x_best, _f_best, _n_evals) =
        optimize::nelder_mead(objective, &x0, &bounds, 2000, 1e-10).ok()?;

    let qmax = x_best[0];
    let kl = x_best[1];

    let (r2, rmse) = goodness_of_fit(ce, qe, |c| isotherm::langmuir(c, qmax, kl));

    Some(IsothermFit {
        model: "langmuir",
        params: vec![qmax, kl],
        r_squared: r2,
        rmse,
    })
}

/// Fit Freundlich model using `barracuda::optimize::nelder_mead`.
///
/// Minimizes `Σ(qe_i - KF·Ce_i^(1/n))²` starting from linearized guess.
///
/// # Errors
///
/// Returns `None` if optimization fails or data is insufficient.
pub fn fit_freundlich_nm(ce: &[f64], qe: &[f64]) -> Option<IsothermFit> {
    if ce.len() < 2 || ce.len() != qe.len() {
        return None;
    }

    let lin_fit = isotherm::fit_freundlich(ce, qe)?;
    let kf_init = lin_fit.params[0];
    let n_init = lin_fit.params[1];

    let ce_owned: Vec<f64> = ce.to_vec();
    let qe_owned: Vec<f64> = qe.to_vec();

    let objective = |x: &[f64]| -> f64 {
        let kf = x[0];
        let n = x[1];
        ce_owned
            .iter()
            .zip(&qe_owned)
            .map(|(&c, &q)| {
                let pred = isotherm::freundlich(c, kf, 1.0 / n);
                (q - pred).powi(2)
            })
            .sum()
    };

    let x0 = vec![kf_init, n_init];
    let bounds = vec![(1e-4, 1000.0), (0.2, 15.0)];

    let (x_best, _f_best, _n_evals) =
        optimize::nelder_mead(objective, &x0, &bounds, 2000, 1e-10).ok()?;

    let kf = x_best[0];
    let n = x_best[1];

    let (r2, rmse) = goodness_of_fit(ce, qe, |c| isotherm::freundlich(c, kf, 1.0 / n));

    Some(IsothermFit {
        model: "freundlich",
        params: vec![kf, n],
        r_squared: r2,
        rmse,
    })
}

/// Fit Langmuir model using `barracuda::optimize::multi_start_nelder_mead`.
///
/// Runs `n_starts` independent Nelder-Mead optimizations from Latin Hypercube
/// initial guesses, returning the globally best fit. More robust than single-start
/// NM for ill-conditioned or noisy data.
///
/// # Errors
///
/// Returns `None` if optimization fails or data is insufficient.
pub fn fit_langmuir_global(ce: &[f64], qe: &[f64], n_starts: usize) -> Option<IsothermFit> {
    if ce.len() < 2 || ce.len() != qe.len() {
        return None;
    }

    let ce_owned: Vec<f64> = ce.to_vec();
    let qe_owned: Vec<f64> = qe.to_vec();

    let objective = |x: &[f64]| -> f64 {
        let qmax = x[0];
        let kl = x[1];
        ce_owned
            .iter()
            .zip(&qe_owned)
            .map(|(&c, &q)| {
                let pred = isotherm::langmuir(c, qmax, kl);
                (q - pred).powi(2)
            })
            .sum()
    };

    let bounds = vec![(0.1, 1000.0), (1e-6, 100.0)];

    let (best, _cache, _results) =
        optimize::multi_start_nelder_mead(objective, &bounds, n_starts, 2000, 1e-10, 42).ok()?;

    let qmax = best.x_best[0];
    let kl = best.x_best[1];

    let (r2, rmse) = goodness_of_fit(ce, qe, |c| isotherm::langmuir(c, qmax, kl));

    Some(IsothermFit {
        model: "langmuir",
        params: vec![qmax, kl],
        r_squared: r2,
        rmse,
    })
}

/// Fit Freundlich model using `barracuda::optimize::multi_start_nelder_mead`.
///
/// Runs `n_starts` independent Nelder-Mead optimizations from Latin Hypercube
/// initial guesses for global search.
///
/// # Errors
///
/// Returns `None` if optimization fails or data is insufficient.
pub fn fit_freundlich_global(ce: &[f64], qe: &[f64], n_starts: usize) -> Option<IsothermFit> {
    if ce.len() < 2 || ce.len() != qe.len() {
        return None;
    }

    let ce_owned: Vec<f64> = ce.to_vec();
    let qe_owned: Vec<f64> = qe.to_vec();

    let objective = |x: &[f64]| -> f64 {
        let kf = x[0];
        let n = x[1];
        ce_owned
            .iter()
            .zip(&qe_owned)
            .map(|(&c, &q)| {
                let pred = isotherm::freundlich(c, kf, 1.0 / n);
                (q - pred).powi(2)
            })
            .sum()
    };

    let bounds = vec![(1e-4, 1000.0), (0.2, 15.0)];

    let (best, _cache, _results) =
        optimize::multi_start_nelder_mead(objective, &bounds, n_starts, 2000, 1e-10, 42).ok()?;

    let kf = best.x_best[0];
    let n = best.x_best[1];

    let (r2, rmse) = goodness_of_fit(ce, qe, |c| isotherm::freundlich(c, kf, 1.0 / n));

    Some(IsothermFit {
        model: "freundlich",
        params: vec![kf, n],
        r_squared: r2,
        rmse,
    })
}

/// Batch-fit isotherms on CPU using linearized LS (`eco::isotherm`).
///
/// Returns `(langmuir_fit, freundlich_fit)` for each dataset.
pub fn fit_batch_cpu(
    datasets: &[(&[f64], &[f64])],
) -> Vec<(Option<IsothermFit>, Option<IsothermFit>)> {
    datasets
        .iter()
        .map(|&(ce, qe)| {
            (
                isotherm::fit_langmuir(ce, qe),
                isotherm::fit_freundlich(ce, qe),
            )
        })
        .collect()
}

/// Batch-fit isotherms using Nelder-Mead (`barracuda::optimize`).
///
/// Returns `(langmuir_fit, freundlich_fit)` for each dataset.
pub fn fit_batch_nm(
    datasets: &[(&[f64], &[f64])],
) -> Vec<(Option<IsothermFit>, Option<IsothermFit>)> {
    datasets
        .iter()
        .map(|&(ce, qe)| (fit_langmuir_nm(ce, qe), fit_freundlich_nm(ce, qe)))
        .collect()
}

/// Batch-fit isotherms using multi-start Nelder-Mead (global search).
///
/// Returns `(langmuir_fit, freundlich_fit)` for each dataset.
pub fn fit_batch_global(
    datasets: &[(&[f64], &[f64])],
    n_starts: usize,
) -> Vec<(Option<IsothermFit>, Option<IsothermFit>)> {
    datasets
        .iter()
        .map(|&(ce, qe)| {
            (
                fit_langmuir_global(ce, qe, n_starts),
                fit_freundlich_global(ce, qe, n_starts),
            )
        })
        .collect()
}

fn goodness_of_fit<F: Fn(f64) -> f64>(ce: &[f64], qe: &[f64], predict: F) -> (f64, f64) {
    let n = crate::len_f64(qe);
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

    const CE_WOOD: [f64; 9] = [1.0, 2.5, 5.0, 10.0, 20.0, 40.0, 60.0, 80.0, 100.0];
    const QE_WOOD: [f64; 9] = [0.85, 1.92, 3.45, 5.8, 8.9, 12.1, 13.8, 14.5, 14.9];

    #[test]
    fn test_langmuir_nm_wood() {
        let fit = fit_langmuir_nm(&CE_WOOD, &QE_WOOD).unwrap();
        assert!(
            fit.params[0] >= 12.0 && fit.params[0] <= 22.0,
            "qmax={}",
            fit.params[0]
        );
        assert!(fit.params[1] > 0.0, "KL={}", fit.params[1]);
        assert!(fit.r_squared > 0.95, "R²={}", fit.r_squared);
    }

    #[test]
    fn test_freundlich_nm_wood() {
        let fit = fit_freundlich_nm(&CE_WOOD, &QE_WOOD).unwrap();
        assert!(fit.params[0] > 0.0, "KF={}", fit.params[0]);
        assert!(fit.params[1] >= 1.0, "n={}", fit.params[1]);
        assert!(fit.r_squared > 0.90, "R²={}", fit.r_squared);
    }

    #[test]
    fn test_nm_matches_linearized_langmuir() {
        let lin = isotherm::fit_langmuir(&CE_WOOD, &QE_WOOD).unwrap();
        let nm = fit_langmuir_nm(&CE_WOOD, &QE_WOOD).unwrap();
        assert!(
            nm.r_squared >= lin.r_squared - 0.01,
            "NM R²={} should be ≥ linear R²={}",
            nm.r_squared,
            lin.r_squared
        );
    }

    #[test]
    fn test_batch_cpu() {
        let results = fit_batch_cpu(&[(&CE_WOOD, &QE_WOOD)]);
        assert_eq!(results.len(), 1);
        assert!(results[0].0.is_some());
        assert!(results[0].1.is_some());
    }

    #[test]
    fn test_batch_nm() {
        let results = fit_batch_nm(&[(&CE_WOOD, &QE_WOOD)]);
        assert_eq!(results.len(), 1);
        assert!(results[0].0.is_some());
        assert!(results[0].1.is_some());
    }

    #[test]
    fn test_langmuir_global_wood() {
        let fit = fit_langmuir_global(&CE_WOOD, &QE_WOOD, 8).unwrap();
        assert!(
            fit.params[0] >= 12.0 && fit.params[0] <= 22.0,
            "qmax={}",
            fit.params[0]
        );
        assert!(fit.params[1] > 0.0, "KL={}", fit.params[1]);
        assert!(fit.r_squared > 0.95, "R²={}", fit.r_squared);
    }

    #[test]
    fn test_freundlich_global_wood() {
        let fit = fit_freundlich_global(&CE_WOOD, &QE_WOOD, 8).unwrap();
        assert!(fit.params[0] > 0.0, "KF={}", fit.params[0]);
        assert!(fit.params[1] >= 1.0, "n={}", fit.params[1]);
        assert!(fit.r_squared > 0.90, "R²={}", fit.r_squared);
    }

    #[test]
    fn test_global_at_least_as_good_as_nm() {
        let nm = fit_langmuir_nm(&CE_WOOD, &QE_WOOD).unwrap();
        let global = fit_langmuir_global(&CE_WOOD, &QE_WOOD, 8).unwrap();
        assert!(
            global.r_squared >= nm.r_squared - 0.02,
            "Global R²={} should be ≥ NM R²={} (within tolerance)",
            global.r_squared,
            nm.r_squared
        );
    }

    #[test]
    fn test_batch_global() {
        let results = fit_batch_global(&[(&CE_WOOD, &QE_WOOD)], 4);
        assert_eq!(results.len(), 1);
        assert!(results[0].0.is_some());
        assert!(results[0].1.is_some());
    }

    #[test]
    fn test_synthetic_perfect_langmuir() {
        let qmax = 18.0;
        let kl = 0.05;
        let ce: Vec<f64> = vec![1.0, 5.0, 10.0, 20.0, 40.0, 80.0, 100.0];
        let qe: Vec<f64> = ce
            .iter()
            .map(|&c| isotherm::langmuir(c, qmax, kl))
            .collect();

        let fit = fit_langmuir_nm(&ce, &qe).unwrap();
        assert!((fit.params[0] - qmax).abs() < 1.0, "qmax={}", fit.params[0]);
        assert!((fit.params[1] - kl).abs() < 0.01, "KL={}", fit.params[1]);
        assert!(fit.r_squared > 0.999, "R²={}", fit.r_squared);
    }
}
