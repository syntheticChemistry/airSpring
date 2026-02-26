// SPDX-License-Identifier: AGPL-3.0-or-later
//! Langmuir and Freundlich isotherm models.
//!
//! Pure Rust, zero dependencies. Absorption target: `barracuda::optimize`
//! (curve fitting via Nelder-Mead or batched GPU optimization).
//!
//! # References
//!
//! - Langmuir (1918) J Am Chem Soc 40:1361-1403
//! - Freundlich (1906) Z Phys Chem 57:385-470
//! - Kumari, Dong & Safferman (2025) Applied Water Science 15(7):162

use crate::len_f64;

/// Langmuir isotherm: qe = qmax × KL × Ce / (1 + KL × Ce)
///
/// # Examples
///
/// ```
/// use airspring_forge::isotherm::langmuir;
///
/// let q = langmuir(10.0, 18.0, 0.05);
/// assert!((q - 6.0).abs() < 0.1);
/// ```
#[must_use]
pub fn langmuir(ce: f64, qmax: f64, kl: f64) -> f64 {
    let denom = kl.mul_add(ce, 1.0);
    if denom.abs() < f64::EPSILON {
        return qmax;
    }
    qmax * kl * ce / denom
}

/// Freundlich isotherm: qe = KF × Ce^(1/n)
///
/// # Examples
///
/// ```
/// use airspring_forge::isotherm::freundlich;
///
/// let q = freundlich(10.0, 2.0, 0.5);
/// assert!((q - 2.0 * 10.0_f64.sqrt()).abs() < 1e-10);
/// ```
#[must_use]
pub fn freundlich(ce: f64, kf: f64, n_inv: f64) -> f64 {
    kf * ce.max(1e-10).powf(n_inv)
}

/// Langmuir separation factor: RL = 1 / (1 + KL × C0)
#[must_use]
pub fn separation_factor(kl: f64, c0: f64) -> f64 {
    1.0 / kl.mul_add(c0, 1.0)
}

/// Result of isotherm fitting.
#[derive(Debug, Clone)]
pub struct IsothermFit {
    /// Model identifier.
    pub model: &'static str,
    /// Fitted parameters: [qmax, KL] for Langmuir, [KF, n] for Freundlich.
    pub params: Vec<f64>,
    /// R² in original space.
    pub r_squared: f64,
    /// RMSE (mg/g).
    pub rmse: f64,
}

/// Fit Langmuir via linearized LS: Ce/qe = 1/(qmax×KL) + Ce/qmax.
///
/// # Errors
///
/// Returns `None` if data is insufficient or singular.
#[must_use]
pub fn fit_langmuir(ce: &[f64], qe: &[f64]) -> Option<IsothermFit> {
    if ce.len() < 2 || ce.len() != qe.len() {
        return None;
    }
    let valid: Vec<(f64, f64)> = ce
        .iter()
        .zip(qe)
        .filter(|(_, &q)| q > 1e-10)
        .map(|(&c, &q)| (c, q))
        .collect();
    if valid.len() < 2 {
        return None;
    }
    let x: Vec<f64> = valid.iter().map(|&(c, _)| c).collect();
    let y: Vec<f64> = valid.iter().map(|&(c, q)| c / q).collect();
    let (slope, intercept) = linear_fit(&x, &y)?;
    if slope.abs() < 1e-30 {
        return None;
    }
    let qmax = 1.0 / slope;
    let kl = if (qmax * intercept).abs() < 1e-30 {
        return None;
    } else {
        1.0 / (qmax * intercept)
    };

    let ce_orig: Vec<f64> = valid.iter().map(|v| v.0).collect();
    let qe_orig: Vec<f64> = valid.iter().map(|v| v.1).collect();
    let (r2, rmse) = goodness_of_fit(&ce_orig, &qe_orig, |c| langmuir(c, qmax, kl));

    Some(IsothermFit {
        model: "langmuir",
        params: vec![qmax, kl],
        r_squared: r2,
        rmse,
    })
}

/// Fit Freundlich via log-linearized LS + 1D grid refinement.
///
/// # Errors
///
/// Returns `None` if data is insufficient or singular.
#[must_use]
pub fn fit_freundlich(ce: &[f64], qe: &[f64]) -> Option<IsothermFit> {
    if ce.len() < 2 || ce.len() != qe.len() {
        return None;
    }
    let valid: Vec<(f64, f64)> = ce
        .iter()
        .zip(qe)
        .filter(|(&c, &q)| c > 1e-10 && q > 1e-10)
        .map(|(&c, &q)| (c, q))
        .collect();
    if valid.len() < 2 {
        return None;
    }
    let x: Vec<f64> = valid.iter().map(|&(c, _)| c.ln()).collect();
    let y: Vec<f64> = valid.iter().map(|&(_, q)| q.ln()).collect();
    let (slope, _) = linear_fit(&x, &y)?;
    if slope.abs() < 1e-30 {
        return None;
    }
    let n_init = 1.0 / slope;
    let ce_v: Vec<f64> = valid.iter().map(|v| v.0).collect();
    let qe_v: Vec<f64> = valid.iter().map(|v| v.1).collect();

    let n_lo = (n_init * 0.5).max(0.2);
    let n_hi = (n_init * 2.0).min(15.0);
    let mut best = (f64::INFINITY, 1.0, n_init);
    for i in 0..=50 {
        let t = f64::from(i) / 50.0;
        let n = n_lo + t * (n_hi - n_lo);
        let n_inv = 1.0 / n;
        let (mut sn, mut sd) = (0.0, 0.0);
        for (&c, &q) in ce_v.iter().zip(&qe_v) {
            let cp = c.powf(n_inv);
            sn += q * cp;
            sd += cp * cp;
        }
        if sd < 1e-30 {
            continue;
        }
        let kf = sn / sd;
        let ss: f64 = ce_v
            .iter()
            .zip(&qe_v)
            .map(|(&c, &q)| (q - freundlich(c, kf, n_inv)).powi(2))
            .sum();
        if ss < best.0 {
            best = (ss, kf, n);
        }
    }
    let (r2, rmse) = goodness_of_fit(&ce_v, &qe_v, |c| freundlich(c, best.1, 1.0 / best.2));
    Some(IsothermFit {
        model: "freundlich",
        params: vec![best.1, best.2],
        r_squared: r2,
        rmse,
    })
}

fn linear_fit(x: &[f64], y: &[f64]) -> Option<(f64, f64)> {
    let n = len_f64(x);
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
    let slope = n.mul_add(sxy, -(sx * sy)) / det;
    let intercept = sxx.mul_add(sy, -(sx * sxy)) / det;
    Some((slope, intercept))
}

fn goodness_of_fit<F: Fn(f64) -> f64>(ce: &[f64], qe: &[f64], predict: F) -> (f64, f64) {
    let n = len_f64(qe);
    let mean: f64 = qe.iter().sum::<f64>() / n;
    let ss_res: f64 = ce
        .iter()
        .zip(qe)
        .map(|(&c, &q)| (q - predict(c)).powi(2))
        .sum();
    let ss_tot: f64 = qe.iter().map(|&q| (q - mean).powi(2)).sum();
    let r2 = if ss_tot > 0.0 {
        1.0 - ss_res / ss_tot
    } else {
        1.0
    };
    (r2, (ss_res / n).sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_langmuir() {
        let q = langmuir(10.0, 18.0, 0.05);
        assert!((q - 6.0).abs() < 0.01);
    }

    #[test]
    fn test_freundlich() {
        let q = freundlich(10.0, 2.0, 0.5);
        assert!(2.0f64.mul_add(-10.0_f64.sqrt(), q).abs() < 1e-10);
    }

    #[test]
    fn test_separation_factor() {
        let rl = separation_factor(0.1, 100.0);
        assert!(rl > 0.0 && rl < 1.0);
    }

    #[test]
    fn test_fit_langmuir_synthetic() {
        let ce: Vec<f64> = vec![1.0, 5.0, 10.0, 20.0, 40.0, 80.0, 100.0];
        let qe: Vec<f64> = ce.iter().map(|&c| langmuir(c, 18.0, 0.05)).collect();
        let fit = fit_langmuir(&ce, &qe).unwrap();
        assert!((fit.params[0] - 18.0).abs() < 0.5);
        assert!(fit.r_squared > 0.99);
    }

    #[test]
    fn test_fit_freundlich_synthetic() {
        let ce: Vec<f64> = vec![1.0, 2.0, 5.0, 10.0, 20.0, 50.0];
        let qe: Vec<f64> = ce.iter().map(|&c| freundlich(c, 2.0, 0.5)).collect();
        let fit = fit_freundlich(&ce, &qe).unwrap();
        assert!((fit.params[0] - 2.0).abs() < 0.1);
        assert!(fit.r_squared > 0.99);
    }
}
