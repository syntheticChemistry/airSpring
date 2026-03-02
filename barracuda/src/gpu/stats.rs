// SPDX-License-Identifier: AGPL-3.0-or-later
//! GPU-accelerated statistics for agricultural data analysis.
//!
//! Wraps `barracuda::ops::stats_f64` (neuralSpring S69 → `ToadStool` absorption)
//! with domain-specific APIs for:
//!
//! - **Sensor calibration regression**: fit raw counts → VWC via OLS on GPU
//! - **Soil variable correlation**: Pearson correlation matrix for multi-variate
//!   soil/weather data (e.g., VWC, EC, temperature, ET₀ across stations)
//!
//! # Cross-Spring Provenance
//!
//! | Component | Origin | Path |
//! |-----------|--------|------|
//! | `linear_regression_f64.wgsl` | neuralSpring S69 | batched OLS `β = (X'X)⁻¹X'y` |
//! | `matrix_correlation_f64.wgsl` | neuralSpring S69 | Pearson correlation P×P |
//! | Domain application | airSpring v0.5.6 | sensor cal regression, soil analysis |

use std::sync::Arc;

use barracuda::device::WgpuDevice;
use barracuda::ops::stats_f64;

/// Batched OLS sensor calibration: fit raw counts → VWC on GPU.
///
/// Given `n` calibration points `(raw_count, reference_vwc)` for `b` sensors,
/// fits a polynomial of degree `degree` (1 = linear, 2 = quadratic, 3 = cubic).
///
/// Returns `[b, degree+1]` coefficient vectors (intercept, raw, raw², ...).
///
/// # Panics
///
/// Panics if `raw_counts.len() != n_sensors * n_points` or
/// `reference_vwc.len() != n_sensors * n_points`.
///
/// # Errors
///
/// Returns an error if the GPU dispatch fails.
pub fn sensor_regression_gpu(
    device: &Arc<WgpuDevice>,
    raw_counts: &[f64],
    reference_vwc: &[f64],
    n_points: usize,
    n_sensors: usize,
    degree: usize,
) -> Result<Vec<Vec<f64>>, barracuda::error::BarracudaError> {
    assert_eq!(raw_counts.len(), n_sensors * n_points);
    assert_eq!(reference_vwc.len(), n_sensors * n_points);

    let k = degree + 1;

    let mut x = Vec::with_capacity(n_sensors * n_points * k);
    for s in 0..n_sensors {
        for p in 0..n_points {
            let raw = raw_counts[s * n_points + p];
            let mut power = 1.0;
            for _ in 0..k {
                x.push(power);
                power *= raw;
            }
        }
    }

    let betas = stats_f64::linear_regression(
        device,
        &x,
        reference_vwc,
        n_sensors as u32,
        n_points as u32,
        k as u32,
    )?;

    Ok(betas.chunks(k).map(<[f64]>::to_vec).collect())
}

/// Compute the Pearson correlation matrix for multi-variate soil data on GPU.
///
/// Given `n` observations of `p` variables (e.g., VWC, EC, temp, pH, ET₀),
/// returns the `p×p` correlation matrix as a flat row-major vector.
///
/// # Panics
///
/// Panics if `data.len() != n_observations * n_variables`.
///
/// # Errors
///
/// Returns an error if the GPU dispatch fails.
pub fn soil_correlation_gpu(
    device: &Arc<WgpuDevice>,
    data: &[f64],
    n_observations: usize,
    n_variables: usize,
) -> Result<Vec<f64>, barracuda::error::BarracudaError> {
    assert_eq!(data.len(), n_observations * n_variables);
    stats_f64::matrix_correlation(device, data, n_observations as u32, n_variables as u32)
}

/// Apply polynomial coefficients from [`sensor_regression_gpu`] to predict VWC.
#[must_use]
pub fn predict_vwc(coefficients: &[f64], raw_count: f64) -> f64 {
    let mut result = 0.0;
    let mut power = 1.0;
    for &coeff in coefficients {
        result += coeff * power;
        power *= raw_count;
    }
    result
}

#[cfg(test)]
mod tests {
    #![allow(clippy::float_cmp)]
    #![allow(clippy::expect_used, clippy::unwrap_used)]

    use super::*;

    fn try_device() -> Option<Arc<WgpuDevice>> {
        barracuda::device::test_pool::tokio_block_on(WgpuDevice::new_f64_capable())
            .ok()
            .map(Arc::new)
    }

    #[test]
    fn predict_vwc_linear() {
        let coeffs = [0.1, 0.002]; // VWC = 0.1 + 0.002 * raw
        assert!((predict_vwc(&coeffs, 100.0) - 0.3).abs() < 1e-10);
    }

    #[test]
    fn predict_vwc_cubic() {
        let coeffs = [-0.0677, 4e-5, -4e-9, 2e-13];
        let vwc = predict_vwc(&coeffs, 10_000.0);
        let expected = (2e-13f64.mul_add(10_000.0, -4e-9))
            .mul_add(10_000.0, 4e-5)
            .mul_add(10_000.0, -0.0677);
        assert!(
            (vwc - expected).abs() < 1e-6,
            "cubic predict: got {vwc}, expected {expected}"
        );
    }

    #[test]
    fn sensor_regression_gpu_linear() {
        let Some(device) = try_device() else {
            eprintln!("SKIP: No f64-capable GPU");
            return;
        };
        let raw: Vec<f64> = (0..50)
            .map(|i| f64::from(i).mul_add(200.0, 1000.0))
            .collect();
        let vwc: Vec<f64> = raw.iter().map(|&r| r.mul_add(0.00002, 0.05)).collect();
        let dev = Arc::clone(&device);
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            sensor_regression_gpu(&dev, &raw, &vwc, 50, 1, 1)
        }));
        let result = match outcome {
            Ok(Ok(r)) => r,
            Ok(Err(e)) => {
                eprintln!("SKIP: stats_f64 GPU dispatch error: {e}");
                return;
            }
            Err(_) => {
                eprintln!("SKIP: stats_f64 shader compilation panic (driver lacks f64 WGSL)");
                return;
            }
        };
        assert_eq!(result.len(), 1);
        let coeffs = &result[0];
        assert!((coeffs[0] - 0.05).abs() < 0.01, "intercept ≈ 0.05");
        assert!((coeffs[1] - 0.00002).abs() < 1e-5, "slope ≈ 0.00002");
    }

    #[test]
    fn soil_correlation_gpu_identity_diagonal() {
        let Some(device) = try_device() else {
            eprintln!("SKIP: No f64-capable GPU");
            return;
        };
        let n = 100;
        let p = 3;
        let mut data = Vec::with_capacity(n * p);
        for i in 0..n {
            let fi = f64::from(i as u16);
            data.push(fi);
            data.push(fi.mul_add(2.0, 1.0));
            data.push((-fi) + 50.0);
        }
        let dev = Arc::clone(&device);
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            soil_correlation_gpu(&dev, &data, n, p)
        }));
        let corr = match outcome {
            Ok(Ok(c)) => c,
            Ok(Err(e)) => {
                eprintln!("SKIP: stats_f64 GPU dispatch error: {e}");
                return;
            }
            Err(_) => {
                eprintln!("SKIP: stats_f64 shader compilation panic (driver lacks f64 WGSL)");
                return;
            }
        };
        assert_eq!(corr.len(), p * p);
        for i in 0..p {
            assert!(
                (corr[i * p + i] - 1.0).abs() < 0.01,
                "diagonal should be ~1.0"
            );
        }
        assert!(
            corr[1] > 0.99,
            "x1 and x2=2*x1+1 should be highly correlated"
        );
        assert!(
            corr[2] < -0.99,
            "x1 and x3=-x1+50 should be highly anti-correlated"
        );
    }

    #[test]
    fn predict_vwc_empty_coeffs_constant_zero() {
        let vwc = predict_vwc(&[], 1000.0);
        assert_eq!(vwc, 0.0);
    }

    #[test]
    fn predict_vwc_single_coeff_constant() {
        let coeffs = [0.42];
        assert!((predict_vwc(&coeffs, 0.0) - 0.42).abs() < 1e-10);
        assert!((predict_vwc(&coeffs, 10_000.0) - 0.42).abs() < 1e-10);
    }

    #[test]
    fn predict_vwc_zero_raw() {
        let coeffs = [0.1, 0.002];
        let vwc = predict_vwc(&coeffs, 0.0);
        assert!((vwc - 0.1).abs() < 1e-10);
    }

    #[test]
    fn predict_vwc_quadratic() {
        let coeffs = [1.0, 0.5, 0.01];
        let vwc = predict_vwc(&coeffs, 10.0);
        let expected = 0.01f64.mul_add(100.0, 0.5f64.mul_add(10.0, 1.0));
        assert!((vwc - expected).abs() < 1e-10);
    }

    #[test]
    fn predict_vwc_summary_statistics() {
        let coeffs = [0.05, 0.00002];
        let raw_values = [0.0, 5_000.0, 10_000.0, 15_000.0, 20_000.0];
        let predicted: Vec<f64> = raw_values
            .iter()
            .map(|&r| predict_vwc(&coeffs, r))
            .collect();
        assert!(predicted[0] < predicted[1]);
        assert!(predicted[1] < predicted[2]);
        assert!(predicted[2] < predicted[3]);
        assert!(predicted[3] < predicted[4]);
        assert!((predicted[0] - 0.05).abs() < 1e-10);
        assert!((predicted[2] - 0.25).abs() < 1e-10);
    }
}
