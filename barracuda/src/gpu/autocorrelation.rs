// SPDX-License-Identifier: AGPL-3.0-or-later
//! GPU-accelerated autocorrelation for ET₀ and hydrology time series.
//!
//! Wraps [`barracuda::ops::autocorrelation_f64_wgsl::AutocorrelationF64`] for
//! temporal analysis of agricultural time series: ET₀ persistence, seasonal
//! patterns, soil moisture memory, and `IoT` sensor diagnostics.
//!
//! # Cross-Spring Provenance
//!
//! The `autocorrelation_f64.wgsl` shader originated from hotSpring's molecular
//! dynamics VACF (velocity autocorrelation function), was generalised by
//! neuralSpring for spectral analysis, and is now used by airSpring for
//! hydrology temporal persistence. One workgroup per lag, tree reduction
//! within workgroup — all lags computed in a single GPU dispatch.
//!
//! # CPU Fallback
//!
//! [`autocorrelation_cpu`] provides an equivalent CPU implementation for
//! headless environments or when GPU is unavailable.

use std::sync::Arc;

use barracuda::device::WgpuDevice;
use barracuda::ops::autocorrelation_f64_wgsl::AutocorrelationF64;

/// GPU-backed autocorrelation engine for hydrology time series.
pub struct HydroAutocorrelation {
    engine: AutocorrelationF64,
}

impl HydroAutocorrelation {
    /// Create a new device-backed autocorrelation engine.
    ///
    /// # Errors
    ///
    /// Returns an error if the GPU device cannot be initialised.
    pub fn new(device: Arc<WgpuDevice>) -> crate::error::Result<Self> {
        let engine = AutocorrelationF64::new(device)?;
        Ok(Self { engine })
    }

    /// Compute autocorrelation C(lag) for lags 0..`max_lag`.
    ///
    /// Returns a vector of length `max_lag` where `out[k]` = C(k).
    /// Single GPU dispatch for all lags simultaneously.
    /// Falls back to CPU if GPU returns all zeros (NVK/consumer GPU
    /// `Df64Only` shared-memory reduction bug — see `PrecisionRoutingAdvice`).
    ///
    /// # Errors
    ///
    /// Returns an error if both GPU and CPU paths fail.
    pub fn autocorrelation(&self, data: &[f64], max_lag: usize) -> crate::error::Result<Vec<f64>> {
        let gpu_result = self
            .engine
            .autocorrelation(data, max_lag)
            .map_err(crate::error::AirSpringError::from)?;

        // Zero-output detection: consumer GPUs with Df64Only precision
        // return zeros from f64 workgroup reductions. Fall back to CPU.
        if !gpu_result.is_empty()
            && gpu_result.iter().all(|&v| v == 0.0)
            && !data.is_empty()
            && data.iter().any(|&v| v != 0.0)
        {
            return Ok(autocorrelation_cpu(data, max_lag));
        }

        Ok(gpu_result)
    }

    /// Compute normalised autocorrelation (ACF) where C(0) = 1.0.
    ///
    /// # Errors
    ///
    /// Returns an error if the GPU dispatch fails.
    pub fn normalised_acf(&self, data: &[f64], max_lag: usize) -> crate::error::Result<Vec<f64>> {
        let raw = self.autocorrelation(data, max_lag)?;
        if raw.is_empty() || raw[0] == 0.0 {
            return Ok(raw);
        }
        let c0 = raw[0];
        Ok(raw.into_iter().map(|c| c / c0).collect())
    }
}

/// CPU autocorrelation for headless environments.
///
/// `C(lag) = (1 / (N - lag)) × Σ x[t] × x[t + lag]`
#[must_use]
pub fn autocorrelation_cpu(data: &[f64], max_lag: usize) -> Vec<f64> {
    let n = data.len();
    if n == 0 || max_lag == 0 {
        return Vec::new();
    }
    let max_lag = max_lag.min(n);
    (0..max_lag)
        .map(|lag| {
            let pairs = n - lag;
            if pairs == 0 {
                return 0.0;
            }
            let sum: f64 = (0..pairs).map(|t| data[t] * data[t + lag]).sum();
            sum / pairs as f64
        })
        .collect()
}

/// Normalised CPU autocorrelation (ACF) where C(0) = 1.0.
#[must_use]
pub fn normalised_acf_cpu(data: &[f64], max_lag: usize) -> Vec<f64> {
    let raw = autocorrelation_cpu(data, max_lag);
    if raw.is_empty() || raw[0] == 0.0 {
        return raw;
    }
    let c0 = raw[0];
    raw.into_iter().map(|c| c / c0).collect()
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn test_autocorrelation_cpu_known() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0];
        let acf = autocorrelation_cpu(&data, 3);
        assert_eq!(acf.len(), 3);
        // C(0) = (1+4+9+16+25)/5 = 11.0
        assert!((acf[0] - 11.0).abs() < 1e-10);
        // C(1) = (2+6+12+20)/4 = 10.0
        assert!((acf[1] - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_normalised_acf_cpu() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0];
        let nacf = normalised_acf_cpu(&data, 3);
        assert!((nacf[0] - 1.0).abs() < 1e-10);
        assert!(nacf[1] > 0.0 && nacf[1] <= 1.0);
    }

    #[test]
    fn test_empty_and_edge_cases() {
        assert!(autocorrelation_cpu(&[], 5).is_empty());
        assert!(autocorrelation_cpu(&[1.0], 0).is_empty());
        let single = autocorrelation_cpu(&[7.0], 1);
        assert_eq!(single.len(), 1);
        assert!((single[0] - 49.0).abs() < 1e-10);
    }

    #[test]
    fn test_constant_signal_max_correlation() {
        let data = vec![3.0; 100];
        let nacf = normalised_acf_cpu(&data, 10);
        for &v in &nacf {
            assert!(
                (v - 1.0).abs() < 1e-10,
                "constant signal should have perfect autocorrelation"
            );
        }
    }

    #[test]
    fn test_gpu_matches_cpu() {
        let device = barracuda::device::test_pool::tokio_block_on(
            barracuda::device::WgpuDevice::new_f64_capable(),
        );
        let Ok(device) = device else {
            eprintln!("SKIP: No GPU device for AutocorrelationF64");
            return;
        };
        let device = Arc::new(device);
        let engine = HydroAutocorrelation::new(device).unwrap();

        let data: Vec<f64> = (0..50).map(|i| f64::from(i).sin()).collect();
        let gpu_acf = engine.autocorrelation(&data, 10).unwrap();
        let cpu_acf = autocorrelation_cpu(&data, 10);
        assert_eq!(gpu_acf.len(), cpu_acf.len());
        for (i, (&g, &c)) in gpu_acf.iter().zip(&cpu_acf).enumerate() {
            assert!((g - c).abs() < 1e-6, "lag {i}: GPU={g} vs CPU={c}");
        }
    }
}
