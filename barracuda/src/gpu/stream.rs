// SPDX-License-Identifier: AGPL-3.0-or-later
//! GPU-accelerated `IoT` stream processing via `ToadStool` `MovingWindowStats`.
//!
//! Wraps [`barracuda::ops::moving_window_stats::MovingWindowStats`] for sliding
//! window smoothing of agricultural sensor time series. Handles the f64→f32→f64
//! conversion transparently (upstream shader is f32; airSpring works in f64).
//!
//! # Provenance
//!
//! `moving_window.wgsl` was contributed by `wetSpring` for environmental
//! monitoring (S28+), absorbed into `ToadStool` ops, and now wired here for
//! `IoT` sensor stream smoothing — a cross-spring benefit.
//!
//! # Usage
//!
//! ```ignore
//! use airspring_barracuda::gpu::stream::StreamSmoother;
//!
//! let smoother = StreamSmoother::new(device);
//! let result = smoother.smooth(&temperature_data, 24)?; // 24-hour window
//! // result.mean, result.min, result.max, result.variance — all f64
//! ```

use std::sync::Arc;

use barracuda::device::WgpuDevice;
use barracuda::ops::moving_window_stats::MovingWindowStats;

use crate::len_f64;

/// Smoothed time series from a sliding window operation.
#[derive(Debug, Clone)]
pub struct SmoothedSeries {
    /// Sliding window mean.
    pub mean: Vec<f64>,
    /// Sliding window variance.
    pub variance: Vec<f64>,
    /// Sliding window minimum.
    pub min: Vec<f64>,
    /// Sliding window maximum.
    pub max: Vec<f64>,
    /// Window size used.
    pub window_size: usize,
    /// Number of output points (`input_len - window_size + 1`).
    pub len: usize,
}

/// GPU-backed sliding window smoother for `IoT` sensor streams.
///
/// Wraps `ToadStool`'s `MovingWindowStats` (f32 GPU shader, `wetSpring` provenance)
/// with f64 conversion for airSpring's precision requirements.
pub struct StreamSmoother {
    inner: MovingWindowStats,
}

impl StreamSmoother {
    /// Create a new stream smoother backed by the given GPU device.
    pub fn new(device: Arc<WgpuDevice>) -> Self {
        Self {
            inner: MovingWindowStats::new(device),
        }
    }

    /// Smooth a sensor time series using a sliding window on GPU.
    ///
    /// Converts f64 → f32 for GPU dispatch, then f32 → f64 for results.
    /// For typical sensor data (temperature, soil moisture, PAR), f32
    /// precision is more than sufficient for smoothing.
    ///
    /// # Errors
    ///
    /// Returns an error if the GPU dispatch fails or input is shorter than
    /// the window size.
    pub fn smooth(&self, data: &[f64], window_size: usize) -> crate::error::Result<SmoothedSeries> {
        if data.len() < window_size {
            return Err(crate::error::AirSpringError::InvalidInput(format!(
                "data length {} < window_size {window_size}",
                data.len()
            )));
        }

        #[allow(clippy::cast_possible_truncation)]
        let f32_data: Vec<f32> = data.iter().map(|&x| x as f32).collect();

        let result = self
            .inner
            .compute(&f32_data, window_size)
            .map_err(|e| crate::error::AirSpringError::Barracuda(format!("{e}")))?;

        let out_len = result.mean.len();
        Ok(SmoothedSeries {
            mean: result.mean.iter().map(|&x| f64::from(x)).collect(),
            variance: result.variance.iter().map(|&x| f64::from(x)).collect(),
            min: result.min.iter().map(|&x| f64::from(x)).collect(),
            max: result.max.iter().map(|&x| f64::from(x)).collect(),
            window_size,
            len: out_len,
        })
    }
}

/// CPU fallback for sliding window statistics (f64 precision).
///
/// Used when no GPU is available or for small datasets where GPU dispatch
/// overhead exceeds computation time.
pub fn smooth_cpu(data: &[f64], window_size: usize) -> Option<SmoothedSeries> {
    if data.len() < window_size || window_size == 0 {
        return None;
    }

    let out_len = data.len() - window_size + 1;
    let wf = len_f64(&data[..window_size]);
    let mut mean = Vec::with_capacity(out_len);
    let mut variance = Vec::with_capacity(out_len);
    let mut min_vals = Vec::with_capacity(out_len);
    let mut max_vals = Vec::with_capacity(out_len);

    for i in 0..out_len {
        let window = &data[i..i + window_size];
        let sum: f64 = window.iter().sum();
        let m = sum / wf;
        let var: f64 = window.iter().map(|&x| (x - m).powi(2)).sum::<f64>() / wf;
        let wmin = window.iter().copied().fold(f64::INFINITY, f64::min);
        let wmax = window.iter().copied().fold(f64::NEG_INFINITY, f64::max);

        mean.push(m);
        variance.push(var);
        min_vals.push(wmin);
        max_vals.push(wmax);
    }

    Some(SmoothedSeries {
        mean,
        variance,
        min: min_vals,
        max: max_vals,
        window_size,
        len: out_len,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_constant_signal() {
        let data = vec![5.0; 100];
        let result = smooth_cpu(&data, 10).unwrap();
        assert_eq!(result.len, 91);
        for &m in &result.mean {
            assert!((m - 5.0).abs() < 1e-10);
        }
        for &v in &result.variance {
            assert!(v.abs() < 1e-10);
        }
    }

    #[test]
    fn test_cpu_ramp_signal() {
        let data: Vec<f64> = (0..20).map(f64::from).collect();
        let result = smooth_cpu(&data, 5).unwrap();
        assert_eq!(result.len, 16);
        // First window [0,1,2,3,4] → mean=2.0
        assert!((result.mean[0] - 2.0).abs() < 1e-10);
        // Last window [15,16,17,18,19] → mean=17.0
        assert!((result.mean[15] - 17.0).abs() < 1e-10);
        // Min of first window = 0
        assert!(result.min[0].abs() < 1e-10);
        // Max of first window = 4
        assert!((result.max[0] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_cpu_window_too_large() {
        let data = vec![1.0, 2.0, 3.0];
        assert!(smooth_cpu(&data, 5).is_none());
    }

    #[test]
    fn test_cpu_window_equals_data() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = smooth_cpu(&data, 5).unwrap();
        assert_eq!(result.len, 1);
        assert!((result.mean[0] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_cpu_diurnal_temperature() {
        use std::f64::consts::PI;
        let data: Vec<f64> = (0..168)
            .map(|i| {
                let hour = f64::from(i);
                8.0f64.mul_add(((hour % 24.0 - 14.0) * PI / 12.0).cos(), 25.0)
            })
            .collect();
        let result = smooth_cpu(&data, 24).unwrap();
        assert_eq!(result.len, 145);
        // 24-hour window on diurnal cycle → mean ≈ 25°C
        for &m in &result.mean {
            assert!(
                (m - 25.0).abs() < 0.5,
                "24h smoothed mean should be ~25°C, got {m}"
            );
        }
    }

    #[test]
    fn test_smoothed_series_fields() {
        let data = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let result = smooth_cpu(&data, 3).unwrap();
        assert_eq!(result.window_size, 3);
        assert_eq!(result.len, 3);
        assert_eq!(result.mean.len(), 3);
        assert_eq!(result.variance.len(), 3);
        assert_eq!(result.min.len(), 3);
        assert_eq!(result.max.len(), 3);
    }

    // ── Edge cases: window size, empty, zero ───────────────────────────────

    #[test]
    fn test_cpu_window_size_one() {
        // Window size 1: each output = input (no smoothing)
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = smooth_cpu(&data, 1).unwrap();
        assert_eq!(result.len, 5);
        assert_eq!(result.mean, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        for &v in &result.variance {
            assert!(v.abs() < 1e-10);
        }
    }

    #[test]
    fn test_cpu_empty_data() {
        assert!(smooth_cpu(&[], 5).is_none());
    }

    #[test]
    fn test_cpu_zero_window() {
        assert!(smooth_cpu(&[1.0, 2.0, 3.0], 0).is_none());
    }

    #[test]
    fn test_cpu_window_equals_data_length() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = smooth_cpu(&data, 5).unwrap();
        assert_eq!(result.len, 1);
        assert!((result.mean[0] - 3.0).abs() < 1e-10);
        assert!((result.min[0] - 1.0).abs() < 1e-10);
        assert!((result.max[0] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_cpu_window_greater_than_data() {
        let data = vec![1.0, 2.0, 3.0];
        assert!(smooth_cpu(&data, 5).is_none());
    }

    // ── Hand-computed moving average ───────────────────────────────────────

    #[test]
    fn test_cpu_hand_computed_moving_average() {
        // [1, 2, 3, 4, 5] window=3
        // Window 1: [1,2,3] → mean=2.0
        // Window 2: [2,3,4] → mean=3.0
        // Window 3: [3,4,5] → mean=4.0
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = smooth_cpu(&data, 3).unwrap();
        assert_eq!(result.len, 3);
        assert!((result.mean[0] - 2.0).abs() < 1e-10);
        assert!((result.mean[1] - 3.0).abs() < 1e-10);
        assert!((result.mean[2] - 4.0).abs() < 1e-10);
        // Variance of [1,2,3]: mean=2, ss=(1+0+1)/3 = 2/3
        assert!((result.variance[0] - 2.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_cpu_min_max_per_window() {
        let data = vec![10.0, 5.0, 15.0, 3.0, 20.0];
        let result = smooth_cpu(&data, 3).unwrap();
        // Window 1: [10,5,15] → min=5, max=15
        assert!((result.min[0] - 5.0).abs() < 1e-10);
        assert!((result.max[0] - 15.0).abs() < 1e-10);
        // Window 2: [5,15,3] → min=3, max=15
        assert!((result.min[1] - 3.0).abs() < 1e-10);
        assert!((result.max[1] - 15.0).abs() < 1e-10);
        // Window 3: [15,3,20] → min=3, max=20
        assert!((result.min[2] - 3.0).abs() < 1e-10);
        assert!((result.max[2] - 20.0).abs() < 1e-10);
    }

    // ── Smoothing property: output variance ≤ input variance ─────────────────

    #[test]
    fn test_cpu_smoothing_reduces_variance() {
        // Noisy signal: moving average should smooth it
        let data: Vec<f64> = (0..50)
            .map(|i| {
                let x = f64::from(i);
                (x * 0.3).sin().mul_add(2.0, x) // oscillating noise
            })
            .collect();
        let result = smooth_cpu(&data, 10).unwrap();
        let input_var = crate::gpu::reduce::sample_variance(&data);
        let output_var_mean: f64 =
            result.variance.iter().sum::<f64>() / result.variance.len() as f64;
        // Smoothed output should have lower average variance per window
        assert!(
            output_var_mean < input_var * 1.5,
            "smoothing should reduce variance: output_avg_var={output_var_mean} input_var={input_var}"
        );
    }

    #[test]
    fn test_cpu_constant_signal_zero_variance() {
        let data = vec![7.0; 20];
        let result = smooth_cpu(&data, 5).unwrap();
        for &v in &result.variance {
            assert!(v.abs() < 1e-10);
        }
    }

    // ── SmoothedSeries struct ───────────────────────────────────────────────

    #[test]
    fn test_smoothed_series_clone() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = smooth_cpu(&data, 3).unwrap();
        let cloned = result.clone();
        assert_eq!(cloned.len, result.len);
        assert_eq!(cloned.window_size, result.window_size);
    }

    // ── StreamSmoother (device-backed, skips if no GPU) ───────────────────────

    fn try_device() -> Option<std::sync::Arc<barracuda::device::WgpuDevice>> {
        pollster::block_on(barracuda::device::WgpuDevice::new_f64_capable())
            .ok()
            .map(std::sync::Arc::new)
    }

    #[test]
    fn test_stream_smoother_new_and_smooth() {
        let Some(device) = try_device() else {
            eprintln!("SKIP: No GPU device for StreamSmoother");
            return;
        };
        let smoother = StreamSmoother::new(device);
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let result = smoother.smooth(&data, 3).unwrap();
        assert_eq!(result.len, 8);
        assert_eq!(result.window_size, 3);
        let cpu_result = smooth_cpu(&data, 3).unwrap();
        assert_eq!(result.mean.len(), cpu_result.mean.len());
        for (i, (&g, &c)) in result.mean.iter().zip(&cpu_result.mean).enumerate() {
            assert!((g - c).abs() < 0.01, "mean[{i}]: GPU={g} CPU={c}");
        }
    }

    #[test]
    fn test_stream_smoother_empty_data_error() {
        let Some(device) = try_device() else {
            eprintln!("SKIP: No GPU device for StreamSmoother");
            return;
        };
        let smoother = StreamSmoother::new(device);
        let result = smoother.smooth(&[], 5);
        assert!(result.is_err());
    }

    #[test]
    fn test_stream_smoother_window_too_large_error() {
        let Some(device) = try_device() else {
            eprintln!("SKIP: No GPU device for StreamSmoother");
            return;
        };
        let smoother = StreamSmoother::new(device);
        let data = vec![1.0, 2.0, 3.0];
        let result = smoother.smooth(&data, 5);
        assert!(result.is_err());
    }

    #[test]
    fn test_stream_smoother_window_size_one() {
        let Some(device) = try_device() else {
            eprintln!("SKIP: No GPU device for StreamSmoother");
            return;
        };
        let smoother = StreamSmoother::new(device);
        let data = vec![1.0, 2.0, 3.0];
        let result = smoother.smooth(&data, 1).unwrap();
        assert_eq!(result.len, 3);
        assert!((result.mean[0] - 1.0).abs() < 0.01);
        assert!((result.mean[1] - 2.0).abs() < 0.01);
        assert!((result.mean[2] - 3.0).abs() < 0.01);
    }
}
