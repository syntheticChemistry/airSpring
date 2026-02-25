// SPDX-License-Identifier: AGPL-3.0-or-later
//! Seasonal statistics via `ToadStool` fused map-reduce — GPU-accelerated.
//!
//! Wraps [`barracuda::ops::fused_map_reduce_f64::FusedMapReduceF64`] for
//! precision agriculture aggregate statistics: seasonal ET₀ totals,
//! field-level VWC averages, and `IoT` stream summaries.
//!
//! # Two API Levels
//!
//! | API | GPU? | Dependency |
//! |-----|:----:|------------|
//! | Free functions (`seasonal_sum`, etc.) | No | None (CPU iterator) |
//! | [`SeasonalReducer`] | Yes | `Arc<WgpuDevice>` |
//!
//! # `ToadStool` Primitive
//!
//! [`SeasonalReducer`] wraps `FusedMapReduceF64` which automatically dispatches
//! to GPU for N ≥ 1024 elements and CPU for smaller arrays. The fused kernel
//! applies a map function (identity, square, abs, log, etc.) and reduces
//! (sum, max, min, product) in a single pass.
//!
//! **TS-004 resolved** (commit `0c477306`): buffer usage conflict in the
//! partials pipeline is fixed. GPU dispatch for N ≥ 1024 now works correctly.

use std::sync::Arc;

use barracuda::device::WgpuDevice;
use barracuda::ops::fused_map_reduce_f64::FusedMapReduceF64;

use crate::len_f64;

// ── Device-backed reducer (wraps barracuda::ops::fused_map_reduce_f64) ──

/// Device-backed seasonal reducer that dispatches to GPU for large arrays.
///
/// Wraps [`FusedMapReduceF64`] with precision-agriculture semantics.
/// For arrays with N ≥ 1024 elements, computations run on the GPU; smaller
/// arrays use an optimised CPU path.
///
/// # Construction
///
/// Requires an `Arc<WgpuDevice>`. Create one via `WgpuDevice::new()` (async)
/// or `WgpuDevice::new_cpu()` for headless environments.
pub struct SeasonalReducer {
    engine: FusedMapReduceF64,
}

impl SeasonalReducer {
    /// Create a new device-backed seasonal reducer.
    ///
    /// # Errors
    ///
    /// Returns an error if the `FusedMapReduceF64` engine cannot be initialised.
    pub fn new(device: Arc<WgpuDevice>) -> crate::error::Result<Self> {
        let engine = FusedMapReduceF64::new(device)
            .map_err(|e| crate::error::AirSpringError::Barracuda(format!("{e}")))?;
        Ok(Self { engine })
    }

    /// GPU-accelerated seasonal sum.
    ///
    /// # Errors
    ///
    /// Returns an error if the GPU dispatch fails.
    pub fn sum(&self, values: &[f64]) -> crate::error::Result<f64> {
        self.engine
            .sum(values)
            .map_err(|e| crate::error::AirSpringError::Barracuda(format!("{e}")))
    }

    /// GPU-accelerated seasonal maximum.
    ///
    /// # Errors
    ///
    /// Returns an error if the GPU dispatch fails.
    pub fn max(&self, values: &[f64]) -> crate::error::Result<f64> {
        self.engine
            .max(values)
            .map_err(|e| crate::error::AirSpringError::Barracuda(format!("{e}")))
    }

    /// GPU-accelerated seasonal minimum.
    ///
    /// # Errors
    ///
    /// Returns an error if the GPU dispatch fails.
    pub fn min(&self, values: &[f64]) -> crate::error::Result<f64> {
        self.engine
            .min(values)
            .map_err(|e| crate::error::AirSpringError::Barracuda(format!("{e}")))
    }

    /// GPU-accelerated sum of squares (for variance computation).
    ///
    /// # Errors
    ///
    /// Returns an error if the GPU dispatch fails.
    pub fn sum_of_squares(&self, values: &[f64]) -> crate::error::Result<f64> {
        self.engine
            .sum_of_squares(values)
            .map_err(|e| crate::error::AirSpringError::Barracuda(format!("{e}")))
    }

    /// Compute all seasonal statistics via GPU-accelerated reductions.
    ///
    /// Dispatches sum, max, min, and sum-of-squares as separate GPU passes.
    /// Mean, variance, and std deviation are derived on CPU.
    ///
    /// # Errors
    ///
    /// Returns an error if any GPU dispatch fails.
    pub fn compute_stats(&self, values: &[f64]) -> crate::error::Result<SeasonalStats> {
        if values.is_empty() {
            return Ok(SeasonalStats {
                total: 0.0,
                mean: 0.0,
                max: f64::NEG_INFINITY,
                min: f64::INFINITY,
                std_dev: 0.0,
                count: 0,
            });
        }

        let total = self.sum(values)?;
        let max = self.max(values)?;
        let min = self.min(values)?;
        let n = len_f64(values);
        let mean = total / n;

        // Variance: E[X²] - E[X]² (computational formula)
        let sum_sq = self.sum_of_squares(values)?;
        let variance = if values.len() > 1 {
            // Bessel correction: sample variance = (Σx² - n·μ²) / (n - 1)
            (n * mean).mul_add(-mean, sum_sq) / (n - 1.0)
        } else {
            0.0
        };

        Ok(SeasonalStats {
            total,
            mean,
            max,
            min,
            std_dev: variance.max(0.0).sqrt(),
            count: values.len(),
        })
    }
}

// ── Free functions: CPU-only (no device needed) ─────────────────────

/// Compute seasonal sum (e.g., total ET₀ over a growing season).
#[must_use]
pub fn seasonal_sum(values: &[f64]) -> f64 {
    values.iter().sum()
}

/// Compute seasonal mean.
///
/// Returns 0.0 for empty slices.
#[must_use]
pub fn seasonal_mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    seasonal_sum(values) / len_f64(values)
}

/// Compute seasonal maximum.
///
/// Returns `f64::NEG_INFINITY` for empty slices.
#[must_use]
pub fn seasonal_max(values: &[f64]) -> f64 {
    values.iter().copied().fold(f64::NEG_INFINITY, f64::max)
}

/// Compute seasonal minimum.
///
/// Returns `f64::INFINITY` for empty slices.
#[must_use]
pub fn seasonal_min(values: &[f64]) -> f64 {
    values.iter().copied().fold(f64::INFINITY, f64::min)
}

/// Compute sum of squared deviations from mean (for variance).
///
/// When `ToadStool` `FusedMapReduceF64` is wired, this dispatches to
/// `MapOp::Square` + `ReduceOp::Sum` on centered data.
#[must_use]
pub fn sum_of_squares_from_mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mean = seasonal_mean(values);
    values.iter().map(|&v| (v - mean).powi(2)).sum()
}

/// Compute sample variance.
#[must_use]
pub fn sample_variance(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    // Bessel correction: divide by (n − 1)
    sum_of_squares_from_mean(values) / (len_f64(values) - 1.0)
}

/// Compute sample standard deviation.
#[must_use]
pub fn sample_std_dev(values: &[f64]) -> f64 {
    sample_variance(values).sqrt()
}

/// Seasonal summary statistics for a time series.
#[derive(Debug, Clone, Copy)]
pub struct SeasonalStats {
    /// Total (sum).
    pub total: f64,
    /// Mean.
    pub mean: f64,
    /// Maximum.
    pub max: f64,
    /// Minimum.
    pub min: f64,
    /// Sample standard deviation.
    pub std_dev: f64,
    /// Count of values.
    pub count: usize,
}

/// Compute all seasonal statistics in a single pass.
#[must_use]
pub fn compute_seasonal_stats(values: &[f64]) -> SeasonalStats {
    SeasonalStats {
        total: seasonal_sum(values),
        mean: seasonal_mean(values),
        max: seasonal_max(values),
        min: seasonal_min(values),
        std_dev: sample_std_dev(values),
        count: values.len(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_seasonal_sum() {
        let vals = [1.0, 2.0, 3.0, 4.0, 5.0];
        assert!((seasonal_sum(&vals) - 15.0).abs() < 1e-10);
    }

    #[test]
    fn test_seasonal_mean() {
        let vals = [1.0, 2.0, 3.0, 4.0, 5.0];
        assert!((seasonal_mean(&vals) - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_seasonal_max_min() {
        let vals = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0];
        assert!((seasonal_max(&vals) - 9.0).abs() < 1e-10);
        assert!((seasonal_min(&vals) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_sample_variance() {
        // Known: [2, 4, 6, 8, 10] → mean=6, var=10
        let vals = [2.0, 4.0, 6.0, 8.0, 10.0];
        assert!((sample_variance(&vals) - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_stats_summary() {
        let et0 = [4.2, 5.1, 3.8, 6.0, 4.5];
        let stats = compute_seasonal_stats(&et0);
        assert_eq!(stats.count, 5);
        assert!((stats.total - 23.6).abs() < 1e-10);
        assert!((stats.max - 6.0).abs() < 1e-10);
        assert!((stats.min - 3.8).abs() < 1e-10);
    }

    #[test]
    fn test_empty_slice() {
        let vals: &[f64] = &[];
        assert!((seasonal_sum(vals)).abs() < 1e-10);
        assert!((seasonal_mean(vals)).abs() < 1e-10);
        assert!((sample_variance(vals)).abs() < 1e-10);
    }

    #[test]
    fn test_single_element() {
        let vals = [7.0];
        assert!((seasonal_sum(&vals) - 7.0).abs() < 1e-10);
        assert!((seasonal_mean(&vals) - 7.0).abs() < 1e-10);
        assert!((seasonal_max(&vals) - 7.0).abs() < 1e-10);
        assert!((seasonal_min(&vals) - 7.0).abs() < 1e-10);
        assert!((sample_variance(&vals)).abs() < 1e-10);
        assert!((sample_std_dev(&vals)).abs() < 1e-10);
    }

    #[test]
    fn test_sample_std_dev() {
        let vals = [2.0, 4.0, 6.0, 8.0, 10.0];
        let sd = sample_std_dev(&vals);
        assert!((sd - 10.0_f64.sqrt()).abs() < 1e-10, "std_dev={sd}");
    }

    #[test]
    fn test_sum_of_squares_from_mean() {
        // [1, 2, 3] → mean=2, ss = 1+0+1 = 2
        let vals = [1.0, 2.0, 3.0];
        assert!((sum_of_squares_from_mean(&vals) - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_sum_of_squares_empty() {
        assert!((sum_of_squares_from_mean(&[])).abs() < 1e-10);
    }

    #[test]
    fn test_compute_seasonal_stats_empty() {
        let stats = compute_seasonal_stats(&[]);
        assert_eq!(stats.count, 0);
        assert!((stats.total).abs() < 1e-10);
        assert!((stats.mean).abs() < 1e-10);
    }

    #[test]
    fn test_compute_seasonal_stats_large() {
        let vals: Vec<f64> = (1..=100).map(f64::from).collect();
        let stats = compute_seasonal_stats(&vals);
        assert_eq!(stats.count, 100);
        assert!((stats.total - 5050.0).abs() < 1e-10);
        assert!((stats.mean - 50.5).abs() < 1e-10);
        assert!((stats.max - 100.0).abs() < 1e-10);
        assert!((stats.min - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_seasonal_max_min_negative() {
        let vals = [-5.0, -1.0, -3.0, -7.0];
        assert!((seasonal_max(&vals) - (-1.0)).abs() < 1e-10);
        assert!((seasonal_min(&vals) - (-7.0)).abs() < 1e-10);
    }

    #[test]
    fn test_empty_max_min_sentinels() {
        let vals: &[f64] = &[];
        assert!(seasonal_max(vals) == f64::NEG_INFINITY);
        assert!(seasonal_min(vals) == f64::INFINITY);
    }

    #[test]
    fn test_constant_values_zero_variance() {
        let vals = [3.0; 10];
        assert!((sample_variance(&vals)).abs() < 1e-10);
        assert!((sample_std_dev(&vals)).abs() < 1e-10);
    }

    // ── SeasonalStats struct ───────────────────────────────────────────────

    #[test]
    fn test_seasonal_stats_debug_clone_copy() {
        let stats = SeasonalStats {
            total: 10.0,
            mean: 2.0,
            max: 4.0,
            min: 0.0,
            std_dev: 1.5,
            count: 5,
        };
        let _ = format!("{stats:?}");
        let cloned = stats;
        let copy = stats;
        assert_eq!(copy.count, cloned.count);
        assert_eq!(cloned.count, 5);
    }

    #[test]
    fn test_compute_seasonal_stats_all_same_values() {
        let vals = [7.5; 20];
        let stats = compute_seasonal_stats(&vals);
        assert_eq!(stats.count, 20);
        assert!((stats.total - 150.0).abs() < 1e-10);
        assert!((stats.mean - 7.5).abs() < 1e-10);
        assert!((stats.max - 7.5).abs() < 1e-10);
        assert!((stats.min - 7.5).abs() < 1e-10);
        assert!((stats.std_dev).abs() < 1e-10);
    }

    #[test]
    fn test_sample_variance_n_less_than_two() {
        assert!((sample_variance(&[])).abs() < 1e-10);
        assert!((sample_variance(&[42.0])).abs() < 1e-10);
    }

    #[test]
    fn test_hand_computed_stats() {
        // [1, 3, 5, 7, 9] → sum=25, mean=5, var=10, std=√10
        let vals = [1.0, 3.0, 5.0, 7.0, 9.0];
        let stats = compute_seasonal_stats(&vals);
        assert!((stats.total - 25.0).abs() < 1e-10);
        assert!((stats.mean - 5.0).abs() < 1e-10);
        assert!((stats.max - 9.0).abs() < 1e-10);
        assert!((stats.min - 1.0).abs() < 1e-10);
        assert!((stats.std_dev - 10.0_f64.sqrt()).abs() < 1e-10);
    }

    // ── SeasonalReducer (device-backed, skips if no GPU) ────────────────────

    fn try_device() -> Option<std::sync::Arc<barracuda::device::WgpuDevice>> {
        pollster::block_on(barracuda::device::WgpuDevice::new_f64_capable())
            .ok()
            .map(std::sync::Arc::new)
    }

    #[test]
    fn test_seasonal_reducer_new_and_sum() {
        let Some(device) = try_device() else {
            eprintln!("SKIP: No GPU device for SeasonalReducer");
            return;
        };
        let reducer = SeasonalReducer::new(device).unwrap();
        let vals = [1.0, 2.0, 3.0, 4.0, 5.0];
        let sum = reducer.sum(&vals).unwrap();
        assert!((sum - 15.0).abs() < 1e-10);
    }

    #[test]
    fn test_seasonal_reducer_max_min_sum_of_squares() {
        let Some(device) = try_device() else {
            eprintln!("SKIP: No GPU device for SeasonalReducer");
            return;
        };
        let reducer = SeasonalReducer::new(device).unwrap();
        let vals = [2.0, 4.0, 6.0, 8.0];
        assert!((reducer.max(&vals).unwrap() - 8.0).abs() < 1e-10);
        assert!((reducer.min(&vals).unwrap() - 2.0).abs() < 1e-10);
        // sum of squares: 4+16+36+64 = 120
        assert!((reducer.sum_of_squares(&vals).unwrap() - 120.0).abs() < 1e-10);
    }

    #[test]
    fn test_seasonal_reducer_compute_stats_empty() {
        let Some(device) = try_device() else {
            eprintln!("SKIP: No GPU device for SeasonalReducer");
            return;
        };
        let reducer = SeasonalReducer::new(device).unwrap();
        let stats = reducer.compute_stats(&[]).unwrap();
        assert_eq!(stats.count, 0);
        assert!((stats.total).abs() < 1e-10);
        assert!(stats.max == f64::NEG_INFINITY);
        assert!(stats.min == f64::INFINITY);
    }

    #[test]
    fn test_seasonal_reducer_compute_stats_single_element() {
        let Some(device) = try_device() else {
            eprintln!("SKIP: No GPU device for SeasonalReducer");
            return;
        };
        let reducer = SeasonalReducer::new(device).unwrap();
        let vals = [42.0];
        let stats = reducer.compute_stats(&vals).unwrap();
        assert_eq!(stats.count, 1);
        assert!((stats.total - 42.0).abs() < 1e-10);
        assert!((stats.mean - 42.0).abs() < 1e-10);
        assert!((stats.std_dev).abs() < 1e-10);
    }

    #[test]
    fn test_seasonal_reducer_compute_stats_matches_cpu() {
        let Some(device) = try_device() else {
            eprintln!("SKIP: No GPU device for SeasonalReducer");
            return;
        };
        let reducer = SeasonalReducer::new(device).unwrap();
        let vals: Vec<f64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
            .into_iter()
            .collect();
        let gpu_stats = reducer.compute_stats(&vals).unwrap();
        let cpu_stats = compute_seasonal_stats(&vals);
        assert_eq!(gpu_stats.count, cpu_stats.count);
        assert!((gpu_stats.total - cpu_stats.total).abs() < 1e-6);
        assert!((gpu_stats.mean - cpu_stats.mean).abs() < 1e-6);
        assert!((gpu_stats.max - cpu_stats.max).abs() < 1e-10);
        assert!((gpu_stats.min - cpu_stats.min).abs() < 1e-10);
        assert!((gpu_stats.std_dev - cpu_stats.std_dev).abs() < 0.01);
    }

    #[test]
    fn test_seasonal_reducer_cpu_path_small_array() {
        // N=10 < 1024 → CPU path in FusedMapReduceF64
        let Some(device) = try_device() else {
            eprintln!("SKIP: No GPU device for SeasonalReducer");
            return;
        };
        let reducer = SeasonalReducer::new(device).unwrap();
        let vals: Vec<f64> = (0..10).map(f64::from).collect();
        let sum = reducer.sum(&vals).unwrap();
        assert!((sum - 45.0).abs() < 1e-10);
    }

    #[test]
    fn test_seasonal_reducer_gpu_path_large_array() {
        // N=1024 → GPU dispatch threshold
        let Some(device) = try_device() else {
            eprintln!("SKIP: No GPU device for SeasonalReducer");
            return;
        };
        let reducer = SeasonalReducer::new(device).unwrap();
        let vals: Vec<f64> = (0..1024).map(f64::from).collect();
        let expected_sum: f64 = (0..1024).map(f64::from).sum();
        let sum = reducer.sum(&vals).unwrap();
        assert!(
            (sum - expected_sum).abs() < 1e-4,
            "sum={sum} expected={expected_sum}"
        );
    }
}
