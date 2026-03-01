// SPDX-License-Identifier: AGPL-3.0-or-later
//! High-cadence streaming classification pipeline.
//!
//! Real-time sensor processing with exponential rolling statistics and
//! CPU-side pre-screening before optional NPU inference.
//!
//! # Design
//!
//! - `O(1)` per reading (no buffer allocation)
//! - Exponential moving average for trend tracking
//! - Z-score anomaly detection with minimum sample guard

use super::quantize_i8;

/// Multi-sensor fusion input (θ + T + EC) for single-inference field state.
pub struct MultiSensorInput {
    /// Volumetric water content θ (0–0.6 cm³/cm³).
    pub theta: f64,
    /// Soil temperature (−10–50 °C).
    pub temperature: f64,
    /// Electrical conductivity (0–5 dS/m).
    pub ec: f64,
    /// Depletion fraction (0–1).
    pub depletion: f64,
    /// Hour of day normalized (0–1).
    pub hour_norm: f64,
    /// Days since last irrigation normalized (0–1, capped at 14 days).
    pub days_since_irr: f64,
}

impl MultiSensorInput {
    /// Quantize all 6 features to int8 for NPU.
    #[must_use]
    pub fn to_i8(&self) -> Vec<i8> {
        vec![
            quantize_i8(self.theta, 0.0, 0.6),
            quantize_i8(self.temperature, -10.0, 50.0),
            quantize_i8(self.ec, 0.0, 5.0),
            quantize_i8(self.depletion, 0.0, 1.0),
            quantize_i8(self.hour_norm, 0.0, 1.0),
            quantize_i8(self.days_since_irr, 0.0, 1.0),
        ]
    }
}

/// Rolling statistics tracker for streaming sensor data.
///
/// Uses exponential moving average for O(1) updates with no buffer allocation.
pub struct RollingStats {
    mean: f64,
    variance: f64,
    min: f64,
    max: f64,
    count: u64,
    alpha: f64,
}

const MIN_ANOMALY_SAMPLES: u64 = 10;
const SIGMA_FLOOR: f64 = 1e-10;
const STRESS_DEPLETION_THRESHOLD: f64 = 0.55;

impl RollingStats {
    /// Create a new tracker with the given smoothing factor.
    ///
    /// `alpha` controls responsiveness: 0.1 = smooth, 0.3 = responsive.
    #[must_use]
    pub const fn new(alpha: f64) -> Self {
        Self {
            mean: 0.0,
            variance: 0.0,
            min: f64::MAX,
            max: f64::MIN,
            count: 0,
            alpha,
        }
    }

    /// Update with a new reading and return the current sigma.
    pub fn update(&mut self, value: f64) -> f64 {
        if self.count == 0 {
            self.mean = value;
            self.variance = 0.0;
        } else {
            let delta = value - self.mean;
            self.mean += self.alpha * delta;
            self.variance = (1.0 - self.alpha).mul_add(self.variance, self.alpha * delta * delta);
        }
        self.min = self.min.min(value);
        self.max = self.max.max(value);
        self.count += 1;
        self.sigma()
    }

    /// Current standard deviation.
    #[must_use]
    pub fn sigma(&self) -> f64 {
        self.variance.sqrt()
    }

    /// Current mean.
    #[must_use]
    pub const fn mean(&self) -> f64 {
        self.mean
    }

    /// Number of readings processed.
    #[must_use]
    pub const fn count(&self) -> u64 {
        self.count
    }

    /// Observed range.
    #[must_use]
    pub const fn range(&self) -> (f64, f64) {
        (self.min, self.max)
    }
}

/// Streaming classification result for a single reading.
#[derive(Debug, Clone)]
pub struct StreamClassification {
    /// Classified state: 0=normal, 1=stressed, 2=anomaly (or domain-specific).
    pub class: usize,
    /// Rolling sigma at this reading.
    pub sigma: f64,
    /// Rolling mean at this reading.
    pub mean: f64,
    /// Whether this reading triggered the anomaly detector.
    pub is_anomaly: bool,
}

/// Classify a sensor reading against rolling statistics.
///
/// Uses a simple z-score threshold for anomaly detection and depletion
/// threshold for stress classification. Designed for CPU-side pre-screening
/// before optional NPU inference.
#[must_use]
#[allow(clippy::bool_to_int_with_if)]
pub fn classify_reading(
    reading: f64,
    stats: &RollingStats,
    fc: f64,
    wp: f64,
    anomaly_z: f64,
) -> StreamClassification {
    let sigma = stats.sigma();
    let mean = stats.mean();

    let is_anomaly = if sigma > SIGMA_FLOOR && stats.count() > MIN_ANOMALY_SAMPLES {
        ((reading - mean) / sigma).abs() > anomaly_z
    } else {
        false
    };

    let depletion = (fc - reading) / (fc - wp);
    let class = if is_anomaly {
        2
    } else if depletion > STRESS_DEPLETION_THRESHOLD {
        1
    } else {
        0
    };

    StreamClassification {
        class,
        sigma,
        mean,
        is_anomaly,
    }
}

/// Result of a high-cadence streaming session.
#[derive(Debug, Clone)]
pub struct StreamSessionResult {
    /// Total readings processed.
    pub total_readings: usize,
    /// Readings classified as normal.
    pub normal_count: usize,
    /// Readings classified as stressed.
    pub stressed_count: usize,
    /// Readings flagged as anomalies.
    pub anomaly_count: usize,
    /// Mean classification time (nanoseconds, CPU-side).
    pub mean_classify_ns: f64,
    /// Final rolling sigma.
    pub final_sigma: f64,
    /// Final rolling mean.
    pub final_mean: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn multi_sensor_input_quantizes() {
        let input = MultiSensorInput {
            theta: 0.30,
            temperature: 25.0,
            ec: 2.5,
            depletion: 0.4,
            hour_norm: 0.5,
            days_since_irr: 0.3,
        };
        let q = input.to_i8();
        assert_eq!(q.len(), 6);
        assert!(q[0] > 50, "θ=0.30 in [0,0.6] should be ~midrange");
    }

    #[test]
    fn rolling_stats_tracks() {
        let mut stats = RollingStats::new(0.1);
        for i in 0..100 {
            stats.update(f64::from(i));
        }
        assert!(stats.count() == 100);
        assert!(stats.mean() > 50.0);
        assert!(stats.sigma() > 0.0);
        assert_eq!(stats.range(), (0.0, 99.0));
    }

    #[test]
    fn classify_detects_anomaly() {
        let mut stats = RollingStats::new(0.1);
        for i in 0..50 {
            stats.update(0.005f64.mul_add(f64::from(i % 3) - 1.0, 0.30));
        }
        let result = classify_reading(0.95, &stats, 0.38, 0.15, 3.0);
        assert!(result.is_anomaly, "0.95 should be anomalous when mean≈0.30");
        assert_eq!(result.class, 2);
    }

    #[test]
    fn classify_detects_stress() {
        let mut stats = RollingStats::new(0.1);
        for _ in 0..50 {
            stats.update(0.20);
        }
        let result = classify_reading(0.20, &stats, 0.38, 0.15, 3.0);
        assert!(!result.is_anomaly);
        assert_eq!(result.class, 1, "depletion > 0.55 should be stressed");
    }

    #[test]
    fn classify_normal() {
        let mut stats = RollingStats::new(0.1);
        for _ in 0..50 {
            stats.update(0.35);
        }
        let result = classify_reading(0.35, &stats, 0.38, 0.15, 3.0);
        assert!(!result.is_anomaly);
        assert_eq!(result.class, 0, "low depletion should be normal");
    }

    #[test]
    fn rolling_stats_constant_signal() {
        let mut stats = RollingStats::new(0.1);
        for _ in 0..200 {
            stats.update(42.0);
        }
        assert!((stats.mean() - 42.0).abs() < 0.01);
        assert!(
            stats.sigma() < 0.01,
            "σ should be ~0 for constant: {}",
            stats.sigma()
        );
    }

    #[test]
    fn rolling_stats_step_change_adapts() {
        let mut stats = RollingStats::new(0.3);
        for _ in 0..100 {
            stats.update(10.0);
        }
        for _ in 0..100 {
            stats.update(20.0);
        }
        assert!(
            stats.mean() > 17.0,
            "mean should adapt to 20 after step change: {}",
            stats.mean()
        );
    }

    #[test]
    fn rolling_stats_first_reading_sets_mean() {
        let mut stats = RollingStats::new(0.1);
        stats.update(99.0);
        assert!((stats.mean() - 99.0).abs() < f64::EPSILON);
        assert!(stats.sigma().abs() < f64::EPSILON);
    }

    #[test]
    fn rolling_stats_range_tracks_extremes() {
        let mut stats = RollingStats::new(0.1);
        stats.update(5.0);
        stats.update(100.0);
        stats.update(-3.0);
        stats.update(50.0);
        assert_eq!(stats.range(), (-3.0, 100.0));
    }

    #[test]
    fn classify_insufficient_data_never_anomaly() {
        let mut stats = RollingStats::new(0.1);
        for _ in 0..5 {
            stats.update(0.30);
        }
        let result = classify_reading(99.0, &stats, 0.38, 0.15, 3.0);
        assert!(
            !result.is_anomaly,
            "count < 10 should suppress anomaly detection"
        );
    }

    #[test]
    fn classify_zero_sigma_never_anomaly() {
        let mut stats = RollingStats::new(0.1);
        for _ in 0..50 {
            stats.update(0.30);
        }
        let result = classify_reading(0.31, &stats, 0.38, 0.15, 3.0);
        assert!(
            !result.is_anomaly,
            "σ ≈ 0 should suppress anomaly detection"
        );
    }

    #[test]
    fn classify_depletion_boundary() {
        let mut stats = RollingStats::new(0.1);
        for _ in 0..50 {
            stats.update(0.25);
        }
        let fc = 0.38;
        let wp = 0.15;
        let theta_at_boundary = fc - 0.55 * (fc - wp);
        let below = classify_reading(theta_at_boundary - 0.01, &stats, fc, wp, 3.0);
        let above = classify_reading(theta_at_boundary + 0.01, &stats, fc, wp, 3.0);
        assert_eq!(below.class, 1, "below boundary should be stressed");
        assert_eq!(above.class, 0, "above boundary should be normal");
    }

    #[test]
    fn multi_sensor_boundary_values() {
        let low = MultiSensorInput {
            theta: 0.0,
            temperature: -10.0,
            ec: 0.0,
            depletion: 0.0,
            hour_norm: 0.0,
            days_since_irr: 0.0,
        };
        let high = MultiSensorInput {
            theta: 0.6,
            temperature: 50.0,
            ec: 5.0,
            depletion: 1.0,
            hour_norm: 1.0,
            days_since_irr: 1.0,
        };
        assert!(low.to_i8().iter().all(|&v| v == 0));
        assert!(high.to_i8().iter().all(|&v| v == 127));
    }
}
