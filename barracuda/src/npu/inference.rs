// SPDX-License-Identifier: AGPL-3.0-or-later
//! Eco-domain NPU batch inference — crop stress, irrigation, and anomaly classifiers.
//!
//! # Workloads
//!
//! 1. **Crop stress** — binary int8, 4 features → 2 classes
//! 2. **Irrigation decision** — 3-class int8, 6 features → 3 classes
//! 3. **Sensor anomaly** — binary int8, 3 features → 2 classes

use super::{Error, NpuHandle, quantize_i8};

/// Result of a single NPU int8 inference via DMA.
#[derive(Debug, Clone)]
pub struct NpuInferResult {
    /// Raw int8 accumulator outputs (one per class).
    pub raw_i8: Vec<i8>,
    /// Argmax class index.
    pub class: usize,
    /// DMA write latency (nanoseconds).
    pub write_ns: u64,
    /// DMA read latency (nanoseconds).
    pub read_ns: u64,
}

/// Run a single int8 inference round-trip on the NPU via DMA.
///
/// # Errors
///
/// Returns error if DMA transfer fails.
pub fn npu_infer_i8(
    handle: &mut NpuHandle,
    input_i8: &[i8],
    n_outputs: usize,
) -> Result<NpuInferResult, Error> {
    let input_bytes: Vec<u8> = input_i8.iter().map(|&x| x.cast_unsigned()).collect();

    let t = std::time::Instant::now();
    handle.write_raw(&input_bytes)?;
    let write_ns = u64::try_from(t.elapsed().as_nanos()).unwrap_or(u64::MAX);

    let mut out_buf = vec![0u8; n_outputs];
    let t = std::time::Instant::now();
    handle.read_raw(&mut out_buf)?;
    let read_ns = u64::try_from(t.elapsed().as_nanos()).unwrap_or(u64::MAX);

    let raw_i8: Vec<i8> = out_buf.iter().map(|&b| i8::from_ne_bytes([b])).collect();
    let class = raw_i8
        .iter()
        .enumerate()
        .max_by_key(|&(_, v)| *v)
        .map_or(0, |(i, _)| i);

    Ok(NpuInferResult {
        raw_i8,
        class,
        write_ns,
        read_ns,
    })
}

/// Crop stress classification features.
///
/// Quantized to int8 for NPU inference.
pub struct CropStressInput {
    /// Soil depletion fraction (0–1).
    pub depletion: f64,
    /// `ETa / ETc` ratio (0–1.5).
    pub et_ratio: f64,
    /// Volumetric water content θ (0–0.6).
    pub theta: f64,
    /// Stress coefficient `Ks` (0–1).
    pub ks: f64,
}

impl CropStressInput {
    /// Quantize features to int8 vector for NPU.
    #[must_use]
    pub fn to_i8(&self) -> Vec<i8> {
        vec![
            quantize_i8(self.depletion, 0.0, 1.0),
            quantize_i8(self.et_ratio, 0.0, 1.5),
            quantize_i8(self.theta, 0.0, 0.6),
            quantize_i8(self.ks, 0.0, 1.0),
        ]
    }
}

/// Irrigation decision features.
pub struct IrrigationInput {
    /// Forecast `ET₀` (mm/day, 0–15).
    pub forecast_et0: f64,
    /// Current soil moisture θ (0–0.6).
    pub theta: f64,
    /// Total available water `TAW` (mm, 0–300).
    pub taw: f64,
    /// Crop growth stage (0=initial, 1=mid, 2=late) normalized (0–1).
    pub stage: f64,
    /// Stress coefficient `Ks` (0–1).
    pub ks: f64,
    /// Rain probability (0–1).
    pub rain_prob: f64,
}

impl IrrigationInput {
    /// Quantize features to int8 vector for NPU.
    #[must_use]
    pub fn to_i8(&self) -> Vec<i8> {
        vec![
            quantize_i8(self.forecast_et0, 0.0, 15.0),
            quantize_i8(self.theta, 0.0, 0.6),
            quantize_i8(self.taw, 0.0, 300.0),
            quantize_i8(self.stage, 0.0, 1.0),
            quantize_i8(self.ks, 0.0, 1.0),
            quantize_i8(self.rain_prob, 0.0, 1.0),
        ]
    }
}

/// Sensor anomaly detection features.
pub struct SensorAnomalyInput {
    /// Current reading.
    pub reading: f64,
    /// Rolling mean.
    pub mean: f64,
    /// Rolling standard deviation.
    pub sigma: f64,
}

impl SensorAnomalyInput {
    /// Quantize features to int8 vector for NPU.
    ///
    /// The z-score `(reading - mean) / sigma` is the key feature;
    /// we pass all three so the NPU learns arbitrary nonlinear thresholds.
    #[must_use]
    pub fn to_i8(&self, scale_lo: f64, scale_hi: f64) -> Vec<i8> {
        vec![
            quantize_i8(self.reading, scale_lo, scale_hi),
            quantize_i8(self.mean, scale_lo, scale_hi),
            quantize_i8(self.sigma, 0.0, scale_hi - scale_lo),
        ]
    }
}

/// Run batch int8 inferences and return metrics.
///
/// # Errors
///
/// Returns error if any DMA transfer fails.
pub fn npu_batch_infer(
    handle: &mut NpuHandle,
    inputs_i8: &[Vec<i8>],
    n_outputs: usize,
) -> Result<NpuBatchResult, Error> {
    let mut classes = Vec::with_capacity(inputs_i8.len());
    let mut total_write_ns = 0u64;
    let mut total_read_ns = 0u64;

    for input in inputs_i8 {
        let r = npu_infer_i8(handle, input, n_outputs)?;
        classes.push(r.class);
        total_write_ns += r.write_ns;
        total_read_ns += r.read_ns;
    }

    let n = inputs_i8.len() as f64;
    let total_ns = total_write_ns + total_read_ns;
    Ok(NpuBatchResult {
        classes,
        mean_write_ns: total_write_ns as f64 / n,
        mean_read_ns: total_read_ns as f64 / n,
        total_us: total_ns as f64 / 1000.0,
        throughput_hz: if total_ns > 0 {
            n * 1_000_000_000.0 / total_ns as f64
        } else {
            0.0
        },
    })
}

/// Aggregate metrics from batch NPU inference.
#[derive(Debug, Clone)]
pub struct NpuBatchResult {
    /// Classified index for each input.
    pub classes: Vec<usize>,
    /// Mean DMA write latency per inference (ns).
    pub mean_write_ns: f64,
    /// Mean DMA read latency per inference (ns).
    pub mean_read_ns: f64,
    /// Total batch time (µs).
    pub total_us: f64,
    /// Throughput (inferences/second).
    pub throughput_hz: f64,
}

/// Load int8 readout weights onto NPU via DMA.
///
/// This is the weight-swap operation for multi-crop classifier hot-swap.
///
/// # Errors
///
/// Returns error if the DMA write fails.
pub fn load_readout_weights(handle: &mut NpuHandle, weights_i8: &[i8]) -> Result<u64, Error> {
    let bytes: Vec<u8> = weights_i8.iter().map(|&x| x.cast_unsigned()).collect();
    let t = std::time::Instant::now();
    handle.write_raw(&bytes)?;
    Ok(u64::try_from(t.elapsed().as_nanos()).unwrap_or(u64::MAX))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn crop_stress_input_quantizes() {
        let input = CropStressInput {
            depletion: 0.7,
            et_ratio: 0.5,
            theta: 0.25,
            ks: 0.3,
        };
        let q = input.to_i8();
        assert_eq!(q.len(), 4);
        assert!(q[0] > 80, "high depletion should quantize high");
        assert!(q[3] < 50, "low Ks should quantize low");
    }

    #[test]
    fn irrigation_input_quantizes() {
        let input = IrrigationInput {
            forecast_et0: 5.0,
            theta: 0.3,
            taw: 150.0,
            stage: 0.5,
            ks: 0.8,
            rain_prob: 0.1,
        };
        let q = input.to_i8();
        assert_eq!(q.len(), 6);
    }

    #[test]
    fn sensor_anomaly_input_quantizes() {
        let input = SensorAnomalyInput {
            reading: 3.5,
            mean: 4.0,
            sigma: 0.5,
        };
        let q = input.to_i8(0.0, 10.0);
        assert_eq!(q.len(), 3);
    }

    #[test]
    fn crop_stress_boundary_values() {
        let low = CropStressInput {
            depletion: 0.0,
            et_ratio: 0.0,
            theta: 0.0,
            ks: 0.0,
        };
        let high = CropStressInput {
            depletion: 1.0,
            et_ratio: 1.5,
            theta: 0.6,
            ks: 1.0,
        };
        let q_low = low.to_i8();
        let q_high = high.to_i8();
        assert!(
            q_low.iter().all(|&v| v == 0),
            "all-zero input → all-zero quant"
        );
        assert!(
            q_high.iter().all(|&v| v == 127),
            "all-max input → all-127 quant"
        );
    }

    #[test]
    fn irrigation_boundary_values() {
        let low = IrrigationInput {
            forecast_et0: 0.0,
            theta: 0.0,
            taw: 0.0,
            stage: 0.0,
            ks: 0.0,
            rain_prob: 0.0,
        };
        let high = IrrigationInput {
            forecast_et0: 15.0,
            theta: 0.6,
            taw: 300.0,
            stage: 1.0,
            ks: 1.0,
            rain_prob: 1.0,
        };
        assert!(low.to_i8().iter().all(|&v| v == 0));
        assert!(high.to_i8().iter().all(|&v| v == 127));
    }
}
