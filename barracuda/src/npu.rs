// SPDX-License-Identifier: AGPL-3.0-or-later
//! NPU integration via `ToadStool` `akida-driver`
//!
//! Wraps `akida_driver` for airSpring's edge inference pipeline:
//! runtime device discovery, int8 quantization, and crop/irrigation
//! classifiers on `BrainChip` AKD1000.
//!
//! # Architecture
//!
//! - **Zero Mocks**: Real hardware only; tests skip when no device present
//! - **Capability-Based**: Device features discovered at runtime
//! - **Primal Self-Knowledge**: airSpring discovers NPU, never hardcodes
//!
//! # Eco-Domain NPU Workloads
//!
//! 1. Crop stress classifier (binary int8, 4 features → 2 classes)
//! 2. Irrigation decision (3-class int8, 6 features → 3 classes)
//! 3. Sensor anomaly detector (binary int8, 3 features → 2 classes)

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss
)]

pub use akida_driver::{
    AkidaDevice, BackendSelection, BackendType, Capabilities, ChipVersion, DeviceManager,
    InferenceConfig, InferenceExecutor, InferenceResult, LoadConfig, ModelLoader, ModelProgram,
    NpuBackend, NpuConfig,
};

use crate::error::AirSpringError;

type Error = AirSpringError;

/// NPU handle wrapping an opened `AkidaDevice` with its capabilities.
///
/// Created via [`discover_npu`] — never constructed manually.
pub struct NpuHandle {
    device: AkidaDevice,
    caps: Capabilities,
}

impl NpuHandle {
    /// Device capabilities (discovered at runtime).
    #[must_use]
    pub const fn capabilities(&self) -> &Capabilities {
        &self.caps
    }

    /// Chip version (`AKD1000`, `AKD1500`, etc.).
    #[must_use]
    pub const fn chip_version(&self) -> ChipVersion {
        self.caps.chip_version
    }

    /// Number of neural processing units.
    #[must_use]
    pub const fn npu_count(&self) -> u32 {
        self.caps.npu_count
    }

    /// On-chip SRAM in megabytes.
    #[must_use]
    pub const fn memory_mb(&self) -> u32 {
        self.caps.memory_mb
    }

    /// `PCIe` bandwidth in GB/s.
    #[must_use]
    pub const fn bandwidth_gbps(&self) -> f32 {
        self.caps.pcie.bandwidth_gbps
    }

    /// Load a model program onto the NPU.
    ///
    /// # Errors
    ///
    /// Returns error if the model is too large for device SRAM or DMA fails.
    pub fn load_model(&mut self, program: &ModelProgram) -> Result<(), Error> {
        let config = LoadConfig::from_capabilities(&self.caps, self.device.index());
        let loader = ModelLoader::new(config);
        loader
            .load(program, &mut self.device)
            .map_err(|e| Error::Npu(format!("model load failed: {e}")))?;
        Ok(())
    }

    /// Run int8 inference with the currently loaded model.
    ///
    /// # Errors
    ///
    /// Returns error if input size is wrong or inference times out.
    pub fn infer(
        &mut self,
        config: &InferenceConfig,
        input: &[u8],
    ) -> Result<InferenceResult, Error> {
        let executor = InferenceExecutor::new(config.clone());
        executor
            .infer(input, &mut self.device)
            .map_err(|e| Error::Npu(format!("inference failed: {e}")))
    }

    /// Write raw bytes to device SRAM (DMA).
    ///
    /// # Errors
    ///
    /// Returns error if the DMA transfer fails.
    pub fn write_raw(&mut self, data: &[u8]) -> Result<usize, Error> {
        self.device
            .write(data)
            .map_err(|e| Error::Npu(format!("write failed: {e}")))
    }

    /// Read raw bytes from device SRAM (DMA).
    ///
    /// # Errors
    ///
    /// Returns error if the DMA transfer fails.
    pub fn read_raw(&mut self, buffer: &mut [u8]) -> Result<usize, Error> {
        self.device
            .read(buffer)
            .map_err(|e| Error::Npu(format!("read failed: {e}")))
    }
}

/// Discover and open the first available Akida NPU.
///
/// # Errors
///
/// Returns error if no Akida hardware is detected or the device cannot be opened.
pub fn discover_npu() -> Result<NpuHandle, Error> {
    let manager =
        DeviceManager::discover().map_err(|e| Error::Npu(format!("discovery failed: {e}")))?;

    let info = manager
        .devices()
        .first()
        .ok_or_else(|| Error::Npu("no devices found after discovery".into()))?;

    let caps = info.capabilities().clone();

    let device = AkidaDevice::open(info).map_err(|e| Error::Npu(format!("open failed: {e}")))?;

    Ok(NpuHandle { device, caps })
}

/// Check whether an Akida NPU is present without opening it.
#[must_use]
pub fn npu_available() -> bool {
    DeviceManager::discover()
        .map(|m| m.device_count() > 0)
        .unwrap_or(false)
}

/// Quantize `f64` value to int8 for NPU inference.
///
/// Maps `[lo, hi]` → `[0, 127]` with clamping.
#[must_use]
pub fn quantize_i8(val: f64, lo: f64, hi: f64) -> i8 {
    let normalized = ((val - lo) / (hi - lo)).clamp(0.0, 1.0);
    (normalized * 127.0) as i8
}

/// Dequantize int8 back to `f64`.
///
/// Maps `[0, 127]` → `[lo, hi]`.
#[must_use]
pub fn dequantize_i8(val: i8, lo: f64, hi: f64) -> f64 {
    let normalized = f64::from(val) / 127.0;
    normalized.mul_add(hi - lo, lo)
}

/// Summary of discovered NPU hardware.
pub struct NpuSummary {
    /// Chip identifier.
    pub chip: String,
    /// `PCIe` address.
    pub pcie_address: String,
    /// Number of NPUs.
    pub npu_count: u32,
    /// SRAM in MB.
    pub memory_mb: u32,
    /// `PCIe` bandwidth in GB/s.
    pub bandwidth_gbps: f32,
}

/// Discover NPU and return a summary without opening the device.
///
/// # Errors
///
/// Returns error if no Akida hardware is detected.
pub fn npu_summary() -> Result<NpuSummary, Error> {
    let manager =
        DeviceManager::discover().map_err(|e| Error::Npu(format!("discovery failed: {e}")))?;

    let info = manager
        .devices()
        .first()
        .ok_or_else(|| Error::Npu("no devices found".into()))?;

    let caps = info.capabilities();

    Ok(NpuSummary {
        chip: format!("{:?}", caps.chip_version),
        pcie_address: info.pcie_address().to_string(),
        npu_count: caps.npu_count,
        memory_mb: caps.memory_mb,
        bandwidth_gbps: caps.pcie.bandwidth_gbps,
    })
}

// ═══════════════════════════════════════════════════════════════════
// Eco-Domain Inference Helpers
// ═══════════════════════════════════════════════════════════════════

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
    let write_ns = t.elapsed().as_nanos() as u64;

    let mut out_buf = vec![0u8; n_outputs];
    let t = std::time::Instant::now();
    handle.read_raw(&mut out_buf)?;
    let read_ns = t.elapsed().as_nanos() as u64;

    let raw_i8: Vec<i8> = out_buf.iter().map(|&b| b as i8).collect();
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

// ═══════════════════════════════════════════════════════════════════
// High-Cadence Streaming Pipeline
// ═══════════════════════════════════════════════════════════════════

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

    let is_anomaly = if sigma > 1e-10 && stats.count() > 10 {
        ((reading - mean) / sigma).abs() > anomaly_z
    } else {
        false
    };

    let depletion = (fc - reading) / (fc - wp);
    let class = if is_anomaly {
        2
    } else if depletion > 0.55 {
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
    Ok(t.elapsed().as_nanos() as u64)
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
    fn quantize_roundtrip() {
        let val = 7.5_f64;
        let q = quantize_i8(val, 5.0, 10.0);
        let deq = dequantize_i8(q, 5.0, 10.0);
        assert!((val - deq).abs() < 0.1, "roundtrip: {val} -> {q} -> {deq}");
    }

    #[test]
    fn quantize_clamp() {
        assert_eq!(quantize_i8(-1.0, 0.0, 10.0), 0);
        assert_eq!(quantize_i8(20.0, 0.0, 10.0), 127);
    }

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
    fn npu_available_check() {
        let avail = npu_available();
        println!("NPU available: {avail}");
    }

    #[test]
    fn discover_npu_check() {
        match discover_npu() {
            Ok(h) => {
                println!(
                    "NPU: {:?}, {} NPUs, {} MB SRAM, {:.1} GB/s",
                    h.chip_version(),
                    h.npu_count(),
                    h.memory_mb(),
                    h.bandwidth_gbps()
                );
            }
            Err(e) => {
                println!("No NPU: {e} (expected on CI)");
            }
        }
    }
}
