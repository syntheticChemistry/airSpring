// SPDX-License-Identifier: AGPL-3.0-or-later
//! NPU integration via `BarraCuda` `akida-driver`
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
//! # Modules
//!
//! - [`inference`] — Eco-domain batch classifiers (crop stress, irrigation, anomaly)
//! - [`streaming`] — High-cadence real-time classification with rolling statistics

pub mod inference;
pub mod streaming;

pub use inference::{
    CropStressInput, IrrigationInput, NpuBatchResult, NpuInferResult, SensorAnomalyInput,
    load_readout_weights, npu_batch_infer, npu_infer_i8,
};
pub use streaming::{
    MultiSensorInput, RollingStats, StreamClassification, StreamSessionResult, classify_reading,
};

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

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
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

    #[test]
    fn quantize_boundary_lo_equals_hi() {
        assert_eq!(quantize_i8(5.0, 5.0, 5.0), 0);
    }

    #[test]
    fn quantize_exact_midpoint() {
        let q = quantize_i8(5.0, 0.0, 10.0);
        assert_eq!(q, 63, "midpoint of [0,10] should map to 127/2 = 63");
    }

    #[test]
    fn quantize_exact_endpoints() {
        assert_eq!(quantize_i8(0.0, 0.0, 10.0), 0);
        assert_eq!(quantize_i8(10.0, 0.0, 10.0), 127);
    }

    #[test]
    fn dequantize_exact_endpoints() {
        let lo = dequantize_i8(0, 0.0, 10.0);
        let hi = dequantize_i8(127, 0.0, 10.0);
        assert!((lo - 0.0).abs() < f64::EPSILON, "deq(0) = {lo}");
        assert!((hi - 10.0).abs() < f64::EPSILON, "deq(127) = {hi}");
    }

    #[test]
    fn quantize_roundtrip_precision() {
        for domain in &[(0.0, 1.0), (0.0, 0.6), (-10.0, 50.0), (0.0, 300.0)] {
            let (lo, hi) = *domain;
            let step = (hi - lo) / 127.0;
            for i in 0..=127 {
                let original = lo + step * f64::from(i);
                let q = quantize_i8(original, lo, hi);
                let deq = dequantize_i8(q, lo, hi);
                assert!(
                    (original - deq).abs() <= step + 1e-10,
                    "roundtrip error at {original}: deq={deq}, step={step}"
                );
            }
        }
    }

    #[test]
    fn quantize_negative_range() {
        let q = quantize_i8(-5.0, -10.0, 50.0);
        let deq = dequantize_i8(q, -10.0, 50.0);
        assert!((-5.0 - deq).abs() < 1.0, "negative range: deq={deq}");
    }
}
