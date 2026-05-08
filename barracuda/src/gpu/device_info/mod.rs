// SPDX-License-Identifier: AGPL-3.0-or-later
//! Device information and precision strategy reporting.
//!
//! Wraps `barracuda::device` to provide precision-aware device creation
//! and capability reporting for airSpring GPU modules.
//!
//! # Cross-Spring Shader Provenance
//!
//! The precision architecture was evolved across all Springs:
//! - **hotSpring**: `df64_core.wgsl`, `df64_transcendentals.wgsl`, `math_f64.wgsl` — FMA-optimized
//!   double-float arithmetic for lattice QCD and nuclear EOS.
//! - **wetSpring**: `shannon_f64.wgsl`, `kriging_f64.wgsl` — f64 diversity and spatial interpolation.
//! - **neuralSpring**: `NelderMeadGpu`, `ValidationHarness` — optimization and quality infrastructure.
//! - **airSpring**: Richards PDE, regression, hydrology — domain contributions absorbed S40+S66.
//! - **groundSpring**: MC propagation, `batched_multinomial` — uncertainty quantification.
//!
//! `Fp64Strategy` was introduced in S58, evolving from hotSpring's need for
//! native f64 in lattice QCD (where `Df64` accumulated unacceptable phase errors
//! in chained SU(3) matrix multiplications). airSpring benefits: the Titan V
//! runs our ET₀ shaders in native f64, while the RTX 4070 can use `Df64` with
//! ~48-bit mantissa precision — still adequate for FAO-56 (which only needs ~6 digits).
//!
//! **Note**: Session references (S40, S54, S66, etc.) in shader provenance below
//! are historical `BarraCuda` sessions. Since S89, all math primitives live in the
//! standalone `barraCuda` primal (`ecoPrimals/barraCuda`).

pub mod shader_provenance;

pub use shader_provenance::{
    PROVENANCE, ShaderProvenance, upstream_airspring_provenance, upstream_cross_spring_matrix,
    upstream_evolution_report,
};

use std::sync::{Arc, OnceLock};

use barracuda::device::capabilities::DeviceCapabilities;
use barracuda::device::driver_profile::PrecisionRoutingAdvice;
use barracuda::device::probe::F64BuiltinCapabilities;
use barracuda::device::{Fp64Strategy, WgpuDevice};

/// Process-wide cached GPU device.
///
/// `wgpu::Instance` creation is not safe to call concurrently from multiple
/// threads (observed as SIGSEGV on NVK/Mesa). `OnceLock` ensures exactly one
/// probe per process lifetime — the same fix applied in toadStool S158.
static GPU_DEVICE: OnceLock<Option<Arc<WgpuDevice>>> = OnceLock::new();

/// Precision report for a GPU device.
#[derive(Debug, Clone)]
pub struct DevicePrecisionReport {
    /// Device adapter name.
    pub adapter_name: String,
    /// f64 throughput strategy.
    pub fp64_strategy: Fp64Strategy,
    /// f64 builtin capabilities.
    pub builtins: F64BuiltinCapabilities,
    /// Whether the device has native f64 shader support.
    pub has_f64_shaders: bool,
    /// Whether the device supports SPIR-V passthrough.
    pub has_spirv_passthrough: bool,
    /// Minimum subgroup (warp/wavefront) size. Zero if not reported.
    pub subgroup_min_size: u32,
    /// Maximum subgroup (warp/wavefront) size. Zero if not reported.
    pub subgroup_max_size: u32,
    /// Precision routing advice (toadStool S128) — integrates f64 shared-memory
    /// reliability for workgroup-based reductions.
    pub precision_routing: PrecisionRoutingAdvice,
}

impl std::fmt::Display for DevicePrecisionReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "GPU: {}", self.adapter_name)?;
        writeln!(f, "  Fp64Strategy:      {:?}", self.fp64_strategy)?;
        writeln!(f, "  PrecisionRouting:  {:?}", self.precision_routing)?;
        writeln!(f, "  f64 shaders:       {}", self.has_f64_shaders)?;
        writeln!(f, "  SPIR-V pass:       {}", self.has_spirv_passthrough)?;
        writeln!(
            f,
            "  Builtins:          exp={} log={} sin={} cos={} sqrt={} fma={}",
            self.builtins.exp,
            self.builtins.log,
            self.builtins.sin,
            self.builtins.cos,
            self.builtins.sqrt,
            self.builtins.fma
        )?;
        if self.subgroup_min_size > 0 {
            write!(
                f,
                "  Subgroups:         {}-{} lanes",
                self.subgroup_min_size, self.subgroup_max_size
            )
        } else {
            write!(f, "  Subgroups:         not reported")
        }
    }
}

/// Probe a device and produce a precision report.
///
/// Combines `DeviceCapabilities` (for `Fp64Strategy` and precision routing) with
/// `probe_f64_builtins` (for native f64 builtin availability).
#[must_use]
pub fn probe_device(device: &WgpuDevice) -> DevicePrecisionReport {
    let caps = DeviceCapabilities::from_device(device);
    let builtins = barracuda::device::test_pool::tokio_block_on(
        barracuda::device::probe::probe_f64_builtins(device),
    );

    let info = device.adapter_info();
    DevicePrecisionReport {
        adapter_name: info.name.clone(),
        fp64_strategy: caps.fp64_strategy(),
        precision_routing: caps.precision_routing(),
        builtins,
        has_f64_shaders: device.has_f64_shaders(),
        has_spirv_passthrough: device.has_spirv_passthrough(),
        subgroup_min_size: info.subgroup_min_size,
        subgroup_max_size: info.subgroup_max_size,
    }
}

/// Try to obtain the process-wide GPU device.
///
/// The first call probes the GPU via `WgpuDevice::from_env()` (which reads
/// `BARRACUDA_GPU_ADAPTER`), falling back to `new_f64_capable()`. Subsequent
/// calls return the cached result — this prevents SIGSEGV from concurrent
/// `wgpu::Instance` creation in parallel test threads (toadStool S158 pattern).
///
/// Returns `None` if no suitable GPU is available.
#[must_use]
pub fn try_f64_device() -> Option<Arc<WgpuDevice>> {
    GPU_DEVICE
        .get_or_init(|| {
            barracuda::device::test_pool::tokio_block_on(WgpuDevice::from_env())
                .or_else(|_| {
                    barracuda::device::test_pool::tokio_block_on(WgpuDevice::new_f64_capable())
                })
                .ok()
                .map(Arc::new)
        })
        .clone()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_precision_routing_field() {
        let report = DevicePrecisionReport {
            adapter_name: "Test GPU".to_string(),
            fp64_strategy: Fp64Strategy::Native,
            precision_routing: PrecisionRoutingAdvice::F64Native,
            builtins: F64BuiltinCapabilities {
                basic_f64: true,
                exp: true,
                log: true,
                exp2: true,
                log2: true,
                sin: true,
                cos: true,
                sqrt: true,
                fma: true,
                abs_min_max: true,
                composite_transcendental: true,
                exp_log_chain: true,
                shared_mem_f64: true,
                df64_arith: true,
                df64_transcendentals_safe: true,
                df64_fma_two_prod: true,
                df64_workgroup_reduce: true,
            },
            has_f64_shaders: true,
            has_spirv_passthrough: false,
            subgroup_min_size: 32,
            subgroup_max_size: 32,
        };
        assert_eq!(report.precision_routing, PrecisionRoutingAdvice::F64Native);
    }

    #[test]
    fn test_try_f64_device() {
        let _ = try_f64_device();
    }

    #[test]
    fn test_probe_device_if_available() {
        let Some(device) = try_f64_device() else {
            eprintln!("SKIP: No f64-capable GPU");
            return;
        };
        let report = probe_device(&device);
        assert!(!report.adapter_name.is_empty());
        println!("{report}");
    }

    #[test]
    fn test_device_precision_report_display() {
        let report = DevicePrecisionReport {
            adapter_name: "Test GPU".to_string(),
            fp64_strategy: Fp64Strategy::Native,
            precision_routing: PrecisionRoutingAdvice::F64Native,
            builtins: F64BuiltinCapabilities {
                basic_f64: true,
                exp: true,
                log: true,
                exp2: true,
                log2: true,
                sin: true,
                cos: true,
                sqrt: true,
                fma: true,
                abs_min_max: true,
                composite_transcendental: true,
                exp_log_chain: true,
                shared_mem_f64: true,
                df64_arith: true,
                df64_transcendentals_safe: true,
                df64_fma_two_prod: true,
                df64_workgroup_reduce: true,
            },
            has_f64_shaders: true,
            has_spirv_passthrough: false,
            subgroup_min_size: 32,
            subgroup_max_size: 32,
        };
        let s = format!("{report}");
        assert!(s.contains("Test GPU"));
        assert!(s.contains("Native"));
        assert!(s.contains("PrecisionRouting"));
        assert!(s.contains("32-32 lanes"));
    }
}
