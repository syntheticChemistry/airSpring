// SPDX-License-Identifier: AGPL-3.0-or-later

//! Substrate abstraction — runtime-discovered compute devices.
//!
//! A substrate is a compute device found on this machine. GPUs come from
//! wgpu adapter enumeration. NPUs come from device node probing. CPU from procfs.
//!
//! Capabilities drive dispatch — code asks "can you do f64?" not "are you an RTX 4070?".

use std::fmt;

/// Well-known PCI vendor:device IDs for substrate identification.
pub mod pci {
    /// NVIDIA Titan V (GV100).
    pub const NVIDIA_TITAN_V: &str = "10de:1db6";
    /// NVIDIA `GeForce` RTX 4070.
    pub const NVIDIA_RTX_4070: &str = "10de:2786";
    /// `BrainChip` Akida AKD1000 Neural Processor.
    pub const BRAINCHIP_AKD1000: &str = "1e7c:1000";
}

/// A compute substrate discovered at runtime.
#[derive(Debug, Clone)]
pub struct Substrate {
    /// What kind of device this is.
    pub kind: SubstrateKind,
    /// How we found it and what to call it.
    pub identity: Identity,
    /// Measured hardware properties.
    pub properties: Properties,
    /// What this device can do.
    pub capabilities: Vec<Capability>,
}

/// How we found this device and what to call it.
#[derive(Debug, Clone)]
pub struct Identity {
    /// Human-readable device name.
    pub name: String,
    /// GPU driver string from wgpu.
    pub driver: Option<String>,
    /// wgpu backend (Vulkan, Metal, etc.).
    pub backend: Option<String>,
    /// wgpu adapter index for GPU selection.
    pub adapter_index: Option<usize>,
    /// Device node (e.g. `/dev/akida0`).
    pub device_node: Option<String>,
    /// PCI `vendor:device` if available.
    pub pci_id: Option<String>,
}

/// Measured properties of a substrate.
#[derive(Debug, Clone, Default)]
pub struct Properties {
    /// Total memory in bytes.
    pub memory_bytes: Option<u64>,
    /// Physical core count (CPU).
    pub core_count: Option<u32>,
    /// Logical thread count (CPU).
    pub thread_count: Option<u32>,
    /// Cache size in KB (CPU).
    pub cache_kb: Option<u32>,
    /// Supports IEEE 754 f64 in shaders (GPU).
    pub has_f64: bool,
    /// Supports timestamp queries (GPU).
    pub has_timestamps: bool,
}

/// The kind of compute device.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SubstrateKind {
    /// GPU via wgpu/Vulkan.
    Gpu,
    /// Neural Processing Unit (`BrainChip` AKD1000).
    Npu,
    /// Host CPU.
    Cpu,
    /// biomeOS Neural API (ecosystem-wide compute via `capability.call`).
    Neural,
}

/// A capability discovered at runtime on a substrate.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Capability {
    /// IEEE 754 f64 compute.
    F64Compute,
    /// f32 compute.
    F32Compute,
    /// Integer quantized inference at a given bit width.
    QuantizedInference {
        /// Quantization bit width (4 or 8).
        bits: u8,
    },
    /// Batch inference with amortized dispatch.
    BatchInference {
        /// Maximum batch size supported.
        max_batch: usize,
    },
    /// Weight mutation without full reprogramming.
    WeightMutation,
    /// Scalar reduction (GPU fused map-reduce).
    ScalarReduce,
    /// WGSL shader dispatch via wgpu.
    ShaderDispatch,
    /// General CPU compute.
    CpuCompute,
    /// AVX2/SSE SIMD on CPU.
    SimdVector,
    /// GPU timestamp query support.
    TimestampQuery,
    /// biomeOS capability.call routing (JSON-RPC over Unix socket).
    NeuralApiRoute,
}

impl fmt::Display for SubstrateKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Gpu => write!(f, "GPU"),
            Self::Npu => write!(f, "NPU"),
            Self::Cpu => write!(f, "CPU"),
            Self::Neural => write!(f, "Neural"),
        }
    }
}

impl fmt::Display for Substrate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} [{}]", self.identity.name, self.kind)?;
        if let Some(ref driver) = self.identity.driver {
            write!(f, " {driver}")?;
        }
        if let Some(mem) = self.properties.memory_bytes {
            let mb = mem / (1024 * 1024);
            write!(f, " {mb}MB")?;
        }
        Ok(())
    }
}

impl Substrate {
    /// Check if this substrate has a specific capability.
    #[must_use]
    pub fn has(&self, cap: &Capability) -> bool {
        self.capabilities.contains(cap)
    }

    /// Return capabilities as a summary string.
    #[must_use]
    pub fn capability_summary(&self) -> String {
        let labels: Vec<&str> = self.capabilities.iter().map(Capability::label).collect();
        labels.join(", ")
    }
}

impl Capability {
    /// Human-readable label.
    #[must_use]
    pub const fn label(&self) -> &'static str {
        match self {
            Self::F64Compute => "f64",
            Self::F32Compute => "f32",
            Self::QuantizedInference { .. } => "quant",
            Self::BatchInference { .. } => "batch",
            Self::WeightMutation => "weight-mut",
            Self::ScalarReduce => "reduce",
            Self::ShaderDispatch => "shader",
            Self::CpuCompute => "cpu",
            Self::SimdVector => "simd",
            Self::TimestampQuery => "timestamps",
            Self::NeuralApiRoute => "neural-api",
        }
    }
}

impl Identity {
    /// Minimal identity with just a name.
    #[must_use]
    pub fn named(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            driver: None,
            backend: None,
            adapter_index: None,
            device_node: None,
            pci_id: None,
        }
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests {
    use super::*;

    fn test_gpu() -> Substrate {
        Substrate {
            kind: SubstrateKind::Gpu,
            identity: Identity {
                name: String::from("Test GPU"),
                adapter_index: Some(0),
                ..Identity::named("Test GPU")
            },
            properties: Properties {
                has_f64: true,
                ..Properties::default()
            },
            capabilities: vec![Capability::F64Compute, Capability::ShaderDispatch],
        }
    }

    #[test]
    fn has_capability() {
        let gpu = test_gpu();
        assert!(gpu.has(&Capability::F64Compute));
        assert!(gpu.has(&Capability::ShaderDispatch));
        assert!(!gpu.has(&Capability::QuantizedInference { bits: 8 }));
    }

    #[test]
    fn display_shows_kind_and_name() {
        let gpu = test_gpu();
        let s = format!("{gpu}");
        assert!(s.contains("Test GPU"));
        assert!(s.contains("GPU"));
    }

    #[test]
    fn capability_labels() {
        assert_eq!(Capability::F64Compute.label(), "f64");
        assert_eq!(Capability::ShaderDispatch.label(), "shader");
        assert_eq!(Capability::QuantizedInference { bits: 8 }.label(), "quant");
    }

    #[test]
    fn substrate_kind_display() {
        assert_eq!(format!("{}", SubstrateKind::Gpu), "GPU");
        assert_eq!(format!("{}", SubstrateKind::Npu), "NPU");
        assert_eq!(format!("{}", SubstrateKind::Cpu), "CPU");
    }

    #[test]
    fn display_with_memory() {
        let s = Substrate {
            kind: SubstrateKind::Gpu,
            identity: Identity::named("RTX 4070"),
            properties: Properties {
                memory_bytes: Some(12 * 1024 * 1024 * 1024),
                ..Properties::default()
            },
            capabilities: vec![],
        };
        let display = format!("{s}");
        assert!(display.contains("12288MB"));
    }
}
