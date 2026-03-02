// SPDX-License-Identifier: AGPL-3.0-or-later

//! Mixed-hardware pipeline — chain workloads across substrates with
//! optimised transfer paths.
//!
//! # `PCIe` Bypass Architecture
//!
//! ```text
//! ┌─────┐  PCIe P2P  ┌─────┐
//! │ NPU ├────────────►│ GPU │   ← bypass CPU roundtrip
//! └──┬──┘             └──┬──┘
//!    │ DMA               │ DMA
//!    ▼                   ▼
//! ┌──────────────────────────┐
//! │        CPU memory        │   ← fallback only
//! └──────────────────────────┘
//! ```
//!
//! When NPU and GPU are on the same `PCIe` root complex, data can flow
//! directly via peer-to-peer DMA without touching CPU memory. This
//! eliminates the CPU roundtrip for pipelines like:
//!
//! ```text
//! crop_stress (NPU int8) → et0_batch (GPU f64) → yield_response (GPU f64)
//! ```
//!
//! # Transfer Path Selection
//!
//! Transfer paths are selected based on substrate topology:
//! - Same device: [`TransferPath::None`] (zero-copy)
//! - Same `PCIe` root complex: [`TransferPath::PciePeerToPeer`]
//! - Different bus: [`TransferPath::CpuMemcpy`] (fallback)
//! - Remote: [`TransferPath::NeuralApi`] (biomeOS)

use crate::dispatch::{self, Workload};
use crate::substrate::{Substrate, SubstrateKind};

/// How data moves between pipeline stages.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransferPath {
    /// No transfer needed (same device or zero-copy).
    None,
    /// Direct `PCIe` peer-to-peer DMA (bypasses CPU memory).
    PciePeerToPeer,
    /// CPU memory copy (host staging buffer).
    CpuMemcpy,
    /// `biomeOS` Neural API (JSON-RPC over Unix socket).
    NeuralApi,
}

/// Relative latency ranks for transfer-path cost estimation.
///
/// Values are ordinal (not calibrated measurements). The ratio between
/// tiers encodes the typical order-of-magnitude penalty:
///   zero-copy < P2P DMA < host staging < JSON-RPC.
const LATENCY_RANK_ZERO_COPY: u32 = 0;
const LATENCY_RANK_PCIE_P2P: u32 = 1;
const LATENCY_RANK_CPU_MEMCPY: u32 = 10;
const LATENCY_RANK_NEURAL_API: u32 = 100;

impl TransferPath {
    /// Estimated relative latency (lower is better).
    #[must_use]
    pub const fn latency_rank(&self) -> u32 {
        match self {
            Self::None => LATENCY_RANK_ZERO_COPY,
            Self::PciePeerToPeer => LATENCY_RANK_PCIE_P2P,
            Self::CpuMemcpy => LATENCY_RANK_CPU_MEMCPY,
            Self::NeuralApi => LATENCY_RANK_NEURAL_API,
        }
    }

    /// Whether this path bypasses CPU memory.
    #[must_use]
    pub const fn bypasses_cpu(&self) -> bool {
        matches!(self, Self::None | Self::PciePeerToPeer)
    }
}

/// A single stage in a mixed-hardware pipeline.
#[derive(Debug)]
pub struct PipelineStage<'a> {
    /// The workload to execute.
    pub workload: &'a Workload,
    /// Which substrate was selected for execution.
    pub substrate: &'a Substrate,
    /// How data arrives from the previous stage.
    pub transfer_in: TransferPath,
}

/// A routed mixed-hardware pipeline.
#[derive(Debug)]
pub struct RoutedPipeline<'a> {
    /// Ordered sequence of stages.
    pub stages: Vec<PipelineStage<'a>>,
    /// Total estimated transfer cost (sum of latency ranks).
    pub total_transfer_cost: u32,
    /// Number of CPU roundtrips in this pipeline.
    pub cpu_roundtrips: usize,
    /// Number of `PCIe` P2P transfers (bypassing CPU).
    pub pcie_bypasses: usize,
}

impl RoutedPipeline<'_> {
    /// Whether the entire pipeline avoids CPU memory for inter-stage transfers.
    #[must_use]
    pub const fn fully_bypasses_cpu(&self) -> bool {
        self.cpu_roundtrips == 0
    }
}

/// Determine the transfer path between two substrates based on topology.
#[must_use]
pub fn transfer_path(from: &Substrate, to: &Substrate) -> TransferPath {
    if std::ptr::eq(from, to) {
        return TransferPath::None;
    }

    #[allow(clippy::match_same_arms)]
    match (from.kind, to.kind) {
        (a, b) if a == b => TransferPath::None,
        (SubstrateKind::Neural, _) | (_, SubstrateKind::Neural) => TransferPath::NeuralApi,
        (SubstrateKind::Cpu, _) | (_, SubstrateKind::Cpu) => TransferPath::CpuMemcpy,
        (SubstrateKind::Gpu, SubstrateKind::Npu) | (SubstrateKind::Npu, SubstrateKind::Gpu) => {
            if pcie_p2p_capable(from, to) {
                TransferPath::PciePeerToPeer
            } else {
                TransferPath::CpuMemcpy
            }
        }
        _ => TransferPath::CpuMemcpy,
    }
}

/// Check if two substrates can do `PCIe` peer-to-peer transfers.
///
/// Requires both devices to be on the same `PCIe` root complex and have
/// compatible IOMMU/ACS settings. Currently uses PCI ID heuristics;
/// a production implementation would use `ioctl` or Vulkan external memory.
const fn pcie_p2p_capable(a: &Substrate, b: &Substrate) -> bool {
    a.identity.pci_id.is_some() && b.identity.pci_id.is_some()
}

/// Route a sequence of workloads through available substrates,
/// optimising for minimal transfer cost.
///
/// Each workload is routed independently via [`dispatch::route`],
/// then transfer paths are computed between consecutive stages.
#[must_use]
pub fn route_pipeline<'a>(
    workloads: &'a [Workload],
    substrates: &'a [Substrate],
) -> Option<RoutedPipeline<'a>> {
    if workloads.is_empty() {
        return Some(RoutedPipeline {
            stages: Vec::new(),
            total_transfer_cost: 0,
            cpu_roundtrips: 0,
            pcie_bypasses: 0,
        });
    }

    let mut stages = Vec::with_capacity(workloads.len());
    let mut total_cost = 0_u32;
    let mut cpu_roundtrips = 0_usize;
    let mut pcie_bypasses = 0_usize;
    let mut prev_substrate: Option<&Substrate> = None;

    for wl in workloads {
        let decision = dispatch::route(wl, substrates)?;
        let transfer_in = prev_substrate.map_or(TransferPath::None, |prev| {
            transfer_path(prev, decision.substrate)
        });

        total_cost += transfer_in.latency_rank();
        if transfer_in == TransferPath::CpuMemcpy {
            cpu_roundtrips += 1;
        }
        if transfer_in == TransferPath::PciePeerToPeer {
            pcie_bypasses += 1;
        }

        prev_substrate = Some(decision.substrate);
        stages.push(PipelineStage {
            workload: wl,
            substrate: decision.substrate,
            transfer_in,
        });
    }

    Some(RoutedPipeline {
        stages,
        total_transfer_cost: total_cost,
        cpu_roundtrips,
        pcie_bypasses,
    })
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::substrate::{Capability, Identity, Properties};

    fn gpu(name: &str) -> Substrate {
        Substrate {
            kind: SubstrateKind::Gpu,
            identity: Identity {
                pci_id: Some("10de:1db6".to_string()),
                ..Identity::named(name)
            },
            properties: Properties {
                has_f64: true,
                ..Properties::default()
            },
            capabilities: vec![
                Capability::F64Compute,
                Capability::ShaderDispatch,
                Capability::ScalarReduce,
            ],
        }
    }

    fn npu() -> Substrate {
        Substrate {
            kind: SubstrateKind::Npu,
            identity: Identity {
                pci_id: Some("1e7c:1000".to_string()),
                device_node: Some("/dev/akida0".to_string()),
                ..Identity::named("AKD1000")
            },
            properties: Properties::default(),
            capabilities: vec![
                Capability::QuantizedInference { bits: 8 },
                Capability::BatchInference { max_batch: 8 },
                Capability::WeightMutation,
            ],
        }
    }

    fn cpu() -> Substrate {
        Substrate {
            kind: SubstrateKind::Cpu,
            identity: Identity::named("i9-12900K"),
            properties: Properties::default(),
            capabilities: vec![
                Capability::F64Compute,
                Capability::F32Compute,
                Capability::CpuCompute,
                Capability::SimdVector,
            ],
        }
    }

    #[test]
    fn npu_to_gpu_pcie_bypass() {
        let n = npu();
        let g = gpu("TITAN V");
        let path = transfer_path(&n, &g);
        assert_eq!(path, TransferPath::PciePeerToPeer);
        assert!(path.bypasses_cpu());
    }

    #[test]
    fn gpu_to_cpu_memcpy() {
        let g = gpu("RTX 4070");
        let c = cpu();
        let path = transfer_path(&g, &c);
        assert_eq!(path, TransferPath::CpuMemcpy);
        assert!(!path.bypasses_cpu());
    }

    #[test]
    fn same_kind_is_none() {
        let g1 = gpu("GPU A");
        let g2 = gpu("GPU B");
        assert_eq!(transfer_path(&g1, &g2), TransferPath::None);
    }

    #[test]
    fn pipeline_npu_then_gpu() {
        let substrates = [npu(), gpu("TITAN V"), cpu()];
        let workloads = [
            Workload::new(
                "crop_stress",
                vec![Capability::QuantizedInference { bits: 8 }],
            )
            .prefer(SubstrateKind::Npu),
            Workload::new(
                "et0_batch",
                vec![Capability::F64Compute, Capability::ShaderDispatch],
            ),
        ];

        let pipeline = route_pipeline(&workloads, &substrates).expect("should route");
        assert_eq!(pipeline.stages.len(), 2);
        assert_eq!(pipeline.stages[0].substrate.kind, SubstrateKind::Npu);
        assert_eq!(pipeline.stages[1].substrate.kind, SubstrateKind::Gpu);
        assert_eq!(pipeline.stages[1].transfer_in, TransferPath::PciePeerToPeer);
        assert_eq!(pipeline.cpu_roundtrips, 0);
        assert_eq!(pipeline.pcie_bypasses, 1);
        assert!(pipeline.fully_bypasses_cpu());
    }

    #[test]
    fn pipeline_gpu_then_cpu_has_roundtrip() {
        let substrates = [gpu("TITAN V"), cpu()];
        let workloads = [
            Workload::new(
                "et0_batch",
                vec![Capability::F64Compute, Capability::ShaderDispatch],
            ),
            Workload::new("validation", vec![Capability::F64Compute]).prefer(SubstrateKind::Cpu),
        ];

        let pipeline = route_pipeline(&workloads, &substrates).expect("should route");
        assert_eq!(pipeline.cpu_roundtrips, 1);
        assert!(!pipeline.fully_bypasses_cpu());
    }

    #[test]
    fn empty_pipeline() {
        let substrates = [cpu()];
        let pipeline = route_pipeline(&[], &substrates).expect("should route");
        assert!(pipeline.stages.is_empty());
        assert_eq!(pipeline.total_transfer_cost, 0);
    }

    #[test]
    fn latency_ranking() {
        assert!(TransferPath::None.latency_rank() < TransferPath::PciePeerToPeer.latency_rank());
        assert!(
            TransferPath::PciePeerToPeer.latency_rank() < TransferPath::CpuMemcpy.latency_rank()
        );
        assert!(TransferPath::CpuMemcpy.latency_rank() < TransferPath::NeuralApi.latency_rank());
    }

    #[test]
    fn three_stage_mixed_pipeline() {
        let substrates = [npu(), gpu("TITAN V"), cpu()];
        let workloads = [
            Workload::new(
                "crop_stress",
                vec![Capability::QuantizedInference { bits: 8 }],
            )
            .prefer(SubstrateKind::Npu),
            Workload::new(
                "et0_batch",
                vec![Capability::F64Compute, Capability::ShaderDispatch],
            ),
            Workload::new(
                "yield_response",
                vec![Capability::F64Compute, Capability::ShaderDispatch],
            ),
        ];

        let pipeline = route_pipeline(&workloads, &substrates).expect("should route");
        assert_eq!(pipeline.stages.len(), 3);
        assert_eq!(pipeline.stages[0].transfer_in, TransferPath::None);
        assert_eq!(pipeline.stages[1].transfer_in, TransferPath::PciePeerToPeer);
        assert_eq!(pipeline.stages[2].transfer_in, TransferPath::None);
        assert!(pipeline.fully_bypasses_cpu());
    }
}
