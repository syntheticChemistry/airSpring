// SPDX-License-Identifier: AGPL-3.0-or-later

//! NUCLEUS atomics — tower, node, and nest deployment patterns.
//!
//! A NUCLEUS atomic is a minimal, composable deployment unit:
//!
//! | Atomic | Components | Role |
//! |--------|-----------|------|
//! | **Tower** | `BearDog` (crypto/TLS) + `Songbird` (mesh/discovery) + `SkunkBat` (defense) | Trust boundary — crypto + discovery + defense |
//! | **Node** | Tower + `ToadStool` (compute/GPU) | Compute dispatch |
//! | **Nest** | Tower + `NestGate` (storage/provenance) | Data storage and provenance |
//!
//! # Coordination via biomeOS Graphs
//!
//! NUCLEUS atomics are coordinated by biomeOS directed graphs:
//!
//! ```text
//! ┌──────────┐    ┌──────────┐    ┌──────────┐
//! │  Tower   │    │   Node   │    │   Nest   │
//! │ (crypto  │───►│ (compute │───►│ (storage │
//! │  + mesh) │    │  + GPU)  │    │  + prov) │
//! └──────────┘    └──────────┘    └──────────┘
//!       │               │               │
//!       └───────────────┴───────────────┘
//!                       │
//!              biomeOS graph engine
//! ```
//!
//! # Mixed-Hardware Dispatch
//!
//! A Node atomic discovers its local substrates (GPU, NPU, CPU) and
//! advertises capabilities to the biomeOS mesh. Pipelines are routed
//! across the mesh using capability-based dispatch:
//!
//! ```text
//! Node A (GPU: TITAN V)  ──┐
//! Node B (NPU: AKD1000)  ──┼── biomeOS mesh ── Pipeline Router
//! Node C (CPU: i9-12900K) ─┘
//! ```

mod mesh;

pub use mesh::{MeshPipeline, MeshStage, NucleusMesh};

use crate::substrate::Substrate;

/// NUCLEUS atomic deployment mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AtomicKind {
    /// Trust boundary: crypto (`BearDog`) + mesh discovery (`Songbird`) + defense (`SkunkBat`).
    Tower,
    /// Compute layer: Tower + `ToadStool` GPU/CPU dispatch.
    Node,
    /// Storage layer: Tower + `NestGate` provenance tracking.
    Nest,
}

impl AtomicKind {
    /// Capabilities provided by this atomic.
    #[must_use]
    pub const fn capabilities(&self) -> &[&str] {
        match self {
            Self::Tower => &["crypto.tls", "mesh.discovery", "defense.audit"],
            Self::Node => &[
                "crypto.tls",
                "mesh.discovery",
                "defense.audit",
                "compute.dispatch",
            ],
            Self::Nest => &[
                "crypto.tls",
                "mesh.discovery",
                "defense.audit",
                "storage.provenance",
            ],
        }
    }

    /// Human-readable component descriptions (primal names resolved at runtime).
    #[must_use]
    pub const fn component_descriptions(&self) -> &[&str] {
        match self {
            Self::Tower => &[
                "crypto/TLS provider",
                "mesh/discovery provider",
                "defense/audit sentinel",
            ],
            Self::Node => &[
                "crypto/TLS provider",
                "mesh/discovery provider",
                "defense/audit sentinel",
                "compute/GPU dispatch",
            ],
            Self::Nest => &[
                "crypto/TLS provider",
                "mesh/discovery provider",
                "defense/audit sentinel",
                "storage/provenance tracker",
            ],
        }
    }

    /// Whether this atomic includes compute dispatch capability.
    #[must_use]
    pub const fn has_compute(&self) -> bool {
        matches!(self, Self::Node)
    }

    /// Whether this atomic includes storage/provenance capability.
    #[must_use]
    pub const fn has_storage(&self) -> bool {
        matches!(self, Self::Nest)
    }

    /// Whether this atomic includes mesh discovery.
    #[must_use]
    pub const fn has_mesh(&self) -> bool {
        true
    }
}

/// A NUCLEUS atomic instance on a specific machine.
#[derive(Debug, Clone)]
pub struct NucleusAtomic {
    /// What kind of atomic this is.
    pub kind: AtomicKind,
    /// Unique node identifier in the biomeOS mesh.
    pub node_id: String,
    /// Locally discovered substrates (GPU, NPU, CPU).
    pub substrates: Vec<Substrate>,
    /// Whether this atomic is currently reachable on the mesh.
    pub reachable: bool,
}

impl NucleusAtomic {
    /// Create a new atomic with discovered local substrates.
    #[must_use]
    pub fn new(kind: AtomicKind, node_id: impl Into<String>, substrates: Vec<Substrate>) -> Self {
        Self {
            kind,
            node_id: node_id.into(),
            substrates,
            reachable: true,
        }
    }

    /// Total number of substrates this atomic can dispatch to.
    #[must_use]
    pub const fn substrate_count(&self) -> usize {
        self.substrates.len()
    }

    /// Whether this atomic can handle a given workload.
    #[must_use]
    pub fn can_route(&self, workload: &crate::dispatch::Workload) -> bool {
        self.kind.has_compute() && crate::dispatch::route(workload, &self.substrates).is_some()
    }
}

#[cfg(test)]
#[expect(clippy::expect_used, reason = "test assertions")]
mod tests {
    use super::*;
    use crate::substrate::{Capability, Identity, Properties, SubstrateKind};

    fn gpu(name: &str) -> Substrate {
        Substrate {
            kind: SubstrateKind::Gpu,
            identity: Identity::named(name),
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

    fn cpu() -> Substrate {
        Substrate {
            kind: SubstrateKind::Cpu,
            identity: Identity::named("i9-12900K"),
            properties: Properties::default(),
            capabilities: vec![
                Capability::F64Compute,
                Capability::CpuCompute,
                Capability::SimdVector,
            ],
        }
    }

    fn npu() -> Substrate {
        Substrate {
            kind: SubstrateKind::Npu,
            identity: Identity::named("AKD1000"),
            properties: Properties::default(),
            capabilities: vec![
                Capability::QuantizedInference { bits: 8 },
                Capability::BatchInference { max_batch: 8 },
            ],
        }
    }

    #[test]
    fn tower_capabilities() {
        assert_eq!(AtomicKind::Tower.capabilities().len(), 3);
        assert!(AtomicKind::Tower.capabilities().contains(&"crypto.tls"));
        assert!(AtomicKind::Tower.capabilities().contains(&"mesh.discovery"));
        assert!(AtomicKind::Tower.capabilities().contains(&"defense.audit"));
        assert!(!AtomicKind::Tower.has_compute());
        assert!(!AtomicKind::Tower.has_storage());
        assert!(AtomicKind::Tower.has_mesh());
    }

    #[test]
    fn node_has_compute() {
        assert!(AtomicKind::Node.has_compute());
        assert_eq!(AtomicKind::Node.capabilities().len(), 4);
        assert!(
            AtomicKind::Node
                .capabilities()
                .contains(&"compute.dispatch")
        );
    }

    #[test]
    fn nest_has_storage() {
        assert!(AtomicKind::Nest.has_storage());
        assert_eq!(AtomicKind::Nest.capabilities().len(), 4);
        assert!(
            AtomicKind::Nest
                .capabilities()
                .contains(&"storage.provenance")
        );
    }

    #[test]
    fn node_can_route_f64_workload() {
        let node = NucleusAtomic::new(AtomicKind::Node, "node-01", vec![gpu("TITAN V"), cpu()]);
        let wl = crate::dispatch::Workload::new(
            "et0_batch",
            vec![Capability::F64Compute, Capability::ShaderDispatch],
        );
        assert!(node.can_route(&wl));
    }

    #[test]
    fn tower_cannot_route() {
        let tower = NucleusAtomic::new(AtomicKind::Tower, "tower-01", vec![cpu()]);
        let wl = crate::dispatch::Workload::new("anything", vec![Capability::F64Compute]);
        assert!(!tower.can_route(&wl));
    }

    #[test]
    fn nest_cannot_dispatch_workloads() {
        let nest = NucleusAtomic::new(AtomicKind::Nest, "nest-01", vec![gpu("GPU"), cpu()]);
        let wl = crate::dispatch::Workload::new("et0", vec![Capability::F64Compute]);
        assert!(!nest.can_route(&wl), "Nest lacks compute dispatch role");
    }

    #[test]
    fn atomic_capabilities_superset_chain() {
        let tower_caps = AtomicKind::Tower.capabilities();
        let node_caps = AtomicKind::Node.capabilities();
        let nest_caps = AtomicKind::Nest.capabilities();

        for cap in tower_caps {
            assert!(
                node_caps.contains(cap),
                "Node must include all Tower capabilities"
            );
            assert!(
                nest_caps.contains(cap),
                "Nest must include all Tower capabilities"
            );
        }

        assert!(node_caps.contains(&"compute.dispatch"));
        assert!(!node_caps.contains(&"storage.provenance"));
        assert!(nest_caps.contains(&"storage.provenance"));
        assert!(!nest_caps.contains(&"compute.dispatch"));
    }
}
