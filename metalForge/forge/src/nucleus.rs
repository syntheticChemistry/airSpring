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
            Self::Node => &["crypto.tls", "mesh.discovery", "defense.audit", "compute.dispatch"],
            Self::Nest => &["crypto.tls", "mesh.discovery", "defense.audit", "storage.provenance"],
        }
    }

    /// Human-readable component descriptions (primal names resolved at runtime).
    #[must_use]
    pub const fn component_descriptions(&self) -> &[&str] {
        match self {
            Self::Tower => &["crypto/TLS provider", "mesh/discovery provider", "defense/audit sentinel"],
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

/// A mesh of NUCLEUS atomics coordinated via biomeOS.
#[derive(Debug, Default)]
pub struct NucleusMesh {
    /// All discovered atomics on the mesh.
    pub atomics: Vec<NucleusAtomic>,
}

impl NucleusMesh {
    /// Create a new empty mesh.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            atomics: Vec::new(),
        }
    }

    /// Register an atomic on the mesh.
    pub fn register(&mut self, atomic: NucleusAtomic) {
        self.atomics.push(atomic);
    }

    /// Find all Node atomics that can handle a workload.
    #[must_use]
    pub fn find_capable_nodes(&self, workload: &crate::dispatch::Workload) -> Vec<&NucleusAtomic> {
        self.atomics
            .iter()
            .filter(|a| a.reachable && a.can_route(workload))
            .collect()
    }

    /// Count atomics by kind.
    #[must_use]
    pub fn count_by_kind(&self, kind: AtomicKind) -> usize {
        self.atomics.iter().filter(|a| a.kind == kind).count()
    }

    /// All substrates across all reachable compute nodes.
    #[must_use]
    pub fn all_substrates(&self) -> Vec<&Substrate> {
        self.atomics
            .iter()
            .filter(|a| a.reachable && a.kind.has_compute())
            .flat_map(|a| a.substrates.iter())
            .collect()
    }

    /// Route a pipeline across the mesh, selecting the best node per stage.
    ///
    /// For each workload, finds the best capable node and its best substrate.
    /// Returns a `MeshPipeline` with per-stage node assignments and
    /// inter-node transfer characterisation.
    #[must_use]
    pub fn route_pipeline(
        &self,
        workloads: &[crate::dispatch::Workload],
    ) -> Option<MeshPipeline<'_>> {
        if workloads.is_empty() {
            return Some(MeshPipeline {
                stages: Vec::new(),
                cross_node_hops: 0,
                local_pcie_bypasses: 0,
            });
        }

        let mut stages = Vec::with_capacity(workloads.len());
        let mut cross_node_hops = 0_usize;
        let mut local_pcie_bypasses = 0_usize;
        let mut prev_node_id: Option<&str> = None;

        for wl in workloads {
            let capable_nodes = self.find_capable_nodes(wl);
            if capable_nodes.is_empty() {
                return None;
            }

            let node = prev_node_id.map_or_else(
                || capable_nodes[0],
                |prev| {
                    capable_nodes
                        .iter()
                        .find(|n| n.node_id == prev)
                        .copied()
                        .unwrap_or(capable_nodes[0])
                },
            );

            let decision = crate::dispatch::route(wl, &node.substrates)?;
            let same_node = prev_node_id.is_none_or(|p| p == node.node_id);

            if !same_node {
                cross_node_hops += 1;
            }

            if same_node
                && decision.substrate.kind == crate::substrate::SubstrateKind::Gpu
                && let Some(npu_sub) = node
                    .substrates
                    .iter()
                    .find(|s| s.kind == crate::substrate::SubstrateKind::Npu)
                && npu_sub.identity.pci_id.is_some()
                && decision.substrate.identity.pci_id.is_some()
            {
                local_pcie_bypasses += 1;
            }

            stages.push(MeshStage {
                node_id: &node.node_id,
                substrate_kind: decision.substrate.kind,
                same_node,
            });

            prev_node_id = Some(&node.node_id);
        }

        Some(MeshPipeline {
            stages,
            cross_node_hops,
            local_pcie_bypasses,
        })
    }
}

/// A stage in a mesh-routed pipeline.
#[derive(Debug)]
pub struct MeshStage<'a> {
    /// Which mesh node handles this stage.
    pub node_id: &'a str,
    /// Which substrate kind executes the workload.
    pub substrate_kind: crate::substrate::SubstrateKind,
    /// Whether this stage runs on the same node as the previous stage.
    pub same_node: bool,
}

/// A pipeline routed across the NUCLEUS mesh.
#[derive(Debug)]
pub struct MeshPipeline<'a> {
    /// Ordered sequence of mesh stages.
    pub stages: Vec<MeshStage<'a>>,
    /// Number of cross-node hops (biomeOS Neural API transfers).
    pub cross_node_hops: usize,
    /// Number of local `PCIe` P2P bypass opportunities.
    pub local_pcie_bypasses: usize,
}

impl MeshPipeline<'_> {
    /// Whether the entire pipeline runs on a single node.
    #[must_use]
    pub const fn is_single_node(&self) -> bool {
        self.cross_node_hops == 0
    }

    /// Number of stages.
    #[must_use]
    pub const fn stage_count(&self) -> usize {
        self.stages.len()
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
    fn mesh_register_and_count() {
        let mut mesh = NucleusMesh::new();
        mesh.register(NucleusAtomic::new(AtomicKind::Tower, "tower-01", vec![]));
        mesh.register(NucleusAtomic::new(
            AtomicKind::Node,
            "node-01",
            vec![gpu("TITAN V"), cpu()],
        ));
        mesh.register(NucleusAtomic::new(AtomicKind::Nest, "nest-01", vec![]));

        assert_eq!(mesh.count_by_kind(AtomicKind::Tower), 1);
        assert_eq!(mesh.count_by_kind(AtomicKind::Node), 1);
        assert_eq!(mesh.count_by_kind(AtomicKind::Nest), 1);
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
    fn mesh_finds_capable_nodes() {
        let mut mesh = NucleusMesh::new();
        mesh.register(NucleusAtomic::new(
            AtomicKind::Node,
            "gpu-node",
            vec![gpu("TITAN V"), cpu()],
        ));
        mesh.register(NucleusAtomic::new(
            AtomicKind::Node,
            "npu-node",
            vec![npu(), cpu()],
        ));
        mesh.register(NucleusAtomic::new(AtomicKind::Tower, "tower", vec![]));

        let gpu_wl = crate::dispatch::Workload::new(
            "et0_batch",
            vec![Capability::F64Compute, Capability::ShaderDispatch],
        );
        let capable = mesh.find_capable_nodes(&gpu_wl);
        assert_eq!(capable.len(), 1);
        assert_eq!(capable[0].node_id, "gpu-node");

        let npu_wl = crate::dispatch::Workload::new(
            "crop_stress",
            vec![Capability::QuantizedInference { bits: 8 }],
        );
        let capable = mesh.find_capable_nodes(&npu_wl);
        assert_eq!(capable.len(), 1);
        assert_eq!(capable[0].node_id, "npu-node");
    }

    #[test]
    fn all_substrates_from_nodes_only() {
        let mut mesh = NucleusMesh::new();
        mesh.register(NucleusAtomic::new(
            AtomicKind::Node,
            "node-a",
            vec![gpu("TITAN V")],
        ));
        mesh.register(NucleusAtomic::new(AtomicKind::Nest, "nest-a", vec![cpu()]));

        let subs = mesh.all_substrates();
        assert_eq!(subs.len(), 1);
        assert_eq!(subs[0].kind, SubstrateKind::Gpu);
    }

    #[test]
    fn unreachable_node_excluded() {
        let mut mesh = NucleusMesh::new();
        let mut node = NucleusAtomic::new(AtomicKind::Node, "offline", vec![gpu("TITAN V"), cpu()]);
        node.reachable = false;
        mesh.register(node);

        let wl = crate::dispatch::Workload::new(
            "et0_batch",
            vec![Capability::F64Compute, Capability::ShaderDispatch],
        );
        assert!(mesh.find_capable_nodes(&wl).is_empty());
        assert!(mesh.all_substrates().is_empty());
    }

    #[test]
    fn mixed_pipeline_across_mesh() {
        let mut mesh = NucleusMesh::new();
        mesh.register(NucleusAtomic::new(
            AtomicKind::Node,
            "eastgate",
            vec![gpu("TITAN V"), npu(), cpu()],
        ));

        let all_subs = mesh.all_substrates();
        let all_subs_owned: Vec<Substrate> = all_subs.into_iter().cloned().collect();

        let workloads = [
            crate::dispatch::Workload::new(
                "crop_stress",
                vec![Capability::QuantizedInference { bits: 8 }],
            )
            .prefer(SubstrateKind::Npu),
            crate::dispatch::Workload::new(
                "et0_batch",
                vec![Capability::F64Compute, Capability::ShaderDispatch],
            ),
        ];

        let pipeline =
            crate::pipeline::route_pipeline(&workloads, &all_subs_owned).expect("should route");
        assert_eq!(pipeline.stages.len(), 2);
        assert_eq!(pipeline.stages[0].substrate.kind, SubstrateKind::Npu);
        assert_eq!(pipeline.stages[1].substrate.kind, SubstrateKind::Gpu);
    }

    #[test]
    fn mesh_pipeline_single_node() {
        let mut mesh = NucleusMesh::new();
        mesh.register(NucleusAtomic::new(
            AtomicKind::Node,
            "eastgate",
            vec![gpu("TITAN V"), npu(), cpu()],
        ));

        let workloads = [
            crate::dispatch::Workload::new(
                "crop_stress",
                vec![Capability::QuantizedInference { bits: 8 }],
            )
            .prefer(SubstrateKind::Npu),
            crate::dispatch::Workload::new(
                "et0_batch",
                vec![Capability::F64Compute, Capability::ShaderDispatch],
            ),
            crate::dispatch::Workload::new(
                "yield_response",
                vec![Capability::F64Compute, Capability::ShaderDispatch],
            ),
        ];

        let pipeline = mesh.route_pipeline(&workloads).expect("should route");
        assert_eq!(pipeline.stage_count(), 3);
        assert!(pipeline.is_single_node());
        assert_eq!(pipeline.cross_node_hops, 0);
    }

    #[test]
    fn mesh_pipeline_cross_node_hop() {
        let mut mesh = NucleusMesh::new();
        mesh.register(NucleusAtomic::new(
            AtomicKind::Node,
            "gpu-node",
            vec![gpu("TITAN V"), cpu()],
        ));
        mesh.register(NucleusAtomic::new(
            AtomicKind::Node,
            "npu-node",
            vec![npu(), cpu()],
        ));

        let workloads = [
            crate::dispatch::Workload::new(
                "et0_batch",
                vec![Capability::F64Compute, Capability::ShaderDispatch],
            ),
            crate::dispatch::Workload::new(
                "crop_stress",
                vec![Capability::QuantizedInference { bits: 8 }],
            ),
        ];

        let pipeline = mesh.route_pipeline(&workloads).expect("should route");
        assert_eq!(pipeline.stage_count(), 2);
        assert!(!pipeline.is_single_node());
        assert_eq!(pipeline.cross_node_hops, 1);
        assert_eq!(pipeline.stages[0].node_id, "gpu-node");
        assert_eq!(pipeline.stages[1].node_id, "npu-node");
    }

    #[test]
    fn mesh_pipeline_empty() {
        let mesh = NucleusMesh::new();
        let pipeline = mesh.route_pipeline(&[]).expect("empty should route");
        assert!(pipeline.is_single_node());
        assert_eq!(pipeline.stage_count(), 0);
    }

    #[test]
    fn mesh_pipeline_prefers_same_node() {
        let mut mesh = NucleusMesh::new();
        mesh.register(NucleusAtomic::new(
            AtomicKind::Node,
            "node-a",
            vec![gpu("TITAN V"), cpu()],
        ));
        mesh.register(NucleusAtomic::new(
            AtomicKind::Node,
            "node-b",
            vec![gpu("RTX 4070"), cpu()],
        ));

        let workloads = [
            crate::dispatch::Workload::new(
                "et0_batch",
                vec![Capability::F64Compute, Capability::ShaderDispatch],
            ),
            crate::dispatch::Workload::new(
                "water_balance",
                vec![Capability::F64Compute, Capability::ShaderDispatch],
            ),
        ];

        let pipeline = mesh.route_pipeline(&workloads).expect("should route");
        assert!(pipeline.is_single_node(), "should stay on same node");
        assert_eq!(pipeline.stages[0].node_id, pipeline.stages[1].node_id);
    }

    // ── Mixed NUCLEUS Composition Tests ──

    #[test]
    fn full_nucleus_trio_tower_node_nest() {
        let mut mesh = NucleusMesh::new();
        let tower = NucleusAtomic::new(AtomicKind::Tower, "tower-01", vec![]);
        let node = NucleusAtomic::new(
            AtomicKind::Node,
            "node-01",
            vec![gpu("TITAN V"), npu(), cpu()],
        );
        let nest = NucleusAtomic::new(AtomicKind::Nest, "nest-01", vec![]);

        mesh.register(tower);
        mesh.register(node);
        mesh.register(nest);

        assert_eq!(mesh.count_by_kind(AtomicKind::Tower), 1);
        assert_eq!(mesh.count_by_kind(AtomicKind::Node), 1);
        assert_eq!(mesh.count_by_kind(AtomicKind::Nest), 1);
        assert_eq!(mesh.atomics.len(), 3);

        assert_eq!(mesh.all_substrates().len(), 3);

        let gpu_wl = crate::dispatch::Workload::new(
            "et0_batch",
            vec![Capability::F64Compute, Capability::ShaderDispatch],
        );
        let npu_wl = crate::dispatch::Workload::new(
            "crop_stress",
            vec![Capability::QuantizedInference { bits: 8 }],
        );

        let gpu_nodes = mesh.find_capable_nodes(&gpu_wl);
        let npu_nodes = mesh.find_capable_nodes(&npu_wl);
        assert_eq!(gpu_nodes.len(), 1);
        assert_eq!(npu_nodes.len(), 1);
        assert_eq!(gpu_nodes[0].node_id, "node-01");
        assert_eq!(npu_nodes[0].node_id, "node-01");
    }

    #[test]
    fn multi_node_mesh_heterogeneous_hardware() {
        let mut mesh = NucleusMesh::new();

        mesh.register(NucleusAtomic::new(
            AtomicKind::Node,
            "gpu-dense",
            vec![gpu("TITAN V"), gpu("RTX 4070"), cpu()],
        ));
        mesh.register(NucleusAtomic::new(
            AtomicKind::Node,
            "npu-edge",
            vec![npu(), cpu()],
        ));
        mesh.register(NucleusAtomic::new(
            AtomicKind::Node,
            "cpu-only",
            vec![cpu()],
        ));
        mesh.register(NucleusAtomic::new(AtomicKind::Tower, "tower-gw", vec![]));
        mesh.register(NucleusAtomic::new(AtomicKind::Nest, "nest-store", vec![]));

        assert_eq!(mesh.count_by_kind(AtomicKind::Node), 3);
        assert_eq!(mesh.all_substrates().len(), 6);

        let f64_wl = crate::dispatch::Workload::new(
            "richards_pde",
            vec![Capability::F64Compute, Capability::ShaderDispatch],
        );
        let capable = mesh.find_capable_nodes(&f64_wl);
        assert_eq!(capable.len(), 1, "only gpu-dense has ShaderDispatch");
        assert_eq!(capable[0].node_id, "gpu-dense");

        let cpu_wl = crate::dispatch::Workload::new(
            "validation",
            vec![Capability::F64Compute],
        );
        let capable = mesh.find_capable_nodes(&cpu_wl);
        assert_eq!(capable.len(), 3, "gpu-dense, npu-edge (has CPU), cpu-only have F64Compute");
    }

    #[test]
    fn mixed_pipeline_npu_gpu_gpu_stays_single_node() {
        let mut mesh = NucleusMesh::new();
        mesh.register(NucleusAtomic::new(
            AtomicKind::Node,
            "mixed-hw",
            vec![gpu("TITAN V"), npu(), cpu()],
        ));

        let workloads = [
            crate::dispatch::Workload::new(
                "crop_stress",
                vec![Capability::QuantizedInference { bits: 8 }],
            )
            .prefer(SubstrateKind::Npu),
            crate::dispatch::Workload::new(
                "et0_batch",
                vec![Capability::F64Compute, Capability::ShaderDispatch],
            ),
            crate::dispatch::Workload::new(
                "water_balance",
                vec![Capability::F64Compute, Capability::ShaderDispatch],
            ),
            crate::dispatch::Workload::new(
                "yield_response",
                vec![Capability::F64Compute, Capability::ShaderDispatch],
            ),
        ];

        let pipeline = mesh.route_pipeline(&workloads).expect("should route");
        assert_eq!(pipeline.stage_count(), 4);
        assert!(pipeline.is_single_node());
        assert_eq!(pipeline.cross_node_hops, 0);

        assert_eq!(pipeline.stages[0].substrate_kind, SubstrateKind::Npu);
        assert_eq!(pipeline.stages[1].substrate_kind, SubstrateKind::Gpu);
        assert_eq!(pipeline.stages[2].substrate_kind, SubstrateKind::Gpu);
        assert_eq!(pipeline.stages[3].substrate_kind, SubstrateKind::Gpu);
    }

    #[test]
    fn cross_node_forced_by_capability_split() {
        let mut mesh = NucleusMesh::new();

        mesh.register(NucleusAtomic::new(
            AtomicKind::Node,
            "gpu-compute",
            vec![gpu("TITAN V"), cpu()],
        ));
        mesh.register(NucleusAtomic::new(
            AtomicKind::Node,
            "npu-inference",
            vec![npu(), cpu()],
        ));

        let workloads = [
            crate::dispatch::Workload::new(
                "et0_batch",
                vec![Capability::F64Compute, Capability::ShaderDispatch],
            ),
            crate::dispatch::Workload::new(
                "crop_stress",
                vec![Capability::QuantizedInference { bits: 8 }],
            ),
            crate::dispatch::Workload::new(
                "water_balance",
                vec![Capability::F64Compute, Capability::ShaderDispatch],
            ),
        ];

        let pipeline = mesh.route_pipeline(&workloads).expect("should route");
        assert_eq!(pipeline.stage_count(), 3);
        assert!(!pipeline.is_single_node());
        assert!(pipeline.cross_node_hops >= 1);

        assert_eq!(pipeline.stages[0].node_id, "gpu-compute");
        assert_eq!(pipeline.stages[1].node_id, "npu-inference");
        assert_eq!(pipeline.stages[2].node_id, "gpu-compute");
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

    #[test]
    fn large_mesh_routing_performance() {
        let mut mesh = NucleusMesh::new();
        for i in 0..10 {
            mesh.register(NucleusAtomic::new(
                AtomicKind::Node,
                format!("node-{i}"),
                vec![gpu(&format!("GPU-{i}")), cpu()],
            ));
        }
        mesh.register(NucleusAtomic::new(
            AtomicKind::Node,
            "npu-node",
            vec![npu(), cpu()],
        ));

        let workloads: Vec<_> = (0..20)
            .map(|i| {
                crate::dispatch::Workload::new(
                    format!("batch_{i}"),
                    vec![Capability::F64Compute, Capability::ShaderDispatch],
                )
            })
            .collect();

        let pipeline = mesh.route_pipeline(&workloads).expect("should route");
        assert_eq!(pipeline.stage_count(), 20);
        assert!(pipeline.is_single_node(), "sticky routing keeps all on node-0");
    }
}
