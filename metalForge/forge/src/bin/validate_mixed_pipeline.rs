// SPDX-License-Identifier: AGPL-3.0-or-later
#![warn(clippy::pedantic)]
//! Exp 056: metalForge Mixed-Hardware Pipeline + NUCLEUS Atomics.
//!
//! Validates:
//! 1. `PCIe` P2P transfer path selection (NPU→GPU bypass)
//! 2. Pipeline routing across mixed substrates
//! 3. NUCLEUS atomic mesh coordination
//! 4. CPU roundtrip elimination for multi-stage pipelines

use airspring_forge::dispatch::Workload;
use airspring_forge::nucleus::{AtomicKind, NucleusAtomic, NucleusMesh};
use airspring_forge::pipeline::{self, TransferPath};
use airspring_forge::substrate::pci;
use airspring_forge::substrate::{Capability, Identity, Properties, Substrate, SubstrateKind};
use barracuda::validation::ValidationHarness;

fn titan_v() -> Substrate {
    Substrate {
        kind: SubstrateKind::Gpu,
        identity: Identity {
            pci_id: Some(pci::NVIDIA_TITAN_V.to_string()),
            driver: Some("NVIDIA (580.119.02)".to_string()),
            backend: Some("Vulkan".to_string()),
            ..Identity::named("NVIDIA TITAN V")
        },
        properties: Properties {
            has_f64: true,
            has_timestamps: true,
            memory_bytes: Some(12 * 1024 * 1024 * 1024),
            ..Properties::default()
        },
        capabilities: vec![
            Capability::F64Compute,
            Capability::F32Compute,
            Capability::ShaderDispatch,
            Capability::ScalarReduce,
            Capability::TimestampQuery,
        ],
    }
}

fn rtx_4070() -> Substrate {
    Substrate {
        kind: SubstrateKind::Gpu,
        identity: Identity {
            pci_id: Some(pci::NVIDIA_RTX_4070.to_string()),
            driver: Some("NVIDIA (580.119.02)".to_string()),
            backend: Some("Vulkan".to_string()),
            ..Identity::named("NVIDIA GeForce RTX 4070")
        },
        properties: Properties {
            has_f64: true,
            has_timestamps: true,
            memory_bytes: Some(12 * 1024 * 1024 * 1024),
            ..Properties::default()
        },
        capabilities: vec![
            Capability::F64Compute,
            Capability::F32Compute,
            Capability::ShaderDispatch,
            Capability::ScalarReduce,
            Capability::TimestampQuery,
        ],
    }
}

fn akd1000() -> Substrate {
    Substrate {
        kind: SubstrateKind::Npu,
        identity: Identity {
            pci_id: Some(pci::BRAINCHIP_AKD1000.to_string()),
            device_node: Some("/dev/akida0".to_string()),
            ..Identity::named("BrainChip AKD1000")
        },
        properties: Properties {
            memory_bytes: Some(4 * 1024 * 1024),
            ..Properties::default()
        },
        capabilities: vec![
            Capability::QuantizedInference { bits: 8 },
            Capability::BatchInference { max_batch: 8 },
            Capability::WeightMutation,
        ],
    }
}

fn i9_cpu() -> Substrate {
    Substrate {
        kind: SubstrateKind::Cpu,
        identity: Identity::named("12th Gen Intel(R) Core(TM) i9-12900K"),
        properties: Properties {
            core_count: Some(16),
            thread_count: Some(24),
            memory_bytes: Some(32 * 1024 * 1024 * 1024),
            ..Properties::default()
        },
        capabilities: vec![
            Capability::F64Compute,
            Capability::F32Compute,
            Capability::CpuCompute,
            Capability::SimdVector,
        ],
    }
}

fn main() {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .with_target(false)
        .without_time()
        .init();

    println!("═══════════════════════════════════════════════════════════");
    println!("  airSpring Exp 056: Mixed-Hardware Pipeline + NUCLEUS");
    println!("═══════════════════════════════════════════════════════════\n");

    let mut v = ValidationHarness::new("Mixed Pipeline + NUCLEUS");

    validate_transfer_paths(&mut v);
    validate_pcie_bypass_pipeline(&mut v);
    validate_cpu_roundtrip_counting(&mut v);
    validate_nucleus_mesh(&mut v);
    validate_mesh_routing(&mut v);
    validate_full_eco_pipeline(&mut v);
    validate_full_seasonal_cross_system(&mut v);
    validate_mesh_pipeline_routing(&mut v);
    validate_cross_node_pipeline(&mut v);

    v.finish();
}

#[expect(
    clippy::similar_names,
    reason = "transfer path validation uses gpu_npu/gpu_cpu/gpu_gpu for GPU→substrate paths"
)]
fn validate_transfer_paths(v: &mut ValidationHarness) {
    println!("── Transfer Path Selection ──");

    let gpu = titan_v();
    let npu = akd1000();
    let cpu = i9_cpu();

    let npu_gpu = pipeline::transfer_path(&npu, &gpu);
    v.check_bool(
        "NPU→GPU uses `PCIe` P2P",
        npu_gpu == TransferPath::PciePeerToPeer,
    );
    v.check_bool("NPU→GPU bypasses CPU", npu_gpu.bypasses_cpu());

    let gpu_npu = pipeline::transfer_path(&gpu, &npu);
    v.check_bool(
        "GPU→NPU uses `PCIe` P2P",
        gpu_npu == TransferPath::PciePeerToPeer,
    );

    let gpu_cpu = pipeline::transfer_path(&gpu, &cpu);
    v.check_bool("GPU→CPU uses memcpy", gpu_cpu == TransferPath::CpuMemcpy);
    v.check_bool("GPU→CPU does NOT bypass CPU", !gpu_cpu.bypasses_cpu());

    let cpu_gpu = pipeline::transfer_path(&cpu, &gpu);
    v.check_bool("CPU→GPU uses memcpy", cpu_gpu == TransferPath::CpuMemcpy);

    let gpu_gpu = pipeline::transfer_path(&gpu, &gpu);
    v.check_bool("GPU→GPU (same kind) is None", gpu_gpu == TransferPath::None);

    v.check_bool(
        "Latency: None < P2P < Memcpy < Neural",
        TransferPath::None.latency_rank() < TransferPath::PciePeerToPeer.latency_rank()
            && TransferPath::PciePeerToPeer.latency_rank() < TransferPath::CpuMemcpy.latency_rank()
            && TransferPath::CpuMemcpy.latency_rank() < TransferPath::NeuralApi.latency_rank(),
    );
}

fn validate_pcie_bypass_pipeline(v: &mut ValidationHarness) {
    println!("\n── `PCIe` Bypass Pipeline (NPU→GPU) ──");

    let substrates = [akd1000(), titan_v(), i9_cpu()];
    let workloads = [
        Workload::new(
            "crop_stress_classifier",
            vec![Capability::QuantizedInference { bits: 8 }],
        )
        .prefer(SubstrateKind::Npu),
        Workload::new(
            "et0_batch",
            vec![Capability::F64Compute, Capability::ShaderDispatch],
        ),
        Workload::new(
            "yield_response_surface",
            vec![Capability::F64Compute, Capability::ShaderDispatch],
        ),
    ];

    let pipe = pipeline::route_pipeline(&workloads, &substrates).expect("should route");

    v.check_bool("3-stage pipeline", pipe.stages.len() == 3);
    v.check_bool(
        "Stage 1: crop_stress → NPU",
        pipe.stages[0].substrate.kind == SubstrateKind::Npu,
    );
    v.check_bool(
        "Stage 2: et0_batch → GPU",
        pipe.stages[1].substrate.kind == SubstrateKind::Gpu,
    );
    v.check_bool(
        "Stage 3: yield_response → GPU",
        pipe.stages[2].substrate.kind == SubstrateKind::Gpu,
    );
    v.check_bool(
        "NPU→GPU transfer is `PCIe` P2P",
        pipe.stages[1].transfer_in == TransferPath::PciePeerToPeer,
    );
    v.check_bool(
        "GPU→GPU transfer is None (same kind)",
        pipe.stages[2].transfer_in == TransferPath::None,
    );
    v.check_bool("Zero CPU roundtrips", pipe.cpu_roundtrips == 0);
    v.check_bool("One `PCIe` bypass", pipe.pcie_bypasses == 1);
    v.check_bool("Fully bypasses CPU", pipe.fully_bypasses_cpu());
    println!(
        "  Total transfer cost: {} (lower is better)",
        pipe.total_transfer_cost
    );
}

fn validate_cpu_roundtrip_counting(v: &mut ValidationHarness) {
    println!("\n── CPU Roundtrip Detection ──");

    let substrates = [titan_v(), i9_cpu()];
    let workloads = [
        Workload::new(
            "et0_batch",
            vec![Capability::F64Compute, Capability::ShaderDispatch],
        ),
        Workload::new("weather_ingest", vec![Capability::CpuCompute]),
        Workload::new(
            "water_balance_batch",
            vec![Capability::F64Compute, Capability::ShaderDispatch],
        ),
    ];

    let pipe = pipeline::route_pipeline(&workloads, &substrates).expect("should route");
    v.check_bool("GPU→CPU→GPU has 2 roundtrips", pipe.cpu_roundtrips == 2);
    v.check_bool("Does NOT fully bypass CPU", !pipe.fully_bypasses_cpu());
}

fn validate_nucleus_mesh(v: &mut ValidationHarness) {
    println!("\n── NUCLEUS Atomic Mesh ──");

    let mut mesh = NucleusMesh::new();

    let tower = NucleusAtomic::new(AtomicKind::Tower, "eastgate-tower", vec![]);
    mesh.register(tower);

    let node = NucleusAtomic::new(
        AtomicKind::Node,
        "eastgate-node",
        vec![titan_v(), rtx_4070(), akd1000(), i9_cpu()],
    );
    mesh.register(node);

    let nest = NucleusAtomic::new(AtomicKind::Nest, "eastgate-nest", vec![]);
    mesh.register(nest);

    v.check_bool("1 Tower atomic", mesh.count_by_kind(AtomicKind::Tower) == 1);
    v.check_bool("1 Node atomic", mesh.count_by_kind(AtomicKind::Node) == 1);
    v.check_bool("1 Nest atomic", mesh.count_by_kind(AtomicKind::Nest) == 1);
    v.check_bool("4 substrates from Node", mesh.all_substrates().len() == 4);

    v.check_bool("Tower has mesh", AtomicKind::Tower.has_mesh());
    v.check_bool("Node has mesh", AtomicKind::Node.has_mesh());
    v.check_bool("Nest has mesh", AtomicKind::Nest.has_mesh());
    v.check_bool("Tower lacks compute", !AtomicKind::Tower.has_compute());
    v.check_bool("Node has compute", AtomicKind::Node.has_compute());
    v.check_bool("Nest has storage", AtomicKind::Nest.has_storage());

    v.check_bool(
        "Tower: 2 capabilities",
        AtomicKind::Tower.capabilities().len() == 2,
    );
    v.check_bool(
        "Node: 3 capabilities",
        AtomicKind::Node.capabilities().len() == 3,
    );
    v.check_bool(
        "Nest: 3 capabilities",
        AtomicKind::Nest.capabilities().len() == 3,
    );
}

#[expect(
    clippy::similar_names,
    reason = "mesh routing validation uses capable_gpu/capable_npu/capable_cpu for substrate lookup"
)]
fn validate_mesh_routing(v: &mut ValidationHarness) {
    println!("\n── Mesh-Aware Routing ──");

    let mut mesh = NucleusMesh::new();

    mesh.register(NucleusAtomic::new(
        AtomicKind::Node,
        "gpu-node",
        vec![titan_v(), i9_cpu()],
    ));
    mesh.register(NucleusAtomic::new(
        AtomicKind::Node,
        "npu-node",
        vec![akd1000(), i9_cpu()],
    ));

    let mut offline_node =
        NucleusAtomic::new(AtomicKind::Node, "offline-node", vec![rtx_4070(), i9_cpu()]);
    offline_node.reachable = false;
    mesh.register(offline_node);

    mesh.register(NucleusAtomic::new(AtomicKind::Tower, "tower-01", vec![]));

    let gpu_wl = Workload::new(
        "et0_batch",
        vec![Capability::F64Compute, Capability::ShaderDispatch],
    );
    let capable_gpu = mesh.find_capable_nodes(&gpu_wl);
    v.check_bool("1 node can route GPU workload", capable_gpu.len() == 1);
    v.check_bool(
        "GPU workload routes to gpu-node",
        capable_gpu.first().is_some_and(|n| n.node_id == "gpu-node"),
    );

    let npu_wl = Workload::new(
        "crop_stress",
        vec![Capability::QuantizedInference { bits: 8 }],
    );
    let capable_npu = mesh.find_capable_nodes(&npu_wl);
    v.check_bool("1 node can route NPU workload", capable_npu.len() == 1);
    v.check_bool(
        "NPU workload routes to npu-node",
        capable_npu.first().is_some_and(|n| n.node_id == "npu-node"),
    );

    let cpu_wl = Workload::new("validation", vec![Capability::F64Compute]);
    let capable_cpu = mesh.find_capable_nodes(&cpu_wl);
    v.check_bool(
        "2 reachable nodes can route f64 (gpu-node + npu-node)",
        capable_cpu.len() == 2,
    );

    let offline_check = mesh
        .find_capable_nodes(&gpu_wl)
        .iter()
        .any(|n| n.node_id == "offline-node");
    v.check_bool("Offline node excluded from routing", !offline_check);
}

fn validate_full_eco_pipeline(v: &mut ValidationHarness) {
    println!("\n── Full Eco Pipeline (5-stage mixed hardware) ──");

    let substrates = [akd1000(), titan_v(), i9_cpu()];
    let workloads = [
        Workload::new(
            "crop_stress_classifier",
            vec![Capability::QuantizedInference { bits: 8 }],
        )
        .prefer(SubstrateKind::Npu),
        Workload::new(
            "et0_batch",
            vec![Capability::F64Compute, Capability::ShaderDispatch],
        ),
        Workload::new(
            "water_balance_batch",
            vec![Capability::F64Compute, Capability::ShaderDispatch],
        ),
        Workload::new(
            "yield_response_surface",
            vec![Capability::F64Compute, Capability::ShaderDispatch],
        ),
        Workload::new("weather_ingest", vec![Capability::CpuCompute]),
    ];

    let pipe = pipeline::route_pipeline(&workloads, &substrates).expect("should route");

    v.check_bool("5-stage pipeline", pipe.stages.len() == 5);

    let route_summary: Vec<String> = pipe
        .stages
        .iter()
        .map(|s| format!("{}→{}", s.workload.name, s.substrate.kind))
        .collect();
    println!("  Route: {}", route_summary.join(" | "));
    println!(
        "  Transfers: {}",
        pipe.stages
            .iter()
            .map(|s| format!("{:?}", s.transfer_in))
            .collect::<Vec<_>>()
            .join(" → ")
    );
    println!("  CPU roundtrips: {}", pipe.cpu_roundtrips);
    println!("  `PCIe` bypasses: {}", pipe.pcie_bypasses);
    println!("  Total transfer cost: {}", pipe.total_transfer_cost);

    v.check_bool(
        "NPU→GPU via `PCIe` P2P (stage 1→2)",
        pipe.stages[1].transfer_in == TransferPath::PciePeerToPeer,
    );
    v.check_bool(
        "GPU→GPU stays on device (stages 2→3→4)",
        pipe.stages[2].transfer_in == TransferPath::None
            && pipe.stages[3].transfer_in == TransferPath::None,
    );
    v.check_bool(
        "1 CPU roundtrip (GPU→CPU at stage 5)",
        pipe.cpu_roundtrips == 1,
    );
    v.check_bool("1 `PCIe` bypass", pipe.pcie_bypasses == 1);
}

fn validate_full_seasonal_cross_system(v: &mut ValidationHarness) {
    println!("\n── Full Seasonal Pipeline Cross-System (7-stage) ──");

    let substrates = [akd1000(), titan_v(), rtx_4070(), i9_cpu()];
    let workloads = [
        Workload::new("weather_ingest", vec![Capability::CpuCompute]),
        Workload::new(
            "et0_batch",
            vec![Capability::F64Compute, Capability::ShaderDispatch],
        ),
        Workload::new(
            "kc_climate_batch",
            vec![Capability::F64Compute, Capability::ShaderDispatch],
        ),
        Workload::new(
            "water_balance_batch",
            vec![Capability::F64Compute, Capability::ShaderDispatch],
        ),
        Workload::new(
            "yield_response_surface",
            vec![Capability::F64Compute, Capability::ShaderDispatch],
        ),
        Workload::new(
            "crop_stress_classifier",
            vec![Capability::QuantizedInference { bits: 8 }],
        )
        .prefer(SubstrateKind::Npu),
        Workload::new("validation_harness", vec![Capability::CpuCompute]),
    ];

    let pipe = pipeline::route_pipeline(&workloads, &substrates).expect("7-stage route");
    v.check_bool("7-stage seasonal pipeline", pipe.stages.len() == 7);

    v.check_bool(
        "Stage 1 weather_ingest → CPU",
        pipe.stages[0].substrate.kind == SubstrateKind::Cpu,
    );
    v.check_bool(
        "Stage 2 et0_batch → GPU",
        pipe.stages[1].substrate.kind == SubstrateKind::Gpu,
    );
    v.check_bool(
        "Stage 3 kc_climate → GPU",
        pipe.stages[2].substrate.kind == SubstrateKind::Gpu,
    );
    v.check_bool(
        "Stage 4 water_balance → GPU",
        pipe.stages[3].substrate.kind == SubstrateKind::Gpu,
    );
    v.check_bool(
        "Stage 5 yield_response → GPU",
        pipe.stages[4].substrate.kind == SubstrateKind::Gpu,
    );
    v.check_bool(
        "Stage 6 crop_stress → NPU",
        pipe.stages[5].substrate.kind == SubstrateKind::Npu,
    );
    v.check_bool(
        "Stage 7 validation → CPU",
        pipe.stages[6].substrate.kind == SubstrateKind::Cpu,
    );

    v.check_bool(
        "GPU stages stay on device (2→3→4→5)",
        pipe.stages[2].transfer_in == TransferPath::None
            && pipe.stages[3].transfer_in == TransferPath::None
            && pipe.stages[4].transfer_in == TransferPath::None,
    );
    v.check_bool(
        "GPU→NPU via PCIe P2P",
        pipe.stages[5].transfer_in == TransferPath::PciePeerToPeer,
    );

    let route_summary: Vec<String> = pipe
        .stages
        .iter()
        .map(|s| format!("{}→{}", s.workload.name, s.substrate.kind))
        .collect();
    println!("  Route: {}", route_summary.join(" | "));
    println!("  CPU roundtrips: {}", pipe.cpu_roundtrips);
    println!("  PCIe bypasses: {}", pipe.pcie_bypasses);
}

fn validate_mesh_pipeline_routing(v: &mut ValidationHarness) {
    println!("\n── Mesh Pipeline Routing (Single-Node) ──");

    let mut mesh = NucleusMesh::new();
    mesh.register(NucleusAtomic::new(
        AtomicKind::Node,
        "eastgate",
        vec![titan_v(), akd1000(), i9_cpu()],
    ));

    let workloads = [
        Workload::new(
            "crop_stress_classifier",
            vec![Capability::QuantizedInference { bits: 8 }],
        )
        .prefer(SubstrateKind::Npu),
        Workload::new(
            "et0_batch",
            vec![Capability::F64Compute, Capability::ShaderDispatch],
        ),
        Workload::new(
            "water_balance_batch",
            vec![Capability::F64Compute, Capability::ShaderDispatch],
        ),
    ];

    let mp = mesh
        .route_pipeline(&workloads)
        .expect("should route on mesh");
    v.check_bool("Mesh pipeline: 3 stages", mp.stage_count() == 3);
    v.check_bool("Mesh pipeline: single node", mp.is_single_node());
    v.check_bool(
        "Mesh pipeline: zero cross-node hops",
        mp.cross_node_hops == 0,
    );
    v.check_bool(
        "Stage 1 on NPU",
        mp.stages[0].substrate_kind == SubstrateKind::Npu,
    );
    v.check_bool(
        "Stage 2 on GPU",
        mp.stages[1].substrate_kind == SubstrateKind::Gpu,
    );
    v.check_bool(
        "Stage 3 on GPU",
        mp.stages[2].substrate_kind == SubstrateKind::Gpu,
    );
    v.check_bool(
        "All stages same_node=true",
        mp.stages.iter().all(|s| s.same_node),
    );
}

fn validate_cross_node_pipeline(v: &mut ValidationHarness) {
    println!("\n── Cross-Node Pipeline (Multi-Node Mesh) ──");

    let mut mesh = NucleusMesh::new();
    mesh.register(NucleusAtomic::new(
        AtomicKind::Node,
        "gpu-node",
        vec![titan_v(), i9_cpu()],
    ));
    mesh.register(NucleusAtomic::new(
        AtomicKind::Node,
        "npu-node",
        vec![akd1000(), i9_cpu()],
    ));
    mesh.register(NucleusAtomic::new(AtomicKind::Tower, "tower-01", vec![]));

    let workloads = [
        Workload::new(
            "et0_batch",
            vec![Capability::F64Compute, Capability::ShaderDispatch],
        ),
        Workload::new(
            "crop_stress_classifier",
            vec![Capability::QuantizedInference { bits: 8 }],
        ),
        Workload::new(
            "water_balance_batch",
            vec![Capability::F64Compute, Capability::ShaderDispatch],
        ),
    ];

    let mp = mesh
        .route_pipeline(&workloads)
        .expect("should route cross-node");
    v.check_bool("Cross-node pipeline: 3 stages", mp.stage_count() == 3);
    v.check_bool("Cross-node pipeline: NOT single node", !mp.is_single_node());
    v.check_bool("Cross-node: has hops", mp.cross_node_hops > 0);
    v.check_bool(
        "Stage 1 (GPU workload) → gpu-node",
        mp.stages[0].node_id == "gpu-node",
    );
    v.check_bool(
        "Stage 2 (NPU workload) → npu-node",
        mp.stages[1].node_id == "npu-node",
    );
    v.check_bool(
        "Stage 3 (GPU workload) → gpu-node",
        mp.stages[2].node_id == "gpu-node",
    );

    println!("  Mesh route:");
    for (i, stage) in mp.stages.iter().enumerate() {
        println!(
            "    Stage {}: {} → {} (same_node={})",
            i + 1,
            stage.node_id,
            stage.substrate_kind,
            stage.same_node
        );
    }
    println!("  Cross-node hops: {}", mp.cross_node_hops);
}
