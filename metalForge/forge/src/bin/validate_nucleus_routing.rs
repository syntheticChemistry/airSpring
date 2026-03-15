// SPDX-License-Identifier: AGPL-3.0-or-later
#![warn(clippy::pedantic)]
//! Exp 076: NUCLEUS Mixed-Hardware Pipeline Routing.
//!
//! Validates that all 27 eco workloads route correctly through the NUCLEUS
//! mesh with Tower/Node/Nest atomics, demonstrating:
//!
//! 1. Capability-based dispatch (GPU > NPU > Neural > CPU)
//! 2. `PCIe` P2P bypass for GPU↔NPU transfers
//! 3. Mixed-hardware pipeline routing across substrates
//! 4. NUCLEUS mesh coordination via `biomeOS` graph topology
//! 5. Local WGSL shader workloads route to GPU alongside absorbed workloads
//!
//! NUCLEUS atomics:
//! - **Tower**: `BearDog` (crypto) + `Songbird` (mesh discovery)
//! - **Node**: Tower + `ToadStool` (compute/GPU dispatch)
//! - **Nest**: Tower + `NestGate` (storage/provenance)

use airspring_forge::dispatch::{self, Workload};
use airspring_forge::nucleus::{AtomicKind, NucleusAtomic, NucleusMesh};
use airspring_forge::pipeline::{self, TransferPath};
use airspring_forge::substrate::{Capability, Identity, Properties, Substrate, SubstrateKind};
use airspring_forge::workloads;

#[expect(
    clippy::too_many_lines,
    reason = "validation binary sequentially checks many baseline comparisons"
)]
#[expect(
    clippy::similar_names,
    reason = "SPI validation uses py_* and rs_* prefixed variable names for Python vs Rust comparison"
)]
#[expect(
    clippy::or_fun_call,
    reason = "validation uses Option::or for default fallbacks"
)]
fn main() {
    println!("═══ Exp 076: NUCLEUS Mixed-Hardware Pipeline Routing ═══\n");

    let mut pass = 0_u32;
    let mut fail = 0_u32;

    macro_rules! check {
        ($name:expr, $cond:expr, $detail:expr) => {
            if $cond {
                println!("  PASS  {}: {}", $name, $detail);
                pass += 1;
            } else {
                println!("  FAIL  {}: {}", $name, $detail);
                fail += 1;
            }
        };
    }

    // ── Substrate inventory ──

    let gpu = Substrate {
        kind: SubstrateKind::Gpu,
        identity: Identity {
            pci_id: Some("10de:1db4".to_string()),
            ..Identity::named("TITAN V")
        },
        properties: Properties {
            has_f64: true,
            ..Properties::default()
        },
        capabilities: vec![
            Capability::F64Compute,
            Capability::F32Compute,
            Capability::ShaderDispatch,
            Capability::ScalarReduce,
            Capability::TimestampQuery,
        ],
    };

    let npu = Substrate {
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
    };

    let cpu = Substrate {
        kind: SubstrateKind::Cpu,
        identity: Identity::named("i9-12900K"),
        properties: Properties::default(),
        capabilities: vec![
            Capability::F64Compute,
            Capability::F32Compute,
            Capability::CpuCompute,
            Capability::SimdVector,
        ],
    };

    let substrates = [gpu.clone(), npu.clone(), cpu.clone()];

    println!("── 1. Workload Inventory ──\n");

    let all = workloads::all_workloads();
    let (absorbed, local, npu_native, cpu_only) = workloads::origin_summary();
    check!(
        "workload_count",
        all.len() == 27,
        format!("{} workloads", all.len())
    );
    check!(
        "absorbed_count",
        absorbed == 14,
        format!("{absorbed} absorbed by BarraCuda")
    );
    check!(
        "local_count",
        local == 6,
        format!("{local} local WGSL shaders")
    );
    check!(
        "npu_count",
        npu_native == 4,
        format!("{npu_native} NPU-native classifiers")
    );
    check!("cpu_count", cpu_only == 3, format!("{cpu_only} CPU-only"));

    println!("\n── 2. Capability-Based Dispatch ──\n");

    let mut gpu_routed = 0_u32;
    let mut npu_routed = 0_u32;
    let mut cpu_routed = 0_u32;

    for w in &all {
        let decision = dispatch::route(&w.workload, &substrates);
        if let Some(d) = &decision {
            match d.substrate.kind {
                SubstrateKind::Gpu => gpu_routed += 1,
                SubstrateKind::Npu => npu_routed += 1,
                SubstrateKind::Cpu => cpu_routed += 1,
                SubstrateKind::Neural => {}
            }
        }
        check!(
            &format!("route_{}", w.workload.name),
            decision.is_some(),
            format!(
                "→ {:?}",
                decision
                    .as_ref()
                    .map_or("NONE".to_string(), |d| format!("{:?}", d.substrate.kind))
            )
        );
    }

    check!(
        "gpu_workloads",
        gpu_routed >= 20,
        format!("{gpu_routed}/27 → GPU (absorbed + local)")
    );
    check!(
        "npu_workloads",
        npu_routed == 4,
        format!("{npu_routed}/27 → NPU")
    );
    check!(
        "cpu_workloads",
        cpu_routed <= 3,
        format!("{cpu_routed}/27 → CPU (fallback)")
    );

    println!("\n── 3. PCIe P2P Bypass ──\n");

    let npu_to_gpu = pipeline::transfer_path(&npu, &gpu);
    check!(
        "npu_gpu_pcie",
        npu_to_gpu == TransferPath::PciePeerToPeer,
        format!("NPU→GPU: {npu_to_gpu:?}")
    );
    check!(
        "pcie_bypasses_cpu",
        npu_to_gpu.bypasses_cpu(),
        "PCIe P2P bypasses CPU memory"
    );

    let gpu_to_cpu = pipeline::transfer_path(&gpu, &cpu);
    check!(
        "gpu_cpu_memcpy",
        gpu_to_cpu == TransferPath::CpuMemcpy,
        format!("GPU→CPU: {gpu_to_cpu:?}")
    );

    let gpu_to_gpu = pipeline::transfer_path(&gpu, &gpu);
    check!(
        "gpu_gpu_none",
        gpu_to_gpu == TransferPath::None,
        format!("GPU→GPU: {gpu_to_gpu:?}")
    );

    println!("\n── 4. Mixed-Hardware Pipeline ──\n");

    let seven_stage = [
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
            "scs_cn_local",
            vec![Capability::F64Compute, Capability::ShaderDispatch],
        ),
        Workload::new(
            "water_balance",
            vec![Capability::F64Compute, Capability::ShaderDispatch],
        ),
        Workload::new(
            "stewart_yield_local",
            vec![Capability::F64Compute, Capability::ShaderDispatch],
        ),
        Workload::new(
            "makkink_local",
            vec![Capability::F64Compute, Capability::ShaderDispatch],
        ),
        Workload::new("validation", vec![Capability::F64Compute]).prefer(SubstrateKind::Cpu),
    ];

    let pipeline = pipeline::route_pipeline(&seven_stage, &substrates).expect("pipeline");
    check!(
        "pipeline_stages",
        pipeline.stages.len() == 7,
        format!("{} stages", pipeline.stages.len())
    );
    check!(
        "pipeline_npu_start",
        pipeline.stages[0].substrate.kind == SubstrateKind::Npu,
        "stage 0 → NPU"
    );
    check!(
        "pipeline_pcie_bypass",
        pipeline.pcie_bypasses >= 1,
        format!("{} PCIe bypasses", pipeline.pcie_bypasses)
    );
    check!(
        "pipeline_cpu_end",
        pipeline.stages[6].substrate.kind == SubstrateKind::Cpu,
        "stage 6 → CPU (validation)"
    );
    check!(
        "pipeline_gpu_middle",
        pipeline.stages[1].substrate.kind == SubstrateKind::Gpu
            && pipeline.stages[2].substrate.kind == SubstrateKind::Gpu
            && pipeline.stages[3].substrate.kind == SubstrateKind::Gpu
            && pipeline.stages[4].substrate.kind == SubstrateKind::Gpu
            && pipeline.stages[5].substrate.kind == SubstrateKind::Gpu,
        "stages 1-5 → GPU (absorbed + local)"
    );

    println!("\n── 5. NUCLEUS Mesh Routing ──\n");

    let mut mesh = NucleusMesh::new();
    mesh.register(NucleusAtomic::new(
        AtomicKind::Tower,
        "tower-eastgate",
        vec![],
    ));
    mesh.register(NucleusAtomic::new(
        AtomicKind::Node,
        "node-eastgate",
        vec![gpu.clone(), npu.clone(), cpu.clone()],
    ));
    mesh.register(NucleusAtomic::new(
        AtomicKind::Nest,
        "nest-eastgate",
        vec![],
    ));

    check!(
        "mesh_tower",
        mesh.count_by_kind(AtomicKind::Tower) == 1,
        "1 Tower atomic"
    );
    check!(
        "mesh_node",
        mesh.count_by_kind(AtomicKind::Node) == 1,
        "1 Node atomic"
    );
    check!(
        "mesh_nest",
        mesh.count_by_kind(AtomicKind::Nest) == 1,
        "1 Nest atomic"
    );

    let mesh_pipeline = mesh.route_pipeline(&seven_stage).expect("mesh pipeline");
    check!(
        "mesh_single_node",
        mesh_pipeline.is_single_node(),
        "all stages on eastgate"
    );
    check!(
        "mesh_7_stages",
        mesh_pipeline.stage_count() == 7,
        format!("{} mesh stages", mesh_pipeline.stage_count())
    );
    check!(
        "mesh_no_cross_hops",
        mesh_pipeline.cross_node_hops == 0,
        "0 cross-node hops"
    );

    println!("\n── 6. Multi-Node Mesh ──\n");

    let mut multi_mesh = NucleusMesh::new();
    multi_mesh.register(NucleusAtomic::new(
        AtomicKind::Node,
        "gpu-node",
        vec![gpu, cpu.clone()],
    ));
    multi_mesh.register(NucleusAtomic::new(
        AtomicKind::Node,
        "npu-node",
        vec![npu, cpu],
    ));

    let cross_pipeline = [
        Workload::new(
            "et0_batch",
            vec![Capability::F64Compute, Capability::ShaderDispatch],
        ),
        Workload::new(
            "crop_stress",
            vec![Capability::QuantizedInference { bits: 8 }],
        ),
    ];
    let mp = multi_mesh
        .route_pipeline(&cross_pipeline)
        .expect("cross-node");
    check!(
        "multi_node_cross_hop",
        mp.cross_node_hops == 1,
        format!("{} cross-node hop", mp.cross_node_hops)
    );
    check!(
        "multi_node_stage0_gpu",
        mp.stages[0].substrate_kind == SubstrateKind::Gpu,
        "stage 0 → gpu-node"
    );
    check!(
        "multi_node_stage1_npu",
        mp.stages[1].substrate_kind == SubstrateKind::Npu,
        "stage 1 → npu-node"
    );

    println!("\n── 7. Atomic Component Verification ──\n");

    check!(
        "tower_capabilities",
        AtomicKind::Tower.capabilities().len() == 2,
        "crypto.tls + mesh.discovery"
    );
    check!(
        "node_capabilities",
        AtomicKind::Node.capabilities().len() == 3,
        "crypto.tls + mesh.discovery + compute.dispatch"
    );
    check!(
        "nest_capabilities",
        AtomicKind::Nest.capabilities().len() == 3,
        "crypto.tls + mesh.discovery + storage.provenance"
    );
    check!(
        "node_has_compute",
        AtomicKind::Node.has_compute(),
        "Node dispatches compute"
    );
    check!(
        "nest_has_storage",
        AtomicKind::Nest.has_storage(),
        "Nest stores provenance"
    );
    check!(
        "tower_has_mesh",
        AtomicKind::Tower.has_mesh(),
        "Tower discovers mesh"
    );

    println!("\n── 8. Transfer Path Latency Ranking ──\n");

    check!(
        "latency_order",
        TransferPath::None.latency_rank() < TransferPath::PciePeerToPeer.latency_rank()
            && TransferPath::PciePeerToPeer.latency_rank() < TransferPath::CpuMemcpy.latency_rank()
            && TransferPath::CpuMemcpy.latency_rank() < TransferPath::NeuralApi.latency_rank(),
        "None < PCIe < CpuMemcpy < NeuralApi"
    );

    println!("\n══════════════════════════════════════════════════");
    println!("  PASS: {pass}  FAIL: {fail}  TOTAL: {}", pass + fail);
    if fail == 0 {
        println!("  ALL CHECKS PASSED");
    }
    println!("══════════════════════════════════════════════════");

    assert_eq!(fail, 0, "{fail} checks failed");
}
