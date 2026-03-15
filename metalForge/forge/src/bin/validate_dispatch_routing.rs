// SPDX-License-Identifier: AGPL-3.0-or-later
//! Validates airSpring metalForge dispatch routing.
//!
//! Tests that eco workloads route correctly given the hardware inventory:
//! - GPU workloads (ET₀, water balance, Richards) route to GPU when available
//! - NPU workloads (crop stress, irrigation, anomaly) route to NPU when present
//! - Fallback behavior is correct (GPU > NPU > CPU)
//! - No workload is left unroutable with at least a CPU

use airspring_forge::dispatch::{self, Reason, Workload};
use airspring_forge::inventory;
use airspring_forge::substrate::{Capability, Identity, Properties, Substrate, SubstrateKind};
use airspring_forge::workloads;
use barracuda::validation::ValidationHarness;

fn main() {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .with_target(false)
        .without_time()
        .init();

    println!("═══════════════════════════════════════════════════════════════");
    println!("  airSpring metalForge — Dispatch Routing Validation");
    println!("═══════════════════════════════════════════════════════════════\n");

    let mut v = ValidationHarness::new("Dispatch Routing");

    let substrates = validate_live_inventory(&mut v);
    validate_synthetic_dispatch(&mut v);
    validate_workload_catalog(&mut v);
    validate_live_routing(&substrates, &mut v);

    v.finish();
}

// ── Section 1: Live hardware inventory ──────────────────────────────

fn validate_live_inventory(v: &mut ValidationHarness) -> Vec<Substrate> {
    println!("\n── Live Hardware Inventory ─────────────────────────────────");
    let substrates = inventory::discover();
    println!("  Discovered {} substrate(s):", substrates.len());
    for sub in &substrates {
        println!("    {sub}  [{caps}]", caps = sub.capability_summary());
    }

    let has_cpu = substrates.iter().any(|s| s.kind == SubstrateKind::Cpu);
    v.check_bool("CPU always discovered", has_cpu);

    let gpu_ct = substrates
        .iter()
        .filter(|s| s.kind == SubstrateKind::Gpu)
        .count();
    println!("  GPU(s): {gpu_ct}");

    let npu_ct = substrates
        .iter()
        .filter(|s| s.kind == SubstrateKind::Npu)
        .count();
    println!("  NPU(s): {npu_ct}");

    substrates
}

// ── Section 2: Synthetic dispatch tests ─────────────────────────────

fn make_test_gpu() -> Substrate {
    Substrate {
        kind: SubstrateKind::Gpu,
        identity: Identity::named("Synthetic GPU"),
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
    }
}

fn make_test_npu() -> Substrate {
    Substrate {
        kind: SubstrateKind::Npu,
        identity: Identity::named("Synthetic AKD1000"),
        properties: Properties::default(),
        capabilities: vec![
            Capability::F32Compute,
            Capability::QuantizedInference { bits: 8 },
            Capability::QuantizedInference { bits: 4 },
            Capability::BatchInference { max_batch: 8 },
            Capability::WeightMutation,
        ],
    }
}

fn make_test_cpu() -> Substrate {
    Substrate {
        kind: SubstrateKind::Cpu,
        identity: Identity::named("Synthetic CPU"),
        properties: Properties::default(),
        capabilities: vec![
            Capability::F64Compute,
            Capability::F32Compute,
            Capability::SimdVector,
        ],
    }
}

fn validate_synthetic_dispatch(v: &mut ValidationHarness) {
    println!("\n── Synthetic Dispatch Tests ────────────────────────────────");

    let full_inv = [make_test_cpu(), make_test_gpu(), make_test_npu()];
    let cpu_inv = [make_test_cpu()];

    check_gpu_workloads(v, &full_inv);
    check_npu_workloads(v, &full_inv);
    check_npu_preference(v, &full_inv);
    check_cpu_fallback(v, &cpu_inv);
    check_quant_unroutable(v, &cpu_inv);
    check_cpu_preference(v, &full_inv);
}

fn check_gpu_workloads(v: &mut ValidationHarness, inv: &[Substrate]) {
    let gpu_workloads = [
        (
            "ET₀ batch",
            vec![Capability::F64Compute, Capability::ShaderDispatch],
        ),
        (
            "water balance",
            vec![Capability::F64Compute, Capability::ShaderDispatch],
        ),
        (
            "Richards PDE",
            vec![Capability::F64Compute, Capability::ShaderDispatch],
        ),
        (
            "Monte Carlo UQ",
            vec![Capability::F64Compute, Capability::ScalarReduce],
        ),
    ];
    for (name, caps) in &gpu_workloads {
        let work = Workload::new(*name, caps.clone());
        let decision = dispatch::route(&work, inv);
        v.check_bool(
            &format!("{name} → GPU"),
            decision
                .as_ref()
                .is_some_and(|d| d.substrate.kind == SubstrateKind::Gpu),
        );
    }
}

fn check_npu_workloads(v: &mut ValidationHarness, inv: &[Substrate]) {
    let npu_workloads = [
        "crop stress classifier",
        "irrigation decision",
        "sensor anomaly",
    ];
    for name in &npu_workloads {
        let work = Workload::new(*name, vec![Capability::QuantizedInference { bits: 8 }]);
        let decision = dispatch::route(&work, inv);
        v.check_bool(
            &format!("{name} → NPU"),
            decision
                .as_ref()
                .is_some_and(|d| d.substrate.kind == SubstrateKind::Npu),
        );
    }
}

fn check_npu_preference(v: &mut ValidationHarness, inv: &[Substrate]) {
    let prefer_work = Workload::new(
        "crop stress (preferred NPU)",
        vec![Capability::QuantizedInference { bits: 8 }],
    )
    .prefer(SubstrateKind::Npu);
    let decision = dispatch::route(&prefer_work, inv);
    v.check_bool(
        "NPU preference honored",
        decision.as_ref().is_some_and(|d| {
            d.substrate.kind == SubstrateKind::Npu && d.reason == Reason::Preferred
        }),
    );
}

fn check_cpu_fallback(v: &mut ValidationHarness, inv: &[Substrate]) {
    let f64_work = Workload::new("f64 fallback", vec![Capability::F64Compute]);
    let decision = dispatch::route(&f64_work, inv);
    v.check_bool(
        "f64 work falls to CPU when no GPU",
        decision
            .as_ref()
            .is_some_and(|d| d.substrate.kind == SubstrateKind::Cpu),
    );
}

fn check_quant_unroutable(v: &mut ValidationHarness, inv: &[Substrate]) {
    let quant_work = Workload::new(
        "int4 inference",
        vec![Capability::QuantizedInference { bits: 4 }],
    );
    let decision = dispatch::route(&quant_work, inv);
    v.check_bool("quant(4) unroutable on CPU-only", decision.is_none());
}

fn check_cpu_preference(v: &mut ValidationHarness, inv: &[Substrate]) {
    let cpu_pref_work =
        Workload::new("validation", vec![Capability::F64Compute]).prefer(SubstrateKind::Cpu);
    let decision = dispatch::route(&cpu_pref_work, inv);
    v.check_bool(
        "CPU preference honored over GPU",
        decision.as_ref().is_some_and(|d| {
            d.substrate.kind == SubstrateKind::Cpu && d.reason == Reason::Preferred
        }),
    );
}

// ── Section 3: Workload catalog ─────────────────────────────────────

fn validate_workload_catalog(v: &mut ValidationHarness) {
    println!("\n── Workload Catalog ────────────────────────────────────────");
    let all = workloads::all_workloads();
    let (absorbed, local, npu_native, cpu_only_ct) = workloads::origin_summary();
    println!("  Total workloads: {}", all.len());
    println!(
        "  Absorbed: {absorbed}, Local: {local}, NPU-native: {npu_native}, CPU-only: {cpu_only_ct}"
    );

    v.check_bool("18 total workloads", all.len() == 18);
    v.check_bool("9 absorbed GPU", absorbed == 9);
    v.check_bool("4 local WGSL (Tier B)", local == 4);
    v.check_bool("3 NPU-native classifiers", npu_native == 3);
    v.check_bool("2 CPU-only domains", cpu_only_ct == 2);

    let npu_prefer_ok = all
        .iter()
        .filter(|w| w.is_npu_native())
        .all(|w| w.workload.preferred_substrate == Some(SubstrateKind::Npu));
    v.check_bool("all NPU workloads prefer NPU", npu_prefer_ok);

    let prim_ok = all
        .iter()
        .filter(|w| w.is_absorbed())
        .all(|w| w.primitive.is_some());
    v.check_bool("all absorbed have primitive name", prim_ok);

    let mut names: Vec<&str> = all.iter().map(|w| w.workload.name.as_str()).collect();
    names.sort_unstable();
    let pre = names.len();
    names.dedup();
    v.check_bool("no duplicate workload names", pre == names.len());
}

// ── Section 4: Live routing ─────────────────────────────────────────

fn validate_live_routing(substrates: &[Substrate], v: &mut ValidationHarness) {
    if substrates.is_empty() {
        return;
    }

    println!("\n── Live Routing (real hardware) ────────────────────────────");
    let all = workloads::all_workloads();
    for eco in &all {
        let decision = dispatch::route(&eco.workload, substrates);
        let target = decision
            .as_ref()
            .map_or_else(|| "NONE".to_string(), |d| format!("{}", d.substrate.kind));
        println!("  {:30} → {target}", eco.workload.name);
    }

    let validation = Workload::new("live validation", vec![Capability::F64Compute]);
    let decision = dispatch::route(&validation, substrates);
    v.check_bool("live f64 work routes somewhere", decision.is_some());
}
