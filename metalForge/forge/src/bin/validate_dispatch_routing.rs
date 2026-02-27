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

fn main() {
    println!("═══════════════════════════════════════════════════════════════");
    println!("  airSpring metalForge — Dispatch Routing Validation");
    println!("═══════════════════════════════════════════════════════════════");

    let mut pass = 0u32;
    let mut fail = 0u32;

    let substrates = validate_live_inventory(&mut pass, &mut fail);
    validate_synthetic_dispatch(&mut pass, &mut fail);
    validate_workload_catalog(&mut pass, &mut fail);
    validate_live_routing(&substrates, &mut pass, &mut fail);

    let total = pass + fail;
    println!("\n═══════════════════════════════════════════════════════════════");
    println!("  {pass}/{total} checks passed");
    if fail > 0 {
        println!("  *** {fail} FAILURES ***");
        std::process::exit(1);
    } else {
        println!("  All checks passed ✓");
    }
}

fn check(name: &str, ok: bool, pass: &mut u32, fail: &mut u32) {
    if ok {
        *pass += 1;
        println!("  [PASS] {name}");
    } else {
        *fail += 1;
        println!("  [FAIL] {name}");
    }
}

// ── Section 1: Live hardware inventory ──────────────────────────────

fn validate_live_inventory(pass: &mut u32, fail: &mut u32) -> Vec<Substrate> {
    println!("\n── Live Hardware Inventory ─────────────────────────────────");
    let substrates = inventory::discover();
    println!("  Discovered {} substrate(s):", substrates.len());
    for sub in &substrates {
        println!("    {sub}  [{caps}]", caps = sub.capability_summary());
    }

    let has_cpu = substrates.iter().any(|s| s.kind == SubstrateKind::Cpu);
    check("CPU always discovered", has_cpu, pass, fail);

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

#[allow(clippy::too_many_lines)]
fn validate_synthetic_dispatch(pass: &mut u32, fail: &mut u32) {
    println!("\n── Synthetic Dispatch Tests ────────────────────────────────");

    let full_inv = [make_test_cpu(), make_test_gpu(), make_test_npu()];

    // GPU workloads
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
        let decision = dispatch::route(&work, &full_inv);
        check(
            &format!("{name} → GPU"),
            decision
                .as_ref()
                .is_some_and(|d| d.substrate.kind == SubstrateKind::Gpu),
            pass,
            fail,
        );
    }

    // NPU workloads
    let npu_workloads = [
        "crop stress classifier",
        "irrigation decision",
        "sensor anomaly",
    ];

    for name in &npu_workloads {
        let work = Workload::new(*name, vec![Capability::QuantizedInference { bits: 8 }]);
        let decision = dispatch::route(&work, &full_inv);
        check(
            &format!("{name} → NPU"),
            decision
                .as_ref()
                .is_some_and(|d| d.substrate.kind == SubstrateKind::Npu),
            pass,
            fail,
        );
    }

    // NPU preference
    let prefer_work = Workload::new(
        "crop stress (preferred NPU)",
        vec![Capability::QuantizedInference { bits: 8 }],
    )
    .prefer(SubstrateKind::Npu);
    let decision = dispatch::route(&prefer_work, &full_inv);
    check(
        "NPU preference honored",
        decision.as_ref().is_some_and(|d| {
            d.substrate.kind == SubstrateKind::Npu && d.reason == Reason::Preferred
        }),
        pass,
        fail,
    );

    // CPU fallback
    let cpu_inv = [make_test_cpu()];
    let f64_work = Workload::new("f64 fallback", vec![Capability::F64Compute]);
    let decision = dispatch::route(&f64_work, &cpu_inv);
    check(
        "f64 work falls to CPU when no GPU",
        decision
            .as_ref()
            .is_some_and(|d| d.substrate.kind == SubstrateKind::Cpu),
        pass,
        fail,
    );

    // No route when requirements unmet
    let quant_work = Workload::new(
        "int4 inference",
        vec![Capability::QuantizedInference { bits: 4 }],
    );
    let decision = dispatch::route(&quant_work, &cpu_inv);
    check(
        "quant(4) unroutable on CPU-only",
        decision.is_none(),
        pass,
        fail,
    );

    // CPU preference overrides GPU
    let cpu_pref_work =
        Workload::new("validation", vec![Capability::F64Compute]).prefer(SubstrateKind::Cpu);
    let decision = dispatch::route(&cpu_pref_work, &full_inv);
    check(
        "CPU preference honored over GPU",
        decision.as_ref().is_some_and(|d| {
            d.substrate.kind == SubstrateKind::Cpu && d.reason == Reason::Preferred
        }),
        pass,
        fail,
    );
}

// ── Section 3: Workload catalog ─────────────────────────────────────

fn validate_workload_catalog(pass: &mut u32, fail: &mut u32) {
    println!("\n── Workload Catalog ────────────────────────────────────────");
    let all = workloads::all_workloads();
    let (absorbed, local, npu_native, cpu_only_ct) = workloads::origin_summary();
    println!("  Total workloads: {}", all.len());
    println!(
        "  Absorbed: {absorbed}, Local: {local}, NPU-native: {npu_native}, CPU-only: {cpu_only_ct}"
    );

    check("14 total workloads", all.len() == 14, pass, fail);
    check("9 absorbed GPU", absorbed == 9, pass, fail);
    check("0 local WGSL", local == 0, pass, fail);
    check("3 NPU-native classifiers", npu_native == 3, pass, fail);
    check("2 CPU-only domains", cpu_only_ct == 2, pass, fail);

    let npu_prefer_ok = all
        .iter()
        .filter(|w| w.is_npu_native())
        .all(|w| w.workload.preferred_substrate == Some(SubstrateKind::Npu));
    check("all NPU workloads prefer NPU", npu_prefer_ok, pass, fail);

    let prim_ok = all
        .iter()
        .filter(|w| w.is_absorbed())
        .all(|w| w.primitive.is_some());
    check("all absorbed have primitive name", prim_ok, pass, fail);

    let mut names: Vec<&str> = all.iter().map(|w| w.workload.name.as_str()).collect();
    names.sort_unstable();
    let pre = names.len();
    names.dedup();
    check(
        "no duplicate workload names",
        pre == names.len(),
        pass,
        fail,
    );
}

// ── Section 4: Live routing ─────────────────────────────────────────

fn validate_live_routing(substrates: &[Substrate], pass: &mut u32, fail: &mut u32) {
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
    check(
        "live f64 work routes somewhere",
        decision.is_some(),
        pass,
        fail,
    );
}
