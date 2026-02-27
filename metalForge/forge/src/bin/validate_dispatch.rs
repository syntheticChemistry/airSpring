// SPDX-License-Identifier: AGPL-3.0-or-later
#![warn(clippy::pedantic)]
//! Exp 041: metalForge Mixed-Hardware Dispatch Validation.
//!
//! Tests actual Rust `dispatch::route()` logic with synthetic inventories.
//! Validates GPU > NPU > Neural > CPU priority, capability matching, and fallback.
//!
//! Benchmark: `control/metalforge_dispatch/benchmark_metalforge_dispatch.json`
//! Baseline: `control/metalforge_dispatch/metalforge_dispatch.py` (14/14 PASS)

use airspring_forge::dispatch::{self, Reason};
use airspring_forge::substrate::{Capability, Identity, Properties, Substrate, SubstrateKind};
use airspring_forge::workloads;

fn make_sub(kind: SubstrateKind, name: &str, caps: Vec<Capability>) -> Substrate {
    Substrate {
        kind,
        identity: Identity::named(name),
        properties: Properties::default(),
        capabilities: caps,
    }
}

fn full_inventory() -> Vec<Substrate> {
    vec![
        make_sub(
            SubstrateKind::Gpu,
            "Titan V GPU",
            vec![
                Capability::F64Compute,
                Capability::F32Compute,
                Capability::ShaderDispatch,
                Capability::ScalarReduce,
                Capability::TimestampQuery,
            ],
        ),
        make_sub(
            SubstrateKind::Npu,
            "AKD1000 NPU",
            vec![
                Capability::F32Compute,
                Capability::QuantizedInference { bits: 8 },
                Capability::BatchInference { max_batch: 8 },
            ],
        ),
        make_sub(
            SubstrateKind::Neural,
            "biomeOS Neural API",
            vec![
                Capability::F64Compute,
                Capability::F32Compute,
                Capability::NeuralApiRoute,
            ],
        ),
        make_sub(
            SubstrateKind::Cpu,
            "AMD Ryzen 7",
            vec![
                Capability::F64Compute,
                Capability::F32Compute,
                Capability::CpuCompute,
                Capability::SimdVector,
            ],
        ),
    ]
}

fn main() {
    println!("═══════════════════════════════════════════════════════════");
    println!("  airSpring Exp 041: metalForge Mixed-Hardware Dispatch");
    println!("═══════════════════════════════════════════════════════════\n");

    let inv = full_inventory();
    let mut pass = 0_u32;
    let mut fail = 0_u32;

    let check = |name: &str, ok: bool, pass: &mut u32, fail: &mut u32| {
        if ok {
            *pass += 1;
            println!("  [PASS] {name}");
        } else {
            *fail += 1;
            println!("  [FAIL] {name}");
        }
    };

    // ── GPU workload routing ────────────────────────────────────────────

    println!("── GPU Workload Routing ──");

    let gpu_wls = [
        workloads::et0_batch(),
        workloads::water_balance_batch(),
        workloads::richards_pde(),
        workloads::yield_response_surface(),
        workloads::monte_carlo_uq(),
        workloads::isotherm_batch(),
    ];
    for ew in &gpu_wls {
        let r = dispatch::route(&ew.workload, &inv);
        check(
            &format!("{} → GPU", ew.workload.name),
            r.as_ref()
                .is_some_and(|d| d.substrate.kind == SubstrateKind::Gpu),
            &mut pass,
            &mut fail,
        );
    }

    // ── NPU workload routing ────────────────────────────────────────────

    println!("\n── NPU Workload Routing ──");

    let npu_wls = [
        workloads::crop_stress_classifier(),
        workloads::irrigation_decision(),
        workloads::sensor_anomaly(),
    ];
    for ew in &npu_wls {
        let r = dispatch::route(&ew.workload, &inv);
        check(
            &format!("{} → NPU", ew.workload.name),
            r.as_ref()
                .is_some_and(|d| d.substrate.kind == SubstrateKind::Npu),
            &mut pass,
            &mut fail,
        );
    }

    // ── CPU workload routing ────────────────────────────────────────────

    println!("\n── CPU Workload Routing ──");

    let cpu_wls = [workloads::weather_ingest()];
    for ew in &cpu_wls {
        let r = dispatch::route(&ew.workload, &inv);
        check(
            &format!("{} → CPU", ew.workload.name),
            r.as_ref()
                .is_some_and(|d| d.substrate.kind == SubstrateKind::Cpu),
            &mut pass,
            &mut fail,
        );
    }

    // ── Priority chain ──────────────────────────────────────────────────

    println!("\n── Priority Chain ──");

    let r = dispatch::route(&workloads::et0_batch().workload, &inv);
    check(
        "GPU preferred over Neural for F64 workloads",
        r.as_ref()
            .is_some_and(|d| d.substrate.kind == SubstrateKind::Gpu),
        &mut pass,
        &mut fail,
    );

    // ── Fallback behavior ───────────────────────────────────────────────

    println!("\n── Fallback Behavior ──");

    let cpu_only = vec![make_sub(
        SubstrateKind::Cpu,
        "CPU only",
        vec![
            Capability::F64Compute,
            Capability::CpuCompute,
            Capability::ShaderDispatch,
        ],
    )];
    let r = dispatch::route(&workloads::et0_batch().workload, &cpu_only);
    check(
        "ET₀ batch falls back to CPU when no GPU",
        r.as_ref()
            .is_some_and(|d| d.substrate.kind == SubstrateKind::Cpu),
        &mut pass,
        &mut fail,
    );

    let gpu_cpu_only = vec![
        make_sub(
            SubstrateKind::Gpu,
            "GPU",
            vec![Capability::F64Compute, Capability::ShaderDispatch],
        ),
        make_sub(
            SubstrateKind::Cpu,
            "CPU",
            vec![Capability::F64Compute, Capability::CpuCompute],
        ),
    ];
    let r = dispatch::route(&workloads::crop_stress_classifier().workload, &gpu_cpu_only);
    check(
        "NPU workload fails when no NPU available",
        r.is_none(),
        &mut pass,
        &mut fail,
    );

    // ── Reason tracking ─────────────────────────────────────────────────

    println!("\n── Dispatch Reason ──");

    let r = dispatch::route(&workloads::crop_stress_classifier().workload, &inv)
        .expect("should route");
    check(
        "NPU workload reports Preferred reason",
        r.reason == Reason::Preferred,
        &mut pass,
        &mut fail,
    );

    let r = dispatch::route(&workloads::et0_batch().workload, &inv)
        .expect("should route");
    check(
        "GPU workload reports BestAvailable reason",
        r.reason == Reason::BestAvailable,
        &mut pass,
        &mut fail,
    );

    // ── Inventory completeness ──────────────────────────────────────────

    println!("\n── Inventory Completeness ──");

    let all = workloads::all_workloads();
    check(
        &format!("{} workloads in catalog", all.len()),
        all.len() == 14,
        &mut pass,
        &mut fail,
    );

    let all_route = all
        .iter()
        .all(|ew| dispatch::route(&ew.workload, &inv).is_some());
    check(
        "All 14 workloads route in full inventory",
        all_route,
        &mut pass,
        &mut fail,
    );

    let (absorbed, _, npu_native, cpu_only) = workloads::origin_summary();
    check(
        &format!("9 absorbed + 3 NPU + 2 CPU = {}", absorbed + npu_native + cpu_only),
        absorbed == 9 && npu_native == 3 && cpu_only == 2,
        &mut pass,
        &mut fail,
    );

    // ── Summary ─────────────────────────────────────────────────────────

    let total = pass + fail;
    println!("\n=== metalForge Dispatch: {pass}/{total} PASS, {fail} FAIL ===");
    if fail > 0 {
        std::process::exit(1);
    }
}
