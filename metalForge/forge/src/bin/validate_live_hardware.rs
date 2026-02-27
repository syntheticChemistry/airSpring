// SPDX-License-Identifier: AGPL-3.0-or-later
#![warn(clippy::pedantic)]
#![allow(clippy::too_many_lines)]
//! Exp 044: metalForge Live Hardware Probe + Dispatch.
//!
//! Probes actual hardware on this machine and validates dispatch routing
//! against the live inventory. Expects:
//! - 2 GPUs (RTX 4070 + TITAN V)
//! - 1 NPU (`BrainChip` AKD1000 at `/dev/akida0`)
//! - 1 CPU (i9-12900K)

use airspring_forge::dispatch::{self, Reason};
use airspring_forge::probe;
use airspring_forge::substrate::{Capability, SubstrateKind};
use airspring_forge::workloads;

fn main() {
    println!("═══════════════════════════════════════════════════════════");
    println!("  airSpring Exp 044: metalForge Live Hardware Probe");
    println!("═══════════════════════════════════════════════════════════\n");

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

    // ── GPU Probe ───────────────────────────────────────────────────────

    println!("── GPU Probe (wgpu) ──");
    let gpus = probe::probe_gpus();
    println!("  Discovered {} GPU adapter(s)", gpus.len());
    for gpu in &gpus {
        println!(
            "    {} [{}] f64={} timestamps={}",
            gpu.identity.name,
            gpu.identity.backend.as_deref().unwrap_or("?"),
            gpu.properties.has_f64,
            gpu.properties.has_timestamps,
        );
        if let Some(ref driver) = gpu.identity.driver {
            println!("      driver: {driver}");
        }
    }

    check(
        "At least 1 GPU discovered",
        !gpus.is_empty(),
        &mut pass,
        &mut fail,
    );

    let titan_v = gpus
        .iter()
        .find(|g| g.identity.name.to_lowercase().contains("titan"));
    check(
        "TITAN V found in GPU inventory",
        titan_v.is_some(),
        &mut pass,
        &mut fail,
    );

    if let Some(tv) = titan_v {
        check(
            "TITAN V has F64Compute",
            tv.has(&Capability::F64Compute),
            &mut pass,
            &mut fail,
        );
        check(
            "TITAN V has ShaderDispatch",
            tv.has(&Capability::ShaderDispatch),
            &mut pass,
            &mut fail,
        );
    } else {
        println!("  WARN: TITAN V not found — skipping TITAN V-specific checks");
    }

    let rtx_4070 = gpus.iter().find(|g| {
        g.identity.name.to_lowercase().contains("4070")
            || g.identity.name.to_lowercase().contains("rtx")
    });
    check(
        "RTX 4070 found in GPU inventory",
        rtx_4070.is_some(),
        &mut pass,
        &mut fail,
    );

    if let Some(rtx) = rtx_4070 {
        check(
            "RTX 4070 has F64Compute",
            rtx.has(&Capability::F64Compute),
            &mut pass,
            &mut fail,
        );
    }

    // ── NPU Probe ───────────────────────────────────────────────────────

    println!("\n── NPU Probe (/dev/akida*) ──");
    let npus = probe::probe_npus();
    println!("  Discovered {} NPU device(s)", npus.len());
    for npu in &npus {
        println!(
            "    {} at {}",
            npu.identity.name,
            npu.identity.device_node.as_deref().unwrap_or("?"),
        );
    }

    check(
        "At least 1 NPU discovered (/dev/akida0)",
        !npus.is_empty(),
        &mut pass,
        &mut fail,
    );

    if let Some(npu) = npus.first() {
        check(
            "AKD1000 has QuantizedInference(8)",
            npu.has(&Capability::QuantizedInference { bits: 8 }),
            &mut pass,
            &mut fail,
        );
        check(
            "AKD1000 has BatchInference",
            npu.has(&Capability::BatchInference { max_batch: 8 }),
            &mut pass,
            &mut fail,
        );
        check(
            "AKD1000 has WeightMutation",
            npu.has(&Capability::WeightMutation),
            &mut pass,
            &mut fail,
        );
    }

    // ── CPU Probe ───────────────────────────────────────────────────────

    println!("\n── CPU Probe (/proc) ──");
    let cpu = probe::probe_cpu();
    println!("  {}", cpu.identity.name);
    if let Some(cores) = cpu.properties.core_count {
        println!("  Cores: {cores}");
    }
    if let Some(threads) = cpu.properties.thread_count {
        println!("  Threads: {threads}");
    }
    if let Some(mem) = cpu.properties.memory_bytes {
        println!("  Memory: {} GB", mem / (1024 * 1024 * 1024));
    }

    check(
        "CPU has F64Compute",
        cpu.has(&Capability::F64Compute),
        &mut pass,
        &mut fail,
    );
    check(
        "CPU has CpuCompute",
        cpu.has(&Capability::CpuCompute),
        &mut pass,
        &mut fail,
    );
    check(
        "CPU has SimdVector (AVX2)",
        cpu.has(&Capability::SimdVector),
        &mut pass,
        &mut fail,
    );

    // ── Full Inventory Dispatch ─────────────────────────────────────────

    println!("\n── Live Dispatch (full inventory) ──");
    let mut inventory = gpus;
    inventory.extend(npus);
    inventory.push(cpu);
    println!("  Total substrates: {}", inventory.len());

    let all_wl = workloads::all_workloads();
    let mut all_route = true;
    for ew in &all_wl {
        let r = dispatch::route(&ew.workload, &inventory);
        let routed = r.is_some();
        if !routed {
            all_route = false;
        }
        if let Some(decision) = r {
            println!(
                "    {} → {} ({:?})",
                ew.workload.name, decision.substrate.identity.name, decision.reason
            );
        } else {
            println!("    {} → NO ROUTE", ew.workload.name);
        }
    }

    check(
        "All 14 workloads route with live hardware",
        all_route,
        &mut pass,
        &mut fail,
    );

    // Verify ET₀ routes to a real GPU (not CPU)
    let et0_route = dispatch::route(&workloads::et0_batch().workload, &inventory);
    check(
        "ET₀ batch routes to GPU (not CPU fallback)",
        et0_route
            .as_ref()
            .is_some_and(|d| d.substrate.kind == SubstrateKind::Gpu),
        &mut pass,
        &mut fail,
    );

    // Verify NPU workloads route to live AKD1000
    let stress_route = dispatch::route(&workloads::crop_stress_classifier().workload, &inventory);
    check(
        "Crop stress routes to live AKD1000 NPU",
        stress_route
            .as_ref()
            .is_some_and(|d| d.substrate.kind == SubstrateKind::Npu),
        &mut pass,
        &mut fail,
    );
    check(
        "NPU route uses Preferred reason",
        stress_route
            .as_ref()
            .is_some_and(|d| d.reason == Reason::Preferred),
        &mut pass,
        &mut fail,
    );

    // ── Summary ─────────────────────────────────────────────────────────

    let total = pass + fail;
    println!("\n=== metalForge Live Hardware: {pass}/{total} PASS, {fail} FAIL ===");
    if fail > 0 {
        std::process::exit(1);
    }
}
