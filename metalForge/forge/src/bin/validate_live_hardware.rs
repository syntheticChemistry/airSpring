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
use barracuda::validation::ValidationHarness;

fn main() {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .with_target(false)
        .without_time()
        .init();

    println!("═══════════════════════════════════════════════════════════");
    println!("  airSpring Exp 044: metalForge Live Hardware Probe");
    println!("═══════════════════════════════════════════════════════════\n");

    let mut v = ValidationHarness::new("Live Hardware");

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

    v.check_bool("At least 1 GPU discovered", !gpus.is_empty());

    let titan_v = gpus
        .iter()
        .find(|g| g.identity.name.to_lowercase().contains("titan"));
    v.check_bool("TITAN V found in GPU inventory", titan_v.is_some());

    if let Some(tv) = titan_v {
        v.check_bool("TITAN V has F64Compute", tv.has(&Capability::F64Compute));
        v.check_bool(
            "TITAN V has ShaderDispatch",
            tv.has(&Capability::ShaderDispatch),
        );
    } else {
        println!("  WARN: TITAN V not found — skipping TITAN V-specific checks");
    }

    let rtx_4070 = gpus.iter().find(|g| {
        g.identity.name.to_lowercase().contains("4070")
            || g.identity.name.to_lowercase().contains("rtx")
    });
    v.check_bool("RTX 4070 found in GPU inventory", rtx_4070.is_some());

    if let Some(rtx) = rtx_4070 {
        v.check_bool("RTX 4070 has F64Compute", rtx.has(&Capability::F64Compute));
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

    v.check_bool("At least 1 NPU discovered (/dev/akida0)", !npus.is_empty());

    if let Some(npu) = npus.first() {
        v.check_bool(
            "AKD1000 has QuantizedInference(8)",
            npu.has(&Capability::QuantizedInference { bits: 8 }),
        );
        v.check_bool(
            "AKD1000 has BatchInference",
            npu.has(&Capability::BatchInference { max_batch: 8 }),
        );
        v.check_bool(
            "AKD1000 has WeightMutation",
            npu.has(&Capability::WeightMutation),
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

    v.check_bool("CPU has F64Compute", cpu.has(&Capability::F64Compute));
    v.check_bool("CPU has CpuCompute", cpu.has(&Capability::CpuCompute));
    v.check_bool(
        "CPU has SimdVector (AVX2)",
        cpu.has(&Capability::SimdVector),
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

    v.check_bool("All 14 workloads route with live hardware", all_route);

    let et0_route = dispatch::route(&workloads::et0_batch().workload, &inventory);
    v.check_bool(
        "ET₀ batch routes to GPU (not CPU fallback)",
        et0_route
            .as_ref()
            .is_some_and(|d| d.substrate.kind == SubstrateKind::Gpu),
    );

    let stress_route = dispatch::route(&workloads::crop_stress_classifier().workload, &inventory);
    v.check_bool(
        "Crop stress routes to live AKD1000 NPU",
        stress_route
            .as_ref()
            .is_some_and(|d| d.substrate.kind == SubstrateKind::Npu),
    );
    v.check_bool(
        "NPU route uses Preferred reason",
        stress_route
            .as_ref()
            .is_some_and(|d| d.reason == Reason::Preferred),
    );

    v.finish();
}
