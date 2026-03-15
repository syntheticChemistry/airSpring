// SPDX-License-Identifier: AGPL-3.0-or-later
#![warn(clippy::pedantic)]
//! Exp 044: metalForge Live Hardware Probe + Dispatch.
//!
//! Probes actual hardware on this machine and validates dispatch routing
//! against the live inventory. Capability-based: only asserts on hardware
//! that is actually discovered at runtime. Primal code has self-knowledge
//! and discovers other substrates dynamically.

use airspring_forge::dispatch::{self, Reason};
use airspring_forge::probe;
use airspring_forge::substrate::{Capability, SubstrateKind};
use airspring_forge::workloads;
use barracuda::validation::ValidationHarness;

#[expect(
    clippy::too_many_lines,
    reason = "validation binary sequentially checks many baseline comparisons"
)]
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

    let has_npu = !npus.is_empty();
    if has_npu {
        v.check_bool("NPU discovered", true);
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
    } else {
        println!("  INFO: No NPU hardware detected — NPU checks skipped (capability-based)");
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
    let mut routed_count = 0_usize;
    let mut npu_required_count = 0_usize;
    for ew in &all_wl {
        let r = dispatch::route(&ew.workload, &inventory);
        if let Some(decision) = r {
            routed_count += 1;
            println!(
                "    {} → {} ({:?})",
                ew.workload.name, decision.substrate.identity.name, decision.reason
            );
        } else {
            let is_npu_only = ew
                .workload
                .preferred_substrate
                .as_ref()
                .is_some_and(|k| *k == SubstrateKind::Npu);
            if is_npu_only && !has_npu {
                npu_required_count += 1;
                println!(
                    "    {} → SKIP (NPU-preferred, no NPU hardware)",
                    ew.workload.name
                );
            } else {
                println!("    {} → NO ROUTE", ew.workload.name);
            }
        }
    }

    let expected_routable = all_wl.len() - npu_required_count;
    v.check_bool(
        "All routable workloads dispatched",
        routed_count >= expected_routable,
    );

    let et0_route = dispatch::route(&workloads::et0_batch().workload, &inventory);
    v.check_bool(
        "ET₀ batch routes to GPU (not CPU fallback)",
        et0_route
            .as_ref()
            .is_some_and(|d| d.substrate.kind == SubstrateKind::Gpu),
    );

    let stress_route = dispatch::route(&workloads::crop_stress_classifier().workload, &inventory);
    if has_npu {
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
    } else {
        println!("  INFO: NPU dispatch checks skipped (no NPU hardware)");
        if let Some(ref decision) = stress_route {
            println!(
                "    Crop stress fallback → {} ({:?})",
                decision.substrate.identity.name, decision.reason
            );
        }
    }

    v.finish();
}
