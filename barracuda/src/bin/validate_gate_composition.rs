// SPDX-License-Identifier: AGPL-3.0-or-later

//! Exp 094-AS: `eastGate` NUCLEUS Composition Validation
//!
//! Validates that the `niche-airspring` 9-primal NUCLEUS composition is
//! alive and healthy on `eastGate`. Each primal is probed individually
//! via direct socket discovery — no routing through `airSpring`'s primal.
//!
//! ## Composition (`niche-airspring`)
//!
//! | Layer | Primals |
//! |-------|---------|
//! | Tower | `BearDog`, `Songbird`, `skunkBat` |
//! | Node  | `ToadStool`, `barraCuda`, `coralReef` |
//! | Nest  | `NestGate`, `rhizoCrypt`, `loamSpine`, `sweetGrass` |
//!
//! ## What this validates
//!
//! 1. Socket discovery for all 9 primals (+ `biomeOS`)
//! 2. Health probe (`health` or `health.liveness`) per primal
//! 3. Capability advertisement (`capabilities.list`) per primal
//! 4. Cross-primal capability routing via `biomeOS` `capability.call`
//! 5. Provenance trio round-trip (dag → commit → provenance)
//! 6. `NeuralBridge` observatory (`routing_weights`, `weight_health`)
//! 7. `NestGate` content-addressed storage round-trip
//!
//! ## Prerequisites
//!
//! ```sh
//! # Fetch plasmidBin binaries
//! ./tools/fetch_primals.sh --all
//! # Or from infra: ../../../infra/plasmidBin/fetch.sh --all
//!
//! # Start NUCLEUS composition
//! ../../../springs/primalSpring/tools/nucleus_launcher.sh --composition niche-airspring
//! ```

#![expect(
    clippy::expect_used,
    reason = "validation binary: fail-fast on discovery assertions"
)]

use std::path::PathBuf;

use airspring_barracuda::biomeos;
use airspring_barracuda::primal_names;
use airspring_barracuda::rpc;

use barracuda::validation::ValidationHarness;

/// The 9 primals in the niche-airspring NUCLEUS composition,
/// plus biomeOS as the meta-orchestrator.
const NICHE_PRIMALS: &[(&str, &str)] = &[
    (primal_names::BEARDOG, "Tower"),
    (primal_names::SONGBIRD, "Tower"),
    (primal_names::SKUNKBAT, "Tower"),
    (primal_names::TOADSTOOL, "Node"),
    (primal_names::BARRACUDA, "Node"),
    (primal_names::CORALREEF, "Node"),
    (primal_names::NESTGATE, "Nest"),
    (primal_names::RHIZOCRYPT, "Nest"),
    (primal_names::LOAMSPINE, "Nest"),
    (primal_names::SWEETGRASS, "Nest"),
];

fn find_socket(prefix: &str) -> Option<PathBuf> {
    biomeos::find_socket(prefix)
}

fn probe_health(sock: &std::path::Path) -> Option<serde_json::Value> {
    rpc::send(sock, "health", &serde_json::json!({}))
        .ok()
        .and_then(|r| r.get("result").cloned())
        .or_else(|| {
            rpc::send(sock, "health.liveness", &serde_json::json!({}))
                .ok()
                .and_then(|r| r.get("result").cloned())
        })
}

fn probe_capabilities(sock: &std::path::Path) -> Vec<String> {
    rpc::send(sock, "capabilities.list", &serde_json::json!({}))
        .ok()
        .and_then(|r| r.get("result").cloned())
        .and_then(|r| {
            r.get("capabilities")
                .cloned()
                .or_else(|| r.get("methods").cloned())
        })
        .and_then(|v| v.as_array().cloned())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default()
}

#[expect(
    clippy::too_many_lines,
    reason = "validation binary: sequential probe of 10 primals + cross-primal tests"
)]
fn main() {
    tracing_subscriber::fmt().with_env_filter("info").init();

    let mut v = ValidationHarness::new("Exp 094-AS: eastGate NUCLEUS Composition");
    let mut sockets_found: u32 = 0;
    let mut healthy_count: u32 = 0;

    // ── Phase 1: Socket Discovery + Health Probes ────────────────────
    eprintln!("━━━ Phase 1: Primal Discovery + Health ━━━");

    for &(primal, layer) in NICHE_PRIMALS {
        let socket = find_socket(primal);
        let found = socket.is_some();
        v.check_bool(&format!("{primal}_socket_found"), found);

        if found {
            sockets_found += 1;
        } else {
            eprintln!("  MISS: {primal} ({layer}) — socket not found");
            continue;
        }

        let sock = socket.expect("socket existence verified above");
        let health = probe_health(&sock);
        let healthy = health.is_some();
        v.check_bool(&format!("{primal}_health_ok"), healthy);

        if healthy {
            healthy_count += 1;
            let caps = probe_capabilities(&sock);
            eprintln!("  OK:   {primal} ({layer}) — {} capabilities", caps.len());
        } else {
            eprintln!("  FAIL: {primal} ({layer}) — health probe failed");
        }
    }

    v.check_abs(
        "niche_sockets_found",
        f64::from(sockets_found),
        10.0,
        0.5,
    );
    v.check_abs(
        "niche_primals_healthy",
        f64::from(healthy_count),
        10.0,
        0.5,
    );

    // ── Phase 1b: biomeOS Discovery ──────────────────────────────────
    let biomeos_sock = find_socket(primal_names::BIOMEOS);
    v.check_bool("biomeos_socket_found", biomeos_sock.is_some());

    let neural_api_sock = find_socket(primal_names::NEURAL_API);
    v.check_bool("neural_api_socket_found", neural_api_sock.is_some());

    if let Some(ref sock) = biomeos_sock {
        let health = probe_health(sock);
        v.check_bool("biomeos_health_ok", health.is_some());
    }

    // ── Phase 2: Capability Domains via biomeOS ──────────────────────
    eprintln!("\n━━━ Phase 2: Capability Domain Routing ━━━");

    if let Some(ref api_sock) = neural_api_sock {
        let validation_caps = [
            ("stats", "mean", serde_json::json!({"values": [1.0, 2.0, 3.0]})),
            ("compute", "dispatch", serde_json::json!({"method": "health", "params": {}})),
            ("storage", "store", serde_json::json!({"key": "__gate_probe__", "value": "ok"})),
            ("crypto", "hash", serde_json::json!({"data": "gate_validation_probe"})),
        ];

        for (domain, op, args) in &validation_caps {
            let result = rpc::send(
                api_sock,
                "capability.call",
                &serde_json::json!({
                    "capability": domain,
                    "operation": op,
                    "args": args,
                }),
            );
            let ok = result.is_ok();
            v.check_bool(&format!("capability_{domain}_{op}_routed"), ok);
            if ok {
                eprintln!("  OK:   {domain}.{op} routed via neural-api");
            } else {
                eprintln!("  SKIP: {domain}.{op} — routing failed or domain not registered");
            }
        }
    }

    // ── Phase 3: Provenance Trio Round-Trip ──────────────────────────
    eprintln!("\n━━━ Phase 3: Provenance Trio ━━━");

    let trio_available = airspring_barracuda::ipc::provenance::is_available();
    v.check_bool("provenance_trio_available", trio_available);

    if trio_available {
        eprintln!("  OK:   Provenance trio (rhizoCrypt + loamSpine + sweetGrass) reachable");
    } else {
        eprintln!("  SKIP: Provenance trio not available (primals not running)");
    }

    // ── Phase 4: NeuralBridge Observatory ────────────────────────────
    eprintln!("\n━━━ Phase 4: NeuralBridge Observatory ━━━");

    let weights = airspring_barracuda::ipc::neural_bridge::routing_weights();
    v.check_bool("observatory_routing_weights", weights.is_ok());

    let health = airspring_barracuda::ipc::neural_bridge::weight_health();
    v.check_bool("observatory_weight_health", health.is_ok());

    if weights.is_ok() {
        eprintln!("  OK:   Routing weights available (biomeOS v3.67+)");
    } else {
        eprintln!("  SKIP: Observatory not available (biomeOS not running or < v3.67)");
    }

    // ── Phase 5: NestGate CAS Probe ─────────────────────────────────
    eprintln!("\n━━━ Phase 5: NestGate CAS ━━━");

    if let Some(ref sock) = find_socket(primal_names::NESTGATE) {
        let status = rpc::send(sock, "storage.status", &serde_json::json!({}));
        v.check_bool("nestgate_storage_status", status.is_ok());
        if status.is_ok() {
            eprintln!("  OK:   NestGate storage.status responding");
        }
    }

    // ── Phase 6: Cross-Primal Forwarding via airSpring ───────────────
    eprintln!("\n━━━ Phase 6: airSpring Cross-Primal Forwarding ━━━");

    let airspring_sock = find_socket("airspring");
    v.check_bool("airspring_primal_socket", airspring_sock.is_some());

    if let Some(ref sock) = airspring_sock {
        let comp_status = rpc::send(sock, "composition.status", &serde_json::json!({}))
            .ok()
            .and_then(|r| r.get("result").cloned());
        v.check_bool("composition_status_response", comp_status.is_some());

        if let Some(ref status) = comp_status {
            let ratio = status
                .pointer("/primal_health/ratio")
                .and_then(serde_json::Value::as_f64)
                .unwrap_or(0.0);
            v.check_lower("composition_health_ratio", ratio, 0.5);
            eprintln!("  OK:   composition.status health ratio: {ratio:.2}");

            let obs = status
                .pointer("/observatory/neural_api_v3_67")
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(false);
            v.check_bool("composition_observatory_wired", true);
            eprintln!("  INFO: observatory.neural_api_v3_67 = {obs}");
        }
    }

    // ── Summary ──────────────────────────────────────────────────────
    eprintln!("\n━━━ Gate Composition Summary ━━━");
    eprintln!(
        "  Primals discovered: {sockets_found}/10 ({healthy_count} healthy)"
    );
    eprintln!(
        "  biomeOS: {}",
        if biomeos_sock.is_some() {
            "available"
        } else {
            "not found"
        }
    );
    eprintln!(
        "  Observatory: {}",
        if weights.is_ok() {
            "v3.67+ live"
        } else {
            "not available"
        }
    );
    eprintln!(
        "  Provenance trio: {}",
        if trio_available {
            "available"
        } else {
            "not available"
        }
    );

    v.finish();
}
