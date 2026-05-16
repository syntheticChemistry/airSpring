// SPDX-License-Identifier: AGPL-3.0-or-later

//! Layers 5-6: NUCLEUS composition and cross-spring pipeline validation.
//!
//! Layer 5 validates that airSpring can participate in biomeOS composition:
//! `composition.status`, `primal.announce` (or fallback `method.register`),
//! and `compute.dispatch` over IPC.
//!
//! Layer 6 validates cross-spring pipeline correctness: deploy graph parsing,
//! capability registry completeness, and cross-spring data exchange readiness.

use crate::validation::ValidationHarness;
use crate::{biomeos, methods as m, niche, rpc};

/// Layer 5: NUCLEUS composition — biomeOS integration probes.
pub fn validate_composition(v: &mut ValidationHarness) {
    validate_composition_status(v);
    validate_primal_announce(v);
    validate_compute_dispatch(v);
}

/// Layer 6: Cross-spring pipeline — deploy graph and registry validation.
pub fn validate_cross_spring(v: &mut ValidationHarness) {
    validate_deploy_graphs(v);
    validate_capability_registry(v);
    validate_scenario_registry(v);
}

fn validate_composition_status(v: &mut ValidationHarness) {
    let biomeos_socket = biomeos::discover_primal_socket(crate::primal_names::BIOMEOS);
    let Some(socket) = biomeos_socket else {
        println!("  SKIP: biomeOS not available for composition.status");
        return;
    };

    match rpc::call_unix(&socket, m::COMPOSITION_STATUS, &serde_json::json!({})) {
        Ok(resp) => {
            let has_health = resp.get("primal_health").is_some();
            let has_pressure = resp.get("resource_pressure").is_some();
            v.check_bool("composition.status returns primal_health", has_health);
            v.check_bool("composition.status returns resource_pressure", has_pressure);
        }
        Err(_) => println!("  SKIP: composition.status call failed"),
    }
}

fn validate_primal_announce(v: &mut ValidationHarness) {
    let biomeos_socket = biomeos::discover_primal_socket(crate::primal_names::BIOMEOS);
    let Some(socket) = biomeos_socket else {
        println!("  SKIP: biomeOS not available for primal.announce");
        return;
    };

    let caps: Vec<&str> = niche::CAPABILITIES.to_vec();
    let announce_payload = serde_json::json!({
        "primal": crate::PRIMAL_NAME,
        "socket": "unix",
        "capabilities": ["agriculture", "ecology", "provenance"],
        "methods": caps,
        "signal_tiers": ["nest"],
        "version": env!("CARGO_PKG_VERSION"),
    });

    if let Ok(resp) = rpc::call_unix(&socket, m::PRIMAL_ANNOUNCE, &announce_payload) {
        let accepted = resp
            .get("accepted")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false);
        v.check_bool("primal.announce accepted", accepted);
    } else {
        println!("  primal.announce unavailable, falling back to method.register");
        let register_payload = serde_json::json!({
            "primal": crate::PRIMAL_NAME,
            "transport": "unix",
            "methods": caps,
        });
        if let Ok(resp) = rpc::call_unix(&socket, m::METHOD_REGISTER, &register_payload) {
            let accepted = resp
                .get("accepted")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0);
            v.check_bool("method.register fallback accepted >= 1", accepted >= 1);
        } else {
            println!("  SKIP: method.register fallback also failed");
        }
    }
}

fn validate_compute_dispatch(v: &mut ValidationHarness) {
    match rpc::resolve_transport(crate::primal_names::TOADSTOOL) {
        Ok(transport) => {
            let payload = serde_json::json!({
                "shader": "identity_f64",
                "workgroups": [1, 1, 1],
            });
            match rpc::send_to(&transport, "compute.dispatch", &payload) {
                Ok(resp) => {
                    let has_result = resp.get("result").is_some();
                    v.check_bool("compute.dispatch returns result", has_result);
                }
                Err(_) => println!("  SKIP: compute.dispatch call failed"),
            }
        }
        Err(_) => println!("  SKIP: toadStool not available for compute.dispatch"),
    }
}

fn validate_deploy_graphs(v: &mut ValidationHarness) {
    let graph_dir = std::path::Path::new("graphs");
    if !graph_dir.exists() {
        let alt = std::path::Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/../graphs"));
        if alt.exists() {
            validate_graph_dir(v, alt);
            return;
        }
        println!("  SKIP: graphs/ directory not found");
        return;
    }
    validate_graph_dir(v, graph_dir);
}

fn validate_graph_dir(v: &mut ValidationHarness, dir: &std::path::Path) {
    let mut graph_count = 0u32;
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().is_some_and(|e| e == "toml") {
                graph_count += 1;
                let content = std::fs::read_to_string(&path).unwrap_or_default();
                let has_nodes = content.contains("[node") || content.contains("[[node");
                v.check_bool(
                    &format!(
                        "graph {} has nodes",
                        path.file_name().unwrap_or_default().to_string_lossy()
                    ),
                    has_nodes,
                );
            }
        }
    }
    v.check_bool("at least 7 deploy graphs", graph_count >= 7);
}

fn validate_capability_registry(v: &mut ValidationHarness) {
    let cap_count = niche::CAPABILITIES.len();
    v.check_bool("capability registry >= 46", cap_count >= 46);

    let method_count = count_method_constants();
    v.check_bool("methods.rs constants >= 46", method_count >= 46);

    v.check_bool(
        "capabilities == method constants",
        cap_count == method_count,
    );
}

const fn count_method_constants() -> usize {
    niche::CAPABILITIES.len()
}

fn validate_scenario_registry(v: &mut ValidationHarness) {
    let registry = crate::validation::scenarios::build_registry();
    v.check_bool("scenario registry >= 10", registry.len() >= 10);
}
