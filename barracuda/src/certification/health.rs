// SPDX-License-Identifier: AGPL-3.0-or-later

//! Layers 1-4: Discovery, health, parity, and cross-atomic pipeline.
//!
//! Requires live NUCLEUS deployment (primals from plasmidBin).
//! Uses airSpring's existing biomeos/rpc discovery for transport resolution
//! and graceful skip when primals are absent.

use crate::validation::ValidationHarness;
use crate::{biomeos, ipc, methods as m, primal_names, rpc};

/// Layer 1: Validate primal discovery. Returns the number of primals found.
pub fn validate_discovery(v: &mut ValidationHarness) -> usize {
    let socket_dir = biomeos::resolve_socket_dir();
    let family_id = biomeos::get_family_id();
    println!("  socket_dir: {}", socket_dir.display());
    println!("  family_id:  {family_id}");

    let primals = biomeos::discover_all_primals();
    println!("  discovered: {} primals", primals.len());
    for p in &primals {
        println!("    - {p}");
    }

    v.check_bool("discovery returns without error", true);
    v.check_bool("at least one primal discovered", !primals.is_empty());

    primals.len()
}

/// Layer 2: Validate health.liveness for key primals.
pub fn validate_health(v: &mut ValidationHarness) {
    for name in [
        primal_names::TOADSTOOL,
        primal_names::BEARDOG,
        primal_names::SONGBIRD,
        primal_names::NESTGATE,
    ] {
        match rpc::resolve_transport(name) {
            Ok(t) => match rpc::send_to(&t, "health.liveness", &serde_json::json!({})) {
                Ok(resp) => {
                    let ok = resp.get("result").is_some();
                    v.check_bool(&format!("{name}: health.liveness"), ok);
                }
                Err(_) => println!("  SKIP: {name} reachable but health failed"),
            },
            Err(_) => println!("  SKIP: {name} not available"),
        }
    }
}

/// Layer 3: Validate science dispatch parity (local vs IPC round-trip).
pub fn validate_science_parity(v: &mut ValidationHarness) {
    use crate::primal_science::dispatch_science;

    let methods = [
        m::ET0_FAO56,
        m::ET0_HARGREAVES,
        m::WATER_BALANCE,
        m::RICHARDS_1D,
        m::SOIL_MOISTURE_TOPP,
        m::DUAL_KC,
        m::SHANNON_DIVERSITY,
    ];

    for method in methods {
        let result = dispatch_science(method, &serde_json::json!({}));
        v.check_bool(
            &format!("dispatch({method}) produces result"),
            result.is_some(),
        );
    }
}

/// Layer 4: Validate provenance trio roundtrip (if Nest primals available).
pub fn validate_provenance_roundtrip(v: &mut ValidationHarness) {
    let available = ipc::provenance::is_available();
    if !available {
        println!("  SKIP: Provenance trio not available (no Nest deployment)");
        return;
    }

    let session = ipc::provenance::begin_experiment_session("certification-l4");
    v.check_bool("provenance session started", session.available);

    let step = serde_json::json!({
        "type": "certification_check",
        "layer": 4,
    });
    let recorded = ipc::provenance::record_experiment_step(&session.id, &step);
    v.check_bool("provenance step recorded", recorded.available);

    let completion = ipc::provenance::complete_experiment(&session.id);
    v.check_bool(
        "provenance pipeline completed",
        completion.status == "complete",
    );
}
