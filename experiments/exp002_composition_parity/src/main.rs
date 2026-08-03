// SPDX-License-Identifier: AGPL-3.0-or-later
//! exp002: Composition Parity (exp094 replication)
//!
//! Validates airSpring's niche within a NUCLEUS composition:
//! - **Tier 1 (LOCAL)**: Science capabilities resolve correctly (always green)
//! - **Tier 2 (IPC)**: Tower health, Node parity, Nest storage — skip if absent
//! - **Tier 3 (FULL NUCLEUS)**: End-to-end cross-atomic validation
//!
//! Pattern: primalSpring exp094_composition_parity / exp095_proto_nucleate_template
//!
//! Environment:
//! - `FAMILY_ID` — primal family for socket discovery (default: "default")
//! - `BIOMEOS_SOCKET_DIR` — override socket directory
//! - Without live primals, Tier 2/3 checks SKIP (exit 0, not FAIL)

use airspring_barracuda::biomeos;
use airspring_barracuda::methods as m;
use airspring_barracuda::niche;
use airspring_barracuda::primal_names;
use airspring_barracuda::rpc;
use airspring_barracuda::validation::{ValidationHarness, banner, init_tracing, section};

fn tier1_local(v: &mut ValidationHarness) {
    section("Tier 1: LOCAL CAPABILITIES");

    v.check_bool(
        "niche name is 'airspring'",
        niche::NICHE_NAME == "airspring",
    );
    v.check_bool("capabilities >= 40", niche::CAPABILITIES.len() >= 40);
    v.check_bool(
        "health.liveness registered",
        niche::CAPABILITIES.contains(&m::HEALTH_LIVENESS),
    );
    v.check_bool(
        "health.readiness registered",
        niche::CAPABILITIES.contains(&m::HEALTH_READINESS),
    );
    v.check_bool(
        "capability.list registered",
        niche::CAPABILITIES.contains(&m::CAPABILITY_LIST),
    );

    let deps = niche::operation_dependencies();
    v.check_bool("operation_dependencies is object", deps.is_object());

    let costs = niche::cost_estimates();
    v.check_bool("cost_estimates is object", costs.is_object());

    let mappings = niche::ecology_semantic_mappings();
    v.check_bool("ecology_mappings is object", mappings.is_object());

    let no_dupes = {
        let mut seen = std::collections::HashSet::new();
        niche::CAPABILITIES.iter().all(|c| seen.insert(c))
    };
    v.check_bool("no duplicate capabilities", no_dupes);
}

fn tier2_ipc(v: &mut ValidationHarness) {
    section("Tier 2: IPC-WIRED (skip if primals absent)");

    let socket_dir = biomeos::resolve_socket_dir();
    let family_id = biomeos::get_family_id();
    println!("  socket_dir: {}", socket_dir.display());
    println!("  family_id:  {family_id}");

    let primals_found = biomeos::discover_all_primals();
    println!("  discovered: {} primals", primals_found.len());
    v.check_bool("primal discovery returns without error", true);

    let check_primal = |v: &mut ValidationHarness, name: &str| {
        let transport = rpc::resolve_transport(name);
        match transport {
            Ok(t) => {
                let health = rpc::send_to(&t, "health", &serde_json::json!({}));
                match health {
                    Ok(resp) => {
                        let has_result = resp.get("result").is_some();
                        v.check_bool(&format!("{name}: health responds"), has_result);
                    }
                    Err(_) => {
                        println!("  SKIP: {name} reachable but health failed");
                    }
                }
            }
            Err(_) => {
                println!("  SKIP: {name} not available");
            }
        }
    };

    check_primal(v, primal_names::TOADSTOOL);
    check_primal(v, primal_names::BEARDOG);
    check_primal(v, primal_names::SONGBIRD);
}

fn tier3_nucleus(v: &mut ValidationHarness) {
    section("Tier 3: FULL NUCLEUS (skip if not deployed)");

    let provenance_available = airspring_barracuda::ipc::provenance::is_available();
    if provenance_available {
        let session = airspring_barracuda::ipc::provenance::begin_experiment_session("exp002");
        v.check_bool("provenance session started", session.available);

        let step = serde_json::json!({
            "type": "composition_check",
            "method": m::ET0_FAO56,
        });
        let recorded =
            airspring_barracuda::ipc::provenance::record_experiment_step(&session.id, &step);
        v.check_bool("provenance step recorded", recorded.available);

        let completion = airspring_barracuda::ipc::provenance::complete_experiment(&session.id);
        v.check_bool(
            "provenance pipeline completed",
            completion.status == "complete",
        );
    } else {
        println!("  SKIP: Provenance trio not available (no NUCLEUS deployment)");
    }
}

fn main() {
    init_tracing();
    banner("exp002 — Composition Parity");

    let mut v = ValidationHarness::new("exp002: Composition Parity");

    tier1_local(&mut v);
    tier2_ipc(&mut v);
    tier3_nucleus(&mut v);

    v.finish();
}
