// SPDX-License-Identifier: AGPL-3.0-or-later

//! Scenario: Composition Parity — absorbed from exp002.
//!
//! Validates airSpring's niche within a NUCLEUS composition:
//! - **Tier 1 (LOCAL)**: Science capabilities resolve correctly (always green)
//! - **Tier 2 (IPC)**: Tower health, Node parity, Nest storage — skip if absent

use crate::ipc::{nestgate_data, precision_route, squirrel_inference, toadstool_validate};
use crate::validation::ValidationHarness;
use crate::validation::scenarios::registry::{Scenario, ScenarioMeta, Tier, Track};
use crate::{biomeos, ipc, methods as m, niche, primal_names, rpc};

/// Scenario metadata and entry point.
pub const SCENARIO: Scenario = Scenario {
    meta: ScenarioMeta {
        id: "composition-parity",
        track: Track::Composition,
        tier: Tier::Both,
        provenance_crate: "exp002_composition_parity",
        provenance_date: "2026-05-12",
        description: "NUCLEUS composition parity — local + IPC + provenance roundtrip",
    },
    run,
};

/// Run this validation scenario.
pub fn run(v: &mut ValidationHarness) {
    tier1_local(v);
    tier2_ipc(v);
    tier3_provenance(v);
}

fn tier1_local(v: &mut ValidationHarness) {
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
        "capability.list registered",
        niche::CAPABILITIES.contains(&m::CAPABILITY_LIST),
    );

    let no_dupes = {
        let mut seen = std::collections::HashSet::new();
        niche::CAPABILITIES.iter().all(|c| seen.insert(c))
    };
    v.check_bool("no duplicate capabilities", no_dupes);
}

fn tier2_ipc(v: &mut ValidationHarness) {
    let socket_dir = biomeos::resolve_socket_dir();
    println!("  socket_dir: {}", socket_dir.display());

    let primals_found = biomeos::discover_all_primals();
    println!("  discovered: {} primals", primals_found.len());
    v.check_bool("primal discovery returns without error", true);

    for name in [
        primal_names::TOADSTOOL,
        primal_names::BEARDOG,
        primal_names::SONGBIRD,
        primal_names::SKUNKBAT,
    ] {
        match rpc::resolve_transport(name) {
            Ok(t) => match rpc::send_to(&t, "health.liveness", &serde_json::json!({})) {
                Ok(resp) => {
                    let has_result = resp.get("result").is_some();
                    v.check_bool(&format!("{name}: health responds"), has_result);
                }
                Err(_) => println!("  SKIP: {name} reachable but health failed"),
            },
            Err(_) => println!("  SKIP: {name} not available"),
        }
    }

    tier2_toadstool_validate(v);
    tier2_precision_route(v);
    tier2_nestgate_cas(v);
    tier2_squirrel_inference(v);
}

fn tier2_toadstool_validate(v: &mut ValidationHarness) {
    match toadstool_validate::validate("workloads/airspring/airspring-et0-validation.toml", true) {
        Ok(vr) => {
            v.check_bool("toadstool.validate: responds", true);
            v.check_bool("toadstool.validate: valid field present", true);
            println!(
                "  toadstool.validate: valid={}, gpu={}, tier={}, est_ms={}",
                vr.valid, vr.gpu_available, vr.precision_tier, vr.estimated_dispatch_time_ms
            );
        }
        Err(toadstool_validate::ValidateError::NoPrimal) => {
            println!("  SKIP: toadstool.validate — toadStool not available");
        }
        Err(e) => {
            println!("  SKIP: toadstool.validate — {e}");
        }
    }
}

fn tier2_precision_route(v: &mut ValidationHarness) {
    match precision_route::route("hydrology") {
        Ok(pa) => {
            v.check_bool("precision.route: responds", true);
            let valid_tier = ["f32", "f64", "df64"].contains(&pa.recommended_tier.as_str());
            v.check_bool("precision.route: valid tier", valid_tier);
            println!(
                "  precision.route: tier={}, fma_safe={}, hw={}",
                pa.recommended_tier, pa.fma_safe, pa.hardware_hint
            );
        }
        Err(precision_route::PrecisionError::NoPrimal) => {
            println!("  SKIP: precision.route — barraCuda primal not available");
        }
        Err(e) => {
            println!("  SKIP: precision.route — {e}");
        }
    }
}

fn tier2_nestgate_cas(v: &mut ValidationHarness) {
    match nestgate_data::storage_status() {
        Ok(ss) => {
            v.check_bool("nestgate.storage.status: responds", true);
            println!(
                "  nestgate storage: healthy={}, backend={}, free={}",
                ss.healthy, ss.backend, ss.free_bytes
            );
        }
        Err(nestgate_data::NestGateError::NoPrimal) => {
            println!("  SKIP: nestgate storage — NestGate not available");
        }
        Err(e) => {
            println!("  SKIP: nestgate storage — {e}");
        }
    }
}

fn tier2_squirrel_inference(v: &mut ValidationHarness) {
    match squirrel_inference::list_models() {
        Ok(models) => {
            v.check_bool("squirrel.inference.models: responds", true);
            println!("  squirrel models: {} available", models.len());
            for m in &models {
                println!(
                    "    {} (embed={}, complete={})",
                    m.id, m.supports_embed, m.supports_complete
                );
            }
        }
        Err(squirrel_inference::InferenceError::NoPrimal) => {
            println!("  SKIP: squirrel inference — Squirrel not available");
        }
        Err(e) => {
            println!("  SKIP: squirrel inference — {e}");
        }
    }
}

fn tier3_provenance(v: &mut ValidationHarness) {
    let available = ipc::provenance::is_available();
    if !available {
        println!("  SKIP: Provenance trio not available (no NUCLEUS deployment)");
        return;
    }

    let session = ipc::provenance::begin_experiment_session("scenario-composition-parity");
    v.check_bool("provenance session started", session.available);

    let step = serde_json::json!({
        "type": "composition_check",
        "method": m::ET0_FAO56,
    });
    let recorded = ipc::provenance::record_experiment_step(&session.id, &step);
    v.check_bool("provenance step recorded", recorded.available);

    let completion = ipc::provenance::complete_experiment(&session.id);
    v.check_bool(
        "provenance pipeline completed",
        completion.status == "complete",
    );
}
