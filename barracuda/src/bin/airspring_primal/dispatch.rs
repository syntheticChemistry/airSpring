// SPDX-License-Identifier: AGPL-3.0-or-later
//! JSON-RPC method dispatch and provenance auto-recording.

use airspring_barracuda::ipc::DispatchOutcome;
use airspring_barracuda::{niche, primal_science};

use super::NicheState;
use super::handlers;

pub fn dispatch(
    method: &str,
    params: &serde_json::Value,
    state: &NicheState,
) -> DispatchOutcome<serde_json::Value> {
    if matches!(
        method,
        "lifecycle.health" | "health" | "health.check" | "science.health"
    ) {
        return DispatchOutcome::Ok(handlers::handle_health(state));
    }

    if method == "health.liveness" {
        return DispatchOutcome::Ok(handlers::handle_liveness());
    }

    if method == "health.readiness" {
        return DispatchOutcome::Ok(handlers::handle_readiness(state));
    }

    if method == "science.version" {
        return DispatchOutcome::Ok(serde_json::json!({
            "niche": niche::NICHE_NAME,
            "version": env!("CARGO_PKG_VERSION"),
        }));
    }

    if let Some(result) = primal_science::dispatch_science(method, params) {
        auto_record_provenance(method, params, &result);
        return DispatchOutcome::Ok(result);
    }

    match method {
        "ecology.experiment" => DispatchOutcome::Ok(handlers::handle_ecology_experiment(params)),
        "capability.list" => DispatchOutcome::Ok(handlers::handle_capability_list(state)),
        "provenance.begin" => DispatchOutcome::Ok(handlers::handle_provenance_begin(params)),
        "provenance.record" => DispatchOutcome::Ok(handlers::handle_provenance_record(params)),
        "provenance.complete" => DispatchOutcome::Ok(handlers::handle_provenance_complete(params)),
        "provenance.status" => DispatchOutcome::Ok(handlers::handle_provenance_status()),
        "data.cross_spring_weather" => {
            DispatchOutcome::Ok(handlers::handle_cross_spring_weather(params))
        }
        "primal.forward" => DispatchOutcome::Ok(handlers::handle_primal_forward(params)),
        "primal.discover" => DispatchOutcome::Ok(handlers::handle_primal_discover()),
        "compute.offload" => DispatchOutcome::Ok(handlers::handle_compute_offload(params)),
        "data.weather" => DispatchOutcome::Ok(handlers::handle_data_weather(params)),
        _ => DispatchOutcome::MethodNotFound(method.to_string()),
    }
}

fn auto_record_provenance(method: &str, params: &serde_json::Value, result: &serde_json::Value) {
    let Some(session_id) = params.get("session_id").and_then(|v| v.as_str()) else {
        return;
    };
    if session_id.is_empty() {
        return;
    }

    let step = serde_json::json!({
        "type": "science_dispatch",
        "method": method,
        "params_summary": summarize_json_keys(params),
        "result_summary": {
            "keys": summarize_json_keys(result),
            "has_error": result.get("error").is_some(),
        },
        "niche": niche::NICHE_NAME,
        "version": env!("CARGO_PKG_VERSION"),
    });
    let _ = airspring_barracuda::ipc::provenance::record_experiment_step(session_id, &step);
}

fn summarize_json_keys(v: &serde_json::Value) -> serde_json::Value {
    let keys: Vec<&str> = v
        .as_object()
        .map(|o| o.keys().map(String::as_str).collect())
        .unwrap_or_default();
    serde_json::json!({ "keys": keys, "count": keys.len() })
}
