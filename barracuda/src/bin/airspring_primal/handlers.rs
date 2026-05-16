// SPDX-License-Identifier: AGPL-3.0-or-later
//! JSON-RPC handler implementations for niche capabilities.

use std::sync::atomic::Ordering;

use airspring_barracuda::{biomeos, niche, primal_names, rpc};

use super::NicheState;
use super::discovery::{discover_compute_primal, discover_data_primal};

pub fn handle_health(state: &NicheState) -> serde_json::Value {
    serde_json::json!({
        "status": "healthy",
        "niche": niche::NICHE_NAME,
        "version": env!("CARGO_PKG_VERSION"),
        "uptime_secs": state.start_time.elapsed().as_secs(),
        "requests_served": state.requests_served.load(Ordering::Relaxed),
        "capabilities": niche::CAPABILITIES,
        "backend": "cpu",
    })
}

/// Minimal liveness probe — confirms the process is running and responsive.
pub fn handle_liveness() -> serde_json::Value {
    serde_json::json!({
        "alive": true,
        "niche": niche::NICHE_NAME,
    })
}

/// Readiness probe — confirms subsystems are operational.
pub fn handle_readiness(state: &NicheState) -> serde_json::Value {
    let trio_available = airspring_barracuda::ipc::provenance::is_available();
    let nestgate_available = discover_data_primal().is_some();
    let toadstool_available = discover_compute_primal().is_some();

    serde_json::json!({
        "ready": true,
        "niche": niche::NICHE_NAME,
        "version": env!("CARGO_PKG_VERSION"),
        "uptime_secs": state.start_time.elapsed().as_secs(),
        "subsystems": {
            "science_dispatch": true,
            "provenance_trio": trio_available,
            (primal_names::NESTGATE): nestgate_available,
            (primal_names::TOADSTOOL): toadstool_available,
        },
    })
}

pub fn handle_ecology_experiment(params: &serde_json::Value) -> serde_json::Value {
    let experiment_name = params
        .get("experiment")
        .or_else(|| params.get("name"))
        .and_then(|v| v.as_str())
        .unwrap_or("unnamed");

    let methods: Vec<&str> = params
        .get("methods")
        .and_then(|v| v.as_array())
        .map(|a| a.iter().filter_map(|v| v.as_str()).collect())
        .or_else(|| {
            params
                .get("method")
                .and_then(|v| v.as_str())
                .map(|m| vec![m])
        })
        .unwrap_or_default();

    if methods.is_empty() {
        return serde_json::json!({
            "error": "provide 'method' (string) or 'methods' (array) to execute",
        });
    }

    let science_params = params
        .get("params")
        .cloned()
        .unwrap_or_else(|| serde_json::json!({}));

    let session = airspring_barracuda::ipc::provenance::begin_experiment_session(experiment_name);
    let session_id = session.id;

    let mut results = Vec::new();
    for method in &methods {
        let result = airspring_barracuda::primal_science::dispatch_science(method, &science_params)
            .unwrap_or_else(|| serde_json::json!({"error": "unknown method", "method": method}));

        let step = serde_json::json!({
            "type": "science_dispatch",
            "method": method,
            "result_keys": result.as_object().map(|o| o.keys().collect::<Vec<_>>()),
            "has_error": result.get("error").is_some(),
        });
        let _ = airspring_barracuda::ipc::provenance::record_experiment_step(&session_id, &step);

        results.push(serde_json::json!({ "method": method, "result": result }));
    }

    let completion = airspring_barracuda::ipc::provenance::complete_experiment(&session_id);

    if params.get("cache").and_then(serde_json::Value::as_bool) == Some(true) {
        cache_experiment_result(experiment_name, &results, &completion);
    }

    serde_json::json!({
        "experiment": experiment_name,
        "session_id": session_id,
        "provenance": completion.to_json(),
        "results": results,
        "methods_executed": methods.len(),
    })
}

fn cache_experiment_result(
    experiment_name: &str,
    results: &[serde_json::Value],
    completion: &airspring_barracuda::ipc::provenance::ProvenanceCompletion,
) {
    if let Some(socket) = discover_data_primal() {
        let _ = rpc::send(
            &socket,
            "storage.store",
            &serde_json::json!({
                "key": format!("airspring:experiment:{experiment_name}"),
                "value": {
                    "schema": "ecoPrimals/experiment-result/v1",
                    "experiment": experiment_name,
                    "results": results,
                    "provenance": completion.to_json(),
                },
                "family_id": niche::NICHE_NAME,
            }),
        );
    }
}

pub fn handle_capability_list(state: &NicheState) -> serde_json::Value {
    let science: Vec<&str> = niche::CAPABILITIES
        .iter()
        .filter(|c| c.starts_with("science.") || c.starts_with("ecology."))
        .copied()
        .collect();

    let infra: Vec<&str> = niche::CAPABILITIES
        .iter()
        .filter(|c| {
            c.starts_with("primal.")
                || c.starts_with("compute.")
                || c.starts_with("data.")
                || c.starts_with("capability.")
                || c.starts_with("provenance.")
        })
        .copied()
        .collect();

    let capabilities: Vec<&str> = niche::CAPABILITIES.to_vec();

    serde_json::json!({
        "capabilities": capabilities,
        "count": niche::CAPABILITIES.len(),
        "primal": niche::NICHE_NAME,
        "version": env!("CARGO_PKG_VERSION"),
        "domain": "ecology",
        "total": niche::CAPABILITIES.len(),
        "science": science,
        "infrastructure": infra,
        "composition": {
            "provenance_trio": airspring_barracuda::ipc::provenance::is_available(),
            (primal_names::NESTGATE): discover_data_primal().is_some(),
            (primal_names::TOADSTOOL): discover_compute_primal().is_some(),
        },
        "operation_dependencies": niche::operation_dependencies(),
        "cost_estimates": niche::cost_estimates(),
        "uptime_secs": state.start_time.elapsed().as_secs(),
    })
}

pub fn handle_provenance_begin(params: &serde_json::Value) -> serde_json::Value {
    let name = params
        .get("experiment")
        .or_else(|| params.get("name"))
        .and_then(|v| v.as_str())
        .unwrap_or("unnamed_experiment");
    let r = airspring_barracuda::ipc::provenance::begin_experiment_session(name);
    serde_json::json!({
        "session_id": r.id,
        "provenance": if r.available { "available" } else { "unavailable" },
        "data": r.data,
    })
}

pub fn handle_provenance_record(params: &serde_json::Value) -> serde_json::Value {
    let sid = params
        .get("session_id")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let step = params
        .get("step")
        .or_else(|| params.get("event"))
        .cloned()
        .unwrap_or_else(|| serde_json::json!({}));
    let r = airspring_barracuda::ipc::provenance::record_experiment_step(sid, &step);
    serde_json::json!({
        "vertex_id": r.id,
        "provenance": if r.available { "available" } else { "unavailable" },
        "data": r.data,
    })
}

pub fn handle_provenance_complete(params: &serde_json::Value) -> serde_json::Value {
    let sid = params
        .get("session_id")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    airspring_barracuda::ipc::provenance::complete_experiment(sid).to_json()
}

pub fn handle_provenance_status() -> serde_json::Value {
    serde_json::json!({
        "available": airspring_barracuda::ipc::provenance::is_available(),
        "trio": {
            (primal_names::RHIZOCRYPT): format!("{}.* via capability.call", primal_names::domains::DAG),
            (primal_names::LOAMSPINE): format!("{}.* via capability.call", primal_names::domains::COMMIT),
            (primal_names::SWEETGRASS): format!("{}.* via capability.call", primal_names::domains::PROVENANCE),
        },
        "degradation": "domain logic succeeds without provenance",
    })
}

pub fn handle_cross_spring_weather(params: &serde_json::Value) -> serde_json::Value {
    let lat = params
        .get("latitude")
        .and_then(serde_json::Value::as_f64)
        .unwrap_or(42.7);
    let lon = params
        .get("longitude")
        .and_then(serde_json::Value::as_f64)
        .unwrap_or(-84.48);
    let start = params
        .get("start_date")
        .and_then(|v| v.as_str())
        .unwrap_or("2025-01-01");
    let end = params
        .get("end_date")
        .and_then(|v| v.as_str())
        .unwrap_or("2025-12-31");

    let Ok(transport) =
        airspring_barracuda::rpc::resolve_transport(airspring_barracuda::primal_names::NESTGATE)
    else {
        return serde_json::json!({
            "error": "NestGate not available — weather routing requires NUCLEUS",
            "schema": "ecoPrimals/time-series/v1",
            "degradation": "nestgate_unavailable",
        });
    };

    let rpc_params = serde_json::json!({
        "capability": "data",
        "operation": "weather.daily",
        "args": {
            "latitude": lat,
            "longitude": lon,
            "start_date": start,
            "end_date": end,
        },
    });

    match airspring_barracuda::rpc::send_to(&transport, "capability.call", &rpc_params) {
        Ok(resp) => resp.get("result").map_or_else(
            || {
                resp.get("error").map_or_else(
                    || resp.clone(),
                    |err| {
                        serde_json::json!({
                            "error": err,
                            "schema": "ecoPrimals/time-series/v1",
                        })
                    },
                )
            },
            |result| {
                serde_json::json!({
                    "data": result,
                    "provider": "nestgate_routed",
                    "schema": "ecoPrimals/time-series/v1",
                })
            },
        ),
        Err(e) => serde_json::json!({
            "error": format!("NestGate IPC failed: {e}"),
            "schema": "ecoPrimals/time-series/v1",
        }),
    }
}

pub fn handle_compute_offload(params: &serde_json::Value) -> serde_json::Value {
    let op = params
        .get("operation")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let Some(socket) = discover_compute_primal() else {
        return serde_json::json!({
            "error": "compute primal not found — Node Atomic not running",
            "hint": "start Node Atomic to enable GPU offload",
            "env_override": super::discovery::COMPUTE_PRIMAL_ENV,
        });
    };
    let inner = params
        .get("params")
        .cloned()
        .unwrap_or_else(|| serde_json::json!({}));
    rpc::send(&socket, &format!("compute.{op}"), &inner).map_or_else(
        |_| serde_json::json!({"error": format!("compute.{op} dispatch failed"), "fallback": "cpu"}),
        |resp| {
            serde_json::json!({
                "offloaded_to": socket.display().to_string(),
                "operation": op,
                "response": resp,
                "transport": "node_atomic_unix_socket",
            })
        },
    )
}

pub fn handle_data_weather(params: &serde_json::Value) -> serde_json::Value {
    let Some(socket) = discover_data_primal() else {
        return serde_json::json!({
            "error": "data primal not found — using direct HTTP",
            "hint": "start Nest Atomic for content-addressed caching",
            "env_override": super::discovery::DATA_PRIMAL_ENV,
            "transport": "standalone",
        });
    };
    // NestGate is a storage primal — it does not implement `data.*` methods.
    // Route through `capability.call` with the standard NestGate operations.
    let call_params = serde_json::json!({
        "capability": "storage",
        "operation": "weather.daily",
        "params": params,
    });
    rpc::send(&socket, "capability.call", &call_params).map_or_else(
        |_| {
            serde_json::json!({
                "error": "NestGate capability.call dispatch failed",
                "fallback": "direct_http",
            })
        },
        |resp| {
            serde_json::json!({
                "source": socket.display().to_string(),
                "transport": "nest_atomic_unix_socket",
                "response": resp,
            })
        },
    )
}

pub fn handle_primal_forward(params: &serde_json::Value) -> serde_json::Value {
    let Some(primal) = params.get("primal").and_then(|v| v.as_str()) else {
        return serde_json::json!({"error": "missing 'primal' parameter"});
    };
    let Some(method) = params.get("method").and_then(|v| v.as_str()) else {
        return serde_json::json!({"error": "missing 'method' parameter"});
    };
    let inner = params
        .get("params")
        .cloned()
        .unwrap_or_else(|| serde_json::json!({}));
    let Some(socket) = biomeos::discover_primal_socket(primal) else {
        return serde_json::json!({"error": format!("primal '{primal}' not found")});
    };
    rpc::send(&socket, method, &inner).map_or_else(
        |_| serde_json::json!({"error": format!("forward to {primal}:{method} failed")}),
        |resp| serde_json::json!({"forwarded_to": primal, "method": method, "response": resp}),
    )
}

/// biomeOS `composition.status` — reports active users, primal health,
/// and resource pressure for the Pathway Learner and neuralAPI dashboard.
pub fn handle_composition_status(state: &NicheState) -> serde_json::Value {
    let trio_available = airspring_barracuda::ipc::provenance::is_available();
    let nestgate = discover_data_primal().is_some();
    let toadstool = discover_compute_primal().is_some();
    let skunkbat = biomeos::discover_primal_socket(primal_names::SKUNKBAT).is_some();

    let healthy_count = u32::from(trio_available)
        + u32::from(nestgate)
        + u32::from(toadstool)
        + u32::from(skunkbat);
    let total_optional = 4_u32;
    let health_ratio = f64::from(healthy_count) / f64::from(total_optional);

    serde_json::json!({
        "niche": niche::NICHE_NAME,
        "version": env!("CARGO_PKG_VERSION"),
        "active_users": state.requests_served.load(Ordering::Relaxed),
        "primal_health": {
            "ratio": health_ratio,
            "healthy": healthy_count,
            "total_optional": total_optional,
            "provenance_trio": trio_available,
            (primal_names::NESTGATE): nestgate,
            (primal_names::TOADSTOOL): toadstool,
            (primal_names::SKUNKBAT): skunkbat,
        },
        "resource_pressure": {
            "uptime_secs": state.start_time.elapsed().as_secs(),
            "capabilities": niche::CAPABILITIES.len(),
        },
    })
}

/// biomeOS v3.51 dynamic method registration acknowledgement.
///
/// Accepts inbound `method.register` calls from other primals that want
/// to register their methods through this niche.
pub fn handle_method_register(params: &serde_json::Value) -> serde_json::Value {
    let method_name = params.get("method").and_then(|v| v.as_str()).unwrap_or("");
    let methods = params
        .get("methods")
        .and_then(|v| v.as_array())
        .map_or_else(|| usize::from(!method_name.is_empty()), Vec::len);
    tracing::info!(
        target: primal_names::BIOMEOS,
        method = method_name,
        count = methods,
        "method.register received"
    );
    serde_json::json!({
        "registered": true,
        "method": method_name,
        "count": methods,
        "primal": niche::NICHE_NAME,
    })
}

/// Wave 17 single-call registration: `primal.announce` replaces the 3-call
/// `lifecycle.register + capability.register + method.register` pattern.
///
/// Accepts inbound announces from other primals that want to register
/// through this niche (forwarding to biomeOS when available).
pub fn handle_primal_announce(params: &serde_json::Value) -> serde_json::Value {
    let primal_id = params
        .get("primal")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown");
    let methods = params
        .get("methods")
        .and_then(|v| v.as_array())
        .map_or(0, Vec::len);
    let socket = params
        .get("socket")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    tracing::info!(
        target: primal_names::BIOMEOS,
        primal = primal_id,
        methods,
        socket,
        "primal.announce received"
    );
    serde_json::json!({
        "accepted": true,
        "primal": primal_id,
        "methods_registered": methods,
        "protocol": "primal.announce/v1",
    })
}

/// Return primal metadata for introspection by other ecosystem participants.
pub fn handle_primal_info() -> serde_json::Value {
    serde_json::json!({
        "primal": niche::NICHE_NAME,
        "version": env!("CARGO_PKG_VERSION"),
        "niche": "ecological & agricultural science",
        "capabilities": niche::CAPABILITIES,
        "capability_count": niche::CAPABILITIES.len(),
        "signal_tiers": ["nest"],
        "guidestone_level": "L4",
    })
}

pub fn handle_primal_discover() -> serde_json::Value {
    let socket_dir = biomeos::resolve_socket_dir();
    let primals = biomeos::discover_all_primals();
    serde_json::json!({
        "socket_dir": socket_dir.to_string_lossy(),
        "primals": primals,
        "count": primals.len(),
    })
}
