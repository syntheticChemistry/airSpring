// SPDX-License-Identifier: AGPL-3.0-or-later

//! Transitional niche adapter for airSpring.
//!
//! airSpring is a niche deployment — not a primal. It proves scientific
//! Python baselines can be faithfully ported to sovereign Rust + GPU compute
//! using the ecoPrimals stack. The niche deploys as a biomeOS graph
//! (`graphs/airspring_niche_deploy.toml`) that composes real primals.
//!
//! This binary is the transitional adapter: a JSON-RPC 2.0 server that
//! exposes the niche's ecology capabilities via Unix domain socket until
//! biomeOS can orchestrate the niche directly from deploy graphs.
//!
//! Socket: `$XDG_RUNTIME_DIR/biomeos/airspring-{family_id}.sock`

use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use airspring_barracuda::biomeos;
use airspring_barracuda::niche;
use airspring_barracuda::primal_science;
use airspring_barracuda::rpc;

const READ_TIMEOUT_SECS: u64 = 60;
const WRITE_TIMEOUT_SECS: u64 = 10;
const HEARTBEAT_INTERVAL_SECS: u64 = 30;

struct NicheState {
    start_time: Instant,
    requests_served: AtomicU64,
}

fn orchestrator_socket_name() -> String {
    std::env::var("BIOMEOS_ORCHESTRATOR_SOCKET").unwrap_or_else(|_| "biomeOS.sock".to_string())
}

fn discover_compute_primal() -> Option<std::path::PathBuf> {
    std::env::var("AIRSPRING_COMPUTE_PRIMAL")
        .ok()
        .and_then(|name| biomeos::discover_primal_socket(&name))
}

fn discover_data_primal() -> Option<std::path::PathBuf> {
    std::env::var("AIRSPRING_DATA_PRIMAL")
        .ok()
        .and_then(|name| biomeos::discover_primal_socket(&name))
}

// ═══════════════════════════════════════════════════════════════════
// Request dispatch
// ═══════════════════════════════════════════════════════════════════

fn dispatch(method: &str, params: &serde_json::Value, state: &NicheState) -> serde_json::Value {
    if matches!(
        method,
        "lifecycle.health" | "health" | "health.check" | "science.health"
    ) {
        return handle_health(state);
    }

    if method == "science.version" {
        return serde_json::json!({
            "niche": niche::NICHE_NAME,
            "version": env!("CARGO_PKG_VERSION"),
        });
    }

    if let Some(result) = primal_science::dispatch_science(method, params) {
        auto_record_provenance(method, params, &result);
        return result;
    }

    match method {
        "ecology.experiment" => handle_ecology_experiment(params),
        "capability.list" => handle_capability_list(state),
        "provenance.begin" => handle_provenance_begin(params),
        "provenance.record" => handle_provenance_record(params),
        "provenance.complete" => handle_provenance_complete(params),
        "provenance.status" => handle_provenance_status(),
        "data.cross_spring_weather" => handle_cross_spring_weather(params),
        "primal.forward" => handle_primal_forward(params),
        "primal.discover" => handle_primal_discover(),
        "compute.offload" => handle_compute_offload(params),
        "data.weather" => handle_data_weather(params),
        _ => serde_json::json!({"error": "method_not_found", "method": method}),
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

// ═══════════════════════════════════════════════════════════════════
// Handlers
// ═══════════════════════════════════════════════════════════════════

fn handle_health(state: &NicheState) -> serde_json::Value {
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

fn handle_ecology_experiment(params: &serde_json::Value) -> serde_json::Value {
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
        let result = primal_science::dispatch_science(method, &science_params)
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
    if let Some(socket) = biomeos::discover_primal_socket("nestgate") {
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

fn handle_capability_list(state: &NicheState) -> serde_json::Value {
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

    serde_json::json!({
        "niche": niche::NICHE_NAME,
        "version": env!("CARGO_PKG_VERSION"),
        "domain": "ecology",
        "total": niche::CAPABILITIES.len(),
        "science": science,
        "infrastructure": infra,
        "composition": {
            "provenance_trio": airspring_barracuda::ipc::provenance::is_available(),
            "nestgate": discover_data_primal().is_some(),
            "toadstool": discover_compute_primal().is_some(),
        },
        "operation_dependencies": niche::operation_dependencies(),
        "cost_estimates": niche::cost_estimates(),
        "uptime_secs": state.start_time.elapsed().as_secs(),
    })
}

fn handle_provenance_begin(params: &serde_json::Value) -> serde_json::Value {
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

fn handle_provenance_record(params: &serde_json::Value) -> serde_json::Value {
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

fn handle_provenance_complete(params: &serde_json::Value) -> serde_json::Value {
    let sid = params
        .get("session_id")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    airspring_barracuda::ipc::provenance::complete_experiment(sid).to_json()
}

fn handle_provenance_status() -> serde_json::Value {
    serde_json::json!({
        "available": airspring_barracuda::ipc::provenance::is_available(),
        "trio": {
            "rhizocrypt": "dag.* via capability.call",
            "loamspine": "commit.* via capability.call",
            "sweetgrass": "provenance.* via capability.call",
        },
        "degradation": "domain logic succeeds without provenance",
    })
}

fn handle_cross_spring_weather(params: &serde_json::Value) -> serde_json::Value {
    use airspring_barracuda::data::Provider;

    let provider = airspring_barracuda::data::NestGateProvider::new();
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

    match provider.fetch_daily_weather(lat, lon, start, end) {
        Ok(resp) => resp.to_cross_spring_v1("nestgate_routed"),
        Err(e) => serde_json::json!({
            "error": format!("fetch failed: {e}"),
            "schema": "ecoPrimals/time-series/v1",
        }),
    }
}

fn handle_compute_offload(params: &serde_json::Value) -> serde_json::Value {
    let op = params
        .get("operation")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let Some(socket) = discover_compute_primal() else {
        return serde_json::json!({
            "error": "compute primal not found — Node Atomic not running",
            "hint": "start Node Atomic to enable GPU offload",
            "env_override": "AIRSPRING_COMPUTE_PRIMAL",
        });
    };
    let inner = params
        .get("params")
        .cloned()
        .unwrap_or_else(|| serde_json::json!({}));
    rpc::send(&socket, &format!("compute.{op}"), &inner).map_or_else(
        || serde_json::json!({"error": format!("compute.{op} dispatch failed"), "fallback": "cpu"}),
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

fn handle_data_weather(params: &serde_json::Value) -> serde_json::Value {
    let Some(socket) = discover_data_primal() else {
        return serde_json::json!({
            "error": "data primal not found — using direct HTTP",
            "hint": "start Nest Atomic for content-addressed caching",
            "env_override": "AIRSPRING_DATA_PRIMAL",
            "transport": "standalone",
        });
    };
    rpc::send(&socket, "data.open_meteo_weather", params).map_or_else(
        || {
            serde_json::json!({
                "error": "data.open_meteo_weather dispatch failed",
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

fn handle_primal_forward(params: &serde_json::Value) -> serde_json::Value {
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
        || serde_json::json!({"error": format!("forward to {primal}:{method} failed")}),
        |resp| serde_json::json!({"forwarded_to": primal, "method": method, "response": resp}),
    )
}

fn handle_primal_discover() -> serde_json::Value {
    let socket_dir = biomeos::resolve_socket_dir();
    let primals = biomeos::discover_all_primals();
    serde_json::json!({
        "socket_dir": socket_dir.to_string_lossy(),
        "primals": primals,
        "count": primals.len(),
    })
}

// ═══════════════════════════════════════════════════════════════════
// biomeOS registration
// ═══════════════════════════════════════════════════════════════════

fn register_with_biomeos(our_socket: &Path) {
    let biomeos_socket = biomeos::resolve_socket_dir().join(orchestrator_socket_name());
    if !biomeos_socket.exists() {
        eprintln!(
            "[biomeos] No orchestrator at {}, running standalone",
            biomeos_socket.display()
        );
        if let Some(fallback_name) = biomeos::fallback_registration_primal() {
            if let Some(ref fallback_sock) = biomeos::discover_primal_socket(&fallback_name) {
                eprintln!(
                    "[biomeos] Found {fallback_name} at {}, registering via fallback",
                    fallback_sock.display()
                );
                niche::register_with_target(fallback_sock, our_socket);
                return;
            }
            eprintln!("[biomeos] Fallback '{fallback_name}' not found — fully standalone");
        }
        return;
    }
    niche::register_with_target(&biomeos_socket, our_socket);
}

// ═══════════════════════════════════════════════════════════════════
// Connection handler + metrics
// ═══════════════════════════════════════════════════════════════════

#[expect(
    clippy::needless_pass_by_value,
    reason = "BufReader::new consumes the stream"
)]
fn handle_connection(stream: UnixStream, state: &NicheState) {
    stream
        .set_read_timeout(Some(Duration::from_secs(READ_TIMEOUT_SECS)))
        .ok();
    stream
        .set_write_timeout(Some(Duration::from_secs(WRITE_TIMEOUT_SECS)))
        .ok();

    let reader = BufReader::new(&stream);
    let mut writer = &stream;

    for line_result in reader.lines() {
        let Ok(line) = line_result else { break };
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        let parsed: serde_json::Value = match serde_json::from_str(trimmed) {
            Ok(v) => v,
            Err(e) => {
                let resp = rpc::error(
                    &serde_json::Value::Null,
                    rpc::PARSE_ERROR,
                    &format!("Parse error: {e}"),
                );
                let _ = writeln!(writer, "{resp}");
                let _ = writer.flush();
                continue;
            }
        };

        let id = parsed.get("id").cloned().unwrap_or(serde_json::Value::Null);
        let method = parsed.get("method").and_then(|v| v.as_str()).unwrap_or("");
        let params = parsed
            .get("params")
            .cloned()
            .unwrap_or_else(|| serde_json::json!({}));

        state.requests_served.fetch_add(1, Ordering::Relaxed);
        let t0 = Instant::now();
        let result = dispatch(method, &params, state);
        let latency_ms = t0.elapsed().as_secs_f64() * 1000.0;
        emit_metrics(method, latency_ms, result.get("error").is_none());

        let _ = writeln!(writer, "{}", rpc::success(&id, &result));
        let _ = writer.flush();
    }
}

fn emit_metrics(operation: &str, latency_ms: f64, success: bool) {
    eprintln!(
        "[metrics] niche={} operation={operation} latency_ms={latency_ms:.2} success={success}",
        niche::NICHE_NAME
    );
    if let Ok(socket_path) = std::env::var("BIOMEOS_METRICS_SOCKET") {
        let payload = serde_json::json!({
            "niche": niche::NICHE_NAME,
            "operation": operation,
            "latency_ms": latency_ms,
            "success": success,
            "version": env!("CARGO_PKG_VERSION"),
        });
        if let Ok(mut stream) = std::os::unix::net::UnixStream::connect(&socket_path) {
            let s = serde_json::to_string(&payload).unwrap_or_default();
            let _ = std::io::Write::write_all(&mut stream, s.as_bytes());
            let _ = std::io::Write::write_all(&mut stream, b"\n");
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// Server lifecycle
// ═══════════════════════════════════════════════════════════════════

fn run() -> Result<(), String> {
    let family_id = biomeos::get_family_id();
    let socket_path = biomeos::resolve_socket_path(niche::NICHE_NAME, &family_id);

    if let Some(parent) = socket_path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("Cannot create socket directory {}: {e}", parent.display()))?;
    }
    if socket_path.exists() {
        std::fs::remove_file(&socket_path)
            .map_err(|e| format!("Cannot remove stale socket {}: {e}", socket_path.display()))?;
    }

    let state = Arc::new(NicheState {
        start_time: Instant::now(),
        requests_served: AtomicU64::new(0),
    });

    let listener = UnixListener::bind(&socket_path)
        .map_err(|e| format!("Cannot bind to {}: {e}", socket_path.display()))?;

    eprintln!(
        "{} niche listening on {}",
        niche::NICHE_NAME,
        socket_path.display()
    );
    eprintln!("  Family ID: {family_id}");
    eprintln!("  Version: {}", env!("CARGO_PKG_VERSION"));
    eprintln!("  Capabilities ({}):", niche::CAPABILITIES.len());
    for cap in niche::CAPABILITIES {
        eprintln!("    - {cap}");
    }

    register_with_biomeos(&socket_path);

    let running = Arc::new(AtomicBool::new(true));
    let heartbeat_state = state.clone();
    let heartbeat_running = running.clone();
    std::thread::spawn(move || {
        let target = {
            let biomeos_sock = biomeos::resolve_socket_dir().join(orchestrator_socket_name());
            if biomeos_sock.exists() {
                Some(biomeos_sock)
            } else {
                biomeos::fallback_registration_primal()
                    .and_then(|name| biomeos::discover_primal_socket(&name))
            }
        };

        while heartbeat_running.load(Ordering::Relaxed) {
            std::thread::sleep(Duration::from_secs(HEARTBEAT_INTERVAL_SECS));
            if let Some(ref t) = target {
                let _ = rpc::send(
                    t,
                    "lifecycle.status",
                    &serde_json::json!({
                        "name": niche::NICHE_NAME,
                        "socket_path": socket_path.to_string_lossy(),
                        "status": "healthy",
                        "requests_served": heartbeat_state.requests_served.load(Ordering::Relaxed),
                        "version": env!("CARGO_PKG_VERSION"),
                        "capabilities_total": niche::CAPABILITIES.len(),
                        "composition": {
                            "provenance_trio": airspring_barracuda::ipc::provenance::is_available(),
                            "nestgate": discover_data_primal().is_some(),
                            "toadstool": discover_compute_primal().is_some(),
                        },
                    }),
                );
            }
        }
    });

    eprintln!("[ready] Accepting connections...");
    for stream in listener.incoming() {
        if !running.load(Ordering::Relaxed) {
            break;
        }
        match stream {
            Ok(s) => {
                let st = state.clone();
                std::thread::spawn(move || handle_connection(s, &st));
            }
            Err(e) => eprintln!("[error] Accept failed: {e}"),
        }
    }
    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("[fatal] {e}");
        std::process::exit(1);
    }
}
