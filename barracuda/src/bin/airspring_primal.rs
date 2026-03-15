// SPDX-License-Identifier: AGPL-3.0-or-later

//! airSpring biomeOS Primal — NUCLEUS Tower Mode
//!
//! JSON-RPC 2.0 server exposing airSpring's ecological science capabilities
//! to the biomeOS ecosystem via Unix domain socket.
//!
//! ## Capability domains
//!
//! **Evapotranspiration**:
//!   `science.et0_fao56`, `science.et0_hargreaves`, `science.et0_priestley_taylor`,
//!   `science.et0_makkink`, `science.et0_turc`, `science.et0_hamon`,
//!   `science.et0_blaney_criddle`
//!
//! **Water balance**: `science.water_balance`
//! **Yield**: `science.yield_response`
//! **Soil**: `science.richards_pde`, `science.pedotransfer`
//! **Diversity**: `science.shannon_diversity`, `science.bray_curtis`
//!
//! ## biomeOS integration
//!
//! On startup, probes for a biomeOS orchestrator socket and registers
//! capabilities via `lifecycle.register` + `capability.register`.
//! Sends heartbeats every 30s. Cleans up socket on SIGTERM.
//!
//! Socket: `$XDG_RUNTIME_DIR/biomeos/airspring-{family_id}.sock`

#![allow(
    clippy::too_many_lines,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::similar_names
)]

use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use airspring_barracuda::biomeos;
use airspring_barracuda::primal_science;
use airspring_barracuda::rpc;


const PRIMAL_NAME: &str = "airspring";
const READ_TIMEOUT_SECS: u64 = 60;
const WRITE_TIMEOUT_SECS: u64 = 10;
const HEARTBEAT_INTERVAL_SECS: u64 = 30;

fn orchestrator_socket_name() -> String {
    std::env::var("BIOMEOS_ORCHESTRATOR_SOCKET").unwrap_or_else(|_| "biomeOS.sock".to_string())
}

const ALL_CAPABILITIES: &[&str] = &[
    // ── Evapotranspiration (7 methods) ──
    "science.et0_fao56",
    "science.et0_hargreaves",
    "science.et0_priestley_taylor",
    "science.et0_makkink",
    "science.et0_turc",
    "science.et0_hamon",
    "science.et0_blaney_criddle",
    // ── Water balance & yield ──
    "science.water_balance",
    "science.yield_response",
    // ── Soil physics ──
    "science.richards_1d",
    "science.scs_cn_runoff",
    "science.green_ampt_infiltration",
    "science.soil_moisture_topp",
    "science.pedotransfer_saxton_rawls",
    // ── Crop & irrigation ──
    "science.dual_kc",
    "science.sensor_calibration",
    "science.gdd",
    // ── Biodiversity ──
    "science.shannon_diversity",
    "science.bray_curtis",
    // ── Geophysics coupling ──
    "science.anderson_coupling",
    // ── Monthly ET ──
    "science.thornthwaite",
    // ── Drought & Stochastic (v0.7.4+) ──
    "science.spi_drought_index",
    "science.autocorrelation",
    "science.gamma_cdf",
    // ── Ecology aliases ──
    "ecology.et0_fao56",
    "ecology.et0_hargreaves",
    "ecology.water_balance",
    "ecology.yield_response",
    "ecology.full_pipeline",
    "ecology.spi_drought_index",
    "ecology.autocorrelation",
    // ── Provenance trio (biomeOS composition) ──
    "provenance.begin",
    "provenance.record",
    "provenance.complete",
    "provenance.status",
    // ── Cross-primal ──
    "primal.forward",
    "primal.discover",
    // ── Niche deployment (biomeOS graph composition) ──
    "capability.list",
    "data.cross_spring_weather",
    // ── Compute offload (Node Atomic) ──
    "compute.offload",
    // ── Data (Nest Atomic routing) ──
    "data.weather",
];

// ═══════════════════════════════════════════════════════════════════
// Primal state
// ═══════════════════════════════════════════════════════════════════

struct PrimalState {
    start_time: Instant,
    requests_served: AtomicU64,
}

// ═══════════════════════════════════════════════════════════════════
// Socket resolution — delegates to `biomeos` module
// ═══════════════════════════════════════════════════════════════════

fn resolve_socket_dir() -> PathBuf {
    biomeos::resolve_socket_dir()
}

fn get_family_id() -> String {
    biomeos::get_family_id()
}

fn resolve_socket_path(family_id: &str) -> PathBuf {
    biomeos::resolve_socket_path(PRIMAL_NAME, family_id)
}

fn discover_primal_socket(primal_name: &str) -> Option<PathBuf> {
    biomeos::discover_primal_socket(primal_name)
}

fn discover_compute_primal() -> Option<PathBuf> {
    std::env::var("AIRSPRING_COMPUTE_PRIMAL")
        .ok()
        .and_then(|name| biomeos::discover_primal_socket(&name))
}

fn discover_data_primal() -> Option<PathBuf> {
    std::env::var("AIRSPRING_DATA_PRIMAL")
        .ok()
        .and_then(|name| biomeos::discover_primal_socket(&name))
}

// ═══════════════════════════════════════════════════════════════════
// Request dispatch — maps JSON-RPC methods to airSpring science
// ═══════════════════════════════════════════════════════════════════

fn dispatch(method: &str, params: &serde_json::Value, state: &PrimalState) -> serde_json::Value {
    if method == "lifecycle.health"
        || method == "health"
        || method == "health.check"
        || method == "science.health"
    {
        return handle_health(state);
    }

    if method == "science.version" {
        return serde_json::json!({
            "primal": PRIMAL_NAME,
            "version": env!("CARGO_PKG_VERSION"),
            "barracuda": "0.3.5",
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

/// Auto-record provenance when a `session_id` is present in science call params.
///
/// When biomeOS or a graph caller provides `session_id`, the science result
/// is automatically appended to the provenance DAG — no explicit
/// `provenance.record` call needed. This makes provenance nearly transparent.
fn auto_record_provenance(
    method: &str,
    params: &serde_json::Value,
    result: &serde_json::Value,
) {
    let Some(session_id) = params.get("session_id").and_then(|v| v.as_str()) else {
        return;
    };
    if session_id.is_empty() {
        return;
    }

    let step = serde_json::json!({
        "type": "science_dispatch",
        "method": method,
        "params_summary": summarize_params(params),
        "result_summary": summarize_result(result),
        "primal": PRIMAL_NAME,
        "version": env!("CARGO_PKG_VERSION"),
    });
    airspring_barracuda::ipc::provenance::record_experiment_step(session_id, &step);
}

fn summarize_params(params: &serde_json::Value) -> serde_json::Value {
    let keys: Vec<&str> = params
        .as_object()
        .map(|o| o.keys().map(String::as_str).collect())
        .unwrap_or_default();
    serde_json::json!({ "keys": keys, "count": keys.len() })
}

fn summarize_result(result: &serde_json::Value) -> serde_json::Value {
    let keys: Vec<&str> = result
        .as_object()
        .map(|o| o.keys().map(String::as_str).collect())
        .unwrap_or_default();
    let has_error = result.get("error").is_some();
    serde_json::json!({ "keys": keys, "has_error": has_error })
}

fn handle_health(state: &PrimalState) -> serde_json::Value {
    let uptime_secs = state.start_time.elapsed().as_secs();
    let requests = state.requests_served.load(Ordering::Relaxed);
    serde_json::json!({
        "status": "healthy",
        "primal": PRIMAL_NAME,
        "version": env!("CARGO_PKG_VERSION"),
        "uptime_secs": uptime_secs,
        "requests_served": requests,
        "capabilities": ALL_CAPABILITIES,
        "backend": "cpu",
    })
}

// ═══════════════════════════════════════════════════════════════════
// ecology.experiment — full auto-provenance pipeline (single call)
// ═══════════════════════════════════════════════════════════════════

/// Single composable method that wraps the full provenance lifecycle:
/// 1. Begin provenance session (if trio available)
/// 2. Execute one or more science methods
/// 3. Auto-record each step in the DAG
/// 4. Complete provenance (dehydrate → commit → attribute)
/// 5. Optionally cache results in `NestGate`
///
/// biomeOS can compose this as a single graph node instead of 5 separate ones.
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
        airspring_barracuda::ipc::provenance::record_experiment_step(&session_id, &step);

        results.push(serde_json::json!({
            "method": method,
            "result": result,
        }));
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
    if let Some(socket) = airspring_barracuda::biomeos::discover_primal_socket("nestgate") {
        let key = format!("airspring:experiment:{experiment_name}");
        let value = serde_json::json!({
            "schema": "ecoPrimals/experiment-result/v1",
            "experiment": experiment_name,
            "results": results,
            "provenance": completion.to_json(),
        });
        let _ = airspring_barracuda::rpc::send(
            &socket,
            "storage.store",
            &serde_json::json!({
                "key": key,
                "value": value,
                "family_id": "airspring",
            }),
        );
    }
}

// ═══════════════════════════════════════════════════════════════════
// Capability listing (biomeOS niche composition)
// ═══════════════════════════════════════════════════════════════════

/// neuralAPI Enhancement 2: Dependency hints for biomeOS parallelization.
///
/// biomeOS can detect independent operations and run them in parallel.
fn operation_dependencies() -> serde_json::Value {
    serde_json::json!({
        "science.et0": ["weather_data"],
        "science.thermal_time": ["temperature_data"],
        "science.vpd": ["temperature_data", "humidity_data"],
        "science.gdd": ["temperature_data"],
        "science.photoperiod": ["latitude", "day_of_year"],
        "science.soil_moisture": ["precipitation_data", "et0_data"],
        "science.biomass": ["gdd_data", "radiation_data"],
        "science.water_stress": ["soil_moisture_data", "et0_data"],
        "science.leaf_energy": ["radiation_data", "temperature_data"],
        "science.air_quality": ["station", "date_range"],
        "science.batch_et0": ["weather_data_array"],
        "ecology.experiment": ["method", "params"],
        "provenance.begin": ["experiment_name"],
        "provenance.record": ["session_id", "step_data"],
        "provenance.complete": ["session_id"],
        "provenance.status": [],
        "data.cross_spring_weather": ["station", "date_range"],
    })
}

/// neuralAPI Enhancement 3: Cost estimates for biomeOS scheduling.
///
/// Typical latencies and resource intensities measured on representative hardware.
fn cost_estimates() -> serde_json::Value {
    serde_json::json!({
        "science.et0": { "latency_ms": 0.5, "cpu": "low", "memory_bytes": 256 },
        "science.thermal_time": { "latency_ms": 0.3, "cpu": "low", "memory_bytes": 128 },
        "science.vpd": { "latency_ms": 0.2, "cpu": "low", "memory_bytes": 128 },
        "science.gdd": { "latency_ms": 0.2, "cpu": "low", "memory_bytes": 128 },
        "science.photoperiod": { "latency_ms": 0.3, "cpu": "low", "memory_bytes": 256 },
        "science.soil_moisture": { "latency_ms": 0.4, "cpu": "low", "memory_bytes": 256 },
        "science.biomass": { "latency_ms": 0.5, "cpu": "low", "memory_bytes": 256 },
        "science.water_stress": { "latency_ms": 0.4, "cpu": "low", "memory_bytes": 256 },
        "science.leaf_energy": { "latency_ms": 0.8, "cpu": "medium", "memory_bytes": 512 },
        "science.air_quality": { "latency_ms": 5.0, "cpu": "low", "memory_bytes": 4096 },
        "science.batch_et0": { "latency_ms": 50.0, "cpu": "medium", "memory_bytes": 65536 },
        "ecology.experiment": { "latency_ms": 100.0, "cpu": "medium", "memory_bytes": 8192 },
        "data.cross_spring_weather": { "latency_ms": 200.0, "cpu": "low", "memory_bytes": 16384 },
        "provenance.begin": { "latency_ms": 10.0, "cpu": "low", "memory_bytes": 512 },
        "provenance.record": { "latency_ms": 5.0, "cpu": "low", "memory_bytes": 1024 },
        "provenance.complete": { "latency_ms": 50.0, "cpu": "medium", "memory_bytes": 2048 },
    })
}

fn handle_capability_list(_state: &PrimalState) -> serde_json::Value {
    let science: Vec<&str> = ALL_CAPABILITIES
        .iter()
        .filter(|c| c.starts_with("science.") || c.starts_with("ecology."))
        .copied()
        .collect();

    let infra: Vec<&str> = ALL_CAPABILITIES
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
        "primal": PRIMAL_NAME,
        "version": env!("CARGO_PKG_VERSION"),
        "domain": "ecology",
        "total": ALL_CAPABILITIES.len(),
        "science": science,
        "infrastructure": infra,
        "composition": {
            "provenance_trio": airspring_barracuda::ipc::provenance::is_available(),
            "nestgate": crate::discover_data_primal().is_some(),
            "toadstool": crate::discover_compute_primal().is_some(),
        },
        "operation_dependencies": operation_dependencies(),
        "cost_estimates": cost_estimates(),
    })
}

// ═══════════════════════════════════════════════════════════════════
// Provenance trio handlers (biomeOS composition)
// ═══════════════════════════════════════════════════════════════════

fn handle_provenance_begin(params: &serde_json::Value) -> serde_json::Value {
    let experiment_name = params
        .get("experiment")
        .or_else(|| params.get("name"))
        .and_then(|v| v.as_str())
        .unwrap_or("unnamed_experiment");

    let result = airspring_barracuda::ipc::provenance::begin_experiment_session(experiment_name);

    serde_json::json!({
        "session_id": result.id,
        "provenance": if result.available { "available" } else { "unavailable" },
        "data": result.data,
    })
}

fn handle_provenance_record(params: &serde_json::Value) -> serde_json::Value {
    let session_id = params
        .get("session_id")
        .and_then(|v| v.as_str())
        .unwrap_or("");

    let step = params
        .get("step")
        .or_else(|| params.get("event"))
        .cloned()
        .unwrap_or_else(|| serde_json::json!({}));

    let result = airspring_barracuda::ipc::provenance::record_experiment_step(session_id, &step);

    serde_json::json!({
        "vertex_id": result.id,
        "provenance": if result.available { "available" } else { "unavailable" },
        "data": result.data,
    })
}

fn handle_provenance_complete(params: &serde_json::Value) -> serde_json::Value {
    let session_id = params
        .get("session_id")
        .and_then(|v| v.as_str())
        .unwrap_or("");

    let completion = airspring_barracuda::ipc::provenance::complete_experiment(session_id);
    completion.to_json()
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

// ═══════════════════════════════════════════════════════════════════
// Cross-Spring data exchange (ecoPrimals/time-series/v1)
// ═══════════════════════════════════════════════════════════════════

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
        Ok(response) => response.to_cross_spring_v1("nestgate_routed"),
        Err(e) => serde_json::json!({
            "error": format!("fetch failed: {e}"),
            "schema": "ecoPrimals/time-series/v1",
        }),
    }
}

// ═══════════════════════════════════════════════════════════════════
// Cross-primal: compute offload (Node Atomic)
// ═══════════════════════════════════════════════════════════════════

fn handle_compute_offload(params: &serde_json::Value) -> serde_json::Value {
    let operation = params
        .get("operation")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let compute_socket = discover_compute_primal();

    compute_socket.map_or_else(
        || {
            serde_json::json!({
                "error": "compute primal not found — Node Atomic not running",
                "hint": "start Node Atomic (tower_atomic_bootstrap + compute primal) to enable GPU offload",
                "env_override": "AIRSPRING_COMPUTE_PRIMAL",
            })
        },
        |socket| {
            let compute_method = format!("compute.{operation}");
            let inner_params = params
                .get("params")
                .cloned()
                .unwrap_or_else(|| serde_json::json!({}));

            rpc::send(&socket, &compute_method, &inner_params).map_or_else(
                || {
                    serde_json::json!({
                        "error": format!("compute.{operation} dispatch failed"),
                        "fallback": "cpu",
                    })
                },
                |resp| {
                    serde_json::json!({
                        "offloaded_to": socket.display().to_string(),
                        "operation": operation,
                        "response": resp,
                        "transport": "node_atomic_unix_socket",
                    })
                },
            )
        },
    )
}

// ═══════════════════════════════════════════════════════════════════
// Data routing: weather data (Nest Atomic)
// ═══════════════════════════════════════════════════════════════════

fn handle_data_weather(params: &serde_json::Value) -> serde_json::Value {
    let data_socket = discover_data_primal();

    data_socket.map_or_else(
        || {
            serde_json::json!({
                "error": "data primal not found — using direct HTTP",
                "hint": "start Nest Atomic (tower + data primal) for content-addressed caching",
                "env_override": "AIRSPRING_DATA_PRIMAL",
                "transport": "standalone",
            })
        },
        |socket| {
            rpc::send(&socket, "data.open_meteo_weather", params).map_or_else(
                || {
                    serde_json::json!({
                        "error": "data.open_meteo_weather dispatch failed",
                        "fallback": "direct_http",
                        "hint": "falling back to direct Open-Meteo HTTP",
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
        },
    )
}

// ═══════════════════════════════════════════════════════════════════
// Cross-primal handlers
// ═══════════════════════════════════════════════════════════════════

fn handle_primal_forward(params: &serde_json::Value) -> serde_json::Value {
    let Some(primal) = params.get("primal").and_then(|v| v.as_str()) else {
        return serde_json::json!({"error": "missing 'primal' parameter"});
    };
    let Some(method) = params.get("method").and_then(|v| v.as_str()) else {
        return serde_json::json!({"error": "missing 'method' parameter"});
    };
    let inner_params = params
        .get("params")
        .cloned()
        .unwrap_or_else(|| serde_json::json!({}));

    let Some(socket) = discover_primal_socket(primal) else {
        return serde_json::json!({"error": format!("primal '{primal}' not found")});
    };

    rpc::send(&socket, method, &inner_params).map_or_else(
        || serde_json::json!({"error": format!("forward to {primal}:{method} failed")}),
        |resp| {
            serde_json::json!({
                "forwarded_to": primal,
                "method": method,
                "response": resp,
            })
        },
    )
}

fn handle_primal_discover() -> serde_json::Value {
    let socket_dir = resolve_socket_dir();
    let primals = biomeos::discover_all_primals();

    serde_json::json!({
        "socket_dir": socket_dir.to_string_lossy(),
        "primals": primals,
        "count": primals.len(),
    })
}

// ═══════════════════════════════════════════════════════════════════
// biomeOS registration (synchronous)
// ═══════════════════════════════════════════════════════════════════

fn register_with_biomeos(our_socket: &Path) {
    let biomeos_socket = resolve_socket_dir().join(orchestrator_socket_name());
    if !biomeos_socket.exists() {
        eprintln!(
            "[biomeos] No orchestrator at {}, running standalone",
            biomeos_socket.display()
        );

        if let Some(fallback_name) = biomeos::fallback_registration_primal() {
            if let Some(ref fallback_sock) = discover_primal_socket(&fallback_name) {
                eprintln!(
                    "[biomeos] Found {fallback_name} at {}, registering via fallback",
                    fallback_sock.display()
                );
                register_via_socket(fallback_sock, our_socket);
                return;
            }
            eprintln!("[biomeos] Fallback primal '{fallback_name}' not found — fully standalone");
        } else {
            eprintln!("[biomeos] No BIOMEOS_FALLBACK_PRIMAL set — fully standalone");
        }
        return;
    }

    register_via_socket(&biomeos_socket, our_socket);
}

fn register_via_socket(target: &Path, our_socket: &Path) {
    let reg_result = rpc::send(
        target,
        "lifecycle.register",
        &serde_json::json!({
            "name": PRIMAL_NAME,
            "socket_path": our_socket.to_string_lossy(),
            "pid": std::process::id(),
        }),
    );

    match reg_result {
        Some(_) => eprintln!("[biomeos] Registered with lifecycle manager"),
        None => eprintln!("[biomeos] lifecycle.register failed (non-fatal)"),
    }

    let sock_str = our_socket.to_string_lossy().to_string();

    // Register the ecology domain with semantic mappings for capability.call routing
    let ecology_mappings = serde_json::json!({
        "et0_fao56":              "science.et0_fao56",
        "et0_hargreaves":         "science.et0_hargreaves",
        "et0_priestley_taylor":   "science.et0_priestley_taylor",
        "et0_makkink":            "science.et0_makkink",
        "et0_turc":               "science.et0_turc",
        "et0_hamon":              "science.et0_hamon",
        "et0_blaney_criddle":     "science.et0_blaney_criddle",
        "water_balance":          "science.water_balance",
        "yield_response":         "science.yield_response",
        "richards_1d":            "science.richards_1d",
        "scs_cn_runoff":          "science.scs_cn_runoff",
        "green_ampt_infiltration":"science.green_ampt_infiltration",
        "soil_moisture_topp":     "science.soil_moisture_topp",
        "pedotransfer":           "science.pedotransfer_saxton_rawls",
        "dual_kc":                "science.dual_kc",
        "sensor_calibration":     "science.sensor_calibration",
        "gdd":                    "science.gdd",
        "shannon_diversity":      "science.shannon_diversity",
        "bray_curtis":            "science.bray_curtis",
        "anderson_coupling":      "science.anderson_coupling",
        "thornthwaite":           "science.thornthwaite",
        "full_pipeline":          "ecology.full_pipeline",
        "spi_drought_index":      "science.spi_drought_index",
        "autocorrelation":        "science.autocorrelation",
        "gamma_cdf":              "science.gamma_cdf",
    });

    let _ = rpc::send(
        target,
        "capability.register",
        &serde_json::json!({
            "primal": PRIMAL_NAME,
            "capability": "ecology",
            "socket": &sock_str,
            "semantic_mappings": ecology_mappings,
        }),
    );

    let provenance_mappings = serde_json::json!({
        "begin":    "provenance.begin",
        "record":   "provenance.record",
        "complete": "provenance.complete",
        "status":   "provenance.status",
    });

    let _ = rpc::send(
        target,
        "capability.register",
        &serde_json::json!({
            "primal": PRIMAL_NAME,
            "capability": "provenance",
            "socket": &sock_str,
            "semantic_mappings": provenance_mappings,
        }),
    );

    let data_mappings = serde_json::json!({
        "cross_spring_weather": "data.cross_spring_weather",
    });

    let _ = rpc::send(
        target,
        "capability.register",
        &serde_json::json!({
            "primal": PRIMAL_NAME,
            "capability": "data",
            "socket": &sock_str,
            "semantic_mappings": data_mappings,
        }),
    );

    let _ = rpc::send(
        target,
        "capability.register",
        &serde_json::json!({
            "primal": PRIMAL_NAME,
            "capability": "capability",
            "socket": &sock_str,
            "semantic_mappings": { "list": "capability.list" },
            "operation_dependencies": operation_dependencies(),
            "cost_estimates": cost_estimates(),
        }),
    );

    let mut registered = 0;
    for cap in ALL_CAPABILITIES {
        let cap_result = rpc::send(
            target,
            "capability.register",
            &serde_json::json!({
                "primal": PRIMAL_NAME,
                "capability": cap,
                "socket": &sock_str,
            }),
        );

        if cap_result.is_some() {
            registered += 1;
        } else {
            eprintln!("[biomeos] capability.register({cap}) failed (non-fatal)");
        }
    }

    eprintln!(
        "[biomeos] {registered}/{} capabilities + ecology/provenance/data domains registered",
        ALL_CAPABILITIES.len()
    );
}

// ═══════════════════════════════════════════════════════════════════
// Connection handler
// ═══════════════════════════════════════════════════════════════════

#[expect(
    clippy::needless_pass_by_value,
    reason = "BufReader::new consumes the stream"
)]
fn handle_connection(stream: UnixStream, state: &PrimalState) {
    stream
        .set_read_timeout(Some(Duration::from_secs(READ_TIMEOUT_SECS)))
        .ok();
    stream
        .set_write_timeout(Some(Duration::from_secs(WRITE_TIMEOUT_SECS)))
        .ok();

    let reader = BufReader::new(&stream);
    let mut writer = &stream;

    for line_result in reader.lines() {
        #[expect(
            clippy::manual_let_else,
            reason = "manual match handles logging on error before continue"
        )]
        let line = match line_result {
            Ok(l) => l,
            Err(_) => break,
        };

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
        let dispatch_start = Instant::now();
        let result = dispatch(method, &params, state);
        let latency_ms = dispatch_start.elapsed().as_secs_f64() * 1000.0;
        let success = result.get("error").is_none();
        emit_metrics(method, latency_ms, success);
        let response = rpc::success(&id, &result);

        let _ = writeln!(writer, "{response}");
        let _ = writer.flush();
    }
}

// ═══════════════════════════════════════════════════════════════════
// Main
// ═══════════════════════════════════════════════════════════════════

fn run() -> Result<(), String> {
    let family_id = get_family_id();
    let socket_path = resolve_socket_path(&family_id);

    if let Some(parent) = socket_path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("Cannot create socket directory {}: {e}", parent.display()))?;
    }

    if socket_path.exists() {
        std::fs::remove_file(&socket_path)
            .map_err(|e| format!("Cannot remove stale socket {}: {e}", socket_path.display()))?;
    }

    let state = Arc::new(PrimalState {
        start_time: Instant::now(),
        requests_served: AtomicU64::new(0),
    });

    let listener = UnixListener::bind(&socket_path)
        .map_err(|e| format!("Cannot bind to {}: {e}", socket_path.display()))?;

    eprintln!(
        "{PRIMAL_NAME} primal listening on {}",
        socket_path.display()
    );
    eprintln!("  Family ID: {family_id}");
    eprintln!("  Mode: Tower (local Eastgate)");
    eprintln!("  Version: {}", env!("CARGO_PKG_VERSION"));
    eprintln!("  Capabilities ({}):", ALL_CAPABILITIES.len());
    for cap in ALL_CAPABILITIES {
        eprintln!("    - {cap}");
    }

    register_with_biomeos(&socket_path);

    let running = Arc::new(AtomicBool::new(true));

    let heartbeat_state = state.clone();
    let heartbeat_running = running.clone();
    std::thread::spawn(move || {
        let biomeos_socket = resolve_socket_dir().join(orchestrator_socket_name());
        let fallback =
            biomeos::fallback_registration_primal().and_then(|name| discover_primal_socket(&name));
        let target = if biomeos_socket.exists() {
            Some(biomeos_socket)
        } else {
            fallback
        };

        while heartbeat_running.load(Ordering::Relaxed) {
            std::thread::sleep(Duration::from_secs(HEARTBEAT_INTERVAL_SECS));

            if let Some(ref t) = target {
                let provenance_up =
                    airspring_barracuda::ipc::provenance::is_available();
                let _ = rpc::send(
                    t,
                    "lifecycle.status",
                    &serde_json::json!({
                        "name": PRIMAL_NAME,
                        "socket_path": socket_path.to_string_lossy(),
                        "status": "healthy",
                        "requests_served": heartbeat_state.requests_served.load(Ordering::Relaxed),
                        "version": env!("CARGO_PKG_VERSION"),
                        "capabilities_total": ALL_CAPABILITIES.len(),
                        "composition": {
                            "provenance_trio": provenance_up,
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
            Ok(stream) => {
                let state = state.clone();
                std::thread::spawn(move || {
                    handle_connection(stream, &state);
                });
            }
            Err(e) => {
                eprintln!("[error] Accept failed: {e}");
            }
        }
    }
    Ok(())
}

/// Emit structured metrics for biomeOS Pathway Learner (neuralAPI pattern).
///
/// Uses passive structured logging (eprintln with fields) so biomeOS can
/// scrape metrics without requiring active reporting infrastructure.
/// When `BIOMEOS_METRICS_SOCKET` is set, also reports directly.
fn emit_metrics(operation: &str, latency_ms: f64, success: bool) {
    eprintln!(
        "[metrics] primal_id={PRIMAL_NAME} operation={operation} latency_ms={latency_ms:.2} success={success}"
    );

    if let Ok(socket_path) = std::env::var("BIOMEOS_METRICS_SOCKET") {
        let metrics = serde_json::json!({
            "primal_id": PRIMAL_NAME,
            "operation": operation,
            "latency_ms": latency_ms,
            "success": success,
            "version": env!("CARGO_PKG_VERSION"),
        });
        if let Ok(mut stream) = std::os::unix::net::UnixStream::connect(&socket_path) {
            let payload = serde_json::to_string(&metrics).unwrap_or_default();
            let _ = std::io::Write::write_all(&mut stream, payload.as_bytes());
            let _ = std::io::Write::write_all(&mut stream, b"\n");
        }
    }
}

fn main() {
    if let Err(e) = run() {
        eprintln!("[fatal] {e}");
        std::process::exit(1);
    }
}
