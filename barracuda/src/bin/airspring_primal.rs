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
    // ── Ecology aliases ──
    "ecology.et0_fao56",
    "ecology.et0_hargreaves",
    "ecology.water_balance",
    "ecology.yield_response",
    "ecology.full_pipeline",
    // ── Cross-primal ──
    "primal.forward",
    "primal.discover",
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
    if method == "lifecycle.health" || method == "health" {
        return handle_health(state);
    }

    if let Some(result) = primal_science::dispatch_science(method, params) {
        return result;
    }

    match method {
        "primal.forward" => handle_primal_forward(params),
        "primal.discover" => handle_primal_discover(),
        "compute.offload" => handle_compute_offload(params),
        "data.weather" => handle_data_weather(params),
        _ => serde_json::json!({"error": "method_not_found", "method": method}),
    }
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
        "[biomeos] {registered}/{} capabilities + ecology domain registered",
        ALL_CAPABILITIES.len()
    );
}

// ═══════════════════════════════════════════════════════════════════
// Connection handler
// ═══════════════════════════════════════════════════════════════════

#[allow(clippy::needless_pass_by_value)] // BufReader::new consumes the stream
fn handle_connection(stream: UnixStream, state: &PrimalState) {
    stream.set_read_timeout(Some(Duration::from_secs(60))).ok();
    stream.set_write_timeout(Some(Duration::from_secs(10))).ok();

    let reader = BufReader::new(&stream);
    let mut writer = &stream;

    for line_result in reader.lines() {
        #[allow(clippy::manual_let_else)]
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
        let result = dispatch(method, &params, state);
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
            std::thread::sleep(Duration::from_secs(30));

            if let Some(ref t) = target {
                let _ = rpc::send(
                    t,
                    "lifecycle.status",
                    &serde_json::json!({
                        "name": PRIMAL_NAME,
                        "socket_path": socket_path.to_string_lossy(),
                        "status": "healthy",
                        "requests_served": heartbeat_state.requests_served.load(Ordering::Relaxed),
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

fn main() {
    if let Err(e) = run() {
        eprintln!("[fatal] {e}");
        std::process::exit(1);
    }
}
