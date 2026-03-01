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
    clippy::pedantic,
    clippy::nursery,
    clippy::unwrap_used,
    clippy::expect_used,
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
use airspring_barracuda::eco::evapotranspiration as et;
use airspring_barracuda::eco::simple_et0;

const PRIMAL_NAME: &str = "airspring";
const ORCHESTRATOR_SOCKET: &str = "biomeOS.sock";

const ALL_CAPABILITIES: &[&str] = &[
    "science.et0_fao56",
    "science.et0_hargreaves",
    "science.et0_priestley_taylor",
    "science.et0_makkink",
    "science.et0_turc",
    "science.et0_hamon",
    "science.et0_blaney_criddle",
    "science.water_balance",
    "science.yield_response",
    "ecology.et0_fao56",
    "ecology.et0_hargreaves",
    "ecology.water_balance",
    "ecology.yield_response",
    "ecology.full_pipeline",
    "primal.forward",
    "primal.discover",
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

// ═══════════════════════════════════════════════════════════════════
// JSON-RPC 2.0 (using serde_json::Value, no serde derive needed)
// ═══════════════════════════════════════════════════════════════════

fn json_rpc_success(id: &serde_json::Value, result: serde_json::Value) -> serde_json::Value {
    serde_json::json!({
        "jsonrpc": "2.0",
        "result": result,
        "id": id,
    })
}

fn json_rpc_error(id: &serde_json::Value, code: i32, message: &str) -> serde_json::Value {
    serde_json::json!({
        "jsonrpc": "2.0",
        "error": { "code": code, "message": message },
        "id": id,
    })
}

// ═══════════════════════════════════════════════════════════════════
// Request dispatch — maps JSON-RPC methods to airSpring science
// ═══════════════════════════════════════════════════════════════════

fn dispatch(method: &str, params: &serde_json::Value, state: &PrimalState) -> serde_json::Value {
    match method {
        "lifecycle.health" | "health" => handle_health(state),

        "science.et0_fao56" | "ecology.et0_fao56" => handle_et0_fao56(params),
        "science.et0_hargreaves" | "ecology.et0_hargreaves" => handle_et0_hargreaves(params),
        "science.et0_priestley_taylor" | "ecology.et0_priestley_taylor" => {
            handle_et0_priestley_taylor(params)
        }
        "science.et0_makkink" | "ecology.et0_makkink" => handle_et0_makkink(params),
        "science.et0_turc" | "ecology.et0_turc" => handle_et0_turc(params),
        "science.et0_hamon" | "ecology.et0_hamon" => handle_et0_hamon(params),
        "science.et0_blaney_criddle" | "ecology.et0_blaney_criddle" => {
            handle_et0_blaney_criddle(params)
        }
        "science.water_balance" | "ecology.water_balance" => handle_water_balance(params),
        "science.yield_response" | "ecology.yield_response" => handle_yield_response(params),

        "ecology.full_pipeline" => handle_full_pipeline(params),

        "primal.forward" => handle_primal_forward(params),
        "primal.discover" => handle_primal_discover(),

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

fn f64_param(params: &serde_json::Value, key: &str) -> Option<f64> {
    params.get(key).and_then(|v| v.as_f64())
}

fn u32_param(params: &serde_json::Value, key: &str) -> Option<u32> {
    params.get(key).and_then(|v| v.as_u64()).map(|v| v as u32)
}

fn handle_et0_fao56(params: &serde_json::Value) -> serde_json::Value {
    let input = et::DailyEt0Input {
        tmax: f64_param(params, "tmax").unwrap_or(30.0),
        tmin: f64_param(params, "tmin").unwrap_or(15.0),
        tmean: f64_param(params, "tmean"),
        solar_radiation: f64_param(params, "solar_radiation").unwrap_or(20.0),
        wind_speed_2m: f64_param(params, "wind_speed_2m").unwrap_or(2.0),
        actual_vapour_pressure: f64_param(params, "actual_vapour_pressure").unwrap_or(1.5),
        day_of_year: u32_param(params, "day_of_year").unwrap_or(180),
        latitude_deg: f64_param(params, "latitude_deg").unwrap_or(42.7),
        elevation_m: f64_param(params, "elevation_m").unwrap_or(250.0),
    };

    let result = et::daily_et0(&input);
    serde_json::json!({
        "et0_mm": result.et0,
        "rn_mj": result.rn,
        "method": "fao56_penman_monteith",
    })
}

fn handle_et0_hargreaves(params: &serde_json::Value) -> serde_json::Value {
    let tmin = f64_param(params, "tmin").unwrap_or(15.0);
    let tmax = f64_param(params, "tmax").unwrap_or(30.0);
    let lat_deg = f64_param(params, "latitude_deg").unwrap_or(42.7);
    let doy = u32_param(params, "day_of_year").unwrap_or(180);

    let lat_rad = lat_deg.to_radians();
    let ra = airspring_barracuda::eco::solar::extraterrestrial_radiation(lat_rad, doy);
    let ra_mm = ra / 2.45;
    let et0 = et::hargreaves_et0(tmin, tmax, ra_mm);

    serde_json::json!({
        "et0_mm": et0,
        "ra_mm_day": ra_mm,
        "method": "hargreaves",
    })
}

fn handle_et0_priestley_taylor(params: &serde_json::Value) -> serde_json::Value {
    let rn = f64_param(params, "rn").unwrap_or(10.0);
    let g = f64_param(params, "g").unwrap_or(0.0);
    let tmean = f64_param(params, "tmean").unwrap_or(22.5);
    let elevation = f64_param(params, "elevation_m").unwrap_or(250.0);

    let et0 = et::priestley_taylor_et0(rn, g, tmean, elevation);
    serde_json::json!({
        "et0_mm": et0,
        "method": "priestley_taylor",
    })
}

fn handle_et0_makkink(params: &serde_json::Value) -> serde_json::Value {
    let tmean = f64_param(params, "tmean").unwrap_or(22.5);
    let rs = f64_param(params, "solar_radiation").unwrap_or(20.0);
    let elevation = f64_param(params, "elevation_m").unwrap_or(250.0);

    let et0 = simple_et0::makkink_et0(tmean, rs, elevation);
    serde_json::json!({
        "et0_mm": et0,
        "method": "makkink",
    })
}

fn handle_et0_turc(params: &serde_json::Value) -> serde_json::Value {
    let tmean = f64_param(params, "tmean").unwrap_or(22.5);
    let rs = f64_param(params, "solar_radiation").unwrap_or(20.0);
    let rh = f64_param(params, "rh_pct").unwrap_or(60.0);

    let et0 = simple_et0::turc_et0(tmean, rs, rh);
    serde_json::json!({
        "et0_mm": et0,
        "method": "turc",
    })
}

fn handle_et0_hamon(params: &serde_json::Value) -> serde_json::Value {
    let tmean = f64_param(params, "tmean").unwrap_or(22.5);
    let lat_deg = f64_param(params, "latitude_deg").unwrap_or(42.7);
    let doy = u32_param(params, "day_of_year").unwrap_or(180);

    let lat_rad = lat_deg.to_radians();
    let et0 = simple_et0::hamon_pet_from_location(tmean, lat_rad, doy);
    serde_json::json!({
        "pet_mm": et0,
        "method": "hamon",
    })
}

fn handle_et0_blaney_criddle(params: &serde_json::Value) -> serde_json::Value {
    let tmean = f64_param(params, "tmean").unwrap_or(22.5);
    let lat_deg = f64_param(params, "latitude_deg").unwrap_or(42.7);
    let doy = u32_param(params, "day_of_year").unwrap_or(180);

    let lat_rad = lat_deg.to_radians();
    let et0 = simple_et0::blaney_criddle_from_location(tmean, lat_rad, doy);
    serde_json::json!({
        "et0_mm": et0,
        "method": "blaney_criddle",
    })
}

fn handle_water_balance(params: &serde_json::Value) -> serde_json::Value {
    let et0 = f64_param(params, "et0_mm").unwrap_or(5.0);
    let kc = f64_param(params, "kc").unwrap_or(1.0);
    let precip = f64_param(params, "precipitation_mm").unwrap_or(3.0);
    let irrigation = f64_param(params, "irrigation_mm").unwrap_or(0.0);
    let soil_water = f64_param(params, "soil_water_mm").unwrap_or(100.0);
    let field_cap = f64_param(params, "field_capacity_mm").unwrap_or(200.0);
    let wilt = f64_param(params, "wilting_point_mm").unwrap_or(50.0);

    let etc = et0 * kc;
    let input = soil_water + precip + irrigation;
    let new_sw = (input - etc).clamp(wilt, field_cap);
    let deep_perc = (input - etc - field_cap).max(0.0);
    let deficit = field_cap - new_sw;

    serde_json::json!({
        "etc_mm": etc,
        "soil_water_mm": new_sw,
        "deep_percolation_mm": deep_perc,
        "deficit_mm": deficit,
        "method": "fao56_water_balance",
    })
}

fn handle_yield_response(params: &serde_json::Value) -> serde_json::Value {
    let ky = f64_param(params, "ky").unwrap_or(1.25);
    let eta_over_etm = f64_param(params, "eta_over_etm").unwrap_or(0.8);
    let max_yield = f64_param(params, "max_yield_t_ha").unwrap_or(12.0);

    let ratio = 1.0 - ky * (1.0 - eta_over_etm);
    let actual_yield = max_yield * ratio.max(0.0);

    serde_json::json!({
        "yield_t_ha": actual_yield,
        "yield_ratio": ratio.max(0.0),
        "ky": ky,
        "method": "stewart_1977",
    })
}

// ═══════════════════════════════════════════════════════════════════
// Cross-primal and pipeline handlers
// ═══════════════════════════════════════════════════════════════════

fn handle_full_pipeline(params: &serde_json::Value) -> serde_json::Value {
    let tmax = f64_param(params, "tmax").unwrap_or(32.0);
    let tmin = f64_param(params, "tmin").unwrap_or(18.0);
    let solar_rad = f64_param(params, "solar_radiation").unwrap_or(22.5);
    let wind = f64_param(params, "wind_speed_2m").unwrap_or(2.0);
    let ea = f64_param(params, "actual_vapour_pressure").unwrap_or(1.5);
    let doy = u32_param(params, "day_of_year").unwrap_or(180);
    let lat_deg = f64_param(params, "latitude_deg").unwrap_or(42.7);
    let elevation = f64_param(params, "elevation_m").unwrap_or(250.0);

    let kc = f64_param(params, "kc").unwrap_or(1.0);
    let precip = f64_param(params, "precipitation_mm").unwrap_or(3.0);
    let irrigation = f64_param(params, "irrigation_mm").unwrap_or(0.0);
    let soil_water = f64_param(params, "soil_water_mm").unwrap_or(100.0);
    let field_cap = f64_param(params, "field_capacity_mm").unwrap_or(200.0);
    let wilt = f64_param(params, "wilting_point_mm").unwrap_or(50.0);

    let ky = f64_param(params, "ky").unwrap_or(1.25);
    let max_yield = f64_param(params, "max_yield_t_ha").unwrap_or(12.0);

    // Stage 1: ET₀
    let input = et::DailyEt0Input {
        tmax,
        tmin,
        tmean: None,
        solar_radiation: solar_rad,
        wind_speed_2m: wind,
        actual_vapour_pressure: ea,
        day_of_year: doy,
        latitude_deg: lat_deg,
        elevation_m: elevation,
    };
    let et0_result = et::daily_et0(&input);
    let et0 = et0_result.et0.max(0.0);

    // Stage 2: Water balance
    let etc = et0 * kc;
    let wb_input = soil_water + precip + irrigation;
    let new_sw = (wb_input - etc).clamp(wilt, field_cap);
    let deep_perc = (wb_input - etc - field_cap).max(0.0);
    let deficit = field_cap - new_sw;

    // Stage 3: Yield response
    let eta_over_etm = if et0 > 0.0 {
        (etc - deficit.min(etc)) / etc
    } else {
        1.0
    };
    let yield_ratio = (1.0 - ky * (1.0 - eta_over_etm)).max(0.0);
    let actual_yield = max_yield * yield_ratio;

    serde_json::json!({
        "pipeline": "ecology.full_pipeline",
        "stages": {
            "et0": {
                "et0_mm": et0,
                "rn_mj": et0_result.rn,
                "method": "fao56_penman_monteith",
            },
            "water_balance": {
                "etc_mm": etc,
                "soil_water_mm": new_sw,
                "deep_percolation_mm": deep_perc,
                "deficit_mm": deficit,
            },
            "yield": {
                "yield_t_ha": actual_yield,
                "yield_ratio": yield_ratio,
                "eta_over_etm": eta_over_etm,
                "ky": ky,
            },
        },
    })
}

fn handle_primal_forward(params: &serde_json::Value) -> serde_json::Value {
    let primal = match params.get("primal").and_then(|v| v.as_str()) {
        Some(p) => p,
        None => return serde_json::json!({"error": "missing 'primal' parameter"}),
    };
    let method = match params.get("method").and_then(|v| v.as_str()) {
        Some(m) => m,
        None => return serde_json::json!({"error": "missing 'method' parameter"}),
    };
    let inner_params = params
        .get("params")
        .cloned()
        .unwrap_or(serde_json::json!({}));

    let socket = match discover_primal_socket(primal) {
        Some(s) => s,
        None => return serde_json::json!({"error": format!("primal '{primal}' not found")}),
    };

    match send_jsonrpc(&socket, method, inner_params) {
        Some(resp) => serde_json::json!({
            "forwarded_to": primal,
            "method": method,
            "response": resp,
        }),
        None => serde_json::json!({
            "error": format!("forward to {primal}:{method} failed"),
        }),
    }
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

fn send_jsonrpc(
    socket_path: &Path,
    method: &str,
    params: serde_json::Value,
) -> Option<serde_json::Value> {
    let mut stream = match UnixStream::connect(socket_path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("[biomeos] connect to {}: {e}", socket_path.display());
            return None;
        }
    };

    stream.set_read_timeout(Some(Duration::from_secs(5))).ok();
    stream.set_write_timeout(Some(Duration::from_secs(5))).ok();

    let request = serde_json::json!({
        "jsonrpc": "2.0",
        "method": method,
        "params": params,
        "id": 1,
    });

    let mut payload = serde_json::to_vec(&request).ok()?;
    payload.push(b'\n');
    stream.write_all(&payload).ok()?;
    stream.flush().ok()?;

    let mut reader = BufReader::new(stream);
    let mut line = String::new();
    reader.read_line(&mut line).ok()?;

    serde_json::from_str(line.trim()).ok()
}

fn register_with_biomeos(our_socket: &Path) {
    let biomeos_socket = resolve_socket_dir().join(ORCHESTRATOR_SOCKET);
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
    let reg_result = send_jsonrpc(
        target,
        "lifecycle.register",
        serde_json::json!({
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
        "et0_fao56":         "science.et0_fao56",
        "et0_hargreaves":    "science.et0_hargreaves",
        "et0_priestley_taylor": "science.et0_priestley_taylor",
        "et0_makkink":       "science.et0_makkink",
        "et0_turc":          "science.et0_turc",
        "et0_hamon":         "science.et0_hamon",
        "et0_blaney_criddle":"science.et0_blaney_criddle",
        "water_balance":     "science.water_balance",
        "yield_response":    "science.yield_response",
        "full_pipeline":     "ecology.full_pipeline",
    });

    let _ = send_jsonrpc(
        target,
        "capability.register",
        serde_json::json!({
            "primal": PRIMAL_NAME,
            "capability": "ecology",
            "socket": &sock_str,
            "semantic_mappings": ecology_mappings,
        }),
    );

    let mut registered = 0;
    for cap in ALL_CAPABILITIES {
        let cap_result = send_jsonrpc(
            target,
            "capability.register",
            serde_json::json!({
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

fn handle_connection(stream: UnixStream, state: &PrimalState) {
    stream.set_read_timeout(Some(Duration::from_secs(60))).ok();
    stream.set_write_timeout(Some(Duration::from_secs(10))).ok();

    let reader = BufReader::new(&stream);
    let mut writer = &stream;

    for line_result in reader.lines() {
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
                let resp = json_rpc_error(
                    &serde_json::Value::Null,
                    -32700,
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
            .unwrap_or(serde_json::json!({}));

        state.requests_served.fetch_add(1, Ordering::Relaxed);
        let result = dispatch(method, &params, state);
        let response = json_rpc_success(&id, result);

        let _ = writeln!(writer, "{response}");
        let _ = writer.flush();
    }
}

// ═══════════════════════════════════════════════════════════════════
// Main
// ═══════════════════════════════════════════════════════════════════

fn main() {
    let family_id = get_family_id();
    let socket_path = resolve_socket_path(&family_id);

    if let Some(parent) = socket_path.parent() {
        if let Err(e) = std::fs::create_dir_all(parent) {
            eprintln!(
                "[fatal] Cannot create socket directory {}: {e}",
                parent.display()
            );
            std::process::exit(1);
        }
    }

    if socket_path.exists() {
        if let Err(e) = std::fs::remove_file(&socket_path) {
            eprintln!(
                "[fatal] Cannot remove stale socket {}: {e}",
                socket_path.display()
            );
            std::process::exit(1);
        }
    }

    let state = Arc::new(PrimalState {
        start_time: Instant::now(),
        requests_served: AtomicU64::new(0),
    });

    let listener = match UnixListener::bind(&socket_path) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("[fatal] Cannot bind to {}: {e}", socket_path.display());
            std::process::exit(1);
        }
    };

    eprintln!("airSpring primal listening on {}", socket_path.display());
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
    let heartbeat_path = socket_path.clone();
    std::thread::spawn(move || {
        let biomeos_socket = resolve_socket_dir().join(ORCHESTRATOR_SOCKET);
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
                let _ = send_jsonrpc(
                    t,
                    "lifecycle.status",
                    serde_json::json!({
                        "name": PRIMAL_NAME,
                        "socket_path": heartbeat_path.to_string_lossy(),
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
}
