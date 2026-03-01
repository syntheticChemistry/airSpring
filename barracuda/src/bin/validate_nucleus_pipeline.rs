// SPDX-License-Identifier: AGPL-3.0-or-later

//! Exp 063: NUCLEUS Cross-Primal Pipeline Validation
//!
//! End-to-end validation that airSpring science flows correctly through
//! the biomeOS NUCLEUS ecosystem, including cross-primal interactions.
//!
//! ## What this validates
//!
//! 1. airSpring primal health + 16 registered capabilities
//! 2. Ecology domain routing (ecology.* methods match science.* parity)
//! 3. Full pipeline: ET₀ → Water Balance → Yield (single JSON-RPC call)
//! 4. Cross-primal forwarding: airSpring → ToadStool health
//! 5. Cross-primal forwarding: airSpring → BearDog health
//! 6. Primal discovery: all NUCLEUS primals visible
//! 7. Neural-api capability.call routing to airSpring
//!
//! ## Prerequisites
//!
//! ```sh
//! FAMILY_ID=8ff3b864a4bc589a cargo run --release --bin airspring_primal
//! ```

#![allow(
    clippy::pedantic,
    clippy::nursery,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation
)]

use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::UnixStream;
use std::path::PathBuf;
use std::time::Duration;

use airspring_barracuda::eco::evapotranspiration as et;

use barracuda::validation::ValidationHarness;

fn resolve_socket_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("BIOMEOS_SOCKET_DIR") {
        return PathBuf::from(dir);
    }
    if let Ok(xdg) = std::env::var("XDG_RUNTIME_DIR") {
        return PathBuf::from(xdg).join("biomeos");
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        if let Ok(meta) = std::fs::metadata("/proc/self") {
            let uid = meta.uid();
            let dir = PathBuf::from(format!("/run/user/{uid}/biomeos"));
            if dir.parent().is_some_and(std::path::Path::exists) {
                return dir;
            }
        }
    }
    std::env::temp_dir().join("biomeos")
}

fn find_socket(prefix: &str) -> Option<PathBuf> {
    let socket_dir = resolve_socket_dir();
    if let Ok(entries) = std::fs::read_dir(&socket_dir) {
        for entry in entries.flatten() {
            let name = entry.file_name();
            let s = name.to_string_lossy();
            if s.starts_with(prefix) && s.ends_with(".sock") {
                return Some(entry.path());
            }
        }
    }
    None
}

fn send_jsonrpc(
    socket_path: &std::path::Path,
    method: &str,
    params: serde_json::Value,
) -> Option<serde_json::Value> {
    let mut stream = UnixStream::connect(socket_path).ok()?;
    stream.set_read_timeout(Some(Duration::from_secs(10))).ok();
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

    let resp: serde_json::Value = serde_json::from_str(line.trim()).ok()?;
    resp.get("result").cloned()
}

fn main() {
    tracing_subscriber::fmt().with_env_filter("info").init();

    let mut v = ValidationHarness::new("Exp 063: NUCLEUS Cross-Primal Pipeline");

    let airspring = find_socket("airspring");
    v.check_bool("airspring_socket_found", airspring.is_some());
    if airspring.is_none() {
        eprintln!("ERROR: airspring_primal not running");
        v.finish();
    }
    let sock = airspring.unwrap();

    // ── Phase 1: Health + Capabilities ─────────────────────────────
    let health = send_jsonrpc(&sock, "health", serde_json::json!({}));
    v.check_bool("health_response", health.is_some());
    if let Some(ref h) = health {
        v.check_bool("health_healthy", h.get("status").and_then(|v| v.as_str()) == Some("healthy"));
        let cap_count = h.get("capabilities").and_then(|v| v.as_array()).map_or(0, |a| a.len());
        v.check_abs("capability_count", cap_count as f64, 16.0, 0.5);
    }

    // ── Phase 2: Ecology Domain Routing ────────────────────────────
    // ecology.et0_fao56 should produce identical results to science.et0_fao56
    let test_params = serde_json::json!({
        "tmax": 32.0, "tmin": 18.0, "solar_radiation": 22.5,
        "wind_speed_2m": 1.8, "actual_vapour_pressure": 1.2,
        "day_of_year": 200, "latitude_deg": 42.727, "elevation_m": 256.0,
    });

    let science_et0 = send_jsonrpc(&sock, "science.et0_fao56", test_params.clone());
    let ecology_et0 = send_jsonrpc(&sock, "ecology.et0_fao56", test_params);

    v.check_bool("science_et0_response", science_et0.is_some());
    v.check_bool("ecology_et0_response", ecology_et0.is_some());

    if let (Some(ref s), Some(ref e)) = (&science_et0, &ecology_et0) {
        let s_val = s.get("et0_mm").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let e_val = e.get("et0_mm").and_then(|v| v.as_f64()).unwrap_or(0.0);
        v.check_abs("ecology_science_et0_parity", s_val, e_val, 1e-15);
    }

    // Also verify direct Rust call parity
    let direct = et::daily_et0(&et::DailyEt0Input {
        tmax: 32.0, tmin: 18.0, tmean: None,
        solar_radiation: 22.5, wind_speed_2m: 1.8, actual_vapour_pressure: 1.2,
        day_of_year: 200, latitude_deg: 42.727, elevation_m: 256.0,
    });
    if let Some(ref e) = ecology_et0 {
        let rpc_val = e.get("et0_mm").and_then(|v| v.as_f64()).unwrap_or(0.0);
        v.check_abs("ecology_direct_rust_parity", rpc_val, direct.et0, 1e-10);
    }

    // ── Phase 3: Full Pipeline ─────────────────────────────────────
    let pipeline_result = send_jsonrpc(
        &sock,
        "ecology.full_pipeline",
        serde_json::json!({
            "tmax": 32.0, "tmin": 18.0, "solar_radiation": 22.5,
            "wind_speed_2m": 1.8, "actual_vapour_pressure": 1.2,
            "day_of_year": 200, "latitude_deg": 42.727, "elevation_m": 256.0,
            "kc": 1.15, "precipitation_mm": 3.5,
            "soil_water_mm": 150.0, "field_capacity_mm": 200.0, "wilting_point_mm": 50.0,
            "ky": 1.25, "max_yield_t_ha": 12.0,
        }),
    );

    v.check_bool("full_pipeline_response", pipeline_result.is_some());
    if let Some(ref p) = pipeline_result {
        let stages = p.get("stages");
        v.check_bool("pipeline_has_stages", stages.is_some());

        if let Some(stages) = stages {
            let et0_mm = stages.pointer("/et0/et0_mm").and_then(|v| v.as_f64()).unwrap_or(0.0);
            v.check_lower("pipeline_et0_positive", et0_mm, 0.0);
            v.check_abs("pipeline_et0_value", et0_mm, direct.et0, 1e-10);

            let etc = stages.pointer("/water_balance/etc_mm").and_then(|v| v.as_f64()).unwrap_or(0.0);
            v.check_abs("pipeline_etc_computed", etc, et0_mm * 1.15, 1e-10);

            let sw = stages.pointer("/water_balance/soil_water_mm").and_then(|v| v.as_f64()).unwrap_or(0.0);
            v.check_lower("pipeline_sw_above_wilt", sw, 49.9);
            v.check_upper("pipeline_sw_below_fc", sw, 200.1);

            let yield_ratio = stages.pointer("/yield/yield_ratio").and_then(|v| v.as_f64()).unwrap_or(-1.0);
            v.check_lower("pipeline_yield_ratio_non_negative", yield_ratio, -0.01);
        }
    }

    // ── Phase 4: Cross-Primal Forwarding ───────────────────────────
    let toadstool_fwd = send_jsonrpc(
        &sock,
        "primal.forward",
        serde_json::json!({"primal": "toadstool", "method": "toadstool.health", "params": {}}),
    );
    v.check_bool("forward_toadstool_response", toadstool_fwd.is_some());
    if let Some(ref t) = toadstool_fwd {
        let inner = t.pointer("/response/result/healthy");
        v.check_bool("toadstool_healthy", inner.and_then(|v| v.as_bool()).unwrap_or(false));
    }

    let beardog_fwd = send_jsonrpc(
        &sock,
        "primal.forward",
        serde_json::json!({"primal": "beardog", "method": "health", "params": {}}),
    );
    v.check_bool("forward_beardog_response", beardog_fwd.is_some());
    if let Some(ref b) = beardog_fwd {
        let inner = b.pointer("/response/result/status");
        v.check_bool("beardog_healthy", inner.and_then(|v| v.as_str()) == Some("healthy"));
    }

    // ── Phase 5: Primal Discovery ──────────────────────────────────
    let discovery = send_jsonrpc(&sock, "primal.discover", serde_json::json!({}));
    v.check_bool("primal_discover_response", discovery.is_some());
    if let Some(ref d) = discovery {
        let count = d.get("count").and_then(|v| v.as_u64()).unwrap_or(0);
        v.check_lower("discover_multiple_primals", count as f64, 3.0);

        let primals = d.get("primals").and_then(|v| v.as_array());
        if let Some(primals) = primals {
            let names: Vec<&str> = primals.iter().filter_map(|v| v.as_str()).collect();
            v.check_bool("discover_airspring", names.contains(&"airspring"));
            v.check_bool("discover_beardog", names.contains(&"beardog"));
            v.check_bool("discover_toadstool", names.iter().any(|n| n.starts_with("toadstool")));
        }
    }

    // ── Phase 6: Neural-API Capability Routing ─────────────────────
    let neural_api = find_socket("neural-api");
    v.check_bool("neural_api_socket_found", neural_api.is_some());

    if let Some(ref api_sock) = neural_api {
        let cap_result = send_jsonrpc(
            api_sock,
            "capability.call",
            serde_json::json!({
                "capability": "ecology",
                "operation": "et0_fao56",
                "args": {
                    "tmax": 32.0, "tmin": 18.0, "solar_radiation": 22.5,
                    "wind_speed_2m": 1.8, "actual_vapour_pressure": 1.2,
                    "day_of_year": 200, "latitude_deg": 42.727, "elevation_m": 256.0,
                },
            }),
        );

        v.check_bool("neural_api_capability_call", cap_result.is_some());
        if let Some(ref r) = cap_result {
            let et0 = r.get("et0_mm").and_then(|v| v.as_f64()).unwrap_or(0.0);
            v.check_abs("neural_api_et0_parity", et0, direct.et0, 1e-10);
            eprintln!(
                "  Neural-API capability.call(ecology.et0_fao56): {:.6} mm (routed to airSpring)",
                et0
            );
        }
    }

    v.finish();
}
