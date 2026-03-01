// SPDX-License-Identifier: AGPL-3.0-or-later

//! Exp 062: NUCLEUS Integration Validation
//!
//! Validates that airSpring's science capabilities are correctly accessible
//! through the biomeOS NUCLEUS ecosystem via Unix domain socket JSON-RPC.
//!
//! ## What this validates
//!
//! 1. Socket discovery — finds `airspring-*.sock` in biomeOS runtime dir
//! 2. JSON-RPC health — primal reports healthy with expected capabilities
//! 3. Science parity — JSON-RPC results match direct Rust calls exactly
//! 4. Cross-primal discovery — can find other NUCLEUS primals
//!
//! ## Prerequisites
//!
//! The `airspring_primal` binary must be running:
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

use airspring_barracuda::biomeos;
use airspring_barracuda::eco::evapotranspiration as et;
use airspring_barracuda::eco::simple_et0;

use barracuda::validation::ValidationHarness;

fn resolve_socket_dir() -> PathBuf {
    biomeos::resolve_socket_dir()
}

fn find_airspring_socket() -> Option<PathBuf> {
    biomeos::find_socket("airspring")
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

    let mut v = ValidationHarness::new("Exp 062: NUCLEUS Integration Validation");

    // ── Phase 1: Socket Discovery ──────────────────────────────────
    let socket_dir = resolve_socket_dir();
    v.check_bool("socket_dir_exists", socket_dir.exists());

    let socket_path = find_airspring_socket();
    let socket_found = socket_path.is_some();
    v.check_bool("airspring_socket_found", socket_found);

    if !socket_found {
        eprintln!("ERROR: airspring_primal is not running. Start it with:");
        eprintln!("  FAMILY_ID=8ff3b864a4bc589a cargo run --release --bin airspring_primal");
        v.finish();
    }

    let socket = socket_path.unwrap();
    eprintln!("  Found airSpring socket: {}", socket.display());

    // ── Phase 2: Health Check ──────────────────────────────────────
    let health = send_jsonrpc(&socket, "health", serde_json::json!({}));
    v.check_bool("health_response", health.is_some());

    if let Some(ref h) = health {
        v.check_bool(
            "health_status_healthy",
            h.get("status").and_then(|v| v.as_str()) == Some("healthy"),
        );
        v.check_bool(
            "health_primal_name",
            h.get("primal").and_then(|v| v.as_str()) == Some("airspring"),
        );
        v.check_bool(
            "health_version",
            h.get("version").and_then(|v| v.as_str()) == Some(env!("CARGO_PKG_VERSION")),
        );

        let cap_count = h
            .get("capabilities")
            .and_then(|v| v.as_array())
            .map_or(0, |a| a.len());
        v.check_abs("capability_count", cap_count as f64, 9.0, 0.5);
    }

    // ── Phase 3: FAO-56 ET₀ Parity ────────────────────────────────
    let test_input = et::DailyEt0Input {
        tmax: 32.0,
        tmin: 18.0,
        tmean: None,
        solar_radiation: 22.5,
        wind_speed_2m: 1.8,
        actual_vapour_pressure: 1.2,
        day_of_year: 200,
        latitude_deg: 42.727,
        elevation_m: 256.0,
    };

    let direct_result = et::daily_et0(&test_input);
    let rpc_result = send_jsonrpc(
        &socket,
        "science.et0_fao56",
        serde_json::json!({
            "tmax": 32.0,
            "tmin": 18.0,
            "solar_radiation": 22.5,
            "wind_speed_2m": 1.8,
            "actual_vapour_pressure": 1.2,
            "day_of_year": 200,
            "latitude_deg": 42.727,
            "elevation_m": 256.0,
        }),
    );

    v.check_bool("et0_fao56_response", rpc_result.is_some());
    if let Some(ref r) = rpc_result {
        let rpc_et0 = r.get("et0_mm").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let rpc_rn = r.get("rn_mj").and_then(|v| v.as_f64()).unwrap_or(0.0);

        v.check_abs("et0_fao56_parity", rpc_et0, direct_result.et0, 1e-10);
        v.check_abs("et0_rn_parity", rpc_rn, direct_result.rn, 1e-10);
        eprintln!(
            "  FAO-56 ET₀: {:.6} mm (direct) vs {:.6} mm (JSON-RPC)",
            direct_result.et0, rpc_et0
        );
    }

    // ── Phase 4: Hargreaves ET₀ Parity ─────────────────────────────
    let lat_rad = 42.727_f64.to_radians();
    let ra = airspring_barracuda::eco::solar::extraterrestrial_radiation(lat_rad, 200);
    let ra_mm = ra / 2.45;
    let direct_hg = et::hargreaves_et0(18.0, 32.0, ra_mm);

    let rpc_hg = send_jsonrpc(
        &socket,
        "science.et0_hargreaves",
        serde_json::json!({
            "tmin": 18.0,
            "tmax": 32.0,
            "latitude_deg": 42.727,
            "day_of_year": 200,
        }),
    );

    v.check_bool("et0_hargreaves_response", rpc_hg.is_some());
    if let Some(ref r) = rpc_hg {
        let rpc_val = r.get("et0_mm").and_then(|v| v.as_f64()).unwrap_or(0.0);
        v.check_abs("et0_hargreaves_parity", rpc_val, direct_hg, 1e-10);
        eprintln!(
            "  Hargreaves ET₀: {:.6} mm (direct) vs {:.6} mm (JSON-RPC)",
            direct_hg, rpc_val
        );
    }

    // ── Phase 5: Simplified ET₀ Methods Parity ─────────────────────
    let direct_makkink = simple_et0::makkink_et0(22.5, 20.0, 250.0);
    let rpc_mak = send_jsonrpc(
        &socket,
        "science.et0_makkink",
        serde_json::json!({"tmean": 22.5, "solar_radiation": 20.0, "elevation_m": 250.0}),
    );
    v.check_bool("et0_makkink_response", rpc_mak.is_some());
    if let Some(ref r) = rpc_mak {
        let rpc_val = r.get("et0_mm").and_then(|v| v.as_f64()).unwrap_or(0.0);
        v.check_abs("et0_makkink_parity", rpc_val, direct_makkink, 1e-10);
    }

    let direct_turc = simple_et0::turc_et0(22.5, 20.0, 60.0);
    let rpc_turc = send_jsonrpc(
        &socket,
        "science.et0_turc",
        serde_json::json!({"tmean": 22.5, "solar_radiation": 20.0, "rh_pct": 60.0}),
    );
    v.check_bool("et0_turc_response", rpc_turc.is_some());
    if let Some(ref r) = rpc_turc {
        let rpc_val = r.get("et0_mm").and_then(|v| v.as_f64()).unwrap_or(0.0);
        v.check_abs("et0_turc_parity", rpc_val, direct_turc, 1e-10);
    }

    let direct_hamon = simple_et0::hamon_pet_from_location(22.5, lat_rad, 200);
    let rpc_ham = send_jsonrpc(
        &socket,
        "science.et0_hamon",
        serde_json::json!({"tmean": 22.5, "latitude_deg": 42.727, "day_of_year": 200}),
    );
    v.check_bool("et0_hamon_response", rpc_ham.is_some());
    if let Some(ref r) = rpc_ham {
        let rpc_val = r.get("pet_mm").and_then(|v| v.as_f64()).unwrap_or(0.0);
        v.check_abs("et0_hamon_parity", rpc_val, direct_hamon, 1e-10);
    }

    let direct_bc = simple_et0::blaney_criddle_from_location(22.5, lat_rad, 200);
    let rpc_bc = send_jsonrpc(
        &socket,
        "science.et0_blaney_criddle",
        serde_json::json!({"tmean": 22.5, "latitude_deg": 42.727, "day_of_year": 200}),
    );
    v.check_bool("et0_blaney_criddle_response", rpc_bc.is_some());
    if let Some(ref r) = rpc_bc {
        let rpc_val = r.get("et0_mm").and_then(|v| v.as_f64()).unwrap_or(0.0);
        v.check_abs("et0_blaney_criddle_parity", rpc_val, direct_bc, 1e-10);
    }

    // ── Phase 6: Water Balance Parity ──────────────────────────────
    let rpc_wb = send_jsonrpc(
        &socket,
        "science.water_balance",
        serde_json::json!({
            "et0_mm": 6.0,
            "kc": 1.15,
            "precipitation_mm": 2.5,
            "soil_water_mm": 150.0,
            "field_capacity_mm": 200.0,
            "wilting_point_mm": 50.0,
        }),
    );

    v.check_bool("water_balance_response", rpc_wb.is_some());
    if let Some(ref r) = rpc_wb {
        let etc = r.get("etc_mm").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let expected_etc: f64 = 6.0 * 1.15;
        v.check_abs("wb_etc_parity", etc, expected_etc, 1e-10);

        let sw = r
            .get("soil_water_mm")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);
        let expected_sw: f64 = (150.0_f64 + 2.5 - expected_etc).clamp(50.0, 200.0);
        v.check_abs("wb_soil_water_parity", sw, expected_sw, 1e-10);
    }

    // ── Phase 7: Yield Response Parity ─────────────────────────────
    let rpc_yr = send_jsonrpc(
        &socket,
        "science.yield_response",
        serde_json::json!({"ky": 1.25, "eta_over_etm": 0.75, "max_yield_t_ha": 12.0}),
    );

    v.check_bool("yield_response_response", rpc_yr.is_some());
    if let Some(ref r) = rpc_yr {
        let yield_val = r.get("yield_t_ha").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let expected_ratio: f64 = 1.0 - 1.25 * (1.0 - 0.75);
        let expected_yield: f64 = 12.0 * expected_ratio;
        v.check_abs("yield_parity", yield_val, expected_yield, 1e-10);
    }

    // ── Phase 8: Cross-Primal Discovery ────────────────────────────
    let primal_names_env = std::env::var("BIOMEOS_EXPECTED_PRIMALS")
        .unwrap_or_else(|_| "beardog,songbird,squirrel,toadstool".to_string());
    let primals: Vec<&str> = primal_names_env.split(',').map(str::trim).collect();
    let discovered = biomeos::discover_all_primals();
    for name in &primals {
        let discoverable = discovered.iter().any(|d| d == name);
        v.check_bool(&format!("primal_{name}_discoverable"), discoverable);
    }

    v.finish();
}
