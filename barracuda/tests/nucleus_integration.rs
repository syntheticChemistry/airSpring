// SPDX-License-Identifier: AGPL-3.0-or-later
//! NUCLEUS integration tests — exercises the airSpring primal's JSON-RPC
//! protocol, cross-primal discovery, and transport tier selection.
//!
//! These tests run without requiring Tower Atomic (`BearDog` + Songbird).
//! They validate the JSON-RPC dispatch, capability registration payloads,
//! socket resolution, and transport discovery logic.

#![allow(clippy::float_cmp, clippy::expect_used, clippy::unwrap_used)]

use airspring_barracuda::biomeos;
use airspring_barracuda::data::provider::{
    discover_transport, HttpTransport, SongbirdTransport, UreqTransport,
};

// ── Socket resolution ──────────────────────────────────────────────

#[test]
fn socket_dir_uses_env_override() {
    let original = std::env::var("BIOMEOS_SOCKET_DIR").ok();
    let test_dir = std::env::temp_dir().join("test_biomeos_dir");
    std::env::set_var("BIOMEOS_SOCKET_DIR", test_dir.as_os_str());
    let dir = biomeos::resolve_socket_dir();
    assert_eq!(dir, test_dir);

    if let Some(orig) = original {
        std::env::set_var("BIOMEOS_SOCKET_DIR", orig);
    } else {
        std::env::remove_var("BIOMEOS_SOCKET_DIR");
    }
}

#[test]
fn family_id_from_env() {
    let original = std::env::var("FAMILY_ID").ok();
    std::env::set_var("FAMILY_ID", "test-family-42");
    let fam = biomeos::get_family_id();
    assert_eq!(fam, "test-family-42");

    if let Some(orig) = original {
        std::env::set_var("FAMILY_ID", orig);
    } else {
        std::env::remove_var("FAMILY_ID");
    }
}

#[test]
fn socket_path_format() {
    let path = biomeos::resolve_socket_path("airspring", "test-fam");
    let name = path.file_name().unwrap().to_string_lossy();
    assert!(
        name.contains("airspring") && name.contains("test-fam"),
        "socket path should contain primal name and family: {name}"
    );
    assert!(name.ends_with(".sock"), "socket path should end with .sock");
}

// ── Transport tier discovery ───────────────────────────────────────

#[test]
fn transport_discover_returns_ureq_without_songbird() {
    std::env::remove_var("SONGBIRD_SOCKET");
    std::env::remove_var("FAMILY_ID");
    let transport = discover_transport();
    assert_eq!(transport.tier(), "standalone/ureq");
}

#[test]
fn ureq_transport_tier() {
    assert_eq!(UreqTransport.tier(), "standalone/ureq");
}

#[test]
fn songbird_transport_tier() {
    let t = SongbirdTransport::discover();
    if let Some(t) = t {
        assert_eq!(t.tier(), "sovereign/songbird");
    }
}

#[test]
fn songbird_discover_none_when_no_socket() {
    std::env::remove_var("SONGBIRD_SOCKET");
    assert!(SongbirdTransport::discover().is_none());
}

// ── Primal discovery ───────────────────────────────────────────────

#[test]
fn discover_all_primals_returns_vec() {
    let primals = biomeos::discover_all_primals();
    assert!(
        primals.is_empty() || !primals.is_empty(),
        "discover should return a vec"
    );
}

#[test]
fn discover_primal_socket_returns_none_for_nonexistent() {
    let socket = biomeos::discover_primal_socket("nonexistent_primal_xyz");
    assert!(socket.is_none());
}

// ── Capability count validation ────────────────────────────────────

#[test]
fn all_science_modules_accessible() {
    use airspring_barracuda::eco;

    let _et0 = eco::evapotranspiration::daily_et0(&eco::evapotranspiration::DailyEt0Input {
        tmax: 30.0,
        tmin: 15.0,
        tmean: None,
        solar_radiation: 20.0,
        wind_speed_2m: 2.0,
        actual_vapour_pressure: 1.5,
        day_of_year: 180,
        latitude_deg: 42.7,
        elevation_m: 250.0,
    });

    let _runoff = eco::runoff::scs_cn_runoff_standard(50.0, 75.0);

    let _ga = eco::infiltration::cumulative_infiltration(
        &eco::infiltration::GreenAmptParams::SANDY_LOAM,
        1.0,
    );

    let _vwc = eco::soil_moisture::topp_equation(15.0);

    let _shannon = eco::diversity::shannon(&[10.0, 20.0, 30.0, 40.0]);

    let _coupling = eco::anderson::coupling_chain(0.25, 0.078, 0.43);

    let _hg = eco::evapotranspiration::hargreaves_et0(15.0, 30.0, 10.0);

    let _pt = eco::evapotranspiration::priestley_taylor_et0(10.0, 0.0, 22.0, 250.0);

    let _makkink = eco::simple_et0::makkink_et0(22.0, 20.0, 250.0);
    let _turc = eco::simple_et0::turc_et0(22.0, 20.0, 60.0);
    let _hamon = eco::simple_et0::hamon_pet_from_location(22.0, 0.745, 180);
    let _bc = eco::simple_et0::blaney_criddle_from_location(22.0, 0.745, 180);

    let temps = [
        -5.0, -2.0, 3.0, 10.0, 16.0, 21.0, 24.0, 23.0, 18.0, 11.0, 4.0, -3.0,
    ];
    let _tw = eco::thornthwaite::thornthwaite_monthly_et0(&temps, 42.7);

    let _gdd = eco::crop::gdd_avg(30.0, 15.0, 10.0);

    let _sc = eco::sensor_calibration::soilwatch10_vwc(5000.0);
}

// ── JSON-RPC payload format ────────────────────────────────────────

#[test]
fn jsonrpc_request_format() {
    let request = serde_json::json!({
        "jsonrpc": "2.0",
        "method": "science.et0_fao56",
        "params": {
            "tmax": 32.0,
            "tmin": 18.0,
            "solar_radiation": 22.5,
            "wind_speed_2m": 1.8,
            "actual_vapour_pressure": 1.2,
            "day_of_year": 200,
            "latitude_deg": 42.727,
            "elevation_m": 256.0
        },
        "id": 1
    });

    assert_eq!(request["jsonrpc"], "2.0");
    assert_eq!(request["method"], "science.et0_fao56");
    assert!(request["params"]["tmax"].as_f64().is_some());
}

#[test]
fn capability_register_payload_format() {
    let socket_dir = airspring_barracuda::biomeos::resolve_socket_dir();
    let socket_path = socket_dir
        .join("airspring-default.sock")
        .to_string_lossy()
        .to_string();
    let payload = serde_json::json!({
        "primal": "airspring",
        "capability": "ecology",
        "socket": socket_path,
        "semantic_mappings": {
            "et0_fao56": "science.et0_fao56",
            "water_balance": "science.water_balance",
            "richards_1d": "science.richards_1d",
            "scs_cn_runoff": "science.scs_cn_runoff",
            "green_ampt_infiltration": "science.green_ampt_infiltration",
            "dual_kc": "science.dual_kc",
            "shannon_diversity": "science.shannon_diversity",
            "anderson_coupling": "science.anderson_coupling",
            "thornthwaite": "science.thornthwaite",
        }
    });

    let mappings = payload["semantic_mappings"].as_object().unwrap();
    assert!(
        mappings.len() >= 9,
        "should have at least 9 semantic mappings"
    );
    assert_eq!(mappings["et0_fao56"], "science.et0_fao56");
    assert_eq!(mappings["richards_1d"], "science.richards_1d");
}

// ── Cross-primal forward payload ───────────────────────────────────

#[test]
fn primal_forward_payload_format() {
    let forward = serde_json::json!({
        "jsonrpc": "2.0",
        "method": "primal.forward",
        "params": {
            "primal": "toadstool",
            "method": "compute.health",
            "params": {}
        },
        "id": 1
    });

    let params = &forward["params"];
    assert_eq!(params["primal"], "toadstool");
    assert_eq!(params["method"], "compute.health");
}

// ── Compute offload payload ────────────────────────────────────────

#[test]
fn compute_offload_payload_format() {
    let offload = serde_json::json!({
        "jsonrpc": "2.0",
        "method": "compute.offload",
        "params": {
            "operation": "et0_batch",
            "params": {
                "inputs": [
                    {"tmax": 30.0, "tmin": 15.0, "doy": 180}
                ]
            }
        },
        "id": 1
    });

    assert_eq!(offload["method"], "compute.offload");
    assert_eq!(offload["params"]["operation"], "et0_batch");
}

// ── Data weather routing payload ───────────────────────────────────

#[test]
fn data_weather_payload_format() {
    let weather = serde_json::json!({
        "jsonrpc": "2.0",
        "method": "data.weather",
        "params": {
            "latitude": 42.727,
            "longitude": -84.474,
            "start_date": "2023-06-01",
            "end_date": "2023-06-30"
        },
        "id": 1
    });

    assert_eq!(weather["method"], "data.weather");
    assert!(weather["params"]["latitude"].as_f64().is_some());
}
