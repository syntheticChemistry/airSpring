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
//! 4. Cross-primal forwarding: airSpring → `ToadStool` health
//! 5. Cross-primal forwarding: airSpring → `BearDog` health
//! 6. Primal discovery: all NUCLEUS primals visible
//! 7. Neural-api capability.call routing to airSpring
//!
//! ## Prerequisites
//!
//! ```sh
//! FAMILY_ID=8ff3b864a4bc589a cargo run --release --bin airspring_primal
//! ```
//!
//! Provenance: `biomeOS` NUCLEUS mixed-hardware pipeline validation

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::path::PathBuf;

use airspring_barracuda::biomeos;
use airspring_barracuda::eco::evapotranspiration as et;
use airspring_barracuda::rpc;

use barracuda::validation::ValidationHarness;

fn find_socket(prefix: &str) -> Option<PathBuf> {
    biomeos::find_socket(prefix)
}

#[expect(
    clippy::too_many_lines,
    reason = "validation binary sequentially checks many baseline comparisons"
)]
fn main() {
    tracing_subscriber::fmt().with_env_filter("info").init();

    let mut v = ValidationHarness::new("Exp 063: NUCLEUS Cross-Primal Pipeline");

    let airspring = find_socket("airspring");
    v.check_bool("airspring_socket_found", airspring.is_some());
    if airspring.is_none() {
        eprintln!("ERROR: airspring_primal not running");
        v.finish();
    }
    let sock = airspring.expect("airspring socket required after socket_found check");

    // ── Phase 1: Health + Capabilities ─────────────────────────────
    let health =
        rpc::send(&sock, "health", &serde_json::json!({})).and_then(|r| r.get("result").cloned());
    v.check_bool("health_response", health.is_some());
    if let Some(ref h) = health {
        v.check_bool(
            "health_healthy",
            h.get("status").and_then(|v| v.as_str()) == Some("healthy"),
        );
        let cap_count = h
            .get("capabilities")
            .and_then(|v| v.as_array())
            .map_or(0, |a| a.len());
        // Architectural: 16 capabilities registered by airspring_primal main()
        // v0.6.0: 30 capabilities (21 science + 5 ecology + 2 primal + 1 compute + 1 data).
        // Tolerance 1.0 accommodates capability evolution across versions.
        v.check_abs("capability_count", cap_count as f64, 30.0, 1.0);
    }

    // ── Phase 2: Ecology Domain Routing ────────────────────────────
    // ecology.et0_fao56 should produce identical results to science.et0_fao56
    let test_params = serde_json::json!({
        "tmax": 32.0, "tmin": 18.0, "solar_radiation": 22.5,
        "wind_speed_2m": 1.8, "actual_vapour_pressure": 1.2,
        "day_of_year": 200, "latitude_deg": 42.727, "elevation_m": 256.0,
    });

    let science_et0 =
        rpc::send(&sock, "science.et0_fao56", &test_params).and_then(|r| r.get("result").cloned());
    let ecology_et0 =
        rpc::send(&sock, "ecology.et0_fao56", &test_params).and_then(|r| r.get("result").cloned());

    v.check_bool("science_et0_response", science_et0.is_some());
    v.check_bool("ecology_et0_response", ecology_et0.is_some());

    if let (Some(s), Some(e)) = (&science_et0, &ecology_et0) {
        let s_val = s.get("et0_mm").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let e_val = e.get("et0_mm").and_then(|v| v.as_f64()).unwrap_or(0.0);
        v.check_abs("ecology_science_et0_parity", s_val, e_val, 1e-15);
    }

    // Also verify direct Rust call parity
    let direct = et::daily_et0(&et::DailyEt0Input {
        tmax: 32.0,
        tmin: 18.0,
        tmean: None,
        solar_radiation: 22.5,
        wind_speed_2m: 1.8,
        actual_vapour_pressure: 1.2,
        day_of_year: 200,
        latitude_deg: 42.727,
        elevation_m: 256.0,
    });
    if let Some(ref e) = ecology_et0 {
        let rpc_val = e.get("et0_mm").and_then(|v| v.as_f64()).unwrap_or(0.0);
        v.check_abs("ecology_direct_rust_parity", rpc_val, direct.et0, 1e-10);
    }

    // ── Phase 3: Full Pipeline ─────────────────────────────────────
    let pipeline_result = rpc::send(
        &sock,
        "ecology.full_pipeline",
        &serde_json::json!({
            "tmax": 32.0, "tmin": 18.0, "solar_radiation": 22.5,
            "wind_speed_2m": 1.8, "actual_vapour_pressure": 1.2,
            "day_of_year": 200, "latitude_deg": 42.727, "elevation_m": 256.0,
            "kc": 1.15, "precipitation_mm": 3.5,
            "soil_water_mm": 150.0, "field_capacity_mm": 200.0, "wilting_point_mm": 50.0,
            "ky": 1.25, "max_yield_t_ha": 12.0,
        }),
    )
    .and_then(|r| r.get("result").cloned());

    v.check_bool("full_pipeline_response", pipeline_result.is_some());
    if let Some(ref p) = pipeline_result {
        let stages = p.get("stages");
        v.check_bool("pipeline_has_stages", stages.is_some());

        if let Some(stages) = stages {
            let et0_mm = stages
                .pointer("/et0/et0_mm")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0);
            v.check_lower("pipeline_et0_positive", et0_mm, 0.0);
            v.check_abs("pipeline_et0_value", et0_mm, direct.et0, 1e-10);

            let etc = stages
                .pointer("/water_balance/etc_mm")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0);
            v.check_abs("pipeline_etc_computed", etc, et0_mm * 1.15, 1e-10);

            let sw = stages
                .pointer("/water_balance/soil_water_mm")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0);
            // Bounds from pipeline soil params: WP=50 mm, FC=200 mm.
            // Margins ±0.1 mm guard against f64 rounding at boundaries.
            v.check_lower("pipeline_sw_above_wilt", sw, 49.9);
            v.check_upper("pipeline_sw_below_fc", sw, 200.1);

            let yield_ratio = stages
                .pointer("/yield/yield_ratio")
                .and_then(|v| v.as_f64())
                .unwrap_or(-1.0);
            v.check_lower("pipeline_yield_ratio_non_negative", yield_ratio, -0.01);
        }
    }

    // ── Phase 4: Capability-Based Forwarding ────────────────────────
    let compute_fwd = rpc::send(
        &sock,
        "capability.forward",
        &serde_json::json!({"capability": "compute.dispatch", "method": "health", "params": {}}),
    )
    .and_then(|r| r.get("result").cloned());
    v.check_bool("forward_compute_response", compute_fwd.is_some());
    if let Some(ref t) = compute_fwd {
        let inner = t.pointer("/response/result/healthy");
        v.check_bool(
            "compute_provider_healthy",
            inner.and_then(|v| v.as_bool()).unwrap_or(false),
        );
    }

    let crypto_fwd = rpc::send(
        &sock,
        "capability.forward",
        &serde_json::json!({"capability": "crypto.tls", "method": "health", "params": {}}),
    )
    .and_then(|r| r.get("result").cloned());
    v.check_bool("forward_crypto_response", crypto_fwd.is_some());
    if let Some(ref b) = crypto_fwd {
        let inner = b.pointer("/response/result/status");
        v.check_bool(
            "crypto_provider_healthy",
            inner.and_then(|v| v.as_str()) == Some("healthy"),
        );
    }

    // ── Phase 5: Capability Discovery ──────────────────────────────
    let discovery = rpc::send(&sock, "capability.discover", &serde_json::json!({}))
        .and_then(|r| r.get("result").cloned());
    v.check_bool("capability_discover_response", discovery.is_some());
    if let Some(ref d) = discovery {
        let count = d.get("count").and_then(|v| v.as_u64()).unwrap_or(0);
        v.check_lower("discover_multiple_capabilities", count as f64, 3.0);

        let capabilities = d.get("capabilities").and_then(|v| v.as_array());
        if let Some(caps) = capabilities {
            let names: Vec<&str> = caps.iter().filter_map(|v| v.as_str()).collect();
            v.check_bool(
                "discover_science",
                names.iter().any(|n| n.starts_with("science.")),
            );
            v.check_bool(
                "discover_crypto",
                names.iter().any(|n| n.starts_with("crypto.")),
            );
            v.check_bool(
                "discover_compute",
                names.iter().any(|n| n.starts_with("compute.")),
            );
        }
    }

    // ── Phase 6: Neural-API Capability Routing ─────────────────────
    let neural_api = find_socket("neural-api");
    v.check_bool("neural_api_socket_found", neural_api.is_some());

    if let Some(ref api_sock) = neural_api {
        let cap_result = rpc::send(
            api_sock,
            "capability.call",
            &serde_json::json!({
                "capability": "ecology",
                "operation": "et0_fao56",
                "args": {
                    "tmax": 32.0, "tmin": 18.0, "solar_radiation": 22.5,
                    "wind_speed_2m": 1.8, "actual_vapour_pressure": 1.2,
                    "day_of_year": 200, "latitude_deg": 42.727, "elevation_m": 256.0,
                },
            }),
        )
        .and_then(|r| r.get("result").cloned());

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
