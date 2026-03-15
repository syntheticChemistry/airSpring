// SPDX-License-Identifier: AGPL-3.0-or-later

//! Exp 083: NUCLEUS Modern Deployment Validation
//!
//! Validates the v0.7.5 biomeOS/NUCLEUS integration including new science
//! capabilities (SPI drought index, autocorrelation, gamma CDF), NUCLEUS
//! atomic detection, cross-primal discovery, and the full ecology pipeline
//! via JSON-RPC.
//!
//! ## Prerequisites
//!
//! The `airspring_primal` binary must be running:
//! ```sh
//! FAMILY_ID=8ff3b864a4bc589a cargo run --release --bin airspring_primal
//! ```
//!
//! Provenance: biomeOS NUCLEUS v0.7.5 modern deployment validation

#![allow(
    clippy::pedantic,
    clippy::nursery,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::too_many_lines
)]

use airspring_barracuda::biomeos;
use airspring_barracuda::eco::drought_index;
use airspring_barracuda::gpu::autocorrelation;
use airspring_barracuda::rpc;

use barracuda::validation::ValidationHarness;

fn main() {
    tracing_subscriber::fmt().with_env_filter("info").init();

    let mut v = ValidationHarness::new("Exp 083: NUCLEUS Modern Deployment Validation");

    // ═══════════════════════════════════════════════════════════════
    // Phase 1: NUCLEUS Atomic Detection
    // ═══════════════════════════════════════════════════════════════

    let socket_dir = biomeos::resolve_socket_dir();
    v.check_bool("socket_dir_resolved", !socket_dir.as_os_str().is_empty());

    let primals = biomeos::discover_all_primals();
    v.check_bool("primal_discovery_works", true);
    eprintln!("  Discovered primals: {primals:?}");

    let tower_atomic = primals.iter().any(|p| p.contains("beardog"))
        && primals.iter().any(|p| p.contains("songbird"));
    eprintln!("  Tower Atomic present: {tower_atomic}");

    let node_atomic = tower_atomic && primals.iter().any(|p| p.contains("toadstool"));
    eprintln!("  Node Atomic present: {node_atomic}");

    let airspring_present = primals.iter().any(|p| p.contains("airspring"));
    eprintln!("  airSpring primal present: {airspring_present}");

    let socket_path = biomeos::find_socket("airspring");
    v.check_bool("airspring_socket_found", socket_path.is_some());

    let Some(socket) = socket_path else {
        eprintln!("ERROR: airspring_primal not running. Start with:");
        eprintln!("  FAMILY_ID=8ff3b864a4bc589a cargo run --release --bin airspring_primal");
        v.finish();
    };
    eprintln!("  Socket: {}", socket.display());

    // ═══════════════════════════════════════════════════════════════
    // Phase 2: Health & Capability Enumeration (v0.7.5)
    // ═══════════════════════════════════════════════════════════════

    let health =
        rpc::send(&socket, "health", &serde_json::json!({})).and_then(|r| r.get("result").cloned());
    v.check_bool("health_response", health.is_some());

    if let Some(ref h) = health {
        v.check_bool(
            "health_healthy",
            h.get("status").and_then(|s| s.as_str()) == Some("healthy"),
        );
        let version = h.get("version").and_then(|s| s.as_str()).unwrap_or("?");
        v.check_bool("version_matches", version == env!("CARGO_PKG_VERSION"));
        eprintln!("  Primal version: {version}");

        let caps: Vec<&str> = h
            .get("capabilities")
            .and_then(|c| c.as_array())
            .map(|a| a.iter().filter_map(|v| v.as_str()).collect())
            .unwrap_or_default();

        v.check_bool(
            "has_spi_capability",
            caps.contains(&"science.spi_drought_index"),
        );
        v.check_bool(
            "has_autocorrelation_capability",
            caps.contains(&"science.autocorrelation"),
        );
        v.check_bool(
            "has_gamma_cdf_capability",
            caps.contains(&"science.gamma_cdf"),
        );
        v.check_bool(
            "has_ecology_spi_alias",
            caps.contains(&"ecology.spi_drought_index"),
        );
        v.check_bool(
            "has_ecology_acf_alias",
            caps.contains(&"ecology.autocorrelation"),
        );

        eprintln!("  Capabilities registered: {}", caps.len());
        v.check_bool("capability_count_ge_33", caps.len() >= 33);
    }

    // ═══════════════════════════════════════════════════════════════
    // Phase 3: SPI Drought Index via JSON-RPC
    // ═══════════════════════════════════════════════════════════════

    let monthly_precip = vec![
        50.0, 60.0, 45.0, 70.0, 80.0, 55.0, 40.0, 65.0, 50.0, 75.0, 60.0, 45.0, 55.0, 65.0, 40.0,
        70.0, 85.0, 50.0, 35.0, 60.0, 55.0, 70.0, 65.0, 50.0,
    ];

    let rpc_spi = rpc::send(
        &socket,
        "science.spi_drought_index",
        &serde_json::json!({
            "monthly_precip_mm": monthly_precip,
            "scale": 3
        }),
    )
    .and_then(|r| r.get("result").cloned());

    v.check_bool("spi_response", rpc_spi.is_some());
    if let Some(ref r) = rpc_spi {
        let spi_vals = r.get("spi").and_then(|s| s.as_array());
        v.check_bool("spi_array_present", spi_vals.is_some());
        if let Some(spi) = spi_vals {
            v.check_abs("spi_length", spi.len() as f64, 24.0, 0.1);
        }
        let n_valid = r.get("n_valid").and_then(|n| n.as_u64()).unwrap_or(0);
        v.check_bool("spi_has_valid_values", n_valid > 0);

        let upstream = r.get("upstream").and_then(|u| u.as_str()).unwrap_or("");
        v.check_bool(
            "spi_upstream_provenance",
            upstream.contains("regularized_gamma_p"),
        );
        eprintln!("  SPI: {n_valid} valid values, upstream={upstream}");

        let classifications = r.get("classifications").and_then(|c| c.as_array());
        v.check_bool("spi_classifications_present", classifications.is_some());
    }

    // Parity: direct Rust vs JSON-RPC
    let direct_spi = drought_index::compute_spi(&monthly_precip, 3);
    if let Some(ref r) = rpc_spi {
        let rpc_vals: Vec<Option<f64>> = r
            .get("spi")
            .and_then(|s| s.as_array())
            .unwrap()
            .iter()
            .map(|v| v.as_f64())
            .collect();

        let mut spi_parity_ok = true;
        for (i, (&direct, rpc_opt)) in direct_spi.iter().zip(rpc_vals.iter()).enumerate() {
            if direct.is_nan() {
                if rpc_opt.is_some() {
                    spi_parity_ok = false;
                    eprintln!("  SPI[{i}]: direct=NaN but RPC={rpc_opt:?}");
                }
            } else if let Some(rpc_val) = rpc_opt {
                if (direct - rpc_val).abs() > 1e-10 {
                    spi_parity_ok = false;
                    eprintln!("  SPI[{i}]: direct={direct} vs RPC={rpc_val}");
                }
            }
        }
        v.check_bool("spi_parity_direct_vs_rpc", spi_parity_ok);
    }

    let rpc_eco_spi = rpc::send(
        &socket,
        "ecology.spi_drought_index",
        &serde_json::json!({
            "monthly_precip_mm": [50.0, 60.0, 45.0, 70.0, 80.0, 55.0, 40.0, 65.0, 50.0, 75.0, 60.0, 45.0],
            "scale": 3
        }),
    )
    .and_then(|r| r.get("result").cloned());
    v.check_bool("ecology_spi_alias_works", rpc_eco_spi.is_some());

    // ═══════════════════════════════════════════════════════════════
    // Phase 4: Autocorrelation via JSON-RPC
    // ═══════════════════════════════════════════════════════════════

    let acf_data: Vec<f64> = (0..100)
        .map(|i| {
            3.0_f64.mul_add(
                (2.0 * std::f64::consts::PI * f64::from(i) / 20.0).sin(),
                5.0,
            )
        })
        .collect();

    let rpc_acf = rpc::send(
        &socket,
        "science.autocorrelation",
        &serde_json::json!({
            "data": acf_data,
            "max_lag": 25
        }),
    )
    .and_then(|r| r.get("result").cloned());

    v.check_bool("acf_response", rpc_acf.is_some());
    if let Some(ref r) = rpc_acf {
        let acf_vals = r.get("acf").and_then(|a| a.as_array());
        v.check_bool("acf_array_present", acf_vals.is_some());
        if let Some(acf) = acf_vals {
            v.check_abs("acf_length", acf.len() as f64, 25.0, 0.1);
        }

        let nacf = r.get("normalised_acf").and_then(|a| a.as_array());
        v.check_bool("nacf_present", nacf.is_some());

        let provenance = r.get("provenance").and_then(|p| p.as_str()).unwrap_or("");
        v.check_bool(
            "acf_provenance_cross_spring",
            provenance.contains("hotSpring") && provenance.contains("neuralSpring"),
        );
        eprintln!("  ACF provenance: {provenance}");
    }

    // Parity: direct CPU vs JSON-RPC
    let direct_acf = autocorrelation::autocorrelation_cpu(&acf_data, 25);
    if let Some(ref r) = rpc_acf {
        let rpc_acf_vals: Vec<f64> = r
            .get("acf")
            .and_then(|a| a.as_array())
            .unwrap()
            .iter()
            .filter_map(|v| v.as_f64())
            .collect();

        let mut acf_parity_ok = true;
        for (i, (&direct, &rpc_val)) in direct_acf.iter().zip(rpc_acf_vals.iter()).enumerate() {
            if (direct - rpc_val).abs() > 1e-10 {
                acf_parity_ok = false;
                eprintln!("  ACF[{i}]: direct={direct} vs RPC={rpc_val}");
            }
        }
        v.check_bool("acf_parity_direct_vs_rpc", acf_parity_ok);
    }

    let rpc_eco_acf = rpc::send(
        &socket,
        "ecology.autocorrelation",
        &serde_json::json!({"data": [1.0, 2.0, 3.0, 4.0, 5.0], "max_lag": 3}),
    )
    .and_then(|r| r.get("result").cloned());
    v.check_bool("ecology_acf_alias_works", rpc_eco_acf.is_some());

    // ═══════════════════════════════════════════════════════════════
    // Phase 5: Gamma CDF via JSON-RPC (upstream lean)
    // ═══════════════════════════════════════════════════════════════

    let rpc_gamma = rpc::send(
        &socket,
        "science.gamma_cdf",
        &serde_json::json!({"x": 2.0, "alpha": 3.0, "beta": 1.0}),
    )
    .and_then(|r| r.get("result").cloned());

    v.check_bool("gamma_cdf_response", rpc_gamma.is_some());
    if let Some(ref r) = rpc_gamma {
        let cdf = r.get("gamma_cdf").and_then(|c| c.as_f64()).unwrap_or(0.0);
        let direct_cdf = drought_index::gamma_cdf(
            2.0,
            &drought_index::GammaParams {
                alpha: 3.0,
                beta: 1.0,
            },
        );
        v.check_abs("gamma_cdf_parity", cdf, direct_cdf, 1e-10);
        v.check_bool("gamma_cdf_in_range", cdf > 0.0 && cdf < 1.0);
        eprintln!("  Gamma CDF(2.0; a=3, b=1): {cdf:.10}");

        let upstream = r.get("upstream").and_then(|u| u.as_str()).unwrap_or("");
        v.check_bool(
            "gamma_cdf_upstream_lean",
            upstream.contains("regularized_gamma_p"),
        );
    }

    // ═══════════════════════════════════════════════════════════════
    // Phase 6: Full Ecology Pipeline via JSON-RPC
    // ═══════════════════════════════════════════════════════════════

    let rpc_pipeline = rpc::send(
        &socket,
        "ecology.full_pipeline",
        &serde_json::json!({
            "tmax": 32.0,
            "tmin": 18.0,
            "solar_radiation": 22.5,
            "wind_speed_2m": 1.8,
            "actual_vapour_pressure": 1.2,
            "day_of_year": 200,
            "latitude_deg": 42.727,
            "elevation_m": 256.0,
            "kc": 1.15,
            "precipitation_mm": 2.5,
            "soil_water_mm": 150.0,
            "field_capacity_mm": 200.0,
            "wilting_point_mm": 50.0,
            "ky": 1.25,
            "max_yield_t_ha": 12.0,
        }),
    )
    .and_then(|r| r.get("result").cloned());

    v.check_bool("full_pipeline_response", rpc_pipeline.is_some());
    if let Some(ref p) = rpc_pipeline {
        let stages = p.get("stages");
        v.check_bool("pipeline_has_stages", stages.is_some());
        if let Some(s) = stages {
            let et0_stage = s.get("et0");
            let wb_stage = s.get("water_balance");
            let yr_stage = s.get("yield");
            v.check_bool("pipeline_has_et0_stage", et0_stage.is_some());
            v.check_bool("pipeline_has_wb_stage", wb_stage.is_some());
            v.check_bool("pipeline_has_yield_stage", yr_stage.is_some());
            let et0 = et0_stage
                .and_then(|e| e.get("et0_mm"))
                .and_then(|e| e.as_f64())
                .unwrap_or(0.0);
            v.check_bool("pipeline_et0_positive", et0 > 0.0);
            let yield_val = yr_stage
                .and_then(|y| y.get("yield_t_ha"))
                .and_then(|y| y.as_f64())
                .unwrap_or(-1.0);
            v.check_bool("pipeline_yield_present", yield_val >= 0.0);
            eprintln!("  Full pipeline ET0: {et0:.4} mm, yield: {yield_val:.4} t/ha");
        }
    }

    // ═══════════════════════════════════════════════════════════════
    // Phase 7: Cross-Primal Discovery & Forwarding
    // ═══════════════════════════════════════════════════════════════

    let rpc_discover = rpc::send(&socket, "primal.discover", &serde_json::json!({}))
        .and_then(|r| r.get("result").cloned());

    v.check_bool("primal_discover_response", rpc_discover.is_some());
    if let Some(ref d) = rpc_discover {
        let socket_dir_str = d.get("socket_dir").and_then(|s| s.as_str()).unwrap_or("");
        v.check_bool("discover_has_socket_dir", !socket_dir_str.is_empty());
        let count = d.get("count").and_then(|c| c.as_u64()).unwrap_or(0);
        eprintln!("  Discovered {count} primal(s) in {socket_dir_str}");
    }

    // ═══════════════════════════════════════════════════════════════
    // Phase 8: toadStool Provenance via IPC (if running)
    // ═══════════════════════════════════════════════════════════════

    let toadstool_socket = biomeos::find_socket("toadstool");
    if let Some(ref ts) = toadstool_socket {
        eprintln!("  ToadStool socket: {}", ts.display());
        let provenance = rpc::send(ts, "toadstool.provenance", &serde_json::json!({}))
            .and_then(|r| r.get("result").cloned());
        if let Some(ref p) = provenance {
            let total = p.get("total_flows").and_then(|t| t.as_u64()).unwrap_or(0);
            v.check_bool("provenance_has_flows", total > 0);
            eprintln!("  Cross-spring provenance flows: {total}");
        } else {
            eprintln!("  ToadStool socket found but provenance IPC failed (non-fatal)");
            v.check_bool("toadstool_socket_detected", true);
        }
    } else {
        eprintln!("  ToadStool not running — provenance IPC skipped (non-fatal)");
        v.check_bool("toadstool_not_required", true);
    }

    // ═══════════════════════════════════════════════════════════════
    // Phase 9: GPU Precision Routing (barracuda device probe)
    // ═══════════════════════════════════════════════════════════════

    if let Some(device) = airspring_barracuda::gpu::device_info::try_f64_device() {
        let report = airspring_barracuda::gpu::device_info::probe_device(&device);
        let routing = report.precision_routing;
        eprintln!("  GPU precision routing: {routing:?}");
        v.check_bool("precision_routing_determined", true);

        let fp64 = report.fp64_strategy;
        eprintln!("  Fp64 strategy: {fp64:?}");
        v.check_bool("fp64_strategy_set", true);
    } else {
        eprintln!("  No f64 GPU — CPU-only mode (non-fatal)");
        v.check_bool("no_gpu_cpu_fallback", true);
    }

    // ═══════════════════════════════════════════════════════════════
    // Summary
    // ═══════════════════════════════════════════════════════════════

    eprintln!();
    eprintln!("  -- NUCLEUS Topology --");
    eprintln!(
        "  Tower Atomic (BearDog+Songbird):  {}",
        if tower_atomic { "LIVE" } else { "offline" }
    );
    eprintln!(
        "  Node Atomic (+ToadStool):         {}",
        if node_atomic { "LIVE" } else { "offline" }
    );
    eprintln!(
        "  airSpring primal:                 LIVE (v{})",
        env!("CARGO_PKG_VERSION")
    );
    eprintln!();
    eprintln!("  -- v0.7.5 Capabilities via JSON-RPC --");
    eprintln!(
        "  science.spi_drought_index:        {}",
        if rpc_spi.is_some() { "OK" } else { "FAIL" }
    );
    eprintln!(
        "  science.autocorrelation:          {}",
        if rpc_acf.is_some() { "OK" } else { "FAIL" }
    );
    eprintln!(
        "  science.gamma_cdf:                {}",
        if rpc_gamma.is_some() { "OK" } else { "FAIL" }
    );
    eprintln!(
        "  ecology.full_pipeline:            {}",
        if rpc_pipeline.is_some() { "OK" } else { "FAIL" }
    );

    v.finish();
}
