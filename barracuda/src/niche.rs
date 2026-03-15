// SPDX-License-Identifier: AGPL-3.0-or-later
//! Niche deployment self-knowledge for airSpring.
//!
//! A Spring is a niche validation domain — not a primal. It proves that
//! scientific Python baselines can be faithfully ported to sovereign
//! Rust + GPU compute using the ecoPrimals stack. The niche deploys as
//! a biomeOS graph (`graphs/airspring_niche_deploy.toml`) that composes
//! real primals (`BearDog`, `Songbird`, `ToadStool`, etc.).
//!
//! This module holds the niche's self-knowledge:
//! - Capability table (what the niche exposes via biomeOS)
//! - Semantic mappings (capability domain → science methods)
//! - Operation dependencies (parallelization hints for Pathway Learner)
//! - Cost estimates (scheduling hints for biomeOS)
//! - Registration logic (how the niche advertises itself)
//!
//! # Evolution
//!
//! The transitional `airspring_primal` binary exposes these capabilities
//! via a JSON-RPC server. The final form is graph-only deployment where
//! biomeOS orchestrates the niche directly from deploy graphs.

use std::path::Path;

/// Niche identity.
pub const NICHE_NAME: &str = "airspring";

/// All capabilities this niche exposes to biomeOS.
pub const CAPABILITIES: &[&str] = &[
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
    // ── Drought & Stochastic ──
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

/// Operation dependency hints for biomeOS Pathway Learner parallelization.
#[must_use]
pub fn operation_dependencies() -> serde_json::Value {
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

/// Cost estimates for biomeOS scheduling (measured on representative hardware).
#[must_use]
pub fn cost_estimates() -> serde_json::Value {
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

/// Semantic mappings for ecology capability domain routing.
#[must_use]
pub fn ecology_semantic_mappings() -> serde_json::Value {
    serde_json::json!({
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
    })
}

/// Register all niche capability domains with a biomeOS target socket.
///
/// Sends `lifecycle.register` followed by per-domain `capability.register`
/// calls and individual capability advertisements.
pub fn register_with_target(target: &Path, our_socket: &Path) {
    let reg_result = crate::rpc::send(
        target,
        "lifecycle.register",
        &serde_json::json!({
            "name": NICHE_NAME,
            "socket_path": our_socket.to_string_lossy(),
            "pid": std::process::id(),
        }),
    );

    match reg_result {
        Some(_) => eprintln!("[biomeos] Registered with lifecycle manager"),
        None => eprintln!("[biomeos] lifecycle.register failed (non-fatal)"),
    }

    let sock_str = our_socket.to_string_lossy().to_string();

    let domains: &[(&str, serde_json::Value)] = &[
        ("ecology", ecology_semantic_mappings()),
        (
            "provenance",
            serde_json::json!({
                "begin":    "provenance.begin",
                "record":   "provenance.record",
                "complete": "provenance.complete",
                "status":   "provenance.status",
            }),
        ),
        (
            "data",
            serde_json::json!({
                "cross_spring_weather": "data.cross_spring_weather",
            }),
        ),
        (
            "capability",
            serde_json::json!({ "list": "capability.list" }),
        ),
    ];

    for (domain, mappings) in domains {
        let mut payload = serde_json::json!({
            "primal": NICHE_NAME,
            "capability": domain,
            "socket": &sock_str,
            "semantic_mappings": mappings,
        });
        if *domain == "capability" {
            payload["operation_dependencies"] = operation_dependencies();
            payload["cost_estimates"] = cost_estimates();
        }
        let _ = crate::rpc::send(target, "capability.register", &payload);
    }

    let mut registered = 0u32;
    for cap in CAPABILITIES {
        if crate::rpc::send(
            target,
            "capability.register",
            &serde_json::json!({
                "primal": NICHE_NAME,
                "capability": cap,
                "socket": &sock_str,
            }),
        )
        .is_some()
        {
            registered += 1;
        } else {
            eprintln!("[biomeos] capability.register({cap}) failed (non-fatal)");
        }
    }

    eprintln!(
        "[biomeos] {registered}/{} capabilities + {} domains registered",
        CAPABILITIES.len(),
        domains.len()
    );
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test code uses unwrap for clarity")]
mod tests {
    use super::*;

    #[test]
    fn capabilities_are_not_empty() {
        assert!(!CAPABILITIES.is_empty());
    }

    #[test]
    fn capabilities_follow_semantic_naming() {
        for cap in CAPABILITIES {
            assert!(
                cap.contains('.'),
                "capability '{cap}' should follow domain.operation format"
            );
        }
    }

    #[test]
    fn operation_dependencies_is_object() {
        let deps = operation_dependencies();
        assert!(deps.is_object());
    }

    #[test]
    fn cost_estimates_is_object() {
        let costs = cost_estimates();
        assert!(costs.is_object());
    }

    #[test]
    fn ecology_mappings_cover_all_science_capabilities() {
        let mappings = ecology_semantic_mappings();
        let map = mappings.as_object().unwrap();
        let science_caps: Vec<&&str> = CAPABILITIES
            .iter()
            .filter(|c| {
                c.starts_with("science.")
                    && !c.starts_with("science.spi")
                    && !c.starts_with("science.autocorrelation")
                    && !c.starts_with("science.gamma")
            })
            .collect();
        for cap in &science_caps {
            let short = cap.strip_prefix("science.").unwrap();
            assert!(
                map.values().any(|v| v.as_str() == Some(cap)),
                "science capability '{cap}' (key '{short}') should appear in ecology mappings"
            );
        }
    }

    #[test]
    fn niche_name_matches_convention() {
        assert_eq!(NICHE_NAME, "airspring");
        assert!(NICHE_NAME.chars().all(|c| c.is_ascii_lowercase()));
    }
}
