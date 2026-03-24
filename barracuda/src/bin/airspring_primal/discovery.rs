// SPDX-License-Identifier: AGPL-3.0-or-later
//! Runtime primal discovery — capability-based, zero hardcoded paths.
//!
//! Each function follows the three-tier pattern:
//!   1. Environment override (explicit primal name or socket path)
//!   2. Named socket scan (well-known ecosystem primal)
//!   3. Capability probe (discover any primal exposing the required domain)
//!
//! The primal only has self-knowledge and discovers peers at runtime via
//! biomeOS socket resolution. Named defaults (tier 2) are discovery hints,
//! not compile-time coupling — any primal exposing the right capabilities
//! will be found by tier 3.

use airspring_barracuda::{biomeos, primal_names};

/// Env var for overriding the orchestrator socket basename.
pub const ORCHESTRATOR_SOCKET_ENV: &str = "BIOMEOS_ORCHESTRATOR_SOCKET";
/// Env var for overriding which primal handles compute dispatch.
pub const COMPUTE_PRIMAL_ENV: &str = "AIRSPRING_COMPUTE_PRIMAL";
/// Env var for overriding which primal handles data storage/retrieval.
pub const DATA_PRIMAL_ENV: &str = "AIRSPRING_DATA_PRIMAL";

pub fn discover_orchestrator_socket() -> Option<std::path::PathBuf> {
    if let Ok(name) = std::env::var(ORCHESTRATOR_SOCKET_ENV) {
        let path = biomeos::resolve_socket_dir().join(name);
        if path.exists() {
            return Some(path);
        }
    }
    biomeos::discover_primal_socket(primal_names::BIOMEOS)
}

pub fn discover_compute_primal() -> Option<std::path::PathBuf> {
    if let Ok(name) = std::env::var(COMPUTE_PRIMAL_ENV)
        && let Some(path) = biomeos::discover_primal_socket(&name)
    {
        return Some(path);
    }
    if let Some(path) = biomeos::discover_primal_socket(primal_names::TOADSTOOL) {
        return Some(path);
    }
    biomeos::discover_primal_by_capability(primal_names::domains::COMPUTE)
}

pub fn discover_data_primal() -> Option<std::path::PathBuf> {
    if let Ok(name) = std::env::var(DATA_PRIMAL_ENV)
        && let Some(path) = biomeos::discover_primal_socket(&name)
    {
        return Some(path);
    }
    if let Some(path) = biomeos::discover_primal_socket(primal_names::NESTGATE) {
        return Some(path);
    }
    biomeos::discover_primal_by_capability("storage")
}
