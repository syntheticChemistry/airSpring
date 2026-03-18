// SPDX-License-Identifier: AGPL-3.0-or-later
//! Runtime primal discovery — capability-based, zero hardcoded paths.
//!
//! Each function follows the three-tier pattern: env override → named socket
//! scan → capability probe. The primal only has self-knowledge and discovers
//! peers at runtime via biomeOS socket resolution.

use airspring_barracuda::{biomeos, primal_names};

pub fn discover_orchestrator_socket() -> Option<std::path::PathBuf> {
    if let Ok(name) = std::env::var("BIOMEOS_ORCHESTRATOR_SOCKET") {
        let path = biomeos::resolve_socket_dir().join(name);
        if path.exists() {
            return Some(path);
        }
    }
    biomeos::discover_primal_socket(primal_names::BIOMEOS)
}

pub fn discover_compute_primal() -> Option<std::path::PathBuf> {
    if let Ok(name) = std::env::var("AIRSPRING_COMPUTE_PRIMAL")
        && let Some(path) = biomeos::discover_primal_socket(&name)
    {
        return Some(path);
    }
    biomeos::discover_primal_socket(primal_names::TOADSTOOL)
}

pub fn discover_data_primal() -> Option<std::path::PathBuf> {
    if let Ok(name) = std::env::var("AIRSPRING_DATA_PRIMAL")
        && let Some(path) = biomeos::discover_primal_socket(&name)
    {
        return Some(path);
    }
    biomeos::discover_primal_socket(primal_names::NESTGATE)
}
