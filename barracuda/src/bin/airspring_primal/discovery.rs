// SPDX-License-Identifier: AGPL-3.0-or-later
//! Runtime primal discovery — capability-based, zero hardcoded paths.

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
    std::env::var("AIRSPRING_COMPUTE_PRIMAL")
        .ok()
        .and_then(|name| biomeos::discover_primal_socket(&name))
}

pub fn discover_data_primal() -> Option<std::path::PathBuf> {
    std::env::var("AIRSPRING_DATA_PRIMAL")
        .ok()
        .and_then(|name| biomeos::discover_primal_socket(&name))
}
