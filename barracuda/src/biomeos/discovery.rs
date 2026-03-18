// SPDX-License-Identifier: AGPL-3.0-or-later
//! Primal socket discovery for biomeOS.

use std::path::{Path, PathBuf};

use super::capabilities;
use super::{SocketConfig, get_family_id_with, resolve_socket_dir, resolve_socket_dir_with};

/// Discover a primal's socket by scanning a specific directory.
///
/// Tries `{name}-{family}.sock` first, then `{name}.sock`, then any
/// file starting with `{name}` and ending with `.sock`.
#[must_use]
pub fn discover_primal_socket_in(
    primal_name: &str,
    socket_dir: &Path,
    family_id: &str,
) -> Option<PathBuf> {
    let with_family = socket_dir.join(format!("{primal_name}-{family_id}.sock"));
    if with_family.exists() {
        return Some(with_family);
    }

    let without_family = socket_dir.join(format!("{primal_name}.sock"));
    if without_family.exists() {
        return Some(without_family);
    }

    if let Ok(entries) = std::fs::read_dir(socket_dir) {
        for entry in entries.flatten() {
            let name = entry.file_name();
            let name_str = name.to_string_lossy();
            if name_str.starts_with(primal_name) && name_str.ends_with(".sock") {
                return Some(entry.path());
            }
        }
    }

    None
}

/// Find a socket by prefix in a specific directory.
#[must_use]
pub fn find_socket_in(prefix: &str, socket_dir: &Path) -> Option<PathBuf> {
    if let Ok(entries) = std::fs::read_dir(socket_dir) {
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

/// List all discovered primals in a specific directory.
#[must_use]
pub fn discover_all_primals_in(socket_dir: &Path) -> Vec<String> {
    let mut primals = Vec::new();

    if let Ok(entries) = std::fs::read_dir(socket_dir) {
        for entry in entries.flatten() {
            let name = entry.file_name();
            let name_str = name.to_string_lossy().to_string();
            if Path::new(&name_str)
                .extension()
                .is_some_and(|ext| ext.eq_ignore_ascii_case("sock"))
            {
                let primal_name = name_str
                    .split('-')
                    .next()
                    .unwrap_or(&name_str)
                    .trim_end_matches(".sock")
                    .to_string();

                if !primals.contains(&primal_name) {
                    primals.push(primal_name);
                }
            }
        }
    }

    primals.sort();
    primals
}

/// Discover a primal's socket by scanning the socket directory.
///
/// Tries `{name}-{family}.sock` first, then `{name}.sock`, then any
/// file starting with `{name}` and ending with `.sock`.
#[must_use]
pub fn discover_primal_socket(primal_name: &str) -> Option<PathBuf> {
    let config = SocketConfig::from_env();
    let socket_dir = resolve_socket_dir_with(&config);
    let family_id = get_family_id_with(&config);
    discover_primal_socket_in(primal_name, &socket_dir, &family_id)
}

/// Find a socket by prefix (e.g., `"airspring"` finds `airspring-*.sock`).
#[must_use]
pub fn find_socket(prefix: &str) -> Option<PathBuf> {
    let socket_dir = resolve_socket_dir();
    find_socket_in(prefix, &socket_dir)
}

/// List all discovered primals in the socket directory.
#[must_use]
pub fn discover_all_primals() -> Vec<String> {
    discover_all_primals_in(&resolve_socket_dir())
}

/// Discover the coralReef sovereign shader compiler.
///
/// Three-tier resolution:
/// 1. Environment override (`CORALREEF_SOCKET`)
/// 2. Named socket scan for `coralreef` in biomeOS socket dir
/// 3. Capability probe: scan all primals for `shader.*` capabilities
///
/// Returns the socket path if found.
#[must_use]
pub fn discover_shader_compiler() -> Option<PathBuf> {
    if let Ok(path) = std::env::var(crate::primal_names::socket_env_var(
        crate::primal_names::CORALREEF,
    )) {
        let p = PathBuf::from(path);
        if p.exists() {
            return Some(p);
        }
    }

    if let Some(path) = discover_primal_socket(crate::primal_names::CORALREEF) {
        return Some(path);
    }

    discover_primal_by_capability(crate::primal_names::domains::SHADER)
}

/// Discover the Squirrel inference / model routing primal.
///
/// Three-tier resolution:
/// 1. Environment override (`SQUIRREL_SOCKET`)
/// 2. Named socket scan for `squirrel` in biomeOS socket dir
/// 3. Capability probe: scan all primals for `inference.*` capabilities
///
/// Returns the socket path if found.
#[must_use]
pub fn discover_inference_primal() -> Option<PathBuf> {
    if let Ok(path) = std::env::var(crate::primal_names::socket_env_var(
        crate::primal_names::SQUIRREL,
    )) {
        let p = PathBuf::from(path);
        if p.exists() {
            return Some(p);
        }
    }

    if let Some(path) = discover_primal_socket(crate::primal_names::SQUIRREL) {
        return Some(path);
    }

    discover_primal_by_capability(crate::primal_names::domains::INFERENCE)
}

/// Discover the petalTongue visualization / interactive exploration primal.
///
/// Three-tier resolution:
/// 1. Environment override (`PETALTONGUE_SOCKET`)
/// 2. Named socket scan for `petaltongue` in biomeOS socket dir
/// 3. Capability probe: scan all primals for `visualization.*` capabilities
///
/// Returns the socket path if found.
#[must_use]
pub fn discover_visualization_primal() -> Option<PathBuf> {
    if let Ok(path) = std::env::var(crate::primal_names::socket_env_var(
        crate::primal_names::PETALTONGUE,
    )) {
        let p = PathBuf::from(path);
        if p.exists() {
            return Some(p);
        }
    }

    if let Some(path) = discover_primal_socket(crate::primal_names::PETALTONGUE) {
        return Some(path);
    }

    discover_primal_by_capability(crate::primal_names::domains::VISUALIZATION)
}

/// Discover a primal socket by probing all known sockets for a capability domain.
///
/// Scans the socket directory, connects to each primal's `capability.list`,
/// and checks if any returned capability starts with the given domain prefix.
fn discover_primal_by_capability(domain: &str) -> Option<PathBuf> {
    let socket_dir = resolve_socket_dir();
    let Ok(entries) = std::fs::read_dir(&socket_dir) else {
        return None;
    };

    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().is_none_or(|ext| ext != "sock") {
            continue;
        }

        if let Ok(resp) = crate::rpc::send(&path, "capability.list", &serde_json::json!({})) {
            let caps = capabilities::parse_capabilities(&resp);
            if caps.iter().any(|c| c.starts_with(domain)) {
                return Some(path);
            }
        }
    }

    None
}
