// SPDX-License-Identifier: AGPL-3.0-or-later
//! barraCuda IPC routing — forwards compute calls to a live barraCuda primal.
//!
//! Socket discovery: `BARRACUDA_SOCKET` env → `/tmp/barracuda.sock` fallback.
//! Returns `None` on any failure so callers can fall back to in-process compute.

use std::path::PathBuf;

/// Discover the barraCuda primal socket.
pub fn discover() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("BARRACUDA_SOCKET") {
        let path = PathBuf::from(p);
        if path.exists() {
            return Some(path);
        }
    }
    let default = PathBuf::from("/tmp/barracuda.sock");
    if default.exists() {
        Some(default)
    } else {
        None
    }
}

/// Attempt to forward a method call to a live barraCuda primal.
///
/// Returns `Some(result)` on success, `None` on any failure.
pub fn try_forward(method: &str, params: &serde_json::Value) -> Option<serde_json::Value> {
    let socket = discover()?;
    crate::rpc::call_unix(&socket, method, params).ok()
}
