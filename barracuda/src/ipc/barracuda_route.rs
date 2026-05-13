// SPDX-License-Identifier: AGPL-3.0-or-later
//! barraCuda IPC routing — forwards compute calls to a live barraCuda primal.
//!
//! Discovery uses the standard `resolve_transport` pipeline:
//! `BARRACUDA_SOCKET` env → `BARRACUDA_ADDRESS` env → XDG runtime dir → biomeOS fallback.
//! Returns `None` on any failure so callers can fall back to in-process compute.

use crate::primal_names;
use crate::rpc;

/// Attempt to forward a method call to a live barraCuda primal.
///
/// Returns `Some(result)` on success, `None` on any failure (no primal,
/// transport error, RPC error).
#[must_use]
pub fn try_forward(method: &str, params: &serde_json::Value) -> Option<serde_json::Value> {
    let transport = rpc::resolve_transport(primal_names::BARRACUDA).ok()?;
    rpc::send_to(&transport, method, params)
        .ok()
        .and_then(|resp| resp.get("result").cloned())
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test code")]
mod tests {
    use super::*;

    #[test]
    fn try_forward_returns_none_when_primal_absent() {
        let result = try_forward("precision.route", &serde_json::json!({"domain": "test"}));
        assert!(result.is_none());
    }

    #[test]
    fn uses_standard_discovery() {
        assert_eq!(
            primal_names::socket_env_var(primal_names::BARRACUDA),
            "BARRACUDA_SOCKET"
        );
    }
}
