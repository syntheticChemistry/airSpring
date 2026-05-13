// SPDX-License-Identifier: AGPL-3.0-or-later

//! skunkBat audit event emission via `security.audit_log`.
//!
//! Sends structured audit events to the skunkBat primal for cross-primal
//! audit forwarding to rhizoCrypt DAG + sweetGrass braid (JH-5).
//!
//! Non-fatal when skunkBat is unavailable — the spring continues without
//! audit logging. When Phase 3 ships, forwarding is automatic.

use std::path::Path;
use std::time::SystemTime;

use tracing::warn;

use crate::{biomeos, niche, primal_names, rpc};

fn epoch_secs() -> u64 {
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

/// Discover the skunkBat socket via biomeOS primal discovery.
#[must_use]
pub fn discover() -> Option<std::path::PathBuf> {
    biomeos::discover_primal_socket(primal_names::SKUNKBAT)
}

/// Emit an audit event to skunkBat via `security.audit_log`.
///
/// Returns the response on success, `None` if skunkBat is unavailable.
#[must_use]
pub fn audit_log(event_type: &str, payload: &serde_json::Value) -> Option<serde_json::Value> {
    let socket = discover()?;
    audit_log_to(&socket, event_type, payload)
}

/// Emit an audit event to a specific skunkBat socket.
pub fn audit_log_to(
    socket: &Path,
    event_type: &str,
    payload: &serde_json::Value,
) -> Option<serde_json::Value> {
    let params = serde_json::json!({
        "event_type": event_type,
        "source": niche::NICHE_NAME,
        "payload": payload,
        "timestamp": epoch_secs().to_string(),
    });

    match rpc::send(socket, "security.audit_log", &params) {
        Ok(resp) => Some(resp),
        Err(e) => {
            warn!(
                target: crate::primal_names::SKUNKBAT,
                error = %e,
                event_type,
                "audit_log failed (non-fatal)"
            );
            None
        }
    }
}

/// Emit a certification audit event (layer completion).
#[must_use]
pub fn audit_certification(tier: u8, passed: u32, failed: u32) -> Option<serde_json::Value> {
    audit_log(
        "certification",
        &serde_json::json!({
            "tier": tier,
            "passed": passed,
            "failed": failed,
        }),
    )
}

/// Emit a startup audit event.
#[must_use]
pub fn audit_startup(capabilities: usize) -> Option<serde_json::Value> {
    audit_log(
        "startup",
        &serde_json::json!({
            "version": env!("CARGO_PKG_VERSION"),
            "capabilities": capabilities,
        }),
    )
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test assertions use unwrap for clarity")]
mod tests {
    use super::*;

    #[test]
    fn audit_params_shape() {
        let params = serde_json::json!({
            "event_type": "test",
            "source": niche::NICHE_NAME,
            "payload": {"key": "value"},
            "timestamp": epoch_secs().to_string(),
        });

        assert_eq!(params["source"], "airspring");
        assert_eq!(params["event_type"], "test");
        assert!(params["timestamp"].as_str().unwrap().parse::<u64>().is_ok());
    }

    #[test]
    fn discover_returns_none_when_no_skunkbat() {
        assert!(discover().is_none());
    }
}
