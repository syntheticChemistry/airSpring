// SPDX-License-Identifier: AGPL-3.0-or-later

//! Dynamic method registration with biomeOS via `method.register`.
//!
//! Sends the spring's registered capabilities to biomeOS for dynamic
//! semantic routing. Follows the biomeOS v3.51 batch contract:
//!
//! ```json
//! { "primal": "airspring", "transport": "/path/to.sock", "methods": [...] }
//! ```
//!
//! Non-fatal when biomeOS is unavailable — the spring continues standalone.

use std::path::Path;

use tracing::{info, warn};

use crate::{niche, rpc};

/// Register all niche methods with biomeOS `method.register`.
///
/// Returns `Some(count)` of registered methods on success, `None` if
/// biomeOS is unavailable or the call fails.
pub fn register_methods(biomeos_socket: &Path, our_socket: &Path) -> Option<u32> {
    let methods: Vec<&str> = niche::CAPABILITIES.to_vec();

    let params = serde_json::json!({
        "primal": niche::NICHE_NAME,
        "transport": our_socket.to_string_lossy(),
        "methods": methods,
    });

    match rpc::send(biomeos_socket, "method.register", &params) {
        Ok(resp) => {
            let registered = resp
                .get("result")
                .and_then(|r| r.get("registered"))
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0);
            let count = u32::try_from(registered).unwrap_or(0);
            info!(
                target: crate::primal_names::BIOMEOS,
                registered = count,
                total = methods.len(),
                "method.register accepted"
            );
            Some(count)
        }
        Err(e) => {
            warn!(
                target: crate::primal_names::BIOMEOS,
                error = %e,
                "method.register failed (non-fatal)"
            );
            None
        }
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test assertions use unwrap for clarity")]
mod tests {
    use super::*;

    #[test]
    fn register_params_shape() {
        let methods: Vec<&str> = niche::CAPABILITIES.to_vec();
        let params = serde_json::json!({
            "primal": niche::NICHE_NAME,
            "transport": "/tmp/test.sock",
            "methods": methods,
        });

        assert_eq!(params["primal"], "airspring");
        assert!(params["methods"].is_array());
        let arr = params["methods"].as_array().unwrap();
        assert_eq!(arr.len(), niche::CAPABILITIES.len());
        assert!(arr.iter().any(|v| v == "science.et0_fao56"));
    }
}
