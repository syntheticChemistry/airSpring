// SPDX-License-Identifier: AGPL-3.0-or-later

//! biomeOS Neural API bridge — `capability.call` over Unix sockets.
//!
//! Minimal synchronous JSON-RPC 2.0 client that talks to the biomeOS
//! Neural API. Zero external dependencies beyond `std` + `serde_json`.
//!
//! # Discovery
//!
//! The Neural API socket is discovered using biomeOS's 5-tier resolution:
//!
//! 1. `NEURAL_API_SOCKET` env var
//! 2. `$XDG_RUNTIME_DIR/biomeos/neural-api-{family_id}.sock`
//! 3. `/run/user/{uid}/biomeos/neural-api-{family_id}.sock`
//! 4. `/tmp/biomeos/neural-api-{family_id}.sock`
//!
//! # Usage
//!
//! ```no_run
//! use airspring_forge::neural::NeuralBridge;
//!
//! let bridge = NeuralBridge::discover().unwrap();
//! let result = bridge.capability_call("ecology", "et0_pm", &serde_json::json!({
//!     "tmin": 12.3, "tmax": 21.5
//! }));
//! ```

use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::UnixStream;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use crate::substrate::{Capability, Identity, Properties, Substrate, SubstrateKind};

static REQUEST_ID: AtomicU64 = AtomicU64::new(1);

/// Connection to the biomeOS Neural API.
pub struct NeuralBridge {
    socket_path: PathBuf,
    timeout: Duration,
}

/// Result of a `capability.call` invocation.
#[derive(Debug)]
pub struct CallResult {
    pub value: serde_json::Value,
}

/// Error from Neural API communication.
#[derive(Debug)]
pub enum NeuralError {
    /// Neural API socket not found (biomeOS not running).
    NotFound(String),
    /// Connection failed.
    Connection(std::io::Error),
    /// JSON serialization/deserialization error.
    Json(String),
    /// JSON-RPC error response from the Neural API.
    Rpc { code: i64, message: String },
    /// Timeout waiting for response.
    Timeout,
}

impl std::fmt::Display for NeuralError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotFound(msg) => write!(f, "Neural API not found: {msg}"),
            Self::Connection(e) => write!(f, "Connection error: {e}"),
            Self::Json(msg) => write!(f, "JSON error: {msg}"),
            Self::Rpc { code, message } => write!(f, "RPC error {code}: {message}"),
            Self::Timeout => write!(f, "Timeout"),
        }
    }
}

impl std::error::Error for NeuralError {}

impl NeuralBridge {
    /// Discover the Neural API socket using biomeOS 5-tier resolution.
    ///
    /// Returns `None` if biomeOS is not running (no socket found).
    #[must_use]
    pub fn discover() -> Option<Self> {
        let path = resolve_socket()?;
        Some(Self {
            socket_path: path,
            timeout: Duration::from_secs(30),
        })
    }

    /// Set the timeout for requests.
    #[must_use]
    pub const fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Send a `capability.call` request to the Neural API.
    ///
    /// The Neural API routes this to the appropriate primal based on
    /// the capability translation registry.
    ///
    /// # Errors
    ///
    /// Returns `NeuralError` if the socket connection fails, the request
    /// is malformed, or the remote primal returns an RPC error.
    pub fn capability_call(
        &self,
        capability: &str,
        operation: &str,
        args: &serde_json::Value,
    ) -> Result<CallResult, NeuralError> {
        let id = REQUEST_ID.fetch_add(1, Ordering::Relaxed);
        let request = serde_json::json!({
            "jsonrpc": "2.0",
            "method": "capability.call",
            "params": {
                "capability": capability,
                "operation": operation,
                "args": args,
            },
            "id": id,
        });

        let response = self.send_request(&request)?;
        parse_response(&response)
    }

    /// Discover capabilities available in the ecosystem.
    ///
    /// # Errors
    ///
    /// Returns `NeuralError` on connection or protocol failure.
    pub fn discover_capability(&self, capability: &str) -> Result<serde_json::Value, NeuralError> {
        let id = REQUEST_ID.fetch_add(1, Ordering::Relaxed);
        let request = serde_json::json!({
            "jsonrpc": "2.0",
            "method": "capability.discover",
            "params": { "capability": capability },
            "id": id,
        });
        let response = self.send_request(&request)?;
        parse_response(&response).map(|r| r.value)
    }

    /// Check if the Neural API is reachable.
    ///
    /// # Errors
    ///
    /// Returns `NeuralError` on connection or protocol failure.
    pub fn health_check(&self) -> Result<serde_json::Value, NeuralError> {
        let id = REQUEST_ID.fetch_add(1, Ordering::Relaxed);
        let request = serde_json::json!({
            "jsonrpc": "2.0",
            "method": "lifecycle.status",
            "params": {},
            "id": id,
        });
        let response = self.send_request(&request)?;
        parse_response(&response).map(|r| r.value)
    }

    /// The socket path we're connected to.
    #[must_use]
    pub fn socket_path(&self) -> &std::path::Path {
        &self.socket_path
    }

    fn send_request(&self, request: &serde_json::Value) -> Result<serde_json::Value, NeuralError> {
        let mut stream = UnixStream::connect(&self.socket_path).map_err(NeuralError::Connection)?;
        stream
            .set_read_timeout(Some(self.timeout))
            .map_err(NeuralError::Connection)?;
        stream
            .set_write_timeout(Some(self.timeout))
            .map_err(NeuralError::Connection)?;

        let mut payload =
            serde_json::to_string(request).map_err(|e| NeuralError::Json(e.to_string()))?;
        payload.push('\n');
        stream
            .write_all(payload.as_bytes())
            .map_err(NeuralError::Connection)?;
        stream.flush().map_err(NeuralError::Connection)?;

        let mut reader = BufReader::new(stream);
        let mut line = String::new();
        reader
            .read_line(&mut line)
            .map_err(NeuralError::Connection)?;

        serde_json::from_str(&line).map_err(|e| NeuralError::Json(e.to_string()))
    }
}

/// Probe for a Neural API substrate.
///
/// Returns a `Substrate` if biomeOS is running, with capabilities
/// determined by `capability.discover` or assumed to be full-spectrum.
#[must_use]
pub fn probe_neural() -> Option<Substrate> {
    let bridge = NeuralBridge::discover()?;
    let health_ok = bridge.health_check().is_ok();
    if !health_ok {
        return None;
    }

    Some(Substrate {
        kind: SubstrateKind::Neural,
        identity: Identity {
            name: format!("biomeOS Neural API ({})", bridge.socket_path().display()),
            driver: Some(String::from("biomeos-neural-api")),
            backend: Some(String::from("unix-socket")),
            adapter_index: None,
            device_node: None,
            pci_id: None,
        },
        properties: Properties::default(),
        capabilities: vec![
            Capability::F64Compute,
            Capability::F32Compute,
            Capability::ScalarReduce,
            Capability::NeuralApiRoute,
        ],
    })
}

fn resolve_socket() -> Option<PathBuf> {
    // Tier 1: explicit env var
    if let Ok(path) = std::env::var("NEURAL_API_SOCKET") {
        let p = PathBuf::from(&path);
        if p.exists() {
            return Some(p);
        }
    }

    let family_id = std::env::var("FAMILY_ID").ok()?;

    // Tier 2: XDG_RUNTIME_DIR
    if let Ok(xdg) = std::env::var("XDG_RUNTIME_DIR") {
        let p = PathBuf::from(xdg)
            .join("biomeos")
            .join(format!("neural-api-{family_id}.sock"));
        if p.exists() {
            return Some(p);
        }
    }

    // Tier 3: /run/user/{uid} — derive from XDG_RUNTIME_DIR or procfs
    let uid = uid_from_runtime_dir();
    let p = PathBuf::from(format!(
        "/run/user/{uid}/biomeos/neural-api-{family_id}.sock"
    ));
    if p.exists() {
        return Some(p);
    }

    // Tier 4: /tmp fallback
    let p = PathBuf::from(format!("/tmp/biomeos/neural-api-{family_id}.sock"));
    if p.exists() {
        return Some(p);
    }

    None
}

const PROC_STATUS_PATH: &str = "/proc/self/status";

/// Extract real UID from `/proc/self/status` (safe, no libc).
///
/// Falls back to `nobody` (65534) rather than assuming a specific user.
/// A hardcoded UID like 1000 is fragile — different distros assign different
/// first-user UIDs.  65534 is the POSIX `nobody` sentinel and will fail
/// visibly rather than silently resolve to the wrong user's runtime dir.
fn uid_from_runtime_dir() -> u32 {
    const NOBODY_UID: u32 = 65534;
    std::fs::read_to_string(PROC_STATUS_PATH)
        .ok()
        .and_then(|status| {
            status.lines().find_map(|line| {
                line.strip_prefix("Uid:")
                    .and_then(|rest| rest.split_whitespace().next())
                    .and_then(|s| s.parse::<u32>().ok())
            })
        })
        .unwrap_or(NOBODY_UID)
}

fn parse_response(response: &serde_json::Value) -> Result<CallResult, NeuralError> {
    if let Some(error) = response.get("error") {
        let code = error["code"].as_i64().unwrap_or(-1);
        let message = error["message"]
            .as_str()
            .unwrap_or("unknown error")
            .to_string();
        return Err(NeuralError::Rpc { code, message });
    }
    Ok(CallResult {
        value: response
            .get("result")
            .cloned()
            .unwrap_or(serde_json::Value::Null),
    })
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn no_socket_returns_none() {
        std::env::remove_var("NEURAL_API_SOCKET");
        std::env::remove_var("FAMILY_ID");
        assert!(NeuralBridge::discover().is_none());
    }

    #[test]
    fn substrate_kind_display() {
        assert_eq!(format!("{}", SubstrateKind::Neural), "Neural");
    }

    #[test]
    fn parse_success_response() {
        let resp = serde_json::json!({
            "jsonrpc": "2.0",
            "result": { "et0": 3.88 },
            "id": 1
        });
        let result = parse_response(&resp).unwrap();
        let et0 = result.value["et0"].as_f64().unwrap();
        assert!((et0 - 3.88).abs() < f64::EPSILON);
    }

    #[test]
    fn parse_error_response() {
        let resp = serde_json::json!({
            "jsonrpc": "2.0",
            "error": { "code": -32601, "message": "Method not found" },
            "id": 1
        });
        let err = parse_response(&resp).unwrap_err();
        assert!(matches!(err, NeuralError::Rpc { code: -32601, .. }));
    }

    #[test]
    fn probe_returns_none_without_biomeos() {
        std::env::remove_var("NEURAL_API_SOCKET");
        std::env::remove_var("FAMILY_ID");
        assert!(probe_neural().is_none());
    }
}
