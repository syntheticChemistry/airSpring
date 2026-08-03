// SPDX-License-Identifier: AGPL-3.0-or-later
//! Typed client for toadStool `toadstool.validate` workload pre-flight.
//!
//! Resolves toadStool transport via [`crate::rpc::resolve_transport`] and
//! sends a `toadstool.validate` request with a workload path.  Returns a
//! structured [`ValidateResult`] containing GPU availability, precision tier,
//! estimated dispatch time, and any warnings.
//!
//! Non-fatal when toadStool is unavailable — callers gate on transport
//! resolution and fall back to local validation only.

use crate::primal_names;
use crate::rpc::{self, IpcError, Transport};

/// Result of a `toadstool.validate` pre-flight check.
#[derive(Debug, Clone)]
pub struct ValidateResult {
    /// Whether the workload TOML is structurally valid.
    pub valid: bool,
    /// Whether a compatible GPU is available for dispatch.
    pub gpu_available: bool,
    /// Precision tier the workload would execute at (e.g. "f32", "f64", "df64").
    pub precision_tier: String,
    /// Estimated wall-clock dispatch time in milliseconds (0 if unknown).
    pub estimated_dispatch_time_ms: u64,
    /// Warnings from the pre-flight check.
    pub warnings: Vec<String>,
    /// Capabilities required by the workload.
    pub required_capabilities: Vec<String>,
}

/// Errors from toadStool validate operations.
#[derive(Debug)]
pub enum ValidateError {
    /// No toadStool transport discovered.
    NoPrimal,
    /// IPC transport error.
    Ipc(IpcError),
    /// Server returned an RPC error.
    RpcError {
        /// JSON-RPC error code.
        code: i64,
        /// Human-readable message.
        message: String,
    },
    /// Response missing expected fields.
    MalformedResponse(String),
}

impl std::fmt::Display for ValidateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoPrimal => write!(f, "no toadStool primal discovered"),
            Self::Ipc(e) => write!(f, "IPC error: {e}"),
            Self::RpcError { code, message } => write!(f, "RPC error {code}: {message}"),
            Self::MalformedResponse(detail) => {
                write!(f, "toadstool.validate malformed response: {detail}")
            }
        }
    }
}

impl std::error::Error for ValidateError {}

impl From<IpcError> for ValidateError {
    fn from(e: IpcError) -> Self {
        match e {
            IpcError::SocketNotFound { .. } => Self::NoPrimal,
            e => Self::Ipc(e),
        }
    }
}

fn discover() -> Result<Transport, ValidateError> {
    rpc::resolve_transport(primal_names::TOADSTOOL).map_err(ValidateError::from)
}

fn parse_result(resp: &serde_json::Value) -> Result<ValidateResult, ValidateError> {
    if let Some((code, message)) = rpc::extract_rpc_error(resp) {
        return Err(ValidateError::RpcError { code, message });
    }

    let r = resp.get("result").or(Some(resp));

    let valid = r
        .and_then(|v| v.get("valid"))
        .and_then(serde_json::Value::as_bool)
        .ok_or_else(|| ValidateError::MalformedResponse("missing `valid`".into()))?;

    let gpu_available = r
        .and_then(|v| v.get("gpu_available"))
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false);

    let precision_tier = r
        .and_then(|v| v.get("precision_tier"))
        .and_then(serde_json::Value::as_str)
        .unwrap_or("unknown")
        .to_owned();

    let estimated_dispatch_time_ms = r
        .and_then(|v| v.get("estimated_dispatch_time_ms"))
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);

    let warnings = r
        .and_then(|v| v.get("warnings"))
        .and_then(serde_json::Value::as_array)
        .map(|arr| {
            arr.iter()
                .filter_map(serde_json::Value::as_str)
                .map(str::to_owned)
                .collect()
        })
        .unwrap_or_default();

    let required_capabilities = r
        .and_then(|v| v.get("required_capabilities"))
        .and_then(serde_json::Value::as_array)
        .map(|arr| {
            arr.iter()
                .filter_map(serde_json::Value::as_str)
                .map(str::to_owned)
                .collect()
        })
        .unwrap_or_default();

    Ok(ValidateResult {
        valid,
        gpu_available,
        precision_tier,
        estimated_dispatch_time_ms,
        warnings,
        required_capabilities,
    })
}

/// Pre-flight validate a workload through toadStool.
///
/// # Errors
///
/// Returns [`ValidateError::NoPrimal`] if toadStool is not discovered.
/// Returns [`ValidateError::Ipc`] on transport failure.
/// Returns [`ValidateError::RpcError`] if the server returns an RPC error.
/// Returns [`ValidateError::MalformedResponse`] if the response shape is unexpected.
pub fn validate(workload_path: &str, dry_run: bool) -> Result<ValidateResult, ValidateError> {
    let transport = discover()?;
    validate_via(&transport, workload_path, dry_run)
}

/// Pre-flight validate against a specific transport (for testing).
///
/// # Errors
///
/// Same as [`validate`].
pub fn validate_via(
    transport: &Transport,
    workload_path: &str,
    dry_run: bool,
) -> Result<ValidateResult, ValidateError> {
    let resp = rpc::send_to(
        transport,
        "toadstool.validate",
        &serde_json::json!({
            "workload_path": workload_path,
            "dry_run": dry_run,
        }),
    )?;
    parse_result(&resp)
}

#[cfg(test)]
#[expect(clippy::expect_used, clippy::unwrap_used, reason = "test code")]
mod tests {
    use super::*;
    use std::io::{BufRead, BufReader, Write};
    use std::net::TcpListener;

    #[test]
    fn no_primal_returns_error() {
        let result = validate("workloads/airspring/test.toml", true);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            ValidateError::NoPrimal | ValidateError::Ipc(_)
        ));
    }

    #[test]
    fn tcp_validate_round_trip() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        let addr = listener.local_addr().expect("addr");

        let server = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("accept");
            let mut reader = BufReader::new(&stream);
            let mut line = String::new();
            reader.read_line(&mut line).expect("read");
            let req: serde_json::Value = serde_json::from_str(line.trim()).expect("parse");
            let id = req.get("id").cloned().unwrap_or(serde_json::Value::Null);
            let resp = serde_json::json!({
                "jsonrpc": "2.0",
                "result": {
                    "valid": true,
                    "gpu_available": true,
                    "precision_tier": "f64",
                    "estimated_dispatch_time_ms": 100,
                    "warnings": [],
                    "required_capabilities": ["compute.dispatch.submit"],
                    "dry_run": true,
                },
                "id": id,
            });
            let mut payload = serde_json::to_vec(&resp).expect("serialize");
            payload.push(b'\n');
            stream.write_all(&payload).expect("write");
            stream.flush().ok();
        });

        let transport = Transport::Tcp(addr);
        let result = validate_via(&transport, "workloads/airspring/test.toml", true);
        server.join().expect("join");

        let vr = result.expect("validate via TCP");
        assert!(vr.valid);
        assert!(vr.gpu_available);
        assert_eq!(vr.precision_tier, "f64");
        assert_eq!(vr.estimated_dispatch_time_ms, 100);
        assert!(vr.warnings.is_empty());
        assert_eq!(vr.required_capabilities, vec!["compute.dispatch.submit"]);
    }

    #[test]
    fn tcp_validate_rpc_error() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        let addr = listener.local_addr().expect("addr");

        let server = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("accept");
            let mut reader = BufReader::new(&stream);
            let mut line = String::new();
            reader.read_line(&mut line).expect("read");
            let req: serde_json::Value = serde_json::from_str(line.trim()).expect("parse");
            let id = req.get("id").cloned().unwrap_or(serde_json::Value::Null);
            let resp = serde_json::json!({
                "jsonrpc": "2.0",
                "error": { "code": -32601, "message": "method not found" },
                "id": id,
            });
            let mut payload = serde_json::to_vec(&resp).expect("serialize");
            payload.push(b'\n');
            stream.write_all(&payload).expect("write");
            stream.flush().ok();
        });

        let transport = Transport::Tcp(addr);
        let result = validate_via(&transport, "test.toml", true);
        server.join().expect("join");

        match result {
            Err(ValidateError::RpcError { code, message }) => {
                assert_eq!(code, -32601);
                assert_eq!(message, "method not found");
            }
            other => panic!("expected RpcError, got {other:?}"),
        }
    }

    #[test]
    fn parse_result_valid_response() {
        let resp = serde_json::json!({
            "jsonrpc": "2.0",
            "result": {
                "valid": false,
                "gpu_available": false,
                "precision_tier": "f32",
                "estimated_dispatch_time_ms": 0,
                "warnings": ["no GPU detected"],
                "required_capabilities": [],
            },
            "id": 1,
        });
        let vr = parse_result(&resp).expect("parse");
        assert!(!vr.valid);
        assert!(!vr.gpu_available);
        assert_eq!(vr.precision_tier, "f32");
        assert_eq!(vr.warnings, vec!["no GPU detected"]);
    }

    #[test]
    fn parse_result_missing_valid_field() {
        let resp = serde_json::json!({
            "jsonrpc": "2.0",
            "result": { "gpu_available": true },
            "id": 1,
        });
        let result = parse_result(&resp);
        assert!(matches!(result, Err(ValidateError::MalformedResponse(_))));
    }

    #[test]
    fn validate_error_display() {
        assert_eq!(
            format!("{}", ValidateError::NoPrimal),
            "no toadStool primal discovered"
        );
        let e = ValidateError::RpcError {
            code: -32600,
            message: "bad".into(),
        };
        assert!(format!("{e}").contains("-32600"));
    }

    #[test]
    fn validate_error_is_error_trait() {
        let e: Box<dyn std::error::Error> = Box::new(ValidateError::NoPrimal);
        assert!(!e.to_string().is_empty());
    }

    #[test]
    fn toadstool_transport_uses_standard_env_keys() {
        assert_eq!(
            primal_names::socket_env_var(primal_names::TOADSTOOL),
            "TOADSTOOL_SOCKET"
        );
    }
}
