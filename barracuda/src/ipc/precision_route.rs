// SPDX-License-Identifier: AGPL-3.0-or-later
//! Typed client for barraCuda `precision.route` advisory.
//!
//! Asks the barraCuda primal which precision tier a given domain should
//! use on the current hardware.  Enables Tier 2 convergence where springs
//! dynamically select f32/f64/df64 based on live GPU capabilities.
//!
//! Non-fatal when barraCuda is unavailable — callers fall back to
//! conservative f64 precision locally.

use crate::primal_names;
use crate::rpc::{self, IpcError, Transport};

/// Result of a `precision.route` advisory call.
#[derive(Debug, Clone)]
pub struct PrecisionAdvice {
    /// Recommended precision tier (e.g. "f32", "f64", "df64").
    pub recommended_tier: String,
    /// Whether FMA (fused multiply-add) is safe on this hardware.
    pub fma_safe: bool,
    /// Whether sovereign (coralReef) compilation is needed.
    pub needs_sovereign_compile: bool,
    /// Whether a shader compiler is required (from `requires_compiler`).
    pub requires_compiler: bool,
    /// Hardware hint returned by barraCuda (e.g. "compute", "integrated").
    pub hardware_hint: String,
    /// GPU adapter name (e.g. "NVIDIA TITAN V"), empty if not reported.
    pub adapter: String,
    /// Human-readable rationale for the recommendation.
    pub rationale: String,
}

/// Errors from precision route operations.
#[derive(Debug)]
pub enum PrecisionError {
    /// No barraCuda primal discovered.
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

impl std::fmt::Display for PrecisionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoPrimal => write!(f, "no barraCuda primal discovered"),
            Self::Ipc(e) => write!(f, "IPC error: {e}"),
            Self::RpcError { code, message } => write!(f, "RPC error {code}: {message}"),
            Self::MalformedResponse(detail) => {
                write!(f, "precision.route malformed response: {detail}")
            }
        }
    }
}

impl std::error::Error for PrecisionError {}

impl From<IpcError> for PrecisionError {
    fn from(e: IpcError) -> Self {
        match e {
            IpcError::SocketNotFound { .. } => Self::NoPrimal,
            e => Self::Ipc(e),
        }
    }
}

fn discover() -> Result<Transport, PrecisionError> {
    rpc::resolve_transport(primal_names::BARRACUDA).map_err(PrecisionError::from)
}

fn parse_result(resp: &serde_json::Value) -> Result<PrecisionAdvice, PrecisionError> {
    if let Some((code, message)) = rpc::extract_rpc_error(resp) {
        return Err(PrecisionError::RpcError { code, message });
    }

    let r = resp.get("result").or(Some(resp));

    let recommended_tier = r
        .and_then(|v| v.get("recommended_tier"))
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| {
            PrecisionError::MalformedResponse("missing `recommended_tier`".into())
        })?
        .to_owned();

    let fma_safe = r
        .and_then(|v| v.get("fma_safe"))
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false);

    let needs_sovereign_compile = r
        .and_then(|v| v.get("needs_sovereign_compile"))
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false);

    let hardware_hint = r
        .and_then(|v| v.get("hardware_hint"))
        .and_then(serde_json::Value::as_str)
        .unwrap_or("unknown")
        .to_owned();

    let requires_compiler = r
        .and_then(|v| v.get("requires_compiler"))
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false);

    let adapter = r
        .and_then(|v| v.get("adapter"))
        .and_then(serde_json::Value::as_str)
        .unwrap_or("")
        .to_owned();

    let rationale = r
        .and_then(|v| v.get("rationale"))
        .and_then(serde_json::Value::as_str)
        .unwrap_or("")
        .to_owned();

    Ok(PrecisionAdvice {
        recommended_tier,
        fma_safe,
        needs_sovereign_compile,
        requires_compiler,
        hardware_hint,
        adapter,
        rationale,
    })
}

/// Query barraCuda for precision routing advice for a given domain.
///
/// Domain examples: `"hydrology"`, `"statistics"`, `"general"`, `"lattice_qcd"`.
///
/// # Errors
///
/// Returns [`PrecisionError::NoPrimal`] if barraCuda is not discovered.
/// Returns [`PrecisionError::Ipc`] on transport failure.
/// Returns [`PrecisionError::RpcError`] if the server returns an RPC error.
/// Returns [`PrecisionError::MalformedResponse`] if the response shape is unexpected.
pub fn route(domain: &str) -> Result<PrecisionAdvice, PrecisionError> {
    let transport = discover()?;
    route_via(&transport, domain)
}

/// Query precision routing against a specific transport (for testing).
///
/// # Errors
///
/// Same as [`route`].
pub fn route_via(
    transport: &Transport,
    domain: &str,
) -> Result<PrecisionAdvice, PrecisionError> {
    let resp = rpc::send_to(
        transport,
        "precision.route",
        &serde_json::json!({ "domain": domain }),
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
        let result = route("hydrology");
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            PrecisionError::NoPrimal | PrecisionError::Ipc(_)
        ));
    }

    #[test]
    fn tcp_route_round_trip() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        let addr = listener.local_addr().expect("addr");

        let server = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("accept");
            let mut reader = BufReader::new(&stream);
            let mut line = String::new();
            reader.read_line(&mut line).expect("read");
            let req: serde_json::Value = serde_json::from_str(line.trim()).expect("parse");
            let id = req.get("id").cloned().unwrap_or(serde_json::Value::Null);

            assert_eq!(req["method"], "precision.route");
            assert_eq!(req["params"]["domain"], "hydrology");

            let resp = serde_json::json!({
                "jsonrpc": "2.0",
                "result": {
                    "recommended_tier": "f64",
                    "fma_safe": true,
                    "needs_sovereign_compile": false,
                    "hardware_hint": "compute",
                    "rationale": "Titan V discrete GPU with native f64",
                    "adapter": "NVIDIA TITAN V",
                },
                "id": id,
            });
            let mut payload = serde_json::to_vec(&resp).expect("serialize");
            payload.push(b'\n');
            stream.write_all(&payload).expect("write");
            stream.flush().ok();
        });

        let transport = Transport::Tcp(addr);
        let result = route_via(&transport, "hydrology");
        server.join().expect("join");

        let pa = result.expect("route via TCP");
        assert_eq!(pa.recommended_tier, "f64");
        assert!(pa.fma_safe);
        assert!(!pa.needs_sovereign_compile);
        assert!(!pa.requires_compiler);
        assert_eq!(pa.hardware_hint, "compute");
        assert_eq!(pa.adapter, "NVIDIA TITAN V");
        assert!(pa.rationale.contains("Titan V"));
    }

    #[test]
    fn tcp_route_rpc_error() {
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
        let result = route_via(&transport, "hydrology");
        server.join().expect("join");

        match result {
            Err(PrecisionError::RpcError { code, message }) => {
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
                "recommended_tier": "df64",
                "fma_safe": false,
                "needs_sovereign_compile": true,
                "hardware_hint": "integrated",
                "rationale": "consumer GPU requires DF64 emulation",
            },
            "id": 1,
        });
        let pa = parse_result(&resp).expect("parse");
        assert_eq!(pa.recommended_tier, "df64");
        assert!(!pa.fma_safe);
        assert!(pa.needs_sovereign_compile);
        assert!(!pa.requires_compiler);
        assert_eq!(pa.hardware_hint, "integrated");
        assert!(pa.adapter.is_empty());
    }

    #[test]
    fn parse_result_missing_tier() {
        let resp = serde_json::json!({
            "jsonrpc": "2.0",
            "result": { "fma_safe": true },
            "id": 1,
        });
        assert!(matches!(
            parse_result(&resp),
            Err(PrecisionError::MalformedResponse(_))
        ));
    }

    #[test]
    fn precision_error_display() {
        assert_eq!(
            format!("{}", PrecisionError::NoPrimal),
            "no barraCuda primal discovered"
        );
        let e = PrecisionError::RpcError {
            code: -32600,
            message: "bad".into(),
        };
        assert!(format!("{e}").contains("-32600"));
    }

    #[test]
    fn precision_error_is_error_trait() {
        let e: Box<dyn std::error::Error> = Box::new(PrecisionError::NoPrimal);
        assert!(!e.to_string().is_empty());
    }

    #[test]
    fn barracuda_transport_uses_standard_env_keys() {
        assert_eq!(
            primal_names::socket_env_var(primal_names::BARRACUDA),
            "BARRACUDA_SOCKET"
        );
    }
}
