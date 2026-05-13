// SPDX-License-Identifier: AGPL-3.0-or-later
//! Typed client for `NestGate` content-addressed storage.
//!
//! `NestGate` is a **storage primal** — it does not implement `data.*` feed
//! methods.  Weather data routing goes through Songbird HTTP or the
//! `capability.call` semantic path.  This module provides typed access
//! to `NestGate`'s actual wire surface:
//!
//! - `content.store` / `content.get` — content-addressed storage (CAS)
//! - `storage.status` — storage health and capacity
//!
//! Non-fatal when `NestGate` is unavailable — callers fall back to local
//! storage or Songbird HTTP.

use crate::primal_names;
use crate::rpc::{self, IpcError, Transport};

/// Result of a `content.store` operation.
#[derive(Debug, Clone)]
pub struct StoreResult {
    /// Content hash (BLAKE3 or similar) returned by `NestGate`.
    pub hash: String,
    /// Size in bytes as reported by the store.
    pub size: u64,
    /// Whether the content was already present (deduplicated).
    pub deduplicated: bool,
}

/// Result of a `content.get` operation.
#[derive(Debug, Clone)]
pub struct GetResult {
    /// The retrieved content as a JSON value.
    pub content: serde_json::Value,
    /// Content hash for verification.
    pub hash: String,
}

/// Result of a `storage.status` query.
#[derive(Debug, Clone)]
pub struct StorageStatus {
    /// Whether the storage backend is healthy.
    pub healthy: bool,
    /// Free capacity in bytes (0 if unknown).
    pub free_bytes: u64,
    /// Backend type (e.g. "zfs", "local", "memory").
    pub backend: String,
}

/// Errors from `NestGate` operations.
#[derive(Debug)]
pub enum NestGateError {
    /// No `NestGate` primal discovered.
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

impl std::fmt::Display for NestGateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoPrimal => write!(f, "no NestGate primal discovered"),
            Self::Ipc(e) => write!(f, "IPC error: {e}"),
            Self::RpcError { code, message } => write!(f, "RPC error {code}: {message}"),
            Self::MalformedResponse(detail) => {
                write!(f, "NestGate malformed response: {detail}")
            }
        }
    }
}

impl std::error::Error for NestGateError {}

impl From<IpcError> for NestGateError {
    fn from(e: IpcError) -> Self {
        match e {
            IpcError::SocketNotFound { .. } => Self::NoPrimal,
            e => Self::Ipc(e),
        }
    }
}

fn discover() -> Result<Transport, NestGateError> {
    rpc::resolve_transport(primal_names::NESTGATE).map_err(NestGateError::from)
}

fn check_rpc_error(resp: &serde_json::Value) -> Result<(), NestGateError> {
    if let Some((code, message)) = rpc::extract_rpc_error(resp) {
        return Err(NestGateError::RpcError { code, message });
    }
    Ok(())
}

/// Store content in `NestGate`'s content-addressed storage.
///
/// # Errors
///
/// Returns [`NestGateError::NoPrimal`] if `NestGate` is not discovered.
/// Returns [`NestGateError::Ipc`] on transport failure.
/// Returns [`NestGateError::RpcError`] if the server returns an RPC error.
/// Returns [`NestGateError::MalformedResponse`] if the response shape is unexpected.
pub fn content_store(key: &str, content: &serde_json::Value) -> Result<StoreResult, NestGateError> {
    let transport = discover()?;
    content_store_via(&transport, key, content)
}

/// Store content against a specific transport (for testing).
///
/// # Errors
///
/// Same as [`content_store`].
pub fn content_store_via(
    transport: &Transport,
    key: &str,
    content: &serde_json::Value,
) -> Result<StoreResult, NestGateError> {
    let resp = rpc::send_to(
        transport,
        "content.store",
        &serde_json::json!({
            "key": key,
            "content": content,
        }),
    )?;
    check_rpc_error(&resp)?;

    let r = resp.get("result").or(Some(&resp));
    let hash = r
        .and_then(|v| v.get("hash"))
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| NestGateError::MalformedResponse("missing `hash`".into()))?
        .to_owned();
    let size = r
        .and_then(|v| v.get("size"))
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    let deduplicated = r
        .and_then(|v| v.get("deduplicated"))
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false);

    Ok(StoreResult {
        hash,
        size,
        deduplicated,
    })
}

/// Retrieve content from `NestGate` by hash.
///
/// # Errors
///
/// Same as [`content_store`].
pub fn content_get(hash: &str) -> Result<GetResult, NestGateError> {
    let transport = discover()?;
    content_get_via(&transport, hash)
}

/// Retrieve content against a specific transport (for testing).
///
/// # Errors
///
/// Same as [`content_store`].
pub fn content_get_via(
    transport: &Transport,
    hash: &str,
) -> Result<GetResult, NestGateError> {
    let resp = rpc::send_to(
        transport,
        "content.get",
        &serde_json::json!({ "hash": hash }),
    )?;
    check_rpc_error(&resp)?;

    let r = resp.get("result").or(Some(&resp));
    let content = r
        .and_then(|v| v.get("content"))
        .cloned()
        .unwrap_or(serde_json::Value::Null);
    let hash_out = r
        .and_then(|v| v.get("hash"))
        .and_then(serde_json::Value::as_str)
        .unwrap_or(hash)
        .to_owned();

    Ok(GetResult {
        content,
        hash: hash_out,
    })
}

/// Query `NestGate` storage status.
///
/// # Errors
///
/// Same as [`content_store`].
pub fn storage_status() -> Result<StorageStatus, NestGateError> {
    let transport = discover()?;
    storage_status_via(&transport)
}

/// Query storage status against a specific transport (for testing).
///
/// # Errors
///
/// Same as [`content_store`].
pub fn storage_status_via(transport: &Transport) -> Result<StorageStatus, NestGateError> {
    let resp = rpc::send_to(
        transport,
        "storage.status",
        &serde_json::json!({}),
    )?;
    check_rpc_error(&resp)?;

    let r = resp.get("result").or(Some(&resp));
    let healthy = r
        .and_then(|v| v.get("healthy"))
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false);
    let free_bytes = r
        .and_then(|v| v.get("free_bytes"))
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    let backend = r
        .and_then(|v| v.get("backend"))
        .and_then(serde_json::Value::as_str)
        .unwrap_or("unknown")
        .to_owned();

    Ok(StorageStatus {
        healthy,
        free_bytes,
        backend,
    })
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test code")]
mod tests {
    use super::*;
    use std::io::{BufRead, BufReader, Write};
    use std::net::TcpListener;

    #[test]
    fn no_primal_returns_error() {
        let result = content_store("test-key", &serde_json::json!({"data": 1}));
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            NestGateError::NoPrimal | NestGateError::Ipc(_)
        ));
    }

    #[test]
    fn tcp_content_store_round_trip() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        let addr = listener.local_addr().expect("addr");

        let server = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("accept");
            let mut reader = BufReader::new(&stream);
            let mut line = String::new();
            reader.read_line(&mut line).expect("read");
            let req: serde_json::Value = serde_json::from_str(line.trim()).expect("parse");
            let id = req.get("id").cloned().unwrap_or(serde_json::Value::Null);

            assert_eq!(req["method"], "content.store");
            assert_eq!(req["params"]["key"], "weather/mi/2024");

            let resp = serde_json::json!({
                "jsonrpc": "2.0",
                "result": {
                    "hash": "b3sum_abc123",
                    "size": 4096,
                    "deduplicated": false,
                },
                "id": id,
            });
            let mut payload = serde_json::to_vec(&resp).expect("serialize");
            payload.push(b'\n');
            stream.write_all(&payload).expect("write");
            stream.flush().ok();
        });

        let transport = Transport::Tcp(addr);
        let result = content_store_via(
            &transport,
            "weather/mi/2024",
            &serde_json::json!({"stations": 100}),
        );
        server.join().expect("join");

        let sr = result.expect("store via TCP");
        assert_eq!(sr.hash, "b3sum_abc123");
        assert_eq!(sr.size, 4096);
        assert!(!sr.deduplicated);
    }

    #[test]
    fn tcp_content_get_round_trip() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        let addr = listener.local_addr().expect("addr");

        let server = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("accept");
            let mut reader = BufReader::new(&stream);
            let mut line = String::new();
            reader.read_line(&mut line).expect("read");
            let req: serde_json::Value = serde_json::from_str(line.trim()).expect("parse");
            let id = req.get("id").cloned().unwrap_or(serde_json::Value::Null);

            assert_eq!(req["method"], "content.get");
            assert_eq!(req["params"]["hash"], "b3sum_abc123");

            let resp = serde_json::json!({
                "jsonrpc": "2.0",
                "result": {
                    "content": {"stations": 100},
                    "hash": "b3sum_abc123",
                },
                "id": id,
            });
            let mut payload = serde_json::to_vec(&resp).expect("serialize");
            payload.push(b'\n');
            stream.write_all(&payload).expect("write");
            stream.flush().ok();
        });

        let transport = Transport::Tcp(addr);
        let result = content_get_via(&transport, "b3sum_abc123");
        server.join().expect("join");

        let gr = result.expect("get via TCP");
        assert_eq!(gr.hash, "b3sum_abc123");
        assert_eq!(gr.content["stations"], 100);
    }

    #[test]
    fn tcp_storage_status_round_trip() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        let addr = listener.local_addr().expect("addr");

        let server = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("accept");
            let mut reader = BufReader::new(&stream);
            let mut line = String::new();
            reader.read_line(&mut line).expect("read");
            let req: serde_json::Value = serde_json::from_str(line.trim()).expect("parse");
            let id = req.get("id").cloned().unwrap_or(serde_json::Value::Null);

            assert_eq!(req["method"], "storage.status");

            let resp = serde_json::json!({
                "jsonrpc": "2.0",
                "result": {
                    "healthy": true,
                    "free_bytes": 1_073_741_824_u64,
                    "backend": "zfs",
                },
                "id": id,
            });
            let mut payload = serde_json::to_vec(&resp).expect("serialize");
            payload.push(b'\n');
            stream.write_all(&payload).expect("write");
            stream.flush().ok();
        });

        let transport = Transport::Tcp(addr);
        let result = storage_status_via(&transport);
        server.join().expect("join");

        let ss = result.expect("status via TCP");
        assert!(ss.healthy);
        assert_eq!(ss.free_bytes, 1_073_741_824);
        assert_eq!(ss.backend, "zfs");
    }

    #[test]
    fn tcp_rpc_error() {
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
        let result = content_store_via(
            &transport,
            "test",
            &serde_json::json!({}),
        );
        server.join().expect("join");

        match result {
            Err(NestGateError::RpcError { code, message }) => {
                assert_eq!(code, -32601);
                assert_eq!(message, "method not found");
            }
            other => panic!("expected RpcError, got {other:?}"),
        }
    }

    #[test]
    fn nestgate_error_display() {
        assert_eq!(
            format!("{}", NestGateError::NoPrimal),
            "no NestGate primal discovered"
        );
    }

    #[test]
    fn nestgate_error_is_error_trait() {
        let e: Box<dyn std::error::Error> = Box::new(NestGateError::NoPrimal);
        assert!(!e.to_string().is_empty());
    }

    #[test]
    fn nestgate_transport_uses_standard_env_keys() {
        assert_eq!(
            primal_names::socket_env_var(primal_names::NESTGATE),
            "NESTGATE_SOCKET"
        );
    }
}
