// SPDX-License-Identifier: AGPL-3.0-or-later

//! JSON-RPC 2.0 infrastructure for biomeOS IPC.
//!
//! Provides helpers for constructing JSON-RPC 2.0 requests and responses,
//! and for sending requests over Unix domain sockets or TCP to biomeOS primals.
//!
//! # Platform-Agnostic Transport (ecoBin Standard)
//!
//! The module supports two transports:
//!
//! - **Unix** (`Transport::Unix`) — Unix domain sockets (Unix/macOS only).
//!   Uses `std::os::unix::net::UnixStream` when available.
//!
//! - **TCP** (`Transport::Tcp`) — TCP sockets for Windows and cross-platform use.
//!   Uses `std::net::TcpStream`; no extra dependencies.
//!
//! Use [`resolve_transport`] to resolve a primal's transport from environment
//! variables (`{PRIMAL}_SOCKET`, `{PRIMAL}_ADDRESS`) or biomeOS discovery.
//!
//! Evolution path: Unix + TCP → Songbird relay (sovereign TLS 1.3) for remote primals.

mod error;
mod transport;

pub use error::IpcError;
pub use transport::{Transport, TransportStream, connect_transport};

use std::io::{BufRead, BufReader, Write};
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

// ── JSON-RPC 2.0 error codes ───────────────────────────────────────────────

/// JSON-RPC 2.0 parse error (malformed JSON).
pub const PARSE_ERROR: i32 = -32700;
/// JSON-RPC 2.0 invalid request (missing required fields).
pub const INVALID_REQUEST: i32 = -32600;
/// JSON-RPC 2.0 method not found.
pub const METHOD_NOT_FOUND: i32 = -32601;
/// JSON-RPC 2.0 invalid parameters.
pub const INVALID_PARAMS: i32 = -32602;
/// JSON-RPC 2.0 internal error.
pub const INTERNAL_ERROR: i32 = -32603;

// ── Protocol helpers ────────────────────────────────────────────────────────

/// Strip legacy primal namespace prefix from JSON-RPC method names (barraCuda v0.3.7 semantic naming).
///
/// Canonical names are `{domain}.{operation}` (for example `science.et0_fao56`). Legacy clients may send
/// `airspring.science.et0_fao56`; this removes a leading `{PRIMAL_NAME}.` when present.
///
/// The prefix is derived from [`crate::PRIMAL_NAME`] — no hardcoded primal strings.
#[must_use]
pub fn normalize_method(method: &str) -> &str {
    method
        .strip_prefix(crate::PRIMAL_NAME)
        .and_then(|rest| rest.strip_prefix('.'))
        .unwrap_or(method)
}

/// Extract a JSON-RPC error code and message from a response.
///
/// Returns `Some((code, message))` if the response contains an `error` object
/// with `code` and `message` fields, `None` otherwise.
#[must_use]
pub fn extract_rpc_error(response: &serde_json::Value) -> Option<(i64, String)> {
    let err = response.get("error")?;
    let code = err.get("code")?.as_i64()?;
    let message = err.get("message")?.as_str()?.to_owned();
    Some((code, message))
}

static REQUEST_ID: AtomicU64 = AtomicU64::new(0);

/// Default RPC timeout when `BIOMEOS_RPC_TIMEOUT_SECS` is unset (seconds).
const DEFAULT_RPC_TIMEOUT_SECS: u64 = 5;

fn socket_timeout() -> Duration {
    std::env::var("BIOMEOS_RPC_TIMEOUT_SECS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .map_or(
            Duration::from_secs(DEFAULT_RPC_TIMEOUT_SECS),
            Duration::from_secs,
        )
}

/// Constructs a JSON-RPC 2.0 success response.
///
/// # Examples
///
/// ```
/// # use airspring_barracuda::rpc;
/// let id = serde_json::json!(1);
/// let result = serde_json::json!({"status": "ok"});
/// let resp = rpc::success(&id, &result);
/// assert!(resp.get("result").is_some());
/// assert_eq!(resp["jsonrpc"], "2.0");
/// ```
#[must_use]
pub fn success(id: &serde_json::Value, result: &serde_json::Value) -> serde_json::Value {
    serde_json::json!({
        "jsonrpc": "2.0",
        "result": result,
        "id": id,
    })
}

/// Constructs a JSON-RPC 2.0 error response.
///
/// # Examples
///
/// ```
/// # use airspring_barracuda::rpc;
/// let id = serde_json::json!(1);
/// let resp = rpc::error(&id, rpc::METHOD_NOT_FOUND, "method not found");
/// assert!(resp.get("error").is_some());
/// assert_eq!(resp["error"]["code"], -32601);
/// ```
#[must_use]
pub fn error(id: &serde_json::Value, code: i32, message: &str) -> serde_json::Value {
    serde_json::json!({
        "jsonrpc": "2.0",
        "error": { "code": code, "message": message },
        "id": id,
    })
}

/// Constructs a JSON-RPC 2.0 request with auto-incrementing id.
///
/// Each call returns a new request with a unique id for correlation.
///
/// # Examples
///
/// ```
/// # use airspring_barracuda::rpc;
/// let empty = serde_json::json!({});
/// let req = rpc::request("health", &empty);
/// assert_eq!(req["jsonrpc"], "2.0");
/// assert_eq!(req["method"], "health");
/// assert!(req.get("id").is_some());
/// ```
#[must_use]
pub fn request(method: &str, params: &serde_json::Value) -> serde_json::Value {
    let id = REQUEST_ID.fetch_add(1, Ordering::Relaxed);
    serde_json::json!({
        "jsonrpc": "2.0",
        "method": method,
        "params": params,
        "id": id,
    })
}

// ── Send / resolve ──────────────────────────────────────────────────────────

fn io_error_to_ipc(
    e: std::io::Error,
    method: &str,
    timeout_dur: Duration,
    transport: &Transport,
) -> IpcError {
    if e.kind() == std::io::ErrorKind::TimedOut {
        IpcError::Timeout {
            method: method.to_string(),
            elapsed: timeout_dur,
        }
    } else {
        match transport {
            #[cfg(unix)]
            Transport::Unix(socket) => IpcError::WriteFailed {
                socket: socket.clone(),
                source: e,
            },
            Transport::Tcp(addr) => IpcError::WriteFailedTcp {
                addr: *addr,
                source: e,
            },
        }
    }
}

fn io_read_error_to_ipc(
    e: std::io::Error,
    method: &str,
    timeout_dur: Duration,
    transport: &Transport,
) -> IpcError {
    if e.kind() == std::io::ErrorKind::TimedOut {
        IpcError::Timeout {
            method: method.to_string(),
            elapsed: timeout_dur,
        }
    } else {
        match transport {
            #[cfg(unix)]
            Transport::Unix(socket) => IpcError::ReadFailed {
                socket: socket.clone(),
                source: e,
            },
            Transport::Tcp(addr) => IpcError::ReadFailedTcp {
                addr: *addr,
                source: e,
            },
        }
    }
}

/// Sends a JSON-RPC request over the given transport and reads the response.
///
/// Uses newline-delimited framing with configurable timeouts
/// (`BIOMEOS_RPC_TIMEOUT_SECS`, default 5s).
/// Returns the full JSON-RPC response (including `result` or `error`).
///
/// # Errors
///
/// Returns `Err(IpcError)` if connection, write, or read fails.
pub fn send_to(
    transport: &Transport,
    method: &str,
    params: &serde_json::Value,
) -> Result<serde_json::Value, IpcError> {
    let mut stream = connect_transport(transport)?;
    let timeout_dur = socket_timeout();
    stream.set_timeouts(transport, Some(timeout_dur))?;

    let req = request(method, params);
    let mut payload = serde_json::to_vec(&req).map_err(|e| IpcError::DeserializationFailed {
        method: method.to_string(),
        source: e,
    })?;
    payload.push(b'\n');

    stream
        .write_all(&payload)
        .map_err(|e| io_error_to_ipc(e, method, timeout_dur, transport))?;
    stream
        .flush()
        .map_err(|e| io_error_to_ipc(e, method, timeout_dur, transport))?;

    let mut reader = BufReader::new(stream);
    let mut line = String::new();
    reader
        .read_line(&mut line)
        .map_err(|e| io_read_error_to_ipc(e, method, timeout_dur, transport))?;

    serde_json::from_str(line.trim()).map_err(|e| IpcError::DeserializationFailed {
        method: method.to_string(),
        source: e,
    })
}

/// Resolves a primal's transport from environment or biomeOS discovery.
///
/// Resolution order:
/// 1. `{PRIMAL}_SOCKET` env var → `Transport::Unix` (Unix only)
/// 2. `{PRIMAL}_ADDRESS` env var → `Transport::Tcp` (parse as `host:port`)
/// 3. biomeOS discovery (`discover_primal_socket`) → `Transport::Unix` (Unix only)
///
/// On non-Unix platforms, only TCP transport is available; the `_SOCKET` env var
/// and biomeOS fallback are skipped.
///
/// # Errors
///
/// Returns `Err(IpcError::SocketNotFound)` if no transport can be resolved.
pub fn resolve_transport(primal: &str) -> Result<Transport, IpcError> {
    let socket_var = crate::primal_names::socket_env_var(primal);
    let address_var = crate::primal_names::address_env_var(primal);

    #[cfg(unix)]
    if let Ok(path) = std::env::var(&socket_var) {
        let p = PathBuf::from(path);
        return Ok(Transport::Unix(p));
    }

    if let Ok(addr_str) = std::env::var(&address_var)
        && let Ok(addr) = addr_str.parse::<SocketAddr>()
    {
        return Ok(Transport::Tcp(addr));
    }

    #[cfg(unix)]
    if let Some(path) = crate::biomeos::discover_primal_socket(primal) {
        return Ok(Transport::Unix(path));
    }

    Err(IpcError::SocketNotFound {
        primal: primal.to_string(),
    })
}

/// Sends a JSON-RPC request over a Unix socket and reads the response.
///
/// Uses newline-delimited framing with configurable timeouts
/// (`BIOMEOS_RPC_TIMEOUT_SECS`, default 5s).
/// Returns the full JSON-RPC response (including `result` or `error`).
///
/// On non-Unix platforms, returns `Err(IpcError::UnixNotAvailable)` — use
/// [`send_to`] with `Transport::Tcp` or [`resolve_transport`] instead.
///
/// # Errors
///
/// Returns `Err(IpcError)` if:
/// - Connection to the socket fails
/// - Write/read times out
/// - Response is not valid JSON
///
/// # Examples
///
/// ```rust,no_run
/// use airspring_barracuda::rpc;
/// use std::path::Path;
///
/// let path = Path::new("/run/user/1000/biomeos/airspring-abc.sock");
/// if let Ok(resp) = rpc::send(path, "health", &serde_json::json!({})) {
///     let result = resp.get("result").cloned();
///     // ...
/// }
/// ```
pub fn send(
    socket_path: &Path,
    method: &str,
    params: &serde_json::Value,
) -> Result<serde_json::Value, IpcError> {
    #[cfg(unix)]
    {
        send_to(&Transport::Unix(socket_path.to_path_buf()), method, params)
    }

    #[cfg(not(unix))]
    {
        let _ = (socket_path, method, params);
        Err(IpcError::UnixNotAvailable)
    }
}

/// Extract the JSON-RPC `result` from a Unix socket call.
///
/// # Errors
///
/// Returns [`IpcError`] on transport failure, JSON-RPC `error`, or missing `result`.
pub fn call_unix(
    socket_path: &Path,
    method: &str,
    params: &serde_json::Value,
) -> Result<serde_json::Value, IpcError> {
    let resp = send(socket_path, method, params)?;
    if let Some((code, message)) = extract_rpc_error(&resp) {
        return Err(IpcError::RpcError {
            code: i32::try_from(code).unwrap_or(INTERNAL_ERROR),
            message,
        });
    }
    resp.get("result")
        .cloned()
        .ok_or_else(|| IpcError::EmptyResponse {
            method: method.to_string(),
        })
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
