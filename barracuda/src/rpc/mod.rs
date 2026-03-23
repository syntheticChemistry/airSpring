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

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test code uses unwrap for clarity")]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn test_success() {
        let id = serde_json::json!(42);
        let result = serde_json::json!({"et0_mm": 5.2});
        let resp = success(&id, &result);
        assert_eq!(resp["jsonrpc"], "2.0");
        assert_eq!(resp["id"], 42);
        assert_eq!(resp["result"], result);
    }

    #[test]
    fn test_error() {
        let id = serde_json::json!(1);
        let resp = error(&id, METHOD_NOT_FOUND, "method not found");
        assert_eq!(resp["jsonrpc"], "2.0");
        assert_eq!(resp["id"], 1);
        assert_eq!(resp["error"]["code"], METHOD_NOT_FOUND);
        assert_eq!(resp["error"]["message"], "method not found");
    }

    #[test]
    fn test_error_with_null_id() {
        let id = serde_json::Value::Null;
        let resp = error(&id, PARSE_ERROR, "Parse error");
        assert_eq!(resp["error"]["code"], PARSE_ERROR);
        assert_eq!(resp["id"], serde_json::Value::Null);
    }

    #[test]
    fn test_request() {
        let params = serde_json::json!({"tmax": 32.0});
        let req = request("science.et0_fao56", &params);
        assert_eq!(req["jsonrpc"], "2.0");
        assert_eq!(req["method"], "science.et0_fao56");
        assert_eq!(req["params"]["tmax"], 32.0);
        assert!(req.get("id").is_some());
    }

    #[test]
    fn test_request_auto_incrementing_id() {
        let empty = serde_json::json!({});
        let req1 = request("a", &empty);
        let req2 = request("b", &empty);
        let id1 = req1["id"].as_u64().unwrap();
        let id2 = req2["id"].as_u64().unwrap();
        assert!(id2 > id1);
    }

    #[test]
    fn test_error_codes() {
        assert_eq!(PARSE_ERROR, -32700);
        assert_eq!(INVALID_REQUEST, -32600);
        assert_eq!(METHOD_NOT_FOUND, -32601);
        assert_eq!(INVALID_PARAMS, -32602);
        assert_eq!(INTERNAL_ERROR, -32603);
    }

    #[test]
    fn send_returns_err_for_nonexistent_socket() {
        let path = Path::new("/nonexistent/rpc/socket/path/that/does/not/exist.sock");
        let result = send(path, "health", &serde_json::json!({}));
        assert!(result.is_err());
    }

    #[test]
    fn success_response_has_no_error_field() {
        let id = serde_json::json!(1);
        let result = serde_json::json!({"ok": true});
        let resp = success(&id, &result);
        assert!(resp.get("error").is_none());
        assert!(resp.get("result").is_some());
    }

    #[test]
    fn error_response_has_no_result_field() {
        let id = serde_json::json!(1);
        let resp = error(&id, INVALID_PARAMS, "invalid params");
        assert!(resp.get("result").is_none());
        assert!(resp.get("error").is_some());
    }

    #[test]
    fn test_request_with_empty_object_params() {
        let empty = serde_json::json!({});
        let req = request("health", &empty);
        assert_eq!(req["method"], "health");
        assert_eq!(req["params"], serde_json::json!({}));
    }

    #[test]
    fn test_request_with_array_params() {
        let params = serde_json::json!([1, 2, 3]);
        let req = request("batch", &params);
        assert_eq!(req["params"], params);
    }

    #[test]
    fn test_request_with_null_params() {
        let params = serde_json::Value::Null;
        let req = request("notify", &params);
        assert_eq!(req["params"], serde_json::Value::Null);
    }

    #[test]
    fn test_success_with_string_id() {
        let id = serde_json::json!("req-42");
        let result = serde_json::json!({"status": "ok"});
        let resp = success(&id, &result);
        assert_eq!(resp["id"], "req-42");
        assert_eq!(resp["result"]["status"], "ok");
    }

    #[test]
    fn test_success_with_array_result() {
        let id = serde_json::json!(1);
        let result = serde_json::json!([1.0, 2.0, 3.0]);
        let resp = success(&id, &result);
        assert_eq!(resp["result"], result);
    }

    #[test]
    fn test_error_with_internal_error_code() {
        let id = serde_json::json!(99);
        let resp = error(&id, INTERNAL_ERROR, "internal server error");
        assert_eq!(resp["error"]["code"], INTERNAL_ERROR);
    }

    #[test]
    fn test_error_with_empty_message() {
        let id = serde_json::json!(1);
        let resp = error(&id, INVALID_REQUEST, "");
        assert_eq!(resp["error"]["message"], "");
    }

    #[test]
    fn send_returns_err_when_server_sends_malformed_json() {
        let dir = std::env::temp_dir().join(format!("rpc_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("malformed.sock");
        let _ = std::fs::remove_file(&path);

        let listener = std::os::unix::net::UnixListener::bind(&path).unwrap();
        std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let _ = stream.write_all(b"not valid json at all\n");
            let _ = stream.flush();
        });

        std::thread::sleep(std::time::Duration::from_millis(50));
        let result = send(&path, "health", &serde_json::json!({}));
        std::fs::remove_file(&path).ok();
        assert!(result.is_err());
    }

    #[test]
    fn send_returns_err_when_server_sends_empty_line() {
        let dir = std::env::temp_dir().join(format!("rpc_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("empty.sock");
        let _ = std::fs::remove_file(&path);

        let listener = std::os::unix::net::UnixListener::bind(&path).unwrap();
        std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let _ = stream.write_all(b"\n");
            let _ = stream.flush();
        });

        std::thread::sleep(std::time::Duration::from_millis(50));
        let result = send(&path, "health", &serde_json::json!({}));
        std::fs::remove_file(&path).ok();
        assert!(result.is_err());
    }

    #[test]
    fn send_returns_err_when_server_sends_partial_json() {
        let dir = std::env::temp_dir().join(format!("rpc_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("partial.sock");
        let _ = std::fs::remove_file(&path);

        let listener = std::os::unix::net::UnixListener::bind(&path).unwrap();
        std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let _ = stream.write_all(b"{\"jsonrpc\":\"2.0\"\n");
            let _ = stream.flush();
        });

        std::thread::sleep(std::time::Duration::from_millis(50));
        let result = send(&path, "health", &serde_json::json!({}));
        std::fs::remove_file(&path).ok();
        assert!(result.is_err());
    }

    #[test]
    fn send_returns_ok_when_server_sends_valid_json_response() {
        let dir = std::env::temp_dir().join(format!("rpc_valid_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("valid.sock");
        let _ = std::fs::remove_file(&path);

        let listener = std::os::unix::net::UnixListener::bind(&path).unwrap();
        std::thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            let mut reader = std::io::BufReader::new(&stream);
            let mut req = String::new();
            std::io::BufRead::read_line(&mut reader, &mut req).unwrap();
            let mut writer = &stream;
            let _ = std::io::Write::write_all(
                &mut writer,
                b"{\"jsonrpc\":\"2.0\",\"result\":{\"ok\":true},\"id\":1}\n",
            );
            let _ = std::io::Write::flush(&mut writer);
        });

        std::thread::sleep(std::time::Duration::from_millis(100));
        let result = send(&path, "health", &serde_json::json!({}));
        std::fs::remove_file(&path).ok();
        assert!(result.is_ok());
        let resp = result.unwrap();
        assert_eq!(resp["result"]["ok"], true);
        assert_eq!(resp["id"], 1);
    }

    #[test]
    fn send_returns_ok_when_server_sends_error_response() {
        let dir = std::env::temp_dir().join(format!("rpc_err_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("error_resp.sock");
        let _ = std::fs::remove_file(&path);

        let listener = std::os::unix::net::UnixListener::bind(&path).unwrap();
        std::thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            let mut reader = std::io::BufReader::new(&stream);
            let mut req = String::new();
            std::io::BufRead::read_line(&mut reader, &mut req).unwrap();
            let payload = format!(
                r#"{{"jsonrpc":"2.0","error":{{"code":{METHOD_NOT_FOUND},"message":"method not found"}},"id":1}}"#
            );
            let mut writer = &stream;
            let _ = std::io::Write::write_all(&mut writer, format!("{payload}\n").as_bytes());
            let _ = std::io::Write::flush(&mut writer);
        });

        std::thread::sleep(std::time::Duration::from_millis(100));
        let result = send(&path, "unknown", &serde_json::json!({}));
        std::fs::remove_file(&path).ok();
        assert!(result.is_ok());
        let resp = result.unwrap();
        assert!(resp.get("error").is_some());
        assert_eq!(resp["error"]["code"], METHOD_NOT_FOUND);
    }

    #[test]
    fn resolve_transport_returns_socket_not_found_for_unknown_primal() {
        let result = resolve_transport("nonexistent_rpc_test_primal_xyz");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, IpcError::SocketNotFound { .. }));
    }

    #[test]
    fn connect_transport_returns_err_for_unreachable_tcp() {
        let addr: SocketAddr = "127.0.0.1:1".parse().unwrap();
        let transport = Transport::Tcp(addr);
        let result = connect_transport(&transport);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            IpcError::ConnectionFailedTcp { .. }
        ));
    }

    #[test]
    #[cfg(unix)]
    fn connect_transport_returns_err_for_nonexistent_unix_socket() {
        let path = PathBuf::from("/nonexistent/rpc/connect_test.sock");
        let transport = Transport::Unix(path);
        let result = connect_transport(&transport);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            IpcError::ConnectionFailed { .. }
        ));
    }
}
