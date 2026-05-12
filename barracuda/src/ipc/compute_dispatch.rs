// SPDX-License-Identifier: AGPL-3.0-or-later
//! Typed client for toadStool `compute.dispatch` capability.
//!
//! Routes GPU workloads through toadStool instead of direct `wgpu` access.
//! Discovery is capability-based: `compute.dispatch.submit` is resolved
//! at runtime through [`crate::rpc::resolve_transport`] (env override or
//! biomeOS socket scanning on Unix).

use crate::primal_names;
use crate::rpc::{self, IpcError, Transport};

/// Handle to a dispatched compute job.
#[derive(Debug)]
pub struct DispatchHandle {
    /// Job identifier returned by toadStool.
    pub job_id: String,
    /// Transport used to reach the compute primal (Unix or TCP).
    pub transport: Transport,
}

/// Errors from compute dispatch operations.
#[derive(Debug)]
pub enum DispatchError {
    /// No compute primal discovered.
    NoComputePrimal,
    /// IPC transport error.
    Ipc(IpcError),
    /// Server did not return a `job_id`.
    MissingJobId,
    /// Server returned an RPC error.
    RpcError {
        /// JSON-RPC error code.
        code: i64,
        /// Human-readable error message.
        message: String,
    },
}

impl std::fmt::Display for DispatchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoComputePrimal => write!(f, "no compute primal discovered"),
            Self::Ipc(e) => write!(f, "IPC error: {e}"),
            Self::MissingJobId => write!(f, "compute.dispatch.submit did not return job_id"),
            Self::RpcError { code, message } => write!(f, "RPC error {code}: {message}"),
        }
    }
}

impl std::error::Error for DispatchError {}

impl From<IpcError> for DispatchError {
    fn from(e: IpcError) -> Self {
        match e {
            IpcError::SocketNotFound { .. } => Self::NoComputePrimal,
            e => Self::Ipc(e),
        }
    }
}

/// Discover the compute primal transport via [`rpc::resolve_transport`] for the
/// toadStool primal (`TOADSTOOL_SOCKET` / `TOADSTOOL_ADDRESS` / biomeOS discovery).
fn discover_compute_transport() -> Result<Transport, DispatchError> {
    rpc::resolve_transport(primal_names::TOADSTOOL).map_err(DispatchError::from)
}

fn submit_to_transport(
    transport: &Transport,
    workload_type: &str,
    params: &serde_json::Value,
) -> Result<DispatchHandle, DispatchError> {
    let result = rpc::send_to(
        transport,
        "compute.dispatch.submit",
        &serde_json::json!({
            "workload": workload_type,
            "params": params,
        }),
    )?;

    if let Some((code, message)) = rpc::extract_rpc_error(&result) {
        return Err(DispatchError::RpcError { code, message });
    }

    let job_id = result
        .get("result")
        .or(Some(&result))
        .and_then(|r| r.get("job_id"))
        .and_then(serde_json::Value::as_str)
        .ok_or(DispatchError::MissingJobId)?
        .to_owned();

    Ok(DispatchHandle {
        job_id,
        transport: transport.clone(),
    })
}

/// Submit a GPU workload to the compute primal.
///
/// Returns a [`DispatchHandle`] for polling results.
///
/// # Errors
///
/// Returns [`DispatchError::NoComputePrimal`] if no compute primal transport can be resolved.
/// Returns [`DispatchError::Ipc`] on transport failure.
/// Returns [`DispatchError::MissingJobId`] if the server response lacks `job_id`.
/// Returns [`DispatchError::RpcError`] if the server returns an RPC error.
pub fn submit(
    workload_type: &str,
    params: &serde_json::Value,
) -> Result<DispatchHandle, DispatchError> {
    let transport = discover_compute_transport()?;
    submit_to_transport(&transport, workload_type, params)
}

/// Poll for the result of a dispatched compute job.
///
/// # Errors
///
/// Returns [`DispatchError::Ipc`] on transport failure.
/// Returns [`DispatchError::RpcError`] if the server returns an RPC error.
pub fn result(handle: &DispatchHandle) -> Result<serde_json::Value, DispatchError> {
    let resp = rpc::send_to(
        &handle.transport,
        "compute.dispatch.result",
        &serde_json::json!({ "job_id": handle.job_id }),
    )?;

    if let Some((code, message)) = rpc::extract_rpc_error(&resp) {
        return Err(DispatchError::RpcError { code, message });
    }

    Ok(resp)
}

/// Query available compute capabilities from the compute primal.
///
/// # Errors
///
/// Returns [`DispatchError::NoComputePrimal`] if no compute primal transport can be resolved.
/// Returns [`DispatchError::Ipc`] on transport failure.
/// Returns [`DispatchError::RpcError`] if the server returns an RPC error.
pub fn capabilities() -> Result<serde_json::Value, DispatchError> {
    let transport = discover_compute_transport()?;

    let resp = rpc::send_to(
        &transport,
        "compute.dispatch.capabilities",
        &serde_json::json!({}),
    )?;

    if let Some((code, message)) = rpc::extract_rpc_error(&resp) {
        return Err(DispatchError::RpcError { code, message });
    }

    Ok(resp)
}

#[cfg(test)]
#[expect(clippy::expect_used, reason = "test code uses expect for clarity")]
mod tests {
    use super::*;
    use std::io::{BufRead, BufReader, Write};
    use std::net::TcpListener;

    #[test]
    fn no_compute_primal_returns_error() {
        // When no toadstool transport exists (typical in CI), submit must return an error.
        let result = submit("test_workload", &serde_json::json!({}));
        assert!(result.is_err());
    }

    /// TCP path: same [`rpc::send_to`] stack used after [`rpc::resolve_transport`] yields
    /// [`Transport::Tcp`] (e.g. `TOADSTOOL_ADDRESS` on Windows or when set explicitly).
    #[test]
    fn tcp_submit_round_trip_via_send_to() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind test listener");
        let addr = listener.local_addr().expect("local addr");

        let server = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("accept");
            let mut reader = BufReader::new(&stream);
            let mut line = String::new();
            reader.read_line(&mut line).expect("read request");
            let req: serde_json::Value = serde_json::from_str(line.trim()).expect("parse json");
            let id = req.get("id").cloned().unwrap_or(serde_json::Value::Null);
            let resp = serde_json::json!({
                "jsonrpc": "2.0",
                "result": { "job_id": "tcp-test-job" },
                "id": id,
            });
            let mut payload = serde_json::to_vec(&resp).expect("serialize");
            payload.push(b'\n');
            stream.write_all(&payload).expect("write");
            stream.flush().ok();
        });

        let transport = Transport::Tcp(addr);
        let outcome = submit_to_transport(&transport, "test_workload", &serde_json::json!({}));
        server.join().expect("server thread");

        let handle = outcome.expect("submit over TCP");
        assert_eq!(handle.job_id, "tcp-test-job");
        match handle.transport {
            Transport::Tcp(a) => assert_eq!(a, addr),
            #[cfg(unix)]
            Transport::Unix(_) => panic!("expected Tcp transport"),
        }
    }

    #[test]
    fn toadstool_dispatch_uses_standard_ecobin_env_keys() {
        assert_eq!(
            primal_names::socket_env_var(primal_names::TOADSTOOL),
            "TOADSTOOL_SOCKET"
        );
        assert_eq!(
            primal_names::address_env_var(primal_names::TOADSTOOL),
            "TOADSTOOL_ADDRESS"
        );
    }

    #[test]
    fn dispatch_error_display_no_compute_primal() {
        let e = DispatchError::NoComputePrimal;
        assert_eq!(format!("{e}"), "no compute primal discovered");
    }

    #[test]
    fn dispatch_error_display_missing_job_id() {
        let e = DispatchError::MissingJobId;
        assert!(format!("{e}").contains("job_id"));
    }

    #[test]
    fn dispatch_error_display_rpc_error() {
        let e = DispatchError::RpcError {
            code: -32600,
            message: "invalid request".to_string(),
        };
        let s = format!("{e}");
        assert!(s.contains("-32600"));
        assert!(s.contains("invalid request"));
    }

    #[test]
    fn dispatch_error_display_ipc() {
        let inner = IpcError::EmptyResponse {
            method: "test".to_string(),
        };
        let e = DispatchError::Ipc(inner);
        assert!(format!("{e}").contains("IPC error"));
    }

    #[test]
    fn dispatch_error_from_socket_not_found() {
        let ipc_err = IpcError::SocketNotFound {
            primal: crate::primal_names::TOADSTOOL.to_string(),
        };
        let e = DispatchError::from(ipc_err);
        assert!(matches!(e, DispatchError::NoComputePrimal));
    }

    #[test]
    fn dispatch_error_from_other_ipc_error() {
        let ipc_err = IpcError::EmptyResponse {
            method: "compute.dispatch.submit".to_string(),
        };
        let e = DispatchError::from(ipc_err);
        assert!(matches!(e, DispatchError::Ipc(_)));
    }

    #[test]
    fn tcp_submit_rpc_error_extracted() {
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
        let result = submit_to_transport(&transport, "bad_workload", &serde_json::json!({}));
        server.join().expect("join");

        match result {
            Err(DispatchError::RpcError { code, message }) => {
                assert_eq!(code, -32601);
                assert_eq!(message, "method not found");
            }
            other => panic!("expected RpcError, got {other:?}"),
        }
    }

    #[test]
    fn tcp_submit_missing_job_id() {
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
                "result": { "status": "ok" },
                "id": id,
            });
            let mut payload = serde_json::to_vec(&resp).expect("serialize");
            payload.push(b'\n');
            stream.write_all(&payload).expect("write");
            stream.flush().ok();
        });

        let transport = Transport::Tcp(addr);
        let result = submit_to_transport(&transport, "test", &serde_json::json!({}));
        server.join().expect("join");

        assert!(matches!(result, Err(DispatchError::MissingJobId)));
    }

    #[test]
    fn dispatch_error_is_error_trait() {
        let e: Box<dyn std::error::Error> = Box::new(DispatchError::NoComputePrimal);
        assert!(!e.to_string().is_empty());
    }
}
