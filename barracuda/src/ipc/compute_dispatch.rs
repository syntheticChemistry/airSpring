// SPDX-License-Identifier: AGPL-3.0-or-later
//! Typed client for toadStool `compute.dispatch` capability.
//!
//! Routes GPU workloads through toadStool instead of direct `wgpu` access.
//! Discovery is capability-based: `compute.dispatch.submit` is resolved
//! at runtime through biomeOS socket scanning.

use std::path::PathBuf;

use crate::biomeos;
use crate::rpc::{self, IpcError};

/// Handle to a dispatched compute job.
#[derive(Debug)]
pub struct DispatchHandle {
    /// Job identifier returned by toadStool.
    pub job_id: String,
    /// Socket path of the compute primal.
    pub socket: PathBuf,
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
        Self::Ipc(e)
    }
}

/// Discover the compute primal socket via capability-based discovery.
fn discover_compute_socket() -> Result<PathBuf, DispatchError> {
    if let Ok(path) = std::env::var("AIRSPRING_COMPUTE_PRIMAL") {
        let p = PathBuf::from(path);
        if p.exists() {
            return Ok(p);
        }
    }

    biomeos::discover_primal_socket(crate::primal_names::TOADSTOOL)
        .ok_or(DispatchError::NoComputePrimal)
}

/// Submit a GPU workload to the compute primal.
///
/// Returns a [`DispatchHandle`] for polling results.
///
/// # Errors
///
/// Returns [`DispatchError::NoComputePrimal`] if no compute primal socket is discovered.
/// Returns [`DispatchError::Ipc`] on transport failure.
/// Returns [`DispatchError::MissingJobId`] if the server response lacks `job_id`.
/// Returns [`DispatchError::RpcError`] if the server returns an RPC error.
pub fn submit(
    workload_type: &str,
    params: &serde_json::Value,
) -> Result<DispatchHandle, DispatchError> {
    let socket = discover_compute_socket()?;

    let result = rpc::send(
        &socket,
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

    Ok(DispatchHandle { job_id, socket })
}

/// Poll for the result of a dispatched compute job.
///
/// # Errors
///
/// Returns [`DispatchError::Ipc`] on transport failure.
/// Returns [`DispatchError::RpcError`] if the server returns an RPC error.
pub fn result(handle: &DispatchHandle) -> Result<serde_json::Value, DispatchError> {
    let resp = rpc::send(
        &handle.socket,
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
/// Returns [`DispatchError::NoComputePrimal`] if no compute primal socket is discovered.
/// Returns [`DispatchError::Ipc`] on transport failure.
/// Returns [`DispatchError::RpcError`] if the server returns an RPC error.
pub fn capabilities() -> Result<serde_json::Value, DispatchError> {
    let socket = discover_compute_socket()?;

    let resp = rpc::send(
        &socket,
        "compute.dispatch.capabilities",
        &serde_json::json!({}),
    )?;

    if let Some((code, message)) = rpc::extract_rpc_error(&resp) {
        return Err(DispatchError::RpcError { code, message });
    }

    Ok(resp)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn no_compute_primal_returns_error() {
        // When no toadstool socket exists (typical in CI), submit must return an error.
        let result = submit("test_workload", &serde_json::json!({}));
        assert!(result.is_err());
    }
}
