// SPDX-License-Identifier: AGPL-3.0-or-later

//! IPC error types for JSON-RPC 2.0 communication.

use std::net::SocketAddr;
use std::path::PathBuf;
use std::time::Duration;

/// IPC transport errors (biomeOS standard).
#[derive(Debug, thiserror::Error)]
pub enum IpcError {
    /// Connection to the socket failed.
    #[error("connection failed to {}: {source}", socket.display())]
    ConnectionFailed {
        /// Target socket path.
        socket: PathBuf,
        /// Underlying I/O error.
        source: std::io::Error,
    },
    /// Connection to TCP address failed.
    #[error("connection failed to {addr}: {source}")]
    ConnectionFailedTcp {
        /// Target TCP address.
        addr: SocketAddr,
        /// Underlying I/O error.
        source: std::io::Error,
    },
    /// Write to socket failed (distinct from connect for recovery).
    #[error("write failed to {}: {source}", socket.display())]
    WriteFailed {
        /// Target socket path.
        socket: PathBuf,
        /// Underlying I/O error.
        source: std::io::Error,
    },
    /// Write to TCP address failed.
    #[error("write failed to {addr}: {source}")]
    WriteFailedTcp {
        /// Target TCP address.
        addr: SocketAddr,
        /// Underlying I/O error.
        source: std::io::Error,
    },
    /// Read from socket failed (distinct from connect for recovery).
    #[error("read failed from {}: {source}", socket.display())]
    ReadFailed {
        /// Target socket path.
        socket: PathBuf,
        /// Underlying I/O error.
        source: std::io::Error,
    },
    /// Read from TCP address failed.
    #[error("read failed from {addr}: {source}")]
    ReadFailedTcp {
        /// Target TCP address.
        addr: SocketAddr,
        /// Underlying I/O error.
        source: std::io::Error,
    },
    /// Request timed out.
    #[error("timeout calling {method} after {elapsed:?}")]
    Timeout {
        /// JSON-RPC method that timed out.
        method: String,
        /// Duration before timeout.
        elapsed: Duration,
    },
    /// Server returned a JSON-RPC error.
    #[error("RPC error {code}: {message}")]
    RpcError {
        /// JSON-RPC error code.
        code: i32,
        /// Human-readable error message.
        message: String,
    },
    /// Response deserialization failed.
    #[error("deserialization failed for {method}: {source}")]
    DeserializationFailed {
        /// JSON-RPC method whose response failed parsing.
        method: String,
        /// Underlying JSON error.
        source: serde_json::Error,
    },
    /// Socket path could not be resolved.
    #[error("socket path not found for {primal}")]
    SocketNotFound {
        /// Primal name that was being discovered.
        primal: String,
    },
    /// Response contained no result or error (empty/malformed).
    #[error("empty response from {method}")]
    EmptyResponse {
        /// JSON-RPC method that returned nothing.
        method: String,
    },
    /// Unix sockets not available on this platform (use TCP via {PRIMAL}_ADDRESS).
    #[error("Unix sockets not available on this platform; use TCP via {{PRIMAL}}_ADDRESS")]
    UnixNotAvailable,
}

impl IpcError {
    /// Whether this error is transient and the operation may succeed on retry.
    ///
    /// Returns `true` for `ConnectionFailed`, `ConnectionFailedTcp`, `Timeout`,
    /// `WriteFailed`, `WriteFailedTcp`, `ReadFailed`, and `ReadFailedTcp` —
    /// these indicate network or scheduling issues, not protocol violations.
    #[must_use]
    pub const fn is_recoverable(&self) -> bool {
        matches!(
            self,
            Self::ConnectionFailed { .. }
                | Self::ConnectionFailedTcp { .. }
                | Self::Timeout { .. }
                | Self::WriteFailed { .. }
                | Self::WriteFailedTcp { .. }
                | Self::ReadFailed { .. }
                | Self::ReadFailedTcp { .. }
        )
    }
}
