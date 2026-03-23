// SPDX-License-Identifier: AGPL-3.0-or-later
//! IPC resilience primitives — circuit breaker and retry with exponential backoff.
//!
//! Absorbed from healthSpring V32 `resilient_capability_call()` pattern.
//! Provides automatic retry for transient IPC failures and circuit-breaking
//! to avoid cascading failures when a primal is down.

use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use crate::rpc::{self, IpcError};

/// Circuit breaker state for a single IPC target.
///
/// When consecutive failures exceed `MAX_RETRIES`, the circuit opens for
/// `CIRCUIT_OPEN_DURATION`. During this period, calls fail immediately
/// without attempting the socket connection, reducing load on a degraded primal.
pub struct CircuitBreaker {
    open: AtomicBool,
    opened_at: std::sync::Mutex<Option<Instant>>,
    failure_count: AtomicU64,
}

const MAX_RETRIES: u32 = 2;
const RETRY_BASE_MS: u64 = 50;
const CIRCUIT_OPEN_DURATION: Duration = Duration::from_secs(5);

impl CircuitBreaker {
    /// Create a new circuit breaker in the closed (healthy) state.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            open: AtomicBool::new(false),
            opened_at: std::sync::Mutex::new(None),
            failure_count: AtomicU64::new(0),
        }
    }

    /// Whether the circuit is currently open (rejecting calls).
    #[must_use]
    pub fn is_open(&self) -> bool {
        if !self.open.load(Ordering::Relaxed) {
            return false;
        }
        let guard = self
            .opened_at
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(opened) = *guard
            && opened.elapsed() >= CIRCUIT_OPEN_DURATION
        {
            drop(guard);
            self.reset();
            return false;
        }
        true
    }

    /// Record a successful call — resets failure count and closes circuit.
    pub fn record_success(&self) {
        self.failure_count.store(0, Ordering::Relaxed);
        if self.open.load(Ordering::Relaxed) {
            self.reset();
        }
    }

    /// Record a failed call — increments failure count and may open circuit.
    pub fn record_failure(&self) {
        let count = self.failure_count.fetch_add(1, Ordering::Relaxed) + 1;
        if count > u64::from(MAX_RETRIES) {
            self.open.store(true, Ordering::Relaxed);
            let mut guard = self
                .opened_at
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            *guard = Some(Instant::now());
        }
    }

    fn reset(&self) {
        self.open.store(false, Ordering::Relaxed);
        self.failure_count.store(0, Ordering::Relaxed);
        let mut guard = self
            .opened_at
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        *guard = None;
    }
}

impl Default for CircuitBreaker {
    fn default() -> Self {
        Self::new()
    }
}

/// Send a JSON-RPC request with retry and exponential backoff.
///
/// Retries up to `MAX_RETRIES` times with backoff (`50ms → 100ms → fail`).
/// Only retries on recoverable errors (connection, timeout, write/read failures).
///
/// # Errors
///
/// Returns the last `IpcError` if all retries are exhausted.
pub fn resilient_send(
    socket_path: &Path,
    method: &str,
    params: &serde_json::Value,
    breaker: &CircuitBreaker,
) -> Result<serde_json::Value, IpcError> {
    if breaker.is_open() {
        return Err(IpcError::ConnectionFailed {
            socket: socket_path.to_path_buf(),
            source: std::io::Error::new(
                std::io::ErrorKind::ConnectionRefused,
                "circuit breaker open",
            ),
        });
    }

    let mut last_err = None;
    for attempt in 0..=MAX_RETRIES {
        match rpc::send(socket_path, method, params) {
            Ok(resp) => {
                breaker.record_success();
                return Ok(resp);
            }
            Err(e) if e.is_recoverable() && attempt < MAX_RETRIES => {
                let backoff = Duration::from_millis(RETRY_BASE_MS << attempt);
                std::thread::sleep(backoff);
                last_err = Some(e);
            }
            Err(e) => {
                breaker.record_failure();
                return Err(e);
            }
        }
    }

    breaker.record_failure();
    Err(last_err.unwrap_or_else(|| IpcError::EmptyResponse {
        method: method.to_string(),
    }))
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test code uses unwrap for clarity")]
mod tests {
    use super::*;

    #[test]
    fn circuit_breaker_starts_closed() {
        let cb = CircuitBreaker::new();
        assert!(!cb.is_open());
    }

    #[test]
    fn circuit_breaker_opens_after_failures() {
        let cb = CircuitBreaker::new();
        for _ in 0..=MAX_RETRIES {
            cb.record_failure();
        }
        assert!(cb.is_open());
    }

    #[test]
    fn circuit_breaker_resets_on_success() {
        let cb = CircuitBreaker::new();
        for _ in 0..=MAX_RETRIES {
            cb.record_failure();
        }
        assert!(cb.is_open());
        cb.record_success();
        assert!(!cb.is_open());
    }

    #[test]
    fn default_creates_closed_breaker() {
        let cb = CircuitBreaker::default();
        assert!(!cb.is_open());
    }

    #[test]
    fn resilient_send_fails_when_circuit_open() {
        let cb = CircuitBreaker::new();
        for _ in 0..=MAX_RETRIES {
            cb.record_failure();
        }
        let result = resilient_send(
            Path::new("/nonexistent.sock"),
            "health",
            &serde_json::json!({}),
            &cb,
        );
        assert!(result.is_err());
    }

    #[test]
    fn is_recoverable_variants() {
        let conn = IpcError::ConnectionFailed {
            socket: "/tmp/x.sock".into(),
            source: std::io::Error::new(std::io::ErrorKind::ConnectionRefused, "refused"),
        };
        assert!(conn.is_recoverable());

        let timeout = IpcError::Timeout {
            method: "health".into(),
            elapsed: Duration::from_secs(5),
        };
        assert!(timeout.is_recoverable());

        let write = IpcError::WriteFailed {
            socket: "/tmp/x.sock".into(),
            source: std::io::Error::new(std::io::ErrorKind::BrokenPipe, "broken"),
        };
        assert!(write.is_recoverable());

        let read = IpcError::ReadFailed {
            socket: "/tmp/x.sock".into(),
            source: std::io::Error::new(std::io::ErrorKind::UnexpectedEof, "eof"),
        };
        assert!(read.is_recoverable());

        let rpc = IpcError::RpcError {
            code: -32601,
            message: "not found".into(),
        };
        assert!(!rpc.is_recoverable());

        let deser = IpcError::DeserializationFailed {
            method: "health".into(),
            source: serde_json::from_str::<serde_json::Value>("{bad").unwrap_err(),
        };
        assert!(!deser.is_recoverable());

        let not_found = IpcError::SocketNotFound {
            primal: "nestgate".into(),
        };
        assert!(!not_found.is_recoverable());

        let empty = IpcError::EmptyResponse {
            method: "health".into(),
        };
        assert!(!empty.is_recoverable());
    }
}
