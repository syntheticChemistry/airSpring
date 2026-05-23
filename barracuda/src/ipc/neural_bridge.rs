// SPDX-License-Identifier: AGPL-3.0-or-later
//! `NeuralBridge` observatory client — biomeOS v3.67+ adaptive routing.
//!
//! Wraps the Neural API observatory surface:
//! - `neural_api.routing_weights` — adaptive routing weight snapshot
//! - `neural_api.route_explain` — explain routing decision for a method
//! - `neural_api.utilization` — real-time utilization metrics
//! - `neural_api.weight_health` — convergence diagnostics (v3.70+)
//!
//! Also provides `capability_call_instrumented` which records a
//! [`BridgeOutcome`] capturing latency and success for each round-trip.
//! Callers feed outcomes into metrics for adaptive routing analysis.
//!
//! Non-fatal when biomeOS is unavailable — all methods return typed errors.

use crate::rpc::{self, IpcError, Transport};

/// Outcome of a bridge round-trip for adaptive routing feedback.
#[derive(Debug, Clone)]
pub struct BridgeOutcome {
    /// Capability domain dispatched.
    pub capability: String,
    /// Operation within the domain.
    pub operation: String,
    /// Wall-clock latency of the round-trip (ms).
    pub latency_ms: u64,
    /// Whether the call succeeded.
    pub success: bool,
    /// Unix epoch milliseconds when dispatch occurred.
    pub timestamp_epoch_ms: u64,
}

/// Errors from `NeuralBridge` operations.
#[derive(Debug)]
pub enum BridgeError {
    /// No biomeOS Neural API transport discovered.
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
}

impl std::fmt::Display for BridgeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoPrimal => write!(f, "no biomeOS Neural API discovered"),
            Self::Ipc(e) => write!(f, "IPC error: {e}"),
            Self::RpcError { code, message } => write!(f, "RPC error {code}: {message}"),
        }
    }
}

impl std::error::Error for BridgeError {}

impl From<IpcError> for BridgeError {
    fn from(e: IpcError) -> Self {
        Self::Ipc(e)
    }
}

fn resolve_neural_api() -> Result<Transport, BridgeError> {
    rpc::resolve_transport("biomeos").map_err(|_| BridgeError::NoPrimal)
}

fn call_neural_api(
    transport: &Transport,
    method: &str,
    params: &serde_json::Value,
) -> Result<serde_json::Value, BridgeError> {
    let resp = rpc::send_to(transport, method, params).map_err(BridgeError::Ipc)?;
    if let Some(err) = resp.get("error") {
        let code = err.get("code").and_then(serde_json::Value::as_i64).unwrap_or(-1);
        let message = err
            .get("message")
            .and_then(|m| m.as_str())
            .unwrap_or("unknown")
            .to_string();
        return Err(BridgeError::RpcError { code, message });
    }
    Ok(resp.get("result").cloned().unwrap_or(serde_json::Value::Null))
}

fn epoch_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| {
            u64::try_from(d.as_millis()).unwrap_or(u64::MAX)
        })
}

/// Invoke `capability.call` through biomeOS and record the round-trip.
///
/// Returns both the result and a [`BridgeOutcome`] for adaptive routing
/// feedback. The outcome is always produced regardless of success.
///
/// # Errors
///
/// Returns `BridgeError` on transport failure or RPC error.
pub fn capability_call_instrumented(
    capability: &str,
    operation: &str,
    args: &serde_json::Value,
) -> (Result<serde_json::Value, BridgeError>, BridgeOutcome) {
    let start = std::time::Instant::now();
    let result = (|| {
        let transport = resolve_neural_api()?;
        call_neural_api(
            &transport,
            "capability.call",
            &serde_json::json!({
                "capability": capability,
                "operation": operation,
                "args": args,
            }),
        )
    })();
    let latency_ms = u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX);
    let success = result.is_ok();
    let outcome = BridgeOutcome {
        capability: capability.to_string(),
        operation: operation.to_string(),
        latency_ms,
        success,
        timestamp_epoch_ms: epoch_ms(),
    };
    (result, outcome)
}

/// Query adaptive routing weights from biomeOS (v3.67+).
///
/// Returns the full weight table snapshot. Non-fatal when biomeOS is
/// unavailable or pre-v3.67.
///
/// # Errors
///
/// Returns `BridgeError` if biomeOS is unreachable or pre-v3.67.
pub fn routing_weights() -> Result<serde_json::Value, BridgeError> {
    let transport = resolve_neural_api()?;
    call_neural_api(
        &transport,
        "neural_api.routing_weights",
        &serde_json::Value::Null,
    )
}

/// Explain the routing decision for a specific method (v3.67+).
///
/// # Errors
///
/// Returns `BridgeError` if biomeOS is unreachable or pre-v3.67.
pub fn route_explain(method: &str) -> Result<serde_json::Value, BridgeError> {
    let transport = resolve_neural_api()?;
    call_neural_api(
        &transport,
        "neural_api.route_explain",
        &serde_json::json!({ "method": method }),
    )
}

/// Query real-time utilization metrics from biomeOS (v3.67+).
///
/// # Errors
///
/// Returns `BridgeError` if biomeOS is unreachable or pre-v3.67.
pub fn utilization() -> Result<serde_json::Value, BridgeError> {
    let transport = resolve_neural_api()?;
    call_neural_api(
        &transport,
        "neural_api.utilization",
        &serde_json::Value::Null,
    )
}

/// Query routing weight health diagnostics from biomeOS (v3.70+).
///
/// Returns convergence diagnostics: healthy flag, persistence status,
/// convergence stats, and open circuit breaker details.
///
/// # Errors
///
/// Returns `BridgeError` if biomeOS is unreachable or pre-v3.70.
pub fn weight_health() -> Result<serde_json::Value, BridgeError> {
    let transport = resolve_neural_api()?;
    call_neural_api(
        &transport,
        "neural_api.weight_health",
        &serde_json::Value::Null,
    )
}

/// Query composition patterns from biomeOS (v3.67+).
///
/// # Errors
///
/// Returns `BridgeError` if biomeOS is unreachable or pre-v3.67.
pub fn composition_patterns() -> Result<serde_json::Value, BridgeError> {
    let transport = resolve_neural_api()?;
    call_neural_api(
        &transport,
        "neural_api.composition_patterns",
        &serde_json::Value::Null,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bridge_outcome_captures_fields() {
        let outcome = BridgeOutcome {
            capability: "science".to_string(),
            operation: "et0_fao56".to_string(),
            latency_ms: 42,
            success: true,
            timestamp_epoch_ms: 1_700_000_000_000,
        };
        assert_eq!(outcome.capability, "science");
        assert_eq!(outcome.operation, "et0_fao56");
        assert!(outcome.success);
    }

    #[test]
    fn bridge_error_display() {
        let err = BridgeError::NoPrimal;
        assert!(err.to_string().contains("no biomeOS"));
        let err = BridgeError::RpcError {
            code: -32601,
            message: "Method not found".to_string(),
        };
        assert!(err.to_string().contains("-32601"));
    }

    #[test]
    fn no_primal_returns_gracefully() {
        let result = routing_weights();
        assert!(
            result.is_err(),
            "should fail gracefully without biomeOS running"
        );
    }

    #[test]
    fn instrumented_call_produces_outcome() {
        let (result, outcome) = capability_call_instrumented(
            "science",
            "et0_fao56",
            &serde_json::json!({}),
        );
        assert!(result.is_err(), "no biomeOS in test env");
        assert!(!outcome.success);
        assert_eq!(outcome.capability, "science");
        assert_eq!(outcome.operation, "et0_fao56");
        assert!(outcome.timestamp_epoch_ms > 0);
    }
}
