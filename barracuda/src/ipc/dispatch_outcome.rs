// SPDX-License-Identifier: AGPL-3.0-or-later
//! Generic dispatch outcome for JSON-RPC method routing.
//!
//! Follows the wetSpring V126 / groundSpring V112 pattern: every spring
//! exposes `DispatchOutcome<T>` as a library type so toadStool dispatch,
//! validation binaries, and metalForge can all classify results uniformly.

/// Outcome of dispatching a JSON-RPC method.
///
/// The generic `T` is typically `serde_json::Value` for the airSpring niche
/// server, but validation binaries can use domain-specific result types.
#[derive(Debug)]
pub enum DispatchOutcome<T> {
    /// Successful dispatch — contains the result payload.
    Ok(T),
    /// The requested method is not implemented by this niche.
    MethodNotFound(String),
    /// The method exists but the parameters are invalid.
    InvalidParams {
        /// JSON-RPC method name.
        method: String,
        /// Human-readable reason the params were rejected.
        reason: String,
    },
    /// The method exists, params are valid, but execution failed.
    InternalError {
        /// JSON-RPC method name.
        method: String,
        /// Error source description.
        source: String,
    },
}

impl<T> DispatchOutcome<T> {
    /// Returns `true` if the outcome is `Ok`.
    #[must_use]
    pub const fn is_ok(&self) -> bool {
        matches!(self, Self::Ok(_))
    }

    /// Returns `true` if the outcome is a recoverable error
    /// (anything except `InternalError`).
    #[must_use]
    pub const fn is_recoverable(&self) -> bool {
        !matches!(self, Self::InternalError { .. })
    }

    /// Extract the success value, or `None` for errors.
    #[must_use]
    pub fn ok(self) -> Option<T> {
        match self {
            Self::Ok(v) => Some(v),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ok_variant() {
        let outcome: DispatchOutcome<i32> = DispatchOutcome::Ok(42);
        assert!(outcome.is_ok());
        assert!(outcome.is_recoverable());
    }

    #[test]
    fn method_not_found_is_recoverable() {
        let outcome: DispatchOutcome<i32> = DispatchOutcome::MethodNotFound("foo".into());
        assert!(!outcome.is_ok());
        assert!(outcome.is_recoverable());
    }

    #[test]
    fn internal_error_not_recoverable() {
        let outcome: DispatchOutcome<i32> = DispatchOutcome::InternalError {
            method: "bar".into(),
            source: "boom".into(),
        };
        assert!(!outcome.is_ok());
        assert!(!outcome.is_recoverable());
    }

    #[test]
    fn ok_extracts_value() {
        let outcome: DispatchOutcome<String> = DispatchOutcome::Ok("hello".into());
        assert_eq!(outcome.ok(), Some("hello".into()));
    }

    #[test]
    fn error_ok_returns_none() {
        let outcome: DispatchOutcome<String> = DispatchOutcome::MethodNotFound("x".into());
        assert_eq!(outcome.ok(), None);
    }
}
