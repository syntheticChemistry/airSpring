//! Error types for airSpring `BarraCUDA`.
//!
//! Provides a unified error type that replaces ad-hoc `String` errors
//! throughout the crate, enabling proper error propagation with `?`.

use std::fmt;

/// Unified error type for airSpring operations.
#[derive(Debug)]
pub enum AirSpringError {
    /// I/O errors (file open, read, write).
    Io(std::io::Error),
    /// CSV parsing errors (malformed input, missing columns).
    CsvParse(String),
    /// JSON parsing errors (benchmark files).
    JsonParse(serde_json::Error),
    /// Invalid input (out of range, wrong dimensions).
    InvalidInput(String),
    /// Errors propagated from barracuda primitives.
    Barracuda(String),
}

impl fmt::Display for AirSpringError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "I/O error: {e}"),
            Self::CsvParse(msg) => write!(f, "CSV parse error: {msg}"),
            Self::JsonParse(e) => write!(f, "JSON parse error: {e}"),
            Self::InvalidInput(msg) => write!(f, "Invalid input: {msg}"),
            Self::Barracuda(msg) => write!(f, "barracuda error: {msg}"),
        }
    }
}

impl std::error::Error for AirSpringError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            Self::JsonParse(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for AirSpringError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

impl From<serde_json::Error> for AirSpringError {
    fn from(e: serde_json::Error) -> Self {
        Self::JsonParse(e)
    }
}

/// Convenience alias used throughout the crate.
pub type Result<T> = std::result::Result<T, AirSpringError>;
