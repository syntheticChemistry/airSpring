// SPDX-License-Identifier: AGPL-3.0-or-later
//! Error types for airSpring `BarraCuda`.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_io_error_display() {
        let err = AirSpringError::Io(std::io::Error::new(std::io::ErrorKind::NotFound, "gone"));
        assert!(format!("{err}").contains("I/O error"));
    }

    #[test]
    fn test_csv_parse_display() {
        let err = AirSpringError::CsvParse("missing column".into());
        assert!(format!("{err}").contains("CSV parse error"));
        assert!(format!("{err}").contains("missing column"));
    }

    #[test]
    fn test_json_parse_display() {
        let bad: std::result::Result<serde_json::Value, _> = serde_json::from_str("{bad");
        let err = AirSpringError::JsonParse(bad.unwrap_err());
        assert!(format!("{err}").contains("JSON parse error"));
    }

    #[test]
    fn test_invalid_input_display() {
        let err = AirSpringError::InvalidInput("negative value".into());
        assert!(format!("{err}").contains("Invalid input"));
    }

    #[test]
    fn test_barracuda_display() {
        let err = AirSpringError::Barracuda("GPU fail".into());
        assert!(format!("{err}").contains("barracuda error"));
    }

    #[test]
    fn test_io_error_source() {
        let inner = std::io::Error::new(std::io::ErrorKind::PermissionDenied, "nope");
        let err = AirSpringError::Io(inner);
        assert!(std::error::Error::source(&err).is_some());
    }

    #[test]
    fn test_json_error_source() {
        let bad: std::result::Result<serde_json::Value, _> = serde_json::from_str("{bad");
        let err = AirSpringError::JsonParse(bad.unwrap_err());
        assert!(std::error::Error::source(&err).is_some());
    }

    #[test]
    fn test_csv_error_no_source() {
        let err = AirSpringError::CsvParse("col missing".into());
        assert!(std::error::Error::source(&err).is_none());
    }

    #[test]
    fn test_from_io_error() {
        let io_err = std::io::Error::other("disk");
        let err: AirSpringError = io_err.into();
        assert!(matches!(err, AirSpringError::Io(_)));
    }

    #[test]
    fn test_from_json_error() {
        let json_err = serde_json::from_str::<serde_json::Value>("{x").unwrap_err();
        let err: AirSpringError = json_err.into();
        assert!(matches!(err, AirSpringError::JsonParse(_)));
    }

    #[test]
    fn test_debug_format() {
        let err = AirSpringError::Barracuda("test".into());
        let debug = format!("{err:?}");
        assert!(debug.contains("Barracuda"));
    }
}
