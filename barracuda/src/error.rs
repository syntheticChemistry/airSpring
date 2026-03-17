// SPDX-License-Identifier: AGPL-3.0-or-later
//! Error types for airSpring `BarraCuda`.
//!
//! Provides a unified error type that replaces ad-hoc `String` errors
//! throughout the crate, enabling proper error propagation with `?`.

/// Unified error type for airSpring operations.
#[derive(Debug, thiserror::Error)]
pub enum AirSpringError {
    /// I/O errors (file open, read, write).
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    /// CSV parsing errors (malformed input, missing columns).
    #[error("CSV parse error: {0}")]
    CsvParse(String),
    /// JSON parsing errors (benchmark files).
    #[error("JSON parse error: {0}")]
    JsonParse(#[from] serde_json::Error),
    /// Benchmark JSON structure errors (missing keys, wrong types).
    #[error("Benchmark parse error: {0}")]
    BenchmarkParse(String),
    /// Invalid input (out of range, wrong dimensions).
    #[error("Invalid input: {0}")]
    InvalidInput(String),
    /// Errors propagated from barracuda primitives (preserves source chain).
    #[error("barracuda error: {0}")]
    Barracuda(#[from] barracuda::error::BarracudaError),
    /// NPU errors (discovery, DMA, inference).
    #[error("NPU error: {0}")]
    Npu(String),
    /// IPC errors (socket connect, timeout, protocol).
    #[error("IPC error: {0}")]
    Ipc(#[from] crate::rpc::IpcError),
}

impl AirSpringError {
    /// Wrap a string as a barracuda error (for cases where the original
    /// error type is not available, e.g. formatted NPU driver messages).
    pub fn barracuda_msg(msg: impl Into<String>) -> Self {
        Self::Barracuda(barracuda::error::BarracudaError::Internal(msg.into()))
    }
}

/// Convenience alias used throughout the crate.
pub type Result<T> = std::result::Result<T, AirSpringError>;

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test code uses unwrap for clarity")]
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
        let err = AirSpringError::from(barracuda::error::BarracudaError::Gpu("GPU fail".into()));
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
    fn test_npu_display() {
        let err = AirSpringError::Npu("no device found".into());
        assert!(format!("{err}").contains("NPU error"));
        assert!(format!("{err}").contains("no device found"));
    }

    #[test]
    fn test_npu_no_source() {
        let err = AirSpringError::Npu("discovery failed".into());
        assert!(std::error::Error::source(&err).is_none());
    }

    #[test]
    fn test_ipc_display() {
        let err = AirSpringError::Ipc(crate::rpc::IpcError::SocketNotFound {
            primal: "nestgate".into(),
        });
        assert!(format!("{err}").contains("IPC error"));
        assert!(format!("{err}").contains("nestgate"));
    }

    #[test]
    fn test_ipc_source() {
        let err = AirSpringError::Ipc(crate::rpc::IpcError::SocketNotFound {
            primal: "toadstool".into(),
        });
        assert!(std::error::Error::source(&err).is_some());
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
    fn test_barracuda_source() {
        let err = AirSpringError::from(barracuda::error::BarracudaError::Gpu("x".into()));
        assert!(std::error::Error::source(&err).is_some());
    }

    #[test]
    fn test_barracuda_msg_helper() {
        let err = AirSpringError::barracuda_msg("custom context");
        assert!(format!("{err}").contains("custom context"));
    }

    #[test]
    fn test_debug_format() {
        let err = AirSpringError::from(barracuda::error::BarracudaError::Internal("test".into()));
        let debug = format!("{err:?}");
        assert!(debug.contains("Barracuda"));
    }

    #[test]
    fn test_from_barracuda_error() {
        let barr_err = barracuda::error::BarracudaError::Device("gone".into());
        let err: AirSpringError = barr_err.into();
        assert!(matches!(err, AirSpringError::Barracuda(_)));
    }
}
