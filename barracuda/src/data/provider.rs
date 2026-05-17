// SPDX-License-Identifier: AGPL-3.0-or-later
//! Data provider trait — abstracts data acquisition from transport.
//!
//! # Transport Tiers
//!
//! | Tier | Transport | TLS Stack | When |
//! |------|-----------|-----------|------|
//! | **Sovereign** | Songbird `network.http_request` | Pure Rust TLS 1.3 + `BearDog` crypto | Tower Atomic running |
//! | **NUCLEUS** | `capability.call` → `NestGate` | Content-addressed cache | Full NUCLEUS mesh |
//!
//! [`HttpTransport`] abstracts the HTTP layer so providers don't care which
//! tier is active. Discovery is automatic via Songbird socket.
//!
//! All providers are synchronous. airSpring operates on batch data (daily weather
//! arrays, annual yield tables), not streaming endpoints.

use crate::data::weather::DailyWeather;
use crate::primal_names;
use std::fmt;

/// Unified error for data acquisition.
#[derive(Debug)]
pub enum DataError {
    /// HTTP request failed (status code, body).
    Http(u16, String),
    /// Response parsing failed.
    Parse(String),
    /// Missing required configuration (e.g., API key).
    Config(String),
    /// I/O error (file, socket).
    Io(std::io::Error),
    /// Rate-limited — caller should back off after the given delay.
    RateLimited {
        /// Seconds to wait before retrying.
        retry_after_secs: u64,
    },
}

impl fmt::Display for DataError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Http(code, body) => write!(f, "HTTP {code}: {body}"),
            Self::Parse(msg) => write!(f, "parse error: {msg}"),
            Self::Config(msg) => write!(f, "config error: {msg}"),
            Self::Io(e) => write!(f, "I/O error: {e}"),
            Self::RateLimited { retry_after_secs } => {
                write!(f, "rate limited, retry after {retry_after_secs}s")
            }
        }
    }
}

impl std::error::Error for DataError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for DataError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

/// Weather data provider — fetches daily weather for a location and date range.
pub trait WeatherProvider {
    /// Provider name for provenance tracking.
    fn name(&self) -> &'static str;

    /// Fetch daily weather data for a location and date range.
    ///
    /// Returns one `DailyWeather` per day in the requested range.
    ///
    /// # Errors
    ///
    /// Returns `DataError` on HTTP failure, rate limiting, or parse errors.
    fn fetch_daily(
        &self,
        lat: f64,
        lon: f64,
        elevation_m: f64,
        station_id: &str,
        start_date: &str,
        end_date: &str,
    ) -> Result<Vec<DailyWeather>, DataError>;
}

/// Crop yield record from USDA NASS or similar source.
#[derive(Debug, Clone)]
pub struct YieldRecord {
    /// Crop commodity name (e.g. `"CORN"`).
    pub crop: String,
    /// Harvest year.
    pub year: u32,
    /// County name.
    pub county: String,
    /// State abbreviation or name.
    pub state: String,
    /// Yield value in the unit specified by [`unit`](Self::unit).
    pub yield_value: f64,
    /// Unit of yield (e.g. `"BU / ACRE"`).
    pub unit: String,
}

/// Yield data provider — fetches real crop yield data.
pub trait YieldProvider {
    /// Provider name for provenance tracking.
    fn name(&self) -> &'static str;

    /// Fetch county-level crop yields for a state and year range.
    ///
    /// # Errors
    ///
    /// Returns `DataError` on HTTP failure, rate limiting, or parse errors.
    fn fetch_yields(
        &self,
        commodity: &str,
        state: &str,
        year_start: u32,
        year_end: u32,
    ) -> Result<Vec<YieldRecord>, DataError>;
}

// ── HTTP Transport Abstraction ──────────────────────────────────────

/// Raw HTTP response from any transport tier.
#[derive(Debug)]
pub struct HttpResponse {
    /// HTTP status code.
    pub status: u16,
    /// Response body as UTF-8 string.
    pub body: String,
}

/// Transport-agnostic HTTP GET.
///
/// Implementations:
/// - [`SongbirdTransport`] — Tower Atomic sovereign TLS via Songbird + `BearDog`
pub trait HttpTransport {
    /// Tier name for provenance logging.
    fn tier(&self) -> &'static str;

    /// Perform a synchronous HTTP GET.
    ///
    /// # Errors
    ///
    /// Returns `DataError` on transport failure.
    fn get(&self, url: &str) -> Result<HttpResponse, DataError>;
}

/// Songbird transport via JSON-RPC over Unix socket (Tower Atomic).
///
/// When Tower Atomic is running, Songbird provides `network.http_request`
/// as a capability. TLS is handled by Songbird's pure-Rust TLS 1.3 stack
/// with crypto delegated to `BearDog` — zero C dependencies in the HTTPS path.
pub struct SongbirdTransport {
    socket_path: std::path::PathBuf,
}

impl SongbirdTransport {
    /// Discover Songbird socket using the standard primal discovery order.
    ///
    /// Search order:
    /// 1. `SONGBIRD_SOCKET` env var
    /// 2. biomeOS standard discovery (XDG runtime, capability scan)
    /// 3. `XDG_RUNTIME_DIR/primal/songbird/songbird.sock`
    #[must_use]
    pub fn discover() -> Option<Self> {
        let env_key = primal_names::socket_env_var(primal_names::SONGBIRD);
        if let Ok(path) = std::env::var(&env_key) {
            let p = std::path::PathBuf::from(&path);
            if p.exists() {
                return Some(Self { socket_path: p });
            }
        }

        if let Some(p) = crate::biomeos::discover_primal_socket(primal_names::SONGBIRD) {
            return Some(Self { socket_path: p });
        }

        let sock_name = primal_names::socket_filename(primal_names::SONGBIRD);
        if let Ok(xdg) = std::env::var("XDG_RUNTIME_DIR") {
            let p = std::path::PathBuf::from(xdg)
                .join("primal")
                .join(primal_names::SONGBIRD)
                .join(&sock_name);
            if p.exists() {
                return Some(Self { socket_path: p });
            }
        }

        None
    }
}

impl HttpTransport for SongbirdTransport {
    fn tier(&self) -> &'static str {
        "sovereign"
    }

    fn get(&self, url: &str) -> Result<HttpResponse, DataError> {
        use std::io::{Read, Write};
        use std::os::unix::net::UnixStream;

        let request = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "network.http_request",
            "params": {
                "method": "GET",
                "url": url,
            }
        });

        let mut stream = UnixStream::connect(&self.socket_path).map_err(|e| {
            DataError::Io(std::io::Error::new(
                e.kind(),
                format!("{} socket {}: {e}", primal_names::SONGBIRD, self.socket_path.display()),
            ))
        })?;

        let payload = serde_json::to_vec(&request)
            .map_err(|e| DataError::Parse(format!("serialize request: {e}")))?;
        stream.write_all(&payload).map_err(|e| {
            DataError::Io(std::io::Error::new(
                e.kind(),
                format!("write to {}: {e}", primal_names::SONGBIRD),
            ))
        })?;
        stream
            .shutdown(std::net::Shutdown::Write)
            .map_err(DataError::Io)?;

        let mut buf = String::new();
        stream.read_to_string(&mut buf).map_err(|e| {
            DataError::Io(std::io::Error::new(
                e.kind(),
                format!("read from {}: {e}", primal_names::SONGBIRD),
            ))
        })?;

        let resp: serde_json::Value = serde_json::from_str(&buf)
            .map_err(|e| DataError::Parse(format!("{} response: {e}", primal_names::SONGBIRD)))?;

        if let Some(err) = resp.get("error") {
            let msg = err["message"].as_str().unwrap_or("unknown");
            let code = err["code"].as_i64().unwrap_or(0);
            return Err(DataError::Http(
                u16::try_from(code).unwrap_or(0),
                msg.to_string(),
            ));
        }

        let result = &resp["result"];
        let status = result["status"]
            .as_u64()
            .and_then(|s| u16::try_from(s).ok())
            .unwrap_or(200);
        let body = result["body"].as_str().unwrap_or("").to_string();

        Ok(HttpResponse { status, body })
    }
}

/// Auto-discover the best available HTTP transport.
///
/// Returns `Some(SongbirdTransport)` if a Songbird socket is available,
/// `None` otherwise.
#[must_use]
pub fn discover_transport() -> Option<Box<dyn HttpTransport>> {
    SongbirdTransport::discover().map(|s| Box::new(s) as Box<dyn HttpTransport>)
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test assertions use unwrap for clarity")]
mod tests {
    use super::*;
    use crate::testutil::EnvGuard;
    use serial_test::serial;
    use std::error::Error;

    #[test]
    fn data_error_display_http() {
        let e = DataError::Http(404, "Not Found".into());
        let s = format!("{e}");
        assert!(s.contains("404"));
        assert!(s.contains("Not Found"));
    }

    #[test]
    fn data_error_display_parse() {
        let e = DataError::Parse("invalid JSON".into());
        let s = format!("{e}");
        assert!(s.contains("parse error"));
        assert!(s.contains("invalid JSON"));
    }

    #[test]
    fn data_error_display_config() {
        let e = DataError::Config("missing API key".into());
        let s = format!("{e}");
        assert!(s.contains("config error"));
        assert!(s.contains("missing API key"));
    }

    #[test]
    fn data_error_display_io() {
        let e = DataError::Io(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "file not found",
        ));
        let s = format!("{e}");
        assert!(s.contains("I/O error"));
        assert!(s.contains("file not found"));
    }

    #[test]
    fn data_error_display_rate_limited() {
        let e = DataError::RateLimited {
            retry_after_secs: 60,
        };
        let s = format!("{e}");
        assert!(s.contains("rate limited"));
        assert!(s.contains("60"));
    }

    #[test]
    fn data_error_source_io_returns_some() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "test");
        let e = DataError::Io(io_err);
        assert!(e.source().is_some());
    }

    #[test]
    fn data_error_source_http_returns_none() {
        let e = DataError::Http(500, "error".into());
        assert!(e.source().is_none());
    }

    #[test]
    fn data_error_source_parse_returns_none() {
        let e = DataError::Parse("error".into());
        assert!(e.source().is_none());
    }

    #[test]
    fn data_error_from_io_error() {
        let io_err = std::io::Error::new(std::io::ErrorKind::PermissionDenied, "denied");
        let e: DataError = io_err.into();
        match &e {
            DataError::Io(inner) => assert_eq!(inner.to_string(), "denied"),
            _ => panic!("expected Io variant"),
        }
    }

    #[test]
    fn yield_record_clone_and_debug() {
        let r = YieldRecord {
            crop: "corn".to_string(),
            year: 2020,
            county: "Ingham".to_string(),
            state: "Michigan".to_string(),
            yield_value: 175.5,
            unit: "BU / ACRE".to_string(),
        };
        let r2 = r.clone();
        assert_eq!(r.crop, r2.crop);
        assert_eq!(r.year, r2.year);
        let _ = format!("{r:?}");
    }

    // ── HttpTransport tests ─────────────────────────────────────────

    #[test]
    fn songbird_transport_tier_name() {
        let t = SongbirdTransport {
            socket_path: std::path::PathBuf::from("/nonexistent"),
        };
        assert_eq!(t.tier(), "sovereign");
    }

    #[test]
    #[serial]
    fn songbird_discover_returns_none_when_no_env() {
        let _g = EnvGuard::remove("SONGBIRD_SOCKET");
        assert!(SongbirdTransport::discover().is_none());
    }

    #[test]
    #[serial]
    fn songbird_discover_returns_none_when_socket_path_not_exists() {
        let _g1 = EnvGuard::remove("XDG_RUNTIME_DIR");
        let _g2 = EnvGuard::remove("FAMILY_ID");
        let _g3 = EnvGuard::set(
            "SONGBIRD_SOCKET",
            "/tmp/nonexistent_songbird_provider_test_xyz.sock",
        );
        assert!(SongbirdTransport::discover().is_none());
    }

    #[test]
    fn http_response_debug() {
        let r = HttpResponse {
            status: 201,
            body: "created".to_string(),
        };
        let s = format!("{r:?}");
        assert!(s.contains("201"));
        assert!(s.contains("created"));
    }

    #[test]
    fn songbird_get_fails_on_nonexistent_socket() {
        let t = SongbirdTransport {
            socket_path: std::path::PathBuf::from("/tmp/nonexistent_songbird_test.sock"),
        };
        let result = t.get("https://example.com");
        assert!(result.is_err());
        match result.unwrap_err() {
            DataError::Io(_) => {}
            other => panic!("expected Io error, got: {other}"),
        }
    }

    #[test]
    #[serial]
    fn discover_transport_returns_none_without_songbird() {
        let _g1 = EnvGuard::remove("SONGBIRD_SOCKET");
        let _g2 = EnvGuard::remove("FAMILY_ID");
        assert!(discover_transport().is_none());
    }

    #[test]
    fn http_response_fields() {
        let r = HttpResponse {
            status: 200,
            body: "hello".to_string(),
        };
        assert_eq!(r.status, 200);
        assert_eq!(r.body, "hello");
    }
}
