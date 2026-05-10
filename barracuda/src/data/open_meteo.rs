// SPDX-License-Identifier: AGPL-3.0-or-later
//! Open-Meteo weather data provider.
//!
//! Mirrors `NestGate`'s `OpenMeteoLiveProvider` API — same endpoints, same
//! parameter semantics, byte-identical data.
//!
//! # Transport Tiers
//!
//! Uses [`HttpTransport`] for HTTP requests. When Songbird is available
//! (Tower Atomic running), HTTPS routes through pure-Rust TLS 1.3 via
//! `BearDog` crypto delegation.
//!
//! # Cross-Spring Provenance
//!
//! Open-Meteo Archive provides ERA5 reanalysis at 10km resolution globally,
//! including all FAO-56 variables (tmax, tmin, humidity, wind, radiation).

use crate::data::provider::{DataError, HttpTransport, WeatherProvider, discover_transport};
use crate::data::weather::DailyWeather;

const ARCHIVE_URL: &str = "https://archive-api.open-meteo.com/v1/archive";

const DAILY_VARS: &str = "temperature_2m_max,temperature_2m_min,temperature_2m_mean,\
relative_humidity_2m_max,relative_humidity_2m_min,\
windspeed_10m_mean,shortwave_radiation_sum,precipitation_sum,\
et0_fao_evapotranspiration";

/// FAO-56 Eq. 47: convert 10m wind speed to 2m.
fn wind_10m_to_2m(u10: f64) -> f64 {
    u10 * 4.87 / 67.8_f64.mul_add(10.0, -5.42).ln()
}

/// Open-Meteo Archive API provider.
///
/// Free, no authentication. Rate-limited to ~10k requests/day for free tier.
/// Transport auto-discovers Songbird (sovereign).
pub struct OpenMeteoProvider {
    max_retries: u32,
    transport: Box<dyn HttpTransport>,
}

impl OpenMeteoProvider {
    /// Create with auto-discovered transport (Songbird preferred, ureq fallback).
    ///
    /// # Errors
    ///
    /// Returns `DataError::Config` if no HTTP transport is available
    /// (Songbird not running).
    pub fn try_new() -> Result<Self, DataError> {
        let transport = discover_transport().ok_or_else(|| {
            DataError::Config("no HTTP transport available — start Songbird primal".to_string())
        })?;
        Ok(Self {
            max_retries: 5,
            transport,
        })
    }

    /// Create with auto-discovered transport (Songbird preferred, ureq fallback).
    ///
    /// # Panics
    ///
    /// Panics if no HTTP transport is available. Prefer [`Self::try_new`]
    /// for fallible construction.
    #[must_use]
    #[expect(
        clippy::expect_used,
        reason = "panicking ctor retained for API compat; prefer try_new()"
    )]
    pub fn new() -> Self {
        Self::try_new().expect("no HTTP transport available")
    }

    /// Create with a specific transport (for testing or explicit tier selection).
    #[must_use]
    pub fn with_transport(transport: Box<dyn HttpTransport>) -> Self {
        Self {
            max_retries: 5,
            transport,
        }
    }

    fn parse_daily_response(
        json: &serde_json::Value,
        station_id: &str,
        lat: f64,
        lon: f64,
        elevation_m: f64,
    ) -> Result<Vec<DailyWeather>, DataError> {
        let daily = json
            .get("daily")
            .ok_or_else(|| DataError::Parse("missing 'daily' in response".into()))?;

        let dates = daily["time"]
            .as_array()
            .ok_or_else(|| DataError::Parse("missing 'time' array".into()))?;
        let tmax = daily["temperature_2m_max"]
            .as_array()
            .ok_or_else(|| DataError::Parse("missing tmax".into()))?;
        let tmin = daily["temperature_2m_min"]
            .as_array()
            .ok_or_else(|| DataError::Parse("missing tmin".into()))?;
        let tmean = daily["temperature_2m_mean"]
            .as_array()
            .ok_or_else(|| DataError::Parse("missing tmean".into()))?;
        let rh_max = daily["relative_humidity_2m_max"]
            .as_array()
            .ok_or_else(|| DataError::Parse("missing rh_max".into()))?;
        let rh_min = daily["relative_humidity_2m_min"]
            .as_array()
            .ok_or_else(|| DataError::Parse("missing rh_min".into()))?;
        let wind = daily["windspeed_10m_mean"]
            .as_array()
            .ok_or_else(|| DataError::Parse("missing wind".into()))?;
        let solar = daily["shortwave_radiation_sum"]
            .as_array()
            .ok_or_else(|| DataError::Parse("missing solar".into()))?;
        let precip = daily["precipitation_sum"]
            .as_array()
            .ok_or_else(|| DataError::Parse("missing precip".into()))?;
        let et0_ref = daily
            .get("et0_fao_evapotranspiration")
            .and_then(|v| v.as_array());

        let extract_f64 = |arr: &[serde_json::Value], idx: usize| -> f64 {
            arr[idx].as_f64().unwrap_or(f64::NAN)
        };

        let records = dates
            .iter()
            .enumerate()
            .map(|(i, date_val)| {
                let u10 = extract_f64(wind, i);
                DailyWeather {
                    date: date_val.as_str().unwrap_or("").to_string(),
                    tmax_c: extract_f64(tmax, i),
                    tmin_c: extract_f64(tmin, i),
                    tmean_c: extract_f64(tmean, i),
                    rh_max_pct: extract_f64(rh_max, i),
                    rh_min_pct: extract_f64(rh_min, i),
                    wind_2m_m_s: wind_10m_to_2m(u10),
                    solar_rad_mj_m2: extract_f64(solar, i),
                    precip_mm: extract_f64(precip, i),
                    et0_reference_mm: et0_ref.map(|arr| extract_f64(arr, i)),
                    station_id: station_id.to_string(),
                    lat,
                    lon,
                    elevation_m,
                }
            })
            .collect();

        Ok(records)
    }
}

impl Default for OpenMeteoProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl WeatherProvider for OpenMeteoProvider {
    fn name(&self) -> &'static str {
        "open_meteo_archive"
    }

    fn fetch_daily(
        &self,
        lat: f64,
        lon: f64,
        elevation_m: f64,
        station_id: &str,
        start_date: &str,
        end_date: &str,
    ) -> Result<Vec<DailyWeather>, DataError> {
        let url = format!(
            "{ARCHIVE_URL}?latitude={lat}&longitude={lon}\
             &start_date={start_date}&end_date={end_date}\
             &daily={DAILY_VARS}\
             &timezone=America%2FDetroit\
             &windspeed_unit=ms&precipitation_unit=mm"
        );

        for attempt in 0..self.max_retries {
            match self.transport.get(&url) {
                Ok(resp) => {
                    let json: serde_json::Value = serde_json::from_str(&resp.body)
                        .map_err(|e| DataError::Parse(format!("JSON: {e}")))?;
                    return Self::parse_daily_response(&json, station_id, lat, lon, elevation_m);
                }
                Err(DataError::RateLimited { retry_after_secs }) => {
                    if attempt + 1 < self.max_retries {
                        let wait = retry_after_secs.max(u64::from(15 * (attempt + 1)));
                        std::thread::sleep(std::time::Duration::from_secs(wait));
                        continue;
                    }
                    return Err(DataError::RateLimited { retry_after_secs });
                }
                Err(e) => return Err(e),
            }
        }
        Err(DataError::RateLimited {
            retry_after_secs: 60,
        })
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test assertions use unwrap for clarity")]
mod tests {
    use super::*;
    use crate::data::provider::{HttpResponse, HttpTransport};

    /// Mock transport for testing.
    struct MockTransport {
        body: String,
    }

    impl MockTransport {
        fn ok(body: &str) -> Self {
            Self {
                body: body.to_string(),
            }
        }
    }

    impl HttpTransport for MockTransport {
        fn tier(&self) -> &'static str {
            "mock"
        }
        fn get(&self, _url: &str) -> Result<HttpResponse, DataError> {
            Ok(HttpResponse {
                status: 200,
                body: self.body.clone(),
            })
        }
    }

    #[test]
    fn try_new_fails_without_transport() {
        let result = OpenMeteoProvider::try_new();
        assert!(
            result.is_err(),
            "try_new should fail without Songbird or standalone-http"
        );
    }

    #[test]
    fn with_transport_creates_provider() {
        let mock = MockTransport::ok("{}");
        let p = OpenMeteoProvider::with_transport(Box::new(mock));
        assert_eq!(p.name(), "open_meteo_archive");
    }

    #[test]
    fn wind_conversion_fao56_eq47() {
        let u10 = 3.0;
        let u2 = wind_10m_to_2m(u10);
        assert!(
            (2.0..2.5).contains(&u2),
            "u10=3.0 m/s → u2≈2.10 m/s per FAO-56 Eq 47, got u2={u2}"
        );
    }

    #[test]
    fn provider_name() {
        let mock = MockTransport::ok("{}");
        let p = OpenMeteoProvider::with_transport(Box::new(mock));
        assert_eq!(p.name(), "open_meteo_archive");
    }

    #[test]
    fn parse_daily_response_valid_json() {
        let json = serde_json::json!({
            "daily": {
                "time": ["2020-06-01", "2020-06-02"],
                "temperature_2m_max": [28.5, 29.0],
                "temperature_2m_min": [15.0, 16.0],
                "temperature_2m_mean": [21.75, 22.5],
                "relative_humidity_2m_max": [85.0, 82.0],
                "relative_humidity_2m_min": [45.0, 48.0],
                "windspeed_10m_mean": [3.0, 2.5],
                "shortwave_radiation_sum": [25.0, 26.0],
                "precipitation_sum": [0.0, 5.2],
                "et0_fao_evapotranspiration": [4.5, 4.8]
            }
        });
        let result =
            OpenMeteoProvider::parse_daily_response(&json, "test_station", 42.7, -84.5, 256.0);
        let records = result.unwrap();
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].date, "2020-06-01");
        assert!((records[0].tmax_c - 28.5).abs() < 0.01);
        assert!((records[0].wind_2m_m_s - wind_10m_to_2m(3.0)).abs() < 0.01);
        assert_eq!(records[0].station_id, "test_station");
        assert!((records[0].lat - 42.7).abs() < 0.01);
        assert_eq!(records[0].et0_reference_mm, Some(4.5));
    }

    #[test]
    fn default_fails_gracefully() {
        let result = OpenMeteoProvider::try_new();
        assert!(result.is_err(), "Default/new should fail without Songbird");
    }

    #[test]
    fn parse_daily_response_missing_daily() {
        let json = serde_json::json!({"count": 0});
        let result = OpenMeteoProvider::parse_daily_response(&json, "station", 42.0, -84.0, 200.0);
        assert!(matches!(result, Err(DataError::Parse(msg)) if msg.contains("daily")));
    }

    #[test]
    fn parse_daily_response_missing_time() {
        let json = serde_json::json!({
            "daily": {
                "temperature_2m_max": [28.0],
                "temperature_2m_min": [15.0],
                "temperature_2m_mean": [21.5],
                "relative_humidity_2m_max": [80.0],
                "relative_humidity_2m_min": [50.0],
                "windspeed_10m_mean": [3.0],
                "shortwave_radiation_sum": [25.0],
                "precipitation_sum": [0.0]
            }
        });
        let result = OpenMeteoProvider::parse_daily_response(&json, "station", 42.0, -84.0, 200.0);
        assert!(matches!(result, Err(DataError::Parse(msg)) if msg.contains("time")));
    }

    #[test]
    fn parse_daily_response_missing_tmax() {
        let json = serde_json::json!({
            "daily": {
                "time": ["2020-06-01"],
                "temperature_2m_min": [15.0],
                "temperature_2m_mean": [21.5],
                "relative_humidity_2m_max": [80.0],
                "relative_humidity_2m_min": [50.0],
                "windspeed_10m_mean": [3.0],
                "shortwave_radiation_sum": [25.0],
                "precipitation_sum": [0.0]
            }
        });
        let result = OpenMeteoProvider::parse_daily_response(&json, "station", 42.0, -84.0, 200.0);
        assert!(matches!(result, Err(DataError::Parse(msg)) if msg.contains("tmax")));
    }

    #[test]
    fn parse_daily_response_without_et0_optional() {
        let json = serde_json::json!({
            "daily": {
                "time": ["2020-06-01"],
                "temperature_2m_max": [28.0],
                "temperature_2m_min": [15.0],
                "temperature_2m_mean": [21.5],
                "relative_humidity_2m_max": [80.0],
                "relative_humidity_2m_min": [50.0],
                "windspeed_10m_mean": [3.0],
                "shortwave_radiation_sum": [25.0],
                "precipitation_sum": [0.0]
            }
        });
        let result = OpenMeteoProvider::parse_daily_response(&json, "station", 42.0, -84.0, 200.0);
        let records = result.unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].et0_reference_mm, None);
    }

    #[test]
    fn fetch_daily_url_and_parsing_via_mock() {
        let json = serde_json::json!({
            "daily": {
                "time": ["2020-07-15"],
                "temperature_2m_max": [30.0],
                "temperature_2m_min": [18.0],
                "temperature_2m_mean": [24.0],
                "relative_humidity_2m_max": [90.0],
                "relative_humidity_2m_min": [55.0],
                "windspeed_10m_mean": [2.0],
                "shortwave_radiation_sum": [22.0],
                "precipitation_sum": [0.0],
                "et0_fao_evapotranspiration": [5.0]
            }
        });
        let mock = MockTransport::ok(&json.to_string());
        let p = OpenMeteoProvider::with_transport(Box::new(mock));
        let records = p
            .fetch_daily(43.0, -85.0, 250.0, "fetch_test", "2020-07-15", "2020-07-15")
            .unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].date, "2020-07-15");
        assert!((records[0].tmax_c - 30.0).abs() < 0.01);
        assert_eq!(records[0].et0_reference_mm, Some(5.0));
    }

    #[test]
    fn wind_10m_to_2m_zero_input() {
        let u2 = wind_10m_to_2m(0.0);
        assert!(u2.abs() < 0.01, "zero wind should give ~0");
    }
}
