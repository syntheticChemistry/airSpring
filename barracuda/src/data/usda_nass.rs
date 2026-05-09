// SPDX-License-Identifier: AGPL-3.0-or-later
//! USDA NASS Quick Stats yield data provider.
//!
//! Mirrors `NestGate`'s `UsdaNassLiveProvider` API. Requires a free API key
//! from <https://quickstats.nass.usda.gov/api/>.
//!
//! # Transport Tiers
//!
//! Uses [`HttpTransport`] for HTTP requests. When Songbird is available
//! (Tower Atomic running), HTTPS routes through pure-Rust TLS 1.3.

use crate::data::provider::{
    DataError, HttpTransport, YieldProvider, YieldRecord, discover_transport,
};

const API_BASE: &str = "https://quickstats.nass.usda.gov/api/api_GET/";

/// USDA NASS Quick Stats API provider.
pub struct NassProvider {
    api_key: String,
    transport: Box<dyn HttpTransport>,
}

impl NassProvider {
    /// Create with auto-discovered transport.
    ///
    /// # Errors
    ///
    /// Returns `DataError::Config` if no HTTP transport is available
    /// (Songbird not running).
    pub fn try_new(api_key: String) -> Result<Self, DataError> {
        let transport = discover_transport().ok_or_else(|| {
            DataError::Config("no HTTP transport available — start Songbird primal".to_string())
        })?;
        Ok(Self { api_key, transport })
    }

    /// Create with auto-discovered transport.
    ///
    /// # Panics
    ///
    /// Panics if no HTTP transport is available. Prefer [`Self::try_new`]
    /// for fallible construction.
    #[must_use]
    #[allow(clippy::expect_used)]
    pub fn new(api_key: String) -> Self {
        Self::try_new(api_key).expect("no HTTP transport available")
    }

    /// Create with a specific transport.
    #[must_use]
    pub fn with_transport(api_key: String, transport: Box<dyn HttpTransport>) -> Self {
        Self { api_key, transport }
    }

    /// Create from `NASS_API_KEY` environment variable.
    ///
    /// # Errors
    ///
    /// Returns `DataError::Config` if the env var is not set.
    pub fn from_env() -> Result<Self, DataError> {
        let key = std::env::var("NASS_API_KEY")
            .map_err(|_| DataError::Config("NASS_API_KEY not set".into()))?;
        Self::try_new(key)
    }

    /// Create from a file containing the API key.
    ///
    /// # Errors
    ///
    /// Returns `DataError::Io` if the file cannot be read, or `DataError::Config`
    /// if the file is empty.
    pub fn from_file(path: &std::path::Path) -> Result<Self, DataError> {
        let key = std::fs::read_to_string(path)?;
        let key = key.trim().to_string();
        if key.is_empty() {
            return Err(DataError::Config("API key file is empty".into()));
        }
        Self::try_new(key)
    }
}

impl YieldProvider for NassProvider {
    fn name(&self) -> &'static str {
        "usda_nass_quickstats"
    }

    fn fetch_yields(
        &self,
        commodity: &str,
        state: &str,
        year_start: u32,
        year_end: u32,
    ) -> Result<Vec<YieldRecord>, DataError> {
        let url = format!(
            "{API_BASE}?key={key}\
             &source_desc=SURVEY\
             &sector_desc=CROPS\
             &commodity_desc={commodity}\
             &statisticcat_desc=YIELD\
             &state_alpha={state}\
             &agg_level_desc=COUNTY\
             &year__GE={year_start}\
             &year__LE={year_end}\
             &format=JSON",
            key = self.api_key
        );

        let resp = self.transport.get(&url)?;
        let json: serde_json::Value =
            serde_json::from_str(&resp.body).map_err(|e| DataError::Parse(format!("JSON: {e}")))?;

        let data = json
            .get("data")
            .and_then(|d| d.as_array())
            .ok_or_else(|| DataError::Parse("missing 'data' array".into()))?;

        let mut records = Vec::new();
        for row in data {
            let value_str = row["Value"]
                .as_str()
                .unwrap_or("")
                .replace(',', "")
                .trim()
                .to_string();

            if value_str.is_empty()
                || value_str == "(D)"
                || value_str == "(NA)"
                || value_str == "(S)"
                || value_str == "(Z)"
            {
                continue;
            }

            let Ok(yield_value) = value_str.parse::<f64>() else {
                continue;
            };

            let year = row["year"]
                .as_str()
                .and_then(|y| y.parse::<u32>().ok())
                .unwrap_or(0);
            if year == 0 {
                continue;
            }

            records.push(YieldRecord {
                crop: commodity.to_string(),
                year,
                county: row["county_name"].as_str().unwrap_or("").to_string(),
                state: state.to_string(),
                yield_value,
                unit: row["unit_desc"].as_str().unwrap_or("").to_string(),
            });
        }

        Ok(records)
    }
}

#[cfg(test)]
#[expect(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test assertions use unwrap/expect for clarity"
)]
#[expect(
    unsafe_code,
    reason = "Rust 2024: env::set_var/remove_var require unsafe in test cleanup"
)]
mod tests {
    use super::*;
    use crate::data::provider::HttpResponse;
    use std::cell::RefCell;
    use std::path::Path;
    use std::rc::Rc;

    /// Mock transport for testing — records URL and returns configurable response.
    struct MockTransport {
        body: String,
        url_captured: Rc<RefCell<Option<String>>>,
    }

    impl MockTransport {
        fn ok(body: &str, url_captured: Rc<RefCell<Option<String>>>) -> Self {
            Self {
                body: body.to_string(),
                url_captured,
            }
        }
    }

    impl crate::data::provider::HttpTransport for MockTransport {
        fn tier(&self) -> &'static str {
            "mock"
        }
        fn get(&self, url: &str) -> Result<HttpResponse, DataError> {
            *self.url_captured.borrow_mut() = Some(url.to_string());
            Ok(HttpResponse {
                status: 200,
                body: self.body.clone(),
            })
        }
    }

    #[test]
    fn try_new_fails_without_transport() {
        let result = NassProvider::try_new("test_key".into());
        assert!(
            result.is_err(),
            "try_new should fail without Songbird or standalone-http"
        );
    }

    #[test]
    fn provider_name() {
        let url_captured = Rc::new(RefCell::new(None));
        let mock = MockTransport::ok("{}", url_captured);
        let p = NassProvider::with_transport("test_key".into(), Box::new(mock));
        assert_eq!(p.name(), "usda_nass_quickstats");
    }

    #[test]
    fn from_env_missing() {
        unsafe {
            std::env::remove_var("NASS_API_KEY");
        }
        assert!(NassProvider::from_env().is_err());
    }

    #[test]
    fn from_file_nonexistent() {
        let path = Path::new("/nonexistent/path/that/does/not/exist/api_key.txt");
        let result = NassProvider::from_file(path);
        assert!(matches!(result, Err(DataError::Io(_))));
    }

    #[test]
    fn from_file_empty() {
        let dir = std::env::temp_dir();
        let path = dir.join("airspring_test_empty_key.txt");
        std::fs::write(&path, "").unwrap();
        let result = NassProvider::from_file(&path);
        std::fs::remove_file(&path).ok();
        assert!(matches!(result, Err(DataError::Config(_))));
    }

    #[test]
    fn from_file_valid_key_no_transport() {
        let dir = std::env::temp_dir();
        let path = dir.join("airspring_test_valid_key.txt");
        std::fs::write(&path, "my_api_key_123").unwrap();
        let result = NassProvider::from_file(&path);
        std::fs::remove_file(&path).ok();
        assert!(
            result.is_err(),
            "from_file should propagate transport-missing error"
        );
    }

    #[test]
    fn fetch_yields_url_construction() {
        let url_captured = Rc::new(RefCell::new(None));
        let mock = MockTransport::ok(r#"{"data":[]}"#, Rc::clone(&url_captured));
        let p = NassProvider::with_transport("my_key".into(), Box::new(mock));
        let _ = p.fetch_yields("CORN", "MI", 2019, 2021);
        let last_url = url_captured.borrow();
        let url = last_url.as_ref().expect("URL should have been captured");
        assert!(url.contains("quickstats.nass.usda.gov"));
        assert!(url.contains("key=my_key"));
        assert!(url.contains("commodity_desc=CORN"));
        assert!(url.contains("state_alpha=MI"));
        assert!(url.contains("year__GE=2019"));
        assert!(url.contains("year__LE=2021"));
    }

    #[test]
    fn fetch_yields_parses_valid_response() {
        let body = r#"{
            "data": [
                {"Value": "175.5", "year": "2020", "county_name": "Ingham", "unit_desc": "BU / ACRE"}
            ]
        }"#;
        let url_captured = Rc::new(RefCell::new(None));
        let mock = MockTransport::ok(body, url_captured);
        let p = NassProvider::with_transport("key".into(), Box::new(mock));
        let records = p.fetch_yields("CORN", "Michigan", 2020, 2020).unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].crop, "CORN");
        assert_eq!(records[0].year, 2020);
        assert_eq!(records[0].county, "Ingham");
        assert_eq!(records[0].state, "Michigan");
        assert!((records[0].yield_value - 175.5).abs() < 0.01);
        assert_eq!(records[0].unit, "BU / ACRE");
    }

    #[test]
    fn fetch_yields_filters_suppressed_values() {
        let body = r#"{
            "data": [
                {"Value": "(D)", "year": "2020", "county_name": "A", "unit_desc": "BU"},
                {"Value": "(NA)", "year": "2020", "county_name": "B", "unit_desc": "BU"},
                {"Value": "(S)", "year": "2020", "county_name": "C", "unit_desc": "BU"},
                {"Value": "(Z)", "year": "2020", "county_name": "D", "unit_desc": "BU"},
                {"Value": "150.0", "year": "2020", "county_name": "E", "unit_desc": "BU"}
            ]
        }"#;
        let url_captured = Rc::new(RefCell::new(None));
        let mock = MockTransport::ok(body, url_captured);
        let p = NassProvider::with_transport("key".into(), Box::new(mock));
        let records = p.fetch_yields("CORN", "MI", 2020, 2020).unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].county, "E");
        assert!((records[0].yield_value - 150.0).abs() < 0.01);
    }

    #[test]
    fn fetch_yields_handles_comma_in_value() {
        let body = r#"{
            "data": [
                {"Value": "1,234.5", "year": "2020", "county_name": "Test", "unit_desc": "BU"}
            ]
        }"#;
        let url_captured = Rc::new(RefCell::new(None));
        let mock = MockTransport::ok(body, url_captured);
        let p = NassProvider::with_transport("key".into(), Box::new(mock));
        let records = p.fetch_yields("CORN", "MI", 2020, 2020).unwrap();
        assert_eq!(records.len(), 1);
        assert!((records[0].yield_value - 1234.5).abs() < 0.01);
    }

    #[test]
    fn fetch_yields_missing_data_array() {
        let mock = MockTransport::ok(r#"{"count": 0}"#, Rc::new(RefCell::new(None)));
        let p = NassProvider::with_transport("key".into(), Box::new(mock));
        let result = p.fetch_yields("CORN", "MI", 2020, 2020);
        assert!(matches!(result, Err(DataError::Parse(msg)) if msg.contains("data")));
    }

    #[test]
    fn fetch_yields_invalid_json() {
        let mock = MockTransport::ok("not json", Rc::new(RefCell::new(None)));
        let p = NassProvider::with_transport("key".into(), Box::new(mock));
        let result = p.fetch_yields("CORN", "MI", 2020, 2020);
        assert!(matches!(result, Err(DataError::Parse(_))));
    }
}
