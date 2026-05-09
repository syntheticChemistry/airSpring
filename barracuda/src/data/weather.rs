// SPDX-License-Identifier: AGPL-3.0-or-later
//! Shared weather data types used across data providers.

/// A single day of weather observations for FAO-56 ET₀ computation.
#[derive(Debug, Clone)]
pub struct DailyWeather {
    /// Date string (`YYYY-MM-DD`).
    pub date: String,
    /// Maximum air temperature (degrees C).
    pub tmax_c: f64,
    /// Minimum air temperature (degrees C).
    pub tmin_c: f64,
    /// Mean air temperature (degrees C).
    pub tmean_c: f64,
    /// Maximum relative humidity (percent).
    pub rh_max_pct: f64,
    /// Minimum relative humidity (percent).
    pub rh_min_pct: f64,
    /// Wind speed at 2 m height (m/s).
    pub wind_2m_m_s: f64,
    /// Incoming solar radiation (MJ/m^2/day).
    pub solar_rad_mj_m2: f64,
    /// Precipitation (mm/day).
    pub precip_mm: f64,
    /// Open-Meteo's own FAO-56 ET₀ (for cross-check).
    pub et0_reference_mm: Option<f64>,
    /// Station identifier.
    pub station_id: String,
    /// Latitude in decimal degrees (WGS-84).
    pub lat: f64,
    /// Longitude in decimal degrees (WGS-84).
    pub lon: f64,
    /// Elevation above sea level (metres).
    pub elevation_m: f64,
}

/// Station metadata for a weather observation point.
#[derive(Debug, Clone)]
pub struct Station {
    /// Unique identifier (e.g. `"east_lansing"`).
    pub id: &'static str,
    /// Human-readable station name.
    pub name: &'static str,
    /// Latitude in decimal degrees (WGS-84).
    pub lat: f64,
    /// Longitude in decimal degrees (WGS-84).
    pub lon: f64,
    /// Elevation above sea level in metres.
    pub elevation_m: f64,
}

/// Michigan core stations — 6 locations matching Open-Meteo download scripts.
pub static MICHIGAN_CORE_STATIONS: &[Station] = &[
    Station {
        id: "east_lansing",
        name: "East Lansing (MSU)",
        lat: 42.727,
        lon: -84.474,
        elevation_m: 256.0,
    },
    Station {
        id: "grand_junction",
        name: "Grand Junction",
        lat: 42.375,
        lon: -86.060,
        elevation_m: 197.0,
    },
    Station {
        id: "sparta",
        name: "Sparta",
        lat: 43.160,
        lon: -85.710,
        elevation_m: 262.0,
    },
    Station {
        id: "hart",
        name: "Hart",
        lat: 43.698,
        lon: -86.364,
        elevation_m: 244.0,
    },
    Station {
        id: "west_olive",
        name: "West Olive",
        lat: 42.917,
        lon: -86.167,
        elevation_m: 192.0,
    },
    Station {
        id: "manchester",
        name: "Manchester",
        lat: 42.153,
        lon: -84.037,
        elevation_m: 290.0,
    },
];
