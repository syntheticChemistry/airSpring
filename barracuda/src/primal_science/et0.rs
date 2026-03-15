// SPDX-License-Identifier: AGPL-3.0-or-later
//! Evapotranspiration (ET₀) handlers for the airSpring primal.

use crate::eco::evapotranspiration as et;
use crate::eco::simple_et0;
use serde_json::Value;

use super::{DEFAULT_DOY, DEFAULT_ELEVATION_M, DEFAULT_LATITUDE_DEG, f64_p, u32_p};

pub(super) fn et0_fao56(params: &Value) -> Value {
    let input = et::DailyEt0Input {
        tmax: f64_p(params, "tmax").unwrap_or(30.0),
        tmin: f64_p(params, "tmin").unwrap_or(15.0),
        tmean: f64_p(params, "tmean"),
        solar_radiation: f64_p(params, "solar_radiation").unwrap_or(20.0),
        wind_speed_2m: f64_p(params, "wind_speed_2m").unwrap_or(2.0),
        actual_vapour_pressure: f64_p(params, "actual_vapour_pressure").unwrap_or(1.5),
        day_of_year: u32_p(params, "day_of_year").unwrap_or(DEFAULT_DOY),
        latitude_deg: f64_p(params, "latitude_deg").unwrap_or(DEFAULT_LATITUDE_DEG),
        elevation_m: f64_p(params, "elevation_m").unwrap_or(DEFAULT_ELEVATION_M),
    };
    let result = et::daily_et0(&input);
    serde_json::json!({"et0_mm": result.et0, "rn_mj": result.rn, "method": "fao56_penman_monteith"})
}

pub(super) fn et0_hargreaves(params: &Value) -> Value {
    let tmin = f64_p(params, "tmin").unwrap_or(15.0);
    let tmax = f64_p(params, "tmax").unwrap_or(30.0);
    let lat_rad = f64_p(params, "latitude_deg")
        .unwrap_or(DEFAULT_LATITUDE_DEG)
        .to_radians();
    let doy = u32_p(params, "day_of_year").unwrap_or(DEFAULT_DOY);
    let ra_mm = crate::eco::solar::extraterrestrial_radiation(lat_rad, doy) / 2.45;
    let et0 = et::hargreaves_et0(tmin, tmax, ra_mm);
    serde_json::json!({"et0_mm": et0, "ra_mm_day": ra_mm, "method": "hargreaves"})
}

pub(super) fn et0_priestley_taylor(params: &Value) -> Value {
    let et0 = et::priestley_taylor_et0(
        f64_p(params, "rn").unwrap_or(10.0),
        f64_p(params, "g").unwrap_or(0.0),
        f64_p(params, "tmean").unwrap_or(22.5),
        f64_p(params, "elevation_m").unwrap_or(DEFAULT_ELEVATION_M),
    );
    serde_json::json!({"et0_mm": et0, "method": "priestley_taylor"})
}

pub(super) fn et0_makkink(params: &Value) -> Value {
    let et0 = simple_et0::makkink_et0(
        f64_p(params, "tmean").unwrap_or(22.5),
        f64_p(params, "solar_radiation").unwrap_or(20.0),
        f64_p(params, "elevation_m").unwrap_or(DEFAULT_ELEVATION_M),
    );
    serde_json::json!({"et0_mm": et0, "method": "makkink"})
}

pub(super) fn et0_turc(params: &Value) -> Value {
    let et0 = simple_et0::turc_et0(
        f64_p(params, "tmean").unwrap_or(22.5),
        f64_p(params, "solar_radiation").unwrap_or(20.0),
        f64_p(params, "rh_pct").unwrap_or(60.0),
    );
    serde_json::json!({"et0_mm": et0, "method": "turc"})
}

pub(super) fn et0_hamon(params: &Value) -> Value {
    let lat_rad = f64_p(params, "latitude_deg")
        .unwrap_or(DEFAULT_LATITUDE_DEG)
        .to_radians();
    let et0 = simple_et0::hamon_pet_from_location(
        f64_p(params, "tmean").unwrap_or(22.5),
        lat_rad,
        u32_p(params, "day_of_year").unwrap_or(DEFAULT_DOY),
    );
    serde_json::json!({"pet_mm": et0, "method": "hamon"})
}

pub(super) fn et0_blaney_criddle(params: &Value) -> Value {
    let lat_rad = f64_p(params, "latitude_deg")
        .unwrap_or(DEFAULT_LATITUDE_DEG)
        .to_radians();
    let et0 = simple_et0::blaney_criddle_from_location(
        f64_p(params, "tmean").unwrap_or(22.5),
        lat_rad,
        u32_p(params, "day_of_year").unwrap_or(DEFAULT_DOY),
    );
    serde_json::json!({"et0_mm": et0, "method": "blaney_criddle"})
}
