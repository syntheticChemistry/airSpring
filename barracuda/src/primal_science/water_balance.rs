// SPDX-License-Identifier: AGPL-3.0-or-later
//! Water balance and yield response handlers for the airSpring primal.

use crate::eco::evapotranspiration as et;
use serde_json::Value;

use super::{DEFAULT_DOY, DEFAULT_ELEVATION_M, DEFAULT_LATITUDE_DEG, f64_p, u32_p};

pub(super) fn water_balance(params: &Value) -> Value {
    let et0 = f64_p(params, "et0_mm").unwrap_or(5.0);
    let kc = f64_p(params, "kc").unwrap_or(1.0);
    let precip = f64_p(params, "precipitation_mm").unwrap_or(3.0);
    let irrigation = f64_p(params, "irrigation_mm").unwrap_or(0.0);
    let sw = f64_p(params, "soil_water_mm").unwrap_or(100.0);
    let fc = f64_p(params, "field_capacity_mm").unwrap_or(200.0);
    let wp = f64_p(params, "wilting_point_mm").unwrap_or(50.0);

    let etc = et0 * kc;
    let input = sw + precip + irrigation;
    let new_sw = (input - etc).clamp(wp, fc);
    let deep = (input - etc - fc).max(0.0);
    serde_json::json!({"etc_mm": etc, "soil_water_mm": new_sw, "deep_percolation_mm": deep, "deficit_mm": fc - new_sw, "method": "fao56_water_balance"})
}

pub(super) fn yield_response(params: &Value) -> Value {
    let ky = f64_p(params, "ky").unwrap_or(1.25);
    let eta = f64_p(params, "eta_over_etm").unwrap_or(0.8);
    let max_y = f64_p(params, "max_yield_t_ha").unwrap_or(12.0);
    let ratio = (1.0 - ky * (1.0 - eta)).max(0.0);
    serde_json::json!({"yield_t_ha": max_y * ratio, "yield_ratio": ratio, "ky": ky, "method": "stewart_1977"})
}

pub(super) fn full_pipeline(params: &Value) -> Value {
    let tmax = f64_p(params, "tmax").unwrap_or(32.0);
    let tmin = f64_p(params, "tmin").unwrap_or(18.0);
    let input = et::DailyEt0Input {
        tmax,
        tmin,
        tmean: None,
        solar_radiation: f64_p(params, "solar_radiation").unwrap_or(22.5),
        wind_speed_2m: f64_p(params, "wind_speed_2m").unwrap_or(2.0),
        actual_vapour_pressure: f64_p(params, "actual_vapour_pressure").unwrap_or(1.5),
        day_of_year: u32_p(params, "day_of_year").unwrap_or(DEFAULT_DOY),
        latitude_deg: f64_p(params, "latitude_deg").unwrap_or(DEFAULT_LATITUDE_DEG),
        elevation_m: f64_p(params, "elevation_m").unwrap_or(DEFAULT_ELEVATION_M),
    };
    let et0_result = et::daily_et0(&input);
    let et0 = et0_result.et0.max(0.0);

    let kc = f64_p(params, "kc").unwrap_or(1.0);
    let precip = f64_p(params, "precipitation_mm").unwrap_or(3.0);
    let irr = f64_p(params, "irrigation_mm").unwrap_or(0.0);
    let sw = f64_p(params, "soil_water_mm").unwrap_or(100.0);
    let fc = f64_p(params, "field_capacity_mm").unwrap_or(200.0);
    let wp = f64_p(params, "wilting_point_mm").unwrap_or(50.0);

    let etc = et0 * kc;
    let wb_in = sw + precip + irr;
    let new_sw = (wb_in - etc).clamp(wp, fc);
    let deep = (wb_in - etc - fc).max(0.0);
    let deficit = fc - new_sw;

    let ky = f64_p(params, "ky").unwrap_or(1.25);
    let max_y = f64_p(params, "max_yield_t_ha").unwrap_or(12.0);
    let eta = if et0 > 0.0 {
        (etc - deficit.min(etc)) / etc
    } else {
        1.0
    };
    let yr = (1.0 - ky * (1.0 - eta)).max(0.0);

    serde_json::json!({
        "pipeline": "ecology.full_pipeline",
        "stages": {
            "et0": {"et0_mm": et0, "rn_mj": et0_result.rn, "method": "fao56_penman_monteith"},
            "water_balance": {"etc_mm": etc, "soil_water_mm": new_sw, "deep_percolation_mm": deep, "deficit_mm": deficit},
            "yield": {"yield_t_ha": max_y * yr, "yield_ratio": yr, "eta_over_etm": eta, "ky": ky},
        },
    })
}
