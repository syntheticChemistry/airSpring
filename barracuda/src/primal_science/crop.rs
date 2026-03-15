// SPDX-License-Identifier: AGPL-3.0-or-later
//! Crop and irrigation handlers for the airSpring primal.

use crate::eco::dual_kc;
use crate::eco::sensor_calibration;
use serde_json::Value;

use super::f64_p;

pub(super) fn dual_kc_handler(params: &Value) -> Value {
    let kcb = f64_p(params, "kcb").unwrap_or(1.10);
    let et0 = f64_p(params, "et0_mm").unwrap_or(5.0);
    serde_json::json!({
        "kc_max": dual_kc::kc_max(f64_p(params, "wind_2m").unwrap_or(2.0), f64_p(params, "rh_min").unwrap_or(45.0), f64_p(params, "crop_height_m").unwrap_or(0.5), kcb),
        "etc_mm": dual_kc::etc_dual(kcb, 1.0, 0.0, et0),
        "kcb": kcb,
        "method": "fao56_dual_kc",
    })
}

pub(super) fn sensor_cal(params: &Value) -> Value {
    let raw = f64_p(params, "raw_count").unwrap_or(5000.0);
    serde_json::json!({"volumetric_water_content": sensor_calibration::soilwatch10_vwc(raw), "raw_count": raw, "method": "soilwatch10_topp_polynomial"})
}

pub(super) fn gdd(params: &Value) -> Value {
    let tbase = f64_p(params, "tbase").unwrap_or(10.0);
    serde_json::json!({"gdd": crate::eco::crop::gdd_avg(f64_p(params, "tmax").unwrap_or(30.0), f64_p(params, "tmin").unwrap_or(15.0), tbase), "tbase": tbase, "method": "gdd_average"})
}
