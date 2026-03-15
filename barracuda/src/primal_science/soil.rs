// SPDX-License-Identifier: AGPL-3.0-or-later
//! Soil physics handlers for the airSpring primal.

use crate::eco::infiltration;
use crate::eco::richards;
use crate::eco::runoff;
use crate::eco::soil_moisture;
use serde_json::Value;

use super::{f64_p, u32_p};

pub(super) fn richards_1d(params: &Value) -> Value {
    let vg = richards::VanGenuchtenParams {
        theta_s: f64_p(params, "theta_s").unwrap_or(0.43),
        theta_r: f64_p(params, "theta_r").unwrap_or(0.078),
        alpha: f64_p(params, "alpha").unwrap_or(0.036),
        n_vg: f64_p(params, "n_vg").unwrap_or(1.56),
        ks: f64_p(params, "ks_cm_day").unwrap_or(24.96),
    };
    match richards::solve_richards_1d(
        &vg,
        f64_p(params, "depth_cm").unwrap_or(100.0),
        u32_p(params, "n_nodes").unwrap_or(20) as usize,
        f64_p(params, "h_initial_cm").unwrap_or(-100.0),
        f64_p(params, "h_top_cm").unwrap_or(-75.0),
        false,
        true,
        f64_p(params, "total_days").unwrap_or(10.0),
        f64_p(params, "dt_days").unwrap_or(0.1),
    ) {
        Ok(profiles) => {
            let Some(last) = profiles.last() else {
                return serde_json::json!({"error": "solver returned no profiles"});
            };
            let mean = last.theta.iter().sum::<f64>() / last.theta.len() as f64;
            serde_json::json!({"mean_theta": mean, "n_nodes": last.theta.len(), "n_timesteps": profiles.len(), "final_theta": last.theta, "method": "richards_1d_implicit_euler_picard"})
        }
        Err(e) => serde_json::json!({"error": format!("{e}")}),
    }
}

pub(super) fn scs_cn_runoff(params: &Value) -> Value {
    let precip = f64_p(params, "precipitation_mm").unwrap_or(50.0);
    let cn = f64_p(params, "curve_number").unwrap_or(75.0);
    let s = runoff::potential_retention(cn);
    serde_json::json!({"runoff_mm": runoff::scs_cn_runoff_standard(precip, cn), "potential_retention_mm": s, "initial_abstraction_mm": runoff::initial_abstraction(s, 0.2), "curve_number": cn, "method": "scs_cn"})
}

pub(super) fn green_ampt(params: &Value) -> Value {
    let ga = infiltration::GreenAmptParams {
        ks_cm_hr: f64_p(params, "ks_cm_hr").unwrap_or(1.0),
        psi_cm: f64_p(params, "psi_f_cm").unwrap_or(11.01),
        delta_theta: f64_p(params, "delta_theta").unwrap_or(0.312),
    };
    let t = f64_p(params, "time_hr").unwrap_or(1.0);
    let mut result = serde_json::json!({"cumulative_infiltration_cm": infiltration::cumulative_infiltration(&ga, t), "infiltration_rate_cm_hr": infiltration::infiltration_rate_at(&ga, t), "method": "green_ampt"});
    if let Some(ri) = f64_p(params, "rain_intensity_cm_hr") {
        result["ponding_time_hr"] = serde_json::json!(infiltration::ponding_time(&ga, ri));
    }
    result
}

pub(super) fn soil_moisture_topp(params: &Value) -> Value {
    let d = f64_p(params, "dielectric_constant").unwrap_or(15.0);
    let vwc = soil_moisture::topp_equation(d);
    serde_json::json!({"volumetric_water_content": vwc, "inverse_dielectric": soil_moisture::inverse_topp(vwc), "method": "topp_equation_1980"})
}

pub(super) fn pedotransfer(params: &Value) -> Value {
    let sr = soil_moisture::saxton_rawls(&soil_moisture::SaxtonRawlsInput {
        sand: f64_p(params, "sand_pct").unwrap_or(40.0) / 100.0,
        clay: f64_p(params, "clay_pct").unwrap_or(20.0) / 100.0,
        om_pct: f64_p(params, "organic_matter_pct").unwrap_or(2.5),
    });
    serde_json::json!({"field_capacity": sr.theta_fc, "wilting_point": sr.theta_wp, "saturation": sr.theta_s, "saturated_conductivity_mm_hr": sr.ksat_mm_hr, "lambda": sr.lambda, "method": "saxton_rawls_2006"})
}
