// SPDX-License-Identifier: AGPL-3.0-or-later
//! Drought index, autocorrelation, gamma CDF, and monthly ET handlers.

use crate::eco::anderson;
use crate::eco::drought_index;
use crate::eco::thornthwaite;
use crate::gpu::autocorrelation;
use serde_json::Value;

use super::{DEFAULT_LATITUDE_DEG, f64_p};

pub(super) fn anderson_coupling(params: &Value) -> Value {
    let r = anderson::coupling_chain(
        f64_p(params, "theta").unwrap_or(0.25),
        f64_p(params, "theta_r").unwrap_or(0.078),
        f64_p(params, "theta_s").unwrap_or(0.43),
    );
    serde_json::json!({"effective_saturation": r.se, "pore_connectivity": r.connectivity, "coordination_number": r.coordination, "effective_dimension": r.d_eff, "disorder_parameter": r.disorder, "regime": format!("{:?}", r.regime), "method": "anderson_soil_coupling"})
}

pub(super) fn thornthwaite_handler(params: &Value) -> Value {
    let temps: Vec<f64> = params
        .get("monthly_temps_c")
        .and_then(|v| v.as_array())
        .map(|a| a.iter().filter_map(serde_json::Value::as_f64).collect())
        .unwrap_or_default();
    if temps.len() != 12 {
        return serde_json::json!({"error": "monthly_temps_c must have exactly 12 values"});
    }
    let mut arr = [0.0; 12];
    arr.copy_from_slice(&temps);
    let m = thornthwaite::thornthwaite_monthly_et0(
        &arr,
        f64_p(params, "latitude_deg").unwrap_or(DEFAULT_LATITUDE_DEG),
    );
    serde_json::json!({"monthly_et0_mm": m.to_vec(), "annual_et0_mm": m.iter().sum::<f64>(), "method": "thornthwaite_1948"})
}

pub(super) fn spi_drought(params: &Value) -> Value {
    let monthly_precip: Vec<f64> = params
        .get("monthly_precip_mm")
        .and_then(|v| v.as_array())
        .map(|a| a.iter().filter_map(serde_json::Value::as_f64).collect())
        .unwrap_or_default();
    if monthly_precip.len() < 3 {
        return serde_json::json!({"error": "monthly_precip_mm must have ≥3 values"});
    }
    let scale = params
        .get("scale")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(3) as usize;
    let spi = drought_index::compute_spi(&monthly_precip, scale);
    let n_valid = spi.iter().filter(|v| v.is_finite()).count();
    let classifications: Vec<&str> = spi
        .iter()
        .map(|&v| {
            if v.is_nan() {
                "insufficient_data"
            } else {
                drought_index::DroughtClass::from_spi(v).label()
            }
        })
        .collect();
    serde_json::json!({
        "spi": spi.iter().map(|v| if v.is_nan() { serde_json::Value::Null } else { serde_json::json!(v) }).collect::<Vec<_>>(),
        "scale_months": scale,
        "n_months": monthly_precip.len(),
        "n_valid": n_valid,
        "classifications": classifications,
        "method": "mckee_1993_spi",
        "upstream": "barracuda::special::gamma::regularized_gamma_p"
    })
}

pub(super) fn autocorrelation_handler(params: &Value) -> Value {
    let data: Vec<f64> = params
        .get("data")
        .and_then(|v| v.as_array())
        .map(|a| a.iter().filter_map(serde_json::Value::as_f64).collect())
        .unwrap_or_default();
    if data.is_empty() {
        return serde_json::json!({"error": "data must be non-empty"});
    }
    let max_lag = params
        .get("max_lag")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(20) as usize;
    let max_lag = max_lag.min(data.len());
    let acf = autocorrelation::autocorrelation_cpu(&data, max_lag);
    let nacf = autocorrelation::normalised_acf_cpu(&data, max_lag);
    serde_json::json!({
        "acf": acf,
        "normalised_acf": nacf,
        "max_lag": max_lag,
        "n_samples": data.len(),
        "provenance": "hotSpring_md_vacf → neuralSpring_spectral → airSpring_hydrology"
    })
}

pub(super) fn gamma_cdf_handler(params: &Value) -> Value {
    let x = f64_p(params, "x").unwrap_or(1.0);
    let alpha = f64_p(params, "alpha").unwrap_or(2.0);
    let beta = f64_p(params, "beta").unwrap_or(1.0);
    let params_g = drought_index::GammaParams { alpha, beta };
    let cdf = drought_index::gamma_cdf(x, &params_g);
    serde_json::json!({
        "gamma_cdf": cdf,
        "x": x,
        "alpha": alpha,
        "beta": beta,
        "upstream": "barracuda::special::gamma::regularized_gamma_p"
    })
}
