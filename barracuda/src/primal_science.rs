// SPDX-License-Identifier: AGPL-3.0-or-later
//! JSON-RPC science handlers for the airSpring primal binary.
//!
//! Separates domain-science dispatch from the primal server infrastructure.
//! Each handler accepts JSON-RPC `params` and returns a JSON result value.

use crate::eco::anderson;
use crate::eco::diversity;
use crate::eco::dual_kc;
use crate::eco::evapotranspiration as et;
use crate::eco::infiltration;
use crate::eco::richards;
use crate::eco::runoff;
use crate::eco::sensor_calibration;
use crate::eco::simple_et0;
use crate::eco::soil_moisture;
use crate::eco::thornthwaite;

fn f64_p(params: &serde_json::Value, key: &str) -> Option<f64> {
    params.get(key).and_then(serde_json::Value::as_f64)
}

fn u32_p(params: &serde_json::Value, key: &str) -> Option<u32> {
    params
        .get(key)
        .and_then(serde_json::Value::as_u64)
        .map(|v| v as u32)
}

/// Dispatch a science method to the appropriate handler.
///
/// Returns `Some(result)` if the method is a known science method,
/// `None` if it should be handled elsewhere (cross-primal, lifecycle, etc.).
#[must_use]
#[allow(clippy::too_many_lines)]
pub fn dispatch_science(method: &str, params: &serde_json::Value) -> Option<serde_json::Value> {
    let result = match method {
        "science.et0_fao56" | "ecology.et0_fao56" => et0_fao56(params),
        "science.et0_hargreaves" | "ecology.et0_hargreaves" => et0_hargreaves(params),
        "science.et0_priestley_taylor" | "ecology.et0_priestley_taylor" => {
            et0_priestley_taylor(params)
        }
        "science.et0_makkink" | "ecology.et0_makkink" => et0_makkink(params),
        "science.et0_turc" | "ecology.et0_turc" => et0_turc(params),
        "science.et0_hamon" | "ecology.et0_hamon" => et0_hamon(params),
        "science.et0_blaney_criddle" | "ecology.et0_blaney_criddle" => et0_blaney_criddle(params),
        "science.water_balance" | "ecology.water_balance" => water_balance(params),
        "science.yield_response" | "ecology.yield_response" => yield_response(params),
        "ecology.full_pipeline" => full_pipeline(params),
        "science.richards_1d" => richards_1d(params),
        "science.scs_cn_runoff" => scs_cn_runoff(params),
        "science.green_ampt_infiltration" => green_ampt(params),
        "science.soil_moisture_topp" => soil_moisture_topp(params),
        "science.pedotransfer_saxton_rawls" => pedotransfer(params),
        "science.dual_kc" => dual_kc_handler(params),
        "science.sensor_calibration" => sensor_cal(params),
        "science.gdd" => gdd(params),
        "science.shannon_diversity" => shannon_diversity(params),
        "science.bray_curtis" => bray_curtis(params),
        "science.anderson_coupling" => anderson_coupling(params),
        "science.thornthwaite" => thornthwaite_handler(params),
        _ => return None,
    };
    Some(result)
}

// ── Evapotranspiration ─────────────────────────────────────────────

fn et0_fao56(params: &serde_json::Value) -> serde_json::Value {
    let input = et::DailyEt0Input {
        tmax: f64_p(params, "tmax").unwrap_or(30.0),
        tmin: f64_p(params, "tmin").unwrap_or(15.0),
        tmean: f64_p(params, "tmean"),
        solar_radiation: f64_p(params, "solar_radiation").unwrap_or(20.0),
        wind_speed_2m: f64_p(params, "wind_speed_2m").unwrap_or(2.0),
        actual_vapour_pressure: f64_p(params, "actual_vapour_pressure").unwrap_or(1.5),
        day_of_year: u32_p(params, "day_of_year").unwrap_or(180),
        latitude_deg: f64_p(params, "latitude_deg").unwrap_or(42.7),
        elevation_m: f64_p(params, "elevation_m").unwrap_or(250.0),
    };
    let result = et::daily_et0(&input);
    serde_json::json!({"et0_mm": result.et0, "rn_mj": result.rn, "method": "fao56_penman_monteith"})
}

fn et0_hargreaves(params: &serde_json::Value) -> serde_json::Value {
    let tmin = f64_p(params, "tmin").unwrap_or(15.0);
    let tmax = f64_p(params, "tmax").unwrap_or(30.0);
    let lat_rad = f64_p(params, "latitude_deg").unwrap_or(42.7).to_radians();
    let doy = u32_p(params, "day_of_year").unwrap_or(180);
    let ra_mm = crate::eco::solar::extraterrestrial_radiation(lat_rad, doy) / 2.45;
    let et0 = et::hargreaves_et0(tmin, tmax, ra_mm);
    serde_json::json!({"et0_mm": et0, "ra_mm_day": ra_mm, "method": "hargreaves"})
}

fn et0_priestley_taylor(params: &serde_json::Value) -> serde_json::Value {
    let et0 = et::priestley_taylor_et0(
        f64_p(params, "rn").unwrap_or(10.0),
        f64_p(params, "g").unwrap_or(0.0),
        f64_p(params, "tmean").unwrap_or(22.5),
        f64_p(params, "elevation_m").unwrap_or(250.0),
    );
    serde_json::json!({"et0_mm": et0, "method": "priestley_taylor"})
}

fn et0_makkink(params: &serde_json::Value) -> serde_json::Value {
    let et0 = simple_et0::makkink_et0(
        f64_p(params, "tmean").unwrap_or(22.5),
        f64_p(params, "solar_radiation").unwrap_or(20.0),
        f64_p(params, "elevation_m").unwrap_or(250.0),
    );
    serde_json::json!({"et0_mm": et0, "method": "makkink"})
}

fn et0_turc(params: &serde_json::Value) -> serde_json::Value {
    let et0 = simple_et0::turc_et0(
        f64_p(params, "tmean").unwrap_or(22.5),
        f64_p(params, "solar_radiation").unwrap_or(20.0),
        f64_p(params, "rh_pct").unwrap_or(60.0),
    );
    serde_json::json!({"et0_mm": et0, "method": "turc"})
}

fn et0_hamon(params: &serde_json::Value) -> serde_json::Value {
    let lat_rad = f64_p(params, "latitude_deg").unwrap_or(42.7).to_radians();
    let et0 = simple_et0::hamon_pet_from_location(
        f64_p(params, "tmean").unwrap_or(22.5),
        lat_rad,
        u32_p(params, "day_of_year").unwrap_or(180),
    );
    serde_json::json!({"pet_mm": et0, "method": "hamon"})
}

fn et0_blaney_criddle(params: &serde_json::Value) -> serde_json::Value {
    let lat_rad = f64_p(params, "latitude_deg").unwrap_or(42.7).to_radians();
    let et0 = simple_et0::blaney_criddle_from_location(
        f64_p(params, "tmean").unwrap_or(22.5),
        lat_rad,
        u32_p(params, "day_of_year").unwrap_or(180),
    );
    serde_json::json!({"et0_mm": et0, "method": "blaney_criddle"})
}

// ── Water balance & yield ──────────────────────────────────────────

fn water_balance(params: &serde_json::Value) -> serde_json::Value {
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

fn yield_response(params: &serde_json::Value) -> serde_json::Value {
    let ky = f64_p(params, "ky").unwrap_or(1.25);
    let eta = f64_p(params, "eta_over_etm").unwrap_or(0.8);
    let max_y = f64_p(params, "max_yield_t_ha").unwrap_or(12.0);
    let ratio = (1.0 - ky * (1.0 - eta)).max(0.0);
    serde_json::json!({"yield_t_ha": max_y * ratio, "yield_ratio": ratio, "ky": ky, "method": "stewart_1977"})
}

fn full_pipeline(params: &serde_json::Value) -> serde_json::Value {
    let tmax = f64_p(params, "tmax").unwrap_or(32.0);
    let tmin = f64_p(params, "tmin").unwrap_or(18.0);
    let input = et::DailyEt0Input {
        tmax,
        tmin,
        tmean: None,
        solar_radiation: f64_p(params, "solar_radiation").unwrap_or(22.5),
        wind_speed_2m: f64_p(params, "wind_speed_2m").unwrap_or(2.0),
        actual_vapour_pressure: f64_p(params, "actual_vapour_pressure").unwrap_or(1.5),
        day_of_year: u32_p(params, "day_of_year").unwrap_or(180),
        latitude_deg: f64_p(params, "latitude_deg").unwrap_or(42.7),
        elevation_m: f64_p(params, "elevation_m").unwrap_or(250.0),
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

// ── Soil physics ───────────────────────────────────────────────────

fn richards_1d(params: &serde_json::Value) -> serde_json::Value {
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
            let last = profiles.last().unwrap();
            let mean = last.theta.iter().sum::<f64>() / last.theta.len() as f64;
            serde_json::json!({"mean_theta": mean, "n_nodes": last.theta.len(), "n_timesteps": profiles.len(), "final_theta": last.theta, "method": "richards_1d_implicit_euler_picard"})
        }
        Err(e) => serde_json::json!({"error": format!("{e}")}),
    }
}

fn scs_cn_runoff(params: &serde_json::Value) -> serde_json::Value {
    let precip = f64_p(params, "precipitation_mm").unwrap_or(50.0);
    let cn = f64_p(params, "curve_number").unwrap_or(75.0);
    let s = runoff::potential_retention(cn);
    serde_json::json!({"runoff_mm": runoff::scs_cn_runoff_standard(precip, cn), "potential_retention_mm": s, "initial_abstraction_mm": runoff::initial_abstraction(s, 0.2), "curve_number": cn, "method": "scs_cn"})
}

fn green_ampt(params: &serde_json::Value) -> serde_json::Value {
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

fn soil_moisture_topp(params: &serde_json::Value) -> serde_json::Value {
    let d = f64_p(params, "dielectric_constant").unwrap_or(15.0);
    let vwc = soil_moisture::topp_equation(d);
    serde_json::json!({"volumetric_water_content": vwc, "inverse_dielectric": soil_moisture::inverse_topp(vwc), "method": "topp_equation_1980"})
}

fn pedotransfer(params: &serde_json::Value) -> serde_json::Value {
    let sr = soil_moisture::saxton_rawls(&soil_moisture::SaxtonRawlsInput {
        sand: f64_p(params, "sand_pct").unwrap_or(40.0) / 100.0,
        clay: f64_p(params, "clay_pct").unwrap_or(20.0) / 100.0,
        om_pct: f64_p(params, "organic_matter_pct").unwrap_or(2.5),
    });
    serde_json::json!({"field_capacity": sr.theta_fc, "wilting_point": sr.theta_wp, "saturation": sr.theta_s, "saturated_conductivity_mm_hr": sr.ksat_mm_hr, "lambda": sr.lambda, "method": "saxton_rawls_2006"})
}

// ── Crop & irrigation ──────────────────────────────────────────────

fn dual_kc_handler(params: &serde_json::Value) -> serde_json::Value {
    let kcb = f64_p(params, "kcb").unwrap_or(1.10);
    let et0 = f64_p(params, "et0_mm").unwrap_or(5.0);
    serde_json::json!({
        "kc_max": dual_kc::kc_max(f64_p(params, "wind_2m").unwrap_or(2.0), f64_p(params, "rh_min").unwrap_or(45.0), f64_p(params, "crop_height_m").unwrap_or(0.5), kcb),
        "etc_mm": dual_kc::etc_dual(kcb, 1.0, 0.0, et0),
        "kcb": kcb,
        "method": "fao56_dual_kc",
    })
}

fn sensor_cal(params: &serde_json::Value) -> serde_json::Value {
    let raw = f64_p(params, "raw_count").unwrap_or(5000.0);
    serde_json::json!({"volumetric_water_content": sensor_calibration::soilwatch10_vwc(raw), "raw_count": raw, "method": "soilwatch10_topp_polynomial"})
}

fn gdd(params: &serde_json::Value) -> serde_json::Value {
    let tbase = f64_p(params, "tbase").unwrap_or(10.0);
    serde_json::json!({"gdd": crate::eco::crop::gdd_avg(f64_p(params, "tmax").unwrap_or(30.0), f64_p(params, "tmin").unwrap_or(15.0), tbase), "tbase": tbase, "method": "gdd_average"})
}

// ── Biodiversity ───────────────────────────────────────────────────

fn shannon_diversity(params: &serde_json::Value) -> serde_json::Value {
    let counts: Vec<f64> = params
        .get("counts")
        .and_then(|v| v.as_array())
        .map(|a| a.iter().filter_map(serde_json::Value::as_f64).collect())
        .unwrap_or_default();
    if counts.is_empty() {
        return serde_json::json!({"error": "missing or empty 'counts' array"});
    }
    let a = diversity::alpha_diversity(&counts);
    serde_json::json!({"shannon": a.shannon, "simpson": a.simpson, "pielou": a.evenness, "observed_species": a.observed, "chao1": a.chao1, "method": "shannon_simpson_chao1"})
}

fn bray_curtis(params: &serde_json::Value) -> serde_json::Value {
    let parse = |k: &str| -> Vec<f64> {
        params
            .get(k)
            .and_then(|v| v.as_array())
            .map(|a| a.iter().filter_map(serde_json::Value::as_f64).collect())
            .unwrap_or_default()
    };
    let (a, b) = (parse("sample_a"), parse("sample_b"));
    if a.is_empty() || b.is_empty() || a.len() != b.len() {
        return serde_json::json!({"error": "sample_a and sample_b must be non-empty and equal length"});
    }
    let bc = diversity::bray_curtis(&a, &b);
    serde_json::json!({"bray_curtis_dissimilarity": bc, "similarity": 1.0 - bc, "method": "bray_curtis"})
}

// ── Geophysics + Monthly ET ────────────────────────────────────────

fn anderson_coupling(params: &serde_json::Value) -> serde_json::Value {
    let r = anderson::coupling_chain(
        f64_p(params, "theta").unwrap_or(0.25),
        f64_p(params, "theta_r").unwrap_or(0.078),
        f64_p(params, "theta_s").unwrap_or(0.43),
    );
    serde_json::json!({"effective_saturation": r.se, "pore_connectivity": r.connectivity, "coordination_number": r.coordination, "effective_dimension": r.d_eff, "disorder_parameter": r.disorder, "regime": format!("{:?}", r.regime), "method": "anderson_soil_coupling"})
}

fn thornthwaite_handler(params: &serde_json::Value) -> serde_json::Value {
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
    let m =
        thornthwaite::thornthwaite_monthly_et0(&arr, f64_p(params, "latitude_deg").unwrap_or(42.7));
    serde_json::json!({"monthly_et0_mm": m.to_vec(), "annual_et0_mm": m.iter().sum::<f64>(), "method": "thornthwaite_1948"})
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::dispatch_science;

    fn empty_params() -> serde_json::Value {
        serde_json::json!({})
    }

    #[test]
    fn test_dispatch_returns_some_for_all_science_methods() {
        let params = empty_params();
        let science_methods = [
            "science.et0_fao56",
            "science.et0_hargreaves",
            "science.et0_priestley_taylor",
            "science.et0_makkink",
            "science.et0_turc",
            "science.et0_hamon",
            "science.et0_blaney_criddle",
            "science.water_balance",
            "science.yield_response",
            "science.richards_1d",
            "science.scs_cn_runoff",
            "science.green_ampt_infiltration",
            "science.soil_moisture_topp",
            "science.pedotransfer_saxton_rawls",
            "science.dual_kc",
            "science.sensor_calibration",
            "science.gdd",
            "science.shannon_diversity",
            "science.bray_curtis",
            "science.anderson_coupling",
            "science.thornthwaite",
        ];
        for method in science_methods {
            let params_for_method = if method == "science.shannon_diversity" {
                serde_json::json!({"counts": [10.0, 5.0, 3.0]})
            } else if method == "science.bray_curtis" {
                serde_json::json!({"sample_a": [1.0, 2.0, 3.0], "sample_b": [2.0, 3.0, 4.0]})
            } else if method == "science.thornthwaite" {
                serde_json::json!({"monthly_temps_c": [5.0, 6.0, 8.0, 12.0, 16.0, 20.0, 22.0, 21.0, 18.0, 13.0, 8.0, 6.0]})
            } else {
                params.clone()
            };
            let result = dispatch_science(method, &params_for_method);
            assert!(
                result.is_some(),
                "dispatch_science should return Some for {method}"
            );
        }
    }

    #[test]
    fn test_dispatch_returns_some_for_all_ecology_methods() {
        let params = empty_params();
        let ecology_methods = [
            "ecology.et0_fao56",
            "ecology.et0_hargreaves",
            "ecology.et0_priestley_taylor",
            "ecology.et0_makkink",
            "ecology.et0_turc",
            "ecology.et0_hamon",
            "ecology.et0_blaney_criddle",
            "ecology.water_balance",
            "ecology.yield_response",
            "ecology.full_pipeline",
        ];
        for method in ecology_methods {
            let result = dispatch_science(method, &params);
            assert!(
                result.is_some(),
                "dispatch_science should return Some for {method}"
            );
        }
    }

    #[test]
    fn test_dispatch_returns_none_for_unknown_methods() {
        let params = empty_params();
        let unknown = [
            "science.unknown",
            "ecology.unknown",
            "foo.bar",
            "science.",
            "ecology.",
            "",
        ];
        for method in unknown {
            let result = dispatch_science(method, &params);
            assert!(
                result.is_none(),
                "dispatch_science should return None for '{method}'"
            );
        }
    }

    #[test]
    fn test_dispatch_et0_fao56_has_et0_mm() {
        let r = dispatch_science("science.et0_fao56", &empty_params()).unwrap();
        assert!(r.get("et0_mm").is_some());
    }

    #[test]
    fn test_dispatch_et0_hargreaves_has_et0_mm() {
        let r = dispatch_science("science.et0_hargreaves", &empty_params()).unwrap();
        assert!(r.get("et0_mm").is_some());
    }

    #[test]
    fn test_dispatch_et0_priestley_taylor_has_et0_mm() {
        let r = dispatch_science("science.et0_priestley_taylor", &empty_params()).unwrap();
        assert!(r.get("et0_mm").is_some());
    }

    #[test]
    fn test_dispatch_et0_makkink_has_et0_mm() {
        let r = dispatch_science("science.et0_makkink", &empty_params()).unwrap();
        assert!(r.get("et0_mm").is_some());
    }

    #[test]
    fn test_dispatch_et0_turc_has_et0_mm() {
        let r = dispatch_science("science.et0_turc", &empty_params()).unwrap();
        assert!(r.get("et0_mm").is_some());
    }

    #[test]
    fn test_dispatch_et0_hamon_has_pet_mm() {
        let r = dispatch_science("science.et0_hamon", &empty_params()).unwrap();
        assert!(r.get("pet_mm").is_some());
    }

    #[test]
    fn test_dispatch_et0_blaney_criddle_has_et0_mm() {
        let r = dispatch_science("science.et0_blaney_criddle", &empty_params()).unwrap();
        assert!(r.get("et0_mm").is_some());
    }

    #[test]
    fn test_dispatch_water_balance_has_etc_and_soil_water() {
        let r = dispatch_science("science.water_balance", &empty_params()).unwrap();
        assert!(r.get("etc_mm").is_some());
        assert!(r.get("soil_water_mm").is_some());
    }

    #[test]
    fn test_dispatch_yield_response_has_yield_fields() {
        let r = dispatch_science("science.yield_response", &empty_params()).unwrap();
        assert!(r.get("yield_t_ha").is_some());
        assert!(r.get("yield_ratio").is_some());
    }

    #[test]
    fn test_dispatch_full_pipeline_has_pipeline_and_stages() {
        let r = dispatch_science("ecology.full_pipeline", &empty_params()).unwrap();
        assert!(r.get("pipeline").is_some());
        assert!(r.get("stages").is_some());
    }

    #[test]
    fn test_dispatch_richards_1d_has_mean_theta_or_error() {
        let r = dispatch_science("science.richards_1d", &empty_params()).unwrap();
        assert!(r.get("mean_theta").is_some() || r.get("error").is_some());
    }

    #[test]
    fn test_dispatch_scs_cn_runoff_has_runoff_mm() {
        let r = dispatch_science("science.scs_cn_runoff", &empty_params()).unwrap();
        assert!(r.get("runoff_mm").is_some());
    }

    #[test]
    fn test_dispatch_green_ampt_has_cumulative_infiltration_cm() {
        let r = dispatch_science("science.green_ampt_infiltration", &empty_params()).unwrap();
        assert!(r.get("cumulative_infiltration_cm").is_some());
    }

    #[test]
    fn test_dispatch_soil_moisture_topp_has_volumetric_water_content() {
        let r = dispatch_science("science.soil_moisture_topp", &empty_params()).unwrap();
        assert!(r.get("volumetric_water_content").is_some());
    }

    #[test]
    fn test_dispatch_pedotransfer_has_field_capacity() {
        let r = dispatch_science("science.pedotransfer_saxton_rawls", &empty_params()).unwrap();
        assert!(r.get("field_capacity").is_some());
    }

    #[test]
    fn test_dispatch_dual_kc_has_kc_max() {
        let r = dispatch_science("science.dual_kc", &empty_params()).unwrap();
        assert!(r.get("kc_max").is_some());
    }

    #[test]
    fn test_dispatch_sensor_cal_has_volumetric_water_content() {
        let r = dispatch_science("science.sensor_calibration", &empty_params()).unwrap();
        assert!(r.get("volumetric_water_content").is_some());
    }

    #[test]
    fn test_dispatch_gdd_has_gdd() {
        let r = dispatch_science("science.gdd", &empty_params()).unwrap();
        assert!(r.get("gdd").is_some());
    }

    #[test]
    fn test_dispatch_shannon_diversity_has_shannon() {
        let params = serde_json::json!({"counts": [10.0, 5.0, 3.0, 2.0]});
        let r = dispatch_science("science.shannon_diversity", &params).unwrap();
        assert!(r.get("shannon").is_some());
    }

    #[test]
    fn test_dispatch_bray_curtis_has_bray_curtis_dissimilarity() {
        let params = serde_json::json!({"sample_a": [1.0, 2.0, 3.0], "sample_b": [2.0, 3.0, 4.0]});
        let r = dispatch_science("science.bray_curtis", &params).unwrap();
        assert!(r.get("bray_curtis_dissimilarity").is_some());
    }

    #[test]
    fn test_dispatch_anderson_coupling_has_effective_saturation() {
        let r = dispatch_science("science.anderson_coupling", &empty_params()).unwrap();
        assert!(r.get("effective_saturation").is_some());
    }

    #[test]
    fn test_dispatch_thornthwaite_has_monthly_et0_mm() {
        let params = serde_json::json!({
            "monthly_temps_c": [5.0, 6.0, 8.0, 12.0, 16.0, 20.0, 22.0, 21.0, 18.0, 13.0, 8.0, 6.0]
        });
        let r = dispatch_science("science.thornthwaite", &params).unwrap();
        assert!(r.get("monthly_et0_mm").is_some());
    }

    #[test]
    fn test_dispatch_shannon_diversity_empty_counts_returns_error() {
        let params = serde_json::json!({"counts": []});
        let r = dispatch_science("science.shannon_diversity", &params).unwrap();
        assert!(r.get("error").is_some());
    }

    #[test]
    fn test_dispatch_bray_curtis_mismatched_lengths_returns_error() {
        let params = serde_json::json!({"sample_a": [1.0, 2.0], "sample_b": [1.0, 2.0, 3.0]});
        let r = dispatch_science("science.bray_curtis", &params).unwrap();
        assert!(r.get("error").is_some());
    }

    #[test]
    fn test_dispatch_thornthwaite_wrong_months_returns_error() {
        let params = serde_json::json!({"monthly_temps_c": [5.0, 6.0, 8.0]});
        let r = dispatch_science("science.thornthwaite", &params).unwrap();
        assert!(r.get("error").is_some());
    }
}
