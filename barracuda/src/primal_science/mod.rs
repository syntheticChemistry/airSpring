// SPDX-License-Identifier: AGPL-3.0-or-later
//! JSON-RPC science handlers for the airSpring primal binary.
//!
//! Separates domain-science dispatch from the primal server infrastructure.
//! Each handler accepts JSON-RPC `params` and returns a JSON result value.
//!
//! # Default Parameter Strategy
//!
//! When optional JSON-RPC parameters are omitted, handlers use sensible
//! defaults centred on the MSU Biosystems & Agricultural Engineering
//! research context (East Lansing, MI):
//!
//! | Parameter | Default | Rationale |
//! |-----------|---------|-----------|
//! | `latitude_deg` | 42.7 | MSU campus, East Lansing MI |
//! | `elevation_m` | 250.0 | Southern Michigan average elevation |
//! | `day_of_year` | 180 | Late June (peak growing season) |
//! | `tmax` / `tmin` | 30.0 / 15.0 | Typical Michigan summer day |
//! | `solar_radiation` | 20.0 MJ/m²/day | Michigan summer average |
//! | `wind_speed_2m` | 2.0 m/s | Typical continental interior |
//!
//! These defaults ensure the primal returns physically meaningful results
//! even when called with minimal parameters. They are **not** hardcoded
//! validation targets — all validation uses explicit benchmark JSON with
//! full provenance (see [`crate::tolerances`]).

mod biodiversity;
mod crop;
mod drought_stats;
mod et0;
mod soil;
mod water_balance;

/// Default latitude for Michigan-centric research (MSU East Lansing).
pub(crate) const DEFAULT_LATITUDE_DEG: f64 = 42.7;

/// Default elevation for southern Michigan (metres above sea level).
pub(crate) const DEFAULT_ELEVATION_M: f64 = 250.0;

/// Default day-of-year: late June, peak growing season.
pub(crate) const DEFAULT_DOY: u32 = 180;

fn f64_p(params: &serde_json::Value, key: &str) -> Option<f64> {
    params.get(key).and_then(serde_json::Value::as_f64)
}

fn u32_p(params: &serde_json::Value, key: &str) -> Option<u32> {
    params
        .get(key)
        .and_then(serde_json::Value::as_u64)
        .and_then(|v| u32::try_from(v).ok())
}

/// Dispatch a science method to the appropriate handler.
///
/// Returns `Some(result)` if the method is a known science method,
/// `None` if it should be handled elsewhere (cross-primal, lifecycle, etc.).
#[must_use]
pub fn dispatch_science(method: &str, params: &serde_json::Value) -> Option<serde_json::Value> {
    let result = match method {
        "science.et0_fao56" | "ecology.et0_fao56" => et0::et0_fao56(params),
        "science.et0_hargreaves" | "ecology.et0_hargreaves" => et0::et0_hargreaves(params),
        "science.et0_priestley_taylor" | "ecology.et0_priestley_taylor" => {
            et0::et0_priestley_taylor(params)
        }
        "science.et0_makkink" | "ecology.et0_makkink" => et0::et0_makkink(params),
        "science.et0_turc" | "ecology.et0_turc" => et0::et0_turc(params),
        "science.et0_hamon" | "ecology.et0_hamon" => et0::et0_hamon(params),
        "science.et0_blaney_criddle" | "ecology.et0_blaney_criddle" => {
            et0::et0_blaney_criddle(params)
        }
        "science.water_balance" | "ecology.water_balance" => water_balance::water_balance(params),
        "science.yield_response" | "ecology.yield_response" => {
            water_balance::yield_response(params)
        }
        "ecology.full_pipeline" => water_balance::full_pipeline(params),
        "science.richards_1d" => soil::richards_1d(params),
        "science.scs_cn_runoff" => soil::scs_cn_runoff(params),
        "science.green_ampt_infiltration" => soil::green_ampt(params),
        "science.soil_moisture_topp" => soil::soil_moisture_topp(params),
        "science.pedotransfer_saxton_rawls" => soil::pedotransfer(params),
        "science.dual_kc" => crop::dual_kc_handler(params),
        "science.sensor_calibration" => crop::sensor_cal(params),
        "science.gdd" => crop::gdd(params),
        "science.shannon_diversity" => biodiversity::shannon_diversity(params),
        "science.bray_curtis" => biodiversity::bray_curtis(params),
        "science.anderson_coupling" => drought_stats::anderson_coupling(params),
        "science.thornthwaite" => drought_stats::thornthwaite_handler(params),
        "science.spi_drought_index" | "ecology.spi_drought_index" => {
            drought_stats::spi_drought(params)
        }
        "science.autocorrelation" | "ecology.autocorrelation" => {
            drought_stats::autocorrelation_handler(params)
        }
        "science.gamma_cdf" => drought_stats::gamma_cdf_handler(params),
        _ => return None,
    };
    Some(result)
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test code uses unwrap for clarity")]
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
            "science.spi_drought_index",
            "science.autocorrelation",
            "science.gamma_cdf",
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

    #[test]
    fn test_dispatch_spi_drought_index() {
        let params = serde_json::json!({
            "monthly_precip_mm": [50.0, 60.0, 45.0, 70.0, 80.0, 55.0, 40.0, 65.0, 50.0, 75.0, 60.0, 45.0],
            "scale": 3
        });
        let r = dispatch_science("science.spi_drought_index", &params).unwrap();
        assert!(r.get("spi").is_some());
        assert_eq!(r["n_months"], 12);
        assert!(r["n_valid"].as_u64().unwrap() > 0);
        assert!(r.get("classifications").is_some());
    }

    #[test]
    fn test_dispatch_spi_drought_index_ecology_alias() {
        let precip: Vec<f64> = vec![50.0; 24];
        let params = serde_json::json!({
            "monthly_precip_mm": precip
        });
        let r = dispatch_science("ecology.spi_drought_index", &params).unwrap();
        assert!(r.get("spi").is_some());
    }

    #[test]
    fn test_dispatch_autocorrelation() {
        let params = serde_json::json!({
            "data": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "max_lag": 5
        });
        let r = dispatch_science("science.autocorrelation", &params).unwrap();
        let acf = r["acf"].as_array().unwrap();
        assert_eq!(acf.len(), 5);
        assert!(r.get("normalised_acf").is_some());
        assert!(r.get("provenance").is_some());
    }

    #[test]
    fn test_dispatch_gamma_cdf() {
        let params = serde_json::json!({"x": 1.0, "alpha": 2.0, "beta": 1.0});
        let r = dispatch_science("science.gamma_cdf", &params).unwrap();
        let cdf = r["gamma_cdf"].as_f64().unwrap();
        assert!(cdf > 0.0 && cdf < 1.0);
    }

    #[test]
    fn test_dispatch_spi_insufficient_data() {
        let params = serde_json::json!({"monthly_precip_mm": [10.0, 20.0]});
        let r = dispatch_science("science.spi_drought_index", &params).unwrap();
        assert!(r.get("error").is_some());
    }

    #[test]
    fn test_dispatch_autocorrelation_empty() {
        let params = serde_json::json!({"data": []});
        let r = dispatch_science("science.autocorrelation", &params).unwrap();
        assert!(r.get("error").is_some());
    }
}
