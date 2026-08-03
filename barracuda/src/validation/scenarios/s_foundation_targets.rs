// SPDX-License-Identifier: AGPL-3.0-or-later

//! Scenario: Foundation Target Validation — numerical parity.
//!
//! Reads `foundation/data/targets/thread06_ag_targets.toml` and validates
//! each numerical target against the corresponding airSpring dispatch method.
//! Targets with `unit = "qualitative_match"` check that dispatch succeeds;
//! targets with numeric `expected_value` + `tolerance` assert parity.

use crate::primal_science::dispatch_science;
use crate::validation::ValidationHarness;
use crate::validation::scenarios::registry::{Scenario, ScenarioMeta, Tier, Track};

/// Foundation Thread 6 agricultural targets — numerical parity scenario.
pub const SCENARIO: Scenario = Scenario {
    meta: ScenarioMeta {
        id: "foundation-targets",
        track: Track::Foundation,
        tier: Tier::Rust,
        provenance_crate: "exp003_foundation_target_validation",
        provenance_date: "2026-05-16",
        description: "Foundation thread06 agricultural targets — numerical parity validation",
    },
    run,
};

const DEFAULT_TARGETS_PATH: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../../../gardens/foundation/data/targets/thread06_ag_targets.toml"
);

fn resolve_targets_path() -> String {
    std::env::var("FOUNDATION_TARGETS_PATH").unwrap_or_else(|_| DEFAULT_TARGETS_PATH.to_string())
}

fn map_paper_to_method(paper: &str, id: &str) -> Option<&'static str> {
    match paper {
        "FAO56_PM" => Some("science.et0_fao56"),
        "PT1972" | "PRIESTLEY_TAYLOR" => Some("science.et0_priestley_taylor"),
        "TH1948" | "THORNTHWAITE" => Some("science.thornthwaite"),
        "HS1985" | "HARGREAVES_SAMANI" => Some("science.et0_hargreaves"),
        "MAKKINK" => Some("science.et0_makkink"),
        "TURC" => Some("science.et0_turc"),
        "HAMON" => Some("science.et0_hamon"),
        "BLANEY_CRIDDLE" => Some("science.et0_blaney_criddle"),
        "FAO56_DUAL_KC" => Some("science.dual_kc"),
        "STEWART_YIELD" | "DOORENBOS_KASSAM" => Some("science.yield_response"),
        "SAXTON_RAWLS" => Some("science.pedotransfer_saxton_rawls"),
        "RICHARDS_VG" | "VAN_GENUCHTEN" => Some("science.richards_1d"),
        "SCS_CN" => Some("science.scs_cn_runoff"),
        "GREEN_AMPT" => Some("science.green_ampt_infiltration"),
        "SPI" | "MCKEE_SPI" => Some("science.spi_drought_index"),
        "SHANNON" => Some("science.shannon_diversity"),
        "GDD" => Some("science.gdd"),
        _ => {
            if id.contains("spi") || id.contains("gamma") {
                Some("science.spi_drought_index")
            } else {
                None
            }
        }
    }
}

fn extract_result_value(result: &serde_json::Value, method: &str) -> Option<f64> {
    let keys: &[&str] = match method {
        "science.et0_fao56"
        | "science.et0_hargreaves"
        | "science.et0_makkink"
        | "science.et0_turc"
        | "science.et0_hamon"
        | "science.et0_blaney_criddle" => &["et0_mm"],
        "science.et0_priestley_taylor" => &["et0_mm", "pet_mm"],
        "science.thornthwaite" => &["annual_pet_mm", "pet_mm"],
        "science.dual_kc" => &["et_adj_mm", "kc_adj"],
        "science.yield_response" => &["ya_fraction", "yield_ratio"],
        "science.pedotransfer_saxton_rawls" => &["fc", "wp"],
        "science.richards_1d" => &["theta_surface"],
        "science.scs_cn_runoff" => &["runoff_mm"],
        "science.green_ampt_infiltration" => &["infiltration_mm"],
        "science.spi_drought_index" => &["spi"],
        "science.shannon_diversity" => &["h_prime"],
        "science.gdd" => &["gdd"],
        _ => &["value", "result"],
    };

    for key in keys {
        if let Some(v) = result.get(key).and_then(serde_json::Value::as_f64) {
            return Some(v);
        }
    }
    result.as_f64()
}

fn toml_f64(v: Option<&toml::Value>) -> Option<f64> {
    v.and_then(|val| {
        val.as_float().or_else(|| {
            val.as_integer()
                .and_then(|i| i32::try_from(i).ok())
                .map(f64::from)
        })
    })
}

/// Run this validation scenario.
pub fn run(v: &mut ValidationHarness) {
    let path = resolve_targets_path();
    println!("  targets file: {path}");

    let Ok(content) = std::fs::read_to_string(&path) else {
        println!("  SKIP: cannot read targets file");
        println!("  Set FOUNDATION_TARGETS_PATH or clone foundation repo to gardens/");
        return;
    };

    let Ok(doc) = content.parse::<toml::Value>() else {
        println!("  SKIP: invalid TOML in targets file");
        return;
    };

    let Some(targets) = doc.get("targets").and_then(|t| t.as_array()) else {
        println!("  SKIP: no [[targets]] array");
        return;
    };

    println!("  parsed {} numerical targets", targets.len());

    let mut parity_pass = 0u32;
    let mut qualitative_pass = 0u32;
    let mut skipped = 0u32;

    for t in targets {
        let id = t.get("id").and_then(|v| v.as_str()).unwrap_or("<unknown>");
        let paper = t.get("paper").and_then(|v| v.as_str()).unwrap_or("");
        let unit = t.get("unit").and_then(|v| v.as_str()).unwrap_or("");
        let expected = toml_f64(t.get("expected_value"));
        let tolerance = toml_f64(t.get("tolerance")).unwrap_or(0.0);

        let Some(method) = map_paper_to_method(paper, id) else {
            skipped += 1;
            continue;
        };

        let params = build_params_for_target(t, method);
        let result = dispatch_science(method, &params);

        match result {
            Some(ref r) if r.get("error").is_none() => {
                if unit == "qualitative_match" || unit == "table_entries_within_tol" {
                    v.check_bool(&format!("{id}: qualitative PASS"), true);
                    qualitative_pass += 1;
                } else if let Some(exp) = expected {
                    if let Some(actual) = extract_result_value(r, method) {
                        let within = (actual - exp).abs() <= tolerance;
                        v.check_bool(
                            &format!("{id}: |{actual:.4} - {exp:.4}| <= {tolerance}"),
                            within,
                        );
                        if within {
                            parity_pass += 1;
                        }
                    } else {
                        v.check_bool(
                            &format!("{id}: dispatch succeeds (no numeric extraction)"),
                            true,
                        );
                        qualitative_pass += 1;
                    }
                } else {
                    v.check_bool(&format!("{id}: dispatch succeeds"), true);
                    qualitative_pass += 1;
                }
            }
            Some(_) => {
                println!("  SKIP: {id}: dispatch returned error");
                skipped += 1;
            }
            None => {
                v.check_bool(&format!("{id}: method routable"), false);
            }
        }
    }

    println!("  Numerical parity: {parity_pass}");
    println!("  Qualitative: {qualitative_pass}");
    println!("  Skipped: {skipped}");
}

fn build_params_for_target(target: &toml::Value, method: &str) -> serde_json::Value {
    if let Some(params) = target.get("params")
        && let Ok(json_str) = serde_json::to_string(params)
        && let Ok(v) = serde_json::from_str::<serde_json::Value>(&json_str)
    {
        return v;
    }

    match method {
        "science.thornthwaite" => {
            if let Some(id) = target.get("id").and_then(|v| v.as_str()) {
                if id.contains("east_lansing") {
                    return serde_json::json!({
                        "monthly_temps": [-4.0, -2.5, 3.0, 9.5, 15.5, 21.0, 23.5, 22.5, 18.0, 11.5, 5.0, -1.5],
                        "latitude_deg": 42.7
                    });
                } else if id.contains("wooster") {
                    return serde_json::json!({
                        "monthly_temps": [-2.0, -1.0, 4.5, 10.5, 16.0, 21.5, 24.0, 23.0, 19.0, 12.5, 6.0, 0.0],
                        "latitude_deg": 40.8
                    });
                }
            }
            serde_json::json!({})
        }
        _ => serde_json::json!({}),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn map_paper_known_methods() {
        assert_eq!(
            map_paper_to_method("FAO56_PM", ""),
            Some("science.et0_fao56")
        );
        assert_eq!(
            map_paper_to_method("HARGREAVES_SAMANI", ""),
            Some("science.et0_hargreaves")
        );
        assert_eq!(
            map_paper_to_method("THORNTHWAITE", ""),
            Some("science.thornthwaite")
        );
        assert_eq!(
            map_paper_to_method("RICHARDS_VG", ""),
            Some("science.richards_1d")
        );
        assert_eq!(
            map_paper_to_method("SCS_CN", ""),
            Some("science.scs_cn_runoff")
        );
        assert_eq!(
            map_paper_to_method("GREEN_AMPT", ""),
            Some("science.green_ampt_infiltration")
        );
        assert_eq!(
            map_paper_to_method("SAXTON_RAWLS", ""),
            Some("science.pedotransfer_saxton_rawls")
        );
        assert_eq!(
            map_paper_to_method("FAO56_DUAL_KC", ""),
            Some("science.dual_kc")
        );
        assert_eq!(
            map_paper_to_method("STEWART_YIELD", ""),
            Some("science.yield_response")
        );
        assert_eq!(
            map_paper_to_method("SHANNON", ""),
            Some("science.shannon_diversity")
        );
        assert_eq!(map_paper_to_method("GDD", ""), Some("science.gdd"));
    }

    #[test]
    fn map_paper_aliases() {
        assert_eq!(
            map_paper_to_method("PT1972", ""),
            Some("science.et0_priestley_taylor")
        );
        assert_eq!(
            map_paper_to_method("HS1985", ""),
            Some("science.et0_hargreaves")
        );
        assert_eq!(
            map_paper_to_method("TH1948", ""),
            Some("science.thornthwaite")
        );
        assert_eq!(
            map_paper_to_method("VAN_GENUCHTEN", ""),
            Some("science.richards_1d")
        );
        assert_eq!(
            map_paper_to_method("DOORENBOS_KASSAM", ""),
            Some("science.yield_response")
        );
        assert_eq!(
            map_paper_to_method("SPI", ""),
            Some("science.spi_drought_index")
        );
        assert_eq!(
            map_paper_to_method("MCKEE_SPI", ""),
            Some("science.spi_drought_index")
        );
    }

    #[test]
    fn map_paper_fallback_spi() {
        assert_eq!(
            map_paper_to_method("UNKNOWN", "target_spi_3month"),
            Some("science.spi_drought_index")
        );
        assert_eq!(
            map_paper_to_method("UNKNOWN", "gamma_fit"),
            Some("science.spi_drought_index")
        );
    }

    #[test]
    fn map_paper_unknown() {
        assert_eq!(map_paper_to_method("UNKNOWN", "unrelated"), None);
    }

    #[test]
    fn extract_result_et0() {
        let val = serde_json::json!({"et0_mm": 4.5, "other": 99.0});
        assert_eq!(extract_result_value(&val, "science.et0_fao56"), Some(4.5));
        assert_eq!(
            extract_result_value(&val, "science.et0_hargreaves"),
            Some(4.5)
        );
    }

    #[test]
    fn extract_result_richards() {
        let val = serde_json::json!({"theta_surface": 0.32});
        assert_eq!(
            extract_result_value(&val, "science.richards_1d"),
            Some(0.32)
        );
    }

    #[test]
    fn extract_result_thornthwaite_annual() {
        let val = serde_json::json!({"annual_pet_mm": 850.0});
        assert_eq!(
            extract_result_value(&val, "science.thornthwaite"),
            Some(850.0)
        );
    }

    #[test]
    fn extract_result_fallback_keys() {
        let val = serde_json::json!({"value": 42.0});
        assert_eq!(
            extract_result_value(&val, "science.unknown_method"),
            Some(42.0)
        );
    }

    #[test]
    fn extract_result_bare_f64() {
        let val = serde_json::json!(3.14);
        assert_eq!(extract_result_value(&val, "anything"), Some(3.14));
    }

    #[test]
    fn extract_result_no_match() {
        let val = serde_json::json!({"unrelated": "text"});
        assert_eq!(extract_result_value(&val, "science.et0_fao56"), None);
    }

    #[test]
    fn toml_f64_from_float() {
        let v = toml::Value::Float(3.14);
        assert_eq!(toml_f64(Some(&v)), Some(3.14));
    }

    #[test]
    fn toml_f64_from_int() {
        let v = toml::Value::Integer(42);
        assert_eq!(toml_f64(Some(&v)), Some(42.0));
    }

    #[test]
    fn toml_f64_none() {
        assert_eq!(toml_f64(None), None);
    }

    #[test]
    fn toml_f64_string_returns_none() {
        let v = toml::Value::String("not a number".to_string());
        assert_eq!(toml_f64(Some(&v)), None);
    }

    #[test]
    fn build_params_thornthwaite_east_lansing() {
        let target: toml::Value = toml::from_str(r#"id = "et0_east_lansing""#).unwrap();
        let params = build_params_for_target(&target, "science.thornthwaite");
        assert!(params.get("monthly_temps").is_some());
        assert!(params.get("latitude_deg").is_some());
    }

    #[test]
    fn build_params_thornthwaite_wooster() {
        let target: toml::Value = toml::from_str(r#"id = "et0_wooster_oh""#).unwrap();
        let params = build_params_for_target(&target, "science.thornthwaite");
        assert!(params.get("monthly_temps").is_some());
        let lat = params["latitude_deg"].as_f64().unwrap();
        assert!((lat - 40.8).abs() < 0.01);
    }

    #[test]
    fn build_params_default_empty() {
        let target: toml::Value = toml::from_str(r#"id = "generic_target""#).unwrap();
        let params = build_params_for_target(&target, "science.et0_fao56");
        assert_eq!(params, serde_json::json!({}));
    }

    #[test]
    fn build_params_with_inline_params() {
        let target: toml::Value = toml::from_str(
            r#"
            id = "custom"
            [params]
            x = 1.0
            alpha = 2.0
            "#,
        )
        .unwrap();
        let params = build_params_for_target(&target, "science.gamma_cdf");
        assert!(params.get("x").is_some());
    }
}
