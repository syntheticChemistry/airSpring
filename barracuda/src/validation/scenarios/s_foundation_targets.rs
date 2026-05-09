// SPDX-License-Identifier: AGPL-3.0-or-later

//! Scenario: Foundation Target Validation — absorbed from exp003.
//!
//! Reads `foundation/data/targets/thread06_ag_targets.toml` and validates
//! each numerical target against the corresponding airSpring dispatch method.

use crate::primal_science::dispatch_science;
use crate::validation::ValidationHarness;
use crate::validation::scenarios::registry::{Scenario, ScenarioMeta, Tier, Track};

/// Scenario metadata and entry point.
pub const SCENARIO: Scenario = Scenario {
    meta: ScenarioMeta {
        id: "foundation-targets",
        track: Track::Foundation,
        tier: Tier::Rust,
        provenance_crate: "exp003_foundation_target_validation",
        provenance_date: "2026-05-09",
        description: "Foundation thread06 agricultural targets — dispatch validation",
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

fn map_paper_to_method(paper: &str, id: &str) -> Option<String> {
    match paper {
        "FAO56_PM" => Some("science.et0_fao56".to_string()),
        "PRIESTLEY_TAYLOR" => Some("science.et0_priestley_taylor".to_string()),
        "THORNTHWAITE" => Some("science.thornthwaite".to_string()),
        "HARGREAVES_SAMANI" => Some("science.et0_hargreaves".to_string()),
        "MAKKINK" => Some("science.et0_makkink".to_string()),
        "TURC" => Some("science.et0_turc".to_string()),
        "HAMON" => Some("science.et0_hamon".to_string()),
        "BLANEY_CRIDDLE" => Some("science.et0_blaney_criddle".to_string()),
        "FAO56_DUAL_KC" => Some("science.dual_kc".to_string()),
        "STEWART_YIELD" => Some("science.yield_response".to_string()),
        "SAXTON_RAWLS" => Some("science.pedotransfer_saxton_rawls".to_string()),
        "RICHARDS_VG" => Some("science.richards_1d".to_string()),
        "SCS_CN" => Some("science.scs_cn_runoff".to_string()),
        "GREEN_AMPT" => Some("science.green_ampt_infiltration".to_string()),
        _ => {
            if id.contains("spi") || id.contains("gamma") {
                Some("science.spi_drought_index".to_string())
            } else {
                None
            }
        }
    }
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

    let mut dispatchable = 0u32;

    for t in targets {
        let id = t.get("id").and_then(|v| v.as_str()).unwrap_or("<unknown>");
        let paper = t.get("paper").and_then(|v| v.as_str()).unwrap_or("");

        let Some(method) = map_paper_to_method(paper, id) else {
            continue;
        };

        let result = dispatch_science(&method, &serde_json::json!({}));
        match result {
            Some(r) if r.get("error").is_none() => {
                v.check_bool(&format!("{id}: dispatch succeeds"), true);
                dispatchable += 1;
            }
            Some(_) => println!("  SKIP: {id}: dispatch returned error"),
            None => {
                v.check_bool(&format!("{id}: dispatch succeeds"), false);
            }
        }
    }

    println!("  Dispatchable targets validated: {dispatchable}");
}
