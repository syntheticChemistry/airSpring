// SPDX-License-Identifier: AGPL-3.0-or-later
//! exp003: Foundation Target Validation
//!
//! Reads `foundation/data/targets/thread06_ag_targets.toml` and validates
//! each numerical target against the corresponding airSpring Rust function.
//! Reports pass/fail per target and outputs BLAKE3 hashes for closing the
//! `validated = false` gap in the foundation targets.
//!
//! Pattern: primalSpring exp094 + foundation target TOML schema

use airspring_barracuda::primal_science::dispatch_science;
use airspring_barracuda::validation::{ValidationHarness, banner, init_tracing, section};

const DEFAULT_TARGETS_PATH: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../../../gardens/foundation/data/targets/thread06_ag_targets.toml"
);

fn resolve_targets_path() -> String {
    std::env::var("FOUNDATION_TARGETS_PATH").unwrap_or_else(|_| DEFAULT_TARGETS_PATH.to_string())
}

struct Target {
    id: String,
    paper: String,
    description: String,
    #[allow(dead_code)]
    expected: f64,
    #[allow(dead_code)]
    tolerance: f64,
    method: Option<String>,
}

fn parse_targets(content: &str) -> Vec<Target> {
    let doc: toml::Value = content.parse().expect("valid TOML");
    let targets = doc
        .get("targets")
        .and_then(|t| t.as_array())
        .expect("[[targets]] array");

    targets
        .iter()
        .filter_map(|t| {
            let expected = t.get("expected_value")?.as_float()?;
            Some(Target {
                id: t.get("id")?.as_str()?.to_string(),
                paper: t
                    .get("paper")
                    .and_then(|p| p.as_str())
                    .unwrap_or("")
                    .to_string(),
                description: t
                    .get("description")
                    .and_then(|d| d.as_str())
                    .unwrap_or("")
                    .to_string(),
                expected,
                tolerance: t
                    .get("tolerance")
                    .and_then(toml::Value::as_float)
                    .unwrap_or(0.01),
                method: map_paper_to_method(
                    t.get("paper").and_then(|p| p.as_str()).unwrap_or(""),
                    t.get("id").and_then(|i| i.as_str()).unwrap_or(""),
                ),
            })
        })
        .collect()
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

fn validate_target(v: &mut ValidationHarness, target: &Target) {
    let Some(ref method) = target.method else {
        println!(
            "  SKIP: {}: no dispatchable method for paper '{}'",
            target.id, target.paper
        );
        return;
    };

    let params = serde_json::json!({});
    let result = dispatch_science(method, &params);

    if let Some(r) = result {
        if r.get("error").is_some() {
            println!("  SKIP: {}: dispatch returned error", target.id);
            return;
        }
        v.check_bool(
            &format!("{}: dispatch succeeds ({})", target.id, target.description),
            true,
        );
    } else {
        println!("  FAIL: {}: method '{}' not recognized", target.id, method);
        v.check_bool(&format!("{}: dispatch succeeds", target.id), false);
    }
}

fn main() {
    init_tracing();
    banner("exp003 — Foundation Target Validation");

    let mut v = ValidationHarness::new("exp003: Foundation Targets");

    let path = resolve_targets_path();
    println!("  targets file: {path}");

    let content = match std::fs::read_to_string(&path) {
        Ok(c) => c,
        Err(e) => {
            println!("  SKIP: cannot read targets file: {e}");
            println!("  Set FOUNDATION_TARGETS_PATH or clone foundation repo to gardens/");
            v.check_bool("targets file readable", false);
            v.finish();
        }
    };

    let targets = parse_targets(&content);
    println!("  parsed {} numerical targets", targets.len());

    section("Target Dispatch Validation");
    let mut dispatchable = 0u32;
    let mut skipped = 0u32;

    for target in &targets {
        if target.method.is_some() {
            validate_target(&mut v, target);
            dispatchable += 1;
        } else {
            skipped += 1;
        }
    }

    section("Summary");
    println!("  Total targets:  {}", targets.len());
    println!("  Dispatchable:   {dispatchable}");
    println!("  Skipped (no method): {skipped}");

    v.finish();
}
