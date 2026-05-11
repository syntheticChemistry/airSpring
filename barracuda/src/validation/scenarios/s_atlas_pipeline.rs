// SPDX-License-Identifier: AGPL-3.0-or-later

//! Scenario: Michigan Crop Water Atlas pipeline.
//!
//! Validates the full ecology pipeline: weather → ET₀ → Kc → WB → yield,
//! exercising the complete agricultural science chain.

use crate::methods as m;
use crate::primal_science::dispatch_science;
use crate::validation::ValidationHarness;
use crate::validation::scenarios::registry::{Scenario, ScenarioMeta, Tier, Track};

/// Atlas pipeline scenario metadata and entry point.
pub const SCENARIO: Scenario = Scenario {
    meta: ScenarioMeta {
        id: "atlas-pipeline",
        track: Track::ScienceDispatch,
        tier: Tier::Rust,
        provenance_crate: "validate_atlas",
        provenance_date: "2026-05-11",
        description: "Full atlas pipeline: ET₀ → water balance → yield",
    },
    run,
};

fn check(v: &mut ValidationHarness, method: &str, params: &serde_json::Value) {
    let result = dispatch_science(method, params);
    v.check_bool(&format!("{method}: returns result"), result.is_some());
    if let Some(ref r) = result {
        v.check_bool(&format!("{method}: no error"), r.get("error").is_none());
    }
}

/// Run atlas pipeline validation scenario.
pub fn run(v: &mut ValidationHarness) {
    let empty = serde_json::json!({});
    check(v, m::ET0_FAO56, &empty);
    check(v, m::WATER_BALANCE, &empty);
    check(v, m::YIELD_RESPONSE, &empty);
    check(v, m::DUAL_KC, &empty);
    check(v, m::ECO_FULL_PIPELINE, &empty);
}
