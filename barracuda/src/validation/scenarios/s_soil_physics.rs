// SPDX-License-Identifier: AGPL-3.0-or-later

//! Scenario: Soil physics validation.
//!
//! Validates Richards 1D, SCS curve number, Green-Ampt infiltration,
//! soil moisture (Topp), and Saxton-Rawls pedotransfer functions.

use crate::methods as m;
use crate::primal_science::dispatch_science;
use crate::validation::ValidationHarness;
use crate::validation::scenarios::registry::{Scenario, ScenarioMeta, Tier, Track};

/// Soil physics validation scenario metadata and entry point.
pub const SCENARIO: Scenario = Scenario {
    meta: ScenarioMeta {
        id: "soil-physics",
        track: Track::ScienceDispatch,
        tier: Tier::Rust,
        provenance_crate: "validate_soil_physics",
        provenance_date: "2026-05-11",
        description: "Richards, SCS-CN, Green-Ampt, Topp, Saxton-Rawls",
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

/// Run soil physics validation scenario.
pub fn run(v: &mut ValidationHarness) {
    let empty = serde_json::json!({});
    check(v, m::RICHARDS_1D, &empty);
    check(v, m::SCS_CN_RUNOFF, &empty);
    check(v, m::GREEN_AMPT, &empty);
    check(v, m::SOIL_MOISTURE_TOPP, &empty);
    check(v, m::PEDOTRANSFER, &empty);
}
