// SPDX-License-Identifier: AGPL-3.0-or-later

//! Scenario: Water balance and yield response.
//!
//! Validates FAO-56 Chapter 8 water balance scheduling, Stewart yield
//! response, and dual crop coefficient models.

use crate::methods as m;
use crate::primal_science::dispatch_science;
use crate::validation::ValidationHarness;
use crate::validation::scenarios::registry::{Scenario, ScenarioMeta, Tier, Track};

/// Water balance and yield response scenario metadata and entry point.
pub const SCENARIO: Scenario = Scenario {
    meta: ScenarioMeta {
        id: "water-balance",
        track: Track::ScienceDispatch,
        tier: Tier::Rust,
        provenance_crate: "validate_water_balance",
        provenance_date: "2026-05-11",
        description: "Water balance, yield response, dual Kc, GDD",
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

/// Run water balance and yield response scenario.
pub fn run(v: &mut ValidationHarness) {
    let empty = serde_json::json!({});
    check(v, m::WATER_BALANCE, &empty);
    check(v, m::YIELD_RESPONSE, &empty);
    check(v, m::DUAL_KC, &empty);
    check(v, m::GDD, &empty);
    check(v, m::SENSOR_CALIBRATION, &empty);
}
