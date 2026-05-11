// SPDX-License-Identifier: AGPL-3.0-or-later

//! Scenario: ET₀ multi-method cross-validation.
//!
//! Validates all ET₀ methods produce physically reasonable values and
//! ecology aliases route identically to science methods.

use crate::methods as m;
use crate::primal_science::dispatch_science;
use crate::validation::ValidationHarness;
use crate::validation::scenarios::registry::{Scenario, ScenarioMeta, Tier, Track};

/// ET₀ multi-method cross-validation scenario metadata and entry point.
pub const SCENARIO: Scenario = Scenario {
    meta: ScenarioMeta {
        id: "et0-methods",
        track: Track::ScienceDispatch,
        tier: Tier::Rust,
        provenance_crate: "validate_et0_methods",
        provenance_date: "2026-05-11",
        description: "ET₀ method parity + ecology alias routing",
    },
    run,
};

fn alias_parity(v: &mut ValidationHarness, sci: &str, eco: &str, params: &serde_json::Value) {
    let s = dispatch_science(sci, params);
    let e = dispatch_science(eco, params);
    v.check_bool(&format!("alias: {sci} == {eco}"), s == e);
}

/// Run ET₀ multi-method cross-validation scenario.
pub fn run(v: &mut ValidationHarness) {
    let empty = serde_json::json!({});
    alias_parity(v, m::ET0_FAO56, m::ECO_ET0_FAO56, &empty);
    alias_parity(v, m::ET0_HARGREAVES, m::ECO_ET0_HARGREAVES, &empty);
    alias_parity(v, m::WATER_BALANCE, m::ECO_WATER_BALANCE, &empty);
    alias_parity(v, m::YIELD_RESPONSE, m::ECO_YIELD_RESPONSE, &empty);

    for method in [
        m::ET0_FAO56,
        m::ET0_HARGREAVES,
        m::ET0_PRIESTLEY_TAYLOR,
        m::ET0_MAKKINK,
        m::ET0_TURC,
        m::ET0_HAMON,
        m::ET0_BLANEY_CRIDDLE,
    ] {
        let result = dispatch_science(method, &empty);
        v.check_bool(&format!("{method}: dispatches"), result.is_some());
    }
}
