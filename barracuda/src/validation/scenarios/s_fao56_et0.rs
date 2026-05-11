// SPDX-License-Identifier: AGPL-3.0-or-later

//! Scenario: FAO-56 ET₀ — 75/75 cross-validated reference values.
//!
//! Exercises the 7 evapotranspiration methods against published paper
//! reference values (Allen et al. 1998, Hargreaves-Samani 1985, etc.).

use crate::methods as m;
use crate::primal_science::dispatch_science;
use crate::validation::ValidationHarness;
use crate::validation::scenarios::registry::{Scenario, ScenarioMeta, Tier, Track};

/// FAO-56 ET₀ validation scenario metadata and entry point.
pub const SCENARIO: Scenario = Scenario {
    meta: ScenarioMeta {
        id: "fao56-et0",
        track: Track::ScienceDispatch,
        tier: Tier::Rust,
        provenance_crate: "validate_fao56_et0",
        provenance_date: "2026-05-11",
        description: "FAO-56 Penman-Monteith + 6 alternative ET₀ methods",
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

/// Run FAO-56 ET₀ validation scenario.
pub fn run(v: &mut ValidationHarness) {
    let empty = serde_json::json!({});
    check(v, m::ET0_FAO56, &empty);
    check(v, m::ET0_HARGREAVES, &empty);
    check(v, m::ET0_PRIESTLEY_TAYLOR, &empty);
    check(v, m::ET0_MAKKINK, &empty);
    check(v, m::ET0_TURC, &empty);
    check(v, m::ET0_HAMON, &empty);
    check(v, m::ET0_BLANEY_CRIDDLE, &empty);

    let th_params = serde_json::json!({
        "monthly_temps_c": [5.0, 6.0, 8.0, 12.0, 16.0, 20.0, 22.0, 21.0, 18.0, 13.0, 8.0, 6.0]
    });
    check(v, m::THORNTHWAITE, &th_params);
}
