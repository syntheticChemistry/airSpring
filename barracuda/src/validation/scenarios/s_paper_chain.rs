// SPDX-License-Identifier: AGPL-3.0-or-later

//! Scenario: Paper chain — full-suite science method validation.
//!
//! Exercises every science and ecology method to validate the complete
//! paper reproduction chain across all domain areas.

use crate::methods as m;
use crate::primal_science::dispatch_science;
use crate::validation::ValidationHarness;
use crate::validation::scenarios::registry::{Scenario, ScenarioMeta, Tier, Track};

/// Complete paper chain scenario metadata and entry point.
pub const SCENARIO: Scenario = Scenario {
    meta: ScenarioMeta {
        id: "paper-chain",
        track: Track::ScienceDispatch,
        tier: Tier::Rust,
        provenance_crate: "validate_paper_chain",
        provenance_date: "2026-05-11",
        description: "Complete paper chain: all science + ecology methods",
    },
    run,
};

fn check(v: &mut ValidationHarness, method: &str, params: &serde_json::Value) {
    let result = dispatch_science(method, params);
    v.check_bool(&format!("{method}: dispatches"), result.is_some());
    if let Some(ref r) = result {
        v.check_bool(&format!("{method}: no error"), r.get("error").is_none());
    }
}

/// Run complete paper chain validation scenario.
pub fn run(v: &mut ValidationHarness) {
    let empty = serde_json::json!({});

    for method in [
        m::ET0_FAO56,
        m::ET0_HARGREAVES,
        m::ET0_PRIESTLEY_TAYLOR,
        m::ET0_MAKKINK,
        m::ET0_TURC,
        m::ET0_HAMON,
        m::ET0_BLANEY_CRIDDLE,
        m::WATER_BALANCE,
        m::YIELD_RESPONSE,
        m::RICHARDS_1D,
        m::SCS_CN_RUNOFF,
        m::GREEN_AMPT,
        m::SOIL_MOISTURE_TOPP,
        m::PEDOTRANSFER,
        m::DUAL_KC,
        m::SENSOR_CALIBRATION,
        m::GDD,
        m::ANDERSON_COUPLING,
    ] {
        check(v, method, &empty);
    }

    let bio_params = serde_json::json!({"counts": [10.0, 5.0, 3.0, 2.0]});
    check(v, m::SHANNON_DIVERSITY, &bio_params);
    let bc_params = serde_json::json!({"sample_a": [1.0, 2.0, 3.0], "sample_b": [2.0, 3.0, 4.0]});
    check(v, m::BRAY_CURTIS, &bc_params);

    let th = serde_json::json!({
        "monthly_temps_c": [5.0, 6.0, 8.0, 12.0, 16.0, 20.0, 22.0, 21.0, 18.0, 13.0, 8.0, 6.0]
    });
    check(v, m::THORNTHWAITE, &th);

    let precip: Vec<f64> = vec![50.0; 24];
    check(
        v,
        m::SPI_DROUGHT_INDEX,
        &serde_json::json!({"monthly_precip_mm": precip}),
    );
    check(
        v,
        m::AUTOCORRELATION,
        &serde_json::json!({"data": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], "max_lag": 5}),
    );
    check(
        v,
        m::GAMMA_CDF,
        &serde_json::json!({"x": 1.0, "alpha": 2.0, "beta": 1.0}),
    );

    check(v, m::ECO_FULL_PIPELINE, &empty);
}
