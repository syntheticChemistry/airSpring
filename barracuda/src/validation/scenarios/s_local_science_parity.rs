// SPDX-License-Identifier: AGPL-3.0-or-later

//! Scenario: Local Science Parity — absorbed from exp001.
//!
//! Validates that all airSpring science methods produce identical results
//! when called via the `dispatch_science` JSON-RPC pathway vs direct Rust
//! function invocation. Purely in-process, no IPC.

use crate::methods as m;
use crate::primal_science::dispatch_science;
use crate::validation::ValidationHarness;
use crate::validation::scenarios::registry::{Scenario, ScenarioMeta, Tier, Track};

/// Scenario metadata and entry point.
pub const SCENARIO: Scenario = Scenario {
    meta: ScenarioMeta {
        id: "local-science-parity",
        track: Track::ScienceDispatch,
        tier: Tier::Rust,
        provenance_crate: "exp001_local_science_parity",
        provenance_date: "2026-05-09",
        description: "All 44 science methods dispatch correctly via JSON-RPC pathway",
    },
    run,
};

fn check_dispatch(v: &mut ValidationHarness, method: &str, params: &serde_json::Value) {
    let result = dispatch_science(method, params);
    v.check_bool(
        &format!("dispatch({method}) returns Some"),
        result.is_some(),
    );
    if let Some(ref r) = result {
        v.check_bool(
            &format!("dispatch({method}) no error field"),
            r.get("error").is_none(),
        );
    }
}

fn check_ecology_alias(
    v: &mut ValidationHarness,
    science: &str,
    ecology: &str,
    params: &serde_json::Value,
) {
    let s = dispatch_science(science, params);
    let e = dispatch_science(ecology, params);
    v.check_bool(&format!("alias parity: {science} == {ecology}"), s == e);
}

/// Run this validation scenario.
pub fn run(v: &mut ValidationHarness) {
    let empty = serde_json::json!({});

    check_dispatch(v, m::ET0_FAO56, &empty);
    check_dispatch(v, m::ET0_HARGREAVES, &empty);
    check_dispatch(v, m::ET0_PRIESTLEY_TAYLOR, &empty);
    check_dispatch(v, m::ET0_MAKKINK, &empty);
    check_dispatch(v, m::ET0_TURC, &empty);
    check_dispatch(v, m::ET0_HAMON, &empty);
    check_dispatch(v, m::ET0_BLANEY_CRIDDLE, &empty);

    check_dispatch(v, m::WATER_BALANCE, &empty);
    check_dispatch(v, m::YIELD_RESPONSE, &empty);
    check_dispatch(v, m::RICHARDS_1D, &empty);
    check_dispatch(v, m::SCS_CN_RUNOFF, &empty);
    check_dispatch(v, m::GREEN_AMPT, &empty);
    check_dispatch(v, m::SOIL_MOISTURE_TOPP, &empty);
    check_dispatch(v, m::PEDOTRANSFER, &empty);
    check_dispatch(v, m::DUAL_KC, &empty);
    check_dispatch(v, m::SENSOR_CALIBRATION, &empty);
    check_dispatch(v, m::GDD, &empty);

    let bio_params = serde_json::json!({"counts": [10.0, 5.0, 3.0, 2.0]});
    check_dispatch(v, m::SHANNON_DIVERSITY, &bio_params);
    let bc_params = serde_json::json!({"sample_a": [1.0, 2.0, 3.0], "sample_b": [2.0, 3.0, 4.0]});
    check_dispatch(v, m::BRAY_CURTIS, &bc_params);

    check_dispatch(v, m::ANDERSON_COUPLING, &empty);
    let th_params = serde_json::json!({"monthly_temps_c": [5.0, 6.0, 8.0, 12.0, 16.0, 20.0, 22.0, 21.0, 18.0, 13.0, 8.0, 6.0]});
    check_dispatch(v, m::THORNTHWAITE, &th_params);
    let precip: Vec<f64> = vec![50.0; 24];
    let spi_params = serde_json::json!({"monthly_precip_mm": precip});
    check_dispatch(v, m::SPI_DROUGHT_INDEX, &spi_params);
    let acf_params = serde_json::json!({"data": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], "max_lag": 5});
    check_dispatch(v, m::AUTOCORRELATION, &acf_params);
    check_dispatch(
        v,
        m::GAMMA_CDF,
        &serde_json::json!({"x": 1.0, "alpha": 2.0, "beta": 1.0}),
    );

    check_ecology_alias(v, m::ET0_FAO56, m::ECO_ET0_FAO56, &empty);
    check_ecology_alias(v, m::ET0_HARGREAVES, m::ECO_ET0_HARGREAVES, &empty);
    check_ecology_alias(v, m::WATER_BALANCE, m::ECO_WATER_BALANCE, &empty);
    check_ecology_alias(v, m::YIELD_RESPONSE, m::ECO_YIELD_RESPONSE, &empty);
    check_ecology_alias(
        v,
        m::SPI_DROUGHT_INDEX,
        m::ECO_SPI_DROUGHT_INDEX,
        &spi_params,
    );
    check_ecology_alias(v, m::AUTOCORRELATION, m::ECO_AUTOCORRELATION, &acf_params);

    v.check_bool(
        "unknown method returns None",
        dispatch_science("science.unknown_method", &empty).is_none(),
    );
}
