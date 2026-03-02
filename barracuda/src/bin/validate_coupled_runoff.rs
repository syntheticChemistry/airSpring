// SPDX-License-Identifier: AGPL-3.0-or-later
#![warn(clippy::pedantic)]
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
//! Exp 052: Coupled SCS-CN + Green-Ampt Rainfall Partitioning.
//!
//! Validates the coupling of SCS Curve Number runoff (USDA-SCS 1972) with
//! Green-Ampt infiltration (Green & Ampt 1911) for rainfall partitioning:
//!
//! P = Q (runoff) + F (infiltration) + ΔS (surface storage)
//!
//! Benchmark: `control/coupled_runoff_infiltration/benchmark_coupled_runoff.json`
//! Baseline: `control/coupled_runoff_infiltration/coupled_runoff_infiltration.py` (292/292 PASS)
//!
//! References:
//!   USDA-SCS (1972) NEH-4 Section 4.
//!   Green WH, Ampt GA (1911) J Agr Sci 4(1):1-24.
//!
//! script=`control/coupled_runoff_infiltration/coupled_runoff_infiltration.py`, commit=6be822f, date=2026-02-28
//! Run: `python3 control/coupled_runoff_infiltration/coupled_runoff_infiltration.py`

use airspring_barracuda::eco::infiltration::{cumulative_infiltration, GreenAmptParams};
use airspring_barracuda::eco::runoff::scs_cn_runoff_standard;
use airspring_barracuda::tolerances;
use airspring_barracuda::validation::{self, parse_benchmark_json, ValidationHarness};

const BENCHMARK_JSON: &str =
    include_str!("../../../control/coupled_runoff_infiltration/benchmark_coupled_runoff.json");

struct SoilParams {
    name: &'static str,
    ga: GreenAmptParams,
}

const SOILS: &[SoilParams] = &[
    SoilParams {
        name: "sandy_loam",
        ga: GreenAmptParams {
            ks_cm_hr: 1.09,
            psi_cm: 11.01,
            delta_theta: 0.312,
        },
    },
    SoilParams {
        name: "loam",
        ga: GreenAmptParams {
            ks_cm_hr: 0.34,
            psi_cm: 8.89,
            delta_theta: 0.405,
        },
    },
    SoilParams {
        name: "silt_loam",
        ga: GreenAmptParams {
            ks_cm_hr: 0.65,
            psi_cm: 16.68,
            delta_theta: 0.400,
        },
    },
    SoilParams {
        name: "clay_loam",
        ga: GreenAmptParams {
            ks_cm_hr: 0.10,
            psi_cm: 20.88,
            delta_theta: 0.309,
        },
    },
];

struct Storm {
    name: &'static str,
    precip_mm: f64,
    duration_hr: f64,
}

const STORMS: &[Storm] = &[
    Storm {
        name: "light",
        precip_mm: 15.0,
        duration_hr: 6.0,
    },
    Storm {
        name: "moderate",
        precip_mm: 40.0,
        duration_hr: 4.0,
    },
    Storm {
        name: "heavy",
        precip_mm: 80.0,
        duration_hr: 3.0,
    },
    Storm {
        name: "extreme",
        precip_mm: 150.0,
        duration_hr: 2.0,
    },
];

struct LandUse {
    name: &'static str,
    cn: f64,
}

const LAND_USES: &[LandUse] = &[
    LandUse {
        name: "row_crops_B",
        cn: 81.0,
    },
    LandUse {
        name: "pasture_B",
        cn: 61.0,
    },
    LandUse {
        name: "woods_B",
        cn: 55.0,
    },
];

fn partition_rainfall(
    precip_mm: f64,
    cn: f64,
    ga: &GreenAmptParams,
    duration_hr: f64,
) -> (f64, f64, f64) {
    let q_mm = scs_cn_runoff_standard(precip_mm, cn);
    let p_net_mm = precip_mm - q_mm;
    let cumul_infil = cumulative_infiltration(ga, duration_hr) * 10.0;
    let f_actual_mm = cumul_infil.min(p_net_mm);
    let surface_mm = (p_net_mm - f_actual_mm).max(0.0);
    (q_mm, f_actual_mm, surface_mm)
}

fn validate_storm_matrix(v: &mut ValidationHarness) {
    validation::section("Storm Matrix");
    let tol = tolerances::SCS_CN_ANALYTICAL.abs_tol;

    for storm in STORMS {
        for soil in SOILS {
            for lu in LAND_USES {
                let (q, f, s) =
                    partition_rainfall(storm.precip_mm, lu.cn, &soil.ga, storm.duration_hr);
                let mass_err = (storm.precip_mm - q - f - s).abs();
                let label = format!("{}_{}_{}", storm.name, soil.name, lu.name);
                v.check_abs(&format!("{label}_mass"), mass_err, 0.0, tol);
                v.check_bool(&format!("{label}_q_leq_p"), q <= storm.precip_mm + tol);
                v.check_bool(
                    &format!("{label}_non_neg"),
                    q >= -tol && f >= -tol && s >= -tol,
                );
            }
        }
    }
}

fn validate_benchmark_parity(v: &mut ValidationHarness, benchmark: &serde_json::Value) {
    validation::section("Python Benchmark Parity");
    let tol = tolerances::CROSS_VALIDATION.abs_tol;

    if let Some(matrix) = benchmark.get("storm_matrix").and_then(|v| v.as_array()) {
        for tc in matrix {
            let storm_name = tc["storm"].as_str().unwrap_or("?");
            let soil_name = tc["soil"].as_str().unwrap_or("?");
            let lu_name = tc["land_use"].as_str().unwrap_or("?");
            let cn = tc["cn"].as_f64().unwrap_or(0.0);
            let duration_hr = tc["storm_duration_hr"].as_f64().unwrap_or(4.0);
            let precip = tc["precip_mm"].as_f64().unwrap_or(0.0);

            let soil = SOILS.iter().find(|s| s.name == soil_name);
            let Some(soil) = soil else { continue };

            let (q, f, s) = partition_rainfall(precip, cn, &soil.ga, duration_hr);
            let py_q = tc["runoff_mm"].as_f64().unwrap_or(0.0);
            let py_f = tc["infiltration_mm"].as_f64().unwrap_or(0.0);
            let py_s = tc["surface_storage_mm"].as_f64().unwrap_or(0.0);

            let label = format!("{storm_name}_{soil_name}_{lu_name}");
            v.check_abs(&format!("{label}_Q"), q, py_q, tol);
            v.check_abs(&format!("{label}_F"), f, py_f, tol);
            v.check_abs(&format!("{label}_S"), s, py_s, tol);
        }
    }
}

fn validate_conservation(v: &mut ValidationHarness) {
    validation::section("Mass Conservation Sweep");
    let tol = tolerances::WATER_BALANCE_MASS.abs_tol;

    for &p in &[5.0, 10.0, 25.0, 50.0, 75.0, 100.0, 150.0, 200.0] {
        for &cn in &[55.0, 65.0, 75.0, 85.0, 95.0] {
            for soil in &SOILS[..2] {
                let (q, f, s) = partition_rainfall(p, cn, &soil.ga, 4.0);
                let err = (p - q - f - s).abs();
                v.check_abs(&format!("P{p}_CN{cn}_{}", soil.name), err, 0.0, tol);
                v.check_bool(&format!("P{p}_CN{cn}_{}_q<=p", soil.name), q <= p + tol);
            }
        }
    }
}

fn validate_monotonicity(v: &mut ValidationHarness) {
    validation::section("Monotonicity");

    for &cn in &[65.0, 85.0] {
        let precips = [10.0, 25.0, 50.0, 100.0];
        let qs: Vec<f64> = precips
            .iter()
            .map(|&p| scs_cn_runoff_standard(p, cn))
            .collect();
        v.check_bool(
            &format!("precip_mono_CN{cn}"),
            qs.windows(2).all(|w| w[0] <= w[1]),
        );
    }

    for &storm_mm in &[40.0, 80.0] {
        let sl = &SOILS[0]; // sandy_loam
        let cl = &SOILS[3]; // clay_loam
        let (_, infil_sandy, _) = partition_rainfall(storm_mm, 75.0, &sl.ga, 4.0);
        let (_, infil_clay, _) = partition_rainfall(storm_mm, 75.0, &cl.ga, 4.0);
        v.check_bool(&format!("sandy>clay_P{storm_mm}"), infil_sandy > infil_clay);
    }
}

fn main() {
    let benchmark = parse_benchmark_json(BENCHMARK_JSON).expect("valid JSON");
    let mut v =
        ValidationHarness::new("Exp 052: Coupled SCS-CN + Green-Ampt Rainfall Partitioning");
    validate_storm_matrix(&mut v);
    validate_benchmark_parity(&mut v, &benchmark);
    validate_conservation(&mut v);
    validate_monotonicity(&mut v);
    v.finish();
}
