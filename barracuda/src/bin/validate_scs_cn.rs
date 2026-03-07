// SPDX-License-Identifier: AGPL-3.0-or-later
#![warn(clippy::pedantic)]
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
//! Exp 050: SCS Curve Number Runoff Validation.
//!
//! Validates the USDA SCS-CN runoff method against analytical benchmarks,
//! AMC adjustment formulas, and physical constraints.
//!
//! Benchmark: `control/scs_curve_number/benchmark_scs_cn.json`
//! Baseline: `control/scs_curve_number/scs_curve_number.py` (38/38 PASS)
//!
//! Reference: USDA-SCS (1972) NEH-4; USDA-SCS (1986) TR-55.
//!
//! Provenance: script=`control/scs_curve_number/scs_curve_number.py`, commit=97e7533, date=2026-02-28

use airspring_barracuda::eco::runoff::{
    amc_cn_dry, amc_cn_wet, potential_retention, scs_cn_runoff, scs_cn_runoff_standard, LandUse,
    SoilGroup,
};
use airspring_barracuda::tolerances;
use airspring_barracuda::validation::{self, json_field, parse_benchmark_json, ValidationHarness};

const BENCHMARK_JSON: &str =
    include_str!("../../../control/scs_curve_number/benchmark_scs_cn.json");

fn validate_analytical(v: &mut ValidationHarness, benchmark: &serde_json::Value) {
    validation::section("Analytical Benchmarks");
    let checks = &benchmark["analytical_benchmarks"];
    for tc in checks.as_array().expect("array") {
        let name = tc["name"].as_str().unwrap_or("?");
        let cn = json_field(&tc["inputs"], "cn");
        let precip = json_field(&tc["inputs"], "precip_mm");
        let ia_ratio = tc["inputs"]
            .get("ia_ratio")
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(0.2);
        let expected_q = json_field(tc, "expected_Q_mm");
        let tol = json_field(tc, "tolerance");

        let computed = scs_cn_runoff(precip, cn, ia_ratio);
        v.check_abs(&format!("Q({name})"), computed, expected_q, tol);

        if let Some(s_expected) = tc.get("S_mm").and_then(serde_json::Value::as_f64) {
            let s = potential_retention(cn);
            v.check_abs(
                &format!("S({name})"),
                s,
                s_expected,
                tolerances::SCS_CN_ANALYTICAL.abs_tol,
            );
        }
    }
}

fn validate_amc(v: &mut ValidationHarness, benchmark: &serde_json::Value) {
    validation::section("AMC Adjustment");
    let cases = &benchmark["amc_adjustment"]["test_cases"];
    for tc in cases.as_array().expect("array") {
        let cn_ii = json_field(tc, "cn_ii");
        let expected_i = json_field(tc, "expected_cn_i");
        let expected_iii = json_field(tc, "expected_cn_iii");
        let tol = json_field(tc, "tolerance");

        let cn_i = amc_cn_dry(cn_ii);
        let cn_iii = amc_cn_wet(cn_ii);
        v.check_abs(&format!("AMC-I(CN={cn_ii})"), cn_i, expected_i, tol);
        v.check_abs(&format!("AMC-III(CN={cn_ii})"), cn_iii, expected_iii, tol);
    }
}

fn validate_monotonicity(v: &mut ValidationHarness) {
    validation::section("Monotonicity");

    // CN monotonic
    let cns = [30, 50, 65, 75, 85, 90, 95, 98];
    let qs: Vec<f64> = cns
        .iter()
        .map(|&cn| scs_cn_runoff_standard(50.0, f64::from(cn)))
        .collect();
    v.check_bool("cn_monotonic", qs.windows(2).all(|w| w[0] <= w[1]));

    // Precip monotonic
    let ps = [0.0, 10.0, 20.0, 30.0, 50.0, 75.0, 100.0, 150.0];
    let qs: Vec<f64> = ps
        .iter()
        .map(|&p| scs_cn_runoff_standard(p, 75.0))
        .collect();
    v.check_bool("precip_monotonic", qs.windows(2).all(|w| w[0] <= w[1]));

    // Q ≤ P always
    let ok = ps.iter().all(|&p| {
        cns.iter()
            .all(|&cn| scs_cn_runoff_standard(p, f64::from(cn)) <= p + 0.001)
    });
    v.check_bool("Q_leq_P", ok);

    // Higher Ia ratio → less runoff
    let q02 = scs_cn_runoff(50.0, 75.0, 0.2);
    let q05 = scs_cn_runoff(50.0, 75.0, 0.05);
    v.check_bool("ia_ratio_effect", q05 > q02);
}

fn validate_cn_table(v: &mut ValidationHarness) {
    validation::section("CN Table Soil Group Ordering");

    let land_uses = [
        LandUse::FallowBare,
        LandUse::RowCropsStraight,
        LandUse::RowCropsContoured,
        LandUse::SmallGrainStraight,
        LandUse::PastureGood,
        LandUse::Meadow,
        LandUse::WoodsGood,
        LandUse::Farmstead,
        LandUse::Impervious,
    ];
    for lu in land_uses {
        let cn_a = lu.cn(SoilGroup::A);
        let cn_d = lu.cn(SoilGroup::D);
        v.check_bool(&format!("{lu:?}_A<=D"), cn_a <= cn_d);
    }
}

fn validate_edge_cases(v: &mut ValidationHarness) {
    validation::section("Edge Cases");
    let tol = tolerances::SCS_CN_ANALYTICAL.abs_tol;
    v.check_abs("CN100", scs_cn_runoff_standard(50.0, 100.0), 50.0, tol);
    v.check_abs("CN1", scs_cn_runoff_standard(50.0, 1.0), 0.0, tol);
}

fn main() {
    let benchmark = parse_benchmark_json(BENCHMARK_JSON).expect("valid JSON");
    let mut v = ValidationHarness::new("Exp 050: SCS Curve Number Runoff");
    validate_analytical(&mut v, &benchmark);
    validate_amc(&mut v, &benchmark);
    validate_monotonicity(&mut v);
    validate_cn_table(&mut v);
    validate_edge_cases(&mut v);
    v.finish();
}
