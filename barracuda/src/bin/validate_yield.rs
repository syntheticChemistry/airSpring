// SPDX-License-Identifier: AGPL-3.0-or-later
#![warn(clippy::pedantic)]
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
//! Validate yield response model against FAO-56 Table 24 and Stewart (1977).
//!
//! Benchmark source: `control/yield_response/benchmark_yield_response.json`
//! Provenance: analytical — FAO-56 Ch. 10, Stewart (1977).
//!
//! Tests:
//! 1. Ky values match FAO-56 Table 24
//! 2. Single-stage Stewart equation against analytical solutions
//! 3. Multi-stage product formula against analytical solutions
//! 4. Water use efficiency (WUE) calculations
//! 5. Scheduling strategy comparison (corn, Michigan synthetic weather)

use airspring_barracuda::eco::water_balance;
use airspring_barracuda::eco::yield_response::{
    clamp_yield_ratio, ky_table, water_use_efficiency, yield_ratio_multistage, yield_ratio_single,
};
use airspring_barracuda::validation::{self, json_str, parse_benchmark_json, ValidationHarness};

const BENCHMARK_JSON: &str =
    include_str!("../../../control/yield_response/benchmark_yield_response.json");

fn f64_field(v: &serde_json::Value, key: &str) -> f64 {
    v[key]
        .as_f64()
        .unwrap_or_else(|| panic!("missing f64 key '{key}'"))
}

fn validate_ky_table(v: &mut ValidationHarness, benchmark: &serde_json::Value) {
    validation::section("Ky Table Values (FAO-56 Table 24)");

    let test_cases = benchmark["validation_checks"]["ky_table_values"]["test_cases"]
        .as_array()
        .expect("test_cases must be array");

    for tc in test_cases {
        let crop = json_str(tc, "crop");
        let expected = f64_field(tc, "ky_total");
        let tol = f64_field(tc, "tolerance");

        let ky = ky_table(crop).expect("crop must be in table");
        v.check_abs(&format!("Ky({crop})"), ky.ky_total, expected, tol);
    }
}

fn validate_single_stage(v: &mut ValidationHarness, benchmark: &serde_json::Value) {
    validation::section("Single-Stage Yield Response (Stewart 1977)");

    let test_cases = benchmark["validation_checks"]["single_stage_analytical"]["test_cases"]
        .as_array()
        .expect("test_cases must be array");

    for tc in test_cases {
        let label = json_str(tc, "label");
        let ky = f64_field(tc, "ky");
        let eta_etc = f64_field(tc, "eta_etc");
        let expected = f64_field(tc, "expected_ratio");
        let tol = f64_field(tc, "tolerance");

        let computed = yield_ratio_single(ky, eta_etc);
        v.check_abs(label, computed, expected, tol);
    }
}

fn validate_multi_stage(v: &mut ValidationHarness, benchmark: &serde_json::Value) {
    validation::section("Multi-Stage Yield Response (FAO-56 Eq. 90)");

    let test_cases = benchmark["validation_checks"]["multi_stage_analytical"]["test_cases"]
        .as_array()
        .expect("test_cases must be array");

    for tc in test_cases {
        let label = json_str(tc, "label");
        let stages_ky: Vec<f64> = tc["stages_ky"]
            .as_array()
            .expect("stages_ky must be array")
            .iter()
            .map(|v| v.as_f64().expect("ky must be f64"))
            .collect();
        let stages_eta_etc: Vec<f64> = tc["stages_eta_etc"]
            .as_array()
            .expect("stages_eta_etc must be array")
            .iter()
            .map(|v| v.as_f64().expect("eta_etc must be f64"))
            .collect();
        let expected = f64_field(tc, "expected_ratio");
        let tol = f64_field(tc, "tolerance");

        let stages: Vec<(f64, f64)> = stages_ky
            .iter()
            .zip(stages_eta_etc.iter())
            .map(|(&k, &e)| (k, e))
            .collect();

        let computed = yield_ratio_multistage(&stages).expect("multistage must succeed");
        v.check_abs(label, computed, expected, tol);
    }
}

fn validate_wue(v: &mut ValidationHarness, benchmark: &serde_json::Value) {
    validation::section("Water Use Efficiency");

    let test_cases = benchmark["validation_checks"]["water_use_efficiency"]["test_cases"]
        .as_array()
        .expect("test_cases must be array");

    for tc in test_cases {
        let label = json_str(tc, "label");
        let yield_kg = f64_field(tc, "yield_kg_ha");
        let eta_mm = f64_field(tc, "eta_mm");
        let expected = f64_field(tc, "expected_wue_kg_m3");
        let tol = f64_field(tc, "tolerance");

        let computed = water_use_efficiency(yield_kg, eta_mm).expect("wue must succeed");
        v.check_abs(label, computed, expected, tol);
    }
}

fn validate_scheduling(v: &mut ValidationHarness, benchmark: &serde_json::Value) {
    validation::section("Scheduling Strategy Comparison (Corn, Michigan)");

    let scenario = &benchmark["validation_checks"]["scheduling_comparison"]["scenario"];
    let strategies = &benchmark["validation_checks"]["scheduling_comparison"]["strategies"];

    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "season_days from JSON f64 is a non-negative integer"
    )]
    let season_days = f64_field(scenario, "season_days") as usize;
    let ky_total = f64_field(scenario, "ky_total");
    let theta_fc = f64_field(scenario, "theta_fc");
    let theta_wp = f64_field(scenario, "theta_wp");
    let root_depth_m = f64_field(scenario, "root_depth_m");
    let p = f64_field(scenario, "p");
    let et0_mean = f64_field(scenario, "et0_mean_mm_day");

    let root_depth_mm = root_depth_m * 1000.0;
    let taw = water_balance::total_available_water(theta_fc, theta_wp, root_depth_mm);
    let raw = water_balance::readily_available_water(taw, p);

    let (et0_daily, precip_daily) = generate_synthetic_weather(season_days, et0_mean);

    let strategy_names = ["no_irrigation", "threshold_50pct", "threshold_mad"];
    let mut yield_ratios: Vec<f64> = Vec::new();
    let mut stress_counts: Vec<usize> = Vec::new();

    for &name in &strategy_names {
        let strat = &strategies[name];

        let thresh_frac: Option<f64> = strat
            .get("irrigation_threshold_frac")
            .and_then(serde_json::Value::as_f64);
        let irrig_depth: f64 = strat
            .get("irrigation_depth_mm")
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(25.0);

        let sim = SimConfig {
            et0: &et0_daily,
            precip: &precip_daily,
            kc: 1.2,
            taw,
            raw,
            thresh_frac,
            irrig_depth,
        };
        let (actual_et, potential_et, stress_days) = simulate_season(season_days, &sim);

        let eta_etc_ratio = if potential_et > 0.0 {
            actual_et / potential_et
        } else {
            1.0
        };
        let yr = clamp_yield_ratio(yield_ratio_single(ky_total, eta_etc_ratio));

        yield_ratios.push(yr);
        stress_counts.push(stress_days);

        let yr_range = strat["expected_yield_ratio_range"]
            .as_array()
            .expect("range array");
        let yr_lo = yr_range[0].as_f64().unwrap();
        let yr_hi = yr_range[1].as_f64().unwrap();

        let sd_range = strat["expected_stress_days_range"]
            .as_array()
            .expect("range array");
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "yield range bounds from JSON f64 are non-negative integers"
        )]
        let sd_lo = sd_range[0].as_f64().unwrap() as usize;
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "yield range upper bound from JSON f64 is a non-negative integer"
        )]
        let sd_hi = sd_range[1].as_f64().unwrap() as usize;

        v.check_bool(
            &format!("{name}_yield_ratio in [{yr_lo}, {yr_hi}]"),
            yr >= yr_lo && yr <= yr_hi,
        );
        v.check_bool(
            &format!("{name}_stress_days in [{sd_lo}, {sd_hi}]"),
            stress_days >= sd_lo && stress_days <= sd_hi,
        );
    }

    v.check_bool(
        "irrigation_improves_yield (MAD > rainfed)",
        yield_ratios[2] > yield_ratios[0],
    );
    v.check_bool(
        "irrigation_reduces_stress (MAD < rainfed)",
        stress_counts[2] < stress_counts[0],
    );
}

/// Deterministic Michigan-like weather: sinusoidal ET₀ + periodic rain.
/// Avoids RNG mismatch between Python (numpy MT19937) and Rust.
fn generate_synthetic_weather(n: usize, et0_mean: f64) -> (Vec<f64>, Vec<f64>) {
    let mut et0 = Vec::with_capacity(n);
    let mut precip = Vec::with_capacity(n);

    for day in 0..n {
        let t = day as f64 / n as f64;
        let seasonal = (std::f64::consts::PI * t).sin();
        et0.push(2.0f64.mul_add(seasonal, et0_mean).max(0.5));

        if day % 3 == 0 {
            precip.push(8.0);
        } else {
            precip.push(0.0);
        }
    }

    (et0, precip)
}

struct SimConfig<'a> {
    et0: &'a [f64],
    precip: &'a [f64],
    kc: f64,
    taw: f64,
    raw: f64,
    thresh_frac: Option<f64>,
    irrig_depth: f64,
}

fn simulate_season(n: usize, cfg: &SimConfig<'_>) -> (f64, f64, usize) {
    let mut dr: f64 = 0.0;
    let mut actual_et_sum: f64 = 0.0;
    let mut potential_et_sum: f64 = 0.0;
    let mut stress_days: usize = 0;

    for i in 0..n {
        let ks = water_balance::stress_coefficient(dr, cfg.taw, cfg.raw);
        if ks < 1.0 {
            stress_days += 1;
        }

        let etc = cfg.kc * cfg.et0[i];
        let eta = ks * etc;

        let irrig = cfg.thresh_frac.map_or(0.0, |frac| {
            if dr > frac * cfg.taw {
                cfg.irrig_depth
            } else {
                0.0
            }
        });

        dr = (dr - cfg.precip[i] - irrig + eta).clamp(0.0, cfg.taw);
        actual_et_sum += eta;
        potential_et_sum += etc;
    }

    (actual_et_sum, potential_et_sum, stress_days)
}

fn main() {
    validation::init_tracing();
    validation::banner("Yield Response Validation (FAO-56 Ch. 10 / Stewart 1977)");
    let mut v = ValidationHarness::new("Yield Response Validation");
    let benchmark =
        parse_benchmark_json(BENCHMARK_JSON).expect("benchmark_yield_response.json must parse");

    validate_ky_table(&mut v, &benchmark);
    validate_single_stage(&mut v, &benchmark);
    validate_multi_stage(&mut v, &benchmark);
    validate_wue(&mut v, &benchmark);
    validate_scheduling(&mut v, &benchmark);

    v.finish();
}
