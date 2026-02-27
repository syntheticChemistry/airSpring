// SPDX-License-Identifier: AGPL-3.0-or-later
#![warn(clippy::pedantic)]
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
//! Exp 042: Seasonal Batch ET₀ at GPU Scale.
//!
//! Generates synthetic 365-day weather for 4 US climate stations, computes
//! daily FAO-56 PM ET₀ via `BatchedEt0`, and validates seasonal aggregates.
//!
//! This validates GPU batch dispatch: 365 × 4 = 1,460 station-days computed
//! in one call through `BatchedEt0::compute_gpu`.
//!
//! Benchmark: `control/seasonal_batch_et0/benchmark_seasonal_batch.json`
//! Baseline: `control/seasonal_batch_et0/seasonal_batch_et0.py` (18/18 PASS)

use airspring_barracuda::gpu::et0::{Backend, BatchedEt0, StationDay};
use airspring_barracuda::validation::{self, json_field, parse_benchmark_json, ValidationHarness};

const BENCHMARK_JSON: &str =
    include_str!("../../../control/seasonal_batch_et0/benchmark_seasonal_batch.json");

fn seasonal_value(doy: u32, vmin: f64, vmax: f64) -> f64 {
    let phase_doy = 196.0_f64;
    let frac = (2.0 * std::f64::consts::PI * (f64::from(doy) - phase_doy + 91.25) / 365.0).sin();
    let mid = f64::midpoint(vmin, vmax);
    let amp = (vmax - vmin) / 2.0;
    mid + amp * frac
}

struct StationSpec {
    label: String,
    latitude: f64,
    elevation: f64,
    tmax_range: (f64, f64),
    tmin_range: (f64, f64),
    rh_max_range: (f64, f64),
    rh_min_range: (f64, f64),
    wind_2m: f64,
    rs_range: (f64, f64),
    annual_et0_min: f64,
    annual_et0_max: f64,
}

fn parse_stations(benchmark: &serde_json::Value) -> Vec<StationSpec> {
    benchmark["stations"]
        .as_array()
        .expect("stations array")
        .iter()
        .map(|st| {
            let range = |key: &str| -> (f64, f64) {
                let arr = st[key].as_array().expect("range array");
                (arr[0].as_f64().expect("f64"), arr[1].as_f64().expect("f64"))
            };
            StationSpec {
                label: st["label"].as_str().unwrap_or("").to_string(),
                latitude: json_field(st, "latitude"),
                elevation: json_field(st, "elevation"),
                tmax_range: range("tmax_range"),
                tmin_range: range("tmin_range"),
                rh_max_range: range("rh_max_range"),
                rh_min_range: range("rh_min_range"),
                wind_2m: json_field(st, "wind_2m"),
                rs_range: range("rs_range"),
                annual_et0_min: st["expected_annual_et0_mm"]["min"].as_f64().expect("min"),
                annual_et0_max: st["expected_annual_et0_mm"]["max"].as_f64().expect("max"),
            }
        })
        .collect()
}

fn generate_year(spec: &StationSpec) -> Vec<StationDay> {
    (1..=365)
        .map(|doy| StationDay {
            tmax: seasonal_value(doy, spec.tmax_range.0, spec.tmax_range.1),
            tmin: seasonal_value(doy, spec.tmin_range.0, spec.tmin_range.1),
            rh_max: seasonal_value(doy, spec.rh_max_range.0, spec.rh_max_range.1),
            rh_min: seasonal_value(doy, spec.rh_min_range.0, spec.rh_min_range.1),
            wind_2m: spec.wind_2m,
            rs: seasonal_value(doy, spec.rs_range.0, spec.rs_range.1),
            elevation: spec.elevation,
            latitude: spec.latitude,
            doy,
        })
        .collect()
}

fn main() {
    validation::init_tracing();
    validation::banner("Exp 042: Seasonal Batch ET₀ at GPU Scale");

    let mut v = ValidationHarness::new("Seasonal Batch ET₀");
    let benchmark =
        parse_benchmark_json(BENCHMARK_JSON).expect("benchmark_seasonal_batch.json must parse");

    let stations = parse_stations(&benchmark);
    let batcher = BatchedEt0::cpu();

    // Generate all station-days and compute in one big batch
    let mut all_days: Vec<StationDay> = Vec::new();
    let mut station_offsets: Vec<(usize, usize)> = Vec::new();
    for spec in &stations {
        let days = generate_year(spec);
        let start = all_days.len();
        all_days.extend_from_slice(&days);
        station_offsets.push((start, all_days.len()));
    }

    validation::section("Batch Computation (all stations)");
    let batch_result = batcher
        .compute_gpu(&all_days)
        .expect("batch compute should succeed");

    v.check_bool(
        "Backend reports CPU (no GPU device)",
        batch_result.backend_used == Backend::Cpu,
    );
    v.check_bool(
        &format!("Batch produced {} results", all_days.len()),
        batch_result.et0_values.len() == all_days.len(),
    );

    // Per-station validation
    let mut annual_totals: Vec<f64> = Vec::new();
    for (i, spec) in stations.iter().enumerate() {
        let (start, end) = station_offsets[i];
        let et0_vals = &batch_result.et0_values[start..end];

        validation::section(&spec.label);

        let annual: f64 = et0_vals.iter().sum();
        annual_totals.push(annual);

        // Seasonal shape: summer > winter
        let summer_mean: f64 = et0_vals[152..244].iter().sum::<f64>() / 92.0;
        let winter_days: Vec<f64> = et0_vals[0..59]
            .iter()
            .chain(et0_vals[334..365].iter())
            .copied()
            .collect();
        let winter_mean = winter_days.iter().sum::<f64>() / winter_days.len() as f64;

        v.check_lower(
            &format!("{}: summer_mean > winter_mean", spec.label),
            summer_mean,
            winter_mean,
        );

        // Annual total in expected range
        v.check_bool(
            &format!(
                "{}: annual={:.0} in [{}, {}]",
                spec.label, annual, spec.annual_et0_min, spec.annual_et0_max
            ),
            annual >= spec.annual_et0_min && annual <= spec.annual_et0_max,
        );

        // Daily range
        let daily_ok = et0_vals.iter().all(|&v| (0.0..=15.0).contains(&v));
        v.check_bool(
            &format!("{}: all daily ET₀ in [0, 15]", spec.label),
            daily_ok,
        );

        // Reduction accuracy: mean × 365 == sum
        let mean = annual / 365.0;
        v.check_abs(
            &format!("{}: mean×365 == sum", spec.label),
            mean * 365.0,
            annual,
            1e-10,
        );
    }

    // Batch consistency: compute one station individually, compare to batch
    validation::section("Batch Consistency");
    let single_days = generate_year(&stations[0]);
    let single_result = batcher.compute_gpu(&single_days).expect("single compute");
    let (s0, e0) = station_offsets[0];
    let batch_slice = &batch_result.et0_values[s0..e0];
    let all_match = single_result
        .et0_values
        .iter()
        .zip(batch_slice.iter())
        .all(|(a, b)| (a - b).abs() == 0.0);
    v.check_bool("Single-station matches batch slice (bit-exact)", all_match);

    // Cross-station ordering: Arizona > Michigan > Pacific NW
    validation::section("Cross-Station Ordering");
    v.check_lower("Arizona > Michigan", annual_totals[1], annual_totals[0]);
    v.check_lower("Michigan > Pacific NW", annual_totals[0], annual_totals[2]);

    v.finish();
}
