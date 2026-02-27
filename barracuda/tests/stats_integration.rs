// SPDX-License-Identifier: AGPL-3.0-or-later
#![allow(clippy::cast_precision_loss)]
//! Statistical primitives integration tests for airSpring `BarraCuda`.
//!
//! Cross-validates `airSpring` testutil and `csv_ts::column_stats` against
//! `barracuda::stats` shared primitives. Also tests RMSE, MBE, R²,
//! Nash-Sutcliffe, Spearman, bootstrap, and variance.

use airspring_barracuda::eco::evapotranspiration as et;
use airspring_barracuda::testutil;

// ── BarraCuda primitive cross-validation ─────────────────────────────

#[test]
fn test_barracuda_pearson_cross_validation() {
    let temps: Vec<f64> = (0..50).map(|i| f64::from(i).mul_add(0.6, 5.0)).collect();
    let es_values: Vec<f64> = temps
        .iter()
        .map(|&t| et::saturation_vapour_pressure(t))
        .collect();
    let delta_values: Vec<f64> = temps
        .iter()
        .map(|&t| et::vapour_pressure_slope(t))
        .collect();

    let r = barracuda::stats::pearson_correlation(&es_values, &delta_values).unwrap();
    assert!(r > 0.99, "es and Δ should be highly correlated: R = {r}");

    let r2 = testutil::r_squared(&es_values, &delta_values).unwrap();
    assert!(r2 > 0.98, "R² should be > 0.98: {r2}");
}

#[test]
fn test_barracuda_stats_vs_airspring_stats() {
    // Cross-validate: airSpring's column_stats (population, n divisor)
    // vs barracuda::stats (sample, n-1 divisor).
    let data = testutil::generate_synthetic_iot_data(168);
    let temps = data.column("temperature").unwrap();

    let our_stats = data.column_stats("temperature").unwrap();
    let bc_std = barracuda::stats::correlation::std_dev(temps).unwrap();

    let ratio = our_stats.std_dev / bc_std;
    let expected_ratio = ((temps.len() - 1) as f64 / temps.len() as f64).sqrt();
    assert!(
        (ratio - expected_ratio).abs() < 0.001,
        "Population vs sample std ratio: {ratio} (expected {expected_ratio})"
    );
}

// ── testutil functions ──────────────────────────────────────────────

#[test]
fn test_testutil_rmse_and_mbe() {
    let observed = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let simulated = vec![1.1, 2.0, 2.9, 4.1, 5.0];

    let rmse_val = testutil::rmse(&observed, &simulated);
    assert!(rmse_val < 0.1, "RMSE: {rmse_val}");

    let mbe_val = testutil::mbe(&observed, &simulated);
    assert!(mbe_val.abs() < 0.05, "MBE: {mbe_val}");
}

#[test]
fn test_testutil_perfect_correlation() {
    let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let b = vec![2.0, 4.0, 6.0, 8.0, 10.0];
    let r2 = testutil::r_squared(&a, &b).unwrap();
    assert!((r2 - 1.0).abs() < 1e-10, "R²: {r2}");
}

// ── Index of Agreement & Nash-Sutcliffe ─────────────────────────────

#[test]
fn test_index_of_agreement_perfect() {
    let obs = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let ia = testutil::index_of_agreement(&obs, &obs).unwrap();
    assert!((ia - 1.0).abs() < 1e-10, "IA (perfect) = {ia}");
}

#[test]
fn test_index_of_agreement_constant_bias() {
    let obs = vec![0.10, 0.15, 0.20, 0.25, 0.30];
    let pred: Vec<f64> = obs.iter().map(|&x| x + 0.02).collect();
    let ia = testutil::index_of_agreement(&obs, &pred).unwrap();
    assert!(ia > 0.95, "IA (constant +0.02 bias) = {ia}");
}

#[test]
fn test_index_of_agreement_matches_python() {
    let measured = vec![0.10, 0.15, 0.20, 0.25, 0.30];
    let predicted = vec![0.10, 0.15, 0.20, 0.25, 0.30];
    let ia = testutil::index_of_agreement(&measured, &predicted).unwrap();
    assert!((ia - 1.0).abs() < 1e-10);

    let mbe_val = testutil::mbe(&measured, &predicted);
    assert!(mbe_val.abs() < 1e-10);
}

#[test]
fn test_nash_sutcliffe_perfect() {
    let obs = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let nse = testutil::nash_sutcliffe(&obs, &obs).unwrap();
    assert!((nse - 1.0).abs() < 1e-10, "NSE (perfect) = {nse}");
}

#[test]
fn test_nash_sutcliffe_mean_predictor() {
    let obs = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let mean_val = 3.0;
    let pred = vec![mean_val; 5];
    let nse = testutil::nash_sutcliffe(&obs, &pred).unwrap();
    assert!(nse.abs() < 1e-10, "NSE (mean predictor) = {nse}");
}

#[test]
fn test_coefficient_of_determination_equals_nse() {
    let obs = vec![1.0, 3.0, 5.0, 7.0, 9.0];
    let pred = vec![1.2, 2.8, 5.1, 6.9, 9.2];
    let r2 = testutil::coefficient_of_determination(&obs, &pred).unwrap();
    let nse = testutil::nash_sutcliffe(&obs, &pred).unwrap();
    assert!(
        (r2 - nse).abs() < f64::EPSILON,
        "R² and NSE should be identical: R²={r2}, NSE={nse}"
    );
}

// ── Spearman & advanced statistics ──────────────────────────────────

#[test]
fn test_spearman_r_perfect_monotonic() {
    let x: Vec<f64> = (1..=20).map(f64::from).collect();
    let y: Vec<f64> = x.iter().map(|&v| v * v).collect();
    let rho = testutil::spearman_r(&x, &y).unwrap();
    assert!(
        (rho - 1.0).abs() < 1e-10,
        "Perfect monotonic should give ρ=1, got {rho}"
    );
}

#[test]
fn test_spearman_r_inverse_relationship() {
    let x: Vec<f64> = (1..=20).map(f64::from).collect();
    let y: Vec<f64> = x.iter().rev().copied().collect();
    let rho = testutil::spearman_r(&x, &y).unwrap();
    assert!(
        (rho + 1.0).abs() < 1e-10,
        "Perfect inverse should give ρ=-1, got {rho}"
    );
}

#[test]
fn test_spearman_vs_pearson_nonlinear() {
    let x: Vec<f64> = (1..=50).map(|i| f64::from(i) * 0.1).collect();
    let y: Vec<f64> = x.iter().map(|&v| v.powi(3)).collect();

    let rho = testutil::spearman_r(&x, &y).unwrap();
    let r2 = testutil::r_squared(&x, &y).unwrap();

    assert!(rho > 0.99, "Spearman: {rho}");
    assert!(r2 < 1.0, "Pearson R²: {r2}");
}

#[test]
fn test_bootstrap_rmse_confidence_interval() {
    let observed: Vec<f64> = (0..100).map(|i| f64::from(i) * 0.1).collect();
    let simulated: Vec<f64> = observed.iter().map(|&v| v + 0.5).collect();

    let point_rmse = testutil::rmse(&observed, &simulated);
    assert!((point_rmse - 0.5).abs() < 1e-10);

    let (lower, upper) = testutil::bootstrap_rmse(&observed, &simulated, 500, 0.95).unwrap();

    assert!(
        lower <= point_rmse && point_rmse <= upper,
        "CI [{lower:.4}, {upper:.4}] should contain RMSE {point_rmse:.4}"
    );
    assert!(
        (upper - lower) < 0.2,
        "CI width {:.4} too wide for constant bias",
        upper - lower
    );
}

#[test]
fn test_variance_and_std_dev() {
    let data = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
    let var = testutil::variance(&data).unwrap();
    let sd = testutil::std_deviation(&data).unwrap();

    assert!((var - 4.571_428_571_428_571).abs() < 1e-6, "Var: {var}");
    assert!(
        (sd - var.sqrt()).abs() < 1e-12,
        "SD: {sd} vs sqrt(var): {}",
        var.sqrt()
    );
}

#[test]
fn test_barracuda_variance_matches_manual() {
    let temps: Vec<f64> = testutil::generate_synthetic_iot_data(48)
        .column("temperature")
        .unwrap()
        .to_vec();

    let n = temps.len() as f64;
    let mean = temps.iter().sum::<f64>() / n;
    let manual_var = temps.iter().map(|t| (t - mean).powi(2)).sum::<f64>() / (n - 1.0);
    let bc_var = testutil::variance(&temps).unwrap();

    assert!(
        (bc_var - manual_var).abs() < 1e-10,
        "barracuda var {bc_var} vs manual {manual_var}"
    );
}

// ── Edge case stats ──────────────────────────────────────────────────

#[test]
fn test_rmse_identical_vectors() {
    let obs = [1.0, 2.0, 3.0, 4.0, 5.0];
    let rmse = testutil::rmse(&obs, &obs);
    assert!(
        rmse.abs() < f64::EPSILON,
        "identical vectors should have RMSE=0"
    );
}

#[test]
fn test_nash_sutcliffe_perfect_is_one() {
    let obs = [1.0, 2.0, 3.0, 4.0, 5.0];
    let nse = testutil::nash_sutcliffe(&obs, &obs).unwrap();
    assert!(
        (nse - 1.0).abs() < f64::EPSILON,
        "perfect match should give NSE=1.0"
    );
}

#[test]
fn test_index_of_agreement_perfect_is_one() {
    let obs = [1.0, 2.0, 3.0, 4.0, 5.0];
    let ia = testutil::index_of_agreement(&obs, &obs).unwrap();
    assert!(
        (ia - 1.0).abs() < f64::EPSILON,
        "perfect match should give IA=1.0"
    );
}

#[test]
fn test_mbe_zero_for_identical() {
    let obs = [1.0, 2.0, 3.0];
    let mbe = testutil::mbe(&obs, &obs);
    assert!(
        mbe.abs() < f64::EPSILON,
        "identical vectors should have MBE=0"
    );
}
