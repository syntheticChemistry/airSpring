// SPDX-License-Identifier: AGPL-3.0-or-later
#![allow(clippy::similar_names)]
//! Exp 081: Standardized Precipitation Index (SPI) Validation.
//!
//! Validates the SPI drought classification algorithm (`McKee` et al. 1993)
//! against Python baselines. Tests gamma distribution fitting, SPI computation
//! at multiple time scales (1, 3, 6, 12 months), WMO drought classification,
//! and statistical properties (mean ≈ 0, std ≈ 1 for standard normal).
//!
//! Benchmark: `control/drought_index/benchmark_drought_index.json` (17/17 Python PASS)
//! Baseline: `control/drought_index/drought_index_spi.py`
//!
//! References:
//! - `McKee` TB et al. (1993) Drought frequency and duration to time scales
//! - Edwards DC, `McKee` TB (1997) Characteristics of 20th century drought
//! - WMO (2012) SPI User Guide. WMO-No. 1090
//!
//! Provenance: script=`control/drought_index/drought_index_spi.py`, commit=1c11763, date=2026-03-07

use airspring_barracuda::eco::drought_index::{DroughtClass, compute_spi, gamma_mle_fit};
use airspring_barracuda::tolerances;
use airspring_barracuda::validation::{
    self, ValidationHarness, json_f64_required, parse_benchmark_json,
};

const BENCHMARK_JSON: &str =
    include_str!("../../../control/drought_index/benchmark_drought_index.json");

fn load_precip(benchmark: &serde_json::Value) -> Vec<f64> {
    benchmark["monthly_precip_mm"]
        .as_array()
        .expect("monthly_precip_mm must be array")
        .iter()
        .map(|v| v.as_f64().expect("f64"))
        .collect()
}

fn load_spi_values(benchmark: &serde_json::Value, key: &str) -> Vec<f64> {
    benchmark[key]["values"]
        .as_array()
        .expect("values array")
        .iter()
        .map(|v| {
            if v.is_null() {
                f64::NAN
            } else {
                v.as_f64().unwrap_or(f64::NAN)
            }
        })
        .collect()
}

fn spi_stats(spi: &[f64]) -> (usize, f64, f64, f64, f64) {
    let valid: Vec<f64> = spi.iter().copied().filter(|x| x.is_finite()).collect();
    let n = valid.len();
    if n == 0 {
        return (0, 0.0, 0.0, 0.0, 0.0);
    }
    #[allow(clippy::cast_precision_loss)]
    let nf = n as f64;
    let mean_val = valid.iter().sum::<f64>() / nf;
    let var_val = valid.iter().map(|x| (x - mean_val).powi(2)).sum::<f64>() / nf;
    let std_val = var_val.sqrt();
    let min_val = valid.iter().copied().fold(f64::INFINITY, f64::min);
    let max_val = valid.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    (n, mean_val, std_val, min_val, max_val)
}

fn validate_gamma_fit(v: &mut ValidationHarness, benchmark: &serde_json::Value) {
    validation::section("Gamma MLE fit");

    let known: Vec<f64> = benchmark["gamma_fit_known"]["data"]
        .as_array()
        .expect("data array")
        .iter()
        .map(|x| x.as_f64().expect("f64"))
        .collect();

    let py_alpha = json_f64_required(benchmark, &["gamma_fit_known", "alpha"]);
    let py_beta = json_f64_required(benchmark, &["gamma_fit_known", "beta"]);

    let params = gamma_mle_fit(&known).expect("gamma fit should succeed");

    v.check_abs(
        "alpha matches Python",
        params.alpha,
        py_alpha,
        tolerances::MC_ET0_PROPAGATION.abs_tol,
    );

    v.check_abs(
        "beta matches Python",
        params.beta,
        py_beta,
        tolerances::MC_ET0_PROPAGATION.abs_tol,
    );

    #[allow(clippy::cast_precision_loss)]
    let data_mean = known.iter().sum::<f64>() / known.len() as f64;
    v.check_abs(
        "alpha*beta ≈ data mean",
        params.alpha * params.beta,
        data_mean,
        0.1,
    );

    let pos_alpha = if params.alpha > 0.0 { 1.0 } else { 0.0 };
    let pos_beta = if params.beta > 0.0 { 1.0 } else { 0.0 };
    v.check_abs("alpha > 0", pos_alpha, 1.0, f64::EPSILON);
    v.check_abs("beta > 0", pos_beta, 1.0, f64::EPSILON);

    println!("  Rust: alpha={:.6}, beta={:.6}", params.alpha, params.beta);
}

fn validate_spi1(v: &mut ValidationHarness, benchmark: &serde_json::Value) {
    validation::section("SPI-1 (monthly)");

    let precip = load_precip(benchmark);
    let spi1 = compute_spi(&precip, 1);
    let py_spi1 = load_spi_values(benchmark, "spi1");

    let (n_valid, mean_val, std_val, min_val, max_val) = spi_stats(&spi1);
    let py_n_valid = json_f64_required(benchmark, &["spi1", "n_valid"]) as usize;
    let py_mean = json_f64_required(benchmark, &["spi1", "mean"]);
    let py_std = json_f64_required(benchmark, &["spi1", "std"]);

    #[allow(clippy::cast_precision_loss)]
    let n_f64 = n_valid as f64;
    #[allow(clippy::cast_precision_loss)]
    let py_n_f64 = py_n_valid as f64;
    v.check_abs("all 60 months valid", n_f64, py_n_f64, f64::EPSILON);

    v.check_abs(
        "mean near 0",
        mean_val,
        py_mean,
        tolerances::MC_ET0_PROPAGATION.abs_tol,
    );

    v.check_abs("std near Python", std_val, py_std, 0.1);

    let in_range = if min_val > -4.0 && max_val < 4.0 {
        1.0
    } else {
        0.0
    };
    v.check_abs("all SPI-1 in (-4, 4)", in_range, 1.0, f64::EPSILON);

    // Spot-check first few values against Python
    let mut matched = 0;
    let mut compared = 0;
    for (i, (&rust_val, &py_val)) in spi1.iter().zip(py_spi1.iter()).enumerate() {
        if rust_val.is_finite() && py_val.is_finite() {
            compared += 1;
            if (rust_val - py_val).abs() < 0.1 {
                matched += 1;
            } else {
                println!("  SPI-1[{i}]: Rust={rust_val:.4} vs Py={py_val:.4}");
            }
        }
    }

    #[allow(clippy::cast_precision_loss)]
    let match_ratio = if compared > 0 {
        f64::from(matched) / f64::from(compared)
    } else {
        0.0
    };
    v.check_abs("≥90% values within 0.1 of Python", match_ratio, 1.0, 0.1);

    println!(
        "  Rust: n={n_valid}, mean={mean_val:.4}, std={std_val:.4}, range=[{min_val:.4}, {max_val:.4}]"
    );
}

fn validate_spi3(v: &mut ValidationHarness, benchmark: &serde_json::Value) {
    validation::section("SPI-3 (quarterly)");

    let precip = load_precip(benchmark);
    let spi3 = compute_spi(&precip, 3);
    let (n_valid, mean_val, _, _, _) = spi_stats(&spi3);

    let py_n = json_f64_required(benchmark, &["spi3", "n_valid"]) as usize;
    let py_mean = json_f64_required(benchmark, &["spi3", "mean"]);

    #[allow(clippy::cast_precision_loss)]
    v.check_abs("58 months valid", n_valid as f64, py_n as f64, f64::EPSILON);
    v.check_abs(
        "mean near 0",
        mean_val,
        py_mean,
        tolerances::MC_ET0_PROPAGATION.abs_tol,
    );

    let nan_prefix = spi3[0].is_nan() && spi3[1].is_nan();
    let prefix_ok = if nan_prefix { 1.0 } else { 0.0 };
    v.check_abs("first 2 months NaN", prefix_ok, 1.0, f64::EPSILON);

    println!("  Rust: n={n_valid}, mean={mean_val:.4}");
}

fn validate_spi6_12(v: &mut ValidationHarness, benchmark: &serde_json::Value) {
    validation::section("SPI-6 and SPI-12");

    let precip = load_precip(benchmark);

    let spi6 = compute_spi(&precip, 6);
    let (n6, _, _, _, _) = spi_stats(&spi6);
    let py_n6 = json_f64_required(benchmark, &["spi6", "n_valid"]) as usize;
    #[allow(clippy::cast_precision_loss)]
    v.check_abs("SPI-6: 55 valid", n6 as f64, py_n6 as f64, f64::EPSILON);

    let spi12 = compute_spi(&precip, 12);
    let (n12, _, _, _, _) = spi_stats(&spi12);
    let py_n12 = json_f64_required(benchmark, &["spi12", "n_valid"]) as usize;
    #[allow(clippy::cast_precision_loss)]
    v.check_abs("SPI-12: 49 valid", n12 as f64, py_n12 as f64, f64::EPSILON);

    let first_11_nan = spi12.iter().take(11).all(|x| x.is_nan());
    let nan_ok = if first_11_nan { 1.0 } else { 0.0 };
    v.check_abs("SPI-12: first 11 NaN", nan_ok, 1.0, f64::EPSILON);

    println!("  Rust: SPI-6 n={n6}, SPI-12 n={n12}");
}

fn validate_classification(v: &mut ValidationHarness, benchmark: &serde_json::Value) {
    validation::section("WMO drought classification");

    let precip = load_precip(benchmark);
    let spi1 = compute_spi(&precip, 1);
    let valid: Vec<f64> = spi1.iter().copied().filter(|x| x.is_finite()).collect();

    let mut near_normal = 0_u32;
    let mut total_classes = 0_u32;
    for &val in &valid {
        let class = DroughtClass::from_spi(val);
        if class == DroughtClass::NearNormal {
            near_normal += 1;
        }
        total_classes += 1;
    }

    let nn_frac = f64::from(near_normal) / f64::from(total_classes);
    let dominant = if nn_frac > 0.4 { 1.0 } else { 0.0 };
    v.check_abs("near_normal is dominant class", dominant, 1.0, f64::EPSILON);

    let has_variety = if total_classes > 0 {
        let classes: std::collections::HashSet<&str> = valid
            .iter()
            .map(|&s| DroughtClass::from_spi(s).label())
            .collect();
        if classes.len() >= 2 { 1.0 } else { 0.0 }
    } else {
        0.0
    };
    v.check_abs("≥2 classes present", has_variety, 1.0, f64::EPSILON);

    println!(
        "  Rust: near_normal={near_normal}/{total_classes} ({:.0}%)",
        nn_frac * 100.0
    );
}

fn validate_scale_ordering(v: &mut ValidationHarness, benchmark: &serde_json::Value) {
    validation::section("Scale ordering (SPI-3 smoother than SPI-1)");

    let precip = load_precip(benchmark);

    let spi1 = compute_spi(&precip, 1);
    let spi3 = compute_spi(&precip, 3);
    let (_, _, std1, _, _) = spi_stats(&spi1);
    let (_, _, std3, _, _) = spi_stats(&spi3);

    let smoother = if std3 <= std1 * 1.2 { 1.0 } else { 0.0 };
    v.check_abs("SPI-3 std ≤ 1.2× SPI-1 std", smoother, 1.0, f64::EPSILON);

    // All SPI values should be bounded
    let all_valid: Vec<f64> = spi1
        .iter()
        .chain(spi3.iter())
        .copied()
        .filter(|x| x.is_finite())
        .collect();
    let bounded = if all_valid.iter().all(|&v| (-4.0..=4.0).contains(&v)) {
        1.0
    } else {
        0.0
    };
    v.check_abs("all SPI in [-4, 4]", bounded, 1.0, f64::EPSILON);

    println!("  Rust: std_spi1={std1:.4}, std_spi3={std3:.4}");
}

fn main() {
    validation::init_tracing();
    validation::banner("Exp 081: Standardized Precipitation Index (SPI)");

    let mut v = ValidationHarness::new("Drought Index (SPI)");
    let benchmark =
        parse_benchmark_json(BENCHMARK_JSON).expect("benchmark_drought_index.json must parse");

    validate_gamma_fit(&mut v, &benchmark);
    validate_spi1(&mut v, &benchmark);
    validate_spi3(&mut v, &benchmark);
    validate_spi6_12(&mut v, &benchmark);
    validate_classification(&mut v, &benchmark);
    validate_scale_ordering(&mut v, &benchmark);

    v.finish();
}
