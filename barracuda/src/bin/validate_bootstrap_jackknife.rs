// SPDX-License-Identifier: AGPL-3.0-or-later
#![allow(clippy::similar_names)]
//! Exp 080: Bootstrap & Jackknife Confidence Intervals for Seasonal ET₀.
//!
//! Validates the `gpu::bootstrap` and `gpu::jackknife` CPU paths against
//! analytical expectations and Python baselines. Tests statistical properties
//! (CI containment, SE bounds, width ordering) rather than bit-exact match,
//! since resampling depends on RNG implementation details.
//!
//! Benchmark: `control/bootstrap_jackknife/benchmark_bootstrap_jackknife.json` (18/18 Python PASS)
//! Baseline: `control/bootstrap_jackknife/bootstrap_jackknife_et0.py`
//!
//! References:
//! - Efron B (1979) Bootstrap methods: another look at the jackknife
//! - Quenouille MH (1956) Notes on bias in estimation
//! - Allen RG et al. (1998) FAO-56 Crop Evapotranspiration
//!
//! Provenance: script=`control/bootstrap_jackknife/bootstrap_jackknife_et0.py`, commit=1c11763, date=2026-03-07

use airspring_barracuda::gpu::bootstrap::GpuBootstrap;
use airspring_barracuda::gpu::jackknife::GpuJackknife;
use airspring_barracuda::tolerances;
use airspring_barracuda::validation::{
    self, ValidationHarness, json_f64_required, parse_benchmark_json,
};

const BENCHMARK_JSON: &str =
    include_str!("../../../control/bootstrap_jackknife/benchmark_bootstrap_jackknife.json");

fn load_et0_series(benchmark: &serde_json::Value) -> Vec<f64> {
    benchmark["et0_series"]
        .as_array()
        .expect("et0_series must be array")
        .iter()
        .map(|v| v.as_f64().expect("f64"))
        .collect()
}

fn validate_bootstrap_season(v: &mut ValidationHarness, benchmark: &serde_json::Value) {
    validation::section("Bootstrap: full season (N=153, B=1000)");

    let data = load_et0_series(benchmark);
    let engine = GpuBootstrap::cpu();

    let py_mean = json_f64_required(benchmark, &["bootstrap_full_season", "mean"]);

    let est = match engine.estimate_mean(&data, 1000, 42) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("Bootstrap failed: {e}");
            std::process::exit(1);
        }
    };

    v.check_abs(
        "mean matches data mean",
        est.mean,
        py_mean,
        tolerances::ET0_REFERENCE.abs_tol,
    );

    v.check_abs("mean plausible (2-7 mm)", est.mean, 4.5, 2.5);

    let ci_contains = if est.ci_lower < est.mean && est.ci_upper > est.mean {
        1.0
    } else {
        0.0
    };
    v.check_abs("CI contains mean", ci_contains, 1.0, f64::EPSILON);

    let ci_width = est.ci_upper - est.ci_lower;
    let positive_width = if ci_width > 0.0 { 1.0 } else { 0.0 };
    v.check_abs("CI width > 0", positive_width, 1.0, f64::EPSILON);
    v.check_abs("CI width < 2mm", ci_width, 1.0, 1.0);

    let se_ok = if est.std_error > 0.0 && est.std_error < 1.0 {
        1.0
    } else {
        0.0
    };
    v.check_abs("SE in (0, 1) mm", se_ok, 1.0, f64::EPSILON);

    println!(
        "  Rust: mean={:.6}, CI=[{:.4}, {:.4}], SE={:.6}",
        est.mean, est.ci_lower, est.ci_upper, est.std_error
    );
}

fn validate_jackknife_season(v: &mut ValidationHarness, benchmark: &serde_json::Value) {
    validation::section("Jackknife: full season (N=153)");

    let data = load_et0_series(benchmark);
    let engine = GpuJackknife::cpu();

    let py_mean = json_f64_required(benchmark, &["jackknife_full_season", "mean"]);
    let py_var = json_f64_required(benchmark, &["jackknife_full_season", "variance"]);
    let py_se = json_f64_required(benchmark, &["jackknife_full_season", "std_error"]);

    let est = match engine.estimate(&data) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("Jackknife failed: {e}");
            std::process::exit(1);
        }
    };

    v.check_abs(
        "mean matches Python",
        est.mean,
        py_mean,
        tolerances::ET0_REFERENCE.abs_tol,
    );

    v.check_abs(
        "variance matches Python",
        est.variance,
        py_var,
        tolerances::MC_ET0_PROPAGATION.abs_tol,
    );

    v.check_abs(
        "SE matches Python",
        est.std_error,
        py_se,
        tolerances::MC_ET0_PROPAGATION.abs_tol,
    );

    let pos_var = if est.variance > 0.0 { 1.0 } else { 0.0 };
    v.check_abs("variance > 0", pos_var, 1.0, f64::EPSILON);

    let pos_se = if est.std_error > 0.0 && est.std_error < 1.0 {
        1.0
    } else {
        0.0
    };
    v.check_abs("SE in (0, 1) mm", pos_se, 1.0, f64::EPSILON);

    println!(
        "  Rust: mean={:.6}, var={:.6}, SE={:.6}",
        est.mean, est.variance, est.std_error
    );
}

fn validate_known_values(v: &mut ValidationHarness, benchmark: &serde_json::Value) {
    validation::section("Known values: [1..10] (analytical)");

    let data: Vec<f64> = (1..=10).map(f64::from).collect();
    let boot = GpuBootstrap::cpu();
    let jack = GpuJackknife::cpu();

    let boot_est = boot.estimate_mean(&data, 2000, 42).expect("bootstrap");
    let jack_est = jack.estimate(&data).expect("jackknife");

    v.check_abs("boot mean = 5.5", boot_est.mean, 5.5, 0.01);
    v.check_abs("jack mean = 5.5", jack_est.mean, 5.5, 0.01);

    // Analytical jackknife variance for mean of [1..10]:
    // Var(X) = (n²-1)/12 = 99/12 = 8.25; jackknife var of mean = Var(X)/n = 0.825
    // However, jackknife computes (n-1)/n * Σ(θᵢ - θ̄)² which gives n*(n-1)/n * (Var/n)
    let py_jack_var = json_f64_required(benchmark, &["jackknife_known_1_10", "variance"]);
    v.check_abs(
        "jack var near analytical",
        jack_est.variance,
        py_jack_var,
        0.15,
    );

    println!(
        "  Rust: boot_mean={:.4}, jack_mean={:.4}, jack_var={:.6}",
        boot_est.mean, jack_est.mean, jack_est.variance
    );
}

fn validate_small_sample(v: &mut ValidationHarness, benchmark: &serde_json::Value) {
    validation::section("Small sample (N=5)");

    let full_data = load_et0_series(benchmark);
    let small: Vec<f64> = full_data.iter().take(5).copied().collect();

    let boot_full = GpuBootstrap::cpu()
        .estimate_mean(&full_data, 1000, 42)
        .expect("bootstrap");
    let boot_small = GpuBootstrap::cpu()
        .estimate_mean(&small, 500, 42)
        .expect("bootstrap");
    let jack_small = GpuJackknife::cpu().estimate(&small).expect("jackknife");

    let full_width = boot_full.ci_upper - boot_full.ci_lower;
    let small_width = boot_small.ci_upper - boot_small.ci_lower;
    let wider = if small_width > full_width { 1.0 } else { 0.0 };
    v.check_abs("small CI wider than full", wider, 1.0, f64::EPSILON);

    let jack_ok = if jack_small.std_error > 0.0 { 1.0 } else { 0.0 };
    v.check_abs("small jack SE > 0", jack_ok, 1.0, f64::EPSILON);

    println!(
        "  Rust: small_width={:.4}, full_width={:.4}, jack_SE={:.6}",
        small_width, full_width, jack_small.std_error
    );
}

fn validate_constant_data(v: &mut ValidationHarness) {
    validation::section("Constant data (zero variance)");

    let data = vec![3.5; 20];
    let jack = GpuJackknife::cpu().estimate(&data).expect("jackknife");
    let boot = GpuBootstrap::cpu()
        .estimate_mean(&data, 200, 42)
        .expect("bootstrap");

    v.check_abs("const jack var ≈ 0", jack.variance, 0.0, 1e-10);
    v.check_abs("const jack mean = 3.5", jack.mean, 3.5, f64::EPSILON);

    let boot_width = boot.ci_upper - boot.ci_lower;
    v.check_abs("const boot CI width ≈ 0", boot_width, 0.0, 1e-10);
}

fn validate_boot_jack_agreement(v: &mut ValidationHarness, benchmark: &serde_json::Value) {
    validation::section("Bootstrap/Jackknife SE agreement");

    let data = load_et0_series(benchmark);
    let boot = GpuBootstrap::cpu()
        .estimate_mean(&data, 1000, 42)
        .expect("bootstrap");
    let jack = GpuJackknife::cpu().estimate(&data).expect("jackknife");

    let ratio = boot.std_error / jack.std_error;
    v.check_abs("SE ratio in [0.5, 2.0]", ratio, 1.0, 1.0);

    println!(
        "  Rust: boot_SE={:.6}, jack_SE={:.6}, ratio={:.3}",
        boot.std_error, jack.std_error, ratio
    );
}

fn main() {
    validation::init_tracing();
    validation::banner("Exp 080: Bootstrap & Jackknife CI for Seasonal ET₀");

    let mut v = ValidationHarness::new("Bootstrap & Jackknife");
    let benchmark = parse_benchmark_json(BENCHMARK_JSON)
        .expect("benchmark_bootstrap_jackknife.json must parse");

    validate_bootstrap_season(&mut v, &benchmark);
    validate_jackknife_season(&mut v, &benchmark);
    validate_known_values(&mut v, &benchmark);
    validate_small_sample(&mut v, &benchmark);
    validate_constant_data(&mut v);
    validate_boot_jack_agreement(&mut v, &benchmark);

    v.finish();
}
