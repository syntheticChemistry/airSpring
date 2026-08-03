// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2025-2026 ecoPrimals Collective
//! Validate gamma CDF (regularised incomplete gamma) against Python baseline
//! (`control/gamma_cdf/benchmark_gamma_cdf.json`).
//!
//! Implements cross-tier parity checks for [`gamma_cdf`] used in SPI-style
//! drought indices. Requires `BarraCuda` `regularized_gamma_p` via the `local`
//! feature; without it the binary skips.

#[cfg(feature = "local")]
use airspring_barracuda::eco::drought_index::{GammaParams, gamma_cdf};
#[cfg(feature = "local")]
use airspring_barracuda::validation::{self, ValidationHarness, json_f64, parse_benchmark_json};
#[cfg(feature = "local")]
use serde_json::Value;

#[cfg(feature = "local")]
const BENCHMARK_JSON: &str = include_str!("../../../control/gamma_cdf/benchmark_gamma_cdf.json");

#[cfg(feature = "local")]
fn tolerance_or_default(check: &Value, fallback: f64) -> f64 {
    check
        .get("tolerance")
        .and_then(Value::as_f64)
        .unwrap_or(fallback)
}

#[cfg(feature = "local")]
fn validate_numeric_checks(v: &mut ValidationHarness, bench: &Value) {
    validation::section("Gamma CDF — exact benchmarks");
    let checks = bench["checks"].as_array().expect("checks array");

    for c in checks {
        let Some(name) = c["name"].as_str() else {
            continue;
        };

        match name {
            "exponential_x0.5" => {
                let p = GammaParams {
                    alpha: 1.0,
                    beta: 1.0,
                };
                let obs = gamma_cdf(0.5, &p);
                let expected = json_f64(c, &["expected"]).expect("expected");
                v.check_abs(name, obs, expected, tolerance_or_default(c, 1e-10));
            }
            "exponential_x1.0" => {
                let p = GammaParams {
                    alpha: 1.0,
                    beta: 1.0,
                };
                let obs = gamma_cdf(1.0, &p);
                let expected = json_f64(c, &["expected"]).expect("expected");
                v.check_abs(name, obs, expected, tolerance_or_default(c, 1e-10));
            }
            "exponential_x2.0" => {
                let p = GammaParams {
                    alpha: 1.0,
                    beta: 1.0,
                };
                let obs = gamma_cdf(2.0, &p);
                let expected = json_f64(c, &["expected"]).expect("expected");
                v.check_abs(name, obs, expected, tolerance_or_default(c, 1e-10));
            }
            "exponential_x5.0" => {
                let p = GammaParams {
                    alpha: 1.0,
                    beta: 1.0,
                };
                let obs = gamma_cdf(5.0, &p);
                let expected = json_f64(c, &["expected"]).expect("expected");
                v.check_abs(name, obs, expected, tolerance_or_default(c, 1e-10));
            }
            "chi2_df2_x2" => {
                // χ²(df=2) ≡ Gamma(shape=α=1, scale=θ=2). CDF(2)=1−e^{-1}, same numerical target as Gamma(1,1) @ x=1.
                let p = GammaParams {
                    alpha: 1.0,
                    beta: 2.0,
                };
                let obs = gamma_cdf(2.0, &p);
                let expected = json_f64(c, &["expected"]).expect("expected");
                v.check_abs(name, obs, expected, tolerance_or_default(c, 1e-10));
            }
            "gamma_2_1_at_1" => {
                let p = GammaParams {
                    alpha: 2.0,
                    beta: 1.0,
                };
                let obs = gamma_cdf(1.0, &p);
                let expected = json_f64(c, &["expected"]).expect("expected");
                v.check_abs(name, obs, expected, tolerance_or_default(c, 1e-10));
            }
            "boundary_x0" => {
                let p = GammaParams {
                    alpha: 2.0,
                    beta: 1.0,
                };
                let obs = gamma_cdf(0.0, &p);
                let expected = json_f64(c, &["expected"]).expect("expected");
                v.check_abs(name, obs, expected, tolerance_or_default(c, 1e-15));
            }
            "large_x_near_1" => {
                let p = GammaParams {
                    alpha: 2.0,
                    beta: 1.0,
                };
                let obs = gamma_cdf(50.0, &p);
                let expected = json_f64(c, &["expected"]).expect("expected");
                v.check_abs(name, obs, expected, tolerance_or_default(c, 1e-10));
            }
            _ => {}
        }
    }
}

#[cfg(feature = "local")]
fn validate_range_checks(v: &mut ValidationHarness, bench: &Value) {
    validation::section("Gamma CDF — median / asymptotic bands");
    let checks = bench["checks"].as_array().expect("checks array");

    for c in checks {
        let Some(name) = c["name"].as_str() else {
            continue;
        };

        match name {
            "spi_typical_median" => {
                let p = GammaParams {
                    alpha: 4.5,
                    beta: 12.0,
                };
                let obs = gamma_cdf(54.0, &p);
                let range = c["expected_range"]
                    .as_array()
                    .expect("expected_range array");
                let lo = range[0].as_f64().expect("spi_typical lower");
                let hi = range[1].as_f64().expect("spi_typical upper");
                v.check_bool(
                    &format!("{name}: CDF {obs:.6} in [{lo:.3},{hi:.3}]"),
                    (lo..=hi).contains(&obs),
                );
            }
            "large_alpha_median" => {
                let p = GammaParams {
                    alpha: 100.0,
                    beta: 1.0,
                };
                let obs = gamma_cdf(100.0, &p);
                let range = c["expected_range"]
                    .as_array()
                    .expect("expected_range array");
                let lo = range[0].as_f64().expect("large_alpha lower");
                let hi = range[1].as_f64().expect("large_alpha upper");
                v.check_bool(
                    &format!("{name}: CDF {obs:.6} in [{lo:.3},{hi:.3}]"),
                    (lo..=hi).contains(&obs),
                );
            }
            _ => {}
        }
    }

    validation::section("Gamma CDF — reference excerpt");
    let reference = &bench["reference"];
    let expo = json_f64(reference, &["exponential_cdf_x1"]).expect("exponential_cdf_x1");
    let gamma21 = json_f64(reference, &["gamma_2_1_cdf_x1"]).expect("gamma_2_1_cdf_x1");
    let p1 = GammaParams {
        alpha: 1.0,
        beta: 1.0,
    };
    let p21 = GammaParams {
        alpha: 2.0,
        beta: 1.0,
    };
    v.check_abs(
        "reference exponentialCDF(1)",
        gamma_cdf(1.0, &p1),
        expo,
        1e-10,
    );
    v.check_abs(
        "reference gamma(2,1)CDF(1)",
        gamma_cdf(1.0, &p21),
        gamma21,
        1e-10,
    );
}

#[cfg(feature = "local")]
fn main() {
    validation::init_tracing();
    validation::banner("Gamma CDF Validation");
    let mut v = ValidationHarness::new("Gamma CDF");

    let Ok(bench) = parse_benchmark_json(BENCHMARK_JSON) else {
        eprintln!("[FAIL] benchmark JSON parse error");
        std::process::exit(1);
    };

    validate_numeric_checks(&mut v, &bench);
    validate_range_checks(&mut v, &bench);

    v.finish();
}

#[cfg(not(feature = "local"))]
fn main() {
    eprintln!(
        "Skipping validate_gamma_cdf: rebuild with `--features local` \
         (requires BarraCuda regularised incomplete gamma)."
    );
}
