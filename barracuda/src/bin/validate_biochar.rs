// SPDX-License-Identifier: AGPL-3.0-or-later
//! Validate biochar adsorption isotherms against Python baseline.
//!
//! Benchmark source: `control/biochar/benchmark_biochar.json`
//! Provenance: Kumari, Dong & Safferman (2025) Applied Water Science 15(7):162.
//! Baseline: `control/biochar/biochar_isotherms.py` (14/14 PASS).

use airspring_barracuda::eco::isotherm::{self, langmuir_rl};
use airspring_barracuda::validation::{self, parse_benchmark_json, ValidationHarness};

/// Benchmark JSON embedded at compile time for reproducibility.
const BENCHMARK_JSON: &str = include_str!("../../../control/biochar/benchmark_biochar.json");

/// Maximum mean residual (mg/g) for no systematic bias.
const MAX_MEAN_RESIDUAL: f64 = 0.5;

#[allow(clippy::too_many_lines)]
fn main() {
    validation::banner("Biochar Adsorption Isotherms Validation");
    let mut v = ValidationHarness::new("Biochar Adsorption Isotherms Validation");
    let benchmark =
        parse_benchmark_json(BENCHMARK_JSON).expect("benchmark_biochar.json must parse");

    let datasets = benchmark
        .get("isotherm_data")
        .and_then(|d| d.get("datasets"))
        .and_then(|d| d.as_object())
        .expect("benchmark must have isotherm_data.datasets");

    let validation = benchmark
        .get("validation_checks")
        .expect("benchmark must have validation_checks");

    let mut results: std::collections::HashMap<String, DatasetResult> =
        std::collections::HashMap::new();

    for (ds_name, ds) in datasets {
        let ce: Vec<f64> = ds
            .get("Ce")
            .and_then(|a| a.as_array())
            .expect("dataset must have Ce")
            .iter()
            .filter_map(serde_json::Value::as_f64)
            .collect();
        let qe: Vec<f64> = ds
            .get("qe")
            .and_then(|a| a.as_array())
            .expect("dataset must have qe")
            .iter()
            .filter_map(serde_json::Value::as_f64)
            .collect();

        let source = ds.get("source").and_then(|s| s.as_str()).unwrap_or("");

        println!("\n── Dataset: {ds_name} ──");
        println!("  Source: {source}");

        let lang_fit = isotherm::fit_langmuir(&ce, &qe).expect("Langmuir fit must succeed");
        let freund_fit = isotherm::fit_freundlich(&ce, &qe).expect("Freundlich fit must succeed");

        let qmax = lang_fit.params[0];
        let kl = lang_fit.params[1];
        let kf = freund_fit.params[0];
        let n = freund_fit.params[1];

        println!(
            "\n  Langmuir: qmax={qmax:.4} mg/g, KL={kl:.6} L/mg, R²={:.4}",
            lang_fit.r_squared
        );
        println!(
            "  Freundlich: KF={kf:.4}, n={n:.4}, R²={:.4}",
            freund_fit.r_squared
        );

        let rl = langmuir_rl(kl, 100.0);
        println!("  RL (C0=100 mg/L) = {rl:.4}");

        let pred_lang: Vec<f64> = ce
            .iter()
            .map(|&c| isotherm::langmuir(c, qmax, kl))
            .collect();
        let pred_freund: Vec<f64> = ce
            .iter()
            .map(|&c| isotherm::freundlich(c, kf, 1.0 / n))
            .collect();

        let residuals_lang: Vec<f64> = qe.iter().zip(&pred_lang).map(|(a, b)| a - b).collect();
        let residuals_freund: Vec<f64> = qe.iter().zip(&pred_freund).map(|(a, b)| a - b).collect();

        let mean_res_lang = residuals_lang.iter().sum::<f64>() / len_f64(&residuals_lang);
        let mean_res_freund = residuals_freund.iter().sum::<f64>() / len_f64(&residuals_freund);

        results.insert(
            ds_name.clone(),
            DatasetResult {
                qmax,
                kl,
                r2_langmuir: lang_fit.r_squared,
                kf,
                n,
                r2_freundlich: freund_fit.r_squared,
                max_mean_residual: mean_res_lang.abs().max(mean_res_freund.abs()),
            },
        );
    }

    // ── Langmuir validation checks ─────────────────────────────────────
    validation::section("Langmuir fit validation");

    let lang_checks = validation
        .get("langmuir_fit")
        .and_then(|o| o.get("checks"))
        .and_then(|a| a.as_array())
        .expect("langmuir_fit.checks");

    for c in lang_checks {
        let cid = c.get("id").and_then(|v| v.as_str()).unwrap_or("");
        let desc = c.get("description").and_then(|v| v.as_str()).unwrap_or("");

        if cid == "wood_qmax_range" {
            let r = results.get("wood_biochar_500C").unwrap();
            let min_v = c
                .get("min")
                .and_then(serde_json::Value::as_f64)
                .unwrap_or(0.0);
            let max_v = c
                .get("max")
                .and_then(serde_json::Value::as_f64)
                .unwrap_or(f64::INFINITY);
            v.check_bool(
                &format!("{desc}: qmax={:.4} mg/g", r.qmax),
                r.qmax >= min_v && r.qmax <= max_v,
            );
        } else if cid == "wood_KL_positive" {
            let r = results.get("wood_biochar_500C").unwrap();
            v.check_bool(&format!("{desc}: KL={:.6}", r.kl), r.kl > 0.0);
        } else if cid == "wood_r2" {
            let r = results.get("wood_biochar_500C").unwrap();
            let min_r2 = c
                .get("min_r2")
                .and_then(serde_json::Value::as_f64)
                .unwrap_or(0.0);
            v.check_bool(
                &format!("{desc}: R²={:.4}", r.r2_langmuir),
                r.r2_langmuir >= min_r2,
            );
        } else if cid == "sugar_qmax_range" {
            let r = results.get("sugar_beet_biochar").unwrap();
            let min_v = c
                .get("min")
                .and_then(serde_json::Value::as_f64)
                .unwrap_or(0.0);
            let max_v = c
                .get("max")
                .and_then(serde_json::Value::as_f64)
                .unwrap_or(f64::INFINITY);
            v.check_bool(
                &format!("{desc}: qmax={:.4} mg/g", r.qmax),
                r.qmax >= min_v && r.qmax <= max_v,
            );
        } else if cid == "sugar_r2" {
            let r = results.get("sugar_beet_biochar").unwrap();
            let min_r2 = c
                .get("min_r2")
                .and_then(serde_json::Value::as_f64)
                .unwrap_or(0.0);
            v.check_bool(
                &format!("{desc}: R²={:.4}", r.r2_langmuir),
                r.r2_langmuir >= min_r2,
            );
        }
    }

    // ── Freundlich validation checks ───────────────────────────────────
    println!();
    validation::section("Freundlich fit validation");

    let freund_checks = validation
        .get("freundlich_fit")
        .and_then(|o| o.get("checks"))
        .and_then(|a| a.as_array())
        .expect("freundlich_fit.checks");

    for c in freund_checks {
        let cid = c.get("id").and_then(|v| v.as_str()).unwrap_or("");
        let desc = c.get("description").and_then(|v| v.as_str()).unwrap_or("");

        if cid == "wood_KF_positive" {
            let r = results.get("wood_biochar_500C").unwrap();
            v.check_bool(&format!("{desc}: KF={:.4}", r.kf), r.kf > 0.0);
        } else if cid == "wood_n_favorable" {
            let r = results.get("wood_biochar_500C").unwrap();
            let min_n = c
                .get("min_n")
                .and_then(serde_json::Value::as_f64)
                .unwrap_or(1.0);
            v.check_bool(&format!("{desc}: n={:.4}", r.n), r.n >= min_n);
        } else if cid == "wood_r2" {
            let r = results.get("wood_biochar_500C").unwrap();
            let min_r2 = c
                .get("min_r2")
                .and_then(serde_json::Value::as_f64)
                .unwrap_or(0.0);
            v.check_bool(
                &format!("{desc}: R²={:.4}", r.r2_freundlich),
                r.r2_freundlich >= min_r2,
            );
        } else if cid == "sugar_KF_positive" {
            let r = results.get("sugar_beet_biochar").unwrap();
            v.check_bool(&format!("{desc}: KF={:.4}", r.kf), r.kf > 0.0);
        } else if cid == "sugar_n_range" {
            let r = results.get("sugar_beet_biochar").unwrap();
            let min_v = c
                .get("min")
                .and_then(serde_json::Value::as_f64)
                .unwrap_or(0.0);
            let max_v = c
                .get("max")
                .and_then(serde_json::Value::as_f64)
                .unwrap_or(f64::INFINITY);
            v.check_bool(
                &format!("{desc}: n={:.4}", r.n),
                r.n >= min_v && r.n <= max_v,
            );
        }
    }

    // ── Model comparison ────────────────────────────────────────────────
    println!();
    validation::section("Model comparison");

    let model_checks = validation
        .get("model_comparison")
        .and_then(|o| o.get("checks"))
        .and_then(|a| a.as_array())
        .expect("model_comparison.checks");

    for c in model_checks {
        let cid = c.get("id").and_then(|v| v.as_str()).unwrap_or("");
        let desc = c.get("description").and_then(|v| v.as_str()).unwrap_or("");

        if cid == "langmuir_better_wood" {
            let r = results.get("wood_biochar_500C").unwrap();
            v.check_bool(
                &format!(
                    "{desc}: Langmuir R²={:.4}, Freundlich R²={:.4}",
                    r.r2_langmuir, r.r2_freundlich
                ),
                r.r2_langmuir >= r.r2_freundlich,
            );
        } else if cid == "both_positive_params" {
            let all_pos = results
                .values()
                .all(|r| r.qmax > 0.0 && r.kl > 0.0 && r.kf > 0.0 && r.n > 0.0);
            v.check_bool(desc, all_pos);
        } else if cid == "residuals_random" {
            let max_mean = results
                .values()
                .map(|r| r.max_mean_residual)
                .fold(0.0_f64, f64::max);
            v.check_bool(
                &format!("{desc}: max |mean residual|={max_mean:.4} mg/g"),
                max_mean < MAX_MEAN_RESIDUAL,
            );
        }
    }

    // ── Separation factor ────────────────────────────────────────────
    println!();
    validation::section("Separation factor RL");

    let sep_checks = validation
        .get("separation_factor")
        .and_then(|o| o.get("checks"))
        .and_then(|a| a.as_array())
        .expect("separation_factor.checks");

    for c in sep_checks {
        let cid = c.get("id").and_then(|v| v.as_str()).unwrap_or("");
        let desc = c.get("description").and_then(|v| v.as_str()).unwrap_or("");

        if cid == "rl_favorable" {
            let c0 = c
                .get("C0")
                .and_then(serde_json::Value::as_f64)
                .unwrap_or(100.0);
            let all_favorable = results.values().all(|r| {
                let rl = langmuir_rl(r.kl, c0);
                rl > 0.0 && rl < 1.0
            });
            v.check_bool(&format!("{desc}: C0={c0} mg/L"), all_favorable);
        }
    }

    v.finish();
}

#[allow(clippy::cast_precision_loss)]
fn len_f64(slice: &[f64]) -> f64 {
    slice.len() as f64
}

struct DatasetResult {
    qmax: f64,
    kl: f64,
    r2_langmuir: f64,
    kf: f64,
    n: f64,
    r2_freundlich: f64,
    max_mean_residual: f64,
}
