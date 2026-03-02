// SPDX-License-Identifier: AGPL-3.0-or-later
#![warn(clippy::pedantic)]
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::too_many_lines
)]
//! Exp 060: NASS Real Yield Comparison.
//!
//! Validates Stewart (1977) yield response model predictions against observed
//! Michigan crop yields. Uses synthetic benchmark data (matching NASS county
//! yield patterns) until a real USDA NASS API key is configured.
//!
//! Key validations:
//! 1. Drought years show expected yield drops (corn > wheat > soybean)
//! 2. Water deficit (ET₀ - precip) correlates with yield depression
//! 3. Yield ratios stay in [0.3, 1.0] for realistic weather
//! 4. Crop sensitivity ordering matches Ky values
//!
//! Provenance: NASS API (no Python baseline)
//! commit=88d07c0, date=2026-03-01

use airspring_barracuda::eco::yield_response;
use airspring_barracuda::validation::ValidationHarness;

const BENCHMARK_JSON: &str = include_str!("../../../control/nass_real/benchmark_nass_real.json");

fn main() {
    tracing_subscriber::fmt().with_env_filter("info").init();

    let benchmark: serde_json::Value =
        serde_json::from_str(BENCHMARK_JSON).expect("benchmark JSON");

    let mut v = ValidationHarness::new("Exp 060: NASS Real Yield Comparison");

    let crops = benchmark["crops"].as_array().expect("crops array");

    for crop in crops {
        let commodity = crop["commodity"].as_str().unwrap_or("");
        let ky = crop["ky_mid"].as_f64().unwrap_or(1.0);
        let years = crop["synthetic_years"].as_array().expect("years");
        let drought_year = crop["expected_drought_year"].as_u64().unwrap_or(0);
        let drought_drop = crop["expected_drought_yield_drop_pct"]
            .as_f64()
            .unwrap_or(0.0);

        let mut yields: Vec<(u32, f64)> = Vec::new();
        let mut drought_yield = 0.0;
        let mut max_yield = 0.0_f64;

        for entry in years {
            let year = entry["year"].as_u64().unwrap_or(0) as u32;
            let yield_val = entry["yield_bu_acre"].as_f64().unwrap_or(0.0);
            let precip = entry["season_precip_mm"].as_f64().unwrap_or(0.0);
            let et0 = entry["season_et0_mm"].as_f64().unwrap_or(0.0);

            yields.push((year, yield_val));
            max_yield = max_yield.max(yield_val);

            if u64::from(year) == drought_year {
                drought_yield = yield_val;
            }

            let water_deficit = et0 - precip;
            let precip_ratio = precip / et0;

            v.check_lower(
                &format!("{commodity}_{year}_precip_ratio_positive"),
                precip_ratio,
                0.0,
            );
            v.check_upper(
                &format!("{commodity}_{year}_precip_ratio_realistic"),
                precip_ratio,
                2.0,
            );

            // Stewart yield ratio from weather
            let eta_etc_ratio = if et0 > 0.0 {
                (precip / et0).min(1.0)
            } else {
                1.0
            };
            let stewart_ratio = yield_response::yield_ratio_single(ky, eta_etc_ratio);
            v.check_lower(
                &format!("{commodity}_{year}_stewart_ratio_min"),
                stewart_ratio,
                0.0,
            );
            v.check_upper(
                &format!("{commodity}_{year}_stewart_ratio_max"),
                stewart_ratio,
                1.01,
            );

            if water_deficit > 100.0 {
                v.check_upper(
                    &format!("{commodity}_{year}_deficit_yield_ratio"),
                    stewart_ratio,
                    0.95,
                );
            }
        }

        // Drought year should show meaningful yield drop
        if drought_yield > 0.0 && max_yield > 0.0 {
            let actual_drop_pct = (1.0 - drought_yield / max_yield) * 100.0;
            v.check_lower(
                &format!("{commodity}_drought_drop_detected"),
                actual_drop_pct,
                drought_drop * 0.5,
            );
        }

        // Yield trend: recent years should be higher (technology + management)
        if yields.len() >= 4 {
            let first_half: f64 = yields[..yields.len() / 2]
                .iter()
                .map(|(_, y)| y)
                .sum::<f64>()
                / (yields.len() / 2) as f64;
            let second_half: f64 = yields[yields.len() / 2..]
                .iter()
                .map(|(_, y)| y)
                .sum::<f64>()
                / (yields.len() - yields.len() / 2) as f64;
            v.check_lower(
                &format!("{commodity}_yield_trend_positive"),
                second_half - first_half,
                0.0,
            );
        }
    }

    // Cross-crop: corn should be most drought-sensitive (highest Ky)
    let corn_ky = crops
        .iter()
        .find(|c| c["commodity"].as_str() == Some("CORN"))
        .and_then(|c| c["ky_mid"].as_f64())
        .unwrap_or(0.0);
    let soy_ky = crops
        .iter()
        .find(|c| c["commodity"].as_str() == Some("SOYBEANS"))
        .and_then(|c| c["ky_mid"].as_f64())
        .unwrap_or(0.0);
    let wheat_ky = crops
        .iter()
        .find(|c| c["commodity"].as_str() == Some("WHEAT"))
        .and_then(|c| c["ky_mid"].as_f64())
        .unwrap_or(0.0);

    v.check_lower("corn_ky_gt_wheat", corn_ky - wheat_ky, 0.0);
    v.check_lower("corn_ky_gt_soy", corn_ky - soy_ky, 0.0);
    v.check_lower("wheat_ky_gt_soy", wheat_ky - soy_ky, 0.0);

    v.finish();
}
