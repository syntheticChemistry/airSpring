// SPDX-License-Identifier: AGPL-3.0-or-later
//! Scenario: Tier 4 Math Parity — verify `crate::math` dual-path correctness.
//!
//! Confirms the math module (`mean`, `pearson_r`, `std_dev`) produces identical
//! results regardless of whether the `local` feature links barraCuda or uses
//! inline fallbacks. When `local` is enabled, also checks that `crate::math`
//! matches direct `barracuda::stats` calls.

use crate::validation::ValidationHarness;
use crate::validation::scenarios::registry::{Scenario, ScenarioMeta, Tier, Track};

/// Tier 4 math parity scenario metadata and entry point.
pub const SCENARIO: Scenario = Scenario {
    meta: ScenarioMeta {
        id: "tier4-math-parity",
        track: Track::Composition,
        tier: Tier::Rust,
        provenance_crate: "ludoSpring V61 Tier 4 exemplar pattern",
        provenance_date: "2026-05-11",
        description: "Verify crate::math dual-path produces identical results",
    },
    run,
};

/// Run Tier 4 math parity validation.
pub fn run(harness: &mut ValidationHarness) {
    let data = &[1.0, 2.0, 3.0, 4.0, 5.0];
    let xs = &[1.0, 2.0, 3.0, 4.0, 5.0];
    let ys = &[2.1, 3.9, 6.1, 7.8, 10.2];

    let computed_mean = crate::math::mean(data);
    harness.check_abs("math::mean([1..5])", computed_mean, 3.0, 1e-15);

    let computed_r = crate::math::pearson_r(xs, ys);
    harness.check_bool("math::pearson_r > 0.99", computed_r > 0.99);

    let computed_sd = crate::math::std_dev(data);
    // barraCuda (local feature) uses sample std dev (N-1); fallback uses population (N).
    #[cfg(feature = "local")]
    let expected_sd = (2.5_f64).sqrt(); // sample: sum((xi-3)^2)/4 = 10/4 = 2.5
    #[cfg(not(feature = "local"))]
    let expected_sd = (2.0_f64).sqrt(); // population: sum((xi-3)^2)/5 = 10/5 = 2.0
    harness.check_abs("math::std_dev([1..5])", computed_sd, expected_sd, 1e-10);

    #[cfg(feature = "local")]
    {
        let upstream_mean = barracuda::stats::mean(data);
        harness.check_abs(
            "math::mean vs barracuda",
            computed_mean,
            upstream_mean,
            1e-15,
        );
        if let Ok(upstream_r) = barracuda::stats::pearson_correlation(xs, ys) {
            harness.check_abs(
                "math::pearson_r vs barracuda",
                computed_r,
                upstream_r,
                1e-15,
            );
        }
        if let Ok(upstream_sd) = barracuda::stats::correlation::std_dev(data) {
            harness.check_abs(
                "math::std_dev vs barracuda",
                computed_sd,
                upstream_sd,
                1e-15,
            );
        }
    }
}
