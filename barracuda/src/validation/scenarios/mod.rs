// SPDX-License-Identifier: AGPL-3.0-or-later

//! Validation scenarios — absorbed experiment patterns.
//!
//! Each scenario is a self-contained validation function that exercises
//! a specific airSpring niche capability or composition pattern. Scenarios
//! evolved from the prokaryotic experiment binary era (exp001–exp003) and
//! were absorbed into the library at the interstadial transition.

mod registry;

pub use registry::{Scenario, ScenarioMeta, ScenarioRegistry, Tier, Track};

pub mod s_atlas_pipeline;
pub mod s_composition_parity;
pub mod s_et0_methods;
pub mod s_fao56_et0;
pub mod s_foundation_targets;
pub mod s_local_science_parity;
pub mod s_paper_chain;
pub mod s_soil_physics;
pub mod s_tier4_math_parity;
pub mod s_water_balance;

/// Build the canonical scenario registry with all absorbed scenarios.
#[must_use]
pub fn build_registry() -> ScenarioRegistry {
    let mut r = ScenarioRegistry::new();
    r.register(s_local_science_parity::SCENARIO);
    r.register(s_composition_parity::SCENARIO);
    r.register(s_foundation_targets::SCENARIO);
    r.register(s_fao56_et0::SCENARIO);
    r.register(s_et0_methods::SCENARIO);
    r.register(s_soil_physics::SCENARIO);
    r.register(s_water_balance::SCENARIO);
    r.register(s_atlas_pipeline::SCENARIO);
    r.register(s_paper_chain::SCENARIO);
    r.register(s_tier4_math_parity::SCENARIO);
    r
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::validation::ValidationHarness;

    #[test]
    fn build_registry_has_10_scenarios() {
        let r = build_registry();
        assert_eq!(r.len(), 10);
        assert!(!r.is_empty());
    }

    #[test]
    fn all_scenario_ids_unique() {
        let r = build_registry();
        let mut ids: Vec<&str> = r.all().iter().map(|s| s.meta.id).collect();
        let before = ids.len();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), before, "duplicate scenario IDs found");
    }

    #[test]
    fn all_scenarios_have_provenance() {
        let r = build_registry();
        for s in r.all() {
            assert!(
                !s.meta.provenance_crate.is_empty(),
                "{}: missing provenance_crate",
                s.meta.id
            );
            assert!(
                !s.meta.provenance_date.is_empty(),
                "{}: missing provenance_date",
                s.meta.id
            );
            assert!(
                !s.meta.description.is_empty(),
                "{}: missing description",
                s.meta.id
            );
        }
    }

    #[test]
    fn filter_rust_scenarios() {
        let r = build_registry();
        let rust_count = r.filter_by_tier(Tier::Rust).count();
        assert!(
            rust_count >= 8,
            "expected >= 8 Tier::Rust scenarios, got {rust_count}"
        );
    }

    #[test]
    fn run_local_science_parity() {
        let mut h = ValidationHarness::new("test: local-science-parity");
        (s_local_science_parity::SCENARIO.run)(&mut h);
        assert!(
            h.checks.iter().all(|c| c.passed),
            "local-science-parity had failures: {:?}",
            h.checks
                .iter()
                .filter(|c| !c.passed)
                .map(|c| &c.label)
                .collect::<Vec<_>>()
        );
        assert!(
            h.checks.len() >= 30,
            "expected >=30 checks, got {}",
            h.checks.len()
        );
    }

    #[test]
    fn run_fao56_et0() {
        let mut h = ValidationHarness::new("test: fao56-et0");
        (s_fao56_et0::SCENARIO.run)(&mut h);
        assert!(
            h.checks.iter().all(|c| c.passed),
            "fao56-et0 had failures: {:?}",
            h.checks
                .iter()
                .filter(|c| !c.passed)
                .map(|c| &c.label)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn run_et0_methods() {
        let mut h = ValidationHarness::new("test: et0-methods");
        (s_et0_methods::SCENARIO.run)(&mut h);
        assert!(h.checks.iter().all(|c| c.passed));
    }

    #[test]
    fn run_soil_physics() {
        let mut h = ValidationHarness::new("test: soil-physics");
        (s_soil_physics::SCENARIO.run)(&mut h);
        assert!(h.checks.iter().all(|c| c.passed));
    }

    #[test]
    fn run_water_balance() {
        let mut h = ValidationHarness::new("test: water-balance");
        (s_water_balance::SCENARIO.run)(&mut h);
        assert!(h.checks.iter().all(|c| c.passed));
    }

    #[test]
    fn run_atlas_pipeline() {
        let mut h = ValidationHarness::new("test: atlas-pipeline");
        (s_atlas_pipeline::SCENARIO.run)(&mut h);
        assert!(h.checks.iter().all(|c| c.passed));
    }

    #[test]
    fn run_paper_chain() {
        let mut h = ValidationHarness::new("test: paper-chain");
        (s_paper_chain::SCENARIO.run)(&mut h);
        assert!(h.checks.iter().all(|c| c.passed));
    }

    #[test]
    fn run_tier4_math_parity() {
        let mut h = ValidationHarness::new("test: tier4-math-parity");
        (s_tier4_math_parity::SCENARIO.run)(&mut h);
        assert!(h.checks.iter().all(|c| c.passed));
    }

    #[test]
    fn run_all_rust_tier_scenarios() {
        let r = build_registry();
        for scenario in r.filter_by_tier(Tier::Rust) {
            let mut h = ValidationHarness::new(&format!("test: {}", scenario.meta.id));
            (scenario.run)(&mut h);
            let failures: Vec<_> = h.checks.iter().filter(|c| !c.passed).collect();
            assert!(
                failures.is_empty(),
                "scenario '{}' had {} failures: {:?}",
                scenario.meta.id,
                failures.len(),
                failures.iter().map(|c| &c.label).collect::<Vec<_>>()
            );
        }
    }
}
