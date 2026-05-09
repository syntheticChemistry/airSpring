// SPDX-License-Identifier: AGPL-3.0-or-later

//! Validation scenarios — absorbed experiment patterns.
//!
//! Each scenario is a self-contained validation function that exercises
//! a specific airSpring niche capability or composition pattern. Scenarios
//! evolved from the prokaryotic experiment binary era (exp001–exp003) and
//! were absorbed into the library at the interstadial transition.

mod registry;

pub use registry::{Scenario, ScenarioMeta, ScenarioRegistry, Tier, Track};

pub mod s_composition_parity;
pub mod s_foundation_targets;
pub mod s_local_science_parity;

/// Build the canonical scenario registry with all absorbed scenarios.
#[must_use]
pub fn build_registry() -> ScenarioRegistry {
    let mut r = ScenarioRegistry::new();
    r.register(s_local_science_parity::SCENARIO);
    r.register(s_composition_parity::SCENARIO);
    r.register(s_foundation_targets::SCENARIO);
    r
}
