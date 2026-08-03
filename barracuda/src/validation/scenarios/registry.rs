// SPDX-License-Identifier: AGPL-3.0-or-later

//! Scenario registry — metadata, filtering, and execution.
//!
//! Mirrors the primalSpring `ScenarioRegistry` pattern for airSpring's
//! niche domain (ecology/agriculture).

use crate::validation::ValidationHarness;

/// Validation tier for scenario filtering.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Tier {
    /// Tier 1: Pure Rust structural validation — no IPC needed.
    Rust,
    /// Tier 2: Live NUCLEUS validation — requires deployed primals.
    Live,
    /// Both tiers: has structural and live phases.
    Both,
}

impl std::fmt::Display for Tier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Rust => write!(f, "rust"),
            Self::Live => write!(f, "live"),
            Self::Both => write!(f, "both"),
        }
    }
}

/// Track taxonomy — groups related scenarios by domain.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Track {
    /// Science dispatch: method routing, alias parity.
    ScienceDispatch,
    /// Composition: NUCLEUS parity, cross-atomic pipeline.
    Composition,
    /// Foundation: external target validation.
    Foundation,
    /// Provenance: trio roundtrip.
    Provenance,
}

impl std::fmt::Display for Track {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ScienceDispatch => write!(f, "science-dispatch"),
            Self::Composition => write!(f, "composition"),
            Self::Foundation => write!(f, "foundation"),
            Self::Provenance => write!(f, "provenance"),
        }
    }
}

impl Track {
    /// Parse a track name from a string.
    #[must_use]
    pub fn from_str_loose(s: &str) -> Option<Self> {
        match s {
            "science-dispatch" | "science" => Some(Self::ScienceDispatch),
            "composition" | "comp" => Some(Self::Composition),
            "foundation" | "targets" => Some(Self::Foundation),
            "provenance" | "prov" => Some(Self::Provenance),
            _ => None,
        }
    }
}

/// Scenario metadata — provenance, classification, and description.
#[derive(Debug, Clone)]
pub struct ScenarioMeta {
    /// Unique scenario identifier.
    pub id: &'static str,
    /// Which track this scenario belongs to.
    pub track: Track,
    /// Which validation tier this scenario exercises.
    pub tier: Tier,
    /// Original experiment crate name for provenance.
    pub provenance_crate: &'static str,
    /// Date of last significant update.
    pub provenance_date: &'static str,
    /// One-line description.
    pub description: &'static str,
}

/// A callable scenario: metadata + run function.
pub struct Scenario {
    /// Scenario metadata.
    pub meta: ScenarioMeta,
    /// The validation function.
    pub run: fn(&mut ValidationHarness),
}

impl std::fmt::Debug for Scenario {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Scenario")
            .field("id", &self.meta.id)
            .field("track", &self.meta.track)
            .field("tier", &self.meta.tier)
            .finish_non_exhaustive()
    }
}

/// Registry of all absorbed validation scenarios.
pub struct ScenarioRegistry {
    scenarios: Vec<Scenario>,
}

impl ScenarioRegistry {
    /// Create an empty registry.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            scenarios: Vec::new(),
        }
    }

    /// Register a scenario.
    pub fn register(&mut self, scenario: Scenario) {
        self.scenarios.push(scenario);
    }

    /// All registered scenarios.
    #[must_use]
    pub fn all(&self) -> &[Scenario] {
        &self.scenarios
    }

    /// Filter scenarios by tier.
    pub fn filter_by_tier(&self, tier: Tier) -> impl Iterator<Item = &Scenario> {
        self.scenarios
            .iter()
            .filter(move |s| s.meta.tier == tier || s.meta.tier == Tier::Both || tier == Tier::Both)
    }

    /// Total number of registered scenarios.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.scenarios.len()
    }

    /// Whether the registry is empty.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.scenarios.is_empty()
    }
}

impl Default for ScenarioRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tier_display_roundtrip() {
        assert_eq!(Tier::Rust.to_string(), "rust");
        assert_eq!(Tier::Live.to_string(), "live");
        assert_eq!(Tier::Both.to_string(), "both");
    }

    #[test]
    fn track_display_roundtrip() {
        assert_eq!(Track::ScienceDispatch.to_string(), "science-dispatch");
        assert_eq!(Track::Composition.to_string(), "composition");
        assert_eq!(Track::Foundation.to_string(), "foundation");
        assert_eq!(Track::Provenance.to_string(), "provenance");
    }

    #[test]
    fn track_from_str_loose_canonical() {
        assert_eq!(
            Track::from_str_loose("science-dispatch"),
            Some(Track::ScienceDispatch)
        );
        assert_eq!(
            Track::from_str_loose("composition"),
            Some(Track::Composition)
        );
        assert_eq!(Track::from_str_loose("foundation"), Some(Track::Foundation));
        assert_eq!(Track::from_str_loose("provenance"), Some(Track::Provenance));
    }

    #[test]
    fn track_from_str_loose_aliases() {
        assert_eq!(
            Track::from_str_loose("science"),
            Some(Track::ScienceDispatch)
        );
        assert_eq!(Track::from_str_loose("comp"), Some(Track::Composition));
        assert_eq!(Track::from_str_loose("targets"), Some(Track::Foundation));
        assert_eq!(Track::from_str_loose("prov"), Some(Track::Provenance));
    }

    #[test]
    fn track_from_str_loose_unknown() {
        assert_eq!(Track::from_str_loose("unknown"), None);
        assert_eq!(Track::from_str_loose(""), None);
    }

    #[test]
    fn registry_starts_empty() {
        let r = ScenarioRegistry::new();
        assert!(r.is_empty());
        assert_eq!(r.len(), 0);
        assert!(r.all().is_empty());
    }

    #[test]
    fn registry_default_is_empty() {
        let r = ScenarioRegistry::default();
        assert!(r.is_empty());
    }

    fn dummy_run(_: &mut ValidationHarness) {}

    fn make_scenario(id: &'static str, tier: Tier, track: Track) -> Scenario {
        Scenario {
            meta: ScenarioMeta {
                id,
                track,
                tier,
                provenance_crate: "test",
                provenance_date: "2026-01-01",
                description: "test scenario",
            },
            run: dummy_run,
        }
    }

    #[test]
    fn register_and_retrieve() {
        let mut r = ScenarioRegistry::new();
        r.register(make_scenario("a", Tier::Rust, Track::ScienceDispatch));
        r.register(make_scenario("b", Tier::Live, Track::Composition));
        assert_eq!(r.len(), 2);
        assert!(!r.is_empty());
        assert_eq!(r.all()[0].meta.id, "a");
        assert_eq!(r.all()[1].meta.id, "b");
    }

    #[test]
    fn filter_by_tier_rust() {
        let mut r = ScenarioRegistry::new();
        r.register(make_scenario(
            "rust-only",
            Tier::Rust,
            Track::ScienceDispatch,
        ));
        r.register(make_scenario("live-only", Tier::Live, Track::Composition));
        r.register(make_scenario("both-tiers", Tier::Both, Track::Foundation));

        let rust: Vec<_> = r.filter_by_tier(Tier::Rust).collect();
        assert_eq!(rust.len(), 2);
        assert!(rust.iter().any(|s| s.meta.id == "rust-only"));
        assert!(rust.iter().any(|s| s.meta.id == "both-tiers"));
    }

    #[test]
    fn filter_by_tier_live() {
        let mut r = ScenarioRegistry::new();
        r.register(make_scenario(
            "rust-only",
            Tier::Rust,
            Track::ScienceDispatch,
        ));
        r.register(make_scenario("live-only", Tier::Live, Track::Composition));
        r.register(make_scenario("both-tiers", Tier::Both, Track::Foundation));

        let live: Vec<_> = r.filter_by_tier(Tier::Live).collect();
        assert_eq!(live.len(), 2);
        assert!(live.iter().any(|s| s.meta.id == "live-only"));
        assert!(live.iter().any(|s| s.meta.id == "both-tiers"));
    }

    #[test]
    fn filter_by_tier_both_returns_all() {
        let mut r = ScenarioRegistry::new();
        r.register(make_scenario("a", Tier::Rust, Track::ScienceDispatch));
        r.register(make_scenario("b", Tier::Live, Track::Composition));
        r.register(make_scenario("c", Tier::Both, Track::Foundation));

        let both: Vec<_> = r.filter_by_tier(Tier::Both).collect();
        assert_eq!(both.len(), 3);
    }

    #[test]
    fn scenario_debug_format() {
        let s = make_scenario("test-debug", Tier::Rust, Track::Foundation);
        let dbg = format!("{s:?}");
        assert!(dbg.contains("test-debug"));
        assert!(dbg.contains("Rust"));
        assert!(dbg.contains("Foundation"));
    }
}
