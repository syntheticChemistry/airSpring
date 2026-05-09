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
