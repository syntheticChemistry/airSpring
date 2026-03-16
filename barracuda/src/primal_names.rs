// SPDX-License-Identifier: AGPL-3.0-or-later
//! Known primal name constants for capability-based discovery.
//!
//! Springs discover primals at runtime via `biomeos::find_socket()` and
//! `biomeos::discover_all_primals()`.  These constants eliminate hardcoded
//! string literals when referring to other primals in socket lookups,
//! composition status, and graph node identifiers.
//!
//! Primal code only has self-knowledge (see [`crate::niche`]); these names
//! are discovery hints, not compile-time coupling.

/// Security primal (Ed25519 signing, encryption, key generation).
pub const TOADSTOOL: &str = "toadstool";
/// Hardware discovery and GPU compute orchestration.
pub const BEARDOG: &str = "beardog";
/// Network (TLS, HTTP fetch, DNS) primal.
pub const SONGBIRD: &str = "songbird";
/// Data storage and retrieval primal.
pub const NESTGATE: &str = "nestgate";
/// AI narration and ecology interpretation primal.
pub const SQUIRREL: &str = "squirrel";
/// DAG session management (provenance trio).
pub const RHIZOCRYPT: &str = "rhizocrypt";
/// Immutable ledger / certificate primal (provenance trio).
pub const LOAMSPINE: &str = "loamspine";
/// Provenance braids / attribution primal (provenance trio).
pub const SWEETGRASS: &str = "sweetgrass";
/// Visualization / interactive exploration primal.
pub const PETALTONGUE: &str = "petaltongue";
/// Neural API / capability routing primal (provenance trio gateway).
pub const NEURAL_API: &str = "neural-api";

/// Provenance trio capability domains (used in `capability.call`).
pub mod domains {
    pub const DAG: &str = "dag";
    pub const COMMIT: &str = "commit";
    pub const PROVENANCE: &str = "provenance";
    pub const COMPUTE: &str = "compute";
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_names_are_lowercase() {
        for name in [
            TOADSTOOL, BEARDOG, SONGBIRD, NESTGATE, SQUIRREL, RHIZOCRYPT,
            LOAMSPINE, SWEETGRASS, PETALTONGUE, NEURAL_API,
        ] {
            assert_eq!(name, name.to_lowercase(), "{name} must be lowercase");
        }
    }

    #[test]
    fn domains_are_lowercase() {
        for d in [domains::DAG, domains::COMMIT, domains::PROVENANCE, domains::COMPUTE] {
            assert_eq!(d, d.to_lowercase(), "{d} must be lowercase");
        }
    }
}
