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

/// Hardware discovery and GPU compute orchestration.
pub const TOADSTOOL: &str = "toadstool";
/// Security primal (Ed25519 signing, encryption, key generation).
pub const BEARDOG: &str = "beardog";
/// Ecosystem orchestrator.
pub const BIOMEOS: &str = "biomeos";
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

/// Derive the environment variable name for a primal's socket override.
///
/// Convention: `{PRIMAL_UPPER}_SOCKET`, e.g. `socket_env_var("toadstool")` →
/// `"TOADSTOOL_SOCKET"`. Used for capability-based discovery with env overrides.
///
/// Absorbed from groundSpring V112 / wetSpring V125 ecosystem pattern.
#[must_use]
pub fn socket_env_var(primal: &str) -> String {
    format!("{}_SOCKET", primal.to_ascii_uppercase())
}

/// Derive the environment variable name for a primal's address (HTTP/TCP).
///
/// Convention: `{PRIMAL_UPPER}_ADDRESS`, e.g. `address_env_var("nestgate")` →
/// `"NESTGATE_ADDRESS"`.
#[must_use]
pub fn address_env_var(primal: &str) -> String {
    format!("{}_ADDRESS", primal.to_ascii_uppercase())
}

/// Provenance trio capability domains (used in `capability.call`).
pub mod domains {
    /// DAG (directed acyclic graph) workflow capability.
    pub const DAG: &str = "dag";
    /// Commit/snapshot capability for versioned state.
    pub const COMMIT: &str = "commit";
    /// Provenance tracking and lineage capability.
    pub const PROVENANCE: &str = "provenance";
    /// Compute dispatch and execution capability.
    pub const COMPUTE: &str = "compute";
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_names_are_lowercase() {
        for name in [
            TOADSTOOL, BEARDOG, BIOMEOS, SONGBIRD, NESTGATE, SQUIRREL,
            RHIZOCRYPT, LOAMSPINE, SWEETGRASS, PETALTONGUE, NEURAL_API,
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

    #[test]
    fn socket_env_var_convention() {
        assert_eq!(socket_env_var(TOADSTOOL), "TOADSTOOL_SOCKET");
        assert_eq!(socket_env_var(NESTGATE), "NESTGATE_SOCKET");
        assert_eq!(socket_env_var(BIOMEOS), "BIOMEOS_SOCKET");
        assert_eq!(socket_env_var("neural-api"), "NEURAL-API_SOCKET");
    }

    #[test]
    fn address_env_var_convention() {
        assert_eq!(address_env_var(NESTGATE), "NESTGATE_ADDRESS");
        assert_eq!(address_env_var(SONGBIRD), "SONGBIRD_ADDRESS");
    }
}
