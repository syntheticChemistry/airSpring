// SPDX-License-Identifier: AGPL-3.0-or-later

//! Feature-gated `CompositionContext` integration for Layer 3+ certification.
//!
//! Active only when `guidestone` feature is enabled (brings in `primalspring`).
//! Provides typed composition validation instead of raw RPC calls.
//!
//! Without this feature, Layers 1-4 use direct `biomeos::discover_*` + `rpc::send_to`.
//! With this feature, Layer 3+ can validate through primalSpring's canonical
//! composition contracts — required for guideStone L3+.

pub use primalspring::composition::CompositionContext;

use crate::validation::ValidationHarness;

/// Validate that airSpring's capabilities are reachable through composition.
///
/// Checks each capability domain that airSpring depends on from the
/// running composition, using primalSpring's typed discovery escalation
/// rather than raw socket scanning.
pub fn validate_composition_context(v: &mut ValidationHarness) {
    let ctx = CompositionContext::discover();
    let available = ctx.available_capabilities();

    v.check_bool(
        "composition discovered at least one capability",
        !available.is_empty(),
    );

    let required_capabilities = ["math", "storage", "security", "compute"];
    for cap in required_capabilities {
        v.check_bool(
            &format!("composition has `{cap}` capability"),
            ctx.has_capability(cap),
        );
    }

    eprintln!(
        "  composition: {} capabilities available ({:?})",
        available.len(),
        available
    );
}
