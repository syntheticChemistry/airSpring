// SPDX-License-Identifier: AGPL-3.0-or-later

//! Composition certification engine — absorbed guidestone organelle.
//!
//! Proves airSpring niche correctness through layered validation:
//!
//! | Layer | Name | Description |
//! |-------|------|-------------|
//! | 0     | Bare | manifest structural validation (no primals needed) |
//! | 1     | Discovery | primals in the composition discoverable |
//! | 2     | Health | discovered primals respond to `health.liveness` |
//! | 3     | Capability Parity | science IPC calls produce correct results |
//! | 4     | Cross-Atomic Pipeline | provenance trio roundtrip |
//! | 5     | NUCLEUS Composition | `composition.status`, `method.register`, `compute.dispatch` |
//! | 6     | Cross-Spring Pipeline | deploy graphs, capability registry, scenario registry |
//!
//! Originally evolved as the `airspring_guidestone` binary.
//! Endosymbiosed into the library at the interstadial transition.

pub mod bare;
#[cfg(feature = "guidestone")]
pub mod composition;
pub mod health;
pub mod nucleus;

use crate::validation::{ValidationHarness, banner, section};

/// Maximum certification layer (inclusive).
pub const MAX_LAYER: u8 = 6;

/// Run the full certification engine up to the specified layer.
///
/// Returns the `ValidationHarness` after all layers complete. Callers
/// can inspect `all_passed()` for pass/fail.
///
/// # Exit semantics
///
/// - `0` — all layers passed
/// - `1` — one or more layers failed
/// - `2` — bare-only mode (no primals discovered, structural checks only)
#[must_use]
pub fn certify(max_layer: u8) -> ValidationHarness {
    let mut v = ValidationHarness::new("airSpring Certification — Niche Correctness");
    banner("airSpring Certification — Niche Correctness");

    section("Layer 0: Bare Properties");
    bare::validate_bare_properties(&mut v);

    if max_layer == 0 {
        print_summary(&v);
        return v;
    }

    section("Layer 1: Discovery");
    let primals_found = health::validate_discovery(&mut v);

    if primals_found == 0 {
        eprintln!("[certify] No NUCLEUS primals discovered — bare certification only.");
        eprintln!("  Deploy from plasmidBin and rerun for full certification.");
        print_summary(&v);
        return v;
    }

    if max_layer < 2 {
        print_summary(&v);
        return v;
    }

    section("Layer 2: Health");
    health::validate_health(&mut v);

    if max_layer < 3 {
        print_summary(&v);
        return v;
    }

    section("Layer 3: Capability Parity");
    health::validate_science_parity(&mut v);

    if max_layer < 4 {
        print_summary(&v);
        return v;
    }

    section("Layer 4: Cross-Atomic Pipeline");
    health::validate_provenance_roundtrip(&mut v);

    if max_layer < 5 {
        print_summary(&v);
        return v;
    }

    section("Layer 5: NUCLEUS Composition");
    nucleus::validate_composition(&mut v);

    if max_layer < 6 {
        print_summary(&v);
        return v;
    }

    section("Layer 6: Cross-Spring Pipeline");
    nucleus::validate_cross_spring(&mut v);

    print_summary(&v);
    v
}

fn print_summary(v: &ValidationHarness) {
    println!();
    if v.all_passed() {
        println!(
            "=== airspring certification: {}/{} PASS ===",
            v.passed_count(),
            v.total_count(),
        );
    } else {
        println!(
            "=== airspring certification: {}/{} PASS, {} FAIL ===",
            v.passed_count(),
            v.total_count(),
            v.total_count() - v.passed_count(),
        );
    }
}
