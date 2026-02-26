// SPDX-License-Identifier: AGPL-3.0-or-later
//! Evolution gap and `ToadStool` issue tracking tests.
//!
//! Validates the evolution roadmap metadata: gap catalog completeness,
//! unique IDs, tier classification, and upstream issue resolution status.
//! These are structural invariants — no GPU device required.

// ── Evolution gap infrastructure ────────────────────────────────────

#[test]
fn test_evolution_gaps_catalogued() {
    use airspring_barracuda::gpu::evolution_gaps::{Tier, GAPS};

    assert!(GAPS.len() >= 8, "Expected 8+ gaps, got {}", GAPS.len());

    let tier_a = GAPS.iter().filter(|g| g.tier == Tier::A).count();
    let tier_b = GAPS.iter().filter(|g| g.tier == Tier::B).count();
    let tier_c = GAPS.iter().filter(|g| g.tier == Tier::C).count();

    assert!(tier_a >= 4, "Expected 4+ Tier A gaps, got {tier_a}");
    assert!(tier_b >= 2, "Expected 2+ Tier B gaps, got {tier_b}");
    assert!(tier_c >= 1, "Expected 1+ Tier C gaps, got {tier_c}");

    for gap in GAPS.iter().filter(|g| g.tier == Tier::A) {
        assert!(
            gap.toadstool_primitive.is_some(),
            "Tier A gap '{}' should reference a ToadStool primitive",
            gap.id
        );
    }

    for gap in GAPS {
        assert!(!gap.id.is_empty(), "Gap id must not be empty");
        assert!(
            !gap.description.is_empty(),
            "Gap description must not be empty"
        );
        assert!(!gap.action.is_empty(), "Gap action must not be empty");
    }
}

#[test]
fn test_evolution_gaps_unique_ids() {
    use airspring_barracuda::gpu::evolution_gaps::GAPS;
    use std::collections::HashSet;

    let mut seen = HashSet::new();
    for gap in GAPS {
        assert!(
            seen.insert(gap.id),
            "Duplicate evolution gap id: '{}'",
            gap.id
        );
    }
}

#[test]
fn test_batched_et0_gap_documented() {
    use airspring_barracuda::gpu::evolution_gaps::GAPS;

    let et0_gap = GAPS.iter().find(|g| g.id == "batched_et0_gpu");
    assert!(et0_gap.is_some(), "Batched ET₀ GPU gap must be documented");

    let gap = et0_gap.unwrap();
    assert!(
        gap.toadstool_primitive
            .unwrap()
            .contains("batched_elementwise"),
        "Should reference the batched elementwise shader"
    );
}

#[test]
fn test_kriging_gap_documented() {
    use airspring_barracuda::gpu::evolution_gaps::GAPS;

    let kriging_gap = GAPS.iter().find(|g| g.id == "kriging_soil_moisture");
    assert!(
        kriging_gap.is_some(),
        "Kriging soil moisture gap must be documented"
    );

    let gap = kriging_gap.unwrap();
    assert!(
        gap.toadstool_primitive.unwrap().contains("kriging"),
        "Should reference kriging_f64"
    );
}

// ── ToadStool issue tracking tests ──────────────────────────────────

#[test]
fn test_toadstool_issues_all_resolved() {
    use airspring_barracuda::gpu::evolution_gaps::{IssueStatus, TOADSTOOL_ISSUES};

    assert_eq!(
        TOADSTOOL_ISSUES.len(),
        4,
        "Expected 4 ToadStool issues, got {}",
        TOADSTOOL_ISSUES.len()
    );

    for issue in TOADSTOOL_ISSUES {
        assert_eq!(
            issue.status,
            IssueStatus::Resolved,
            "{} should be Resolved, got {:?}",
            issue.id,
            issue.status
        );
        assert!(!issue.id.is_empty());
        assert!(!issue.file.is_empty());
        assert!(!issue.summary.is_empty());
        assert!(!issue.fix.is_empty());
        assert!(!issue.blocks.is_empty());
    }
}

#[test]
fn test_toadstool_issues_by_id() {
    use airspring_barracuda::gpu::evolution_gaps::TOADSTOOL_ISSUES;

    let ts001 = TOADSTOOL_ISSUES.iter().find(|i| i.id == "TS-001").unwrap();
    assert_eq!(ts001.severity, "CRITICAL");
    assert!(ts001.file.contains("batched_elementwise"));
    assert!(ts001.fix.contains("RESOLVED"));

    let ts002 = TOADSTOOL_ISSUES.iter().find(|i| i.id == "TS-002").unwrap();
    assert_eq!(ts002.severity, "MEDIUM");
    assert!(ts002.fix.contains("RESOLVED"));

    let ts003 = TOADSTOOL_ISSUES.iter().find(|i| i.id == "TS-003").unwrap();
    assert_eq!(ts003.severity, "LOW");
    assert!(ts003.fix.contains("RESOLVED"));

    let ts004 = TOADSTOOL_ISSUES.iter().find(|i| i.id == "TS-004").unwrap();
    assert_eq!(ts004.severity, "HIGH");
    assert!(ts004.fix.contains("RESOLVED"));
}
