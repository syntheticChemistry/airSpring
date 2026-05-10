// SPDX-License-Identifier: AGPL-3.0-or-later
//! Cross-sync: validates airSpring shared-domain method strings against
//! the primalSpring canonical capability registry (403 methods).
//!
//! Spring-specific domains (`science.*`, `ecology.*`) are exempt — those are
//! airSpring-local methods. Shared ecosystem domains (`health`, `capability`,
//! `compute`) must align exactly. Domains where airSpring extends the canonical
//! set (`provenance`, `primal`, `data`) are documented and tracked for upstream
//! registration.

use std::collections::BTreeSet;
use std::path::PathBuf;

fn canonical_registry_path() -> Option<PathBuf> {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let candidate = manifest.join("../../../primalSpring/config/capability_registry.toml");
    candidate.exists().then_some(candidate)
}

fn local_registry_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../capability_registry.toml")
}

fn extract_methods(content: &str) -> BTreeSet<String> {
    content
        .lines()
        .filter_map(|line| {
            let trimmed = line.trim();
            if trimmed.starts_with("method = ") {
                let val = trimmed.strip_prefix("method = ")?;
                let unquoted = val.trim_matches('"');
                Some(unquoted.to_owned())
            } else if trimmed.starts_with('"') && trimmed.ends_with('"') {
                let s = trimmed.trim_matches('"');
                if s.contains('.') {
                    Some(s.to_owned())
                } else {
                    None
                }
            } else if trimmed.starts_with('"') && trimmed.ends_with("\",") {
                let s = trimmed.trim_start_matches('"').trim_end_matches("\",");
                if s.contains('.') {
                    Some(s.to_owned())
                } else {
                    None
                }
            } else {
                None
            }
        })
        .collect()
}

const SPRING_LOCAL_DOMAINS: &[&str] = &["science", "ecology"];

const ALIGNED_DOMAINS: &[&str] = &["health", "capability", "compute"];

const EXTENDING_DOMAINS: &[&str] = &["provenance", "primal", "data", "composition"];

fn domain_of(method: &str) -> &str {
    method.split('.').next().unwrap_or(method)
}

#[test]
fn shared_methods_align_with_canonical() {
    let Some(canonical_path) = canonical_registry_path() else {
        eprintln!(
            "SKIP: primalSpring canonical registry not found at expected path. \
             Run with primalSpring checked out at ../../../primalSpring/ relative to barracuda/."
        );
        return;
    };

    let canonical_content =
        std::fs::read_to_string(&canonical_path).expect("canonical registry must be readable");
    let canonical = extract_methods(&canonical_content);

    let local_content =
        std::fs::read_to_string(local_registry_path()).expect("local registry must exist");
    let local = extract_methods(&local_content);

    assert!(
        canonical.len() >= 400,
        "canonical registry should have ~403 methods, found {}",
        canonical.len()
    );

    let mut drift = Vec::new();

    for method in &local {
        let dom = domain_of(method);

        if SPRING_LOCAL_DOMAINS.contains(&dom) {
            continue;
        }

        if ALIGNED_DOMAINS.contains(&dom) && !canonical.contains(method) {
            drift.push(format!(
                "DRIFT (aligned domain): {method} — must exist in canonical"
            ));
        }
    }

    assert!(
        drift.is_empty(),
        "Shared-domain methods diverge from canonical:\n{}",
        drift.join("\n")
    );
}

#[test]
fn document_extending_methods() {
    let Some(canonical_path) = canonical_registry_path() else {
        return;
    };

    let canonical_content =
        std::fs::read_to_string(&canonical_path).expect("canonical registry must be readable");
    let canonical = extract_methods(&canonical_content);

    let local_content =
        std::fs::read_to_string(local_registry_path()).expect("local registry must exist");
    let local = extract_methods(&local_content);

    let mut extensions = Vec::new();
    for method in &local {
        let dom = domain_of(method);
        if EXTENDING_DOMAINS.contains(&dom) && !canonical.contains(method) {
            extensions.push(method.as_str());
        }
    }

    let expected_extensions = &[
        "provenance.begin",
        "provenance.record",
        "provenance.complete",
        "provenance.status",
        "primal.forward",
        "primal.discover",
        "data.cross_spring_weather",
        "data.weather",
        "composition.status",
    ];

    for ext in &extensions {
        assert!(
            expected_extensions.contains(ext),
            "Unexpected extending method: {ext} — add to expected_extensions or \
             register upstream"
        );
    }

    eprintln!(
        "INFO: {} methods extending canonical in shared domains (tracked for upstream): {:?}",
        extensions.len(),
        extensions
    );
}

#[test]
fn local_capability_count() {
    let local_content =
        std::fs::read_to_string(local_registry_path()).expect("local registry must exist");
    let local = extract_methods(&local_content);

    assert_eq!(
        local.len(),
        45,
        "expected 45 capabilities, found {}",
        local.len()
    );
}
