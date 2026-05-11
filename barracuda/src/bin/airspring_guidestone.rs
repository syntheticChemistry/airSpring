// SPDX-License-Identifier: AGPL-3.0-or-later
//! guideStone Level 1 — standalone manifest reader and local property checks.
//!
//! Reads `downstream_manifest.toml` from primalSpring, extracts airSpring's
//! `validation_capabilities`, and cross-checks against `niche::CAPABILITIES`.
//! No `primalspring` crate dependency (path deps deprecated). Uses `toml`
//! crate for direct TOML parsing.
//!
//! Exit codes: 0 = all checks pass, 1 = drift detected, 2 = manifest not found (skip).

#![forbid(unsafe_code)]

use std::collections::BTreeSet;
use std::path::PathBuf;
use std::process::ExitCode;

use airspring_barracuda::niche;
use airspring_barracuda::primal_names;
use airspring_barracuda::validation::{self, ValidationHarness};
use tracing_subscriber::EnvFilter;

const DEFAULT_MANIFEST_PATH: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../primalSpring/graphs/downstream/downstream_manifest.toml",
);

#[derive(Debug)]
struct ManifestEntry {
    spring_name: String,
    domain: String,
    fragments: Vec<String>,
    depends_on: Vec<String>,
    validation_capabilities: Vec<String>,
}

fn resolve_manifest_path() -> PathBuf {
    if let Ok(p) = std::env::var("AIRSPRING_MANIFEST_PATH") {
        return PathBuf::from(p);
    }
    if let Ok(eco) = std::env::var("ECOPRIMALS_ROOT") {
        return PathBuf::from(eco)
            .join("springs/primalSpring/graphs/downstream/downstream_manifest.toml");
    }
    PathBuf::from(DEFAULT_MANIFEST_PATH)
}

fn extract_string_array(entry: &toml::Value, key: &str) -> Vec<String> {
    entry
        .get(key)
        .and_then(toml::Value::as_array)
        .map(|a| {
            a.iter()
                .filter_map(toml::Value::as_str)
                .map(ToOwned::to_owned)
                .collect()
        })
        .unwrap_or_default()
}

fn parse_airspring_entry(content: &str) -> Option<ManifestEntry> {
    let table: toml::Value = content.parse().ok()?;
    let downstreams = table.get("downstream")?.as_array()?;

    for entry in downstreams {
        let spring_name = entry.get("spring_name")?.as_str()?;
        if spring_name != airspring_barracuda::PRIMAL_NAME {
            continue;
        }
        let domain = entry
            .get("domain")
            .and_then(toml::Value::as_str)
            .unwrap_or("unknown")
            .to_owned();

        return Some(ManifestEntry {
            spring_name: spring_name.to_owned(),
            domain,
            fragments: extract_string_array(entry, "fragments"),
            depends_on: extract_string_array(entry, "depends_on"),
            validation_capabilities: extract_string_array(entry, "validation_capabilities"),
        });
    }
    None
}

fn validate_identity(v: &mut ValidationHarness, entry: &ManifestEntry) {
    validation::section("P1: Identity");
    v.check_bool(
        "spring_name == airspring",
        entry.spring_name == airspring_barracuda::PRIMAL_NAME,
    );
    v.check_bool(
        "domain == ecology_agriculture",
        entry.domain == "ecology_agriculture",
    );
}

fn validate_fragments(v: &mut ValidationHarness, entry: &ManifestEntry) {
    validation::section("P2: Fragment Coverage");
    for frag in ["tower_atomic", "node_atomic", "nest_atomic"] {
        v.check_bool(
            &format!("fragment:{frag}"),
            entry.fragments.iter().any(|f| f == frag),
        );
    }
}

fn validate_dependencies(v: &mut ValidationHarness, entry: &ManifestEntry) {
    validation::section("P3: Dependency Coverage");
    for dep in [
        primal_names::BEARDOG,
        primal_names::SONGBIRD,
        primal_names::CORALREEF,
        primal_names::TOADSTOOL,
        "barracuda",
        primal_names::NESTGATE,
    ] {
        v.check_bool(
            &format!("depends_on:{dep}"),
            entry.depends_on.iter().any(|d| d == dep),
        );
    }
}

fn validate_capabilities(v: &mut ValidationHarness, entry: &ManifestEntry) {
    validation::section("P4: Capability Cross-Check");
    let niche_caps: BTreeSet<&str> = niche::CAPABILITIES.iter().copied().collect();
    let manifest_caps: BTreeSet<&str> = entry
        .validation_capabilities
        .iter()
        .map(String::as_str)
        .collect();

    v.check_bool(
        "manifest validation_capabilities declared",
        !manifest_caps.is_empty(),
    );
    v.check_bool(
        "niche has more capabilities than manifest requires",
        niche_caps.len() > manifest_caps.len(),
    );

    println!();
    println!(
        "  Manifest IPC targets (consumed from primals): {}",
        manifest_caps.len(),
    );
    for cap in &manifest_caps {
        let locality = if niche_caps.contains(cap) {
            "also local"
        } else {
            "upstream only"
        };
        println!("    {cap} ({locality})");
    }
    println!(
        "  Niche capabilities (provided by airSpring): {}",
        niche_caps.len(),
    );
}

fn validate_health(v: &mut ValidationHarness) {
    validation::section("P5: Health & Discovery (Tier 1 — local only)");
    let niche_caps: BTreeSet<&str> = niche::CAPABILITIES.iter().copied().collect();
    v.check_bool(
        "capability.list in niche",
        niche_caps.contains("capability.list"),
    );
    v.check_bool(
        "health.liveness in niche",
        niche_caps.contains("health.liveness"),
    );
    v.check_bool(
        "health.readiness in niche",
        niche_caps.contains("health.readiness"),
    );
}

fn main() -> ExitCode {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::from_default_env().add_directive("info".parse().expect("valid directive")),
        )
        .with_target(false)
        .init();

    let manifest_path = resolve_manifest_path();
    let mut v = ValidationHarness::new("airspring_guidestone");

    validation::section("guideStone Level 1 — Manifest Discovery");
    println!("  manifest: {}", manifest_path.display());

    let Ok(content) = std::fs::read_to_string(&manifest_path) else {
        println!("  SKIP: manifest not readable");
        println!("  Set AIRSPRING_MANIFEST_PATH or ECOPRIMALS_ROOT to override.");
        return ExitCode::from(2);
    };

    let Some(entry) = parse_airspring_entry(&content) else {
        println!("  SKIP: no [[downstream]] entry for airspring in manifest");
        return ExitCode::from(2);
    };

    validate_identity(&mut v, &entry);
    validate_fragments(&mut v, &entry);
    validate_dependencies(&mut v, &entry);
    validate_capabilities(&mut v, &entry);
    validate_health(&mut v);

    println!();
    if v.all_passed() {
        println!(
            "=== airspring_guidestone: {}/{} PASS ===",
            v.passed_count(),
            v.total_count(),
        );
        ExitCode::from(0)
    } else {
        println!(
            "=== airspring_guidestone: {}/{} PASS, {} FAIL ===",
            v.passed_count(),
            v.total_count(),
            v.total_count() - v.passed_count(),
        );
        ExitCode::from(1)
    }
}
