// SPDX-License-Identifier: AGPL-3.0-or-later

//! Layer 0: Bare structural validation — no primals needed.
//!
//! Absorbed from the `airspring_guidestone` binary. Validates niche
//! identity, manifest fragment coverage, dependency declarations, and
//! capability cross-checks against `niche::CAPABILITIES`.

use std::collections::BTreeSet;
use std::path::PathBuf;

use crate::niche;
use crate::primal_names;
use crate::validation::ValidationHarness;

const DEFAULT_MANIFEST_PATH: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../primalSpring/graphs/downstream/downstream_manifest.toml",
);

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

struct ManifestEntry {
    spring_name: String,
    domain: String,
    fragments: Vec<String>,
    depends_on: Vec<String>,
    validation_capabilities: Vec<String>,
}

fn parse_airspring_entry(content: &str) -> Option<ManifestEntry> {
    let table: toml::Value = content.parse().ok()?;
    let downstreams = table.get("downstream")?.as_array()?;

    for entry in downstreams {
        let spring_name = entry.get("spring_name")?.as_str()?;
        if spring_name != crate::PRIMAL_NAME {
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

/// Validate all bare structural properties (Layer 0).
pub fn validate_bare_properties(v: &mut ValidationHarness) {
    let manifest_path = resolve_manifest_path();
    println!("  manifest: {}", manifest_path.display());

    let Ok(content) = std::fs::read_to_string(&manifest_path) else {
        println!("  SKIP: manifest not readable");
        println!("  Set AIRSPRING_MANIFEST_PATH or ECOPRIMALS_ROOT to override.");
        return;
    };

    let Some(entry) = parse_airspring_entry(&content) else {
        println!("  SKIP: no [[downstream]] entry for airspring in manifest");
        return;
    };

    validate_identity(v, &entry);
    validate_fragments(v, &entry);
    validate_dependencies(v, &entry);
    validate_capabilities(v, &entry);
    validate_health_caps(v);
}

fn validate_identity(v: &mut ValidationHarness, entry: &ManifestEntry) {
    v.check_bool(
        "spring_name == airspring",
        entry.spring_name == crate::PRIMAL_NAME,
    );
    v.check_bool(
        "domain == ecology_agriculture",
        entry.domain == "ecology_agriculture",
    );
}

fn validate_fragments(v: &mut ValidationHarness, entry: &ManifestEntry) {
    for frag in ["tower_atomic", "node_atomic", "nest_atomic"] {
        v.check_bool(
            &format!("fragment:{frag}"),
            entry.fragments.iter().any(|f| f == frag),
        );
    }
}

fn validate_dependencies(v: &mut ValidationHarness, entry: &ManifestEntry) {
    for dep in [
        primal_names::BEARDOG,
        primal_names::SONGBIRD,
        primal_names::CORALREEF,
        primal_names::TOADSTOOL,
        primal_names::BARRACUDA,
        primal_names::NESTGATE,
    ] {
        v.check_bool(
            &format!("depends_on:{dep}"),
            entry.depends_on.iter().any(|d| d == dep),
        );
    }
}

fn validate_capabilities(v: &mut ValidationHarness, entry: &ManifestEntry) {
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

    println!(
        "  Manifest IPC targets (consumed from primals): {}",
        manifest_caps.len(),
    );
    println!(
        "  Niche capabilities (provided by airSpring): {}",
        niche_caps.len(),
    );
}

fn validate_health_caps(v: &mut ValidationHarness) {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_string_array_present() {
        let toml: toml::Value =
            toml::from_str(r#"fruits = ["apple", "banana", "cherry"]"#).unwrap();
        let result = extract_string_array(&toml, "fruits");
        assert_eq!(result, vec!["apple", "banana", "cherry"]);
    }

    #[test]
    fn extract_string_array_missing() {
        let toml: toml::Value = toml::from_str(r#"other = 42"#).unwrap();
        assert!(extract_string_array(&toml, "fruits").is_empty());
    }

    #[test]
    fn extract_string_array_mixed_types_filters() {
        let toml: toml::Value = toml::from_str(r#"data = ["text", 42, "more"]"#).unwrap();
        let result = extract_string_array(&toml, "data");
        assert_eq!(result, vec!["text", "more"]);
    }

    #[test]
    fn parse_airspring_entry_valid() {
        let content = r#"
[[downstream]]
spring_name = "airspring"
domain = "ecology_agriculture"
fragments = ["tower_atomic", "node_atomic", "nest_atomic"]
depends_on = ["bearDog", "songBird", "coralReef", "toadStool", "barraCuda", "nestGate"]
validation_capabilities = ["science.et0_fao56", "science.water_balance"]
"#;
        let entry = parse_airspring_entry(content).expect("should parse");
        assert_eq!(entry.spring_name, "airspring");
        assert_eq!(entry.domain, "ecology_agriculture");
        assert_eq!(entry.fragments.len(), 3);
        assert_eq!(entry.depends_on.len(), 6);
        assert_eq!(entry.validation_capabilities.len(), 2);
    }

    #[test]
    fn parse_airspring_entry_missing() {
        let content = r#"
[[downstream]]
spring_name = "hotspring"
domain = "nuclear_physics"
"#;
        assert!(parse_airspring_entry(content).is_none());
    }

    #[test]
    fn parse_airspring_entry_multiple_springs() {
        let content = r#"
[[downstream]]
spring_name = "hotspring"
domain = "nuclear"

[[downstream]]
spring_name = "airspring"
domain = "ecology_agriculture"
fragments = ["tower_atomic"]
depends_on = ["bearDog"]
validation_capabilities = ["science.et0_fao56"]
"#;
        let entry = parse_airspring_entry(content).expect("should find airspring");
        assert_eq!(entry.spring_name, "airspring");
        assert_eq!(entry.fragments, vec!["tower_atomic"]);
    }

    #[test]
    fn parse_airspring_entry_invalid_toml() {
        assert!(parse_airspring_entry("not valid toml {{{}").is_none());
    }

    #[test]
    fn parse_airspring_entry_no_downstream_key() {
        assert!(parse_airspring_entry(r#"[other] key = "val""#).is_none());
    }

    #[test]
    fn validate_health_caps_all_present() {
        let mut h = ValidationHarness::new("test: health caps");
        validate_health_caps(&mut h);
        assert!(
            h.checks.iter().all(|c| c.passed),
            "health caps missing: {:?}",
            h.checks
                .iter()
                .filter(|c| !c.passed)
                .map(|c| &c.label)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn validate_identity_correct() {
        let entry = ManifestEntry {
            spring_name: crate::PRIMAL_NAME.to_string(),
            domain: "ecology_agriculture".to_string(),
            fragments: vec![],
            depends_on: vec![],
            validation_capabilities: vec![],
        };
        let mut h = ValidationHarness::new("test: identity");
        validate_identity(&mut h, &entry);
        assert!(h.checks.iter().all(|c| c.passed));
    }

    #[test]
    fn validate_fragments_full() {
        let entry = ManifestEntry {
            spring_name: String::new(),
            domain: String::new(),
            fragments: vec![
                "tower_atomic".to_string(),
                "node_atomic".to_string(),
                "nest_atomic".to_string(),
            ],
            depends_on: vec![],
            validation_capabilities: vec![],
        };
        let mut h = ValidationHarness::new("test: fragments");
        validate_fragments(&mut h, &entry);
        assert!(h.checks.iter().all(|c| c.passed));
    }

    #[test]
    fn validate_fragments_partial() {
        let entry = ManifestEntry {
            spring_name: String::new(),
            domain: String::new(),
            fragments: vec!["tower_atomic".to_string()],
            depends_on: vec![],
            validation_capabilities: vec![],
        };
        let mut h = ValidationHarness::new("test: partial frags");
        validate_fragments(&mut h, &entry);
        let failures = h.checks.iter().filter(|c| !c.passed).count();
        assert_eq!(failures, 2, "missing node_atomic + nest_atomic");
    }
}
