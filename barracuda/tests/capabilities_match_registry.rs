// SPDX-License-Identifier: AGPL-3.0-or-later
//! Sync test: verifies that `capability_registry.toml` matches `niche::CAPABILITIES`.
//!
//! Every method string in the registry must appear in the Rust constant,
//! and every Rust capability must appear in the registry. This catches
//! drift between the TOML (consumed by Songbird/biomeOS) and the Rust
//! code (used by the primal binary).

use std::collections::BTreeSet;
use std::path::PathBuf;

fn registry_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../capability_registry.toml")
}

fn extract_methods_from_registry(content: &str) -> BTreeSet<String> {
    let table: toml::Value = content
        .parse()
        .expect("capability_registry.toml must be valid TOML");

    let caps = table
        .get("capabilities")
        .and_then(toml::Value::as_table)
        .expect("registry must have [capabilities.*] tables");

    caps.values()
        .filter_map(|v| {
            v.get("method")
                .and_then(toml::Value::as_str)
                .map(ToOwned::to_owned)
        })
        .collect()
}

#[test]
fn capabilities_match_registry_toml() {
    let registry_content =
        std::fs::read_to_string(registry_path()).expect("capability_registry.toml must exist");
    let registry_methods = extract_methods_from_registry(&registry_content);

    let niche_methods: BTreeSet<String> = airspring_barracuda::niche::CAPABILITIES
        .iter()
        .map(|s| (*s).to_owned())
        .collect();

    let in_niche_not_registry: Vec<_> = niche_methods.difference(&registry_methods).collect();
    let in_registry_not_niche: Vec<_> = registry_methods.difference(&niche_methods).collect();

    assert!(
        in_niche_not_registry.is_empty(),
        "niche::CAPABILITIES has methods not in registry: {in_niche_not_registry:?}"
    );
    assert!(
        in_registry_not_niche.is_empty(),
        "registry has methods not in niche::CAPABILITIES: {in_registry_not_niche:?}"
    );

    assert_eq!(
        niche_methods.len(),
        registry_methods.len(),
        "capability count mismatch: niche={} registry={}",
        niche_methods.len(),
        registry_methods.len(),
    );
}
