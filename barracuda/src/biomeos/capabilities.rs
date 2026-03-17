// SPDX-License-Identifier: AGPL-3.0-or-later
//! Capability parsing for biomeOS primal discovery.

/// Parse capability names from flat-array, nested-object, or wrapped formats.
///
/// Handles all formats returned by diverse ecosystem sources:
/// - **Format A** — String array: `["health", "compute.dispatch"]`
/// - **Format B** — Object array: `[{"name": "health", "version": "1.0"}]`
/// - **Format C** — Nested wrapper: `{"capabilities": ["health", ...]}` (neuralSpring S156+)
/// - **Format D** — Double-nested: `{"capabilities": {"capabilities": [...]}}` (toadStool S155+)
///
/// Use when parsing `capability.list`, `health`, or `capability.discover` responses.
#[must_use]
pub fn parse_capabilities(value: &serde_json::Value) -> Vec<String> {
    if let serde_json::Value::Object(obj) = value {
        if let Some(inner) = obj.get("capabilities") {
            return parse_capabilities(inner);
        }
        if let Some(inner) = obj.get("result") {
            return parse_capabilities(inner);
        }
    }

    match value {
        serde_json::Value::Array(arr) => arr
            .iter()
            .filter_map(|v| match v {
                serde_json::Value::String(s) => Some(s.clone()),
                serde_json::Value::Object(obj) => obj
                    .get("name")
                    .or_else(|| obj.get("capability"))
                    .and_then(|n| n.as_str())
                    .map(str::to_owned),
                _ => None,
            })
            .collect(),
        _ => Vec::new(),
    }
}
