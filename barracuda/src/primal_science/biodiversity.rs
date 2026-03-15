// SPDX-License-Identifier: AGPL-3.0-or-later
//! Biodiversity and diversity index handlers for the airSpring primal.

use crate::eco::diversity;
use serde_json::Value;

pub(super) fn shannon_diversity(params: &Value) -> Value {
    let counts: Vec<f64> = params
        .get("counts")
        .and_then(|v| v.as_array())
        .map(|a| a.iter().filter_map(serde_json::Value::as_f64).collect())
        .unwrap_or_default();
    if counts.is_empty() {
        return serde_json::json!({"error": "missing or empty 'counts' array"});
    }
    let a = diversity::alpha_diversity(&counts);
    serde_json::json!({"shannon": a.shannon, "simpson": a.simpson, "pielou": a.evenness, "observed_species": a.observed, "chao1": a.chao1, "method": "shannon_simpson_chao1"})
}

pub(super) fn bray_curtis(params: &Value) -> Value {
    let parse = |k: &str| -> Vec<f64> {
        params
            .get(k)
            .and_then(|v| v.as_array())
            .map(|a| a.iter().filter_map(serde_json::Value::as_f64).collect())
            .unwrap_or_default()
    };
    let (a, b) = (parse("sample_a"), parse("sample_b"));
    if a.is_empty() || b.is_empty() || a.len() != b.len() {
        return serde_json::json!({"error": "sample_a and sample_b must be non-empty and equal length"});
    }
    let bc = diversity::bray_curtis(&a, &b);
    serde_json::json!({"bray_curtis_dissimilarity": bc, "similarity": 1.0 - bc, "method": "bray_curtis"})
}
