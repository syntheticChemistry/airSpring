// SPDX-License-Identifier: AGPL-3.0-or-later

//! Provenance Trio integration via biomeOS capability routing.
//!
//! Provides experiment session lifecycle (begin → record → complete)
//! backed by the provenance trio (rhizoCrypt + loamSpine + sweetGrass)
//! when biomeOS is running, with graceful degradation to local-only
//! operation when the trio is unavailable.
//!
//! # Architecture
//!
//! ```text
//! airSpring experiment
//!   → capability.call("dag", "create_session", ...)    → rhizoCrypt
//!   → capability.call("dag", "append_event", ...)      → rhizoCrypt
//!   → capability.call("dag", "dehydration.trigger", ...) → rhizoCrypt
//!   → capability.call("commit", "session", ...)        → loamSpine
//!   → capability.call("provenance", "create_braid", ...) → sweetGrass
//! ```
//!
//! # Graceful Degradation
//!
//! Domain logic never fails when provenance is unavailable. All functions
//! return `Ok` with a status field indicating provenance availability.
//!
//! # Reference
//!
//! Pattern: `wateringHole/SPRING_PROVENANCE_TRIO_INTEGRATION_PATTERN.md`
//! Derived from: ludoSpring V15 `barracuda/src/ipc/provenance.rs`

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::rpc::{self, Transport};

fn niche_did() -> String {
    format!("did:key:{}", crate::niche::NICHE_NAME)
}

static SESSION_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Result of a provenance operation with availability status.
#[derive(Debug, Clone)]
pub struct ProvenanceResult {
    /// Session or vertex identifier.
    pub id: String,
    /// Whether the provenance trio was reachable.
    pub available: bool,
    /// Raw response data (or degradation status).
    pub data: serde_json::Value,
}

/// Summary of a completed provenance pipeline.
#[derive(Debug, Clone)]
pub struct ProvenanceCompletion {
    /// rhizoCrypt Merkle root.
    pub merkle_root: String,
    /// loamSpine commit reference.
    pub commit_id: String,
    /// sweetGrass braid reference (empty if attribution failed).
    pub braid_id: String,
    /// Pipeline status: `"complete"`, `"partial"`, or `"unavailable"`.
    pub status: String,
}

/// Configuration for provenance transport discovery (DI pattern).
///
/// Production code uses [`ProvenanceConfig::from_env`]; tests construct
/// directly to avoid environment variable mutation.
///
/// Supports platform-agnostic transport (ecoBin standard): Unix domain
/// sockets on Unix/macOS, TCP on all platforms. Resolution order:
/// 1. `transport_override` (explicit `Transport`)
/// 2. `NEURAL_API_SOCKET` env var → Unix socket
/// 3. `NEURAL_API_ADDRESS` env var → TCP socket
/// 4. biomeOS socket directory discovery
#[derive(Debug, Clone, Default)]
pub struct ProvenanceConfig {
    /// Explicit transport override (skips all env/discovery logic).
    pub transport_override: Option<Transport>,
    /// Override for `NEURAL_API_SOCKET` env var.
    pub neural_api_socket: Option<PathBuf>,
    /// Override for `NEURAL_API_ADDRESS` env var (TCP fallback).
    pub neural_api_address: Option<std::net::SocketAddr>,
    /// Override for `BIOMEOS_SOCKET_DIR` env var.
    pub biomeos_socket_dir: Option<PathBuf>,
}

impl ProvenanceConfig {
    /// Build config from the current environment.
    #[must_use]
    pub fn from_env() -> Self {
        Self {
            transport_override: None,
            neural_api_socket: std::env::var("NEURAL_API_SOCKET").ok().map(PathBuf::from),
            neural_api_address: std::env::var("NEURAL_API_ADDRESS")
                .ok()
                .and_then(|s| s.parse().ok()),
            biomeos_socket_dir: std::env::var("BIOMEOS_SOCKET_DIR").ok().map(PathBuf::from),
        }
    }
}

/// Resolve the Neural API transport using biomeOS discovery.
///
/// Used internally by provenance operations and by other IPC consumers
/// (e.g., `NestGateProvider`) that need to route through the Neural API.
///
/// Returns a platform-agnostic [`Transport`] (Unix or TCP).
#[must_use]
pub fn resolve_neural_api_transport() -> Option<Transport> {
    resolve_neural_api_transport_with(&ProvenanceConfig::from_env())
}

/// Resolve Neural API transport with explicit config (DI variant).
pub(crate) fn resolve_neural_api_transport_with(config: &ProvenanceConfig) -> Option<Transport> {
    if let Some(ref transport) = config.transport_override {
        return Some(transport.clone());
    }

    #[cfg(unix)]
    if let Some(ref path) = config.neural_api_socket
        && path.exists()
    {
        return Some(Transport::Unix(path.clone()));
    }

    if let Some(addr) = config.neural_api_address {
        return Some(Transport::Tcp(addr));
    }

    #[cfg(unix)]
    {
        let socket_dir = crate::biomeos::resolve_socket_dir();
        let family_id = crate::biomeos::get_family_id();
        let sock_name = format!("{}-{family_id}.sock", crate::primal_names::NEURAL_API);

        let candidate = socket_dir.join(&sock_name);
        if candidate.exists() {
            return Some(Transport::Unix(candidate));
        }

        if let Some(ref dir) = config.biomeos_socket_dir {
            let p = PathBuf::from(dir).join(&sock_name);
            if p.exists() {
                return Some(Transport::Unix(p));
            }
        }
    }

    None
}

fn capability_call(
    transport: &Transport,
    capability: &str,
    operation: &str,
    args: &serde_json::Value,
) -> std::result::Result<serde_json::Value, crate::error::AirSpringError> {
    let params = serde_json::json!({
        "capability": capability,
        "operation": operation,
        "args": args,
    });

    let response = rpc::send_to(transport, "capability.call", &params)?;

    if let Some(err) = response.get("error") {
        let msg = err
            .get("message")
            .and_then(|m| m.as_str())
            .unwrap_or("unknown");
        return Err(crate::error::AirSpringError::Ipc(
            rpc::IpcError::EmptyResponse {
                method: format!("capability.call({capability}.{operation}): {msg}"),
            },
        ));
    }

    response.get("result").cloned().ok_or_else(|| {
        crate::error::AirSpringError::Ipc(rpc::IpcError::EmptyResponse {
            method: format!("capability.call({capability}.{operation})"),
        })
    })
}

fn local_session_id() -> String {
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_millis());
    let seq = SESSION_COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("local-{}-{ts}-{seq}", crate::niche::NICHE_NAME)
}

/// Begin a provenance-tracked experiment session.
///
/// Creates a DAG session in rhizoCrypt via biomeOS capability routing.
/// If the trio is unavailable, returns a local session ID and
/// `available: false` — the experiment proceeds without provenance.
#[must_use]
pub fn begin_experiment_session(experiment_name: &str) -> ProvenanceResult {
    begin_experiment_session_with(experiment_name, &ProvenanceConfig::from_env())
}

/// DI variant — accepts explicit [`ProvenanceConfig`].
#[must_use]
pub fn begin_experiment_session_with(
    experiment_name: &str,
    config: &ProvenanceConfig,
) -> ProvenanceResult {
    let Some(transport) = resolve_neural_api_transport_with(config) else {
        return ProvenanceResult {
            id: local_session_id(),
            available: false,
            data: serde_json::json!({ "provenance": "unavailable" }),
        };
    };

    let args = serde_json::json!({
        "metadata": {
            "type": "experiment",
            "name": experiment_name,
            "spring": crate::niche::NICHE_NAME,
        },
        "session_type": { "Experiment": { "spring_id": crate::niche::NICHE_NAME } },
        "description": experiment_name,
    });

    capability_call(
        &transport,
        crate::primal_names::domains::DAG,
        "create_session",
        &args,
    )
    .map_or_else(
        |_| ProvenanceResult {
            id: local_session_id(),
            available: false,
            data: serde_json::json!({ "provenance": "unavailable" }),
        },
        |result| {
            let session_id = result
                .get("session_id")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown")
                .to_string();
            ProvenanceResult {
                id: session_id.clone(),
                available: true,
                data: serde_json::json!({ "session_id": session_id }),
            }
        },
    )
}

/// Record an experiment step in the provenance DAG.
///
/// Tries Wave 17 `nest.store` signal dispatch first (biomeOS manages the
/// content put → DAG append → spine seal graph). Falls back to the legacy
/// `capability.call("dag", "append_event", ...)` when `nest.store` is
/// unavailable.
#[must_use]
pub fn record_experiment_step(session_id: &str, step: &serde_json::Value) -> ProvenanceResult {
    record_experiment_step_with(session_id, step, &ProvenanceConfig::from_env())
}

/// DI variant — accepts explicit [`ProvenanceConfig`].
#[must_use]
pub fn record_experiment_step_with(
    session_id: &str,
    step: &serde_json::Value,
    config: &ProvenanceConfig,
) -> ProvenanceResult {
    let Some(transport) = resolve_neural_api_transport_with(config) else {
        return ProvenanceResult {
            id: "unavailable".to_string(),
            available: false,
            data: serde_json::json!({ "provenance": "unavailable" }),
        };
    };

    if let Some(result) = try_nest_store_signal(&transport, session_id, step) {
        return result;
    }

    record_experiment_step_legacy(&transport, session_id, step)
}

/// Attempt `nest.store` signal dispatch (Wave 17 composition collapse).
///
/// biomeOS decomposes `nest.store` into: NestGate.content.put →
/// rhizoCrypt.dag.event.append → loamSpine.spine.seal → sweetGrass.braid.create.
/// Returns `None` if the signal is not available.
fn try_nest_store_signal(
    transport: &Transport,
    session_id: &str,
    content: &serde_json::Value,
) -> Option<ProvenanceResult> {
    let params = serde_json::json!({
        "content": content,
        "author": format!("{}:experiment", crate::niche::NICHE_NAME),
        "session_id": session_id,
    });

    let result = rpc::send_to(transport, "nest.store", &params).ok()?;

    if result.get("error").is_some() {
        return None;
    }

    let r = result.get("result").unwrap_or(&result);
    let vertex_id = r
        .get("vertex_id")
        .or_else(|| r.get("hash"))
        .or_else(|| r.get("id"))
        .and_then(|v| v.as_str())
        .unwrap_or("unknown")
        .to_string();

    Some(ProvenanceResult {
        id: vertex_id.clone(),
        available: true,
        data: serde_json::json!({ "vertex_id": vertex_id, "signal": "nest.store" }),
    })
}

/// Legacy provenance recording (pre-Wave 17).
fn record_experiment_step_legacy(
    transport: &Transport,
    session_id: &str,
    step: &serde_json::Value,
) -> ProvenanceResult {
    let args = serde_json::json!({
        "session_id": session_id,
        "event": step,
    });

    capability_call(
        transport,
        crate::primal_names::domains::DAG,
        "append_event",
        &args,
    )
    .map_or_else(
        |_| ProvenanceResult {
            id: "unavailable".to_string(),
            available: false,
            data: serde_json::json!({ "provenance": "unavailable" }),
        },
        |result| {
            let vertex_id = result
                .get("vertex_id")
                .or_else(|| result.get("id"))
                .and_then(|v| v.as_str())
                .unwrap_or("unknown")
                .to_string();
            ProvenanceResult {
                id: vertex_id.clone(),
                available: true,
                data: serde_json::json!({ "vertex_id": vertex_id }),
            }
        },
    )
}

/// Complete an experiment: dehydrate → commit → attribute.
///
/// Tries Wave 17 `nest.commit` signal dispatch first (single RPC call that
/// lets biomeOS manage the dehydrate → commit → attribute graph). Falls back
/// to the legacy three-phase provenance pipeline when `nest.commit` is
/// unavailable (pre-v3.57 biomeOS or signal dispatch not yet deployed).
///
/// Returns a [`ProvenanceCompletion`] with status indicating how far
/// the pipeline progressed. Domain logic always succeeds regardless.
#[must_use]
pub fn complete_experiment(session_id: &str) -> ProvenanceCompletion {
    complete_experiment_with(session_id, &ProvenanceConfig::from_env())
}

/// DI variant — accepts explicit [`ProvenanceConfig`].
#[must_use]
pub fn complete_experiment_with(
    session_id: &str,
    config: &ProvenanceConfig,
) -> ProvenanceCompletion {
    let Some(transport) = resolve_neural_api_transport_with(config) else {
        return ProvenanceCompletion {
            merkle_root: String::new(),
            commit_id: String::new(),
            braid_id: String::new(),
            status: "unavailable".to_string(),
        };
    };

    if let Some(completion) = try_nest_commit_signal(&transport, session_id) {
        return completion;
    }

    complete_experiment_legacy(&transport, session_id)
}

/// Attempt `nest.commit` signal dispatch (Wave 17 composition collapse).
///
/// biomeOS decomposes `nest.commit` into: rhizoCrypt.dehydrate → bearDog.sign
/// → NestGate.store → loamSpine.seal. Returns `None` if the signal is not
/// available (caller should fall back to legacy).
fn try_nest_commit_signal(
    transport: &Transport,
    session_id: &str,
) -> Option<ProvenanceCompletion> {
    let params = serde_json::json!({
        "session_id": session_id,
        "author": format!("{}:experiment", crate::niche::NICHE_NAME),
        "agents": [{
            "did": niche_did(),
            "role": "author",
            "contribution": 1.0,
        }],
    });

    let result = rpc::send_to(transport, "nest.commit", &params).ok()?;

    if result.get("error").is_some() {
        return None;
    }

    let r = result.get("result").unwrap_or(&result);
    Some(ProvenanceCompletion {
        merkle_root: r
            .get("merkle_root")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string(),
        commit_id: r
            .get("commit_id")
            .or_else(|| r.get("entry_id"))
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string(),
        braid_id: r
            .get("braid_id")
            .or_else(|| r.get("id"))
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string(),
        status: "complete".to_string(),
    })
}

/// Legacy three-phase provenance pipeline (pre-Wave 17).
fn complete_experiment_legacy(
    transport: &Transport,
    session_id: &str,
) -> ProvenanceCompletion {
    let Ok(dehydration) = capability_call(
        transport,
        crate::primal_names::domains::DAG,
        "dehydration.trigger",
        &serde_json::json!({ "session_id": session_id }),
    ) else {
        return ProvenanceCompletion {
            merkle_root: String::new(),
            commit_id: String::new(),
            braid_id: String::new(),
            status: "unavailable".to_string(),
        };
    };

    let merkle_root = dehydration
        .get("merkle_root")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    let Ok(commit_result) = capability_call(
        transport,
        crate::primal_names::domains::COMMIT,
        "session",
        &serde_json::json!({
            "summary": dehydration,
            "content_hash": merkle_root,
        }),
    ) else {
        return ProvenanceCompletion {
            merkle_root,
            commit_id: String::new(),
            braid_id: String::new(),
            status: "partial".to_string(),
        };
    };

    let commit_id = commit_result
        .get("commit_id")
        .or_else(|| commit_result.get("entry_id"))
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    let braid_id = capability_call(
        transport,
        crate::primal_names::domains::PROVENANCE,
        "create_braid",
        &serde_json::json!({
            "commit_ref": commit_id,
            "agents": [{
                "did": niche_did(),
                "role": "author",
                "contribution": 1.0,
            }],
        }),
    )
    .ok()
    .and_then(|r| {
        r.get("braid_id")
            .or_else(|| r.get("id"))
            .and_then(|v| v.as_str())
            .map(str::to_string)
    })
    .unwrap_or_default();

    ProvenanceCompletion {
        merkle_root,
        commit_id,
        braid_id,
        status: "complete".to_string(),
    }
}

/// Record GPU compute provenance for a shader chain execution.
///
/// Tracks the full pipeline: input data → shader invocation → output,
/// including precision tier (`f32`/`f64`), device info, and tolerances.
#[must_use]
pub fn record_gpu_step(
    session_id: &str,
    shader_name: &str,
    precision: &str,
    input_hash: &str,
    output_summary: &serde_json::Value,
) -> ProvenanceResult {
    record_gpu_step_with(
        session_id,
        shader_name,
        precision,
        input_hash,
        output_summary,
        &ProvenanceConfig::from_env(),
    )
}

/// DI variant — accepts explicit [`ProvenanceConfig`].
#[must_use]
pub fn record_gpu_step_with(
    session_id: &str,
    shader_name: &str,
    precision: &str,
    input_hash: &str,
    output_summary: &serde_json::Value,
    config: &ProvenanceConfig,
) -> ProvenanceResult {
    let step = serde_json::json!({
        "type": "gpu_compute",
        "shader": shader_name,
        "precision": precision,
        "input_content_hash": input_hash,
        "output_summary": output_summary,
        "backend": "barracuda_wgsl",
    });
    record_experiment_step_with(session_id, &step, config)
}

/// Check whether the provenance trio is reachable.
#[must_use]
pub fn is_available() -> bool {
    is_available_with(&ProvenanceConfig::from_env())
}

/// DI variant — accepts explicit [`ProvenanceConfig`].
#[must_use]
pub fn is_available_with(config: &ProvenanceConfig) -> bool {
    let Some(transport) = resolve_neural_api_transport_with(config) else {
        return false;
    };
    capability_call(
        &transport,
        crate::primal_names::domains::DAG,
        "health",
        &serde_json::json!({}),
    )
    .is_ok()
}

impl ProvenanceCompletion {
    /// Serialize to JSON for inclusion in experiment results.
    #[must_use]
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "provenance": self.status,
            "merkle_root": self.merkle_root,
            "commit_id": self.commit_id,
            "braid_id": self.braid_id,
        })
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test code uses unwrap for clarity")]
#[path = "provenance_tests.rs"]
mod tests;
