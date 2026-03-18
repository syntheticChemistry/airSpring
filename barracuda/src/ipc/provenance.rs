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

use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

const PROVENANCE_TIMEOUT_SECS: u64 = 10;

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

/// Configuration for provenance socket discovery (DI pattern).
///
/// Production code uses [`ProvenanceConfig::from_env`]; tests construct
/// directly to avoid environment variable mutation.
#[derive(Debug, Clone, Default)]
pub struct ProvenanceConfig {
    /// Explicit socket path override (skips all env/discovery logic).
    pub socket_override: Option<PathBuf>,
    /// Override for `NEURAL_API_SOCKET` env var.
    pub neural_api_socket: Option<PathBuf>,
    /// Override for `BIOMEOS_SOCKET_DIR` env var.
    pub biomeos_socket_dir: Option<PathBuf>,
}

impl ProvenanceConfig {
    /// Build config from the current environment.
    #[must_use]
    pub fn from_env() -> Self {
        Self {
            socket_override: None,
            neural_api_socket: std::env::var("NEURAL_API_SOCKET").ok().map(PathBuf::from),
            biomeos_socket_dir: std::env::var("BIOMEOS_SOCKET_DIR").ok().map(PathBuf::from),
        }
    }
}

/// Resolve the Neural API socket path using biomeOS discovery.
///
/// Used internally by provenance operations and by other IPC consumers
/// (e.g., `NestGateProvider`) that need to route through the Neural API.
pub(crate) fn neural_api_socket_path() -> Option<PathBuf> {
    neural_api_socket_path_with(&ProvenanceConfig::from_env())
}

/// Resolve Neural API socket with explicit config (DI variant).
pub(crate) fn neural_api_socket_path_with(config: &ProvenanceConfig) -> Option<PathBuf> {
    if let Some(ref override_path) = config.socket_override {
        return Some(override_path.clone());
    }

    if let Some(ref path) = config.neural_api_socket
        && path.exists()
    {
        return Some(path.clone());
    }

    let socket_dir = crate::biomeos::resolve_socket_dir();
    let family_id = crate::biomeos::get_family_id();
    let sock_name = format!("{}-{family_id}.sock", crate::primal_names::NEURAL_API);

    let candidate = socket_dir.join(&sock_name);
    if candidate.exists() {
        return Some(candidate);
    }

    if let Some(ref dir) = config.biomeos_socket_dir {
        let p = PathBuf::from(dir).join(&sock_name);
        if p.exists() {
            return Some(p);
        }
    }

    None
}

fn ipc_err(msg: impl Into<String>) -> crate::error::AirSpringError {
    crate::error::AirSpringError::Ipc(crate::rpc::IpcError::EmptyResponse { method: msg.into() })
}

fn capability_call(
    socket_path: &Path,
    capability: &str,
    operation: &str,
    args: &serde_json::Value,
) -> std::result::Result<serde_json::Value, crate::error::AirSpringError> {
    let request = serde_json::json!({
        "jsonrpc": "2.0",
        "method": "capability.call",
        "params": {
            "capability": capability,
            "operation": operation,
            "args": args,
        },
        "id": SESSION_COUNTER.fetch_add(1, Ordering::Relaxed),
    });

    let timeout = Duration::from_secs(PROVENANCE_TIMEOUT_SECS);
    let mut stream =
        UnixStream::connect(socket_path).map_err(|e| ipc_err(format!("connect: {e}")))?;
    stream.set_read_timeout(Some(timeout)).ok();
    stream.set_write_timeout(Some(timeout)).ok();

    let payload = serde_json::to_string(&request)?;
    stream
        .write_all(payload.as_bytes())
        .map_err(|e| ipc_err(format!("write: {e}")))?;
    stream
        .write_all(b"\n")
        .map_err(|e| ipc_err(format!("write newline: {e}")))?;
    stream.flush().map_err(|e| ipc_err(format!("flush: {e}")))?;

    let mut reader = BufReader::new(stream);
    let mut line = String::new();
    reader
        .read_line(&mut line)
        .map_err(|e| ipc_err(format!("read: {e}")))?;

    let parsed: serde_json::Value = serde_json::from_str(line.trim())?;

    if let Some(err) = parsed.get("error") {
        return Err(ipc_err(format!(
            "rpc error: {}",
            err.get("message")
                .and_then(|m| m.as_str())
                .unwrap_or("unknown")
        )));
    }

    parsed
        .get("result")
        .cloned()
        .ok_or_else(|| ipc_err("no result in response"))
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
    let Some(socket) = neural_api_socket_path_with(config) else {
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
        &socket,
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
/// Appends a vertex to the rhizoCrypt DAG for this session. The step
/// payload should include domain-specific data (method name, parameters,
/// result summary, tolerances).
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
    let Some(socket) = neural_api_socket_path_with(config) else {
        return ProvenanceResult {
            id: "unavailable".to_string(),
            available: false,
            data: serde_json::json!({ "provenance": "unavailable" }),
        };
    };

    let args = serde_json::json!({
        "session_id": session_id,
        "event": step,
    });

    capability_call(
        &socket,
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
/// Executes the three-phase provenance pipeline:
/// 1. **Dehydrate** (rhizoCrypt) — content-addressed Merkle root
/// 2. **Commit** (loamSpine) — immutable ledger entry
/// 3. **Attribute** (sweetGrass) — W3C PROV-O braid (best-effort)
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
    let Some(socket) = neural_api_socket_path_with(config) else {
        return ProvenanceCompletion {
            merkle_root: String::new(),
            commit_id: String::new(),
            braid_id: String::new(),
            status: "unavailable".to_string(),
        };
    };

    let Ok(dehydration) = capability_call(
        &socket,
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
        &socket,
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
        &socket,
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
    let Some(socket) = neural_api_socket_path_with(config) else {
        return false;
    };
    capability_call(
        &socket,
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
#[allow(clippy::unwrap_used, reason = "test code uses unwrap for clarity")]
mod tests {
    use super::*;

    fn no_socket_config() -> ProvenanceConfig {
        ProvenanceConfig {
            socket_override: None,
            neural_api_socket: None,
            biomeos_socket_dir: None,
        }
    }

    #[test]
    fn begin_session_degrades_gracefully_without_biomeos() {
        let config = no_socket_config();
        let result = begin_experiment_session_with("test_et0_validation", &config);
        assert!(!result.available);
        let prefix = format!("local-{}-", crate::niche::NICHE_NAME);
        assert!(result.id.starts_with(&prefix));
        assert_eq!(result.data["provenance"], "unavailable");
    }

    #[test]
    fn record_step_degrades_gracefully_without_biomeos() {
        let config = no_socket_config();
        let step = serde_json::json!({
            "method": "science.et0_fao56",
            "result_mm": 5.2,
        });
        let result = record_experiment_step_with("local-session-1", &step, &config);
        assert!(!result.available);
    }

    #[test]
    fn complete_experiment_degrades_gracefully_without_biomeos() {
        let config = no_socket_config();
        let completion = complete_experiment_with("local-session-1", &config);
        assert_eq!(completion.status, "unavailable");
        assert!(completion.merkle_root.is_empty());
    }

    #[test]
    fn record_gpu_step_degrades_gracefully() {
        let config = no_socket_config();
        let result = record_gpu_step_with(
            "local-session-1",
            "fao56_et0_batch",
            "f64",
            "sha256:abc123",
            &serde_json::json!({"mean_et0_mm": 4.8}),
            &config,
        );
        assert!(!result.available);
    }

    #[test]
    fn provenance_availability_false_without_biomeos() {
        let config = no_socket_config();
        assert!(!is_available_with(&config));
    }

    #[test]
    fn local_session_id_is_unique() {
        let id1 = local_session_id();
        let id2 = local_session_id();
        assert_ne!(id1, id2);
        let prefix = format!("local-{}-", crate::niche::NICHE_NAME);
        assert!(id1.starts_with(&prefix));
    }

    #[test]
    fn provenance_completion_to_json() {
        let c = ProvenanceCompletion {
            merkle_root: "abc123".to_string(),
            commit_id: "commit-456".to_string(),
            braid_id: "braid-789".to_string(),
            status: "complete".to_string(),
        };
        let j = c.to_json();
        assert_eq!(j["provenance"], "complete");
        assert_eq!(j["merkle_root"], "abc123");
        assert_eq!(j["commit_id"], "commit-456");
        assert_eq!(j["braid_id"], "braid-789");
    }

    #[test]
    fn partial_completion_to_json() {
        let c = ProvenanceCompletion {
            merkle_root: "abc123".to_string(),
            commit_id: String::new(),
            braid_id: String::new(),
            status: "partial".to_string(),
        };
        let j = c.to_json();
        assert_eq!(j["provenance"], "partial");
        assert!(!j["merkle_root"].as_str().unwrap().is_empty());
        assert!(j["commit_id"].as_str().unwrap().is_empty());
    }

    #[test]
    fn niche_did_uses_niche_name() {
        let did = niche_did();
        assert!(did.starts_with("did:key:"));
        assert!(did.contains(crate::niche::NICHE_NAME));
    }

    #[test]
    fn config_from_env_builds() {
        let config = ProvenanceConfig::from_env();
        assert!(config.socket_override.is_none());
    }
}
