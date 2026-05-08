// SPDX-License-Identifier: AGPL-3.0-or-later

use crate::rpc::Transport;

use super::*;

fn no_socket_config() -> ProvenanceConfig {
    ProvenanceConfig {
        transport_override: None,
        neural_api_socket: None,
        neural_api_address: None,
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
    assert!(config.transport_override.is_none());
}

#[test]
fn resolve_transport_with_override() {
    let config = ProvenanceConfig {
        transport_override: Some(Transport::Tcp("127.0.0.1:9999".parse().unwrap())),
        neural_api_socket: None,
        neural_api_address: None,
        biomeos_socket_dir: None,
    };
    let t = resolve_neural_api_transport_with(&config);
    assert!(t.is_some());
}

#[test]
fn resolve_transport_with_tcp_address() {
    let config = ProvenanceConfig {
        transport_override: None,
        neural_api_socket: None,
        neural_api_address: Some("127.0.0.1:9998".parse().unwrap()),
        biomeos_socket_dir: None,
    };
    let t = resolve_neural_api_transport_with(&config);
    assert!(t.is_some());
    assert!(matches!(t.unwrap(), Transport::Tcp(_)));
}

fn spawn_jsonrpc_server(
    response_fn: impl Fn(&serde_json::Value) -> serde_json::Value + Send + 'static,
) -> (std::net::SocketAddr, std::thread::JoinHandle<()>) {
    use std::io::{BufRead, BufReader, Write};
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let handle = std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        let mut reader = BufReader::new(&stream);
        let mut line = String::new();
        reader.read_line(&mut line).unwrap();
        let req: serde_json::Value = serde_json::from_str(line.trim()).unwrap();
        let id = req.get("id").cloned().unwrap_or(serde_json::Value::Null);
        let mut resp = response_fn(&req);
        resp["jsonrpc"] = serde_json::json!("2.0");
        resp["id"] = id;
        let mut payload = serde_json::to_vec(&resp).unwrap();
        payload.push(b'\n');
        stream.write_all(&payload).unwrap();
        stream.flush().ok();
    });
    (addr, handle)
}

fn tcp_config(addr: std::net::SocketAddr) -> ProvenanceConfig {
    ProvenanceConfig {
        transport_override: Some(Transport::Tcp(addr)),
        neural_api_socket: None,
        neural_api_address: None,
        biomeos_socket_dir: None,
    }
}

#[test]
fn begin_session_with_live_transport() {
    let (addr, server) = spawn_jsonrpc_server(|_req| {
        serde_json::json!({
            "result": { "session_id": "rhizo-sess-001" }
        })
    });

    let config = tcp_config(addr);
    let result = begin_experiment_session_with("test_experiment", &config);
    server.join().unwrap();

    assert!(result.available);
    assert_eq!(result.id, "rhizo-sess-001");
}

#[test]
fn record_step_with_live_transport() {
    let (addr, server) = spawn_jsonrpc_server(|_req| {
        serde_json::json!({
            "result": { "vertex_id": "vtx-123" }
        })
    });

    let config = tcp_config(addr);
    let step = serde_json::json!({ "method": "science.et0_fao56", "result_mm": 5.2 });
    let result = record_experiment_step_with("sess-001", &step, &config);
    server.join().unwrap();

    assert!(result.available);
    assert_eq!(result.id, "vtx-123");
}

#[test]
fn record_gpu_step_with_live_transport() {
    let (addr, server) = spawn_jsonrpc_server(|_req| {
        serde_json::json!({
            "result": { "vertex_id": "gpu-vtx-456" }
        })
    });

    let config = tcp_config(addr);
    let result = record_gpu_step_with(
        "sess-001",
        "fao56_et0_batch",
        "f64",
        "sha256:abc",
        &serde_json::json!({"mean_et0_mm": 4.8}),
        &config,
    );
    server.join().unwrap();

    assert!(result.available);
    assert_eq!(result.id, "gpu-vtx-456");
}

#[test]
fn begin_session_handles_rpc_error() {
    let (addr, server) = spawn_jsonrpc_server(|_req| {
        serde_json::json!({
            "error": { "code": -32600, "message": "invalid" }
        })
    });

    let config = tcp_config(addr);
    let result = begin_experiment_session_with("test_err", &config);
    server.join().unwrap();

    assert!(!result.available);
    assert!(result.id.starts_with("local-"));
}

#[test]
fn provenance_result_fields_accessible() {
    let r = ProvenanceResult {
        id: "test".to_string(),
        available: true,
        data: serde_json::json!({"test": true}),
    };
    assert_eq!(r.id, "test");
    assert!(r.available);
}
