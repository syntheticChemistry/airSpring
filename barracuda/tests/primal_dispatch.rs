// SPDX-License-Identifier: AGPL-3.0-or-later
//! Integration tests for the airspring niche's JSON-RPC dispatch routing.
//!
//! Spawns a server that replicates `airspring_primal`'s dispatch table using
//! the library's public API, then exercises every route through real Unix
//! socket connections. This validates the full dispatch → handler → science
//! pipeline without requiring a pre-built binary.

#![expect(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "integration test clarity"
)]

use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use airspring_barracuda::ipc::DispatchOutcome;
use airspring_barracuda::{niche, primal_science, rpc};

fn tmp_socket(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "airspring_dispatch_test_{}_{}",
        std::process::id(),
        name
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir.join(format!("{name}.sock"))
}

fn dispatch_niche(method: &str, params: &serde_json::Value) -> DispatchOutcome<serde_json::Value> {
    match method {
        "health" | "health.check" | "lifecycle.health" | "science.health" => {
            DispatchOutcome::Ok(serde_json::json!({
                "status": "healthy",
                "niche": niche::NICHE_NAME,
                "version": env!("CARGO_PKG_VERSION"),
                "capabilities": niche::CAPABILITIES,
                "backend": "cpu",
            }))
        }
        "health.liveness" => DispatchOutcome::Ok(serde_json::json!({
            "alive": true,
            "niche": niche::NICHE_NAME,
        })),
        "health.readiness" => DispatchOutcome::Ok(serde_json::json!({
            "ready": true,
            "niche": niche::NICHE_NAME,
            "version": env!("CARGO_PKG_VERSION"),
            "subsystems": {
                "science_dispatch": true,
                "provenance_trio": false,
                "nestgate": false,
                "toadstool": false,
            },
        })),
        "science.version" => DispatchOutcome::Ok(serde_json::json!({
            "niche": niche::NICHE_NAME,
            "version": env!("CARGO_PKG_VERSION"),
        })),
        "capability.list" => DispatchOutcome::Ok(serde_json::json!({
            "capabilities": niche::CAPABILITIES,
            "count": niche::CAPABILITIES.len(),
            "primal": niche::NICHE_NAME,
            "domain": "ecology",
            "total": niche::CAPABILITIES.len(),
            "operation_dependencies": niche::operation_dependencies(),
            "cost_estimates": niche::cost_estimates(),
        })),
        "provenance.begin" => {
            let name = params
                .get("experiment")
                .or_else(|| params.get("name"))
                .and_then(|v| v.as_str())
                .unwrap_or("unnamed_experiment");
            let r = airspring_barracuda::ipc::provenance::begin_experiment_session(name);
            DispatchOutcome::Ok(serde_json::json!({
                "session_id": r.id,
                "provenance": if r.available { "available" } else { "unavailable" },
                "data": r.data,
            }))
        }
        "provenance.record" => {
            let sid = params
                .get("session_id")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let step = params
                .get("step")
                .or_else(|| params.get("event"))
                .cloned()
                .unwrap_or_else(|| serde_json::json!({}));
            let r = airspring_barracuda::ipc::provenance::record_experiment_step(sid, &step);
            DispatchOutcome::Ok(serde_json::json!({
                "vertex_id": r.id,
                "provenance": if r.available { "available" } else { "unavailable" },
                "data": r.data,
            }))
        }
        "provenance.complete" => {
            let sid = params
                .get("session_id")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            DispatchOutcome::Ok(
                airspring_barracuda::ipc::provenance::complete_experiment(sid).to_json(),
            )
        }
        "provenance.status" => DispatchOutcome::Ok(serde_json::json!({
            "available": airspring_barracuda::ipc::provenance::is_available(),
            "degradation": "domain logic succeeds without provenance",
        })),
        "primal.discover" => {
            let primals = airspring_barracuda::biomeos::discover_all_primals();
            DispatchOutcome::Ok(serde_json::json!({
                "primals": primals,
                "count": primals.len(),
            }))
        }
        _ => {
            if let Some(result) = primal_science::dispatch_science(method, params) {
                return DispatchOutcome::Ok(result);
            }
            DispatchOutcome::MethodNotFound(method.to_string())
        }
    }
}

fn spawn_niche_server(path: &std::path::Path) -> (std::thread::JoinHandle<()>, Arc<AtomicBool>) {
    let path = path.to_path_buf();
    let _ = std::fs::remove_file(&path);
    let running = Arc::new(AtomicBool::new(true));
    let server_running = running.clone();

    let handle = std::thread::spawn(move || {
        let listener = UnixListener::bind(&path).expect("bind niche server");
        listener.set_nonblocking(true).expect("set_nonblocking");

        while server_running.load(Ordering::Relaxed) {
            match listener.accept() {
                Ok((stream, _)) => {
                    stream.set_read_timeout(Some(Duration::from_secs(2))).ok();
                    handle_niche_connection(&stream);
                }
                Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                    std::thread::sleep(Duration::from_millis(10));
                }
                Err(_) => break,
            }
        }
    });

    (handle, running)
}

fn handle_niche_connection(stream: &UnixStream) {
    let reader = BufReader::new(stream);
    let mut writer = stream;

    for line_result in reader.lines() {
        let Ok(line) = line_result else { break };
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        let Ok(parsed) = serde_json::from_str::<serde_json::Value>(trimmed) else {
            let resp = rpc::error(&serde_json::Value::Null, rpc::PARSE_ERROR, "Parse error");
            let _ = writeln!(writer, "{resp}");
            let _ = writer.flush();
            continue;
        };

        let id = parsed.get("id").cloned().unwrap_or(serde_json::Value::Null);
        let method = parsed.get("method").and_then(|v| v.as_str()).unwrap_or("");
        let params = parsed
            .get("params")
            .cloned()
            .unwrap_or_else(|| serde_json::json!({}));

        let response = match dispatch_niche(method, &params) {
            DispatchOutcome::Ok(result) => rpc::success(&id, &result),
            DispatchOutcome::MethodNotFound(m) => rpc::error(
                &id,
                rpc::METHOD_NOT_FOUND,
                &format!("Method not found: {m}"),
            ),
            DispatchOutcome::InvalidParams { method: m, reason } => {
                rpc::error(&id, rpc::INVALID_PARAMS, &format!("{m}: {reason}"))
            }
            DispatchOutcome::InternalError { method: m, source } => {
                rpc::error(&id, rpc::INTERNAL_ERROR, &format!("{m}: {source}"))
            }
        };

        let _ = writeln!(writer, "{response}");
        let _ = writer.flush();
    }
}

fn cleanup(path: &std::path::Path, running: &Arc<AtomicBool>, handle: std::thread::JoinHandle<()>) {
    running.store(false, Ordering::Relaxed);
    let _ = handle.join();
    std::fs::remove_file(path).ok();
    if let Some(parent) = path.parent() {
        std::fs::remove_dir(parent).ok();
    }
}

// ── Health probes ─────────────────────────────────────────────────

#[test]
fn dispatch_health_returns_healthy_with_capabilities() {
    let path = tmp_socket("d_health");
    let (handle, running) = spawn_niche_server(&path);
    std::thread::sleep(Duration::from_millis(50));

    let resp = rpc::send(&path, "health.check", &serde_json::json!({})).expect("health.check");
    let result = &resp["result"];
    assert_eq!(result["status"], "healthy");
    assert_eq!(result["niche"], "airspring");
    assert!(result["capabilities"].as_array().unwrap().len() > 20);

    cleanup(&path, &running, handle);
}

#[test]
fn dispatch_liveness_returns_alive() {
    let path = tmp_socket("d_liveness");
    let (handle, running) = spawn_niche_server(&path);
    std::thread::sleep(Duration::from_millis(50));

    let resp =
        rpc::send(&path, "health.liveness", &serde_json::json!({})).expect("health.liveness");
    assert_eq!(resp["result"]["alive"], true);
    assert_eq!(resp["result"]["niche"], "airspring");

    cleanup(&path, &running, handle);
}

#[test]
fn dispatch_readiness_reports_subsystems() {
    let path = tmp_socket("d_readiness");
    let (handle, running) = spawn_niche_server(&path);
    std::thread::sleep(Duration::from_millis(50));

    let resp =
        rpc::send(&path, "health.readiness", &serde_json::json!({})).expect("health.readiness");
    let result = &resp["result"];
    assert_eq!(result["ready"], true);
    assert!(result.get("subsystems").is_some());
    assert_eq!(result["subsystems"]["science_dispatch"], true);

    cleanup(&path, &running, handle);
}

#[test]
fn dispatch_health_aliases_all_respond() {
    let path = tmp_socket("d_aliases");
    let (handle, running) = spawn_niche_server(&path);
    std::thread::sleep(Duration::from_millis(50));

    for method in [
        "health",
        "health.check",
        "lifecycle.health",
        "science.health",
    ] {
        let resp = rpc::send(&path, method, &serde_json::json!({}))
            .unwrap_or_else(|_| panic!("{method} should respond"));
        assert_eq!(resp["result"]["status"], "healthy", "{method} mismatch");
    }

    cleanup(&path, &running, handle);
}

// ── Capability introspection ──────────────────────────────────────

#[test]
fn dispatch_capability_list_returns_full_inventory() {
    let path = tmp_socket("d_caplist");
    let (handle, running) = spawn_niche_server(&path);
    std::thread::sleep(Duration::from_millis(50));

    let resp =
        rpc::send(&path, "capability.list", &serde_json::json!({})).expect("capability.list");
    let result = &resp["result"];
    assert_eq!(result["primal"], "airspring");
    assert_eq!(result["domain"], "ecology");
    assert!(result["count"].as_u64().unwrap() > 20);
    assert!(result["total"].as_u64().unwrap() > 20);
    assert!(result.get("capabilities").is_some());
    assert!(result.get("operation_dependencies").is_some());
    assert!(result.get("cost_estimates").is_some());

    cleanup(&path, &running, handle);
}

#[test]
fn dispatch_science_version() {
    let path = tmp_socket("d_version");
    let (handle, running) = spawn_niche_server(&path);
    std::thread::sleep(Duration::from_millis(50));

    let resp =
        rpc::send(&path, "science.version", &serde_json::json!({})).expect("science.version");
    assert_eq!(resp["result"]["niche"], "airspring");
    assert!(resp["result"]["version"].as_str().is_some());

    cleanup(&path, &running, handle);
}

// ── Science dispatch ──────────────────────────────────────────────

#[test]
fn dispatch_et0_fao56_returns_et0_mm() {
    let path = tmp_socket("d_et0");
    let (handle, running) = spawn_niche_server(&path);
    std::thread::sleep(Duration::from_millis(50));

    let resp = rpc::send(
        &path,
        "science.et0_fao56",
        &serde_json::json!({"tmax": 32.5, "tmin": 18.0}),
    )
    .expect("et0_fao56");
    let et0 = resp["result"]["et0_mm"].as_f64().unwrap();
    assert!(
        et0 > 0.0 && et0 < 20.0,
        "ET₀ should be physically plausible: {et0}"
    );

    cleanup(&path, &running, handle);
}

#[test]
fn dispatch_water_balance_returns_soil_water() {
    let path = tmp_socket("d_wb");
    let (handle, running) = spawn_niche_server(&path);
    std::thread::sleep(Duration::from_millis(50));

    let resp =
        rpc::send(&path, "science.water_balance", &serde_json::json!({})).expect("water_balance");
    assert!(resp["result"]["soil_water_mm"].as_f64().is_some());
    assert!(resp["result"]["etc_mm"].as_f64().is_some());

    cleanup(&path, &running, handle);
}

#[test]
fn dispatch_richards_returns_result() {
    let path = tmp_socket("d_richards");
    let (handle, running) = spawn_niche_server(&path);
    std::thread::sleep(Duration::from_millis(50));

    let resp =
        rpc::send(&path, "science.richards_1d", &serde_json::json!({})).expect("richards_1d");
    let result = &resp["result"];
    assert!(
        result.get("mean_theta").is_some() || result.get("error").is_some(),
        "richards should return mean_theta or error"
    );

    cleanup(&path, &running, handle);
}

#[test]
fn dispatch_shannon_diversity_with_valid_counts() {
    let path = tmp_socket("d_shannon");
    let (handle, running) = spawn_niche_server(&path);
    std::thread::sleep(Duration::from_millis(50));

    let resp = rpc::send(
        &path,
        "science.shannon_diversity",
        &serde_json::json!({"counts": [10.0, 5.0, 3.0, 2.0]}),
    )
    .expect("shannon");
    let h = resp["result"]["shannon"].as_f64().unwrap();
    assert!(h > 0.0, "Shannon diversity should be positive");

    cleanup(&path, &running, handle);
}

// ── Provenance lifecycle ──────────────────────────────────────────

#[test]
fn dispatch_provenance_lifecycle_roundtrip() {
    let path = tmp_socket("d_prov");
    let (handle, running) = spawn_niche_server(&path);
    std::thread::sleep(Duration::from_millis(50));

    let begin = rpc::send(
        &path,
        "provenance.begin",
        &serde_json::json!({"experiment": "test_experiment"}),
    )
    .expect("provenance.begin");
    let session_id = begin["result"]["session_id"].as_str().unwrap();
    assert!(!session_id.is_empty());

    let record = rpc::send(
        &path,
        "provenance.record",
        &serde_json::json!({
            "session_id": session_id,
            "step": {"type": "et0_compute", "method": "fao56"}
        }),
    )
    .expect("provenance.record");
    assert!(record["result"]["vertex_id"].as_str().is_some());

    let complete = rpc::send(
        &path,
        "provenance.complete",
        &serde_json::json!({"session_id": session_id}),
    )
    .expect("provenance.complete");
    assert!(complete.get("result").is_some());

    let status =
        rpc::send(&path, "provenance.status", &serde_json::json!({})).expect("provenance.status");
    assert!(
        status["result"].get("available").is_some(),
        "provenance.status should report availability"
    );

    cleanup(&path, &running, handle);
}

// ── Discovery ─────────────────────────────────────────────────────

#[test]
fn dispatch_primal_discover_returns_list() {
    let path = tmp_socket("d_discover");
    let (handle, running) = spawn_niche_server(&path);
    std::thread::sleep(Duration::from_millis(50));

    let resp =
        rpc::send(&path, "primal.discover", &serde_json::json!({})).expect("primal.discover");
    let result = &resp["result"];
    assert!(result.get("primals").is_some());
    assert!(result.get("count").is_some());

    cleanup(&path, &running, handle);
}

// ── Error handling ────────────────────────────────────────────────

#[test]
fn dispatch_unknown_method_returns_method_not_found() {
    let path = tmp_socket("d_unknown");
    let (handle, running) = spawn_niche_server(&path);
    std::thread::sleep(Duration::from_millis(50));

    let resp = rpc::send(
        &path,
        "nonexistent.method.that.does.not.exist",
        &serde_json::json!({}),
    )
    .expect("unknown method should still return a response");
    assert!(resp.get("error").is_some());
    assert_eq!(resp["error"]["code"], rpc::METHOD_NOT_FOUND);

    cleanup(&path, &running, handle);
}

#[test]
fn dispatch_multiple_science_methods_sequentially() {
    let path = tmp_socket("d_multi");
    let (handle, running) = spawn_niche_server(&path);
    std::thread::sleep(Duration::from_millis(50));

    let methods = [
        ("science.et0_fao56", serde_json::json!({})),
        ("science.et0_hargreaves", serde_json::json!({})),
        ("science.scs_cn_runoff", serde_json::json!({})),
        ("science.gdd", serde_json::json!({})),
        (
            "science.shannon_diversity",
            serde_json::json!({"counts": [10.0, 5.0, 3.0]}),
        ),
    ];

    for (method, params) in &methods {
        let resp =
            rpc::send(&path, method, params).unwrap_or_else(|_| panic!("{method} should succeed"));
        assert!(
            resp.get("result").is_some(),
            "{method} should return result, got: {resp}"
        );
    }

    cleanup(&path, &running, handle);
}
