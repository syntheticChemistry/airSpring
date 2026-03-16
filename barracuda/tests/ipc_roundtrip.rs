// SPDX-License-Identifier: AGPL-3.0-or-later
//! Self-contained IPC round-trip tests.
//!
//! Spawns a minimal JSON-RPC 2.0 server in-process, exercises the rpc module's
//! framing, error handling, and protocol compliance without requiring a running
//! `airspring_primal` binary.

#![expect(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "integration test clarity"
)]

use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use airspring_barracuda::rpc;

fn tmp_socket(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "airspring_ipc_test_{}_{name}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir.join(format!("{name}.sock"))
}

fn spawn_echo_server(path: &std::path::Path) -> (std::thread::JoinHandle<()>, Arc<AtomicBool>) {
    let path = path.to_path_buf();
    let _ = std::fs::remove_file(&path);
    let running = Arc::new(AtomicBool::new(true));
    let server_running = running.clone();

    let handle = std::thread::spawn(move || {
        let listener = UnixListener::bind(&path).expect("bind");
        listener
            .set_nonblocking(true)
            .expect("set_nonblocking");

        while server_running.load(Ordering::Relaxed) {
            match listener.accept() {
                Ok((stream, _)) => {
                    stream
                        .set_read_timeout(Some(Duration::from_secs(2)))
                        .ok();
                    handle_echo_connection(stream);
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

fn handle_echo_connection(stream: UnixStream) {
    let reader = BufReader::new(&stream);
    let mut writer = &stream;

    for line_result in reader.lines() {
        let Ok(line) = line_result else { break };
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        let Ok(parsed) = serde_json::from_str::<serde_json::Value>(trimmed) else {
            let resp = rpc::error(
                &serde_json::Value::Null,
                rpc::PARSE_ERROR,
                "Parse error",
            );
            let _ = writeln!(writer, "{resp}");
            let _ = writer.flush();
            continue;
        };

        let id = parsed
            .get("id")
            .cloned()
            .unwrap_or(serde_json::Value::Null);
        let method = parsed
            .get("method")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let params = parsed
            .get("params")
            .cloned()
            .unwrap_or_else(|| serde_json::json!({}));

        let response = match method {
            "health" => rpc::success(
                &id,
                &serde_json::json!({"status": "healthy", "version": "test"}),
            ),
            "echo" => rpc::success(&id, &params),
            "" => rpc::error(&id, rpc::INVALID_REQUEST, "Missing method"),
            _ => rpc::error(
                &id,
                rpc::METHOD_NOT_FOUND,
                &format!("Method not found: {method}"),
            ),
        };

        let _ = writeln!(writer, "{response}");
        let _ = writer.flush();
    }
}

#[test]
fn rpc_health_roundtrip() {
    let path = tmp_socket("health_rt");
    let (handle, running) = spawn_echo_server(&path);
    std::thread::sleep(Duration::from_millis(50));

    let resp = rpc::send(&path, "health", &serde_json::json!({}));
    assert!(resp.is_ok(), "health should return a response");
    let resp = resp.unwrap();
    assert_eq!(resp["result"]["status"], "healthy");
    assert_eq!(resp["jsonrpc"], "2.0");

    running.store(false, Ordering::Relaxed);
    let _ = handle.join();
    std::fs::remove_file(&path).ok();
}

#[test]
fn rpc_echo_params_roundtrip() {
    let path = tmp_socket("echo_rt");
    let (handle, running) = spawn_echo_server(&path);
    std::thread::sleep(Duration::from_millis(50));

    let params = serde_json::json!({"tmax": 32.5, "station": "MSU"});
    let resp = rpc::send(&path, "echo", &params);
    assert!(resp.is_ok());
    let resp = resp.unwrap();
    assert_eq!(resp["result"]["tmax"], 32.5);
    assert_eq!(resp["result"]["station"], "MSU");

    running.store(false, Ordering::Relaxed);
    let _ = handle.join();
    std::fs::remove_file(&path).ok();
}

#[test]
fn rpc_method_not_found_returns_error_object() {
    let path = tmp_socket("mnf_rt");
    let (handle, running) = spawn_echo_server(&path);
    std::thread::sleep(Duration::from_millis(50));

    let resp = rpc::send(&path, "nonexistent.method", &serde_json::json!({}));
    assert!(resp.is_ok());
    let resp = resp.unwrap();
    assert!(
        resp.get("error").is_some(),
        "method-not-found must return error object, not result"
    );
    assert!(
        resp.get("result").is_none(),
        "method-not-found must not have result field"
    );
    assert_eq!(resp["error"]["code"], rpc::METHOD_NOT_FOUND);

    running.store(false, Ordering::Relaxed);
    let _ = handle.join();
    std::fs::remove_file(&path).ok();
}

#[test]
fn rpc_multiple_requests_same_connection_via_send() {
    let path = tmp_socket("multi_rt");
    let (handle, running) = spawn_echo_server(&path);
    std::thread::sleep(Duration::from_millis(50));

    for i in 0..5 {
        let resp = rpc::send(
            &path,
            "echo",
            &serde_json::json!({"iteration": i}),
        );
        assert!(resp.is_ok(), "request {i} should succeed");
        let resp = resp.unwrap();
        assert_eq!(resp["result"]["iteration"], i);
    }

    running.store(false, Ordering::Relaxed);
    let _ = handle.join();
    std::fs::remove_file(&path).ok();
}

#[test]
fn rpc_response_has_correct_jsonrpc_version() {
    let path = tmp_socket("version_rt");
    let (handle, running) = spawn_echo_server(&path);
    std::thread::sleep(Duration::from_millis(50));

    let resp = rpc::send(&path, "health", &serde_json::json!({}))
        .expect("should get response");
    assert_eq!(
        resp["jsonrpc"], "2.0",
        "JSON-RPC version must be 2.0"
    );

    let err_resp = rpc::send(&path, "unknown", &serde_json::json!({}))
        .expect("should get error response");
    assert_eq!(
        err_resp["jsonrpc"], "2.0",
        "JSON-RPC version must be 2.0 even for errors"
    );

    running.store(false, Ordering::Relaxed);
    let _ = handle.join();
    std::fs::remove_file(&path).ok();
}
