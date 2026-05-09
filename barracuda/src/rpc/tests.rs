// SPDX-License-Identifier: AGPL-3.0-or-later

#![expect(clippy::unwrap_used, reason = "test code uses unwrap for clarity")]

use super::*;
use std::io::Write;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};

#[test]
fn test_success() {
    let id = serde_json::json!(42);
    let result = serde_json::json!({"et0_mm": 5.2});
    let resp = success(&id, &result);
    assert_eq!(resp["jsonrpc"], "2.0");
    assert_eq!(resp["id"], 42);
    assert_eq!(resp["result"], result);
}

#[test]
fn test_error() {
    let id = serde_json::json!(1);
    let resp = error(&id, METHOD_NOT_FOUND, "method not found");
    assert_eq!(resp["jsonrpc"], "2.0");
    assert_eq!(resp["id"], 1);
    assert_eq!(resp["error"]["code"], METHOD_NOT_FOUND);
    assert_eq!(resp["error"]["message"], "method not found");
}

#[test]
fn test_error_with_null_id() {
    let id = serde_json::Value::Null;
    let resp = error(&id, PARSE_ERROR, "Parse error");
    assert_eq!(resp["error"]["code"], PARSE_ERROR);
    assert_eq!(resp["id"], serde_json::Value::Null);
}

#[test]
fn test_request() {
    let params = serde_json::json!({"tmax": 32.0});
    let req = request("science.et0_fao56", &params);
    assert_eq!(req["jsonrpc"], "2.0");
    assert_eq!(req["method"], "science.et0_fao56");
    assert_eq!(req["params"]["tmax"], 32.0);
    assert!(req.get("id").is_some());
}

#[test]
fn test_request_auto_incrementing_id() {
    let empty = serde_json::json!({});
    let req1 = request("a", &empty);
    let req2 = request("b", &empty);
    let id1 = req1["id"].as_u64().unwrap();
    let id2 = req2["id"].as_u64().unwrap();
    assert!(id2 > id1);
}

#[test]
fn normalize_method_strips_airspring_prefix() {
    assert_eq!(
        normalize_method("airspring.science.et0_fao56"),
        "science.et0_fao56"
    );
}

#[test]
fn normalize_method_leaves_bare_semantic_names_unchanged() {
    assert_eq!(normalize_method("science.et0_fao56"), "science.et0_fao56");
    assert_eq!(normalize_method("health"), "health");
}

#[test]
fn normalize_method_does_not_strip_partial_prefix() {
    assert_eq!(normalize_method("airspring"), "airspring");
}

#[test]
fn test_error_codes() {
    assert_eq!(PARSE_ERROR, -32700);
    assert_eq!(INVALID_REQUEST, -32600);
    assert_eq!(METHOD_NOT_FOUND, -32601);
    assert_eq!(INVALID_PARAMS, -32602);
    assert_eq!(INTERNAL_ERROR, -32603);
}

#[test]
fn send_returns_err_for_nonexistent_socket() {
    let path = Path::new("/nonexistent/rpc/socket/path/that/does/not/exist.sock");
    let result = send(path, "health", &serde_json::json!({}));
    assert!(result.is_err());
}

#[test]
fn success_response_has_no_error_field() {
    let id = serde_json::json!(1);
    let result = serde_json::json!({"ok": true});
    let resp = success(&id, &result);
    assert!(resp.get("error").is_none());
    assert!(resp.get("result").is_some());
}

#[test]
fn error_response_has_no_result_field() {
    let id = serde_json::json!(1);
    let resp = error(&id, INVALID_PARAMS, "invalid params");
    assert!(resp.get("result").is_none());
    assert!(resp.get("error").is_some());
}

#[test]
fn test_request_with_empty_object_params() {
    let empty = serde_json::json!({});
    let req = request("health", &empty);
    assert_eq!(req["method"], "health");
    assert_eq!(req["params"], serde_json::json!({}));
}

#[test]
fn test_request_with_array_params() {
    let params = serde_json::json!([1, 2, 3]);
    let req = request("batch", &params);
    assert_eq!(req["params"], params);
}

#[test]
fn test_request_with_null_params() {
    let params = serde_json::Value::Null;
    let req = request("notify", &params);
    assert_eq!(req["params"], serde_json::Value::Null);
}

#[test]
fn test_success_with_string_id() {
    let id = serde_json::json!("req-42");
    let result = serde_json::json!({"status": "ok"});
    let resp = success(&id, &result);
    assert_eq!(resp["id"], "req-42");
    assert_eq!(resp["result"]["status"], "ok");
}

#[test]
fn test_success_with_array_result() {
    let id = serde_json::json!(1);
    let result = serde_json::json!([1.0, 2.0, 3.0]);
    let resp = success(&id, &result);
    assert_eq!(resp["result"], result);
}

#[test]
fn test_error_with_internal_error_code() {
    let id = serde_json::json!(99);
    let resp = error(&id, INTERNAL_ERROR, "internal server error");
    assert_eq!(resp["error"]["code"], INTERNAL_ERROR);
}

#[test]
fn test_error_with_empty_message() {
    let id = serde_json::json!(1);
    let resp = error(&id, INVALID_REQUEST, "");
    assert_eq!(resp["error"]["message"], "");
}

#[test]
fn send_returns_err_when_server_sends_malformed_json() {
    let dir = std::env::temp_dir().join(format!("rpc_test_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("malformed.sock");
    let _ = std::fs::remove_file(&path);

    let listener = std::os::unix::net::UnixListener::bind(&path).unwrap();
    std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        let _ = stream.write_all(b"not valid json at all\n");
        let _ = stream.flush();
    });

    std::thread::sleep(std::time::Duration::from_millis(50));
    let result = send(&path, "health", &serde_json::json!({}));
    std::fs::remove_file(&path).ok();
    assert!(result.is_err());
}

#[test]
fn send_returns_err_when_server_sends_empty_line() {
    let dir = std::env::temp_dir().join(format!("rpc_test_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("empty.sock");
    let _ = std::fs::remove_file(&path);

    let listener = std::os::unix::net::UnixListener::bind(&path).unwrap();
    std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        let _ = stream.write_all(b"\n");
        let _ = stream.flush();
    });

    std::thread::sleep(std::time::Duration::from_millis(50));
    let result = send(&path, "health", &serde_json::json!({}));
    std::fs::remove_file(&path).ok();
    assert!(result.is_err());
}

#[test]
fn send_returns_err_when_server_sends_partial_json() {
    let dir = std::env::temp_dir().join(format!("rpc_test_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("partial.sock");
    let _ = std::fs::remove_file(&path);

    let listener = std::os::unix::net::UnixListener::bind(&path).unwrap();
    std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        let _ = stream.write_all(b"{\"jsonrpc\":\"2.0\"\n");
        let _ = stream.flush();
    });

    std::thread::sleep(std::time::Duration::from_millis(50));
    let result = send(&path, "health", &serde_json::json!({}));
    std::fs::remove_file(&path).ok();
    assert!(result.is_err());
}

#[test]
fn send_returns_ok_when_server_sends_valid_json_response() {
    let dir = std::env::temp_dir().join(format!("rpc_valid_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("valid.sock");
    let _ = std::fs::remove_file(&path);

    let listener = std::os::unix::net::UnixListener::bind(&path).unwrap();
    std::thread::spawn(move || {
        let (stream, _) = listener.accept().unwrap();
        let mut reader = std::io::BufReader::new(&stream);
        let mut req = String::new();
        std::io::BufRead::read_line(&mut reader, &mut req).unwrap();
        let mut writer = &stream;
        let _ = std::io::Write::write_all(
            &mut writer,
            b"{\"jsonrpc\":\"2.0\",\"result\":{\"ok\":true},\"id\":1}\n",
        );
        let _ = std::io::Write::flush(&mut writer);
    });

    std::thread::sleep(std::time::Duration::from_millis(100));
    let result = send(&path, "health", &serde_json::json!({}));
    std::fs::remove_file(&path).ok();
    assert!(result.is_ok());
    let resp = result.unwrap();
    assert_eq!(resp["result"]["ok"], true);
    assert_eq!(resp["id"], 1);
}

#[test]
fn send_returns_ok_when_server_sends_error_response() {
    let dir = std::env::temp_dir().join(format!("rpc_err_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("error_resp.sock");
    let _ = std::fs::remove_file(&path);

    let listener = std::os::unix::net::UnixListener::bind(&path).unwrap();
    std::thread::spawn(move || {
        let (stream, _) = listener.accept().unwrap();
        let mut reader = std::io::BufReader::new(&stream);
        let mut req = String::new();
        std::io::BufRead::read_line(&mut reader, &mut req).unwrap();
        let payload = format!(
            r#"{{"jsonrpc":"2.0","error":{{"code":{METHOD_NOT_FOUND},"message":"method not found"}},"id":1}}"#
        );
        let mut writer = &stream;
        let _ = std::io::Write::write_all(&mut writer, format!("{payload}\n").as_bytes());
        let _ = std::io::Write::flush(&mut writer);
    });

    std::thread::sleep(std::time::Duration::from_millis(100));
    let result = send(&path, "unknown", &serde_json::json!({}));
    std::fs::remove_file(&path).ok();
    assert!(result.is_ok());
    let resp = result.unwrap();
    assert!(resp.get("error").is_some());
    assert_eq!(resp["error"]["code"], METHOD_NOT_FOUND);
}

#[test]
fn resolve_transport_returns_socket_not_found_for_unknown_primal() {
    let result = resolve_transport("nonexistent_rpc_test_primal_xyz");
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(matches!(err, IpcError::SocketNotFound { .. }));
}

#[test]
fn connect_transport_returns_err_for_unreachable_tcp() {
    let addr: SocketAddr = "127.0.0.1:1".parse().unwrap();
    let transport = Transport::Tcp(addr);
    let result = connect_transport(&transport);
    assert!(result.is_err());
    assert!(matches!(
        result.unwrap_err(),
        IpcError::ConnectionFailedTcp { .. }
    ));
}

#[test]
#[cfg(unix)]
fn connect_transport_returns_err_for_nonexistent_unix_socket() {
    let path = PathBuf::from("/nonexistent/rpc/connect_test.sock");
    let transport = Transport::Unix(path);
    let result = connect_transport(&transport);
    assert!(result.is_err());
    assert!(matches!(
        result.unwrap_err(),
        IpcError::ConnectionFailed { .. }
    ));
}
