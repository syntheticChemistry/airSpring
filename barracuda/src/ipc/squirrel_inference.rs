// SPDX-License-Identifier: AGPL-3.0-or-later
//! Typed client for Squirrel `inference.*` methods.
//!
//! Enables airSpring's ecology science path to use Squirrel for:
//! - **Embeddings** (`inference.embed`) — embed soil sensor profiles, diversity
//!   vectors, or management histories for similarity search.
//! - **Completion** (`inference.complete`) — structured JSON crop parameter
//!   suggestions from local θ/ET₀/GDD context.
//! - **Model discovery** (`inference.models`) — prelude check for available
//!   inference providers before experiment dispatch.
//!
//! Non-fatal when Squirrel is unavailable — callers skip inference-assisted
//! paths and proceed with deterministic science only.

use crate::primal_names;
use crate::rpc::{self, IpcError, Transport};

/// Result of an `inference.embed` call.
#[derive(Debug, Clone)]
pub struct EmbedResult {
    /// The embedding vector.
    pub embedding: Vec<f64>,
    /// Model used for embedding.
    pub model: String,
    /// Dimensionality of the embedding.
    pub dimensions: usize,
}

/// Result of an `inference.complete` call.
#[derive(Debug, Clone)]
pub struct CompleteResult {
    /// The completion text or structured JSON output.
    pub content: String,
    /// Model used for completion.
    pub model: String,
    /// Token count (0 if not reported).
    pub tokens: u64,
}

/// Available inference model info from `inference.models`.
#[derive(Debug, Clone)]
pub struct InferenceModel {
    /// Model identifier.
    pub id: String,
    /// Whether this model supports embeddings.
    pub supports_embed: bool,
    /// Whether this model supports completions.
    pub supports_complete: bool,
}

/// Errors from Squirrel inference operations.
#[derive(Debug)]
pub enum InferenceError {
    /// No Squirrel primal discovered.
    NoPrimal,
    /// IPC transport error.
    Ipc(IpcError),
    /// Server returned an RPC error.
    RpcError {
        /// JSON-RPC error code.
        code: i64,
        /// Human-readable message.
        message: String,
    },
    /// Response missing expected fields.
    MalformedResponse(String),
}

impl std::fmt::Display for InferenceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoPrimal => write!(f, "no Squirrel primal discovered"),
            Self::Ipc(e) => write!(f, "IPC error: {e}"),
            Self::RpcError { code, message } => write!(f, "RPC error {code}: {message}"),
            Self::MalformedResponse(detail) => {
                write!(f, "inference malformed response: {detail}")
            }
        }
    }
}

impl std::error::Error for InferenceError {}

impl From<IpcError> for InferenceError {
    fn from(e: IpcError) -> Self {
        match e {
            IpcError::SocketNotFound { .. } => Self::NoPrimal,
            e => Self::Ipc(e),
        }
    }
}

fn discover() -> Result<Transport, InferenceError> {
    rpc::resolve_transport(primal_names::SQUIRREL).map_err(InferenceError::from)
}

fn check_rpc_error(resp: &serde_json::Value) -> Result<(), InferenceError> {
    if let Some((code, message)) = rpc::extract_rpc_error(resp) {
        return Err(InferenceError::RpcError { code, message });
    }
    Ok(())
}

/// Embed text through Squirrel for similarity search.
///
/// Domain examples: soil sensor profile JSON, diversity index vector,
/// management history summary for field matching.
///
/// # Errors
///
/// Returns [`InferenceError::NoPrimal`] if Squirrel is not discovered.
/// Returns [`InferenceError::Ipc`] on transport failure.
/// Returns [`InferenceError::RpcError`] if the server returns an RPC error.
/// Returns [`InferenceError::MalformedResponse`] if the response shape is unexpected.
pub fn embed(text: &str) -> Result<EmbedResult, InferenceError> {
    let transport = discover()?;
    embed_via(&transport, text)
}

/// Embed text against a specific transport (for testing).
///
/// # Errors
///
/// Same as [`embed`].
pub fn embed_via(transport: &Transport, text: &str) -> Result<EmbedResult, InferenceError> {
    let resp = rpc::send_to(
        transport,
        "inference.embed",
        &serde_json::json!({ "input": text }),
    )?;
    check_rpc_error(&resp)?;

    let r = resp.get("result").or(Some(&resp));
    let embedding = r
        .and_then(|v| v.get("embedding"))
        .and_then(serde_json::Value::as_array)
        .map(|arr| {
            arr.iter()
                .filter_map(serde_json::Value::as_f64)
                .collect::<Vec<_>>()
        })
        .ok_or_else(|| InferenceError::MalformedResponse("missing `embedding`".into()))?;

    let model = r
        .and_then(|v| v.get("model"))
        .and_then(serde_json::Value::as_str)
        .unwrap_or("unknown")
        .to_owned();

    let dimensions = embedding.len();

    Ok(EmbedResult {
        embedding,
        model,
        dimensions,
    })
}

/// Request a structured completion from Squirrel.
///
/// Use for crop parameter suggestions, irrigation rule generation, or
/// ecology-domain structured JSON output from field context.
///
/// # Errors
///
/// Same as [`embed`].
pub fn complete(prompt: &str) -> Result<CompleteResult, InferenceError> {
    let transport = discover()?;
    complete_via(&transport, prompt)
}

/// Request completion against a specific transport (for testing).
///
/// # Errors
///
/// Same as [`embed`].
pub fn complete_via(transport: &Transport, prompt: &str) -> Result<CompleteResult, InferenceError> {
    let resp = rpc::send_to(
        transport,
        "inference.complete",
        &serde_json::json!({ "prompt": prompt }),
    )?;
    check_rpc_error(&resp)?;

    let r = resp.get("result").or(Some(&resp));
    let content = r
        .and_then(|v| v.get("content"))
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| InferenceError::MalformedResponse("missing `content`".into()))?
        .to_owned();
    let model = r
        .and_then(|v| v.get("model"))
        .and_then(serde_json::Value::as_str)
        .unwrap_or("unknown")
        .to_owned();
    let tokens = r
        .and_then(|v| v.get("tokens"))
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);

    Ok(CompleteResult {
        content,
        model,
        tokens,
    })
}

/// List available inference models from Squirrel.
///
/// # Errors
///
/// Same as [`embed`].
pub fn list_models() -> Result<Vec<InferenceModel>, InferenceError> {
    let transport = discover()?;
    list_models_via(&transport)
}

/// List models against a specific transport (for testing).
///
/// # Errors
///
/// Same as [`embed`].
pub fn list_models_via(transport: &Transport) -> Result<Vec<InferenceModel>, InferenceError> {
    let resp = rpc::send_to(transport, "inference.models", &serde_json::json!({}))?;
    check_rpc_error(&resp)?;

    let r = resp.get("result").or(Some(&resp));
    let models = r
        .and_then(|v| v.get("models"))
        .and_then(serde_json::Value::as_array)
        .map(|arr| {
            arr.iter()
                .filter_map(|m| {
                    let id = m.get("id")?.as_str()?.to_owned();
                    let supports_embed = m
                        .get("supports_embed")
                        .and_then(serde_json::Value::as_bool)
                        .unwrap_or(false);
                    let supports_complete = m
                        .get("supports_complete")
                        .and_then(serde_json::Value::as_bool)
                        .unwrap_or(false);
                    Some(InferenceModel {
                        id,
                        supports_embed,
                        supports_complete,
                    })
                })
                .collect()
        })
        .unwrap_or_default();

    Ok(models)
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test code")]
mod tests {
    use super::*;
    use std::io::{BufRead, BufReader, Write};
    use std::net::TcpListener;

    #[test]
    fn no_primal_returns_error() {
        let result = embed("test soil profile");
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            InferenceError::NoPrimal | InferenceError::Ipc(_)
        ));
    }

    #[test]
    fn tcp_embed_round_trip() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        let addr = listener.local_addr().expect("addr");

        let server = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("accept");
            let mut reader = BufReader::new(&stream);
            let mut line = String::new();
            reader.read_line(&mut line).expect("read");
            let req: serde_json::Value = serde_json::from_str(line.trim()).expect("parse");
            let id = req.get("id").cloned().unwrap_or(serde_json::Value::Null);

            assert_eq!(req["method"], "inference.embed");
            assert!(req["params"]["input"].as_str().unwrap().contains("soil"));

            let resp = serde_json::json!({
                "jsonrpc": "2.0",
                "result": {
                    "embedding": [0.1, 0.2, 0.3, -0.1, 0.5],
                    "model": "bge-small-en",
                },
                "id": id,
            });
            let mut payload = serde_json::to_vec(&resp).expect("serialize");
            payload.push(b'\n');
            stream.write_all(&payload).expect("write");
            stream.flush().ok();
        });

        let transport = Transport::Tcp(addr);
        let result = embed_via(&transport, "soil profile: sand 40%, clay 20%, OM 2.5%");
        server.join().expect("join");

        let er = result.expect("embed via TCP");
        assert_eq!(er.embedding.len(), 5);
        assert_eq!(er.dimensions, 5);
        assert_eq!(er.model, "bge-small-en");
    }

    #[test]
    fn tcp_complete_round_trip() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        let addr = listener.local_addr().expect("addr");

        let server = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("accept");
            let mut reader = BufReader::new(&stream);
            let mut line = String::new();
            reader.read_line(&mut line).expect("read");
            let req: serde_json::Value = serde_json::from_str(line.trim()).expect("parse");
            let id = req.get("id").cloned().unwrap_or(serde_json::Value::Null);

            assert_eq!(req["method"], "inference.complete");

            let resp = serde_json::json!({
                "jsonrpc": "2.0",
                "result": {
                    "content": "{\"kc_mid\": 1.15, \"kc_end\": 0.35}",
                    "model": "llama3-8b",
                    "tokens": 42,
                },
                "id": id,
            });
            let mut payload = serde_json::to_vec(&resp).expect("serialize");
            payload.push(b'\n');
            stream.write_all(&payload).expect("write");
            stream.flush().ok();
        });

        let transport = Transport::Tcp(addr);
        let result = complete_via(&transport, "suggest Kc for corn in Michigan");
        server.join().expect("join");

        let cr = result.expect("complete via TCP");
        assert!(cr.content.contains("kc_mid"));
        assert_eq!(cr.model, "llama3-8b");
        assert_eq!(cr.tokens, 42);
    }

    #[test]
    fn tcp_list_models_round_trip() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        let addr = listener.local_addr().expect("addr");

        let server = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("accept");
            let mut reader = BufReader::new(&stream);
            let mut line = String::new();
            reader.read_line(&mut line).expect("read");
            let req: serde_json::Value = serde_json::from_str(line.trim()).expect("parse");
            let id = req.get("id").cloned().unwrap_or(serde_json::Value::Null);

            assert_eq!(req["method"], "inference.models");

            let resp = serde_json::json!({
                "jsonrpc": "2.0",
                "result": {
                    "models": [
                        {"id": "bge-small-en", "supports_embed": true, "supports_complete": false},
                        {"id": "llama3-8b", "supports_embed": false, "supports_complete": true},
                    ],
                },
                "id": id,
            });
            let mut payload = serde_json::to_vec(&resp).expect("serialize");
            payload.push(b'\n');
            stream.write_all(&payload).expect("write");
            stream.flush().ok();
        });

        let transport = Transport::Tcp(addr);
        let result = list_models_via(&transport);
        server.join().expect("join");

        let models = result.expect("models via TCP");
        assert_eq!(models.len(), 2);
        assert!(models[0].supports_embed);
        assert!(models[1].supports_complete);
    }

    #[test]
    fn tcp_rpc_error() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        let addr = listener.local_addr().expect("addr");

        let server = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("accept");
            let mut reader = BufReader::new(&stream);
            let mut line = String::new();
            reader.read_line(&mut line).expect("read");
            let req: serde_json::Value = serde_json::from_str(line.trim()).expect("parse");
            let id = req.get("id").cloned().unwrap_or(serde_json::Value::Null);
            let resp = serde_json::json!({
                "jsonrpc": "2.0",
                "error": { "code": -32601, "message": "method not found" },
                "id": id,
            });
            let mut payload = serde_json::to_vec(&resp).expect("serialize");
            payload.push(b'\n');
            stream.write_all(&payload).expect("write");
            stream.flush().ok();
        });

        let transport = Transport::Tcp(addr);
        let result = embed_via(&transport, "test");
        server.join().expect("join");

        match result {
            Err(InferenceError::RpcError { code, message }) => {
                assert_eq!(code, -32601);
                assert_eq!(message, "method not found");
            }
            other => panic!("expected RpcError, got {other:?}"),
        }
    }

    #[test]
    fn inference_error_display() {
        assert_eq!(
            format!("{}", InferenceError::NoPrimal),
            "no Squirrel primal discovered"
        );
    }

    #[test]
    fn inference_error_is_error_trait() {
        let e: Box<dyn std::error::Error> = Box::new(InferenceError::NoPrimal);
        assert!(!e.to_string().is_empty());
    }

    #[test]
    fn squirrel_transport_uses_standard_env_keys() {
        assert_eq!(
            primal_names::socket_env_var(primal_names::SQUIRREL),
            "SQUIRREL_SOCKET"
        );
    }
}
