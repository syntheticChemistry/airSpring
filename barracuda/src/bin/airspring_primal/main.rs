// SPDX-License-Identifier: AGPL-3.0-or-later

//! Transitional niche adapter for airSpring.
//!
//! airSpring is a niche deployment — not a primal. It proves scientific
//! Python baselines can be faithfully ported to sovereign Rust + GPU compute
//! using the ecoPrimals stack. The niche deploys as a biomeOS graph
//! (`graphs/airspring_niche_deploy.toml`) that composes real primals.
//!
//! This binary is the transitional adapter: a JSON-RPC 2.0 server that
//! exposes the niche's ecology capabilities via Unix domain socket until
//! biomeOS can orchestrate the niche directly from deploy graphs.
//!
//! Socket: `$XDG_RUNTIME_DIR/biomeos/airspring-{family_id}.sock`

mod discovery;
mod dispatch;
mod handlers;

use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use tracing::{error, info, warn};

use airspring_barracuda::ipc::DispatchOutcome;
use airspring_barracuda::{biomeos, niche, primal_names, rpc};

const READ_TIMEOUT_SECS: u64 = 60;
const WRITE_TIMEOUT_SECS: u64 = 10;
const HEARTBEAT_INTERVAL_SECS: u64 = 30;

struct NicheState {
    start_time: Instant,
    requests_served: AtomicU64,
}

fn register_with_biomeos(our_socket: &Path) {
    if let Some(orchestrator) = discovery::discover_orchestrator_socket() {
        info!(
            target: primal_names::BIOMEOS,
            socket = %orchestrator.display(),
            "registering with orchestrator"
        );
        niche::register_with_target(&orchestrator, our_socket);
        return;
    }
    info!(target: primal_names::BIOMEOS, "no orchestrator discovered, trying fallback");
    if let Some(fallback_name) = biomeos::fallback_registration_primal() {
        if let Some(ref fallback_sock) = biomeos::discover_primal_socket(&fallback_name) {
            info!(
                target: primal_names::BIOMEOS,
                fallback = fallback_name,
                socket = %fallback_sock.display(),
                "registering via fallback"
            );
            niche::register_with_target(fallback_sock, our_socket);
            return;
        }
        warn!(
            target: primal_names::BIOMEOS,
            fallback = fallback_name,
            "fallback primal not found — fully standalone"
        );
    }
    info!(target: primal_names::BIOMEOS, "running standalone (no orchestrator, no fallback)");
}

#[expect(
    clippy::needless_pass_by_value,
    reason = "BufReader::new consumes the stream"
)]
fn handle_connection(stream: UnixStream, state: &NicheState) {
    stream
        .set_read_timeout(Some(Duration::from_secs(READ_TIMEOUT_SECS)))
        .ok();
    stream
        .set_write_timeout(Some(Duration::from_secs(WRITE_TIMEOUT_SECS)))
        .ok();

    let reader = BufReader::new(&stream);
    let mut writer = &stream;

    for line_result in reader.lines() {
        let Ok(line) = line_result else { break };
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        let parsed: serde_json::Value = match serde_json::from_str(trimmed) {
            Ok(v) => v,
            Err(e) => {
                let resp = rpc::error(
                    &serde_json::Value::Null,
                    rpc::PARSE_ERROR,
                    &format!("Parse error: {e}"),
                );
                let _ = writeln!(writer, "{resp}");
                let _ = writer.flush();
                continue;
            }
        };

        let id = parsed.get("id").cloned().unwrap_or(serde_json::Value::Null);
        let method = parsed.get("method").and_then(|v| v.as_str()).unwrap_or("");
        let params = parsed
            .get("params")
            .cloned()
            .unwrap_or_else(|| serde_json::json!({}));

        state.requests_served.fetch_add(1, Ordering::Relaxed);

        if method.is_empty() {
            let resp = rpc::error(&id, rpc::INVALID_REQUEST, "Missing 'method' field");
            let _ = writeln!(writer, "{resp}");
            let _ = writer.flush();
            emit_metrics("<invalid>", 0.0, false);
            continue;
        }

        let t0 = Instant::now();
        let outcome = dispatch::dispatch(method, &params, state);
        let latency_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let (resp, success) = match outcome {
            DispatchOutcome::Ok(result) => (rpc::success(&id, &result), true),
            DispatchOutcome::MethodNotFound(method) => (
                rpc::error(
                    &id,
                    rpc::METHOD_NOT_FOUND,
                    &format!("Method not found: {method}"),
                ),
                false,
            ),
            DispatchOutcome::InvalidParams { method, reason } => (
                rpc::error(&id, rpc::INVALID_PARAMS, &format!("{method}: {reason}")),
                false,
            ),
            DispatchOutcome::InternalError { method, source } => (
                rpc::error(&id, rpc::INTERNAL_ERROR, &format!("{method}: {source}")),
                false,
            ),
        };
        emit_metrics(method, latency_ms, success);

        let _ = writeln!(writer, "{resp}");
        let _ = writer.flush();
    }
}

fn emit_metrics(operation: &str, latency_ms: f64, success: bool) {
    info!(
        target: "metrics",
        niche = niche::NICHE_NAME,
        operation,
        latency_ms = format!("{latency_ms:.2}"),
        success,
    );
    if let Ok(socket_path) = std::env::var("BIOMEOS_METRICS_SOCKET") {
        let payload = serde_json::json!({
            "niche": niche::NICHE_NAME,
            "operation": operation,
            "latency_ms": latency_ms,
            "success": success,
            "version": env!("CARGO_PKG_VERSION"),
        });
        if let Ok(mut stream) = std::os::unix::net::UnixStream::connect(&socket_path) {
            let s = serde_json::to_string(&payload).unwrap_or_default();
            let _ = std::io::Write::write_all(&mut stream, s.as_bytes());
            let _ = std::io::Write::write_all(&mut stream, b"\n");
        }
    }
}

fn emit_startup_audit() {
    let _ = airspring_barracuda::ipc::skunkbat::audit_startup(niche::CAPABILITIES.len());
}

fn init_tracing() {
    use tracing_subscriber::EnvFilter;
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .with_target(true)
        .init();
}

fn run() -> Result<(), String> {
    init_tracing();
    let family_id = biomeos::get_family_id();
    let socket_path = biomeos::resolve_socket_path(niche::NICHE_NAME, &family_id);

    if let Some(parent) = socket_path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("Cannot create socket directory {}: {e}", parent.display()))?;
    }
    if socket_path.exists() {
        std::fs::remove_file(&socket_path)
            .map_err(|e| format!("Cannot remove stale socket {}: {e}", socket_path.display()))?;
    }

    let state = Arc::new(NicheState {
        start_time: Instant::now(),
        requests_served: AtomicU64::new(0),
    });

    let listener = UnixListener::bind(&socket_path)
        .map_err(|e| format!("Cannot bind to {}: {e}", socket_path.display()))?;

    info!(
        target: "airspring",
        niche = niche::NICHE_NAME,
        socket = %socket_path.display(),
        family_id,
        version = env!("CARGO_PKG_VERSION"),
        capabilities = niche::CAPABILITIES.len(),
        "niche listening"
    );

    register_with_biomeos(&socket_path);
    emit_startup_audit();

    let running = Arc::new(AtomicBool::new(true));
    let heartbeat_state = state.clone();
    let heartbeat_running = running.clone();
    std::thread::spawn(move || {
        let target = discovery::discover_orchestrator_socket().or_else(|| {
            biomeos::fallback_registration_primal()
                .and_then(|name| biomeos::discover_primal_socket(&name))
        });

        while heartbeat_running.load(Ordering::Relaxed) {
            std::thread::sleep(Duration::from_secs(HEARTBEAT_INTERVAL_SECS));
            if let Some(ref t) = target {
                let _ = rpc::send(
                    t,
                    "lifecycle.status",
                    &serde_json::json!({
                        "name": niche::NICHE_NAME,
                        "socket_path": socket_path.to_string_lossy(),
                        "status": "healthy",
                        "requests_served": heartbeat_state.requests_served.load(Ordering::Relaxed),
                        "version": env!("CARGO_PKG_VERSION"),
                        "capabilities_total": niche::CAPABILITIES.len(),
                        "composition": {
                            "provenance_trio": airspring_barracuda::ipc::provenance::is_available(),
                            (primal_names::NESTGATE): discovery::discover_data_primal().is_some(),
                            (primal_names::TOADSTOOL): discovery::discover_compute_primal().is_some(),
                        },
                    }),
                );
            }
        }
    });

    info!(target: "airspring", "accepting connections");
    for stream in listener.incoming() {
        if !running.load(Ordering::Relaxed) {
            break;
        }
        match stream {
            Ok(s) => {
                let st = state.clone();
                std::thread::spawn(move || handle_connection(s, &st));
            }
            Err(e) => error!(target: "airspring", error = %e, "accept failed"),
        }
    }
    Ok(())
}

fn print_version() {
    println!(
        "{} {} (ecoPrimals/airSpring — ecological & agricultural science niche)",
        niche::NICHE_NAME,
        env!("CARGO_PKG_VERSION"),
    );
}

fn print_status() {
    let family_id = biomeos::get_family_id();
    let socket_path = biomeos::resolve_socket_path(niche::NICHE_NAME, &family_id);
    let running = socket_path.exists();
    println!("niche:        {}", niche::NICHE_NAME);
    println!("version:      {}", env!("CARGO_PKG_VERSION"));
    println!("family_id:    {family_id}");
    println!("socket:       {}", socket_path.display());
    println!("running:      {running}");
    println!("capabilities: {}", niche::CAPABILITIES.len());

    let primals = biomeos::discover_all_primals();
    println!("primals:      {} discovered", primals.len());
    for p in primals {
        println!("  - {p}");
    }

    let trio_available = airspring_barracuda::ipc::provenance::is_available();
    println!(
        "provenance:   {}",
        if trio_available {
            "available"
        } else {
            "unavailable"
        }
    );
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let subcommand = args.get(1).map_or("server", String::as_str);

    match subcommand {
        "server" | "serve" => {
            if let Err(e) = run() {
                tracing::error!(error = %e, "fatal server error");
                std::process::exit(1);
            }
        }
        "version" | "--version" | "-V" => print_version(),
        "status" => print_status(),
        "capabilities" | "caps" => {
            for cap in niche::CAPABILITIES {
                println!("{cap}");
            }
        }
        "help" | "--help" | "-h" => {
            print_version();
            println!();
            println!("Usage: airspring_primal [SUBCOMMAND]");
            println!();
            println!("Subcommands:");
            println!("  server        Start JSON-RPC 2.0 niche server (default)");
            println!("  status        Show niche status and discovered primals");
            println!("  version       Print version information");
            println!("  capabilities  List all registered capabilities");
            println!("  help          Show this help message");
        }
        other => {
            eprintln!("Unknown subcommand: {other}");
            eprintln!("Run 'airspring_primal help' for usage.");
            std::process::exit(1);
        }
    }
}
