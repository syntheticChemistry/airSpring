// SPDX-License-Identifier: AGPL-3.0-or-later

//! airSpring `UniBin` — the eukaryotic cell.
//!
//! Single binary consolidating certification (guidestone organelle),
//! validation scenarios (experiment ribosomes), and IPC server (cell membrane).
//!
//! Evolved from the prokaryotic era of separate binaries during the
//! interstadial transition.

#![forbid(unsafe_code)]

mod cli;

use clap::Parser;

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .with_target(true)
        .init();

    let parsed = cli::Cli::parse();

    match parsed.command {
        cli::Commands::Certify { layer, bare } => cmd_certify(layer, bare),
        cli::Commands::Validate {
            ref track,
            ref scenario,
            ref tier,
            list,
            format,
            ref provenance_dir,
        } => cmd_validate(
            track.as_deref(),
            scenario.as_deref(),
            tier.as_deref(),
            list,
            format,
            provenance_dir.as_deref(),
        ),
        cli::Commands::Serve => cmd_serve(),
        cli::Commands::Status => cmd_status(),
        cli::Commands::Version => cmd_version(),
    }
}

fn cmd_certify(layer: Option<u8>, bare: bool) {
    let max_layer = if bare {
        0
    } else {
        layer.unwrap_or(airspring_barracuda::certification::MAX_LAYER)
    };

    let result = airspring_barracuda::certification::certify(max_layer);
    if result.all_passed() {
        std::process::exit(0);
    } else {
        std::process::exit(1);
    }
}

#[expect(
    clippy::too_many_lines,
    reason = "validation harness dispatch — sequential scenario runner"
)]
fn cmd_validate(
    track: Option<&str>,
    scenario_id: Option<&str>,
    tier: Option<&str>,
    list: bool,
    format: cli::OutputFormat,
    provenance_dir: Option<&str>,
) {
    use airspring_barracuda::validation::scenarios::{Tier, Track, build_registry};

    let registry = build_registry();
    let json_mode = matches!(format, cli::OutputFormat::Json);

    if list {
        if json_mode {
            let items: Vec<serde_json::Value> = registry
                .all()
                .iter()
                .map(|s| {
                    serde_json::json!({
                        "id": s.meta.id,
                        "track": s.meta.track.to_string(),
                        "tier": s.meta.tier.to_string(),
                        "provenance_crate": s.meta.provenance_crate,
                    })
                })
                .collect();
            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({
                    "scenarios": items,
                    "count": items.len(),
                }))
                .unwrap_or_default()
            );
        } else {
            println!(
                "airSpring Validation Scenarios ({} registered)\n",
                registry.len()
            );
            let hdr_scenario = "SCENARIO";
            let hdr_track = "TRACK";
            let hdr_tier = "TIER";
            let hdr_provenance = "PROVENANCE";
            println!("{hdr_scenario:<30} {hdr_track:<20} {hdr_tier:<6} {hdr_provenance}");
            println!("{}", "-".repeat(80));
            for s in registry.all() {
                println!(
                    "{:<30} {:<20} {:<6} {}",
                    s.meta.id, s.meta.track, s.meta.tier, s.meta.provenance_crate
                );
            }
        }
        return;
    }

    let tier_filter: Option<Tier> = tier.map(|t| match t {
        "rust" => Tier::Rust,
        "live" => Tier::Live,
        "both" | "all" => Tier::Both,
        _ => {
            eprintln!("unknown tier: {t} (expected: rust, live, both)");
            std::process::exit(1);
        }
    });

    let track_filter: Option<Track> = track.and_then(|t| {
        Track::from_str_loose(t).or_else(|| {
            eprintln!("unknown track: {t}");
            std::process::exit(1);
        })
    });

    let mut v = airspring_barracuda::validation::ValidationHarness::new(
        "airSpring Validation — Scenario Runner",
    );

    if !json_mode {
        airspring_barracuda::validation::banner("airSpring Validation — Scenario Runner");
    }

    let mut ran = 0usize;
    for s in registry.all() {
        if let Some(id) = scenario_id
            && s.meta.id != id
        {
            continue;
        }
        if let Some(track_f) = track_filter
            && s.meta.track != track_f
        {
            continue;
        }
        if let Some(tier_f) = tier_filter
            && tier_f != Tier::Both
            && s.meta.tier != tier_f
            && s.meta.tier != Tier::Both
        {
            continue;
        }

        if !json_mode {
            airspring_barracuda::validation::section(&format!(
                "Scenario: {} [{}] ({})",
                s.meta.id, s.meta.track, s.meta.tier
            ));
        }
        (s.run)(&mut v);
        ran += 1;
    }

    if ran == 0 {
        if json_mode {
            println!(
                "{}",
                serde_json::json!({"error": "no scenarios matched the filter criteria"})
            );
        } else {
            eprintln!("no scenarios matched the filter criteria");
        }
        std::process::exit(1);
    }

    if let Some(dir) = provenance_dir {
        write_provenance_artifacts(dir, &v);
    }

    if json_mode {
        let report = airspring_barracuda::validation::harness_to_json(&v);
        println!(
            "{}",
            serde_json::to_string_pretty(&report).unwrap_or_default()
        );
        if v.all_passed() {
            std::process::exit(0);
        } else {
            std::process::exit(1);
        }
    } else {
        v.finish();
    }
}

fn write_provenance_artifacts(dir: &str, v: &airspring_barracuda::validation::ValidationHarness) {
    let path = std::path::Path::new(dir);
    if let Err(e) = std::fs::create_dir_all(path) {
        eprintln!("warning: cannot create provenance dir {dir}: {e}");
        return;
    }

    let report = airspring_barracuda::validation::harness_to_json(v);
    let results_path = path.join("results.json");
    if let Ok(json) = serde_json::to_string_pretty(&report)
        && let Err(e) = std::fs::write(&results_path, json)
    {
        eprintln!("warning: cannot write results.json: {e}");
    }

    let provenance = format!(
        "[provenance]\n\
         spring = \"airspring\"\n\
         version = \"{}\"\n\
         date = \"{}\"\n\
         passed = {}\n\
         failed = {}\n\
         total = {}\n\
         guidestone_level = \"L4\"\n\
         thread = 6\n\
         thread_name = \"Agricultural Science\"\n",
        env!("CARGO_PKG_VERSION"),
        chrono_date_stub(),
        v.passed_count(),
        v.total_count() - v.passed_count(),
        v.total_count(),
    );
    let prov_path = path.join("provenance.toml");
    if let Err(e) = std::fs::write(&prov_path, provenance) {
        eprintln!("warning: cannot write provenance.toml: {e}");
    }

    println!("  provenance artifacts written to {dir}/");
}

fn chrono_date_stub() -> String {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let days = now / 86400;
    let years = 1970 + days / 365;
    format!("{years}-05-16")
}

fn dispatch_serve(
    method: &str,
    params: &serde_json::Value,
    id: &serde_json::Value,
) -> serde_json::Value {
    use airspring_barracuda::{niche, primal_science, rpc};

    match method {
        "health.liveness" | "lifecycle.health" | "health" | "health.check" => rpc::success(
            id,
            &serde_json::json!({"status": "ok", "primal": niche::NICHE_NAME}),
        ),
        "health.readiness" => rpc::success(
            id,
            &serde_json::json!({"status": "ok", "primal": niche::NICHE_NAME, "ready": true}),
        ),
        "capability.list" => {
            let caps: Vec<&str> = niche::CAPABILITIES.to_vec();
            rpc::success(
                id,
                &serde_json::json!({"capabilities": caps, "count": caps.len()}),
            )
        }
        "science.version" => rpc::success(
            id,
            &serde_json::json!({"niche": niche::NICHE_NAME, "version": env!("CARGO_PKG_VERSION")}),
        ),
        _ => primal_science::dispatch_science(method, params).map_or_else(
            || {
                rpc::error(
                    id,
                    rpc::METHOD_NOT_FOUND,
                    &format!("Method not found: {method}"),
                )
            },
            |result| rpc::success(id, &result),
        ),
    }
}

fn cmd_serve() {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

    use airspring_barracuda::{biomeos, niche, primal_names};

    let family_id = biomeos::get_family_id();
    let socket_path = biomeos::resolve_socket_path(niche::NICHE_NAME, &family_id);

    if let Some(parent) = socket_path.parent()
        && let Err(e) = std::fs::create_dir_all(parent)
    {
        tracing::error!(error = %e, "failed to create socket directory");
        std::process::exit(1);
    }
    if socket_path.exists() {
        let _ = std::fs::remove_file(&socket_path);
    }

    let requests_served = Arc::new(AtomicU64::new(0));
    let listener = match std::os::unix::net::UnixListener::bind(&socket_path) {
        Ok(l) => l,
        Err(e) => {
            tracing::error!(error = %e, "failed to bind Unix socket");
            std::process::exit(1);
        }
    };

    tracing::info!(
        niche = niche::NICHE_NAME,
        socket = %socket_path.display(),
        family_id,
        version = env!("CARGO_PKG_VERSION"),
        capabilities = niche::CAPABILITIES.len(),
        "airspring UniBin serving"
    );

    niche::register_with_target(
        &biomeos::resolve_socket_dir().join(primal_names::socket_filename(primal_names::BIOMEOS)),
        &socket_path,
    );

    let running = Arc::new(AtomicBool::new(true));
    spawn_heartbeat(running.clone(), &socket_path, requests_served.clone());

    tracing::info!("accepting connections");
    for stream in listener.incoming() {
        if !running.load(Ordering::Relaxed) {
            break;
        }
        match stream {
            Ok(s) => {
                let reqs = requests_served.clone();
                std::thread::spawn(move || handle_serve_connection(s, &reqs));
            }
            Err(e) => tracing::error!(error = %e, "accept failed"),
        }
    }
}

fn spawn_heartbeat(
    running: std::sync::Arc<std::sync::atomic::AtomicBool>,
    socket_path: &std::path::Path,
    requests: std::sync::Arc<std::sync::atomic::AtomicU64>,
) {
    use std::sync::atomic::Ordering;

    use airspring_barracuda::{biomeos, niche, primal_names, rpc};

    let sock = socket_path.to_path_buf();
    std::thread::spawn(move || {
        while running.load(Ordering::Relaxed) {
            std::thread::sleep(std::time::Duration::from_secs(30));
            let orchestrator = biomeos::resolve_socket_dir()
                .join(primal_names::socket_filename(primal_names::BIOMEOS));
            if orchestrator.exists() {
                let _ = rpc::send(
                    &orchestrator,
                    "lifecycle.status",
                    &serde_json::json!({
                        "name": niche::NICHE_NAME,
                        "socket_path": sock.to_string_lossy(),
                        "status": "healthy",
                        "requests_served": requests.load(Ordering::Relaxed),
                        "version": env!("CARGO_PKG_VERSION"),
                        "capabilities_total": niche::CAPABILITIES.len(),
                    }),
                );
            }
        }
    });
}

#[expect(
    clippy::needless_pass_by_value,
    reason = "stream ownership transfers from thread::spawn"
)]
fn handle_serve_connection(
    stream: std::os::unix::net::UnixStream,
    requests: &std::sync::atomic::AtomicU64,
) {
    use std::io::{BufRead, BufReader, Write};
    use std::sync::atomic::Ordering;
    use std::time::Duration;

    use airspring_barracuda::rpc;

    stream.set_read_timeout(Some(Duration::from_secs(60))).ok();
    stream.set_write_timeout(Some(Duration::from_secs(10))).ok();

    let reader = BufReader::new(&stream);
    let mut writer = &stream;

    for line_result in reader.lines() {
        let Ok(line) = line_result else { break };
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        requests.fetch_add(1, Ordering::Relaxed);

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

        let resp = dispatch_serve(method, &params, &id);

        let _ = writeln!(writer, "{resp}");
        let _ = writer.flush();
    }
}

fn cmd_status() {
    use airspring_barracuda::{biomeos, ipc, niche};

    let family_id = biomeos::get_family_id();
    let socket_path = biomeos::resolve_socket_path(niche::NICHE_NAME, &family_id);
    let running = socket_path.exists();

    println!("airspring v{} (UniBin)", env!("CARGO_PKG_VERSION"));
    println!("niche:        {}", niche::NICHE_NAME);
    println!("family_id:    {family_id}");
    println!("socket:       {}", socket_path.display());
    println!("running:      {running}");
    println!("capabilities: {}", niche::CAPABILITIES.len());

    let primals = biomeos::discover_all_primals();
    println!("primals:      {} discovered", primals.len());
    for p in primals {
        println!("  - {p}");
    }

    let trio_available = ipc::provenance::is_available();
    println!(
        "provenance:   {}",
        if trio_available {
            "available"
        } else {
            "unavailable"
        }
    );
}

fn cmd_version() {
    println!("airspring {} (UniBin)", env!("CARGO_PKG_VERSION"));
}
