# airSpring V0.8.8 — Cross-Ecosystem Absorption Handoff

**Date**: March 16, 2026
**From**: airSpring V0.8.8
**To**: toadStool, coralReef, barraCuda, biomeOS, sibling springs
**License**: AGPL-3.0-or-later

---

## Executive Summary

- Absorbed high-priority patterns from 6 sibling springs and 5 primals
- `thiserror` replaces all manual `Display`/`Error` impls (error.rs, rpc.rs)
- `health.liveness` + `health.readiness` probes (healthSpring V32 pattern)
- Circuit breaker + exponential backoff for IPC resilience (healthSpring pattern)
- `OrExit<T>` trait + `parse_benchmark()` (groundSpring V112 / wetSpring V125)
- `IpcError` expanded: `WriteFailed`, `ReadFailed`, `SocketNotFound`, `EmptyResponse` + `is_recoverable()`
- `socket_env_var()` + `address_env_var()` generic discovery (groundSpring pattern)
- Structured `tracing` replaces all `eprintln!` in primal binary
- Workspace-level `deny.toml`
- 880 lib + 22 property tests (up from 872)

## What Changed

### 1. `thiserror` for All Error Types

Replaced manual `std::fmt::Display` and `std::error::Error` implementations:
- `AirSpringError` — 8 variants with `#[from]` for `io::Error`, `serde_json::Error`, `BarracudaError`, and `IpcError`
- `IpcError` — 8 variants (expanded from 4) with `#[error("...")]` formatting

### 2. Structured `IpcError` (8 variants)

| Variant | New? | Recoverable? |
|---------|------|-------------|
| `ConnectionFailed` | No | Yes |
| `WriteFailed` | **Yes** | Yes |
| `ReadFailed` | **Yes** | Yes |
| `Timeout` | No | Yes |
| `RpcError` | No | No |
| `DeserializationFailed` | No | No |
| `SocketNotFound` | **Yes** | No |
| `EmptyResponse` | **Yes** | No |

`IpcError::is_recoverable()` returns `true` for transient failures, enabling retry logic.

### 3. Health Probes

| Method | Response | Purpose |
|--------|----------|---------|
| `health.liveness` | `{"alive": true, "niche": "airspring"}` | Minimal process check |
| `health.readiness` | `{"ready": true, "subsystems": {...}}` | Full subsystem status |

Added to niche CAPABILITIES, dispatch, and handlers.

### 4. Circuit Breaker (`ipc::resilience`)

- `CircuitBreaker`: opens after 3 consecutive failures, auto-resets after 5s cooldown
- `resilient_send()`: retry with exponential backoff (50ms → 100ms → fail)
- Only retries on recoverable errors; non-recoverable errors fail immediately

### 5. `OrExit<T>` + `parse_benchmark()`

- `OrExit<T>` trait on `Result<T, E>` and `Option<T>` — `eprintln! + exit(1)` instead of `panic!()`
- `parse_benchmark(json_str)` — one-call benchmark loading for validation binaries
- Absorbed from groundSpring V112 / wetSpring V125 ecosystem pattern

### 6. Generic Discovery Functions

- `primal_names::socket_env_var("toadstool")` → `"TOADSTOOL_SOCKET"`
- `primal_names::address_env_var("nestgate")` → `"NESTGATE_ADDRESS"`
- Eliminates ad-hoc env var name construction across the codebase

### 7. Structured Tracing in Primal Binary

All `eprintln!` calls in `airspring_primal` binary replaced with `tracing::info!`/`warn!`/`error!`:
- `register_with_biomeos()` — structured target/socket fields
- `emit_metrics()` — structured niche/operation/latency fields
- Server startup — single structured log with all boot info
- Accept loop — `error!` with error field

### 8. Workspace `deny.toml`

Added root-level `deny.toml` (matches `barracuda/deny.toml`) for consistent
supply-chain hygiene across `barracuda/` and `metalForge/forge/`.

### 9. Clippy Debt Resolution

- Fixed 30+ pre-existing missing doc warnings in `eco/anderson.rs`, `eco/drought_index.rs`, `eco/dual_kc.rs`, `eco/et0_ensemble.rs`, `eco/soil_moisture.rs`, `gpu/simple_et0.rs`
- Fixed unfulfilled `#[expect]` lint annotations (3 files)
- Fixed `#[must_use]` on `Result`-returning `rpc::send()`

## Metrics

| Metric | Before (v0.8.7) | After (v0.8.8) |
|--------|-----------------|-----------------|
| Library tests | 872 | **880** |
| Property tests | 22 | 22 |
| Clippy warnings (lib) | 0 | 0 |
| `#[allow()]` in production | 0 | 0 |
| `unsafe` in production | 0 | 0 |
| Hardcoded primal names | 0 | 0 |
| `eprintln!` in primal binary | 12 | **0** |
| IpcError variants | 4 | **8** |
| Error types with thiserror | 0 | **2** |

## Ecosystem Alignment

| Pattern | Source | airSpring Status |
|---------|--------|------------------|
| `health.liveness`/`readiness` | healthSpring V32 | **Implemented** |
| Circuit breaker + backoff | healthSpring V32 | **Implemented** |
| `IpcError::is_recoverable()` | neuralSpring S161 | **Implemented** |
| `OrExit<T>` trait | groundSpring V112, wetSpring V125 | **Implemented** |
| `thiserror` for errors | groundSpring V112, healthSpring V32 | **Implemented** |
| `socket_env_var()` | groundSpring V112, wetSpring V125 | **Implemented** |
| Structured tracing | neuralSpring S161, healthSpring V32 | **Implemented** |
| Workspace `deny.toml` | wetSpring V125, healthSpring V32 | **Implemented** |
| Edition 2024 | All springs | Already implemented |
| Zero C deps | Ecosystem standard | Already implemented |

## Next Steps (V0.8.9 Candidates)

- Migrate validation binaries to use `OrExit<T>` and `parse_benchmark()`
- Wire `resilient_send()` into provenance trio IPC calls
- Expand `IpcError` structured variants to all IPC call sites
- Profile `compute_gpu()` at N=100K+ for atlas-scale crossover analysis

## Superseded Handoffs (archived)

- `AIRSPRING_V087_ECOSYSTEM_ABSORPTION_HANDOFF_MAR16_2026.md` → `archive/`
- `AIRSPRING_V087_BARRACUDA_TOADSTOOL_EVOLUTION_HANDOFF_MAR16_2026.md` → `archive/`

---
*ScyBorg Provenance Trio: AGPL-3.0-or-later (code) + ORC (game mechanics) + CC-BY-SA 4.0 (creative content)*
