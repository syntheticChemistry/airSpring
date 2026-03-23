# Handoff: airSpring → barraCuda Team — Transport Absorption & Evolution Findings

**Date:** 2026-03-23
**From:** airSpring V0.10.0
**To:** barraCuda / toadStool team
**License:** AGPL-3.0-or-later

---

## Executive Summary

airSpring completed platform-agnostic IPC migration and deep audit. This handoff
documents patterns, findings, and absorption candidates relevant to the barraCuda
and toadStool evolution.

---

## 1. Transport Pattern (Absorption Candidate)

airSpring's `ipc/provenance.rs` previously used raw `UnixStream` directly
(bypassing the `rpc::Transport` abstraction). This was a sovereignty violation —
Unix-only, not cross-platform. The fix:

```rust
// BEFORE: Unix-only, hardcoded
let stream = UnixStream::connect(socket_path)?;
stream.write_all(payload.as_bytes())?;

// AFTER: Platform-agnostic via Transport
let response = rpc::send_to(&transport, "capability.call", &params)?;
```

**Recommendation for barraCuda**: If any upstream IPC code still uses raw
`UnixStream` outside the transport abstraction, consider the same migration.
The `Transport` enum (`Unix(PathBuf)` + `Tcp(SocketAddr)`) with `cfg(unix)`
gating provides ecoBin compliance with zero runtime cost on Unix.

**Recommendation for toadStool**: The `NEURAL_API_ADDRESS` env var (TCP
fallback) enables cross-platform provenance IPC. If toadStool's orchestrator
advertises TCP endpoints, springs can discover them via `{PRIMAL}_ADDRESS`.

---

## 2. ProvenanceConfig DI Pattern

airSpring uses dependency injection for all IPC configuration to enable testing
without environment mutation:

```rust
pub struct ProvenanceConfig {
    pub transport_override: Option<Transport>,
    pub neural_api_socket: Option<PathBuf>,
    pub neural_api_address: Option<SocketAddr>,
    pub biomeos_socket_dir: Option<PathBuf>,
}
```

Production: `ProvenanceConfig::from_env()`. Tests: construct directly. Zero
`set_var`/`remove_var`, zero `unsafe`, zero `#[serial]`.

**Recommendation**: If barraCuda's `ValidationHarness` or other upstream test
infrastructure uses env vars directly, consider evolving to this DI pattern.

---

## 3. #[expect()] vs #[allow()] (Rust 2024 Idiom)

airSpring completed full migration from `#[allow()]` to `#[expect()]` with
`reason` strings. `#[expect]` is strictly superior — it fails if the suppressed
lint stops firing, preventing stale suppressions.

**Finding**: airSpring has **zero `#[allow()]`** in any code (production or test).
All lint suppression uses `#[expect(lint, reason = "...")]`.

**Recommendation for barraCuda**: If upstream still uses `#[allow()]`, consider
migration. With Edition 2024 and MSRV 1.92, `#[expect]` is stable.

---

## 4. Doctest Evolution (ignore → no_run)

`ignore`d doctests are invisible to CI — they can rot without detection.
airSpring evolved all 4 to `no_run` with hidden setup boilerplate:

```rust
//! ```rust,no_run
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! # let device = todo!("requires GPU device");
//! use airspring_barracuda::gpu::stream::StreamSmoother;
//! let smoother = StreamSmoother::new(device);
//! # Ok(())
//! # }
//! ```
```

**Recommendation**: Audit barraCuda doctests for `ignore` directives. The
`no_run` + hidden setup pattern preserves compile checking without requiring
runtime resources (GPU, sockets).

---

## 5. Dependency Health Report

airSpring's dependency tree analysis:

| Dep | Source | Notes |
|---|---|---|
| `libc` | errno, getrandom, tempfile (via tokio, rand) | Infrastructure — acceptable |
| `cc` | blake3 (build-dep, SIMD acceleration) | Infrastructure — acceptable |
| `wgpu 28` | Direct dep (GPU dispatch) | Pure Rust surface, Vulkan backend |
| `bytemuck 1` | Direct dep (GPU buffer layout) | Pure Rust, zero-dep |
| `akida-driver` | Optional (NPU, feature-gated) | Path dep to toadStool |

No `openssl`, `ring`, `cmake`, `bindgen`, or `pkg-config` in the tree.
ecoBin compliant.

**Note for barraCuda**: `blake3` pulls `cc` as a build-dependency for optional
SIMD acceleration. If ecoBin purity is a concern, `blake3` supports
`features = ["pure"]` to disable the C codegen.

---

## 6. Tolerance Architecture

airSpring uses 58 named `Tolerance` structs with `abs_tol`, `rel_tol`, and
`justification` fields. All are centralized in `tolerances/mod.rs` with a
`Baseline Provenance` table mapping each to its Python control script, commit,
date, and command.

**Status**: All 452 inline test tolerances audited — none are in production
code. Validation binaries (Python↔Rust) use named tolerances exclusively.

---

## 7. Evolution Gaps — Upstream Requests

### Tier C (needs new barraCuda primitive):

1. **`ValidationSink` trait**: airSpring's 91 validation binaries would benefit
   from testable harness output. Pattern from ludoSpring V23: `StderrSink` for
   production, `BufferSink` for testing. Propose upstream absorption into
   `barracuda::validation::ValidationHarness`.

2. **HTTP/JSON data client**: Open-Meteo, NOAA CDO APIs. Not GPU — but needed
   for automated data ingestion. Could live in a `barracuda::io::http` module
   or a separate `primalTools` utility.

### Tier B (available but unwired):

- `BatchedOdeRK45F64` — Adaptive Dormand-Prince for dynamic soil models
- `TensorContext` — Pooled buffers for `SeasonalReducer` optimization
- `UnidirectionalPipeline` — Fire-and-forget GPU streaming for atlas pipeline

---

## 8. Cross-Spring Shader Status

All 20 `BatchedElementwiseF64` ops consumed by airSpring are upstream.
`local_dispatch` retired in v0.7.2. Zero local WGSL shaders remaining.

The Write → Absorb → Lean cycle is **complete** for airSpring.

---

## 9. Verified State

| Metric | Value |
|---|---|
| Lib tests | 946 |
| Integration tests | 20 |
| Forge tests | 61 |
| Doctests | 9 (0 ignored) |
| Validation binaries | 91 |
| Experiments | 87 |
| Line coverage | ~95% (llvm-cov) |
| Clippy | Zero warnings (pedantic + nursery) |
| Unsafe blocks | Zero (`#![forbid(unsafe_code)]`) |
| `#[allow()]` | Zero |
| C dependencies | Zero in application code |
| Hardcoded primals | Zero in production code |
| Platform | Unix + TCP (Transport enum) |
