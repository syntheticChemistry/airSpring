# airSpring V0.8.3 — Deep Debt Resolution & barraCuda Evolution Handoff

**Date:** March 16, 2026
**From:** airSpring V0.8.3 (ecoPrimals)
**To:** barraCuda, ToadStool, biomeOS, all Springs
**Authority:** wateringHole (ecoPrimals Core Standards)
**Supersedes:** V0.8.2 Niche Architecture handoff (retained in `handoffs/archive/`)
**Pins:** barraCuda v0.3.5 (`path = ../../barraCuda/crates/barracuda`), toadStool S94b+, wgpu 28
**License:** AGPL-3.0-or-later

---

## Executive Summary

- **Protocol compliance**: JSON-RPC 2.0 method-not-found now returns proper `{"error": {"code": -32601, ...}}` — was silently wrapped in success (F-10)
- **Safety hardened**: `#![forbid(unsafe_code)]` in both crates (upgraded from deny — cannot be overridden by `#[allow]`)
- **Capability-based discovery**: Orchestrator socket resolved via runtime discovery, not hardcoded `"biomeOS.sock"` name
- **58 centralized tolerances**: 3 new GPU parity tiers (`gpu_simplified_et0`, `gpu_empirical_pet`, `bootstrap_jackknife_known`); 3 missing biodiversity tolerances added to test
- **IPC integration tests**: 5 self-contained round-trip tests exercising JSON-RPC protocol layer
- **Quality gates**: `warn(missing_docs)`, `cargo-deny` aligned (unknown-git deny), `rust-toolchain.toml` at 1.92

---

## §1 What Changed (19 findings resolved)

### Protocol & Safety

| ID | Finding | Resolution |
|----|---------|------------|
| F-10 | `dispatch()` returned `serde_json::Value`, `handle_connection` always wrapped in `rpc::success()` — method-not-found was a silent protocol violation | Introduced `DispatchOutcome` enum; method-not-found returns `rpc::error()` with code `-32601`; added `INVALID_REQUEST` for missing `method` field |
| F-5 | `#![deny(unsafe_code)]` can be overridden with `#[allow(unsafe_code)]` at item level | Upgraded to `#![forbid(unsafe_code)]` in both `airspring-barracuda` and `airspring-forge` |
| F-6 | `forge/src/lib.rs` lacked `deny(clippy::unwrap_used, clippy::expect_used)` | Added — now matches barracuda's lint profile |

### Hardcoding → Capability-Based

| ID | Finding | Resolution |
|----|---------|------------|
| F-16 | `orchestrator_socket_name()` returned hardcoded `"biomeOS.sock"` | New `discover_orchestrator_socket()`: checks `BIOMEOS_ORCHESTRATOR_SOCKET` env → scans for `biomeOS`/`biomeos` sockets via `discover_primal_socket_in()` → falls back to `fallback_registration_primal()`. Primal only has self-knowledge. |
| F-1 | `py_kr` hardcoded in function body without structured provenance | Moved to `const PY_KR_BARE_SOIL_DRYDOWN` with doc comment (script, commit, date) |

### Tolerance Centralization

| ID | Finding | Resolution |
|----|---------|------------|
| F-2 | Ad-hoc inline tolerances (1e-3, 5e-3, 1e-2) in `validate_cross_spring_evolution.rs` | Created 3 new centralized `Tolerance` constants; rewired binary to `tolerances::GPU_SIMPLIFIED_ET0`, `tolerances::GPU_EMPIRICAL_PET`, `tolerances::CROSS_SPRING_EVOLUTION`, `tolerances::CROSS_SPRING_GPU_CPU` |
| F-3 | Tolerance test asserted 52 but missed 3 biodiversity tolerances; TOLERANCE_REGISTRY.md said 57 | Fixed: 58 `Tolerance` structs across 4 modules (15+19+11+13), test includes all, registry aligned |

### Build & Quality

| ID | Finding | Resolution |
|----|---------|------------|
| F-15 | No `rust-toolchain.toml` — builds depend on ambient toolchain | Created `rust-toolchain.toml` pinning channel 1.92 (wgpu 28 MSRV) |
| F-17 | No `missing_docs` lint — public API docs not enforced | Added `#![warn(missing_docs)]` to both crate roots |
| F-19 | `barracuda/deny.toml` used `unknown-git = "allow"` while forge used `"deny"` | Aligned both to `"deny"` |
| F-12 | Stale barraCuda version comment in `forge/Cargo.toml` | Updated `v0.3.3+` → `v0.3.5` |

### Documentation & Specs

| ID | Resolution |
|----|------------|
| F-13 | Updated `BARRACUDA_REQUIREMENTS.md` ValidationHarness count (22 → 84+ binaries) |
| F-14 | Documented early-exit(0) contract in `validation.rs` — CI should distinguish "no GPU" from "validation failure" |
| F-9 | Documented Unix-only IPC as known limitation in `rpc.rs` with evolution path |
| F-7 | Added effort estimates and blocker descriptions to `GPU_PROMOTION_MAP.md` for Tier B/C |
| F-8 | Created `tests/ipc_roundtrip.rs` — 5 self-contained JSON-RPC round-trip tests |

---

## §2 What airSpring Contributes Upstream (complete)

All local WGSL ops absorbed. Write→Absorb→Lean complete for all operations:

| Contribution | Type | Upstream Location |
|-------------|------|-------------------|
| SCS-CN runoff | Absorbed → op=17 | `BatchedElementwiseF64` |
| Stewart yield | Absorbed → op=18 | `BatchedElementwiseF64` |
| Makkink ET₀ | Absorbed → op=14 | `BatchedElementwiseF64` |
| Turc ET₀ | Absorbed → op=15 | `BatchedElementwiseF64` |
| Hamon PET | Absorbed → op=16 | `BatchedElementwiseF64` |
| Blaney-Criddle | Absorbed → op=19 | `BatchedElementwiseF64` |
| Stats metrics | Absorbed (S64) | `barracuda::stats::metrics` |
| Richards PDE | Absorbed (S40) | `barracuda::pde::richards` |
| ValidationHarness | Absorbed (S59) | `barracuda::validation` |
| TS-001 pow_f64 fix | Fix (S54) | `math_f64.wgsl` |
| TS-003 acos/sin fix | Fix (S54) | `math_f64.wgsl` |
| TS-004 reduce buffer fix | Fix (S54) | `fused_map_reduce_f64.wgsl` |

---

## §3 What airSpring Still Needs from barraCuda

| Gap | Tier | Current State | Effort |
|-----|------|---------------|--------|
| Fused GPU seasonal pipeline | B | GPU stages 1-3 wired; stage 4 (yield) CPU | Medium — needs `PipelineSession` or chained persistent buffers |
| `UnidirectionalPipeline` streaming | B | GPU-capable with callback pattern | Medium — needs barraCuda async dispatch |
| Anderson coupling shader | C | CPU-only iterative fixed-point | High — new WGSL pattern for convergence loops |
| VG θ/K batch (new op) | B | `eco::van_genuchten` validated | Low — elementwise, straightforward |
| Batch Nelder-Mead GPU | B | CPU NM wired via `gpu::isotherm` | Medium |
| Tridiagonal solve | B | Available upstream | Low |
| Adaptive ODE (RK45) | B | Available upstream | Low |

---

## §4 barraCuda Evolution Recommendations

### 4.1 PipelineSession API

airSpring's `gpu::seasonal_pipeline` chains ops 0→7→1→yield across 4 GPU stages. Currently each stage does a fresh buffer upload/download. A `PipelineSession` that keeps intermediate buffers on-device would eliminate 3× the PCIe round-trips per field-day.

**toadStool action:** Design `PipelineSession` API that lets springs chain `BatchedElementwiseF64` ops with persistent GPU buffers.

### 4.2 Platform-Agnostic IPC Transport

airSpring's `rpc.rs` is Unix-only (`UnixStream`). ecoBin standard requires platform-agnostic IPC. A transport trait in barraCuda's device module would let all springs share one implementation:

```rust
pub trait RpcTransport: Send + Sync {
    fn send(&self, method: &str, params: &Value) -> Option<Value>;
}
```

**barraCuda action:** Consider a `transport` module with Unix socket, TCP, and Windows named pipe backends.

### 4.3 Tolerance Infrastructure

airSpring now has 58 centralized `Tolerance` structs across 4 domain submodules. The `barracuda::tolerances::{Tolerance, check}` type is the shared foundation. Other springs may benefit from:

- A `tolerance_registry!` macro that auto-generates the count assertion test
- A `tolerance.check_rel()` method (currently only `check()` exists for absolute)
- Cross-spring tolerance namespacing (e.g., `barracuda::tolerances::gpu::GPU_CPU_CROSS`)

### 4.4 Async Dispatch for Streaming

`gpu::atlas_stream` implements fire-and-forget GPU dispatch with completion callbacks. barraCuda could formalize this as:

```rust
pub trait AsyncGpuDispatch {
    fn dispatch_async(&self, input: &[f64], callback: impl FnOnce(Vec<f64>)) -> DispatchHandle;
}
```

This would enable 10,000-station atlas-scale streaming without blocking the main thread.

---

## §5 ToadStool Evolution Recommendations

### 5.1 Session-Aware Precision Routing

`PrecisionRoutingAdvice` (S58) works well for single-op dispatch. For multi-op pipelines, the advice should be session-scoped: if any stage requires DF64, the entire pipeline should use DF64 to avoid mixed-precision intermediate buffers.

### 5.2 Shader Compilation Caching

airSpring's 25 Tier A GPU modules compile shaders on first use. A persistent shader cache keyed on `(shader_source_hash, device_id, precision_tier)` would eliminate cold-start latency.

### 5.3 coralReef Integration Timeline

Once coralReef reaches native f64 lowering (Phase 4), airSpring's DF64 workaround code paths become dead code. Document the migration path: which airSpring GPU modules have DF64 fallback code that should be removed when coralReef provides native f64.

---

## §6 Cross-Spring Learnings

### 6.1 JSON-RPC 2.0 Protocol Compliance

We discovered that wrapping dispatch results in `rpc::success()` without distinguishing protocol errors from application errors violates JSON-RPC 2.0. The `DispatchOutcome` enum pattern should be adopted by all springs running JSON-RPC servers:

```rust
enum DispatchOutcome {
    Ok(serde_json::Value),
    RpcError { code: i32, message: String },
}
```

**All springs action:** Audit `dispatch()` functions for method-not-found handling.

### 6.2 `#![forbid(unsafe_code)]` vs `#![deny(unsafe_code)]`

`deny` can be overridden at item level with `#[allow(unsafe_code)]`. `forbid` cannot. For application crates that should never have unsafe, `forbid` is the correct choice. Library crates that need targeted unsafe (e.g., barraCuda's GPU buffer management) should use `deny` with documented `#[allow]` exceptions.

### 6.3 Capability-Based Discovery Pattern

Hardcoded socket names (e.g., `"biomeOS.sock"`) couple primals to deployment topology. The pattern should be: env override → runtime scan → fallback. This was implemented in `discover_orchestrator_socket()` and should be the standard for all primals.

---

## §7 Quality Gate (v0.8.3)

| Gate | Status |
|------|--------|
| `cargo test --lib` (barracuda) | **863/863 PASS** |
| `cargo test --test ipc_roundtrip` | **5/5 PASS** |
| `cargo test --lib` (metalForge) | **61/61 PASS** |
| `#![forbid(unsafe_code)]` | Both crates |
| `#![deny(clippy::unwrap_used, expect_used)]` | Both crates |
| `#![warn(missing_docs)]` | Both crates (71 + 5 warnings — gradual enforcement) |
| `cargo-deny check` | Both crates, `unknown-git = "deny"` |
| Centralized tolerances | 58 `Tolerance` structs, 4 submodules, test asserts count |
| JSON-RPC 2.0 protocol | Method-not-found returns error object |
| Capability-based discovery | No hardcoded socket names |
| `rust-toolchain.toml` | Pinned 1.92, rustfmt + clippy + llvm-tools |

---

## Action Items

| # | Owner | Action | Priority |
|---|-------|--------|----------|
| 1 | barraCuda | Design `PipelineSession` API for chained GPU ops with persistent buffers | High |
| 2 | barraCuda | Consider platform-agnostic `RpcTransport` trait (Unix, TCP, named pipes) | Medium |
| 3 | barraCuda | Add `tolerance.check_rel()` method and `tolerance_registry!` macro | Low |
| 4 | ToadStool | Session-scoped `PrecisionRoutingAdvice` for multi-op pipelines | Medium |
| 5 | ToadStool | Persistent shader compilation cache | Medium |
| 6 | All Springs | Audit JSON-RPC dispatch for method-not-found protocol compliance | High |
| 7 | All Springs | Upgrade `deny(unsafe_code)` → `forbid(unsafe_code)` in application crates | Medium |
| 8 | All Springs | Replace hardcoded socket names with capability-based discovery | Medium |
| 9 | coralReef | Document DF64→native f64 migration path for spring GPU modules | Low |
