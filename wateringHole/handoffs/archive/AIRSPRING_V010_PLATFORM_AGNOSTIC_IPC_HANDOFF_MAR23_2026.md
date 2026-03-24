# airSpring V0.10.0 — Platform-Agnostic IPC & Deep Debt Resolution Handoff

**Date:** 2026-03-23
**From:** airSpring V0.10.0
**License:** AGPL-3.0-or-later
**Supersedes:** AIRSPRING_V010_DEEP_EVOLUTION_EXECUTION_HANDOFF_MAR22_2026.md

---

## Executive Summary

Completed platform-agnostic IPC migration for the provenance trio, eliminated
all remaining `#[allow()]` in production code, evolved 4 ignored doctests to
compile-checked `no_run`, documented 26 missing experiments (Exp 062-087) in
PAPER_REVIEW_QUEUE.md, and performed full deep audit with zero findings.

---

## What Changed

### 1. Platform-Agnostic Provenance IPC (Unix → Transport)

`ipc/provenance.rs` used raw `std::os::unix::net::UnixStream` — Unix-only,
violating the ecoBin cross-platform standard. Evolved to use the existing
`rpc::Transport` + `rpc::send_to` abstraction:

- `ProvenanceConfig` evolved: `socket_override: Option<PathBuf>` → `transport_override: Option<Transport>`
- Added `neural_api_address: Option<SocketAddr>` for TCP fallback
- `neural_api_socket_path()` → `resolve_neural_api_transport()` (returns `Transport`)
- `capability_call()` reduced from 54 lines of raw I/O to 25 lines delegating to `rpc::send_to`
- `data/nestgate.rs` updated: `rpc::send()` → `rpc::send_to(&transport, ...)`
- All 10 provenance tests updated and passing
- Resolution order: transport_override → NEURAL_API_SOCKET (Unix) → NEURAL_API_ADDRESS (TCP) → biomeOS discovery

### 2. #[allow()] → #[expect()] Final Sweep

- `eco/drought_index.rs` test module: `#[allow(unwrap_used, float_cmp)]` → `#[expect(...)]`
- `tests/common/mod.rs`: 3 attributes (`dead_code`, `unused_macros`, `unused_imports`) → `#[expect(...)]`
- **Zero `#[allow()]` remaining in any production or test code**

### 3. Doctest Evolution (ignore → no_run)

4 doctests were `ignore`d — invisible to CI, never compile-checked:
- `biomeos/mod.rs`: `ignore` → `no_run` with hidden helper stubs
- `gpu/stream.rs`: `ignore` → `no_run` with hidden setup
- `gpu/seasonal_pipeline/mod.rs`: `ignore` → `no_run` with `CropConfig::standard()`
- `rpc/mod.rs`: `ignore` → `no_run` (socket path example)

**Before:** 5 pass + 4 ignored. **After:** 9 pass + 0 ignored.

### 4. Experiment Documentation (PAPER_REVIEW_QUEUE.md)

Added Exp 062-087 rows to the experiment table (26 experiments were in the
header count but missing from the table body). All 87 experiments now fully
documented with binary name, GPU path, and check counts.

---

## Deep Audit Results (All Clean)

| Category | Finding |
|---|---|
| Unsafe code | Zero. `#![forbid(unsafe_code)]` enforced |
| `#[allow()]` in production | Zero (all → `#[expect()]`) |
| Mocks in production | Zero |
| `.unwrap()` in production | Zero. `#![deny(clippy::unwrap_used)]` enforced |
| TODO/FIXME/HACK | Zero in production code |
| Hardcoded primal names | Only test fixtures + log targets |
| External C deps | libc/cc in infrastructure only (tokio, blake3, rand) |
| Primal discovery | Runtime-only, zero compile-time coupling |
| File sizes | All under 1000 LOC |
| Ignored doctests | Zero (was 4) |

---

## Verification

- `cargo fmt --check`: PASS
- `cargo clippy --all-features -- -D warnings`: PASS (zero warnings)
- `cargo check --all-features`: PASS
- `cargo test --all-features`: 947 passed, 0 failed, 0 ignored
- `cargo test --doc`: 9 passed, 0 failed, 0 ignored

---

## For barraCuda Team

See companion handoff: `HANDOFF_AIRSPRING_TO_BARRACUDA_TRANSPORT_ABSORPTION_MAR23_2026.md`

---

## Files Modified

- `barracuda/src/eco/drought_index.rs` — `#[allow]` → `#[expect]`
- `barracuda/tests/common/mod.rs` — `#[allow]` → `#[expect]` (3 sites)
- `barracuda/src/ipc/provenance.rs` — Unix-only → Transport abstraction
- `barracuda/src/data/nestgate.rs` — socket path → transport
- `barracuda/src/gpu/stream.rs` — doctest `ignore` → `no_run`
- `barracuda/src/gpu/seasonal_pipeline/mod.rs` — doctest `ignore` → `no_run`
- `barracuda/src/rpc/mod.rs` — doctest `ignore` → `no_run`
- `barracuda/src/biomeos/mod.rs` — doctest `ignore` → `no_run`
- `specs/PAPER_REVIEW_QUEUE.md` — Exp 062-087 rows added
