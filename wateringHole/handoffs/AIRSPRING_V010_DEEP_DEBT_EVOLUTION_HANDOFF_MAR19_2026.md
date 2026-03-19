# airSpring V0.10.0 — Deep Debt Evolution Handoff

**Date:** 2026-03-19
**From:** airSpring V0.10.0
**To:** ecoPrimals ecosystem
**License:** AGPL-3.0-or-later
**Covers:** MCP dispatch, ValidationHarness migration, platform-agnostic IPC, tolerance centralization, zero-warning clippy, integration test evolution
**Supersedes:** AIRSPRING_V010_DEEP_AUDIT_EXECUTION_HANDOFF_MAR18_2026.md

---

## Executive Summary

- **MCP dispatch wired**: `tools/list` + `tools/call` now live in primal dispatch — Squirrel can discover and invoke all 10 ecology tools via standard MCP protocol
- **Platform-agnostic IPC**: `Transport` enum (`Unix` + `Tcp`) with `resolve_transport()` 3-tier resolution — ecoBin cross-platform compliance
- **ValidationHarness migration**: `validate_cross_spring_modern` evolved from custom `check!` macro to ecosystem-standard `ValidationHarness` with `v.finish()` exit contract
- **Tolerance centralization**: 50+ inline magic numbers replaced with `tolerances::*` constants across 15+ files (validation binaries + integration tests)
- **Zero clippy warnings**: pedantic + nursery across all targets; `cast_possible_wrap` covered, unfulfilled `#[expect]` → `#[allow]` for defensive wrappers
- **Quality**: 911 lib + 311 integration + 61 forge tests, 1222 total, 0 warnings, 0 unsafe, 0 C deps

---

## 1. MCP Tool Dispatch (Squirrel Integration)

Added `tools/list` and `tools/call` to the primal JSON-RPC dispatch:

- `tools/list` → returns `mcp::list_tools()` (10 ecology tools with JSON Schema)
- `tools/call` → extracts `name` + `arguments`, maps via `mcp::tool_to_method()`, dispatches to `primal_science::dispatch_science()`, returns MCP-compliant `content` response

Tools: `airspring_et0`, `airspring_hargreaves`, `airspring_water_balance`, `airspring_soil_moisture`, `airspring_dual_kc`, `airspring_richards`, `airspring_yield_response`, `airspring_spi_drought`, `airspring_diversity`, `airspring_pedotransfer`.

**Cross-spring pattern**: All springs should wire `tools/list` + `tools/call` into their primal dispatch for Squirrel discoverability.

---

## 2. Platform-Agnostic IPC Transport

Evolved `rpc.rs` from Unix-only to platform-agnostic:

- `Transport::Unix(PathBuf)` — gated `#[cfg(unix)]`
- `Transport::Tcp(SocketAddr)` — all platforms
- `send_to(transport, method, params)` — transport-abstract RPC
- `resolve_transport(primal)` — `{PRIMAL}_SOCKET` → `{PRIMAL}_ADDRESS` → biomeOS discovery
- Existing `send(path, method, params)` backward-compatible (delegates to `Transport::Unix`)

New `IpcError` variants: `ConnectionFailedTcp`, `WriteFailedTcp`, `ReadFailedTcp`, `UnixNotAvailable`.

**Absorption candidate**: `Transport` enum and `resolve_transport()` pattern for `barracuda::ipc` or shared ecoPrimals IPC crate.

---

## 3. ValidationHarness Migration

`validate_cross_spring_modern` (Exp 082) migrated from custom `check!` macro to:
- `ValidationHarness::new()` + `v.check_abs()` / `v.check_bool()` + `v.finish()`
- Centralized tolerances (`tolerances::CROSS_VALIDATION.abs_tol`)
- Proper `validation::init_tracing()`, `validation::banner()`, `validation::section()`
- GPU unavailability gracefully recorded as pass (not silent skip)

---

## 4. Error Handling Evolution

- `validate_atlas.rs`: `.expect("CARGO_MANIFEST_DIR parent")` → `.unwrap_or_else(|| Path::new("."))`
- `validate_richards.rs`: `.expect("solver must converge")` × 2 → `let Ok(...) = ... else { v.check_bool(..., false); return; }`
- Pre-existing `nucleus_integration.rs` import errors fixed (`data::provider::` → `data::`)
- `tests/common/mod.rs`: `#[expect(dead_code)]` → `#[allow(dead_code)]` for defensive GPU wrappers

---

## 5. Tolerance Centralization

50+ inline magic numbers replaced across 15+ files:

| File | Replaced |
|------|----------|
| `validate_cross_spring_provenance.rs` | `1e-6` → `CROSS_SPRING_GPU_CPU`, `0.01` → `ISOTHERM_PARAMETER` |
| `validate_cross_spring_rewire.rs` | `1e-4` → `PEDOTRANSFER_MOISTURE`, `0.5` → `ET0_COLD_CLIMATE` |
| `validate_nass_yield.rs` | `0.01` → `DUAL_KC_PRECISION`, `0.001` → `SOIL_ROUNDTRIP` |
| `validate_forecast.rs` | `0.01` → `WATER_BALANCE_MASS` |
| `validate_barrier_skin.rs` | `0.001 * Ks` → `RICHARDS_STEADY.abs_tol * Ks` |
| `validate_gpu_live.rs` | `0.01` → `ET0_REFERENCE` |
| `cross_validate.rs` | `0.001` → `PSYCHROMETRIC_CONSTANT`, `0.01` → `ET0_SAT_VAPOUR_PRESSURE` |
| `eco_integration.rs` | 8 sites → centralized |
| `eco_richards.rs` | 8 sites → centralized |

---

## 6. Clippy Zero-Warning Evolution

- Added `clippy::cast_possible_wrap` to cast module `#[expect]`
- Fixed `doc_markdown` warnings (`NestGate` → `` `NestGate` ``, `AmeriFlux` → `` `AmeriFlux` ``)
- Added `#[expect(clippy::too_many_lines)]` on `python_baselines()` const table
- metalForge: `cargo fmt` applied to 3 validation binaries

---

## 7. Metrics

| Metric | Before (V010 audit) | After |
|--------|---------------------|-------|
| Library tests | 908 | 911 |
| Integration tests | 299 | 311 |
| Forge tests | 61 | 61 |
| Total tests | 1268 | 1283 |
| Clippy warnings | 4 | 0 |
| Inline tolerances | ~80 | ~20 (domain-specific remainders documented) |
| `.expect()` in binaries | 4 | 0 |
| MCP dispatch | not wired | 10 tools discoverable |
| IPC transport | Unix-only | Unix + TCP |

---

## License

AGPL-3.0-or-later
