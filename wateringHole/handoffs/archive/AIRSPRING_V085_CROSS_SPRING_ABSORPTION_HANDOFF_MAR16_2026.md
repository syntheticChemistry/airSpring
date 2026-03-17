# airSpring V0.8.5 — Cross-Spring Absorption Handoff

**Date**: 2026-03-16
**From**: airSpring
**To**: barraCuda / toadStool / All Springs
**License**: AGPL-3.0-or-later
**Previous**: V084 Deep Debt Execution (archived)

---

## Executive Summary

airSpring v0.8.5 completes the cross-spring absorption sprint, absorbing 6 patterns from sibling springs and primals. The headline achievement: **zero C dependencies** — ureq (ring) removed, HTTP now routes through Songbird IPC (Tower Atomic pattern from neuralSpring). All 865 library tests pass, 0 failures, 58 warnings (pre-existing `missing_docs` only).

---

## Changes Executed

### 1. P0: ureq → Songbird IPC (Zero C Dependencies)

**What**: Removed `ureq` (which pulled in `ring`, a C/asm crypto library) and replaced `HttpProvider` with `SongbirdHttpProvider` routing through Songbird's `http.request` capability via JSON-RPC IPC.

**Files Changed**:
- `Cargo.toml` — removed `ureq` dep and `standalone-http` feature
- `src/data/provider.rs` — `HttpProvider` → `SongbirdHttpProvider`
- `src/data/mod.rs` — updated exports, docs
- `tests/nucleus_integration.rs` — updated test names

**Pattern Source**: neuralSpring V108 (`playGround/src/songbird_http.rs`)

**Impact**: airSpring default build is now **fully ecoBin compliant** — zero C dependencies, pure Rust, cross-compilable to any target without C toolchain.

### 2. P1: `#[expect(reason)]` Migration

**What**: Evolved all remaining `#[allow()]` in production code to `#[expect(reason)]` with documented justifications, following wetSpring V122 pattern.

**Files Changed**: `validate_mc_et0.rs`, `validate_drought_index.rs`, `validate_cross_spring_modern.rs`, `bench_cross_spring_evolution/modern.rs`, `data/provider.rs`

**Result**: Zero `#[allow()]` in production code. All `#[allow(unwrap_used, expect_used)]` correctly gated behind `#[cfg(test)]`.

### 3. P2: Zero-Panic Validation Binaries

**What**: Top 9 validation binaries (by expect/unwrap count) evolved from `parse_benchmark_json(...).expect(...)` to `let Ok(...) = ... else { eprintln!("[FAIL]"); std::process::exit(1); }`.

**Pattern Source**: groundSpring V109

**Files Changed**: `validate_priestley_taylor.rs`, `validate_gpu_live.rs`, `validate_cw2d.rs`, `validate_et0.rs`, `validate_thornthwaite.rs`, `validate_yield.rs`, `validate_nass_yield.rs`, `validate_tissue.rs`, `validate_et0_intercomparison.rs`

### 4. P2: Named Physical Constants

**What**: Extracted inline magic numbers to named `const` with source citations.

**New Constants**:
- `evapotranspiration.rs`: `CELSIUS_TO_KELVIN`, `FAO56_PM_WIND_NUMERATOR`, `FAO56_PM_WIND_U2_COEFF`, `PRIESTLEY_TAYLOR_ALPHA`
- `runoff.rs`: `SCS_RETENTION_NUMERATOR`, `SCS_RETENTION_OFFSET`, `AMC_DRY_COEFF`, `AMC_DRY_OFFSET`, `AMC_WET_COEFF`, `AMC_WET_OFFSET`
- `soil_moisture.rs`: Saxton-Rawls constants, 12 texture hydraulic property sets
- `solar.rs`: All 9 constants promoted to `pub const`

### 5. P2: IpcError + DispatchOutcome (biomeOS Alignment)

**What**: Evolved `rpc::send()` from `Option<Value>` to `Result<Value, IpcError>` with structured error variants. Aligned `DispatchOutcome` with biomeOS standard (`Ok`, `MethodNotFound`, `InvalidParams`, `InternalError`).

**New Type**: `IpcError { ConnectionFailed, Timeout, RpcError, DeserializationFailed }`

**Impact**: All IPC callers now have structured error information instead of opaque `None`.

### 6. P3: Dual-Format Capability Discovery

**What**: Added `biomeos::parse_capabilities()` that handles both string-array and object-array capability list formats from diverse ecosystem sources.

**Pattern Source**: ludoSpring

---

## Metrics

| Metric | V084 | V085 |
|--------|------|------|
| C dependencies | 1 (ring via ureq) | **0** |
| Library tests | 863 | **865** |
| `cargo check` warnings | 71 | **58** |
| `#[allow()]` in production | 15 | **0** |
| Named physical constants | ~30 | **~55** |
| `rpc::send` return type | `Option<Value>` | `Result<Value, IpcError>` |
| `DispatchOutcome` variants | 2 | **4** (biomeOS aligned) |
| Validation binaries zero-panic | 0 | **9** (top by count) |

---

## Upstream Action Items

### barraCuda Team
- **Absorb `parse_capabilities`**: Consider adding dual-format capability parsing to `barracuda::biomeos` (or a shared ecosystem crate) — 4 springs now implement this independently.
- **Absorb `IpcError`**: The `IpcError` enum is now consistent across airSpring, groundSpring, neuralSpring — candidate for a shared `barracuda::ipc::IpcError`.
- **Named constants namespace**: Consider `barracuda::constants::fao56::*` for shared agricultural constants (used by airSpring and groundSpring).

### toadStool Team
- **compute.dispatch** wiring: airSpring GPU orchestrators are ready to dispatch through `toadStool` instead of direct `wgpu` — next evolution step.
- **Songbird `http.request`**: Ensure `save_to` and `content_length` fields are supported in Songbird responses (neuralSpring expects them for file downloads).

### All Springs
- **Tower Atomic adoption**: airSpring, neuralSpring, and groundSpring are now zero C deps. wetSpring, hotSpring, healthSpring, ludoSpring should consider the same migration.
- **`#[expect(reason)]` pattern**: Ecosystem-wide adoption recommended — zero `#[allow()]` in production, with documented justifications.
- **Zero-panic validation**: `let Ok else { eprintln!; exit(1) }` pattern should be ecosystem standard for validation binaries.

---

## What's Next

- Wire GPU orchestrators to `toadStool compute.dispatch` for live hardware routing
- Expand zero-panic pattern to remaining ~50 validation binaries
- `compute.dispatch` integration with `CapabilityClient` typed SDK
- Cross-spring weather data sharing via NestGate content-addressed cache

---

*AGPL-3.0-or-later · ScyBorg Provenance Trio*
