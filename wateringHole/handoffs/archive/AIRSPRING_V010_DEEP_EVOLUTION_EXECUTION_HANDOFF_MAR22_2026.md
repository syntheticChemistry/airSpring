# airSpring V0.10.0 — Deep Evolution Execution Handoff

**Date:** 2026-03-22
**From:** airSpring V0.10.0
**License:** AGPL-3.0-or-later
**Supersedes:** AIRSPRING_V010_DEEP_EVOLUTION_AUDIT_HANDOFF_MAR22_2026.md, AIRSPRING_V010_DEEP_DEBT_EVOLUTION_HANDOFF_MAR19_2026.md

---

## Executive Summary

Full deep evolution execution. Migrated from deprecated `GpuDriverProfile` to
`DeviceCapabilities` (barraCuda 0.3.7), smart-refactored `rpc.rs` (834 lines)
into `rpc/` module directory by responsibility, completed second `#[allow]`→`#[expect]`
migration round (66 files, 75 unfulfilled removed), added 7 proptest scientific
invariants, fixed NPU quantization edge case, named magic constants. Zero clippy
warnings pedantic+nursery, zero unsafe, zero C deps. PII scrub complete.

---

## What Changed

### 1. barraCuda API Migration: GpuDriverProfile → DeviceCapabilities

- `barracuda::device::GpuDriverProfile` deprecated upstream (Sprint 14)
- `gpu/device_info/mod.rs` migrated to `DeviceCapabilities::from_device()`
- Dropped `fp64_rate: Fp64Rate` from `DevicePrecisionReport` (driver-internal, not consumer-visible)
- `fp64_strategy()` and `precision_routing()` now sourced from `DeviceCapabilities`
- Doc comments updated: `GpuDriverProfile` → `DeviceCapabilities` across 3 files

### 2. Smart Refactor: rpc.rs → rpc/ Module

Split by responsibility, not by line count:
- `rpc/error.rs` — `IpcError` enum + `is_recoverable()` (most-reused type)
- `rpc/transport.rs` — `Transport`, `TransportStream`, Read/Write impls, `connect_transport()`
- `rpc/mod.rs` — protocol builders, send/resolve, `DEFAULT_RPC_TIMEOUT_SECS`, tests
- Error-mapping logic de-duplicated into `io_error_to_ipc` / `io_read_error_to_ipc`

### 3. #[allow] → #[expect] Migration Round 2

- 66 files migrated (all remaining `#[allow(clippy::unwrap_used)]` and `#[allow(clippy::expect_used)]` in test modules)
- 75 unfulfilled `#[expect]` lines removed from 54 files
- Zero `#[allow]` remaining in library source (only `tests/common/mod.rs` retained with reason)

### 4. NPU Evolution

- Added `cast::f64_i8()` helper with debug_assert bounds checking
- Fixed `quantize_i8()`: `lo == hi` edge case produced NaN → now returns 0 explicitly
- `npu/mod.rs` uses `cast::f64_i8` instead of raw `as i8`

### 5. Proptest Invariants (7 new property tests)

- SVP positive (saturation vapour pressure > 0 for -40..60°C)
- SVP monotonic (strictly increases with temperature)
- Δ positive (vapour pressure slope > 0)
- Hargreaves non-negative (ET₀ ≥ 0, uses `f64::midpoint`)
- TAW non-negative (total available water ≥ 0)
- RAW ≤ TAW (readily available never exceeds total)
- Ks ∈ [0,1] (stress coefficient bounded)

### 6. Named Constants

- `DEFAULT_RPC_TIMEOUT_SECS: u64 = 5` replaces bare `Duration::from_secs(5)`

### 7. PII Scrub

- `specs/NUCLEUS_INTEGRATION.md`: remaining `/home/eastgate/` paths → `$ECOPRIMALS_ROOT`

---

## Quality Metrics

| Metric | Value |
|--------|-------|
| Library tests (all features) | **946 passed**, 0 failed |
| Integration tests | **20 passed** (GPU-gated) |
| Forge tests | **61 passed** |
| Doc tests | **5 passed** |
| Proptest invariants | **7** |
| Clippy (pedantic+nursery) | **0 warnings** (both crates, all features) |
| Formatting | **clean** (both crates) |
| Unsafe code | **0** (`#![forbid(unsafe_code)]`) |
| C dependencies | **0** |
| `#[allow]` in src/ | **0** in library code |

---

## Patterns Worth Absorbing Upstream

### Smart Module Refactoring Pattern

Refactored by **responsibility boundaries**, not line count:
- Error types get their own file (they're the most-imported item)
- Platform abstraction gets its own file (Read/Write impls, cfg gates)
- Protocol logic stays with the public API (cohesive send/resolve surface)
- Tests stay in mod.rs (co-located with what they test)

### proptest for Scientific Invariants

Property tests on domain invariants catch edge cases unit tests miss:
- Physical positivity (SVP > 0, ET₀ ≥ 0)
- Monotonicity (SVP increases with temperature)
- Bounded outputs (Ks ∈ [0,1])
- Conservation laws (RAW ≤ TAW)

### Edge-Case Guards in Quantization

`quantize_i8(val, lo, hi)` with `lo == hi` produces 0/0 = NaN. Guard the division explicitly rather than relying on clamp (NaN propagates through clamp on some platforms).

---

## Open Items

| Item | Priority | Notes |
|------|----------|-------|
| `ipc/provenance.rs` duplicates RPC send logic | Low | Cross-crate boundary; cohesive at 613 lines |
| metalForge neural socket resolution parallel to biomeos | Low | Cross-crate boundary by design |
| `PRIMAL_REGISTRY.md` lists airSpring as v0.7.6 | Medium | Parent doc update deferred |
| Coverage re-measurement (llvm-cov) | Medium | Disk space prevented during this session |

---

## License

AGPL-3.0-or-later
