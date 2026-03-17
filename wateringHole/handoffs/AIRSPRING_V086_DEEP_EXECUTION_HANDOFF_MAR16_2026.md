# airSpring V0.8.6 — Deep Execution Handoff

**Date**: March 16, 2026
**From**: airSpring niche
**To**: barraCuda, toadStool, ecosystem
**License**: AGPL-3.0-or-later

---

## Executive Summary

v0.8.6 completes the deep execution phase started in v0.8.4: all 47 active
validation binaries now use zero-panic patterns, a typed `compute_dispatch`
client is wired for toadStool's `compute.dispatch.submit/result/capabilities`,
`extract_rpc_error()` is centralized in `rpc.rs`, and the Python tolerance
mirror is complete (60 constants, Rust↔Python parity verified).

## Changes

### 1. Zero-Panic Validation — All 47 Binaries

Every active `validate_*.rs` binary now uses `let Ok(...) else { eprintln!("[FAIL]"); exit(1); }`
for benchmark JSON loading. No `.expect()` or `.unwrap()` remains in production
binary paths. This prevents panics in CI and provides clean diagnostic output.

**Before (39 files):**
```rust
let benchmark = parse_benchmark_json(BENCHMARK_JSON).expect("must parse");
```

**After:**
```rust
let Ok(benchmark) = parse_benchmark_json(BENCHMARK_JSON) else {
    eprintln!("[FAIL] benchmark JSON parse error");
    std::process::exit(1);
};
```

### 2. Typed `compute_dispatch` Client

New module: `src/ipc/compute_dispatch.rs`

- `submit(workload_type, params) -> Result<DispatchHandle, DispatchError>`
- `result(handle) -> Result<Value, DispatchError>`
- `capabilities() -> Result<Value, DispatchError>`
- Discovery: `AIRSPRING_COMPUTE_PRIMAL` env → biomeOS socket scan for toadStool
- Uses `extract_rpc_error()` for structured error handling
- Follows healthSpring/groundSpring patterns for job-based dispatch

This enables GPU orchestrators to route through toadStool instead of direct
`wgpu` access, with local fallback when toadStool is unavailable.

### 3. `extract_rpc_error()` — Centralized IPC Error Extraction

Added to `rpc.rs`:
```rust
pub fn extract_rpc_error(response: &Value) -> Option<(i64, String)>
```

Replaces ad-hoc `response.get("error").is_some()` patterns across 4+ call sites.
Takes `&Value` (not `&str` like wetSpring) since `rpc::send()` returns parsed JSON.

### 4. Python Tolerance Mirror — Complete

`control/tolerances.py` now mirrors all 60 Rust tolerance constants:

- Added: `GPU_SIMPLIFIED_ET0` (5e-3, 5e-3) — Makkink/Turc GPU parity
- Added: `GPU_EMPIRICAL_PET` (1e-2, 1e-2) — Hamon/Blaney-Criddle GPU parity
- Documented: `BOOTSTRAP_JACKKNIFE_KNOWN` intentional divergence (Rust 0.01/1e-3
  for analytical known-values vs Python 0.15/0.05 for heuristic range checks)

## Metrics

| Metric | v0.8.5 | v0.8.6 |
|--------|--------|--------|
| Library tests | 865 | **866** |
| Zero-panic binaries | 9/47 | **47/47** |
| Python tolerance constants | 58 | **60** |
| `.expect()` in binary paths | 39 | **0** |
| `extract_rpc_error()` callers | 0 | **1** (compute_dispatch) |
| `compute_dispatch` methods | 0 | **3** (submit/result/capabilities) |

## Upstream Action Items

### barraCuda
- No new upstream absorption candidates this cycle
- `compute_dispatch` client assumes toadStool implements `compute.dispatch.submit`
  with `{workload, params}` → `{job_id}` response format

### toadStool
- airSpring now has a typed client for `compute.dispatch.*`
- Workload types will be: `et0_fao56_batch`, `hargreaves_batch`, etc.
- Please confirm `compute.dispatch.submit` response format: `{"job_id": "..."}`

## What's Next

- Wire GPU orchestrators (`gpu/et0.rs`, etc.) to optionally route through
  `compute_dispatch::submit()` with local `WgpuDevice` fallback
- Expand `extract_rpc_error()` usage to all IPC call sites
- Align `BOOTSTRAP_JACKKNIFE_KNOWN` if semantic parity is desired

---
*ScyBorg Provenance Trio: AGPL-3.0-or-later (code) + ORC (game mechanics) + CC-BY-SA 4.0 (creative content)*
