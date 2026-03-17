# airSpring V0.8.8 — barraCuda / toadStool Evolution Handoff

**Date**: March 16, 2026
**From**: airSpring V0.8.8
**To**: barraCuda, toadStool, coralReef, sibling springs
**License**: AGPL-3.0-or-later

---

## Executive Summary

- airSpring consumes **75+ distinct barraCuda functions/types** across 11 modules
- **20 GPU ops** via `BatchedElementwiseF64` (ops 0–19), all upstream
- **11 specialized GPU primitives** (BrentGpu, RichardsGpu, KrigingF64, etc.)
- **Zero local WGSL shaders** — Write→Absorb→Lean complete
- **3 toadStool IPC methods** via typed `compute_dispatch` client
- v0.8.8 adds `IpcError::is_recoverable()`, `CircuitBreaker`, `resilient_send()`, `thiserror`, health probes

## What Changed in V0.8.8

### 1. `thiserror` for All Error Types

- `AirSpringError`: 8 variants with `#[from]` for `io::Error`, `serde_json::Error`, `BarracudaError`, `IpcError`
- `IpcError`: expanded from 4 → **8 variants** (`WriteFailed`, `ReadFailed`, `SocketNotFound`, `EmptyResponse` added)

### 2. `IpcError::is_recoverable()` for Retry Logic

Returns `true` for `ConnectionFailed`, `Timeout`, `WriteFailed`, `ReadFailed` — transient failures that toadStool dispatch should retry on. **Recommendation for barraCuda**: absorb `is_recoverable()` into a shared IPC error type.

### 3. Circuit Breaker + Exponential Backoff

`ipc::resilience::CircuitBreaker` + `resilient_send()`:
- Opens after 3 consecutive failures, auto-resets after 5s cooldown
- Backoff: 50ms → 100ms → fail
- **Recommendation for toadStool**: consider exposing circuit breaker health in `compute.dispatch.status`

### 4. Health Probes

- `health.liveness`: `{"alive": true, "niche": "airspring"}`
- `health.readiness`: subsystem status (provenance trio, nestgate, toadstool)
- **Recommendation for toadStool/biomeOS**: query `health.readiness` before routing work to airSpring

### 5. `socket_env_var()` / `address_env_var()`

Generic discovery: `socket_env_var("toadstool")` → `"TOADSTOOL_SOCKET"`. **Recommendation for barraCuda**: absorb into shared `primal_names` crate.

## barraCuda Consumption Inventory

### Modules Used (by import count)

| Module | ~Imports | Key Types/Functions |
|--------|----------|---------------------|
| `stats` | 45+ | mean, pearson, spearman, rmse, mbe, bootstrap_ci, norm_ppf, hargreaves_et0_batch, diversity metrics |
| `ops` | 35+ | BatchedElementwiseF64, FusedMapReduceF64, VarianceF64, CorrelationF64, KrigingF64, DiversityFusionGpu |
| `device` | 25+ | WgpuDevice, PrecisionRoutingAdvice, F64BuiltinCapabilities, Fp64Strategy |
| `optimize` | 15+ | brent, BrentGpu, nelder_mead, multi_start_nelder_mead |
| `validation` | 10+ | ValidationHarness, exit_no_gpu, gpu_required |
| `pde` | 8+ | solve_richards, RichardsGpu, CrankNicolsonConfig |
| `tolerances` | 5+ | Tolerance, check |
| `shaders::provenance` | 5+ | shaders_consumed_by, evolution_report, cross_spring_matrix |
| `linalg` | 5+ | tridiagonal_solve, ridge_regression |
| `pipeline` | 2 | StatefulPipeline, BatchedStatefulF64 |
| `special` | 1 | regularized_gamma_p |

### GPU Ops (all via `BatchedElementwiseF64`)

Ops 0–19 (FAO-56 ET₀, water balance, sensor calibration, Hargreaves, Kc climate, dual Kc, van Genuchten, Thornthwaite, GDD, pedotransfer, Makkink, Turc, Hamon, SCS-CN runoff, yield response, Blaney-Criddle).

### toadStool Integration

| Method | Purpose |
|--------|---------|
| `compute.dispatch.submit` | Submit GPU workload → `job_id` |
| `compute.dispatch.result` | Poll result by `job_id` |
| `compute.dispatch.capabilities` | Query available compute |

Discovery: `biomeos::discover_primal_socket(primal_names::TOADSTOOL)` or `TOADSTOOL_SOCKET` env override.

## Upstream Absorption Candidates

Patterns airSpring pioneered or refined that barraCuda could absorb:

| Candidate | What | Benefit |
|-----------|------|---------|
| `parse_capabilities` | 4-format capability discovery (flat, object, nested, double-nested) | Used by 4+ springs |
| `IpcError` + `is_recoverable()` | Structured IPC error type with recovery classification | Replace ad-hoc error handling |
| `extract_rpc_error` | `fn(Value) -> Option<(i64, String)>` | Centralize JSON-RPC error extraction |
| `OrExit<T>` | Zero-panic trait for validation binaries | Ecosystem-wide standard |
| `CircuitBreaker` | IPC resilience primitive | Shared circuit breaker for all springs |
| `socket_env_var()` / `address_env_var()` | Generic primal env var convention | Replace per-primal constants |
| Named physical constants | 55+ FAO-56, SCS-CN, Saxton-Rawls constants | Domain-specific precision validation |

## Upstream Action Items

### For barraCuda

| Request | Priority | Context |
|---------|----------|---------|
| `TensorSession` f64 support | High | airSpring seasonal pipeline: fused multi-op f64 sessions |
| Absorb `IpcError` + `is_recoverable()` | High | 4+ springs use structured IPC errors |
| Absorb `parse_capabilities` 4-format | Medium | Standardize in `barracuda::biomeos` |
| `NpuDispatch` trait | Low | Replace local `npu.rs` |
| `batched_multinomial.wgsl` | Low | groundSpring V10 rarefaction use case |

### For toadStool

| Request | Priority | Context |
|---------|----------|---------|
| Confirm `compute.dispatch.submit` response format | High | airSpring expects `{"job_id": "..."}` |
| Expose circuit breaker health in `compute.dispatch.status` | Medium | airSpring can query before submitting |
| Songbird `http.request` latency headers | Medium | SongbirdHttpProvider needs timeout control |

## What Worked Well

- **Write→Absorb→Lean** cycle — all 6 local ops retired cleanly, zero local WGSL
- **`thiserror`** eliminated 60+ lines of manual error boilerplate
- **Proptest fuzz** (22 properties) provides regression safety for IPC infrastructure
- **Zero-panic** (47 binaries) + **`OrExit<T>`** — clean CI output

## Next Steps (V0.8.9 Candidates)

- Migrate validation binaries to use `OrExit<T>` and `parse_benchmark()`
- Wire `resilient_send()` into provenance trio IPC calls
- Profile `compute_gpu()` at N=100K+ for atlas-scale crossover analysis
- ISSUE-008: ET₀ GPU promotion (Thornthwaite, Makkink, Turc, Hamon → Tier A)

## Superseded Handoffs (archived)

- `AIRSPRING_V087_BARRACUDA_TOADSTOOL_EVOLUTION_HANDOFF_MAR16_2026.md` → `archive/`
- `AIRSPRING_V087_ECOSYSTEM_ABSORPTION_HANDOFF_MAR16_2026.md` → `archive/`

---
*ScyBorg Provenance Trio: AGPL-3.0-or-later (code) + ORC (game mechanics) + CC-BY-SA 4.0 (creative content)*
