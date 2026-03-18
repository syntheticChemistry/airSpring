# airSpring V0.8.9 — barraCuda / toadStool Evolution Handoff

**Date**: March 17, 2026
**From**: airSpring V0.8.9
**To**: barraCuda (math) / toadStool (dispatch) teams
**License**: AGPL-3.0-or-later

---

## Executive Summary

airSpring V0.8.9 completed a cross-ecosystem absorption sprint that produced
several patterns and types suitable for upstream absorption. This handoff
documents what we built, what we learned, and what barraCuda / toadStool
can absorb to benefit all springs.

- **3 upstream absorption candidates** ready now
- **73 barraCuda touchpoints** across 20 source files (healthy, no gaps)
- **0 local WGSL shaders** (Write→Absorb→Lean complete since v0.7.2)
- **891 lib tests**, zero clippy warnings, `#![forbid(unsafe_code)]`

---

## 1. Upstream Absorption Candidates

### 1.1 `cast` Module — Safe Numeric Casts

**Provenance**: neuralSpring S162, healthSpring V33, airSpring V0.8.9

**What**: A centralized module providing type-safe numeric conversion functions
that satisfy `clippy::cast_precision_loss`, `clippy::cast_possible_truncation`,
and `clippy::cast_sign_loss` lints without `#[allow]` or `#[expect]` suppressions.

**Functions**:

| Function | Signature | Safety |
|----------|-----------|--------|
| `usize_f64(v)` | `usize → f64` | Exact for v < 2^53 |
| `f64_usize(v)` | `f64 → usize` | Debug-panic on negative, NaN, overflow |
| `usize_u32(v)` | `usize → u32` | Debug-panic on overflow |
| `i32_f64(v)` | `i32 → f64` | Exact (all i32 fit in f64) |
| `u32_f64(v)` | `u32 → f64` | Exact (all u32 fit in f64) |
| `f64_u32(v)` | `f64 → u32` | Debug-panic on negative, NaN, overflow |

**Why upstream**: Every spring needs these. airSpring migrated ~30 sites.
healthSpring, neuralSpring, wetSpring all built local versions independently.
A canonical `barracuda::cast` module eliminates this repeated work.

**Size**: ~100 lines including tests.

**toadStool action**: Absorb into `barracuda::cast` or `barracuda::numerics::cast`.

### 1.2 `DispatchOutcome<T>` — Generic JSON-RPC Dispatch Result

**Provenance**: wetSpring V126, groundSpring V112, airSpring V0.8.9

**What**: A generic enum for classifying JSON-RPC method dispatch results:

```rust
pub enum DispatchOutcome<T> {
    Ok(T),
    MethodNotFound(String),
    InvalidParams { method: String, reason: String },
    InternalError { method: String, source: String },
}
```

With helper methods: `is_ok()`, `is_recoverable()`, `ok()`.

**Why upstream**: Every primal binary dispatches JSON-RPC methods. wetSpring,
groundSpring, healthSpring, and airSpring all define local variants. A canonical
library type in `barracuda::ipc` or `barracuda::rpc` standardizes error handling.

**Size**: ~100 lines including tests.

**toadStool action**: Absorb into `barracuda::rpc::DispatchOutcome<T>`.

### 1.3 `ValidationSink` Trait — Testable Harness Output

**Provenance**: ludoSpring V23

**What**: A trait abstracting harness output so validation results can be
directed to stderr (production), a buffer (tests), or any other sink.
ludoSpring implemented `StderrSink`, `BufferSink`, and demonstrated
testable validation harness output.

**Current state**: `barracuda::validation::ValidationHarness::finish()` writes
directly to tracing. No trait-based output abstraction exists.

**Why upstream**: All 7 springs have validation binaries (airSpring has 91).
Testable harness output lets springs assert on validation results in
integration tests without parsing stderr.

**toadStool action**: Absorb `ValidationSink` trait into
`barracuda::validation::ValidationHarness`. Add `StderrSink` (default) and
`BufferSink` (testing). Springs opt in by passing a sink to `finish()`.

---

## 2. airSpring's barraCuda Usage Profile

### 2.1 Module Touchpoints (73 total)

| barraCuda Module | Touchpoints | What We Use |
|------------------|:-----------:|-------------|
| `barracuda::device` | 25 | `WgpuDevice`, `Fp64Strategy`, `Fp64Rate`, `GpuDriverProfile`, `PrecisionRoutingAdvice`, `F64BuiltinCapabilities`, `probe::probe_f64_builtins`, `test_pool::tokio_block_on` |
| `barracuda::ops` | 18 | `BatchedElementwiseF64` (ops 0–1, 5–13, 17–19), `FusedMapReduceF64`, `VarianceF64`, `AutocorrelationF64`, `MovingWindowStats`, `CorrelationF64`, `stats_f64` |
| `barracuda::stats` | 15 | `mean`, `percentile`, `norm_ppf`, `bootstrap_ci`, `BootstrapMeanGpu`, `moving_window_stats_f64`, `regression::fit_linear`, `hydrology::hargreaves_et0_batch`, `correlation::variance`, `correlation::std_dev` |
| `barracuda::pde` | 6 | `richards`, `richards_gpu::RichardsGpu`, `crank_nicolson::CrankNicolsonConfig`, `crank_nicolson::HeatEquation1D` |
| `barracuda::error` | 5 | `BarracudaError` |
| `barracuda::validation` | 2 | `ValidationHarness`, `exit_no_gpu`, `gpu_required` |
| `barracuda::linalg` | 1 | `tridiagonal_solve` |
| `barracuda::special` | 1 | `gamma::regularized_gamma_p` |

### 2.2 GPU Ops Profile

All 20 `BatchedElementwiseF64` ops consumed upstream (ops 0–19).
Zero local WGSL shaders. Zero local dispatch. Write→Absorb→Lean complete.

| Op | Domain | Kernel |
|----|--------|--------|
| 0 | FAO-56 PM ET₀ | `batched_elementwise_f64` |
| 1 | Water balance step | `batched_elementwise_f64` |
| 5 | Sensor calibration | `batched_elementwise_f64` |
| 6 | Hargreaves ET₀ | `HargreavesBatchGpu` |
| 7 | Kc climate adjust | `batched_elementwise_f64` |
| 8 | Dual Kc | `batched_elementwise_f64` |
| 9-10 | Van Genuchten θ/K | `batched_elementwise_f64` |
| 11 | Thornthwaite PET | `batched_elementwise_f64` |
| 12 | GDD | `batched_elementwise_f64` |
| 13 | Pedotransfer | `batched_elementwise_f64` |
| 14-16, 19 | Simple ET₀ methods | `batched_elementwise_f64` |
| 17 | SCS-CN runoff | `batched_elementwise_f64` |
| 18 | Stewart yield | `batched_elementwise_f64` |

---

## 3. Patterns Learned (Relevant to All Springs)

### 3.1 `OnceLock` GPU Probe Caching

**Problem**: Parallel `cargo test` threads calling `WgpuDevice::new_f64_capable()`
create multiple `wgpu::Instance` objects. On some drivers this causes SIGSEGV.

**Solution**: `static GPU_DEVICE: OnceLock<Option<Arc<WgpuDevice>>>` initialized
via `get_or_init()`. Single probe per process lifetime.

**Impact**: airSpring refactored 10+ test-local `try_device()` functions to
use a centralized `try_f64_device()`. Eliminates flaky test failures.

**toadStool action**: Consider upstream `barracuda::device::cached_device()` or
document the `OnceLock` pattern in device guides. toadStool S158 already uses
this pattern; springs should all converge on the same approach.

### 3.2 `mul_add()` FMA Numerical Accuracy

**Problem**: `a * b + c` is two-instruction (round between multiply and add).
`a.mul_add(b, c)` is one-instruction (fused, single rounding).

**Impact**: airSpring evolved 18 `a*b+c` patterns across 7 files (eco/richards,
eco/evapotranspiration, eco/thornthwaite, eco/dual_kc, eco/infiltration,
eco/water_balance, gpu/mc_et0). No test regressions; some results improved
by 1-2 ULP.

**toadStool action**: Audit barraCuda CPU paths for `mul_add()` candidates.
The pattern is: any `a * b + c` or `a * b - c` where `c` is not zero.
Avoid in test assertions where exact bit equality matters.

### 3.3 Canonical `PRIMAL_NAME` / `PRIMAL_DOMAIN`

**Pattern**: Every primal should export `pub const PRIMAL_NAME: &str` and
`pub const PRIMAL_DOMAIN: &str` at the crate root. All identity strings
delegate to these constants. Runtime registration, IPC, socket resolution,
and capability advertising all use the single source of truth.

**toadStool action**: If not already present, add `PRIMAL_NAME` / `PRIMAL_DOMAIN`
to `barracuda::lib.rs` and `toadstool::lib.rs`.

### 3.4 Smart Refactoring Pattern

**Pattern**: When a module exceeds ~500 LOC, extract by domain concept,
not arbitrary line count. Create a module directory with focused sub-modules
and re-export everything from `mod.rs` for backward compatibility.

airSpring refactored 4 monoliths into 19 focused modules:

| Module | Before | After |
|--------|--------|-------|
| `eco/evapotranspiration.rs` | 755 LOC | 5 files: atmosphere, penman_monteith, radiation, hargreaves, priestley_taylor |
| `eco/dual_kc.rs` | 712 LOC | 8 files: types, equations, simulation, cover_crop, mulch, crop_basal, evaporation_params, tests |
| `biomeos.rs` | 713 LOC | 3 files: mod (core), discovery, capabilities |
| `validation.rs` | 679 LOC | 2 files: mod (core), json |

Zero downstream breakage. All public APIs preserved via re-exports.

---

## 4. Evolution Gaps — Unreleased barraCuda Features We're Waiting For

| Feature | Status | What airSpring Gains |
|---------|--------|---------------------|
| `TensorContext` | Unreleased | Fused multi-op pipelines (SeasonalPipeline could be single dispatch) |
| `ComputeDispatch::df64()` | Unreleased | DF64 pathway for consumer GPUs without native f64 |
| `BatchedOdeRK45F64` | Unreleased | GPU-batched ODE for Richards PDE (currently CPU only for implicit Picard) |
| Subgroup detection | Unreleased | Performance optimization for reduce ops |
| `mean_variance_to_buffer()` | Unreleased | Avoid readback for intermediate stats |
| Shader provenance registry v2 | Unreleased | Cross-spring evolution tracking with DAG history |

---

## 5. Dependency Health

| Dependency | Version | Notes |
|------------|---------|-------|
| `barracuda` | 0.3.5 (path) | Healthy — 73 touchpoints |
| `wgpu` | 28 | Only non-pure-Rust dep (required for GPU) |
| `serde`, `serde_json`, `toml` | Latest | Pure Rust |
| `tracing`, `tracing-subscriber` | Latest | Pure Rust |
| `thiserror` | Latest | Pure Rust |
| `bytemuck` | Latest | Pure Rust (safe transmute) |
| `bingocube-nautilus` | 0.1.0 (path) | Evolutionary reservoir |

Zero C dependencies in application code. `#![forbid(unsafe_code)]` in both crates.
`cargo deny check` clean.

---

## 6. What airSpring Does NOT Need

- No new WGSL shaders (all 20 ops absorbed upstream)
- No `GemmF64` direct use (consumed via `stats_f64::linear_regression`)
- No `cyclic_reduction_f64` direct use (consumed via `pde::richards`)
- No local GPU dispatch (retired v0.7.2)

---

## 7. Action Items Summary

| # | Action | Owner | Priority |
|---|--------|-------|----------|
| 1 | Absorb `cast` module into `barracuda::cast` | barraCuda | P1 |
| 2 | Absorb `DispatchOutcome<T>` into `barracuda::rpc` | barraCuda | P1 |
| 3 | Absorb `ValidationSink` trait into `barracuda::validation` | barraCuda | P2 |
| 4 | Audit `mul_add()` candidates in barraCuda CPU paths | barraCuda | P2 |
| 5 | Document `OnceLock` GPU probe pattern for springs | toadStool | P2 |
| 6 | Add `PRIMAL_NAME`/`PRIMAL_DOMAIN` to barraCuda/toadStool | toadStool | P3 |
| 7 | Release `TensorContext` for fused pipeline dispatch | barraCuda | P3 |

---

## Verification

```
cargo clippy --all-targets    # zero warnings (barracuda + metalForge)
cargo check                   # green
cargo test --lib              # 891 tests pass
cargo test --test '*'         # integration tests pass
```
