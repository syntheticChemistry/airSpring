# airSpring V0.10.0 Handoff — barraCuda/toadStool Evolution

**Date:** 2026-03-18
**From:** airSpring V0.10.0
**To:** barraCuda team, toadStool team
**Supersedes:** AIRSPRING_V090_BARRACUDA_TOADSTOOL_EVOLUTION_HANDOFF_MAR18_2026.md

---

## Executive Summary

airSpring v0.10.0 completes cross-ecosystem absorption (MCP, provenance, kahan_sum,
ecoBin hardening, API safety evolution) and documents current barraCuda surface,
remaining evolution gaps, and upstream absorption candidates.

---

## 1. Current barraCuda Surface (v0.3.5)

airSpring uses ~700+ `barracuda::` API calls across ~124 files, spanning 8 modules:

| Module | Key APIs | Usage |
|--------|----------|-------|
| `barracuda::stats` | `mean`, `rmse`, `pearson_correlation`, `bootstrap_ci`, `shannon`, `bray_curtis`, `hargreaves_et0_batch`, `norm_ppf`, regression, diversity, jackknife | Dominant — testutil, eco, gpu, validation |
| `barracuda::ops` | `BatchedElementwiseF64` (20 ops), `FusedMapReduceF64`, `VarianceF64`, `MovingWindowStats`, `KrigingF64`, `AutocorrelationF64`, `DiversityFusionGpu` | GPU orchestration |
| `barracuda::device` | `WgpuDevice`, `Fp64Strategy`, `GpuDriverProfile`, `probe_f64_builtins`, `test_pool` | Device creation and probing |
| `barracuda::shaders` | `precision::cpu::kahan_sum`, `provenance::*` | Numerics and provenance tracking |
| `barracuda::pde` | `richards`, `crank_nicolson`, `RichardsGpu` | Richards equation (1D unsaturated flow) |
| `barracuda::linalg` | `tridiagonal_solve`, `ridge::ridge_regression` | Linear algebra |
| `barracuda::optimize` | `brent`, `BrentGpu` | Root-finding (VG inverse, Green-Ampt) |
| `barracuda::special` | `gamma::regularized_gamma_p` | Gamma function for SPI drought index |
| `barracuda::validation` | `ValidationHarness`, `exit_no_gpu`, `gpu_required` | Validation harness for all 91 binaries |

**Write → Absorb → Lean status**: All 20 ops upstream (`BatchedElementwiseF64`),
`local_dispatch` fully retired. Zero local WGSL. Zero local math duplication.

---

## 2. Upstream Absorption Candidates

### 2a. `McpTool` Struct + `list_tools()` / `tool_to_method()`

airSpring and wetSpring both independently implement identical MCP tool definition patterns.
This could be promoted to a `barracuda::ipc::mcp` shared utility:

```rust
pub struct McpTool {
    pub name: &'static str,
    pub description: &'static str,
    pub input_schema: fn() -> Value,
}
```

Both springs use `list_tools()` and `tool_to_method()`. The struct, list builder, and
test pattern (`all_tools_have_input_schema`, `tool_names_are_prefixed`,
`all_tools_have_method_mapping`) are identical.

### 2b. `PythonBaseline` Provenance Registry

Both springs maintain a `provenance.rs` with `PythonBaseline` records. The struct
and category enum could be shared upstream:

```rust
pub struct PythonBaseline {
    pub binary: &'static str,
    pub script: Option<&'static str>,
    pub commit: &'static str,
    pub date: &'static str,
    pub category: BaselineCategory,
}
```

### 2c. `ValidationSink` Trait (from ludoSpring V23)

A testable output trait for validation harnesses. Currently proposed but not yet
absorbed. airSpring's 91 binaries would benefit.

### 2d. Missing `barracuda::special` Functions

airSpring's drought index and gamma CDF code would benefit from:
- `regularized_gamma_q` (complement of `regularized_gamma_p`)
- `lower_incomplete_gamma`, `upper_incomplete_gamma`
- `digamma`
- `beta`, `ln_beta`

### 2e. `BatchedOdeRK45F64`

A GPU-accelerated adaptive RK45 ODE integrator would enable GPU promotion of:
- Richards equation (currently CPU Picard iteration)
- Coupled runoff-infiltration dynamics
- Cover crop phenology models

---

## 3. Evolution Gaps Remaining

### Tier B (Needs Adaptation)
- `nonlinear_solver` — Nelder-Mead/BFGS for multi-parameter optimization
- `rk45_adaptive` — Adaptive RK45 ODE for soil/water dynamics
- `seasonal_pipeline` — GPU stages 1-2 (ET₀ + Kc)
- `atlas_stream` — Streaming multi-year ET₀

### Tier C (Needs New Primitive)
- `data_client` — HTTP/JSON for Open-Meteo, NOAA CDO
- `validation_sink` — `ValidationSink` trait for testable harness output

---

## 4. What airSpring Learned (Relevant to Upstream)

### 4a. `deny.toml` C-Dependency Ban Template

14 crates banned for ecoBin compliance. Other springs should adopt:
```toml
deny = [
    { crate = "openssl-sys" }, { crate = "libz-sys" },
    { crate = "zstd-sys" }, { crate = "curl-sys" },
    { crate = "ring" }, { crate = "native-tls" },
    # ... 8 more
]
```

### 4b. Lint Convention: `#[allow(reason)]` vs `#[expect(reason)]`

- `#[expect(reason)]` for lints **known** to fire (compile error if they don't)
- `#[allow(reason)]` for **blanket** test module suppressions (may or may not fire)

53 test modules migrated. The key learning: `#[expect(clippy::unwrap_used)]` on a
test module that doesn't use `unwrap()` causes an unfulfilled-expectation error.

### 4c. `f64::total_cmp` for NaN-Safe Ordering

`partial_cmp().unwrap_or(Equal)` silently treats NaN as equal. `f64::total_cmp`
provides deterministic ordering. All springs should migrate.

### 4d. `assert!` → `Result<T, InputError>` in Public APIs

Library functions should never panic on bad input. Pattern:
```rust
pub fn wind_speed_at_2m(uz: f64, z_m: f64) -> Result<f64> {
    if z_m <= 0.0 {
        return Err(AirSpringError::InvalidInput("...".into()));
    }
    Ok(uz * 4.87 / (67.8f64.mul_add(z_m, -5.42)).ln())
}
```
Callers: `.expect()` in binaries, `.unwrap()` in tests.

---

## 5. Test Results

| Metric | Value |
|--------|-------|
| Library tests | 908 |
| Integration tests | 299 |
| Validation binaries | 91 |
| Clippy warnings | 0 |
| Format issues | 0 |
| Unsafe code | 0 |
| C dependencies | 0 |
| TODO/FIXME markers | 0 |
| Mocks in production | 0 |
