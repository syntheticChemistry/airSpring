# airSpring V0.10.0 — barraCuda/toadStool Evolution Handoff

**Date:** 2026-03-18
**From:** airSpring V0.10.0
**To:** barraCuda team, toadStool team
**License:** AGPL-3.0-or-later
**Covers:** Deep audit execution, OrExit zero-panic, provenance expansion, tolerance centralization, upstream absorption candidates
**Supersedes:** AIRSPRING_V090_BARRACUDA_TOADSTOOL_EVOLUTION_HANDOFF_MAR18_2026.md

---

## Executive Summary

airSpring v0.10.0 completes a deep audit execution pass: 63 Python baseline
provenance records, zero-panic validation across all 91 binaries, centralized
tolerances, smart module refactors, and Rust 2024 lint evolution. All 20 ops
remain upstream via `BatchedElementwiseF64`. Zero local WGSL. Zero local math
duplication. 908 lib tests, 0 failures, 0 clippy, 0 unsafe, 0 C deps.

This handoff documents the full barraCuda surface, proven patterns from the
deep audit, and upstream absorption candidates for the barraCuda/toadStool team.

---

## 1. Current barraCuda Surface (v0.3.5)

airSpring uses ~700+ `barracuda::` API calls across ~124 files:

| Module | Key APIs | Usage |
|--------|----------|-------|
| `barracuda::stats` | `mean`, `rmse`, `pearson_correlation`, `bootstrap_ci`, `shannon`, `bray_curtis`, `hargreaves_et0_batch`, `norm_ppf`, regression, diversity, jackknife | Dominant — testutil, eco, gpu, validation |
| `barracuda::ops` | `BatchedElementwiseF64` (20 ops), `FusedMapReduceF64`, `VarianceF64`, `MovingWindowStats`, `KrigingF64`, `AutocorrelationF64`, `DiversityFusionGpu` | GPU orchestration |
| `barracuda::device` | `WgpuDevice`, `Fp64Strategy`, `GpuDriverProfile`, `probe_f64_builtins`, `test_pool` | Device creation and probing |
| `barracuda::shaders` | `precision::cpu::kahan_sum`, `provenance::*` | Numerics and provenance tracking |
| `barracuda::pde` | `richards`, `crank_nicolson`, `RichardsGpu` | Richards equation (1D unsaturated flow) |
| `barracuda::linalg` | `tridiagonal_solve`, `ridge::ridge_regression` | Linear algebra |
| `barracuda::optimize` | `brent`, `BrentGpu`, `nelder_mead`, `multi_start` | Root-finding, optimization |
| `barracuda::special` | `gamma::regularized_gamma_p`, `gamma::ln_gamma` | Gamma function for SPI drought index |
| `barracuda::validation` | `ValidationHarness`, `OrExit`, `exit_no_gpu`, `gpu_required` | Validation harness for all 91 binaries |

**Write → Absorb → Lean status**: All 20 ops upstream, `local_dispatch` fully retired.

---

## 2. What V0.10.0 Deep Audit Achieved

### 2a. OrExit Zero-Panic Migration

All 91 validation binaries migrated from `.expect()`/`.unwrap()` to `.or_exit()`.
~180 call sites converted. The `OrExit` trait from `barracuda::validation` handles
both `Result<T, E>` and `Option<T>` with clean `eprintln!` + `exit(1)`.

**toadStool action:** The `OrExit` pattern proved highly effective. Consider promoting
`OrExit` to a first-class `barracuda::validation` export (it already is, but documenting
as a recommended pattern for all spring validation binaries would help).

### 2b. Python Baseline Provenance Registry (11→63)

Every CI validation binary now has a `PythonBaseline` record with script path, commit
hash, date, and category. The registry enables automated provenance audits.

**toadStool action:** The `PythonBaseline` struct and `BaselineCategory` enum are
identical across airSpring and wetSpring. Upstream absorption into
`barracuda::validation::provenance` would eliminate duplication.

### 2c. Tolerance Centralization

All hardcoded thresholds in validation binaries now reference named `Tolerance`
constants from `airspring_barracuda::tolerances`:

| Tolerance | Value | Used By |
|-----------|-------|---------|
| `IA_CRITERION` | 0.80 | validate_lysimeter |
| `RMSE_MAXIMUM` | 1.5 mm/day | validate_lysimeter |
| `R2_MINIMUM` | 0.85 | validate_atlas |
| `SOIL_HYDRAULIC` | 0.01 | validate_richards |
| `RICHARDS_TRANSIENT` | 50.0 | validate_richards |
| `RICHARDS_STEADY` | 5.0 | validate_richards |
| `BIO_DIVERSITY_SHANNON` | 0.01 | validate_diversity |
| `BIO_DIVERSITY_SIMPSON` | 0.01 | validate_diversity |
| `BIO_BRAY_CURTIS` | 0.02 | validate_diversity |

**toadStool action:** This pattern (named, justified, centralized tolerances) should be
recommended across all springs. Consider `barracuda::validation::Tolerance` as a
shared type.

### 2d. Smart Module Refactoring

| Module | Before | After | Strategy |
|--------|--------|-------|----------|
| `data/provider.rs` | 781 LOC | 4 modules (~300 + 160 + 110 + 170) | Extracted providers to submodules, kept trait + helpers |
| `gpu/evolution_gaps.rs` | 731 LOC | 635 LOC + `resolved_issues.rs` (~100) | Extracted historical issues to separate module |

### 2e. Cast Evolution

2 new safe cast helpers: `f64_i32()`, `usize_i32()`. Progressive migration strategy:
when touching a validation binary, add `use cast::*` and replace raw casts, then add
per-binary `#[expect]` to progressively tighten. Goal: remove crate-level allows entirely.

### 2f. Rust 2024 Lint Migration

`#[allow()]` → `#[expect()]` with reason strings in `tests/common/mod.rs`. The key
learning: `#[expect]` causes a compile error if the lint doesn't fire — use `#[allow]`
for blanket suppressions on test modules, `#[expect]` for specific known-to-fire cases.

---

## 3. Upstream Absorption Candidates

### 3a. `McpTool` Struct + `list_tools()` / `tool_to_method()`

airSpring and wetSpring both independently implement identical MCP tool definition
patterns. Promote to `barracuda::ipc::mcp`:

```rust
pub struct McpTool {
    pub name: &'static str,
    pub description: &'static str,
    pub input_schema: fn() -> Value,
}
```

### 3b. `PythonBaseline` Provenance Registry

Both springs maintain `provenance.rs` with identical `PythonBaseline` records. The
struct and `BaselineCategory` enum should be shared upstream.

### 3c. `ValidationSink` Trait (from ludoSpring V23)

A testable output trait for validation harnesses. airSpring's 91 binaries would benefit.

### 3d. Missing `barracuda::special` Functions

- `regularized_gamma_q` (complement of `regularized_gamma_p`)
- `lower_incomplete_gamma`, `upper_incomplete_gamma`
- `digamma`
- `beta`, `ln_beta`

### 3e. `BatchedOdeRK45F64`

GPU-accelerated adaptive RK45 ODE integrator for:
- Richards equation (currently CPU Picard iteration)
- Coupled runoff-infiltration dynamics
- Cover crop phenology models

---

## 4. Evolution Gaps Remaining

### Tier B (Needs Adaptation)
- `nonlinear_solver` — Nelder-Mead/BFGS for multi-parameter optimization
- `rk45_adaptive` — Adaptive RK45 ODE for soil/water dynamics
- `seasonal_pipeline` — GPU stages 3-4 (water balance + yield response)
- `atlas_stream` — Streaming multi-year multi-station ET₀

### Tier C (Needs New Primitive)
- `data_client` — Sovereign HTTP/JSON for Open-Meteo, NOAA CDO
- `validation_sink` — `ValidationSink` trait for testable harness output

---

## 5. What airSpring Learned (Relevant to Upstream Evolution)

### 5a. `deny.toml` C-Dependency Ban Template

14 crates banned for ecoBin compliance. Template for all springs:
```toml
deny = [
    { crate = "openssl-sys" }, { crate = "libz-sys" },
    { crate = "zstd-sys" }, { crate = "curl-sys" },
    { crate = "ring" }, { crate = "native-tls" },
    { crate = "bzip2-sys" }, { crate = "lzma-sys" },
    { crate = "libgit2-sys" }, { crate = "freetype-sys" },
    { crate = "cmake" }, { crate = "cc" },
    { crate = "pkg-config" }, { crate = "vcpkg" },
]
```

### 5b. Lint Convention

- `#[expect(reason)]` for lints **known** to fire (compile error if they don't)
- `#[allow(reason)]` for **blanket** test module suppressions

### 5c. `f64::total_cmp` for NaN-Safe Ordering

Replaces `partial_cmp().unwrap_or(Equal)` which silently treats NaN as equal.

### 5d. Determinism Contract for Nautilus Brain

The brain is structurally deterministic (ridge regression, tournament selection, fixed
topology). No RNG seed needed. Document this pattern for other springs using
bingoCube/nautilus.

### 5e. Data Provenance Accession IDs

All data sources now carry formal identifiers:
- ECMWF Copernicus CDS: `reanalysis-era5-single-levels`
- USDA SCAN: network station IDs per experiment
- AmeriFlux: DOI per site (e.g. `10.17190/AMF/1246153`)
- USDA NASS: `source_desc=SURVEY`
- NCBI 16S: SRA BioProject accession (e.g. `PRJNA*`)
- NOAA CDO: dataset ID `GHCND`

---

## 6. Test Results (V0.10.0)

| Metric | Value |
|--------|-------|
| Library tests | 908 |
| Integration tests | 299 |
| Forge tests | 61 |
| Property tests | 22 |
| Validation binaries | 91 |
| Python baselines | 63 (all with provenance) |
| Clippy warnings | 0 (pedantic + nursery) |
| Format issues | 0 |
| Unsafe code | 0 (`#![forbid(unsafe_code)]` both crates) |
| C dependencies | 0 (deny.toml enforced) |
| TODO/FIXME markers | 0 |
| Mocks in production | 0 |
| Zero-panic binaries | 91/91 |
| Named tolerances | 63 (4 submodules) |

---

## 7. Resolved barraCuda Issues (Historical)

All issues resolved as of S54+S66. No open upstream blockers.

| ID | Issue | Status |
|----|-------|--------|
| TS-001 | `pow_f64` non-integer exponents | **RESOLVED** (S54) |
| TS-002 | No Rust orchestrator for `batched_elementwise_f64` | **RESOLVED** (S54) |
| TS-003 | `acos_simple`/`sin_simple` approximations | **RESOLVED** (S54) |
| TS-004 | `FusedMapReduceF64` buffer conflict N≥1024 | **RESOLVED** (S54) |
| P0 | GPU dispatch bind-group panic | **RESOLVED** (S66 explicit BGL) |
