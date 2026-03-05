<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
# airSpring V0.6.8 Deep Debt Audit Round 2 — ToadStool/barraCuda Handoff

**Date**: March 4, 2026
**From**: airSpring V0.6.8 (ecology/agriculture validation Spring)
**To**: ToadStool S93+ (hardware dispatch) + barraCuda 0.3.1+ (sovereign math)
**License**: AGPL-3.0-or-later

---

## Executive Summary

airSpring completed a comprehensive deep debt audit (round 2) with the following
outcomes relevant to ToadStool/barraCuda evolution:

- **Sovereignty hardening**: `ShaderProvenance.airspring_use` renamed to `domain_use`
  (24 sites). GPU debug labels stripped of primal prefixes. Primal code has
  self-knowledge only, discovers other primals at runtime via capabilities.
- **Dependency gating**: `ureq` (which pulls `ring` C/asm) gated behind
  `standalone-http` feature. Consumers using Songbird TLS get a pure Rust build.
  `testutil` gated behind `testutil` feature (excluded from production).
- **Fallible constructors**: `UsdaNassClient::try_new()` and `OpenMeteoClient::try_new()`
  return `Result<Self, DataError>` instead of panicking on missing HTTP transport.
- **Large-file refactoring**: 3 files exceeding 800 LOC split into focused modules
  while preserving public API and all 1132 tests.
- **Zero-copy I/O**: CSV parser (`io::csv_ts`) eliminates per-row `Vec<&str>` allocation.
- All 1132 tests pass, zero clippy pedantic+nursery warnings, zero fmt diffs.

---

## Part 1: Patterns for Upstream Adoption

### 1.1 Sovereignty: `domain_use` Pattern

The `ShaderProvenance` struct tracked which airSpring domain used each shader.
The field was named `airspring_use` — a sovereignty violation (hardcoded primal name
in a struct that could be shared across Springs).

**Pattern**: Domain-specific fields in shared structs should use generic names
(`domain_use`, `domain_context`) rather than primal-specific names (`airspring_use`,
`hotspring_purpose`). Each Spring fills the field with its own domain context.

**Recommendation for barraCuda**: If `ShaderProvenance` moves upstream, use
`domain_use` as the canonical field name.

### 1.2 Feature Gating for Pure Rust Builds

airSpring gates two dependencies behind Cargo features:

```toml
[features]
default = ["testutil", "standalone-http"]
testutil = []
standalone-http = ["dep:ureq"]
```

- `standalone-http`: Gates `ureq` (which pulls `ring` → C/asm). Consumers using
  Songbird for TLS can build without: `cargo build --no-default-features --features testutil`
- `testutil`: Gates synthetic data generators and stat helpers used only in tests

**Pattern**: Any dependency that brings non-Rust code should be feature-gated.
The `discover_transport()` function returns `Option<Box<dyn HttpTransport>>` so
callers handle the "no transport available" case gracefully.

**Recommendation for barraCuda**: Apply the same pattern if HTTP transport is
needed for data providers. `ureq` is fine as a default, but Songbird should be
the sovereign path.

### 1.3 Fallible Constructors

Data providers previously panicked in `new()` if no HTTP transport was available.
Now they provide `try_new() -> Result<Self, DataError>` with the old `new()` as
a convenience wrapper that calls `.expect()`.

**Pattern**: Any constructor that depends on runtime discovery (GPU device, HTTP
transport, file system) should be fallible. Reserve `new()` for infallible
construction; use `try_new()` for anything that can fail.

### 1.4 Large-File Refactoring Strategy

Three files were refactored:

| File | Before | After | Strategy |
|------|--------|-------|----------|
| `bench_cross_spring_evolution.rs` | 951 LOC (monolithic binary) | `main.rs` + 5 focused submodules | Split by benchmark domain (precision, domain, gpu_ops, paper12, pipeline) |
| `seasonal_pipeline.rs` | 888 LOC | `mod.rs` + `multi_field.rs` | Extract multi-field orchestration from core pipeline |
| `cross_spring_absorption.rs` | 921 LOC | `main.rs` + `s70_cross_spring.rs` | Extract S70+ cross-spring evolution tests into submodule |

**Pattern**: Convert single file to directory module (`file.rs` → `file/mod.rs`),
extract cohesive subsections into submodules, re-export public types from `mod.rs`.
Public API unchanged, all callers unaffected.

---

## Part 2: barraCuda Evolution Intelligence

### 2.1 Consumed Primitives Catalog (March 4, 2026)

airSpring consumes 25+ barraCuda primitives across 7 domains:

| Domain | Primitives | airSpring Modules |
|--------|-----------|-------------------|
| **Device** | `WgpuDevice`, `F64BuiltinCapabilities`, `Fp64Strategy` | All `gpu::*` |
| **Ops** | `BatchedElementwiseF64` (ops 0-13), `KrigingF64`, `FusedMapReduceF64`, `MovingWindowStats`, `DiversityFusionGpu` | ET₀, WB, VG, kriging, stats, diversity |
| **Optimize** | `brent`, `BrentGpu`, `nelder_mead`, `multi_start_nelder_mead` | VG inverse, infiltration, isotherm |
| **Stats** | `pearson_correlation`, `rmse`, `mbe`, `bootstrap_ci`, `BootstrapMeanGpu`, `JackknifeMeanGpu`, `norm_ppf`, `fit_linear/quad/exp/log`, `shannon`, `simpson`, `bray_curtis` | Sensor validation, uncertainty, diversity |
| **PDE** | `CrankNicolsonConfig`, `HeatEquation1D`, `richards`, `RichardsGpu` | Richards equation |
| **Linalg** | `ridge_regression` | Sensor calibration |
| **Validation** | `ValidationHarness`, `exit_no_gpu`, `gpu_required`, `check`, `Tolerance` | All binaries |

### 2.2 No Duplicate Math

airSpring delegates all math to barraCuda. The only intentionally local functions are:
- `testutil::stats::index_of_agreement` — different division-guard convention than upstream
- `testutil::stats::nash_sutcliffe` — same convention difference
- `testutil::stats::r_squared` — Pearson r² vs SS-based (= NSE) in upstream

These are test utilities, not production code, and the semantic difference is documented.

### 2.3 Evolution Gaps (28 Tier A, 4 Tier B, 1 Tier C)

28 Tier A gaps fully integrated. Remaining:
- **Tier B**: `nonlinear_solver` (BFGS available), `tridiagonal_batch` (via Richards),
  `rk45_adaptive` (available upstream), `seasonal_pipeline` (Stages 1-2 GPU)
- **Tier C**: `data_client` (HTTP/JSON for Open-Meteo/NOAA — now feature-gated)

### 2.4 Pending Absorption: 6 Local WGSL Shaders

6 local f32 shaders in `local_elementwise.wgsl` await promotion to f64 canonical
via `batched_elementwise_f64` ops 14-19. See V068 absorption handoff for details.

---

## Part 3: Quality Gates (March 4, 2026)

| Gate | Result |
|------|--------|
| `cargo fmt --check` | 0 diffs (both crates) |
| `cargo clippy --all-targets -W pedantic -W nursery` | 0 warnings (both crates) |
| `cargo doc --no-deps` | 0 warnings |
| `cargo test --workspace` (barracuda) | 1132 passed, 0 failed (846 lib + 286 integration) |
| `cargo test` (metalForge) | 62 passed, 0 failed |
| `#![forbid(unsafe_code)]` | Both crates |
| All files < 1000 lines | Yes (after refactoring) |
| Sovereignty violations | 0 (no hardcoded primal names in production code) |
| Mocks in production | 0 (all isolated to `#[cfg(test)]` or `testutil` feature) |
| `unwrap()` in binaries | 0 |
| barraCuda version | 0.3.1 standalone |
| Feature-gated C deps | `ureq`/`ring` behind `standalone-http` |

---

## Part 4: Learnings for barraCuda/ToadStool Evolution

1. **Feature-gating C dependencies works**: Springs can opt into pure Rust builds.
   The `discover_transport() -> Option<_>` pattern handles missing transports gracefully.

2. **Fallible constructors prevent runtime panics**: `try_new()` is safer than
   `new().expect()` for anything that depends on runtime state.

3. **Directory modules scale better than monolithic files**: Converting `file.rs` →
   `file/mod.rs` + submodules preserves API while improving navigation and ownership.

4. **Sovereignty requires active maintenance**: Hardcoded primal names creep into
   debug labels, struct fields, and log messages. Periodic sovereignty sweeps catch them.

5. **Zero-copy I/O matters at scale**: Eliminating per-row allocations in the CSV
   parser improved throughput for atlas-scale workloads (100 stations × 80 years).
