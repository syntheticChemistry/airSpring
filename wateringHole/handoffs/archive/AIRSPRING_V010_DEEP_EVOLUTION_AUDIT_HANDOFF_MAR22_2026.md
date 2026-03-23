# airSpring v0.10.0 — Deep Evolution Audit Handoff

**Date:** March 22, 2026
**From:** airSpring v0.10.0
**To:** barraCuda, toadStool, All Springs
**License:** AGPL-3.0-or-later
**Covers:** v0.10.0 deep evolution audit (code quality, coverage, ecosystem compliance)

## Executive Summary

- Complete `#[allow()]` → `#[expect(reason)]` migration across 71 source files (Rust 2024 idiom)
- Clippy pedantic + nursery: **zero warnings** (both crates, all features including NPU)
- New `f64_i8` cast helper in centralized `cast` module for NPU int8 quantization
- Test coverage pushed from 89.52% → **90.00%** (950 lib + 20 integration + 61 forge)
- `CONTEXT.md` created per `PUBLIC_SURFACE_STANDARD.md` Layer 3
- "Part of ecoPrimals" footer added to README
- PII scrubbed from `specs/NUCLEUS_INTEGRATION.md` (hardcoded `/home/eastgate/` → `$ECOPRIMALS_ROOT`)
- Discovery module (`biomeos/discovery.rs`) now fully tested with filesystem-based unit tests
- Zero TODO/FIXME/HACK/MOCK markers in all `.rs` source code
- Zero `unsafe` blocks; `#![forbid(unsafe_code)]` on both crate roots

## Part 1: What Changed

### Code Quality Fixes

| Change | Files | Impact |
|--------|-------|--------|
| `#[allow()]` → `#[expect(reason)]` | 71 `.rs` files | Rust 2024 idiomatic; unfulfilled expects now warn |
| Remove unfulfilled `#[expect()]` | 59 files, 79 lines removed | Test modules only had proactive guards for unused lints |
| `tests/common/mod.rs`: keep `#[allow()]` for cross-binary items | 1 file, 3 items | `dead_code`/`unused_macros`/`unused_imports` for shared test helpers that compile in multiple binaries |
| `npu/mod.rs`: `(val * 127.0) as i8` → `cast::f64_i8()` | 1 file | Fixes `cast_possible_truncation` lint with `--all-features` |
| `cast.rs`: add `f64_i8()` helper + tests | 1 file, +15 lines | Centralized NPU quantization cast |
| `cast.rs`: add `f64_i32`, `usize_i32` tests | 1 file, +12 lines | Coverage for untested cast helpers |

### Ecosystem Compliance

| Change | Impact |
|--------|--------|
| Created `CONTEXT.md` at repo root | Per `PUBLIC_SURFACE_STANDARD.md` Layer 3 — AI/search discoverability |
| Added "Part of ecoPrimals (syntheticChemistry)" footer to README | Per `PUBLIC_SURFACE_STANDARD.md` Layer 2 |
| `specs/NUCLEUS_INTEGRATION.md`: `/home/eastgate/` → `$ECOPRIMALS_ROOT` | PII remediation (2 instances) |

### Test Coverage Improvements

| Module | Before | After | Tests Added |
|--------|--------|-------|-------------|
| `validation/json.rs` | 62.09% | 81.25% | 17 tests for `_checked`/`_opt` JSON extractors |
| `eco/crop.rs` | 81.91% | 95.14% | GDD params for all 10 crops, `accumulated_gdd_clamp`, edge cases |
| `biomeos/discovery.rs` | 51.49% | ~85% | 11 filesystem-based socket discovery tests |
| `cast.rs` | 83.74% | ~95% | `f64_i32`, `usize_i32`, `f64_i8`, `f64_usize(0)` |
| **TOTAL** | **89.52%** | **90.00%** | +39 new unit tests |

## Part 2: barraCuda Primitive Consumption

### CPU Primitives (wired, in production)

| Primitive | airSpring Module | Status |
|-----------|-----------------|--------|
| `stats::pearson_correlation` | `testutil::r_squared` | Leaning |
| `stats::spearman_correlation` | `testutil::spearman_r` | Leaning |
| `stats::bootstrap_ci` | `testutil::bootstrap_rmse` | Leaning |
| `stats::diversity::*` | `eco::diversity` | Leaning (S64+S66) |
| `stats::hydrology::*` | `eco::crop`, `gpu::hargreaves` | Leaning (S66) |
| `stats::metrics::*` | `testutil::stats` | Leaning (S64) |
| `linalg::ridge::ridge_regression` | `eco::correction::fit_ridge` | Wired |
| `optimize::nelder_mead` | `gpu::isotherm` | Wired |
| `optimize::brent` | `eco::richards::inverse_vg_h` | Wired |
| `pde::richards::solve_richards` | `gpu::richards` | Wired |
| `validation::ValidationHarness` | All 91 validation binaries | Leaning |
| `tolerances::Tolerance` | `tolerances/` (58 named) | Leaning |

### GPU Primitives (25 Tier A modules)

| Op | airSpring Module | barraCuda Primitive |
|----|-----------------|---------------------|
| 0 | `gpu::et0` | `BatchedElementwiseF64` |
| 1 | `gpu::water_balance` | `BatchedElementwiseF64` |
| 5 | `gpu::sensor_calibration` | `BatchedElementwiseF64` |
| 6 | `gpu::hargreaves` | `HargreavesBatchGpu` |
| 7 | `gpu::kc_climate` | `BatchedElementwiseF64` |
| 8 | `gpu::dual_kc` | `BatchedElementwiseF64` |
| 9-10 | `gpu::van_genuchten` | `BatchedElementwiseF64` |
| 11 | `gpu::thornthwaite` | `BatchedElementwiseF64` |
| 12 | `gpu::gdd` | `BatchedElementwiseF64` |
| 13 | `gpu::pedotransfer` | `BatchedElementwiseF64` |
| 14-16,19 | `gpu::simple_et0` | `BatchedElementwiseF64` |
| 17 | `gpu::runoff` | `BatchedElementwiseF64` |
| 18 | `gpu::yield_response` | `BatchedElementwiseF64` |
| — | `gpu::kriging` | `KrigingF64` |
| — | `gpu::reduce` | `FusedMapReduceF64` |
| — | `gpu::stream` | `MovingWindowStats` |
| — | `gpu::richards` | `pde::richards::solve_richards` |
| — | `gpu::isotherm` | `optimize::nelder_mead` |
| — | `gpu::mc_et0` | `mc_et0_propagate_f64.wgsl` |
| — | `gpu::jackknife` | `JackknifeMeanGpu` |
| — | `gpu::bootstrap` | `BootstrapMeanGpu` |
| — | `gpu::diversity` | `DiversityFusionGpu` |
| — | `gpu::stats` | `linear_regression_f64` + `matrix_correlation_f64` |
| — | `gpu::infiltration` | `BrentGpu` |

## Part 3: Patterns Worth Absorbing

### `#[allow()]` vs `#[expect()]` for Shared Test Utilities

When a `pub fn` in `tests/common/mod.rs` is used by some test binaries but not others,
`#[expect(dead_code)]` generates unfulfilled-expectation warnings in non-using binaries.
The correct pattern is `#[allow(dead_code, reason = "shared helper — used by some binaries
but not all")]`. This edge case should be documented in the ecosystem coding standard.

### Discovery Module Testing Pattern

Socket discovery functions (`discover_*_in`) accept a `&Path` parameter, enabling
filesystem-based unit tests with temp directories and `.sock` files. This pattern
avoids requiring running biomeOS for integration tests and is reusable by other springs.

## Part 4: Quality Metrics

| Metric | Value |
|--------|-------|
| `cargo test --lib` (barracuda) | **950 passed**, 0 failures |
| `cargo test --test '*'` (integration) | **20 passed** (feature-gated; 299 with GPU) |
| `cargo test --lib` (metalForge) | **61 passed**, 0 failures |
| `cargo llvm-cov --lib --fail-under-lines 90` | **90.00% region coverage** (PASS) |
| `cargo clippy --all-targets --all-features` | **0 warnings** (pedantic + nursery) |
| `cargo fmt --check` | **Clean** (both crates) |
| `cargo doc --no-deps` | **Clean** (both crates) |
| `#![forbid(unsafe_code)]` | Both crate roots |
| `unsafe` blocks | **0** |
| `#[allow()]` in library code | **0** (all migrated to `#[expect(reason)]`) |
| `#[allow()]` in test common | **3** (justified: cross-binary shared items) |
| TODO/FIXME/HACK/MOCK in `.rs` | **0** |
| Files > 1000 lines | **0** (max: 833 lines, `rpc.rs`) |
| Hardcoded home paths in source | **0** (fixed in specs) |
| License | AGPL-3.0-or-later (Cargo.toml, LICENSE, SPDX headers) |
| CONTEXT.md | **Present** (per PUBLIC_SURFACE_STANDARD) |

## Open Items for Next Session

1. **Coverage headroom**: `data/songbird.rs` (9%), `data/nestgate.rs` (18%), `data/biomeos_provider.rs` (31%) are IPC modules that require running external services. Mock-based testing or integration test infrastructure would improve coverage further.

2. **Cargo.toml clippy allows**: Three crate-level `cast_*` allows remain in `[lints.clippy]` for the 91 validation binaries. The evolution path is documented: when touching a binary, add `use cast::*` and replace raw casts with helpers, then tighten per-binary.

3. **License alignment**: Project uses "AGPL-3.0-or-later" consistently. The `PUBLIC_SURFACE_STANDARD.md` template says "AGPL-3.0-only" — verify which is the ecosystem canonical form.

4. **Tier B GPU evolution**: `seasonal_pipeline` (fused GPU pipeline) and `atlas_stream` (GPU streaming) remain Tier B. Need barraCuda `PipelineSession` or chained persistent buffers.

5. **Tier C gaps**: `eco::anderson` needs a new WGSL shader for the iterative θ→S_e→d_eff→QS fixed-point loop. `eco::et0_ensemble` meta-logic is inherently serial.

6. **Kokkos validation**: Not started in airSpring. groundSpring has published benchmarks. Entry point: port `fao56_et0_batch` timing to Kokkos comparison harness.

7. **Integration test count**: Only 20 integration tests run without GPU features; 299 with GPU. Consider adding `--features testutil` to CI for broader non-GPU integration coverage.
