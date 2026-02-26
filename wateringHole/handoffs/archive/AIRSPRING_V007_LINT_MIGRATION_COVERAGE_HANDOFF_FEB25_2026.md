# airSpring V007: Lint Migration + Coverage 97.58% + Absorption Readiness

**Date**: February 25, 2026
**From**: airSpring (Precision Agriculture — v0.4.2, 555 total tests)
**To**: ToadStool/BarraCuda core team
**ToadStool PIN**: `02207c4a` (S62+, DF64 expansion, unified_hardware)
**License**: AGPL-3.0-or-later
**Supersedes**: V006 (archived)

---

## Executive Summary

airSpring v0.4.2 completes two audit passes and reaches production-quality gates:

- **555 tests** (407 lib + 95 integration + 53 forge), 0 failures
- **97.58% line coverage** (was 89% pre-audit, 96.84% after pass 1, 97.58% after pass 2)
- **Zero** clippy warnings (pedantic via `[lints.clippy]` in Cargo.toml — modern Rust pattern)
- **Zero** `unsafe`, zero `unwrap()` in library code, zero files > 850 lines
- **16/16 validation binaries PASS** (341 checks), 75/75 cross-validation MATCH
- **8 GPU orchestrators** wired to ToadStool primitives
- **4 metalForge modules** ready for upstream absorption
- Clippy lint configuration migrated from `#![warn]` attributes to `[lints.clippy]` in Cargo.toml
- GPU test suite refactored by domain cohesion (functional / evolution / determinism)
- All benchmark JSON files have complete provenance (script, commit, date, command, reproduction note)
- Baseline Commit Lineage documented (94cc51d Phase 1, 3afc229 Phase 2)

---

## Part 1: BarraCuda Consumption Map (14 primitives)

### CPU Primitives

| BarraCuda Primitive | airSpring Module | Purpose | Tests |
|--------------------|-----------------|---------|:-----:|
| `linalg::tridiagonal_solve_f64` | `eco::richards` | Thomas algorithm | 31 |
| `linalg::ridge::ridge_regression` | `eco::correction` | Sensor correction | 17 |
| `stats::pearson_correlation` | `testutil::stats` | Cross-station r | 34 |
| `stats::spearman_correlation` | `testutil::stats` | Rank correlation | 34 |
| `stats::bootstrap_ci` | `testutil::bootstrap` | Confidence intervals | 3 |
| `validation::ValidationHarness` | All 16 binaries | Pass/fail harness | 33 |

### GPU Primitives

| BarraCuda Primitive | airSpring Module | Dispatch | Tests |
|--------------------|-----------------|----------|:-----:|
| `ops::batched_elementwise_f64` (op=0) | `gpu::et0` | **GPU-FIRST** | 20 |
| `ops::batched_elementwise_f64` (op=1) | `gpu::water_balance` | **GPU-STEP** | 18 |
| `ops::kriging_f64::KrigingF64` | `gpu::kriging` | **INTEGRATED** | 17 |
| `ops::fused_map_reduce_f64` | `gpu::reduce` | **GPU N≥1024** | 26 |
| `ops::moving_window_stats` | `gpu::stream` | **WIRED** | 20 |
| `pde::richards::solve_richards` | `gpu::richards` | **WIRED** | 4 |
| `optimize::nelder_mead` | `gpu::isotherm` | **WIRED** | 10 |
| `optimize::multi_start_nelder_mead` | `gpu::isotherm` | **WIRED** | 10 |

---

## Part 2: What airSpring Learned (for BarraCuda to evolve)

### 2.1 Clippy Lint Configuration via Cargo.toml

airSpring migrated from `#![warn(clippy::pedantic)]` in `lib.rs` to:

```toml
[lints.clippy]
pedantic = { level = "warn", priority = -1 }
module_name_repetitions = "allow"
must_use_candidate = "allow"
return_self_not_must_use = "allow"
cast_precision_loss = "allow"
```

This is the modern Rust pattern (stable since 1.74). Benefits:
- Lints apply uniformly to all crate targets (lib, bins, tests)
- No need for per-item `#[allow]` scattered across files
- Overrides work consistently regardless of CLI flags (`-D warnings` etc.)
- `cast_precision_loss = "allow"` is justified: scientific code casts integer counts to f64; all N < 2^53

**Recommendation**: ToadStool/BarraCuda should migrate `barracuda` crate to the same pattern. We removed ~28 redundant per-item `#[allow(clippy::cast_precision_loss)]` annotations across 14 files after adding the crate-level allow.

### 2.2 `mul_add()` for FMA Parity

All `a * b + c` patterns converted to `mul_add()` across 13+ locations (5 files). This uses hardware FMA when available (one rounding instead of two). WGSL shaders should ensure `fma_f64` is available for numerical parity between CPU and GPU paths.

### 2.3 Named Constants in Richards PDE

8 named constants replace magic numbers in `eco::richards`:
- `VG_H_ABS_MAX` (1e8 cm) — suction cap for van Genuchten
- `VG_POWF_MAX` (50) — exponent overflow guard
- `SATURATED_CAPACITY` (1e-10) — near-saturation capacity floor
- `CAPACITY_H_MIN` (1e-3 cm) — avoids division-by-zero in dθ/dh
- `DT_FACTOR` (10) — Picard timestep reduction factor
- `MAX_PICARD_ITER` (50) — Picard convergence limit
- `PICARD_TOLERANCE` (1e-6 cm) — convergence threshold
- `MASS_BALANCE_EMPTY_GUARD` (0.0) — empty profile safety

**Recommendation**: Upstream `pde::richards` should adopt matching constants for maintainability and cross-Spring consistency.

### 2.4 Preallocation Outside Loops

Picard iteration in Richards solver preallocates `a`, `b`, `c`, `d`, `h_prev`, `h_old`, `q_buf` outside the time-stepping loop — eliminates per-iteration allocations on the hot path. Same pattern should be applied to upstream `pde::richards::solve_richards`.

### 2.5 GPU Test Suite Structure

airSpring refactored `gpu_integration.rs` (1076 lines → 754) by domain cohesion:
- `tests/common/mod.rs` — shared `try_create_device()` + `device_or_skip!` macro
- `tests/gpu_integration.rs` — functional orchestrator tests only
- `tests/gpu_evolution.rs` — evolution gap catalog + ToadStool issue tracking (no GPU needed)
- `tests/gpu_determinism.rs` — bit-identical rerun validation

This pattern separates concerns: functional tests require a GPU, metadata tests don't, determinism tests are cross-cutting. Recommended for all Springs.

### 2.6 Benchmark Provenance Standard

All 9 benchmark JSON files now embed complete provenance:

```json
"_provenance": {
    "script": "control/fao56/penman_monteith.py",
    "baseline_commit": "94cc51d",
    "baseline_command": "python control/fao56/penman_monteith.py",
    "date": "2026-02-16",
    "python_version": "3.10.12",
    "repository": "ecoPrimals/airSpring",
    "reproduction_note": "Re-run baseline_command at baseline_commit to regenerate expected values"
}
```

All 8 Python baseline scripts embed matching Provenance blocks in their docstrings.

---

## Part 3: metalForge Absorption Targets (P1)

Four modules in `metalForge/forge/` (53 tests, all passing) are ready for absorption:

| Module | Target | Tests | Why Absorb |
|--------|--------|:-----:|------------|
| `metrics` | `barracuda::stats::metrics` | 11 | RMSE, MBE, NSE, IA, R² — every Spring needs these |
| `regression` | `barracuda::stats::regression` | 11 | Linear, quadratic, exponential, logarithmic fitting |
| `moving_window_f64` | `barracuda::ops::moving_window_stats_f64` | 7 | f64 CPU moving window (upstream is f32 GPU) |
| `hydrology` | `barracuda::ops::hydrology` | 13 | Hargreaves ET₀, crop Kc, soil water balance |

**Total: 42 tests, pure arithmetic, zero dependencies beyond std.**

See `metalForge/ABSORPTION_MANIFEST.md` for complete function signatures, validation provenance, edge case coverage, and post-absorption rewiring plan.

### Absorption Procedure

1. Copy `forge/src/{metrics,regression,moving_window,hydrology}.rs` into barracuda
2. Add `pub mod` entries to `stats/mod.rs` and `ops/mod.rs`
3. Run forge tests alongside barracuda suite
4. Bump barracuda version
5. airSpring rewires `use barracuda::stats::metrics::*` etc., removes local code
6. metalForge code archived (kept for provenance)

---

## Part 4: Actionable Items for ToadStool/BarraCuda

### P0 — Blocking

*None. airSpring is not blocked on any ToadStool work.*

### P1 — High Value

| # | Item | Justification |
|---|------|---------------|
| 1 | **Absorb 4 metalForge modules** | metrics/regression/moving_window_f64/hydrology — 42 tests, pure arithmetic, every Spring benefits |
| 2 | **`fma_f64` WGSL shader instruction** | `mul_add()` CPU parity — needed for numerical consistency |
| 3 | **`crank_nicolson_f64`** | f32 version exists; Richards PDE requires f64 precision |
| 4 | **Named constants in `pde::richards`** | Match airSpring's 8 named constants for cross-Spring consistency |
| 5 | **Preallocation pattern in `pde::richards`** | Picard buffers outside loops — hot-path allocation elimination |
| 6 | **`[lints.clippy]` in Cargo.toml** | Modern Rust pattern — migrate all crates for consistency |

### P2 — Nice to Have

| # | Item | Justification |
|---|------|---------------|
| 7 | Batch PDE dispatch | `pde::richards::solve_batch_gpu` for M soil columns |
| 8 | BFGS optimizer | Gradient-based fitting for smooth objectives |
| 9 | `adaptive_penalty` | Constrained optimization for parameter bounds |
| 10 | Dual Kc GPU shader | `batched_elementwise_f64` op=8 for multi-field crop coefficients |
| 11 | f64 moving window GPU | Verify `StreamSmoother` f64 precision on GPU |

### P3 — Research

| # | Item | Justification |
|---|------|---------------|
| 12 | `unified_hardware` | metalForge → `HardwareDiscovery` + `ComputeScheduler` |
| 13 | Surrogate learning | Richards PDE → neural surrogate for real-time irrigation |
| 14 | Coupled soil-plant ODE | Dynamic root water uptake models |

---

## Part 5: Evolution Readiness Summary

| Tier | Count | Description |
|------|:-----:|-------------|
| A (Integrated) | 8 | GPU primitive wired, validated, GPU-first with CPU fallback |
| B (Ready to wire) | 11 | Upstream primitive exists, needs domain wiring |
| C (Needs new) | 1 | HTTP/JSON client (not GPU) |
| Absorbed | 3 | ValidationHarness, van_genuchten, isotherm NM |
| metalForge ready | 4 | metrics, regression, moving_window_f64, hydrology |

See `barracuda/EVOLUTION_READINESS.md` for the full tier breakdown.

---

## Part 6: Quality Gates (All Green)

```
cargo fmt --check              → clean
cargo clippy -- -D warnings    → 0 warnings (pedantic via [lints.clippy])
cargo doc --no-deps            → builds, 0 warnings
cargo test                     → 555 total (407 lib + 95 integration + 53 forge)
cargo llvm-cov --lib           → 97.58% line coverage
unsafe code                    → zero
unwrap() in lib                → zero (all in #[cfg(test)])
files > 1000 lines             → zero (max: 845 lines)
validation binaries            → 16/16 PASS (341 checks)
cross-validation               → 75/75 MATCH (tol=1e-5)
Python baselines               → 344/344 PASS (8 scripts, provenance documented)
```

### Coverage by Module

| Module | Line Coverage |
|--------|:------------:|
| `eco::evapotranspiration` | 99.69% |
| `eco::water_balance` | 99.23% |
| `eco::soil_moisture` | 99.55% |
| `eco::dual_kc` | 98.74% |
| `eco::isotherm` | 98.33% |
| `eco::correction` | 98.21% |
| `eco::richards` | 97.23% |
| `eco::crop` | 100.00% |
| `eco::sensor_calibration` | 100.00% |
| `validation` | 100.00% |
| `testutil/*` | 96.36–100% |
| `io::csv_ts` | 97.89% |
| `gpu::*` | 93.77–97.42% |
| **TOTAL** | **97.58%** |

---

## Part 7: Cross-Spring Contributions

### What airSpring Fixed Upstream (benefits ALL Springs)

| Fix | Trigger | Impact |
|-----|---------|--------|
| TS-001: `pow_f64` fractional exponents | van Genuchten θ(h) | All Springs using exponential math |
| TS-003: `acos` precision boundary | Radiation calculations | All Springs using trig in f64 shaders |
| TS-004: reduce buffer N≥1024 | Seasonal statistics | All Springs using `FusedMapReduceF64` |
| Richards PDE absorbed (S40) | 1D unsaturated flow | Available to all Springs via `pde::richards` |

### What airSpring Learned From Other Springs

| Source | Primitive | airSpring Use |
|--------|----------|---------------|
| hotSpring | `pow_f64`, df64 core | VG retention, atmospheric pressure |
| wetSpring | `kriging_f64` | Soil moisture spatial interpolation |
| wetSpring | `fused_map_reduce_f64` | Seasonal ET₀ totals, field aggregation |
| wetSpring | `moving_window_stats` | IoT sensor stream smoothing |
| wetSpring | `ridge_regression` | Sensor correction pipeline |
| neuralSpring | `nelder_mead` + `multi_start` | Isotherm fitting (Langmuir/Freundlich) |
| neuralSpring | `ValidationHarness` | All 16 validation binaries |

---

## Part 8: Artifacts

| Document | Location |
|----------|----------|
| This handoff | `wateringHole/handoffs/AIRSPRING_V007_LINT_MIGRATION_COVERAGE_HANDOFF_FEB25_2026.md` |
| Previous handoff | `wateringHole/handoffs/archive/AIRSPRING_V006_DEEP_AUDIT_ABSORPTION_HANDOFF_FEB25_2026.md` |
| Evolution readiness | `barracuda/EVOLUTION_READINESS.md` |
| Absorption manifest | `metalForge/ABSORPTION_MANIFEST.md` |
| Evolution gaps (code) | `barracuda/src/gpu/evolution_gaps.rs` |
| Cross-spring evolution | `specs/CROSS_SPRING_EVOLUTION.md` |
| BarraCuda requirements | `specs/BARRACUDA_REQUIREMENTS.md` |
| Baseline commit lineage | `specs/README.md` (Baseline Commit Lineage section) |

---

## Version History

| Version | Date | Key Changes |
|---------|------|-------------|
| V001 | 2026-02-25 | Initial GPU handoff (v0.3.8) |
| V002 | 2026-02-25 | Dual Kc, cover crops, deep debt cleanup (v0.3.10) |
| V003 | 2026-02-25 | Richards + isotherm GPU wiring (v0.4.0) |
| V004 | 2026-02-25 | ToadStool S62 sync, multi-start NM (v0.4.1) |
| V005 | 2026-02-25 | Complete status, GPU integration tests (v0.4.2) |
| V006 | 2026-02-25 | Deep audit pass 1, 96.84% coverage, testutil split (v0.4.2+) |
| **V007** | **2026-02-25** | **Lint migration, 97.58% coverage, 555 tests, evolution readiness, absorption roadmap (v0.4.2)** |

---

*End of V007 handoff. Direction: airSpring → ToadStool (unidirectional).
Supersedes V006 (archived). Next handoff: V008 after metalForge absorption
completes or new experiment requires upstream primitives.*
