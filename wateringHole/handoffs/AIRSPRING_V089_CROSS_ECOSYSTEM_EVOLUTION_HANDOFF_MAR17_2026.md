# airSpring V0.8.9 — Cross-Ecosystem Evolution Handoff

**Date**: March 17, 2026
**From**: airSpring V0.8.9
**Scope**: Cross-ecosystem absorption of patterns from 7 springs + 4 primals
**Status**: All 10 plan items complete. Zero clippy warnings. Both crates green.

---

## Absorbed Patterns (with provenance)

### P0: Structural Identity and Safety

#### 1. Canonical `PRIMAL_NAME` / `PRIMAL_DOMAIN` (healthSpring V34)

- Added `pub const PRIMAL_NAME: &str = "airspring"` and `PRIMAL_DOMAIN: &str = "ecology"` to `lib.rs`
- `niche::NICHE_NAME` now delegates to `crate::PRIMAL_NAME` (single source of truth)
- Replaced hardcoded `"airspring"` strings in `validate_nucleus.rs`, `validate_biome_graph.rs`
- `niche.rs` test asserts against `crate::PRIMAL_NAME` instead of literal

**Source**: healthSpring V34 handoff — `PRIMAL_NAME`/`PRIMAL_DOMAIN` pattern

#### 2. `OnceLock` GPU Probe Caching (toadStool S158)

- Added `static GPU_DEVICE: OnceLock<Option<Arc<WgpuDevice>>>` to `gpu/device_info/mod.rs`
- `try_f64_device()` now uses `get_or_init()` — single probe per process lifetime
- Redirected 10+ test-local `try_device()` functions across `gpu/*.rs` to use the centralized `try_f64_device()`
- `water_balance.rs` production `gpu_only()` also uses the centralized probe
- Prevents SIGSEGV from concurrent `wgpu::Instance` creation in `cargo test` parallel threads

**Source**: toadStool S158 handoff — SIGSEGV fix via `OnceLock` GPU probe caching

#### 3. `cast` Module — Safe Numeric Casts (neuralSpring S162, healthSpring V33)

- Created `barracuda/src/cast.rs` with `usize_f64()`, `f64_usize()`, `usize_u32()`, `i32_f64()`, `u32_f64()`, `f64_u32()`
- Debug-panics on out-of-range casts, silent in release
- Migrated ~30 high-value library sites in `eco/`, `gpu/`, `io/`, `ipc/`, `primal_science/`
- Removed unfulfilled `#[expect(clippy::cast_precision_loss)]` from `ipc/timeseries.rs`
- Removed redundant `#[expect(clippy::cast_possible_truncation)]` from `gpu/richards.rs` (3 sites)

**Source**: neuralSpring S162 `safe_cast` module + healthSpring V33 centralized cast pattern

### P1: IPC and Discovery Evolution

#### 4. `DispatchOutcome<T>` as Library Type (wetSpring V126, groundSpring V112)

- Created `ipc/dispatch_outcome.rs` with generic `DispatchOutcome<T>` enum
- Added `is_ok()`, `is_recoverable()`, `ok()` helper methods
- Exported from `ipc` module via `pub use`
- `airspring_primal` binary now imports from library instead of defining locally
- Dead-code `#[expect]` removed (all variants documented and public)

**Source**: wetSpring V126 handoff — `DispatchOutcome<T>` library type pattern

#### 5. coralReef + Squirrel Discovery (healthSpring V34)

- Added `primal_names::CORALREEF` constant
- Added `primal_names::domains::SHADER` and `domains::INFERENCE` capability domains
- Added `biomeos::discover_shader_compiler()` — three-tier resolution (env override → named socket scan → capability probe)
- Added `biomeos::discover_inference_primal()` — same three-tier pattern for Squirrel
- Added `biomeos::discover_primal_by_capability()` — generic capability-based primal discovery
- Tests for all new constants and domains

**Source**: healthSpring V34 handoff — `discover_shader_compiler()` / `discover_inference_primal()` pattern

### P2: Code Quality Evolution

#### 6. `eprintln!` → Structured Tracing (neuralSpring S162)

- Migrated `airspring_primal/main.rs` server fatal error path to `tracing::error!`
- Kept `eprintln!` in CLI error paths (tracing not initialized before `run()`)
- Kept `eprintln!` in test SKIP diagnostics (correct for test stderr)
- Kept `eprintln!` in validation binary diagnostics (stderr is correct channel)
- Assessment: ~300 `eprintln!` sites are almost entirely in test + validation code (correct usage)

**Source**: neuralSpring S162 handoff — zero `eprintln!` in library/server code

#### 7. `mul_add()` FMA Numerical Accuracy (barraCuda Sprint 7)

- Evolved 18 sites across 7 files to use `mul_add()` for fused multiply-add precision:
  - `eco/richards.rs` (8) — flux calculations, trapezoidal integration
  - `eco/evapotranspiration.rs` (1) — soil heat flux coefficient
  - `eco/thornthwaite.rs` (1) — Willmott coefficient
  - `eco/dual_kc.rs` (1) — Kr evaporation reduction
  - `eco/infiltration.rs` (1) — Green-Ampt infiltration rate
  - `eco/water_balance.rs` (2) — total available water
  - `gpu/mc_et0.rs` (4) — Monte Carlo wind/solar perturbation
- Preserved `#[expect(clippy::suboptimal_flops)]` in `eco/soil_moisture.rs` (Topp equation reference implementation)

**Source**: barraCuda Sprint 7 handoff — `mul_add()` evolution pattern

### P3: barraCuda Primitive Wiring

#### 8. GemmF64 Transpose + Cyclic Reduction

- `tridiagonal_solve` already wired in `eco/richards.rs` via `barracuda::linalg::tridiagonal_solve` (S52+)
- `cyclic_reduction_f64` already wired via `pde::richards` GPU solver (S62+)
- `GemmF64::execute_gemm_ex(trans_a=true)` available upstream; sensor calibration uses higher-level `stats_f64::linear_regression` which consumes GEMM internally
- Added `gemm_f64_transpose` entry to `gpu/evolution_gaps.rs` documenting availability

**Source**: groundSpring V113 `GemmF64` transpose handoff

#### 9. ValidationSink Assessment (ludoSpring V23)

- Assessed upstream `barracuda::validation::ValidationHarness`: does NOT support sinks
- `finish()` writes directly to tracing — no trait-based output abstraction
- Added `validation_sink` entry to `gpu/evolution_gaps.rs` proposing upstream absorption
- When upstream absorbs ludoSpring's `ValidationSink` trait, airSpring's 91 validation binaries benefit automatically

**Source**: ludoSpring V23 handoff — `ValidationSink` / `StderrSink` / `BufferSink` pattern

### P4: Documentation and Composition Guidance

#### 10. Composition Guidance

- Created `wateringHole/airspring/AIRSPRING_COMPOSITION_GUIDANCE.md`
- Documents solo usage (40+ capabilities across 7 science categories)
- Documents trio combos (rhizoCrypt session tracking, sweetGrass attribution, loamSpine permanence)
- Documents wider primal compositions (barraCuda GPU, toadStool dispatch, coralReef shaders, petalTongue visualization, Squirrel AI, NestGate data, BearDog crypto, Songbird network)
- Documents cross-spring compositions (groundSpring uncertainty, neuralSpring ML, wetSpring life science, healthSpring human health, hotSpring physics, ludoSpring games)
- Documents 4 novel multi-primal pipelines: Full NUCLEUS Precision Agriculture, Continental-Scale Drought Monitoring, Precision Irrigation Decision System, Cross-Spring Soil-Microbiome-Health Pipeline

**Source**: healthSpring V34 `HEALTHSPRING_COMPOSITION_GUIDANCE.md` template

#### 11. This Handoff

- Updated `EVOLUTION_READINESS.md` header with V0.8.9 status
- Handoff placed in `airSpring/wateringHole/handoffs/` and `ecoPrimals/wateringHole/handoffs/`

### P5: Deep Debt — Smart Refactoring

#### 12. `eco/evapotranspiration.rs` Refactor (755 → 5 sub-modules)

- Extracted `eco/evapotranspiration/` module directory:
  - `mod.rs` (395 LOC) — shared types, ensemble, re-exports, tests
  - `atmosphere.rs` (115 LOC) — pressure, vapour pressure, wind (FAO-56 Eqs. 7–8, 11–17, 47)
  - `penman_monteith.rs` (179 LOC) — FAO-56 PM (Eq. 6), `DailyEt0Input`, `Et0Result`
  - `radiation.rs` (51 LOC) — solar Rs, soil heat flux
  - `hargreaves.rs` (38 LOC) — Hargreaves–Samani (Eq. 52)
  - `priestley_taylor.rs` (36 LOC) — Priestley–Taylor radiation-only
- Public API preserved via re-exports. All 28 evapotranspiration tests pass.

#### 13. `eco/dual_kc.rs` Refactor (712 → 8 sub-modules)

- Extracted `eco/dual_kc/` module directory:
  - `mod.rs` (50 LOC) — docs, re-exports
  - `types.rs` (61 LOC) — `BasalCropCoefficients`, `EvaporationParams`, `DualKcInput`, `DualKcOutput`
  - `equations.rs` (71 LOC) — FAO-56 Eqs. 69, 71–73, 77
  - `simulation.rs` (70 LOC) — `simulate_dual_kc`, `simulate_dual_kc_mulched`
  - `cover_crop.rs` (60 LOC) — `CoverCropType`
  - `mulch.rs` (48 LOC) — `ResidueLevel`, `mulched_ke`
  - `crop_basal.rs` (78 LOC) — `CropType::basal_coefficients` (Table 17)
  - `evaporation_params.rs` (79 LOC) — `SoilTexture::evaporation_params` (Table 19)
  - `tests.rs` (235 LOC) — unit tests
- Public API preserved via re-exports.

#### 14. `biomeos.rs` Refactor (713 → 3 sub-modules)

- Extracted `biomeos/` module directory:
  - `mod.rs` (503 LOC) — core types, platform fallback, env wrappers, re-exports, tests
  - `discovery.rs` (190 LOC) — all discovery functions (`discover_primal_socket_in`, `discover_shader_compiler`, `discover_inference_primal`, etc.)
  - `capabilities.rs` (39 LOC) — `parse_capabilities`
- Public API preserved via re-exports. All 38 biomeos tests pass.

#### 15. `validation.rs` Refactor (679 → 2 sub-modules)

- Extracted `validation/` module directory:
  - `mod.rs` (480 LOC) — `OrExit`, `init_tracing`, `section`, `banner`, re-exports, tests
  - `json.rs` (210 LOC) — JSON benchmark helpers (`json_f64`, `json_str`, `json_field`, `json_array`, `json_object_opt`, `parse_benchmark_json`, etc.)
- Public API preserved via re-exports. All 50 validation tests pass.

---

## Verification

```
cargo clippy --all-targets    # zero warnings (barracuda + metalForge)
cargo check                   # green
cargo test --lib              # 891 tests pass
```

## Files Changed (summary)

| Category | Files | Changes |
|----------|-------|---------|
| Identity | `lib.rs`, `niche.rs`, `primal_names.rs`, 2 binaries | `PRIMAL_NAME`/`PRIMAL_DOMAIN` constants |
| GPU Safety | `gpu/device_info/mod.rs`, 10 `gpu/*.rs` test modules, `water_balance.rs` | `OnceLock` GPU probe |
| Cast | `cast.rs` (new), ~15 library files | `usize_f64()`/`f64_usize()`/`usize_u32()` migrations |
| IPC | `ipc/dispatch_outcome.rs` (new), `ipc/mod.rs`, `airspring_primal/dispatch.rs`, `main.rs` | `DispatchOutcome<T>` |
| Discovery | `biomeos.rs`, `primal_names.rs` | coralReef/Squirrel discovery |
| Tracing | `airspring_primal/main.rs` | `eprintln!` → `tracing::error!` |
| FMA | 7 `eco/*.rs` + `gpu/mc_et0.rs` | 18 `mul_add()` sites |
| Composition | `wateringHole/airspring/AIRSPRING_COMPOSITION_GUIDANCE.md` (new) | Solo/trio/wider compositions |
| Refactoring | `eco/evapotranspiration/` (5 files), `eco/dual_kc/` (9 files), `biomeos/` (3 files), `validation/` (2 files) | 4 monoliths → 19 focused modules |
| Documentation | `evolution_gaps.rs`, `EVOLUTION_READINESS.md` | GEMM, ValidationSink entries |
| Cleanup | `ipc/timeseries.rs`, `gpu/stream.rs`, `biomeos.rs` | Removed unfulfilled `#[expect]`, simplified bool |

## Upstream Proposals

1. **ValidationSink trait**: Propose barracuda upstream absorption of ludoSpring V23's `ValidationSink`/`StderrSink`/`BufferSink` pattern for testable harness output
2. **GemmF64 direct use**: Available when airSpring needs custom regression beyond `stats_f64::linear_regression`
