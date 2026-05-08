# airSpring V010 Deep Debt Evolution Handoff

**Date**: May 8, 2026
**From**: airSpring v0.10.0
**For**: primalSpring, upstream primal teams, projectNUCLEUS, foundation
**Phase**: Deep Debt, Cleanup, and Evolution
**guideStone**: Level 1 → Level 2 (IPC-wired, composition experiment crates)

---

## What Changed

### 1. Centralized Capability Method Constants (`methods.rs`)

Created `barracuda/src/methods.rs` with 44 constants for all `science.*`, `ecology.*`,
`provenance.*`, and infrastructure method strings. Updated `niche.rs`, `primal_science/mod.rs`,
and `ipc/mcp.rs` to use these constants instead of inline string literals.

**Why it matters**: Eliminates the #1 drift risk identified in the parity audit — scattered
capability strings across 4+ modules. Any method name change now requires updating exactly
one file, and the `capabilities_match_registry` integration test catches divergence from
`capability_registry.toml`.

### 2. Composition Experiment Crates (3 new)

| Crate | Result | Pattern |
|-------|--------|---------|
| `exp001_local_science_parity` | **55/55 PASS** | Validates all 25 science methods via `dispatch_science` — in-process, no IPC needed |
| `exp002_composition_parity` | **10/10 PASS** | Replicates exp094 pattern: Tier 1 local (always green), Tier 2 IPC (skip if absent), Tier 3 NUCLEUS (skip if no deployment) |
| `exp003_foundation_target_validation` | **4/4 PASS** | Reads `foundation/data/targets/thread06_ag_targets.toml`, validates dispatchable targets |

**Why it matters**: airSpring had **zero** experiment crates vs ludoSpring (100), healthSpring (94).
These establish the pattern for systematic composition validation. exp003 closes the loop between
`foundation` targets and airSpring science — currently 5 targets are dispatchable, expanding as
method-to-paper mapping grows.

### 3. Large File Refactoring (3 files)

| File | Before | After | Extraction |
|------|--------|-------|------------|
| `ipc/provenance.rs` | 747 LOC | 496 LOC | Tests → `provenance_tests.rs` |
| `rpc/mod.rs` | 650 LOC | 341 LOC | Tests → `rpc/tests.rs` |
| `gpu/seasonal_pipeline/mod.rs` | 738 LOC | 539 LOC | Tests → `seasonal_pipeline/tests.rs` |

### 4. Compilation Error Fixes (3 pre-existing)

- **autobins**: `validate_gpu_rewire_support.rs` was auto-compiled as binary; disabled `autobins`
  in Cargo.toml since all 93 binaries are explicitly declared
- **NestGateProvider**: Replaced unimplemented `data::NestGateProvider` with capability-based
  IPC through `rpc::resolve_transport(NESTGATE)` — proper graceful degradation
- **fhe_ntt**: Feature-gated `FheNtt` check to `cfg!(feature = "domain-fhe")` for upstream
  barraCuda compatibility

### 5. Code Quality Improvements

- **Zero production `.unwrap()`** confirmed — all 401 instances in `#[cfg(test)]` or doc comments
- **All 145 `#[expect()]` suppressions** verified active (none stale)
- **Missing docs** added for `DailyWeather`, `Station`, `YieldRecord`, `HttpResponse`, `DataError`
- **`/proc/*` paths** gated behind `cfg(target_os = "linux")` in metalForge (`probe.rs`, `neural.rs`)
- **`resolve_neural_api_transport`** promoted from `pub(crate)` to `pub` (dead code fix)
- **`standalone-http` feature** declared in Cargo.toml (eliminates cfg warning)

### 6. guideStone Level 1 → 2

Level 2 criteria met:
- Tier 2 IPC checks wired in `exp002` (graceful skip when primals absent)
- Science parity validated via `dispatch_science` (exp001: 55/55)
- Foundation targets cross-validated (exp003: 4/4)

---

## What Other Springs Need to Know

### For primalSpring

- airSpring now has 3 experiment crates following the exp094/exp095 pattern
- `methods.rs` constants module is replicable — centralizes all method strings
- parity audit gap: `barraCuda optional = true` remains open (requires `MathBackend` trait)
- guideStone advanced to Level 2; Level 3 requires live NUCLEUS deployment

### For projectNUCLEUS

- exp002 is ready for live NUCLEUS testing — set `FAMILY_ID` and socket dir
- exp003 reads foundation targets directly — can be wired into CI once NUCLEUS deploys

### For foundation

- 36 targets in `thread06_ag_targets.toml`, 5 currently dispatchable via exp003
- Remaining 31 need `paper → method` mapping expansion (different method signatures
  or qualitative targets that don't map to a single dispatch)

### For barraCuda

- AG-015 (barraCuda optional) remains the largest parity audit gap
- Proposal: `MathBackend` trait that barraCuda implements, with IPC fallback
- AG-010 (TensorSession) and AG-011 (Anderson WGSL shader) still open

---

## Remaining Gaps (prioritized)

1. **barraCuda optional = true** (AG-015) — ecosystem-wide, needs trait design
2. **guideStone L3+** — requires live NUCLEUS from plasmidBin
3. **exp003 target coverage** — 5/36 dispatchable, needs method-to-paper expansion
4. **Kokkos/Galaxy GPU benchmark** — no external GPU parity suite (internal only)
5. **Dong lab field data** — Tier 1 papers #6-7 blocked on external data

---

## File Manifest

| Path | Action |
|------|--------|
| `barracuda/src/methods.rs` | **NEW** — 44 centralized capability constants |
| `barracuda/src/lib.rs` | Updated — added `pub mod methods;`, `cfg_attr(not(test), forbid(unsafe_code))` |
| `barracuda/src/niche.rs` | Updated — CAPABILITIES array now uses `methods::*` constants |
| `barracuda/src/primal_science/mod.rs` | Updated — dispatch match arms use `methods::*` |
| `barracuda/src/ipc/mcp.rs` | Updated — `tool_to_method` uses `methods::*` |
| `barracuda/src/ipc/provenance.rs` | Refactored — tests extracted, `pub fn resolve_neural_api_transport` |
| `barracuda/src/ipc/provenance_tests.rs` | **NEW** — extracted test module |
| `barracuda/src/rpc/mod.rs` | Refactored — tests extracted |
| `barracuda/src/rpc/tests.rs` | **NEW** — extracted test module |
| `barracuda/src/gpu/seasonal_pipeline/mod.rs` | Refactored — tests extracted |
| `barracuda/src/gpu/seasonal_pipeline/tests.rs` | **NEW** — extracted test module |
| `barracuda/Cargo.toml` | Updated — `autobins = false`, `standalone-http` feature |
| `barracuda/src/bin/airspring_primal/handlers.rs` | Fixed — NestGateProvider → IPC |
| `barracuda/src/bin/bench_cross_spring_evolution/pipeline.rs` | Fixed — fhe_ntt cfg gate |
| `barracuda/src/data/weather.rs` | Updated — added field docs |
| `barracuda/src/data/provider.rs` | Updated — added field docs |
| `metalForge/forge/src/probe.rs` | Updated — `/proc/*` gated `cfg(target_os = "linux")` |
| `metalForge/forge/src/neural.rs` | Updated — `/proc/self` gated `cfg(target_os = "linux")` |
| `experiments/exp001_local_science_parity/` | **NEW** — 55/55 local science parity |
| `experiments/exp002_composition_parity/` | **NEW** — 10/10 composition parity |
| `experiments/exp003_foundation_target_validation/` | **NEW** — 4/4 foundation targets |
| `docs/PRIMAL_GAPS.md` | Updated — deep debt checklist, gS Level 2 |
| `sporeprint/validation-summary.md` | Updated — 90 experiments, methods centralized |
| `README.md` | Updated — deep debt evolution footer |
| `whitePaper/baseCamp/README.md` | Updated — deep debt evolution section |

---

*airSpring v0.10.0 — AGPL-3.0-or-later*
