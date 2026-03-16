# airSpring V0.8.2 — Niche Architecture + barraCuda/ToadStool Absorption Handoff

**Date:** March 15, 2026
**From:** airSpring V0.8.2
**To:** barraCuda, ToadStool, biomeOS, all Springs
**Authority:** wateringHole (ecoPrimals Core Standards)
**Supersedes:** V0.8.1 neuralAPI handoff (retained as reference)

---

## Executive Summary

airSpring v0.8.2 completes four major evolutions:

1. **Niche architecture clarification**: airSpring is a *niche deployment of primals*
   via biomeOS graphs — not a standalone primal. The `airspring_primal` binary is a
   transitional niche adapter (635 LOC) that will be replaced by pure biomeOS graph
   orchestration. Niche self-knowledge centralized in `src/niche.rs`. **BYOB niche
   deployment** via `niches/airspring-ecology.yaml` (matches groundSpring/wetSpring).

2. **Deep code quality**: Edition 2024 migration (rust-version 1.87), zero `#[allow()]`
   in production code (redundant lints removed from 91 binaries), `#![deny(unsafe_code)]`
   — **zero unsafe in production AND tests** (DI `_with`/`_in` pattern eliminates
   `set_var`/`remove_var`), zero clippy pedantic+nursery warnings, metalForge forge
   Edition 2024 migrated.

3. **Deep debt resolution**: Zero `panic!()` in library code (14 eliminated — all validation
   binaries use structured `exit(1)`), zero `#[allow()]` in library code (redundant cast
   allows removed, blanket binary allows evolved to targeted `#[expect()]` with reasons),
   `primal_science` refactored from 810 LOC monolith to 7 thematic sub-modules,
   57 centralized named tolerances in 4 domain submodules (`atmospheric`, `soil`, `gpu`,
   `instrument`), hardcoded primal names evolved to capability-based discovery
   (`crate::niche::NICHE_NAME`), ecoBin-clean default build (`standalone-http` opt-in),
   UniBin subcommands (`server`/`status`/`version`/`capabilities`).

4. **Full validation pipeline green** (2026-03-15): 1284/1284 Python control checks,
   863 + 280 + 61 Rust tests, 54+ validation binaries exit 0, 24/24 CPU algorithms
   match Python at 14.3× geometric mean speedup, 21/21 CPU-GPU parity modules validated,
   metalForge 32/32 dispatch + 21/21 routing + 17/17 mixed hardware.

5. **Cross-spring evolution**: Dependency injection pattern (`SocketConfig` + `_with`/`_in`
   variants) adopted from biomeOS V239, eliminating all `unsafe` and `#[serial]` from tests.
   rhizoCrypt semantic RPC alignment (`dag.dehydrate` → `dag.dehydration.trigger`).
   Tolerance hierarchy refactored from flat file to 4-submodule domain structure.

6. **Modern idiomatic Rust — deep debt round 2**: Centralized primal name constants
   (`primal_names.rs` — 9 primals + 4 capability domains), typed IPC errors
   (`AirSpringError::Ipc` replaces `Result<_, String>`), cross-spring time series IPC
   (`ipc/timeseries.rs` — `ecoPrimals/time-series/v1` schema from wetSpring), smart file
   refactoring (`device_info/` and `atlas_stream/` directory modules), `ProvenanceConfig`
   DI pattern eliminates last `unsafe` in provenance tests, shared Python tolerance
   vocabulary (`control/tolerances.py` mirrors all 57 Rust constants), formal tolerance
   registry (`specs/TOLERANCE_REGISTRY.md`), `pollster` 0.3→0.4 upgrade in metalForge.

**Quality: 863 lib + 280 integration + 61 forge tests, 0 failures, 0 clippy warnings.**

---

## §1 Niche Architecture — What This Means for barraCuda/ToadStool

### The Key Insight

A Spring is a niche validation domain — not a primal. It *deploys* primals via biomeOS
graphs. Each Spring is its own niche that can be redeployed and evolved independently.

```
                    biomeOS
                      │
            ┌─────────┼─────────┐
            │         │         │
      airSpring     wetSpring  hotSpring
      (niche)       (niche)    (niche)
            │
    ┌───────┼───────┐
    │       │       │
 BearDog Songbird ToadStool    ← real primals
```

### What airSpring Consumes from barraCuda (34 primitives, all stable)

| barraCuda Domain | Primitives Used | airSpring Modules | Stability |
|-----------------|----------------|-------------------|-----------|
| `ops` | `batched_elementwise_f64` (ops 0-19), `fused_map_reduce_f64`, `variance_f64_wgsl`, `moving_window_stats`, `kriging_f64`, `autocorrelation_f64_wgsl`, `bio::diversity_fusion` | `gpu::*` (25 modules) | 3+ releases stable |
| `optimize` | `brent`, `brent_gpu`, `nelder_mead`, `multi_start` | `eco::richards`, `gpu::van_genuchten`, `gpu::isotherm` | Stable |
| `pde` | `richards`, `richards_gpu`, `crank_nicolson` | `gpu::richards` | Stable (airSpring contributed S40) |
| `stats` | `bootstrap`, `jackknife`, `diversity`, `normal`, `pearson_correlation`, `regression::fit_linear`, `rmse`, `metrics` | `gpu::jackknife`, `gpu::bootstrap`, `eco::*` | Stable |
| `special` | `gamma::regularized_gamma_p`, `gamma::ln_gamma` | `eco::drought_index` | Stable |
| `linalg` | `ridge::ridge_regression` | `eco::correction` | Stable |
| `device` | `WgpuDevice`, `PrecisionRoutingAdvice`, `Fp64Strategy`, `GpuDriverProfile` | All GPU modules | Stable |
| `validation` | `ValidationHarness`, `exit_no_gpu`, `gpu_required` | All `validate_*` bins | Stable |
| `tolerances` | `check`, `Tolerance` | `tolerances/` (4 submodules) | Stable |
| `shaders::provenance` | `SpringDomain` | Cross-spring benchmarks | Stable |

**Zero local WGSL shaders.** Write→Absorb→Lean complete since v0.7.2.

**No duplicate math.** All stats, linalg, PDE, optimization delegate to barraCuda.

### What's NOT a barraCuda Concern

airSpring's domain-specific science (`eco::*` modules — ET₀, water balance, Richards,
isotherms, diversity, drought) stays local. These are validated against published papers
and don't belong in a math engine. barraCuda provides the numerical primitives; airSpring
provides the ecological science that composes them.

---

## §2 What airSpring Contributed Upstream (Complete)

| Contribution | barraCuda Module | Sprint | Status |
|-------------|-----------------|--------|--------|
| Richards PDE solver | `pde::richards` | S40 | **Absorbed** |
| Stats metrics re-exports | `stats::metrics` | S64 | **Absorbed** |
| SCS-CN runoff (op=17) | `ops::batched_elementwise_f64` | S66 | **Absorbed** |
| Stewart/Makkink/Turc/Hamon/Blaney-Criddle | `ops` (ops 14-16, 19) | S66 | **Absorbed** |
| Yield response (op=18) | `ops::batched_elementwise_f64` | S66 | **Absorbed** |
| `pow_f64` fractional exponent fix | `shaders` | TS-001 | **Merged** |
| Reduce buffer N≥1024 fix | `ops` | TS-004 | **Merged** |
| acos precision boundary fix | `shaders` | TS-003 | **Merged** |

**All local GPU ops absorbed. Zero local WGSL. Write→Absorb→Lean complete.**

---

## §3 Edition 2024 Learnings — Relevant for All Springs

### Pattern Matching Changes

Edition 2024 introduces stricter pattern matching. Closures that previously used
`|(_, &val)|` now require `|&(_, &val)|` (outer reference binding). This affected:

- `barracuda/src/eco/isotherm.rs` — `filter` and `map` closures
- `metalForge/forge/src/graph.rs` — topological sort `filter` closure
- 3 validation binaries

**Recommendation for barraCuda**: When upgrading to Edition 2024, run `cargo build`
and fix all "cannot explicitly dereference within an implicitly-borrowing pattern"
errors. They are mechanical fixes.

### `std::env::set_var` / `remove_var` — Eliminated via DI

Edition 2024 makes `std::env::set_var()` and `remove_var()` unsafe because they are
not thread-safe. airSpring initially wrapped these in `unsafe` blocks, then **evolved
past the need entirely** using dependency injection.

**Strategy used by airSpring** (final, recommended):

1. Introduce a config struct (e.g. `SocketConfig`) that captures env-dependent values
2. Create `_with`/`_in` function variants that accept the config struct explicitly
3. Public wrappers call `Config::from_env()` → `_with(config)` for production use
4. Tests construct `Config` directly — **no env mutation, no unsafe, no `#[serial]`**

Example from `biomeos.rs`:

```rust
pub struct SocketConfig {
    pub socket_dir: Option<PathBuf>,
    pub xdg_runtime_dir: Option<PathBuf>,
    pub family_id: Option<String>,
    // ...
}

pub fn resolve_socket_dir_with(config: &SocketConfig) -> PathBuf { /* ... */ }
pub fn resolve_socket_dir() -> PathBuf { resolve_socket_dir_with(&SocketConfig::from_env()) }
```

This pattern (from biomeOS V239) provides:
- **Zero unsafe** in production AND tests
- **No `#[serial]`** — tests run in parallel
- **Deterministic** — no ambient state leakage between tests
- **Composable** — easy to test edge cases by constructing specific configs

**Recommendation for barraCuda**: Adopt this DI pattern for any function that reads
env vars. This is superior to wrapping `set_var` in `unsafe`.

### `ureq` → `ring` C Dependency

When the `standalone-http` feature is enabled, `ureq → rustls → ring` introduces
C/assembly code. airSpring documents the ecoBin-compliant build:

```bash
cargo build --no-default-features --features testutil
```

**Recommendation**: All Springs should audit their `ureq`/`rustls` chains.

---

## §4 Evolution Opportunities for barraCuda

### Priority 1: GPU Provenance Tracking (from V081, still open)

Add a `provenance` feature flag to `WgpuDevice` that emits structured provenance
events for every shader dispatch. Springs would get GPU provenance "for free."

### Priority 2: neuralAPI-Aware Dispatch (from V081, still open)

Accept a `DispatchHint` from biomeOS that biases `PrecisionRoutingAdvice` based on
Pathway Learner observations (latency, device load, cross-Spring routing).

### Priority 3: Streaming Pipeline Provenance (from V081, still open)

Add optional pipeline-level provenance hooks to `UnidirectionalPipeline` and
`gpu_step()` for automatic experiment DAG construction.

### Priority 4: Edition 2024 Migration (NEW)

Migrate barraCuda to Edition 2024 using the pattern described in §3. airSpring's
experience can serve as a reference.

### Priority 5: Structured Metrics on `WgpuDevice::submit()` (from V081, still open)

Add structured logging to `WgpuDevice::submit()` and key ops so biomeOS can learn
GPU dispatch latencies across Springs.

---

## §5 Evolution Opportunities for ToadStool

### 1. `compute.provenance` Capability (from V081, still open)

Springs can merge GPU execution traces with experiment DAGs when ToadStool exposes
`compute.provenance`.

### 2. Pathway Learner Metrics (from V081, still open)

Structured metrics from ToadStool enable biomeOS to learn optimal batch sizes,
device warmup patterns, and GPU vs CPU routing decisions.

### 3. Niche-Aware Dispatch (NEW)

Now that Springs are explicitly niches (not primals), ToadStool can evolve to
dispatch compute for *niches* rather than individual primals. A niche deploy graph
specifies the full compute pipeline; ToadStool can optimize the entire graph rather
than individual `compute.execute` calls.

---

## §6 Cross-Spring Learnings from the Niche Refactoring

### 1. Self-Knowledge Module Pattern

airSpring's `src/niche.rs` centralizes:
- Capability table (41 capabilities)
- Semantic mappings (capability → science method)
- Operation dependencies (parallelization hints)
- Cost estimates (scheduling hints)
- Registration logic (biomeOS advertisement)

**Recommendation for other Springs**: Extract your primal's self-knowledge into a
similar module. This makes the evolution from standalone binary → biomeOS graph
deployment much cleaner.

### 2. Transitional Adapter Pattern

The `airspring_primal` binary is a transitional niche adapter: it exposes niche
capabilities via JSON-RPC while the biomeOS graph deployment is being built. Other
Springs should plan the same evolution.

### 3. Redundant Lint Cleanup

airSpring had `#![warn(clippy::pedantic)]` and `#![allow(clippy::cast_*)]` in 91
binary files — all redundant with the workspace-level `Cargo.toml` lint configuration.
Removing these reduced boilerplate and ensured lint consistency.

**Recommendation for barraCuda**: Audit all crate-level lint attributes. If they
duplicate `Cargo.toml` workspace lint configuration, remove them.

---

## §7 Deep Debt Resolution — Patterns for barraCuda/ToadStool

### `panic!()` Elimination Strategy

airSpring's validation binaries previously used `panic!()` for fatal errors. These were
replaced with `eprintln!("FATAL: {context}") + std::process::exit(1)` — providing a clean
diagnostic message and structured exit code instead of a stack trace. The `_checked` variants
return `Result` for composable error handling; the convenience wrappers provide structured
termination for binary entry points.

**Recommendation for barraCuda**: Audit all `panic!()` in non-test code. Replace with
structured `exit(1)` in binaries, `Result` propagation in library code.

### `#[allow()]` → `#[expect()]` Evolution

Blanket `#![allow(clippy::pedantic, clippy::nursery)]` in 91 binary files were all
redundant with `Cargo.toml` workspace lint configuration and were removed. Binary-specific
`#![allow(clippy::unwrap_used)]` in IPC validation binaries were converted to targeted
`#[expect(clippy::too_many_lines, reason = "validation binary exercises full pipeline")]`
on the specific function where the lint fires.

**Recommendation for barraCuda**: Replace `#[allow()]` with `#[expect()]` wherever
possible — `#[expect()]` warns if the lint it suppresses stops firing, preventing
stale suppressions from accumulating.

### Smart Refactoring (primal_science)

The 810-line `primal_science.rs` monolith was refactored into 7 thematic sub-modules
(`et0.rs`, `water_balance.rs`, `soil.rs`, `drought_stats.rs`, `biodiversity.rs`,
`crop.rs`, `mod.rs`) without changing the external `dispatch_science` signature.

**Recommendation for barraCuda**: Identify monolithic files (>500 LOC) and refactor
into themed sub-modules. Preserve the public API signature.

### Tolerance Centralization

57 named `Tolerance` constants organized into a 4-submodule hierarchy:

| Submodule | Constants | Domains |
|-----------|-----------|---------|
| `atmospheric.rs` | 15 | ET₀, vapour pressure, radiation, MC propagation |
| `soil.rs` | 19 | Water balance, Richards PDE, SCS-CN, Green-Ampt, GDD, yield |
| `gpu.rs` | 9 | GPU/CPU cross-validation, kriging, NUCLEUS IPC |
| `instrument.rs` | 14 | Sensors, IoT, NPU, biodiversity, statistical quality |

Naming convention: `{DOMAIN}_{METRIC}_{QUALIFIER}` (e.g. `ET0_SAT_VAPOUR_PRESSURE`,
`GPU_CPU_CROSS`, `BIO_DIVERSITY_SHANNON`). All tolerances have documented scientific
justification. The hierarchy follows the same pattern as wetSpring V118.

**Recommendation for barraCuda**: Adopt a similar tolerance hierarchy for upstream
validation. The naming convention enables cross-spring tolerance alignment.

---

## §8 Deep Debt Round 2 — Patterns for barraCuda/ToadStool

### Primal Name Centralization

Hardcoded primal name strings (`"toadstool"`, `"beardog"`, etc.) create fragile coupling
and prevent capability-based discovery. airSpring now centralizes these in
`barracuda/src/primal_names.rs`:

```rust
pub const TOADSTOOL: &str = "toadstool";
pub const BEARDOG: &str = "beardog";
pub mod domains {
    pub const DAG: &str = "dag";
    pub const COMPUTE: &str = "compute";
    pub const COMMIT: &str = "commit";
    pub const PROVENANCE: &str = "provenance";
}
```

All IPC callers reference `primal_names::TOADSTOOL` and `primal_names::domains::DAG`
instead of raw strings. Unit tests enforce all names are lowercase.

**Recommendation for barraCuda**: Adopt a similar pattern. Primals should only have
self-knowledge and discover others at runtime via capability queries.

### Typed IPC Errors

IPC functions previously returned `Result<_, String>`. airSpring now uses a dedicated
`AirSpringError::Ipc(String)` variant, enabling pattern matching on error categories:

```rust
pub enum AirSpringError {
    Io(std::io::Error),
    Parse(String),
    Ipc(String),    // socket connect, timeout, protocol errors
    // ...
}
```

All `capability_call` and provenance functions propagate `AirSpringError::Ipc` instead
of opaque strings. This enables callers to distinguish IPC failures from parse failures
from I/O failures.

**Recommendation for barraCuda**: Define typed error enums for IPC. This eliminates
`String` error propagation and enables structured error handling in composing primals.

### Cross-Spring Time Series IPC

airSpring adopted the `ecoPrimals/time-series/v1` JSON schema from wetSpring for
standardized time series data exchange between Springs:

```json
{
  "schema": "ecoPrimals/time-series/v1",
  "source_spring": "airspring",
  "variable": "et0_reference",
  "unit": "mm/day",
  "timestamps": ["2024-01-01T00:00:00Z", ...],
  "values": [3.42, ...]
}
```

Implemented in `ipc/timeseries.rs` with builders, parsers, and domain-specific
constructors (`build_et0_series`, `build_soil_moisture_series`). All builders and
parsers return `Result<_, AirSpringError>` with typed errors.

**Recommendation for barraCuda**: Consider adding time series utilities to the
`ops` or `stats` module. Springs are converging on this schema for cross-spring
data exchange; upstream support would reduce duplication.

### Smart File Refactoring

Two files exceeding 800 LOC were refactored into themed directory modules:

| Original | → Directory | Sub-modules | Rationale |
|----------|-------------|-------------|-----------|
| `gpu/device_info.rs` (946 LOC) | `gpu/device_info/` | `mod.rs` (probe logic), `shader_provenance.rs` (static provenance) | Device probing is runtime; provenance is compile-time static data |
| `gpu/atlas_stream.rs` (887 LOC) | `gpu/atlas_stream/` | `mod.rs` (streaming core), `drift.rs` (drift monitoring) | Core streaming is data I/O; drift detection is statistical analysis |

Public API unchanged — `pub use` re-exports maintain backwards compatibility.

**Recommendation for barraCuda**: Apply the same pattern to large upstream files.
Split along domain boundaries (runtime vs static, I/O vs analysis), not line count.

### Shared Python Tolerance Vocabulary

`control/tolerances.py` mirrors all 57 Rust `Tolerance` constants as Python dataclasses:

```python
ET0_REFERENCE = Tolerance("et0_reference", 0.01, 1e-3,
    "FAO-56 Examples 17-19: validated against 3-decimal tables")
```

This ensures Python baselines and Rust validation use identical thresholds. The formal
registry at `specs/TOLERANCE_REGISTRY.md` documents all 57 tolerances with justifications.

---

## §9 Quality Gate

| Check | Result |
|-------|--------|
| Python control baselines | **1284/1284** checks, 54 scripts |
| `cargo test --lib` (barracuda) | **863 passed**, 0 failures |
| `cargo test --tests` (integration) | **280 passed**, 0 failures |
| `cargo test --lib` (metalForge forge) | **61 passed**, 0 failures |
| Validation binaries (hotSpring exit 0/1) | **54+ binaries** exit 0 |
| CPU benchmark (Rust vs Python) | **24/24 parity**, 14.3× geometric mean |
| CPU-GPU parity | **21/21 modules** validated |
| metalForge dispatch | **32/32** routing, **17/17** mixed hardware |
| `cargo clippy --lib` (both crates) | **0 warnings** (pedantic+nursery) |
| `cargo fmt --check` | **Clean** |
| Edition | **2024** (rust-version 1.87) |
| `unsafe` in production | **Zero** (`#![deny(unsafe_code)]`) |
| `unsafe` in tests | **Zero** (DI `_with`/`_in` pattern, no `set_var`) |
| `#[allow()]` in library | **Zero** (all evolved) |
| `panic!()` in library | **Zero** (all evolved to structured exit) |
| Named tolerances | **57** in 4 domain submodules (all with justification) |
| BYOB niche | `niches/airspring-ecology.yaml` deployed |

---

## §9 Files Changed (V0.8.1 → V0.8.2)

| File | Change |
|------|--------|
| `barracuda/Cargo.toml` | edition 2021→2024, rust-version 1.87, ureq docs |
| `metalForge/forge/Cargo.toml` | edition 2021→2024, rust-version 1.87 |
| `.rustfmt.toml` | edition 2021→2024 |
| `barracuda/src/lib.rs` | `forbid(unsafe_code)` → `deny(unsafe_code)`, removed redundant allow, added `pub mod niche` |
| `barracuda/src/niche.rs` | **NEW** — niche self-knowledge (41 caps, deps, costs, semantic mappings) |
| `barracuda/src/bin/airspring_primal.rs` | Refactored to transitional niche adapter (1034→635 LOC) |
| `barracuda/src/eco/isotherm.rs` | Edition 2024 pattern fix |
| `barracuda/src/ipc/provenance.rs` | `#[must_use]`, `dag.dehydration.trigger`, `crate::niche::NICHE_NAME` |
| `barracuda/src/biomeos.rs` | `SocketConfig` + `_with`/`_in` DI variants, 24 tests rewritten (zero unsafe) |
| `barracuda/tests/nucleus_integration.rs` | Rewritten to use `SocketConfig` (15 tests, zero unsafe, no `#[serial]`) |
| `metalForge/forge/src/lib.rs` | `forbid(unsafe_code)` → `deny(unsafe_code)` |
| `metalForge/forge/src/neural.rs` | Tests rewritten (5 tests, zero unsafe, no env mutation) |
| `metalForge/forge/src/graph.rs` | Edition 2024 pattern fix |
| `niches/airspring-ecology.yaml` | **NEW** — BYOB niche definition for biomeOS deployment |
| `graphs/airspring_niche_deploy.toml` | Version 0.8.2, `dag.dehydration.trigger`, `health.check` |
| `metalForge/deploy/airspring_deploy.toml` | `dag.dehydration.trigger`, `discovery.resolve` capability |
| 88 `barracuda/src/bin/*.rs` files | Removed redundant crate-level lint attributes |
| `barracuda/src/validation.rs` | `panic!()` → structured `exit(1)` in JSON helpers |
| `barracuda/src/primal_science.rs` → `barracuda/src/primal_science/` | Refactored 810 LOC monolith → 7 sub-modules |
| `barracuda/src/tolerances/` | **Refactored** from 790 LOC flat file → 4 domain submodules (atmospheric/soil/gpu/instrument), 57 named constants |
| `barracuda/src/gpu/stats.rs` | Fixed `clippy::let_and_return` |
| `barracuda/src/primal_names.rs` | **NEW** — centralized primal name constants (9 primals + 4 domains) |
| `barracuda/src/error.rs` | Added `Ipc(String)` variant for typed IPC errors |
| `barracuda/src/ipc/timeseries.rs` | **NEW** — cross-spring time series IPC (`ecoPrimals/time-series/v1`) |
| `barracuda/src/ipc/provenance.rs` | `ProvenanceConfig` DI, `_with` variants, typed errors, `niche_did()` |
| `barracuda/src/gpu/device_info.rs` → `gpu/device_info/` | Split: `mod.rs` (probe), `shader_provenance.rs` (static data) |
| `barracuda/src/gpu/atlas_stream.rs` → `gpu/atlas_stream/` | Split: `mod.rs` (streaming), `drift.rs` (drift monitor) |
| `control/tolerances.py` | **NEW** — Python tolerance vocabulary mirroring 57 Rust constants |
| `specs/TOLERANCE_REGISTRY.md` | **NEW** — formal tolerance registry with justifications |
| `metalForge/forge/Cargo.toml` | `pollster` 0.3→0.4 upgrade |
| `metalForge/forge/src/pipeline.rs` | Refactored duplicated match arms, removed `#[allow()]` |
| `metalForge/forge/src/bin/validate_dispatch_routing.rs` | Extracted 6 helper functions, removed `#[allow(too_many_lines)]` |
| 12 `barracuda/src/bin/validate_*.rs` | `panic!()` → structured `exit(1)` |
| 6 IPC/NUCLEUS validation binaries | Blanket `#![allow()]` → targeted `#[expect()]` |

---

## Action Items

| # | Owner | Action | Priority |
|---|-------|--------|----------|
| 1 | barraCuda | Migrate to Edition 2024 (see §3 for strategy) | High |
| 2 | barraCuda | Adopt DI `_with`/`_in` pattern for env-dependent functions (see §3) | High |
| 3 | barraCuda | Add `provenance` feature flag for automatic GPU provenance | Medium |
| 4 | barraCuda | Accept `DispatchHint` for biomeOS-influenced routing | Medium |
| 5 | barraCuda | Add structured metrics to `WgpuDevice::submit()` | High |
| 6 | barraCuda | Adopt tolerance hierarchy (4-submodule pattern, naming convention) | Medium |
| 7 | barraCuda | Audit crate-level lint attributes vs workspace config | Low |
| 8 | ToadStool | Expose `compute.provenance` capability | High |
| 9 | ToadStool | Evolve to niche-aware dispatch (graph-level optimization) | Medium |
| 10 | All Springs | Extract self-knowledge module (`niche.rs` pattern) | Medium |
| 11 | All Springs | Adopt DI pattern for zero-unsafe tests | High |
| 12 | All Springs | BYOB niche YAML + deploy graph alignment | Medium |
| 13 | All Springs | Align rhizoCrypt RPC names (`dag.dehydration.trigger`) | High |
| 14 | All Springs | Plan transitional adapter → pure graph deployment | Low |
| 15 | barraCuda | Centralize primal name constants (eliminate hardcoded strings) | Medium |
| 16 | barraCuda | Add typed IPC error enum (replace `Result<_, String>`) | High |
| 17 | barraCuda | Add time series utilities for `ecoPrimals/time-series/v1` schema | Medium |
| 18 | barraCuda | Smart-refactor files >800 LOC along domain boundaries | Medium |
| 19 | All Springs | Adopt shared Python tolerance vocabulary pattern | Medium |
| 20 | All Springs | Adopt `ProvenanceConfig` DI pattern for provenance tests | High |

---

*AGPL-3.0-or-later — airSpring v0.8.2 (March 15, 2026)*
