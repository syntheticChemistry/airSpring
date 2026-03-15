# airSpring V0.8.2 — Niche Architecture + barraCuda/ToadStool Absorption Handoff

**Date:** March 15, 2026
**From:** airSpring V0.8.2
**To:** barraCuda, ToadStool, biomeOS, all Springs
**Authority:** wateringHole (ecoPrimals Core Standards)
**Supersedes:** V0.8.1 neuralAPI handoff (retained as reference)

---

## Executive Summary

airSpring v0.8.2 completes two major evolutions:

1. **Niche architecture clarification**: airSpring is a *niche deployment of primals*
   via biomeOS graphs — not a standalone primal. The `airspring_primal` binary is a
   transitional niche adapter (635 LOC) that will be replaced by pure biomeOS graph
   orchestration. Niche self-knowledge centralized in `src/niche.rs`.

2. **Deep code quality**: Edition 2024 migration (rust-version 1.87), zero `#[allow()]`
   in production code (redundant lints removed from 95 binaries), `#![deny(unsafe_code)]`
   with unsafe isolated to test `set_var`/`remove_var`, zero clippy pedantic+nursery
   warnings, metalForge forge Edition 2024 migrated.

**Quality: 853 lib + 281 integration + 61 forge tests, 0 failures, 0 clippy warnings.**

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
| `tolerances` | `check`, `Tolerance` | `tolerances.rs` | Stable |
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

### `std::env::set_var` / `remove_var` Now Unsafe

Edition 2024 makes `std::env::set_var()` and `remove_var()` unsafe because they are
not thread-safe. This affects every crate that uses env vars in tests.

**Strategy used by airSpring**:

1. Changed `#![forbid(unsafe_code)]` → `#![deny(unsafe_code)]` in `lib.rs`
2. Added `#![allow(unsafe_code)]` to `#[cfg(test)] mod tests` blocks
3. Wrapped `set_var`/`remove_var` calls in `unsafe { ... }` with safety comments
4. Production code remains zero-unsafe

**Recommendation for barraCuda**: Apply the same strategy. This maintains zero unsafe
in production while allowing the necessary test patterns.

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

airSpring had `#![warn(clippy::pedantic)]` and `#![allow(clippy::cast_*)]` in 95
binary files — all redundant with the workspace-level `Cargo.toml` lint configuration.
Removing these reduced boilerplate and ensured lint consistency.

**Recommendation for barraCuda**: Audit all crate-level lint attributes. If they
duplicate `Cargo.toml` workspace lint configuration, remove them.

---

## §7 Quality Gate

| Check | Result |
|-------|--------|
| `cargo test --lib` (barracuda) | **853 passed**, 0 failures |
| `cargo test --test '*'` (integration) | **281 passed**, 0 failures |
| `cargo test --lib` (metalForge forge) | **61 passed**, 0 failures |
| `cargo clippy --all-targets --all-features` | **0 warnings** (pedantic+nursery) |
| `cargo fmt --check` | **Clean** |
| `cargo doc --no-deps` | **Clean** (both crates) |
| `bench_cross_spring_evolution` | **146/146 PASS** |
| Edition | **2024** (rust-version 1.87) |
| `unsafe` in production | **Zero** (`#![deny(unsafe_code)]`) |
| `#[allow()]` in production | **Zero** (all removed from 95 binaries) |

---

## §8 Files Changed (V0.8.1 → V0.8.2)

| File | Change |
|------|--------|
| `barracuda/Cargo.toml` | edition 2021→2024, rust-version 1.87, ureq docs |
| `metalForge/forge/Cargo.toml` | edition 2021→2024, rust-version 1.87 |
| `.rustfmt.toml` | edition 2021→2024 |
| `barracuda/src/lib.rs` | `forbid(unsafe_code)` → `deny(unsafe_code)`, removed redundant allow, added `pub mod niche` |
| `barracuda/src/niche.rs` | **NEW** — niche self-knowledge (41 caps, deps, costs, semantic mappings) |
| `barracuda/src/bin/airspring_primal.rs` | Refactored to transitional niche adapter (1034→635 LOC) |
| `barracuda/src/eco/isotherm.rs` | Edition 2024 pattern fix |
| `barracuda/src/ipc/provenance.rs` | `#[must_use]`, test unsafe blocks |
| `barracuda/src/biomeos.rs` | Test unsafe blocks |
| `barracuda/tests/nucleus_integration.rs` | Test unsafe blocks |
| `metalForge/forge/src/lib.rs` | `forbid(unsafe_code)` → `deny(unsafe_code)` |
| `metalForge/forge/src/neural.rs` | Test unsafe blocks |
| `metalForge/forge/src/graph.rs` | Edition 2024 pattern fix |
| 95 `barracuda/src/bin/*.rs` files | Removed redundant crate-level lint attributes |

---

## Action Items

| # | Owner | Action | Priority |
|---|-------|--------|----------|
| 1 | barraCuda | Migrate to Edition 2024 (see §3 for strategy) | High |
| 2 | barraCuda | Add `provenance` feature flag for automatic GPU provenance | Medium |
| 3 | barraCuda | Accept `DispatchHint` for biomeOS-influenced routing | Medium |
| 4 | barraCuda | Add structured metrics to `WgpuDevice::submit()` | High |
| 5 | barraCuda | Audit crate-level lint attributes vs workspace config | Low |
| 6 | ToadStool | Expose `compute.provenance` capability | High |
| 7 | ToadStool | Evolve to niche-aware dispatch (graph-level optimization) | Medium |
| 8 | All Springs | Extract self-knowledge module (`niche.rs` pattern) | Medium |
| 9 | All Springs | Plan transitional adapter → pure graph deployment | Low |

---

*AGPL-3.0-or-later — airSpring v0.8.2 (March 15, 2026)*
