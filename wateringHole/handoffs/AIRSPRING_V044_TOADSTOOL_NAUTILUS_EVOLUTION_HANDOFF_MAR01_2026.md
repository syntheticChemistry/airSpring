# airSpring → ToadStool/BarraCUDA — S71 Sync + Nautilus + Cross-Spring Learnings

**Date:** March 1, 2026
**From:** airSpring v0.5.9 (63 experiments, 817 tests, 53/53 cross-spring evolution benchmark)
**To:** ToadStool/BarraCUDA core team
**ToadStool HEAD:** `8dc01a37` (Session 71)
**Supersedes:** V043 S71 Sync Handoff (archived)
**License:** AGPL-3.0-or-later

---

## Executive Summary

- **ToadStool S71 fully synced**: 671 WGSL shaders, 2,773+ barracuda tests, zero regressions.
- **53/53 cross-spring evolution benchmark PASS** (expanded from 44 with 9 S71 checks).
- **Upstream `fao56_et0` cross-validated bit-identical** with airSpring local Penman-Monteith.
- **bingoCube/nautilus discovered as production-ready**: Evolutionary reservoir for ET₀ prediction,
  drift detection, NPU export — available now via path dependency.
- **Deep debt eliminated**: Shared `biomeos` module, configurable `RichardsConfig`, capability-based
  discovery everywhere, zero mocks in production, `lifecycle.health` wateringHole-compliant.
- **817 tests, 0 clippy warnings, 0 fmt issues.**

---

## Part 1: S71 Sync Status

### What airSpring Validated

| Category | S71 Change | airSpring Validation |
|----------|-----------|---------------------|
| 774→671 shaders | f32-only removed; universal precision | All GPU paths still dispatch correctly |
| DF64 transcendentals (15 fn) | sinh, cosh, tanh, atanh, cbrt, hypot, log1p, expm1 added | Available for VG/atmospheric GPU shaders |
| 66 `ComputeDispatch` migrations | Internal refactor | No API change for consumers; zero regressions |
| `HargreavesBatchGpu` | Pre-computed Ra dispatch | Available; airSpring keeps `BatchedElementwiseF64` op=6 (computes Ra internally) |
| `JackknifeMeanGpu` | Leave-one-out GPU | Validated: `jackknife_mean([1,2,3,4,5]) = 3.0 ± 0.5` |
| `BootstrapMeanGpu` | Bootstrap CI GPU | Validated: `bootstrap_ci(data, 1000) → lower < upper` |
| `HistogramGpu` | Atomic histogram GPU | Available for empirical ET₀ distributions |
| `KimuraGpu` | Population genetics GPU | Validated: `kimura_fixation_prob(1000, 0.01, 0.5) > 0.5` |
| `fao56_et0` scalar | Full PM from groundSpring | **Bit-identical to airSpring local PM** (cross-validated) |
| `Fp64Strategy` | Universal precision architecture | Automatic — same shader, best precision per silicon |

### 53/53 Cross-Spring Evolution Benchmark

9 new S71-specific checks added to `bench_cross_spring_evolution`:

| Check | Result | Provenance |
|-------|--------|------------|
| upstream `fao56_et0` FAO-56 Example 18 | 3.975 (±0.15 of 3.88) | groundSpring → S70 |
| upstream `fao56_et0` ≈ local PM | 3.975 = 3.975 | Cross-validation |
| Kimura fixation p > 0.5 (s>0) | 0.9999 | wetSpring bio → S71 |
| Kimura fixation p < 1.0 | 0.9999 | wetSpring bio → S71 |
| Jackknife mean of [1..5] = 3.0 | 3.0 | neuralSpring → S70+ |
| Jackknife variance > 0 | 0.5 | neuralSpring → S70+ |
| Bootstrap CI lower < upper | PASS | S64 |
| Bootstrap mean ≈ 6.5 | 6.5 (±0.5) | S64 |
| Percentile(50) of uniform [0,1) | 0.495 (±0.05 of 0.5) | S64 |

---

## Part 2: bingoCube/nautilus — Available Now

`ecoPrimals/primalTools/bingoCube/nautilus/` is a production-ready evolutionary reservoir
computing crate. Discovered during cross-spring exploration (hotSpring's brain architecture
uses it for QCD prediction). Domain-agnostic: takes `Vec<f64>` inputs, predicts `Vec<f64>`
targets via evolved Bingo board populations.

### What airSpring Will Use

| Capability | API | Application |
|------------|-----|-------------|
| Evolutionary reservoir | `NautilusShell::evolve_generation()` | Weather → ET₀ prediction |
| Drift monitoring | `DriftMonitor::is_drifting()` | Detect regime changes (drought, season shift) |
| Edge seeding | `EdgeSeeder::seed_boards()` | Focus on difficult microclimates |
| AKD1000 NPU export | `export_akd1000_weights()` | ~48µs edge inference (LOCOMOS power budget) |
| Shell transfer/merge | `continue_from()`, `merge_shell()` | Cross-field learning |
| JSON persistence | `to_json()` / `from_json()` | Cross-run bootstrap |

### Integration Path

```toml
# airSpring/barracuda/Cargo.toml
[dependencies]
bingocube-nautilus = { path = "../../primalTools/bingoCube/nautilus" }
```

This is a path dependency while ToadStool evolves its own `esn_v2` absorption of the
Nautilus pattern. Once ToadStool absorbs, airSpring will migrate to the upstream API
following the Write→Absorb→Lean cycle.

---

## Part 3: What airSpring Learned (Relevant to ToadStool Evolution)

### 1. Cross-Validated Implementations Converge

airSpring's local `eco::evapotranspiration::daily_et0()` and upstream `barracuda::stats::fao56_et0()`
were developed independently (airSpring origin vs groundSpring origin). Given equivalent inputs
(after RH→actual vapour pressure conversion), they produce **identical** output. Independent
implementations of the same FAO-56 equations converge — this validates both.

**Recommendation**: When absorbing domain-specific functions, cross-validate against the
originating Spring's implementation. If they diverge, one has a bug.

### 2. `BatchedElementwiseF64` op= Enum Is Getting Crowded

airSpring uses ops 0 (ET₀), 1 (WB), 5 (sensor cal), 6 (Hargreaves), 7 (Kc climate),
8 (dual Kc). The enum-based dispatch pattern works but doesn't scale to N Springs × M ops.
The new dedicated types (`HargreavesBatchGpu`, `JackknifeMeanGpu`, etc.) are the right
direction — each GPU dispatch type owns its own shader and validation.

**Recommendation**: Continue migrating domain ops to dedicated `*Gpu` types. The
`BatchedElementwiseF64` op= pattern should be deprecated once all ops have dedicated types.

### 3. Tolerance Organization Matters

airSpring's cross-spring evolution benchmark documents tolerances per check with provenance.
wetSpring's V86 handoff organizes tolerances by domain (alignment, diversity, phylogeny, etc.).
Both patterns work; the key insight is that tolerances need documented justification.

**Recommendation**: Consider a shared `tolerances` module pattern in barracuda for Springs
to organize their validation thresholds.

### 4. `biomeos` Module Pattern

airSpring extracted socket resolution, family ID, and primal discovery into a shared `biomeos.rs`
module. This eliminated triplicated code across `airspring_primal.rs`, `validate_nucleus.rs`,
and `validate_nucleus_pipeline.rs`.

**Recommendation**: Consider a shared `barracuda::biomeos` module upstream. Every Spring
needs socket resolution and primal discovery.

### 5. Richards PDE: `tridiagonal_solve` Singularity Detection

airSpring's local `thomas_solve` had no singularity detection. Upstream
`barracuda::linalg::tridiagonal_solve` returns `Result`, which we now propagate.
The numerical results are identical, but error handling is strictly better.

**Recommendation**: Springs migrating from local tridiagonal solvers should use the
upstream version for consistent error handling.

### 6. Nautilus Shell Architecture (from hotSpring)

hotSpring's `NautilusBrain` wraps `NautilusShell` with QCD-specific feature extraction
and multi-head classifiers (`plaquette_head`, `cg_head`, `polyakov_head`). airSpring
will build an `AirSpringBrain` wrapper with agricultural heads (ET₀, soil moisture, crop stress).

The pattern: **domain-agnostic shell** + **domain-specific brain wrapper** is reusable.
ToadStool could absorb this as `barracuda::nautilus::Shell` (the raw evolutionary reservoir)
and let Springs provide their own brain wrappers.

---

## Part 4: Absorption Roadmap

### Already Absorbed (airSpring → ToadStool)

| What | Session | Status |
|------|---------|--------|
| Stats metrics (rmse, mbe, NSE, IA, R², hit_rate) | S64 | Upstream in `barracuda::stats::metrics` |
| metalForge modules (6/6) | S64+S66 | Upstream in `barracuda` core |
| 3 bug fixes (TS-001, TS-003, TS-004) | S51-S68 | Upstream |

### Available for Absorption (ToadStool from airSpring)

| Candidate | airSpring Location | Why |
|-----------|-------------------|-----|
| `biomeos` module | `src/biomeos.rs` | Every Spring duplicates socket/discovery logic |
| `RichardsConfig` pattern | `src/eco/richards.rs` | Configurable solver parameters as struct + Default |
| Cross-spring evolution benchmark | `bin/bench_cross_spring_evolution.rs` | Regression gate for 5 Springs (53 checks) |
| Cross-spring absorption tests | `tests/cross_spring_absorption.rs` | 44 integration tests covering all upstream primitives |

### Waiting on ToadStool

| Need | Why | Priority |
|------|-----|----------|
| Remaining `ComputeDispatch` migrations (184 ops) | Clean dispatch API | P2 |
| Dedicated GPU types for ops 0,1,5,7,8 | Replace enum dispatch | P2 |
| `barracuda::nautilus::Shell` upstream | Replace path dependency on bingoCube | P3 |
| `esn_v2` absorption of Nautilus pattern | Unified evolutionary + temporal reservoir | P3 |

---

## Reproduction

```bash
cd barracuda/
cargo fmt -- --check                                    # Clean
cargo clippy --all-targets -- -D warnings               # 0 warnings
cargo test                                              # 817 passed, 0 failed
cargo run --release --bin bench_cross_spring_evolution   # 53/53 PASS
cargo doc --no-deps                                     # Builds
```

---

*Unidirectional handoff — no response expected. airSpring continues autonomous evolution.*
