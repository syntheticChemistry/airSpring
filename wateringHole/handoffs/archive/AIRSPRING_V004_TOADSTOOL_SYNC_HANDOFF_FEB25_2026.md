# airSpring V004: ToadStool Sync + Multi-Start NM + Upstream Audit

**Date**: February 25, 2026
**From**: airSpring (Precision Agriculture — Ecological & Agricultural Sciences)
**To**: ToadStool/BarraCuda core team
**airSpring Version**: 0.4.2 (328 tests, 75/75 cross-validation, 8 GPU orchestrators)
**ToadStool HEAD**: `02207c4a` (S62+, DF64 expansion, unified_hardware)
**License**: AGPL-3.0-or-later
**Supersedes**: V003 (archived)

---

## Executive Summary

airSpring v0.4.1 syncs with ToadStool S62 state, confirms all absorption items
(TS-001 through TS-004) are resolved upstream, wires `multi_start_nelder_mead`
for robust global isotherm fitting, audits all upstream primitives for new
opportunities, and documents the metalForge → `unified_hardware` evolution path.

**By the numbers:**
- 328 barracuda tests + 53 forge tests = 381 total (was 376)
- 75/75 Python↔Rust cross-validation match (tol=1e-5)
- 8 GPU orchestrators, 4 upstream primitives newly wired (v0.4.0-0.4.1)
- 4 new tests: `fit_langmuir_global`, `fit_freundlich_global`, `fit_batch_global`, parity check
- All TS-001 through TS-004 confirmed resolved at ToadStool HEAD
- Zero clippy warnings, zero fmt issues

---

## Part 1: ToadStool S52-S62 Audit — What airSpring Found

### Confirmed Absorptions (All DONE)

| airSpring Item | ToadStool Session | Status |
|---------------|-------------------|--------|
| TS-001: `pow_f64` fractional exponent | S54 (H-011) | **DONE** — `round()` + tolerance for integer detection |
| TS-003: `acos_simple` precision | S54 (H-012) | **DONE** — replaced with `acos_f64` from `math_f64.wgsl` |
| TS-004: `FusedMapReduceF64` N≥1024 | S54 (H-013) | **DONE** — separate `partials_buffer` for pass 2 |
| TS-002: Rust orchestrator | S54 (L-011) | **ALREADY PRESENT** |
| Richards PDE solver | S40 | **ABSORBED** — `pde::richards::solve_richards` |

### New Upstream Capabilities Discovered

| Primitive | Module | airSpring Relevance | Action Taken |
|-----------|--------|---------------------|-------------|
| `multi_start_nelder_mead` | `optimize::multi_start` | Global isotherm fitting with LHS | **WIRED** in v0.4.1 |
| `NelderMeadGpu` | `optimize::nelder_mead_gpu` | GPU-resident optimizer (5-50 params) | Not wired — 2-param isotherms cheaper on CPU |
| `ops::crank_nicolson` | `ops::crank_nicolson` | GPU Crank-Nicolson PDE | **f32 only** — needs f64 for Richards |
| `unified_hardware` | `unified_hardware` | `HardwareDiscovery`, `ComputeScheduler`, `MixedSubstrate` | metalForge evolution target |
| `optimize::bfgs` | `optimize::bfgs` | Quasi-Newton with gradient | Available for smooth objectives |
| `optimize::adaptive_penalty` | `optimize::penalty` | Constrained optimization | Available for bounded fitting |
| `linalg::nmf` | `linalg::nmf` | NMF (Euclidean/KL) | Not relevant for airSpring |
| `stats::spectral_density` | `stats::spectral_density` | RMT diagnostics | Not relevant currently |
| `numerical::ode_bio` | `numerical::ode_bio` | ODE systems | Potential for soil dynamics |

---

## Part 2: What airSpring Rewired (v0.4.1)

### New: Multi-Start Global Fitting

Added `fit_langmuir_global()` and `fit_freundlich_global()` to `gpu::isotherm`.
These use `barracuda::optimize::multi_start_nelder_mead` with Latin Hypercube
Sampling for robust global search. Better than single-start NM when linearized
initial guess is poor (ill-conditioned or noisy data).

```rust
// Single-start (fast, good with linearized initial guess)
let fit = gpu::isotherm::fit_langmuir_nm(&ce, &qe);

// Multi-start (robust, explores parameter space globally)
let fit = gpu::isotherm::fit_langmuir_global(&ce, &qe, 8); // 8 LHS starts
```

Batch variant: `fit_batch_global(datasets, n_starts)` for field-scale mapping.

### All GPU Orchestrators (8 total, unchanged)

| airSpring Module | BarraCuda Primitive | Status |
|-----------------|--------------------|----|
| `gpu::et0` | `ops::batched_elementwise_f64` (op=0) | **GPU-FIRST** |
| `gpu::water_balance` | `ops::batched_elementwise_f64` (op=1) | **GPU-STEP** |
| `gpu::kriging` | `ops::kriging_f64::KrigingF64` | **Integrated** |
| `gpu::reduce` | `ops::fused_map_reduce_f64` | **GPU N≥1024** |
| `gpu::stream` | `ops::moving_window_stats` | **Wired** |
| `gpu::richards` | `pde::richards::solve_richards` | **Wired** |
| `gpu::isotherm` | `optimize::nelder_mead` + `multi_start_nelder_mead` | **Wired** (v0.4.1) |
| `eco::correction::fit_ridge` | `linalg::ridge::ridge_regression` | **Wired** |

---

## Part 3: Recommendations for ToadStool

### Request: f64 Crank-Nicolson GPU

`ops::crank_nicolson` is f32 only. airSpring's Richards equation requires f64
precision for van Genuchten constitutive relations (strong nonlinearity at
dry end: h < -1000 cm). An `ops::crank_nicolson_f64` module would enable
fully GPU-resident Richards solving.

The shader comment already lists "Richards equation for unsaturated flow
(airSpring, wetSpring)" as a use case.

### metalForge → unified_hardware Evolution

ToadStool's `unified_hardware` module provides exactly what airSpring's metalForge
targets:

| metalForge Concept | unified_hardware Equivalent |
|--------------------|-----------------------------|
| Substrate discovery | `HardwareDiscovery::discover_all()` |
| Mixed dispatch | `ComputeScheduler` |
| Transfer cost | `MixedSubstrate`, `TransferCost`, `PcieBridge` |
| Hardware types | `HardwareType`, `HardwareCapabilities` |
| Bandwidth tiers | `BandwidthTier` |

Recommendation: airSpring's metalForge forge modules should evolve to use
`unified_hardware` for device selection and dispatch, rather than maintaining
a separate hardware abstraction. The forge's domain-specific modules
(metrics, regression, hydrology) remain absorption candidates for
`barracuda::stats` and `barracuda::ops`.

### Absorption Candidates (Still Pending)

These metalForge modules have stable APIs, comprehensive tests, and documented
provenance. Ready for absorption into barracuda:

| Module | Target | Tests | Status |
|--------|--------|:-----:|--------|
| `metrics` (rmse, mbe, nse, ia, r2) | `barracuda::stats::metrics` | 11 | Highest priority |
| `regression` (4 curve fits) | `barracuda::stats::regression` | 11 | Medium priority |
| `hydrology` (Hargreaves, Kc, WB) | `barracuda::ops::hydrology` | 13 | Domain-specific |
| `moving_window_f64` | `barracuda::ops::moving_window_stats_f64` | 7 | f64 variant of existing f32 |

---

## Part 4: Compute Pipeline Status

| Exp | Paper | Python | CPU | GPU | Status |
|:---:|-------|:------:|:---:|:---:|--------|
| 001 | FAO-56 PM ET₀ | 64 | 31 | GPU-FIRST | Complete |
| 002 | Dong 2020 soil | 36 | 26 | ridge | Complete |
| 003 | Dong 2024 IoT | 24 | 11 | stream | Complete |
| 004 | FAO-56 Ch 8 WB | 18 | 13 | GPU-STEP | Complete |
| 005 | Real data 918d | R²=0.967 | 23 | all | Complete |
| 006 | Richards PDE | 14 | 15 | pde::richards | Complete |
| 007 | Biochar isotherms | 14 | 14 | NM + multi-start | Complete |
| 009 | Dual Kc Ch 7 | 63 | 61 | Tier B | CPU ready |
| 010 | Regional ET₀ | 61 | 61 | BatchedEt0 | Complete |
| 011 | Cover crops | 40 | 40 | Tier B | CPU ready |
| 015 | 60yr WB | 10 | 11 | Batched | Complete |

---

## Part 5: What Changed Since V003

| Change | Detail |
|--------|--------|
| **Added** | `fit_langmuir_global`, `fit_freundlich_global` (multi-start NM) |
| **Added** | `fit_batch_global` (batch global fitting for field-scale mapping) |
| **Added** | 4 new tests for multi-start NM (v0.4.1) |
| **Added** | 5 new GPU integration tests: Richards + Isotherm orchestrators (v0.4.2) |
| **Added** | Expanded `bench_airspring_gpu`: Richards PDE, VG θ(h), isotherm 3-level |
| **Added** | Cross-spring shader provenance table in benchmarks + CROSS_SPRING_EVOLUTION.md |
| **Updated** | `evolution_gaps.rs` with upstream capability audit (S52-S62) |
| **Updated** | `gpu::mod.rs` and `gpu::isotherm` doc comments |
| **Confirmed** | All TS-001 through TS-004 resolved at ToadStool S54 |
| **Confirmed** | All 328 tests pass, zero clippy warnings, cargo fmt clean |
| **Version** | 0.4.0 → 0.4.1 → 0.4.2 |

---

*End of V004 handoff. Direction: airSpring → ToadStool (unidirectional).
Supersedes V003 (archived). Next handoff: V005 after f64 Crank-Nicolson
GPU integration or metalForge → unified_hardware evolution.*
