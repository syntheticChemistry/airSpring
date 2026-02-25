# airSpring V005: BarraCuda Evolution — Complete Status + Actionable Work

**Date**: February 25, 2026
**From**: airSpring (Precision Agriculture — Ecological & Agricultural Sciences)
**To**: ToadStool/BarraCuda core team
**airSpring Version**: 0.4.2 (328 tests, 75/75 cross-validation, 8 GPU orchestrators)
**ToadStool PIN**: `02207c4a` (S62+, DF64 expansion, unified_hardware)
**License**: AGPL-3.0-or-later
**Supersedes**: V004 (archived)

---

## Executive Summary

airSpring v0.4.2 is the most comprehensively wired BarraCuda consumer Spring.
328 tests pass (231 unit + 97 integration), 16 validation binaries, 75/75
Python↔Rust cross-validation matches (tol=1e-5), and 8 GPU orchestrators
bridge domain science to ToadStool WGSL shaders.

This handoff documents:
1. **What airSpring consumes** from BarraCuda (every primitive, by module)
2. **What airSpring contributed** back (3 critical fixes, 1 PDE solver)
3. **Cross-spring shader evolution** (which Spring helped which, and when)
4. **Actionable work items** for ToadStool (P0/P1/P2 priorities)
5. **metalForge evolution path** toward `unified_hardware`

**By the numbers:**
- 328 barracuda tests + 53 forge tests = 381 total
- 75/75 Python↔Rust cross-validation match (tol=1e-5)
- 8 GPU orchestrators, all verified with integration tests
- 11 papers reproduced, 344/344 Python checks
- CPU benchmarks: 12.5M ET₀/s, 38.9M VG θ/s, 175K NM fits/s
- Zero clippy warnings, cargo fmt clean

---

## Part 1: What airSpring Consumes from BarraCuda

### Primitives in Active Use

| BarraCuda Primitive | airSpring Module | Purpose | Test Count |
|--------------------|-----------------|---------|:----------:|
| `ops::batched_elementwise_f64` | `gpu::et0` | FAO-56 ET₀ batch dispatch (op=0) | 12 |
| `ops::batched_elementwise_f64` | `gpu::water_balance` | Water balance step dispatch (op=1) | 10 |
| `ops::kriging_f64::KrigingF64` | `gpu::kriging` | Soil moisture spatial interpolation | 8 |
| `ops::fused_map_reduce_f64` | `gpu::reduce` | Seasonal statistics (sum, mean, max, min, σ) | 12 |
| `ops::moving_window_stats` | `gpu::stream` | IoT sensor stream smoothing | 5 |
| `pde::richards::solve_richards` | `gpu::richards` | 1D Richards equation (Crank-Nicolson) | 4 |
| `optimize::nelder_mead` | `gpu::isotherm` | Isotherm fitting (single start) | 6 |
| `optimize::multi_start_nelder_mead` | `gpu::isotherm` | Isotherm fitting (LHS global search) | 4 |
| `linalg::ridge::ridge_regression` | `eco::correction` | Sensor correction pipeline | 2 |
| `linalg::tridiagonal_solve_f64` | `eco::richards` | Thomas algorithm verification | 2 |
| `validation::ValidationHarness` | `validation` | All 16 validation binaries | 4 |
| `stats::pearson_correlation` | `testutil` | Cross-station correlation | 2 |
| `stats::spearman_correlation` | `testutil` | Rank correlation | 2 |
| `stats::bootstrap_ci` | `testutil` | Confidence intervals | 2 |

**Total**: 14 BarraCuda primitives consumed, 75+ tests exercising them.

### GPU Orchestrators

| Orchestrator | Backend | GPU Dispatch | CPU Fallback | Integration Tests |
|-------------|---------|:------------:|:------------:|:-----------------:|
| `BatchedEt0` | `batched_elementwise_f64` | Yes | Yes | 5 (+ determinism) |
| `BatchedWaterBalance` | `batched_elementwise_f64` | Yes | Yes | 4 (+ determinism) |
| `KrigingInterpolator` | `kriging_f64` | Yes | Yes | 6 (+ determinism) |
| `SeasonalReducer` | `fused_map_reduce_f64` | Yes (N≥1024) | Yes | 6 (+ determinism) |
| `StreamSmoother` | `moving_window_stats` | CPU only | Yes | 5 |
| `BatchedRichards` | `pde::richards` | CPU cross-validate | Yes | 2 |
| `fit_*_nm` (isotherm) | `optimize::nelder_mead` | CPU | Yes | 3 |
| `fit_*_global` | `multi_start_nelder_mead` | CPU | Yes | 3 |

---

## Part 2: What airSpring Contributed Back

### Bug Fixes (3 critical, all absorbed upstream)

| ID | Severity | Shader | Issue | Fix | Absorbed |
|----|----------|--------|-------|-----|:--------:|
| TS-001 | **CRITICAL** | `batched_elementwise_f64.wgsl` | `pow_f64` fractional exponents returned NaN | Fixed in S50 | Yes |
| TS-003 | LOW | `trig_f64.wgsl` | `acos_f64` boundary precision (|x|≈1) | Fixed in S54 | Yes |
| TS-004 | **HIGH** | `fused_map_reduce_f64.wgsl` | Buffer conflict for N≥1024 elements | Fixed in S54 | Yes |

**Impact**: All 3 fixes benefit ALL Springs — hotSpring, wetSpring, neuralSpring.
TS-001 unlocked GPU dispatch for any power-law computation. TS-004 stabilized
reduce operations at scale.

### Domain Code Absorbed Upstream

| Module | Absorbed Into | Session | Status |
|--------|--------------|---------|--------|
| Richards PDE solver | `barracuda::pde::richards` | S40 | Upstream has Crank-Nicolson variant |
| Isotherm fitting strategy | Pattern documented | V004 | Linearized LS → NM → multi-start NM |

---

## Part 3: Cross-Spring Shader Evolution

### Who Helps Whom

```
hotSpring (56 shaders)                    neuralSpring (20 shaders)
  │ df64 core                                │ nelder_mead
  │ pow/exp/log/trig f64                     │ multi_start_nelder_mead
  │ lattice QCD precision                    │ ValidationHarness
  ▼                                          ▼
╔══════════════════════════════════════════════════════╗
║              ToadStool (608 WGSL shaders)            ║
║  41 categories, 46 cross-spring absorptions (S51-57) ║
╚══════════════════════════════════════════════════════╝
  ▲                                          ▲
  │ kriging_f64                              │ TS-001 pow_f64 fix
  │ fused_map_reduce                         │ TS-003 acos precision
  │ moving_window                            │ TS-004 reduce buffer
  │ ridge_regression                         │ Richards PDE
wetSpring (25 shaders)                    airSpring (3 fixes + 1 PDE)
```

### Cross-Spring Benefits Timeline

| When | Event | Who Benefits |
|------|-------|-------------|
| Jan 2026 | hotSpring df64 core | ALL Springs get f64 GPU math |
| Jan 2026 | wetSpring kriging_f64 | airSpring gets soil moisture interpolation |
| Feb 2026 | airSpring TS-001 (pow_f64) | ALL Springs get fractional exponents |
| Feb 2026 | airSpring TS-004 (reduce) | ALL Springs get stable N≥1024 reduce |
| Feb 2026 | neuralSpring nelder_mead | airSpring gets nonlinear optimization |
| Feb 2026 | S40 Richards PDE | ToadStool gains environmental PDE |
| Feb 2026 | S52-S62 DF64 expansion | hotSpring precision shaders multiply |
| Feb 2026 | neuralSpring multi_start_NM | airSpring gets global search with LHS |

### What Evolved and When

These cross-spring connections create a **positive feedback loop**: hotSpring's
precision fixes enable airSpring's VG retention GPU path. airSpring's Richards
solver enriches ToadStool's PDE library. wetSpring's kriging enables airSpring's
spatial mapping. neuralSpring's optimizer enables airSpring's isotherm fitting.
Each Spring's validation uncovers bugs that, once fixed, benefit everyone.

---

## Part 4: Actionable Work Items for ToadStool

### P0 — Blocking (airSpring cannot proceed without these)

*None currently blocking. All 8 orchestrators functional.*

### P1 — High Value (would unlock significant capability)

| Item | Description | Acceptance Criteria |
|------|-------------|-------------------|
| **f64 Crank-Nicolson GPU** | `ops::crank_nicolson` currently f32 only. airSpring Richards needs f64 for VG constitutive relations. | `ops::crank_nicolson_f64` with same API as f32 variant |
| **Batch PDE dispatch** | Richards solver is sequential — one column at a time. GPU batch dispatch for M soil columns would enable field-scale simulations. | `pde::richards::solve_batch_gpu` accepting `&[SoilParams]` |
| **NelderMeadGpu batch** | `NelderMeadGpu` exists but not yet suitable for 2-parameter isotherms (GPU overhead > benefit for dim=2). For dim≥5 surrogate models, it would be valuable. | Document dim threshold where GPU beats CPU |

### P2 — Nice to Have (future evolution)

| Item | Description | Notes |
|------|-------------|-------|
| **BFGS optimizer** | `optimize::bfgs` exists in ToadStool. airSpring could use it for gradient-based isotherm fitting (faster than NM for smooth objectives). | Needs gradient of objective function |
| **adaptive_penalty** | Constrained optimization for parameter bounds (θ_r < θ_s, α > 0). | Would replace manual clamping in VG parameter fitting |
| **Dual Kc GPU shader** | `BatchedDualKc` currently CPU-only. Multi-field Kcb+Ke partitioning is embarrassingly parallel. | New WGSL shader or extend `batched_elementwise_f64` with op=2 |
| **Moving window GPU** | `StreamSmoother` uses `MovingWindowStats` CPU path only. GPU dispatch for large IoT datasets (>8760 hourly). | Existing shader, just needs f64 path verification |

### P3 — Research (exploration for future handoffs)

| Item | Description |
|------|-------------|
| **unified_hardware** | ToadStool's `unified_hardware` module for device selection and compute scheduling. metalForge could evolve to consume this instead of hardcoding device types. |
| **ODE bio shaders** | wetSpring ODE bio shaders (S55+) for coupled soil-plant models. airSpring could wire these for dynamic root growth simulation. |
| **Surrogate learning** | hotSpring surrogate learning (Phase D). Richards PDE → surrogate model for real-time irrigation prediction. |

---

## Part 5: metalForge Evolution Path

### Current State

`metalForge/forge` v0.2.0: 53 tests, 6 modules

| Module | Status | Upstream Target |
|--------|--------|----------------|
| `van_genuchten` | **Absorbed** (→ `barracuda::pde::richards`) | Done |
| `isotherm` | **Absorbed** (→ `barracuda::optimize`) | Done |
| `metrics` | Pending | `barracuda::stats` or `validation` |
| `regression` | Pending | `barracuda::linalg` |
| `moving_window_f64` | Pending | `barracuda::ops::moving_window_stats` |
| `hydrology` | Pending | `barracuda::pde` |

### Proposed Evolution: metalForge → unified_hardware

ToadStool's `unified_hardware` module provides:
- Device discovery and capability checking
- Compute cost modeling (GPU vs CPU vs NPU)
- Transfer cost estimation

metalForge could evolve from a module staging area into a thin orchestration
layer that uses `unified_hardware` for dispatch decisions:

```
metalForge::dispatch(task, data)
  → unified_hardware::best_device(task.compute_cost, data.size)
  → device.execute(barracuda_primitive, data)
```

This would demonstrate the full "Write → Absorb → Lean → Compose" cycle.

---

## Part 6: Version History

| Version | Date | Key Changes |
|---------|------|-------------|
| V001 | 2026-02-25 | Initial GPU handoff (v0.3.8) |
| V002 | 2026-02-25 | Dual Kc, cover crops, deep debt cleanup (v0.3.10) |
| V003 | 2026-02-25 | Richards + isotherm GPU wiring (v0.4.0) |
| V004 | 2026-02-25 | ToadStool S62 sync, multi-start NM (v0.4.1) |
| **V005** | **2026-02-25** | **Complete status, GPU integration tests, cross-spring benchmarks, actionable items (v0.4.2)** |

---

## Part 7: Benchmark Data (for ToadStool validation)

These numbers are from `bench_airspring_gpu --release` on Eastgate hardware
(i9-12900K, RTX 4070, 64 GB DDR5). Useful for ToadStool to compare
cross-spring benchmark regressions.

| Operation | N | CPU (µs) | Throughput | Shader Family |
|-----------|---|----------|-----------|--------------|
| ET₀ (FAO-56) | 10,000 | 797 | 12.5M ops/s | `batched_elementwise_f64` |
| VG θ(h) batch | 100,000 | 2,575 | 38.9M evals/s | (pure arithmetic, GPU-ready) |
| Dual Kc season | 3,650 | 62 µs | 59M days/s | (CPU, Tier B) |
| Reduce (seasonal) | 100,000 | 254 | 395M elem/s | `fused_map_reduce_f64` |
| Stream smooth | 8,760 (24h) | 276 | 31.7M elem/s | `moving_window_stats` |
| Kriging (20→500) | 500 | 26 µs | — | `kriging_f64` |
| Ridge regression | 5,000 | 48 µs | R²=1.000 | `linalg::ridge` |
| Richards PDE (50 nodes) | 1 sim | 13,930 | 72 sims/s | `pde::richards` |
| Isotherm (linearized) | 9 pts | 0.1 | 8.3M fits/s | `eco::isotherm` |
| Isotherm (NM 1-start) | 9 pts | 5.7 | 175K fits/s | `optimize::nelder_mead` |
| Isotherm (NM 8×LHS) | 9 pts | 23.5 | 42.5K fits/s | `optimize::multi_start_nelder_mead` |

---

*End of V005 handoff. Direction: airSpring → ToadStool (unidirectional).
Supersedes V004 (archived). Next handoff: V006 after f64 Crank-Nicolson
GPU integration, metalForge → unified_hardware evolution, or new experiment
requiring upstream primitives.*
