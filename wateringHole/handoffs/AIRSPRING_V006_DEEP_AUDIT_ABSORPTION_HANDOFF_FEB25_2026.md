# airSpring V006: Deep Audit + Absorption Roadmap

**Date**: February 25, 2026
**From**: airSpring (Precision Agriculture — Ecological & Agricultural Sciences)
**To**: ToadStool/BarraCuda core team
**airSpring Version**: 0.4.2+ (371 lib tests, 97 integration, 96.84% coverage)
**ToadStool PIN**: `02207c4a` (S62+, DF64 expansion, unified_hardware)
**License**: AGPL-3.0-or-later
**Supersedes**: V005 (archived)

---

## Executive Summary

Deep audit and cleanup pass completed. airSpring v0.4.2+ delivers:
- **371 lib tests** (was 328), **96.84% coverage** (was ~89%)
- Zero clippy pedantic+nursery warnings
- **testutil** split into focused submodules (generators, stats, bootstrap)
- All validation binaries hardened (unwrap → proper error handling)
- Hot-path allocations eliminated (richards q_buf prealloc, csv_ts iterator fold)
- Shader Promotion Mapping documented in `evolution_gaps.rs`
- **4 metalForge modules** ready for absorption

---

## Part 1: Current BarraCuda Consumption

Same 14 primitives as V005, now better tested.

| BarraCuda Primitive | airSpring Module | Purpose | Test Count |
|--------------------|-----------------|---------|:----------:|
| `ops::batched_elementwise_f64` | `gpu::et0` | FAO-56 ET₀ batch dispatch (op=0) | 20 |
| `ops::batched_elementwise_f64` | `gpu::water_balance` | Water balance step dispatch (op=1) | 18 |
| `ops::kriging_f64::KrigingF64` | `gpu::kriging` | Soil moisture spatial interpolation | 17 |
| `ops::fused_map_reduce_f64` | `gpu::reduce` | Seasonal statistics (sum, mean, max, min, σ) | 26 |
| `ops::moving_window_stats` | `gpu::stream` | IoT sensor stream smoothing | 20 |
| `pde::richards::solve_richards` | `gpu::richards` | 1D Richards equation (Crank-Nicolson) | 4 |
| `optimize::nelder_mead` | `gpu::isotherm` | Isotherm fitting (single start) | 10 |
| `optimize::multi_start_nelder_mead` | `gpu::isotherm` | Isotherm fitting (LHS global search) | 10 |
| `linalg::ridge::ridge_regression` | `eco::correction` | Sensor correction pipeline | 17 |
| `linalg::tridiagonal_solve_f64` | `eco::richards` | Thomas algorithm verification | 31 |
| `validation::ValidationHarness` | `validation` | All 16 validation binaries | 33 |
| `stats::pearson_correlation` | `testutil` | Cross-station correlation | 34 |
| `stats::spearman_correlation` | `testutil` | Rank correlation | 34 |
| `stats::bootstrap_ci` | `testutil` | Confidence intervals | 3 |

**Total**: 14 BarraCuda primitives consumed, 371 lib tests + 97 integration.

---

## Part 2: What airSpring Learned (for ToadStool to absorb)

Key findings from the audit:

### 1. mul_add() everywhere

All `a * b + c` patterns converted to `mul_add()` for FMA efficiency. ToadStool WGSL shaders should ensure `fma_f64` is available for parity.

### 2. Named constants

8 named constants in Richards solver (`VG_H_ABS_MAX`, etc.). Upstream `pde::richards` should adopt matching constants for consistency and maintainability.

### 3. let...else pattern

Idiomatic Rust for GPU device checks. All GPU tests use:
```rust
let Some(device) = try_device() else { return; }
```

### 4. Preallocation

Picard iteration buffers (`a`, `b`, `c`, `d`, `h_prev`, `h_old`, `q_buf`) preallocated outside loops. Upstream should do the same to avoid hot-path allocations.

---

## Part 3: metalForge Absorption Targets

| Module | Target | Status | Acceptance |
|--------|--------|--------|------------|
| metrics | `barracuda::stats::metrics` | **Ready** (53 tests, all passing) | RMSE, MBE, NSE, IA, R² |
| regression | `barracuda::stats::regression` | **Ready** | fit_linear, fit_quadratic, fit_exponential, fit_logarithmic |
| moving_window_f64 | `barracuda::ops::moving_window_stats_f64` | **Ready** | f64 CPU moving window stats |
| hydrology | `barracuda::ops::hydrology` | **Ready** | hargreaves_et0, crop_coefficient, soil_water_balance |

After absorption, airSpring will rewire:
- `testutil/stats.rs` → `barracuda::stats::metrics`
- `eco/correction.rs` → `barracuda::stats::regression`
- `gpu/stream.rs` smooth_cpu → `barracuda::ops::moving_window_stats_f64`
- `eco/evapotranspiration.rs` hargreaves → `barracuda::ops::hydrology`

---

## Part 4: Shader Promotion Mapping

| Rust Module | GPU Orchestrator | WGSL Shader | Pipeline Stage | Tier |
|---|---|---|---|---|
| eco::evapotranspiration | gpu::et0 | batched_elementwise_f64.wgsl (op=0) | ET₀ computation | A (ready) |
| eco::water_balance | gpu::water_balance | batched_elementwise_f64.wgsl (op=1) | Daily water balance | B (needs BatchedStatefulF64) |
| eco::dual_kc | gpu::dual_kc | Pending (op=8) | Crop coefficient | B (needs conditional shader) |
| eco::soil_moisture | gpu::kriging | kriging_f64.wgsl | Spatial interpolation | A (ready) |
| eco::richards | gpu::richards | pde_richards.wgsl | PDE solve | A (ready) |
| eco::isotherm | gpu::isotherm | nelder_mead.wgsl | Isotherm fitting | B (needs batch NM) |
| testutil | gpu::reduce | fused_map_reduce_f64.wgsl | Seasonal stats | A (ready) |
| io::csv_ts | gpu::stream | moving_window.wgsl | Stream smoothing | A (ready) |

---

## Part 5: Paper Controls Matrix

| Paper | Python Control | BarraCuda CPU | BarraCuda GPU | metalForge |
|:-----:|:-:|:-:|:-:|:-:|
| 1 (FAO-56 ET₀) | 64/64 | 31/31 | BatchedEt0 GPU-FIRST | metrics |
| 2 (Soil sensors) | 36/36 | 40/40 | fit_ridge | regression |
| 3 (IoT) | 24/24 | 11/11 | StreamSmoother | moving_window_f64 |
| 4 (Water balance) | 18/18 | 13/13 | BatchedWB GPU-STEP | hydrology |
| 5 (Real data) | R²=0.967 | 23/23 | All orchestrators | All modules |
| 6 (Dual Kc) | 63/63 | 61/61 | BatchedDualKc (Tier B) | — |
| 7 (Regional ET₀) | 61/61 | 61/61 | BatchedEt0 at scale | — |
| 8 (Cover crops) | 40/40 | 40/40 | BatchedDualKc + mulch | — |
| 9 (Richards) | 14/14 | 15/15 | BatchedRichards WIRED | — |
| 10 (Biochar) | 14/14 | 14/14 | fit_*_nm WIRED | — |
| 11 (60yr WB) | 10/10 | 11/11 | BatchedEt0+BatchedWB | hydrology |

---

## Part 6: Actionable Items (P0/P1/P2)

### P0 — Blocking

*None currently blocking.*

### P1 — High Value

| Item | Description |
|------|-------------|
| **Absorb 4 metalForge modules** | metrics, regression, moving_window_f64, hydrology |
| **fma_f64 shader instruction** | mul_add() parity for WGSL |
| f64 Crank-Nicolson GPU | `ops::crank_nicolson_f64` for Richards |
| Batch PDE dispatch | `pde::richards::solve_batch_gpu` for M soil columns |
| NelderMeadGpu batch | Document dim threshold where GPU beats CPU |

### P2 — Nice to Have

| Item | Description |
|------|-------------|
| **Named constants in upstream pde::richards** | VG_H_ABS_MAX, etc. |
| **Preallocation pattern in upstream pde::richards** | Picard buffers outside loops |
| BFGS optimizer | Gradient-based isotherm fitting |
| adaptive_penalty | Constrained optimization for parameter bounds |
| Dual Kc GPU shader | BatchedDualKc multi-field |
| Moving window GPU | StreamSmoother f64 path verification |

### P3 — Research

| Item | Description |
|------|-------------|
| unified_hardware | metalForge → unified_hardware orchestration |
| ODE bio shaders | wetSpring coupled soil-plant models |
| Surrogate learning | Richards PDE → surrogate for real-time irrigation |

---

## Part 7: Version History

| Version | Date | Key Changes |
|---------|------|-------------|
| V001 | 2026-02-25 | Initial GPU handoff (v0.3.8) |
| V002 | 2026-02-25 | Dual Kc, cover crops, deep debt cleanup (v0.3.10) |
| V003 | 2026-02-25 | Richards + isotherm GPU wiring (v0.4.0) |
| V004 | 2026-02-25 | ToadStool S62 sync, multi-start NM (v0.4.1) |
| V005 | 2026-02-25 | Complete status, GPU integration tests, cross-spring benchmarks (v0.4.2) |
| **V006** | **2026-02-25** | **Deep audit, 96.84% coverage, testutil split, metalForge absorption roadmap (v0.4.2+)** |

---

*End of V006 handoff. Direction: airSpring → ToadStool (unidirectional).
Supersedes V005 (archived). Next handoff: V007 after metalForge absorption,
fma_f64 shader parity, or new experiment requiring upstream primitives.*
