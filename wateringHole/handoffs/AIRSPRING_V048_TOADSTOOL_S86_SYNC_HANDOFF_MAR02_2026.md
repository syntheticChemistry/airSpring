# airSpring V048 — ToadStool S86 Sync + Tier B→A Promotions

**Date**: March 2, 2026
**From**: airSpring v0.6.5
**To**: ToadStool / BarraCuda evolution team
**License**: AGPL-3.0-or-later
**Covers**: v0.6.4 → v0.6.5 (S79 → S86 sync, 138/138 cross-spring)

---

## Executive Summary

- **ToadStool S86 sync**: 7 commits (S80-S86), 160 files, 2,866 tests, 844 WGSL shaders, 144 `ComputeDispatch` ops.
- **Tier B→A promotions**: WB (`BatchedStatefulF64` S83), isotherm (`BatchedNelderMeadGpu` S80).
- **New upstream validated**: `BrentGpu`, `RichardsGpu`, `StatefulPipeline`, `nautilus`, `anderson_4d`, `lbfgs`, hydrology split.
- **Cross-spring benchmark**: 124 → 138 checks (+14 S80-S86 validations).
- **Zero breakage**: 813 lib tests, 0 clippy pedantic warnings, all existing experiments pass.

---

## Part 1: What ToadStool Absorbed from airSpring (Cumulative)

| Module | airSpring Version | ToadStool Session | Status |
|--------|------------------|-------------------|--------|
| `SeasonalPipelineF64` | V039 | S72 | Absorbed — GPU fused ET₀→Kc→WB→stress |
| Brent f64 root-finding | V039 | S72 → `BrentGpu` S83 | Absorbed — now GPU batched |
| batched_elementwise ops 0-8 | V009-V039 | S66-S70+ | Absorbed |
| Ops 9-13 (VG/Thornthwaite/GDD/Pedotransfer) | V039 | S76 | Absorbed |
| `StatefulPipeline` + `WaterBalanceState` | V039 | S80 | Absorbed |
| `BatchedStatefulF64` (GPU ping-pong state) | V045 | S83 | Absorbed |
| Richards PDE → `RichardsGpu` | V045 | S83 | Absorbed |
| Regression, hydrology, moving_window | V009 | S66 | Absorbed |

## Part 2: New Upstream Primitives Validated (S80-S86)

| Primitive | Purpose | airSpring Validation |
|-----------|---------|---------------------|
| `BatchedStatefulF64` | GPU-resident ping-pong state buffer | Type check, API verification |
| `StatefulPipeline<WaterBalanceState>` | CPU day-over-day state pipeline | Passthrough test, state persistence |
| `BrentGpu` | Batched GPU Brent root-finding | Type availability confirmed |
| `RichardsGpu` | GPU Richards PDE (Picard + CN + Thomas) | Type availability confirmed |
| `BatchedNelderMeadGpu` | Parallel Nelder-Mead optimizations | Config type verified |
| `lbfgs` / `lbfgs_numerical` | L-BFGS optimizer | Rosenbrock convergence test |
| `NautilusBrain` | Evolutionary reservoir (bingoCube) | Brain lifecycle, shell export |
| `anderson_4d` | 4D Anderson Hamiltonian | L=3 → 81×81 structure verified |
| `hydrology::fao56_et0` | CPU FAO-56 PM ET₀ | Example 18 range check |
| `hydrology::soil_water_balance` | CPU daily WB update | Clamping and mass balance |
| `hydrology::crop_coefficient` | CPU Kc interpolation | Midpoint interpolation |

## Part 3: Tier B→A Promotions

| Gap | Was | Now | Reason |
|-----|-----|-----|--------|
| Water balance (day-over-day GPU) | Tier B: needs `BatchedStatefulF64` | Tier A: `BatchedStatefulF64` exists (S83) | GPU-resident state buffer for multi-day WB |
| Isotherm batch fitting | Tier B: needs batch NM | Tier A: `BatchedNelderMeadGpu` (S80) | Parallel NM for multi-start fitting |

## Part 4: What airSpring Still Maintains Locally

| Module | Reason |
|--------|--------|
| `eco::dual_kc` | FAO-56 Ch 7/11 domain logic — too specialized |
| `eco::crop` (growth stages) | Domain constants and stage logic |
| `eco::evapotranspiration` (PM params) | Parameter ordering and naming conventions |
| `eco::tissue`, `eco::cytokine` | Paper 12 domain models |
| `gpu::seasonal_pipeline` | Orchestration over ToadStool primitives |
| `gpu::atlas_stream` | Multi-station streaming pattern |
| `gpu::mc_et0` | MC ET₀ uncertainty orchestration |
| All validation binaries | Domain-specific experiment validation |

## Part 5: Next Evolution Targets

1. **Wire `BatchedStatefulF64` into `run_multi_field`**: Replace per-day CPU readback
   with GPU-resident state carry-forward. Target: 3 dispatches per season instead of N+2.
2. **Wire `BrentGpu` into VG inverse**: Replace CPU Brent with GPU batched Brent for
   multi-field VG pressure head inversion.
3. **Wire `RichardsGpu` into Richards validation**: Add GPU path alongside existing CPU.
4. **`BatchedEncoder` for multi-stage**: Use single `queue.submit()` for ET₀+Kc+WB chain.

---

## Metrics

| Metric | v0.6.4 (S79) | v0.6.5 (S86) | Delta |
|--------|-------------:|-------------:|------:|
| Cross-spring checks | 124 | 138 | +14 |
| ToadStool session | S79 | S86 | +7 |
| ToadStool tests | 2,773 | 2,866 | +93 |
| WGSL shaders | 844 | 844 | = |
| ComputeDispatch ops | ~100 | 144 | +44 |
| Tier A gaps | 26 | 28 | +2 |
| Tier B gaps | 6 | 4 | -2 |
