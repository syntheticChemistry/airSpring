# airSpring V033 — Cross-Spring Rewiring + Benchmark Evolution

**Date**: February 28, 2026
**Version**: v0.5.2 (589 lib tests, 53 binaries, 48 experiments)
**ToadStool PIN**: S68+ HEAD (`e96576ee`)
**Previous**: V032 (ToadStool S68 sync + revalidation)

---

## Executive Summary

Completed deep cross-spring rewiring of airSpring to leverage modern ToadStool/BarraCuda
primitives (S66–S68+). Four modules rewired to delegate upstream, two new diversity
functions wired, and the cross-spring benchmark expanded from 16 to 25 benchmarks with
full 6-Spring provenance tracking. All 25/25 benchmarks PASS on RTX 4070 (Hybrid Fp64Strategy).

---

## Rewiring Completed

### 1. `gpu::hargreaves` — CPU batch → ToadStool S66 hydrology

**Before**: Local loop calling `eco::evapotranspiration::hargreaves_et0` per day.
**After**: Pre-computes Ra array, delegates to `barracuda::stats::hargreaves_et0_batch`.
**Provenance**: airSpring metalForge → ToadStool S66 (R-S66-002) → airSpring leans.

### 2. `eco::diversity` — New wrappers for bray_curtis_matrix + shannon_from_frequencies

**`bray_curtis_matrix`**: Full M×M distance matrix for ordination (PCoA, NMDS) and
cluster analysis. Useful for soil microbiome community comparisons.
**`shannon_from_frequencies`**: Pre-normalised Shannon for streaming pipelines where
abundances are already relative frequencies.
**Provenance**: wetSpring S28 (bio/diversity.rs) → ToadStool S64 (absorption) → airSpring wrappers.

### 3. `eco::crop::crop_coefficient_stage` — ToadStool S66 hydrology

**New function**: Wraps `barracuda::stats::crop_coefficient` for calendar-day-based Kc
interpolation. Complements existing `kc_from_gdd` (GDD-based).
**Provenance**: airSpring metalForge → ToadStool S66 (R-S66-002) → airSpring leans.

### 4. `gpu::richards::solve_cn_diffusion` — SoilParams consolidation

**Before**: Manually constructed `SoilParams` inline in map closure.
**After**: Uses `to_barracuda_params()` for consistency — single conversion point.

---

## Benchmark Evolution

### Cross-Spring Provenance Benchmark v0.5.2

| Metric | Before (V032) | After (V033) |
|--------|:------------:|:------------:|
| Benchmarks | 16 | **25** |
| Provenance entries | 10 | **13** |
| Tracked primitives | 33 | **45** |
| Origin Springs | 4 | **6** |

### New Benchmarks Added

| Benchmark | Time (ms) | Origin | What It Tests |
|-----------|----------:|--------|---------------|
| Hargreaves batch CPU (N=365) | 0.02 | airSpring→ToadStool S66 | Rewired batch delegate |
| Hargreaves batch CPU (N=10000) | 0.58 | airSpring→ToadStool S66 | Scale validation |
| Diversity alpha (5-species mix) | 0.00 | wetSpring→ToadStool S64 | Shannon + Simpson + Chao1 |
| Bray-Curtis matrix (20 samples) | 0.03 | wetSpring→ToadStool S64 | Full M×M ordination |
| Shannon from frequencies | 0.00 | wetSpring→ToadStool S66 | Pre-normalised path |
| Crop Kc stage (180d) | 0.00 | airSpring→ToadStool S66 | Calendar Kc interpolation |
| Kc from GDD (corn) | 0.00 | airSpring FAO-56 | GDD-based Kc |
| Anderson coupling chain (10K θ) | 0.20 | groundSpring→airSpring | θ→QS regime batch |
| Anderson regime classification | 0.00 | groundSpring spectral | Regime boundaries |

### Cross-Spring Shader Provenance (13 entries, 6 Springs)

```
Shader                         Origin                 Prims  airSpring Use
───────────────────────────────────────────────────────────────────────
math_f64.wgsl                  hotSpring                  6  Solar declination, atmospheric...
df64_core.wgsl                 hotSpring                  5  Consumer GPU precision (Df64)
df64_transcendentals.wgsl      hotSpring                  5  Df64 precision path for ET₀
batched_elementwise_f64.wgsl   airSpring + ToadStool      2  Primary GPU dispatch
kriging_f64.wgsl               wetSpring                  2  Spatial interpolation
fused_map_reduce_f64.wgsl      wetSpring                  5  Seasonal aggregation, diversity
moving_window_stats.wgsl       wetSpring                  2  IoT stream smoothing
nelder_mead.wgsl               neuralSpring               2  Isotherm fitting
crank_nicolson_f64.wgsl        hotSpring                  2  Richards PDE cross-val
norm_ppf.wgsl (Moro 1995)      hotSpring                  2  MC ET₀ confidence intervals
hydrology (CPU batch)          airSpring                  3  Hargreaves batch, Kc, WB
diversity (CPU bio)            wetSpring                  6  Biodiversity + microbiome
anderson (CPU coupling)        groundSpring               3  θ→QS regime coupling
───────────────────────────────────────────────────────────────────────
13 entries, 45 primitives, 6 origin Springs
```

---

## Revalidation Summary

| Check | Result |
|-------|--------|
| `cargo test --lib` | **589/589 PASS** (+5 from new rewiring tests) |
| `cargo clippy --all-targets` | **0 warnings, 0 errors** |
| `cross_validate` | **33/33 PASS** |
| `validate_atlas` | **1498/1498 PASS** |
| `validate_gpu_math` | **46/46 PASS** |
| `validate_ncbi_16s_coupling` | **29/29 PASS** |
| `bench_cross_spring` | **25/25 PASS** (RTX 4070, Hybrid Fp64) |
| Integration tests | **142/142 PASS** |

---

## Cross-Spring Evolution Notes

### hotSpring → airSpring (precision shaders)

hotSpring contributed the foundational math precision layer: `math_f64.wgsl` (6 primitives),
`df64_core.wgsl` (5 primitives), `df64_transcendentals.wgsl` (5 primitives),
`crank_nicolson_f64.wgsl` (2 primitives), and `norm_ppf.wgsl` (2 primitives).

These evolved through S60-S68 from individual f64 functions to the **universal precision
architecture** where `op_preamble()` injects abstract math ops and `df64_rewrite.rs` handles
naga IR transformation. airSpring benefits from precision-transparent computation on both
Titan V (native f64) and RTX 4070 (Df64 ~48-bit).

### wetSpring → airSpring (bio shaders)

wetSpring contributed biodiversity metrics (Shannon, Simpson, Chao1, Bray-Curtis, rarefaction),
spatial interpolation (kriging), time-series smoothing (moving window), and ridge regression.
These were absorbed into ToadStool S28-S66 and are now used by airSpring for:
- Cover crop biodiversity assessment
- Soil microbiome community analysis (16S amplicon)
- IoT sensor stream smoothing
- Sensor correction fitting

New in V033: `bray_curtis_matrix` (full M×M for PCoA/NMDS) and `shannon_from_frequencies`
(streaming pre-normalised Shannon).

### neuralSpring → airSpring (optimization)

neuralSpring contributed `nelder_mead` and `multi_start_nelder_mead` for derivative-free
optimization. airSpring uses these for isotherm fitting (Langmuir/Freundlich).

Both airSpring and neuralSpring benefit from ToadStool's universal precision shaders —
neuralSpring for neural network training precision, airSpring for ecological computation.

### groundSpring → airSpring (physics coupling)

groundSpring contributed the Anderson localisation model physics chain
(θ → Sₑ → pₓ → z → d_eff → QS regime). airSpring uses this for soil moisture regime
classification and NCBI 16S coupling (Experiment 048).

### airSpring → ToadStool (contributions back)

| Contribution | Impact | Session |
|-------------|--------|---------|
| TS-001: `pow_f64` fractional exponent fix | All Springs using VG/exponential math | S54 |
| TS-003: `acos` precision boundary fix | All Springs using trig in f64 | S54 |
| TS-004: reduce buffer N≥1024 fix | All Springs using `FusedMapReduceF64` | S54 |
| Richards PDE solver | `pde::richards` upstream module | S40 |
| Hydrology batch | `stats::hydrology` (Hargreaves, Kc, WB) | S66 |
| Moving window f64 | `stats::moving_window_f64` | S66 |
| Regression | `stats::regression` | S66 |
| 8 SoilParams named constants | `pde::richards::SoilParams` | S66 |

---

## Next Steps

- Wire `NelderMeadGpu` for GPU-resident isotherm optimization (medium effort)
- Wire `BatchedBisectionGpu` for m/z tolerance search (low effort)
- Track `barracuda::npu` (NpuDispatch trait) when proposed by wetSpring V61
- Track `barracuda::nn` (MLP, LSTM, ESN) for ML regime surrogates
