# airSpring V050 — ToadStool/BarraCuda Evolution Handoff

**Date**: March 2, 2026
**From**: airSpring v0.6.6 → ToadStool/BarraCuda team
**ToadStool PIN**: S86 HEAD (`2fee1969`)
**Status**: 815 lib tests | 83 binaries | 73 experiments | 1237/1237 Python | 138/138 cross-spring | 68/68 rewire

---

## Executive Summary

airSpring has completed its full validation chain:

```
57 papers reproduced → 1237/1237 Python → 815 Rust lib tests → 25 Tier A GPU modules
    → BrentGpu + RichardsGpu wired → 13,000× speedup → metalForge cross-hardware
    → NUCLEUS primal (30 capabilities) → ecology domain in biomeOS
```

This handoff documents **everything airSpring learned** that is relevant to ToadStool's
continued evolution — what we contributed, what we consumed, what worked, and what
the ecosystem needs next.

---

## Part 1: What airSpring Contributed to ToadStool

### Direct Contributions (Write → Absorb)

| What | When | Status | Impact |
|------|------|--------|--------|
| FAO-56 Penman-Monteith ET₀ | V009→S66 | Absorbed | `stats::hydrology::fao56_et0` — core hydrology |
| Hargreaves/Thornthwaite/Makkink/Turc/Hamon ET₀ | V025→S66 | Absorbed | 6 ET₀ methods in `stats::hydrology` |
| Soil water balance | V009→S66 | Absorbed | `stats::hydrology::soil_water_balance` |
| Crop coefficient interpolation | V009→S66 | Absorbed | `stats::hydrology::crop_coefficient` |
| Van Genuchten parameters | V009→S40 | Absorbed | `pde::richards::SoilParams` |
| Richards PDE (Picard+CN) | V040→S40 | Absorbed | `pde::richards::solve_richards` |
| Richards GPU (Picard) | V045→S83 | Absorbed | `pde::richards_gpu::RichardsGpu` |
| Seasonal pipeline ops 0-13 | V039→S72 | Absorbed | `batched_elementwise_f64.wgsl` |
| Moving window f64 | V025→S66 | Absorbed | `stats::moving_window_f64` |
| Regression (linear, quad, exp, log) | V025→S66 | Absorbed | `stats::regression` |
| Metrics (RMSE, MAE, NSE, etc.) | V025→S64 | Absorbed | `stats::metrics` |
| Diversity (Shannon, Simpson, Bray-Curtis) | V035→S64 | Absorbed | `stats::diversity` |
| Van Genuchten named constants | V035→S66 | Absorbed | 8 `SoilParams` constants |
| Isotherm Nelder-Mead fitting | V030→S62 | Absorbed | `optimize::nelder_mead` |

**Total**: 14 major modules contributed upstream. airSpring is ToadStool's largest
hydrology/agriculture contributor.

### Indirect Contributions (Validation Pressure)

airSpring's validation work drove several upstream improvements:

- **f64 precision**: airSpring's Richards PDE and ET₀ validation required f64 throughout,
  accelerating hotSpring's `df64` emulation and universal precision work.
- **Batched state**: The seasonal pipeline's day-over-day water balance motivated
  `StatefulPipeline` (S80) and `BatchedStatefulF64` (S83).
- **BrentGpu**: VG inverse needs drove the batched GPU Brent root-finder (S83).
- **Cross-spring benchmark**: airSpring's 138-check cross-spring benchmark validates
  primitives from all 5 springs, serving as an ecosystem integration test.

---

## Part 2: What airSpring Consumes from ToadStool

### GPU Primitives (25 Tier A)

| Op | Function | WGSL Shader | Origin Spring |
|----|----------|-------------|---------------|
| 0 | FAO-56 ET₀ | `batched_elementwise_f64.wgsl` | airSpring |
| 1 | Water balance | `batched_elementwise_f64.wgsl` | airSpring |
| 5 | Sensor calibration | `batched_elementwise_f64.wgsl` | airSpring |
| 6 | Hargreaves ET₀ | `batched_elementwise_f64.wgsl` | airSpring |
| 7 | Kc climate adjust | `batched_elementwise_f64.wgsl` | airSpring |
| 8 | Dual Kc | `batched_elementwise_f64.wgsl` | airSpring |
| 9 | VG θ(h) | `batched_elementwise_f64.wgsl` | airSpring |
| 10 | VG K(h) | `batched_elementwise_f64.wgsl` | airSpring |
| 11 | Thornthwaite monthly | `batched_elementwise_f64.wgsl` | airSpring |
| 12 | GDD batch | `batched_elementwise_f64.wgsl` | airSpring |
| — | Kriging f64 | `kriging_f64.wgsl` | wetSpring |
| — | Fused map-reduce | `fused_map_reduce_f64.wgsl` | neuralSpring |
| — | Moving window | `moving_window_stats.wgsl` | airSpring |
| — | Nelder-Mead | `nelder_mead_f64.wgsl` | neuralSpring |
| — | BatchedNelderMeadGpu | `batched_nelder_mead_f64.wgsl` | neuralSpring S80 |
| — | BrentGpu | `brent_f64.wgsl` | neuralSpring S83 |
| — | RichardsGpu | `richards_picard_f64.wgsl` | airSpring S83 |
| — | Crank-Nicolson | `crank_nicolson_f64.wgsl` | hotSpring S62 |
| — | Ridge regression | `ridge_regression_f64.wgsl` | neuralSpring S69 |
| — | Matrix correlation | `matrix_correlation_f64.wgsl` | neuralSpring S69 |
| — | Bootstrap mean | `bootstrap_mean_f64.wgsl` | groundSpring |
| — | Jackknife mean | `jackknife_mean_f64.wgsl` | groundSpring |
| — | MC ET₀ propagation | `mc_et0_propagate_f64.wgsl` | groundSpring |

### CPU Primitives

- `stats::hydrology::*` (9 ET₀ methods, water balance, crop coefficient)
- `stats::diversity::*` (Shannon, Simpson, Bray-Curtis, Hill, Chao1, rarefaction)
- `stats::bootstrap_ci`, `stats::jackknife`
- `optimize::brent`, `optimize::lbfgs::lbfgs_numerical`
- `math::erf`, `math::gamma`, `math::ln_gamma`
- `spectral::anderson::anderson_4d` (soil disorder modeling)
- `pipeline::StatefulPipeline<WaterBalanceState>`
- `validation::ValidationHarness`

---

## Part 3: What We Learned — Lessons for ToadStool

### 1. f64 is Non-Negotiable for Hydrology

Richards equation, Van Genuchten, and precision ET₀ all require f64. The hotSpring
df64 emulation work was critical. Any new hydrology shader **must** support f64 natively.

### 2. Batched State is the Key to GPU Scaling

The seasonal pipeline (ET₀→Kc→WB→stress) processes M fields × N days. Without
`BatchedStatefulF64`, each day requires a GPU→CPU→GPU round-trip for state. With it,
soil moisture, snow, and deep percolation stay GPU-resident across time steps.

**Recommendation**: Make `BatchedStatefulF64` the default pattern for any multi-step
pipeline. The current API is good; consider adding `BatchedEncoder` integration for
single-submit multi-step chains.

### 3. Cross-Spring Provenance Matters

airSpring's pipeline uses shaders from all 5 springs:
- **hotSpring**: `pow_f64`, `exp_f64`, `erf`, Crank-Nicolson, Anderson 4D
- **wetSpring**: Shannon diversity, kriging, moving window
- **neuralSpring**: Nelder-Mead, Brent, L-BFGS, ridge regression, bisection
- **groundSpring**: Bootstrap, jackknife, MC propagation
- **airSpring**: FAO-56, hydrology ops, Richards PDE, seasonal pipeline

Document provenance in every shader header. It helps downstream springs debug issues.

### 4. Dispatch Overhead Dominates at Small N

For small grids (Richards PDE with 20 nodes), CPU is 4000× faster than GPU due to
`wgpu` dispatch overhead. GPU wins at N > ~1000 or when batching multiple fields.

**Recommendation**: Add a `BatchedRichardsGpu` that processes M soil columns in
one dispatch (M workgroups, each solving one column). This would amortize dispatch
across fields — same pattern as `HargreavesBatchGpu`.

### 5. The "Pure GPU" Pipeline is Viable

Exp 072 showed all 4 seasonal stages can run on GPU with 19.7× dispatch reduction.
The remaining bottleneck is state readback between stages. `BatchedEncoder` with
buffer chaining eliminates this.

### 6. metalForge Cross-Hardware is Production-Ready

The 7-stage seasonal pipeline routes through GPU→NPU (PCIe P2P)→CPU. metalForge's
substrate discovery and capability routing work correctly on consumer hardware
(RTX 4070 + AKD1000 + i9-12900K).

---

## Part 4: What ToadStool Should Evolve Next

### Priority 1: Fused Seasonal Pipeline

Currently airSpring chains 4 GPU dispatches per day per field. ToadStool's
`SeasonalPipelineF64` (S72) should be the target: a single shader that runs
ET₀→Kc→WB→stress in one dispatch. airSpring should adopt this once it supports
configurable crop databases.

### Priority 2: Batched Richards GPU

A `BatchedRichardsGpu` that solves M columns in one dispatch would eliminate the
per-field overhead that makes GPU slower than CPU for small grids.

### Priority 3: Green-Ampt GPU

airSpring has CPU `eco::infiltration::green_ampt_infiltration` but no GPU path.
The coupled SCS-CN + Green-Ampt runoff-infiltration pipeline (292/292 PASS) would
benefit from a batched GPU Green-Ampt shader.

### Priority 4: Pedotransfer GPU

Saxton-Rawls pedotransfer functions (`eco::soil_moisture`) compute θs, θr, Ks from
soil texture. These are embarrassingly parallel and would benefit from a simple
elementwise GPU shader.

### Priority 5: Multi-GPU Field Parallelism

airSpring's `run_multi_field` currently uses one GPU device. ToadStool's `multi_gpu`
module (S86) could shard fields across devices for true horizontal scaling.

---

## Part 5: Open Data Validation Chain

All 57 reproduced papers use open data. The compute pipeline for each:

```
Paper math (literature)
    → Python control (control/*/benchmark_*.json + *.py)
    → BarraCuda CPU (barracuda/src/eco/*.rs, 815 lib tests)
    → BarraCuda GPU (barracuda/src/gpu/*.rs, 25 Tier A orchestrators)
    → metalForge (metalForge/forge/, 66/66 cross-system routing)
    → NUCLEUS primal (30 capabilities, biomeOS ecology domain)
```

### Papers Still Pending Controls

| # | Paper | Reason | Open Data? |
|---|-------|--------|:----------:|
| 6 | Dong et al. — Multi-sensor network | Awaiting Dong lab field data (2026) | No |
| 7 | Dong et al. — Full IoT + forecast | Awaiting Dong lab field data (2026) | No |
| 16 | Cover crop water use | Awaiting field data | No |
| 23 | Dolson — Evolutionary sensor placement | Future (Tier 4) | N/A |
| 24 | Waters — Soil microbiome dynamics | Future (Tier 4) | N/A |

### BarraCuda CPU → GPU → metalForge Progression

| Domain | CPU Validated | GPU Wired | metalForge | Next |
|--------|:------------:|:---------:|:----------:|------|
| FAO-56 PM ET₀ | ✓ 815 tests | ✓ `BatchedEt0` (op=0) | ✓ 66/66 | Fused pipeline |
| Hargreaves/5 methods | ✓ | ✓ `HargreavesBatchGpu` (op=6) | ✓ | — |
| Water balance | ✓ | ✓ `BatchedWaterBalance` (op=1) | ✓ | `BatchedStatefulF64` |
| Dual Kc / Kc climate | ✓ | ✓ ops 7, 8 | ✓ | — |
| Richards PDE | ✓ | ✓ `RichardsGpu` (Picard) | ✓ | Batched multi-column |
| VG forward/inverse | ✓ | ✓ `BrentGpu` (θ→h) | ✓ | — |
| Isotherm fitting | ✓ | ✓ `BatchedNelderMeadGpu` | ✓ | — |
| Kriging/interpolation | ✓ | ✓ `kriging_f64` | ✓ | — |
| SCS-CN runoff | ✓ | — (CPU only) | — | GPU shader needed |
| Green-Ampt infiltration | ✓ | — (CPU only) | — | GPU shader needed |
| Pedotransfer (Saxton-Rawls) | ✓ | — (CPU only) | — | GPU shader needed |
| Seasonal pipeline | ✓ | ✓ Stages 1-3 | ✓ | `SeasonalPipelineF64` |
| MC uncertainty | ✓ | ✓ `McEt0PropagateGpu` | ✓ | — |
| Diversity indices | ✓ | ✓ GPU diversity | ✓ | — |

---

## Part 6: Metrics Summary

| Metric | Value |
|--------|-------|
| airSpring version | 0.6.6 |
| ToadStool PIN | S86 HEAD |
| Lib tests | 815 |
| Forge tests | 61 |
| Binaries | 83 (79 barracuda + 4 forge) |
| Experiments | 73 |
| Python checks | 1237/1237 |
| Cross-spring evolution | 138/138 PASS |
| Cross-spring rewire | 68/68 PASS (5/5 springs) |
| Tier A GPU modules | 25 + BrentGpu + RichardsGpu |
| CPU speedup vs Python | 13,000× (atlas-scale) |
| metalForge cross-system | 66/66 (GPU→NPU→CPU) |
| NUCLEUS capabilities | 30 |
| Control directories | 56 (56 benchmark JSONs) |
| Open data papers | 57/57 reproduced |
| Papers pending controls | 5 (awaiting field data or future) |
| Clippy pedantic | 0 warnings |
| Coverage | 95.66% line |
